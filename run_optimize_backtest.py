# run_optimize_backtest.py
"""
Orchestrateur principal pour le processus d'Optimisation Walk-Forward (WFO).
Ce script charge la configuration, identifie les tâches WFO à exécuter
(basées sur les stratégies actives et les arguments CLI pour la paire et le contexte),
et lance ces tâches en parallèle en utilisant multiprocessing.
Il intègre les composants refactorisés et la nouvelle structure AppConfig.
"""

import argparse
import logging
import sys
import time
import json
import dataclasses # Pour dataclasses.asdict pour sauvegarder AppConfig
from pathlib import Path
from datetime import datetime, timedelta, timezone
import concurrent.futures
import signal
import os
from typing import Any, Dict, List, Optional, Tuple
import threading

# numpy est utilisé pour EnhancedJSONEncoder
import numpy as np
import pandas as pd # Utilisé dans EnhancedJSONEncoder pour pd.Timestamp

# Détection de la racine du projet et ajout de src au PYTHONPATH
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    print(f"INFO: Ajouté {SRC_PATH} au PYTHONPATH par run_optimize_backtest.py")

# Configuration initiale du logging (sera surchargée par load_all_configs)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__) # Logger spécifique pour cet orchestrateur

# Imports des modules de l'application
try:
    from src.config.loader import load_all_configs, AppConfig
    from src.config.definitions import StrategyParamsConfig # Pour le typage
    from src.utils.run_utils import generate_run_id, ensure_dir_exists, _sanitize_path_component
    from src.backtesting.wfo.task_executor import run_wfo_task_wrapper
    from src.reporting.master_report_generator import generate_all_reports
except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE (run_optimize_backtest.py): Impossible d'importer les modules nécessaires: {e}. "
                    f"Vérifiez PYTHONPATH et les installations. CWD: {Path.cwd()}, sys.path: {sys.path}", exc_info=True)
    sys.exit(1)
except Exception as e_imp:
    logger.critical(f"ÉCHEC CRITIQUE (run_optimize_backtest.py): Erreur inattendue lors des imports initiaux: {e_imp}", exc_info=True)
    sys.exit(1)


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Encodeur JSON amélioré pour sérialiser les dataclasses, Path, datetime, et types numpy.
    """
    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o):
            # Ne pas sérialiser les instances de service qui pourraient être dans AppConfig
            # Cela nécessiterait une logique plus fine ou de ne pas sauvegarder AppConfig directement
            # si elle contient des objets non sérialisables comme des locks ou des instances de cache complexes.
            # Pour l'instant, on tente asdict, mais attention aux champs non sérialisables.
            try:
                return dataclasses.asdict(o)
            except TypeError: # Peut arriver si un champ n'est pas sérialisable par asdict
                return f"<DataclassInstance {type(o).__name__} not fully serializable>"
        if isinstance(o, Path):
            return str(o.as_posix())
        if isinstance(o, (datetime, pd.Timestamp)):
            return o.isoformat()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            if np.isnan(o): return None
            if np.isinf(o): return "Infinity" if o > 0 else "-Infinity"
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        # Gérer les types d'interface/protocole qui ne sont pas des classes concrètes
        if hasattr(o, '__protocol_attrs__'): # Heuristique pour les Protocoles
             return f"<Protocol {o.__class__.__name__}>"
        if isinstance(o, type) and hasattr(o, "__name__"): # Pour les types de classes
            return f"<Type {o.__name__}>"

        try:
            return super().default(o)
        except TypeError:
            logger.debug(f"Type non sérialisable rencontré par EnhancedJSONEncoder: {type(o)}. Rendu comme chaîne.")
            return str(o)

executor_instance: Optional[concurrent.futures.ProcessPoolExecutor] = None
orchestrator_shutdown_event = threading.Event()

def signal_handler(signum, frame):
    global executor_instance, orchestrator_shutdown_event
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') and isinstance(signum, int) else str(signum)
    
    if orchestrator_shutdown_event.is_set():
        logger.info(f"Signal {signal_name} reçu, mais l'arrêt de l'orchestrateur est déjà en cours.")
        return

    logger.info(f"Signal {signal_name} reçu. Demande d'arrêt global de l'orchestrateur...")
    orchestrator_shutdown_event.set()

    if executor_instance:
        logger.info("Arrêt du ProcessPoolExecutor depuis le gestionnaire de signaux...")
        if sys.version_info >= (3, 9):
            executor_instance.shutdown(wait=False, cancel_futures=True)
        else:
            executor_instance.shutdown(wait=False)
        logger.info("Demande d'arrêt envoyée au ProcessPoolExecutor.")

def main():
    global executor_instance, orchestrator_shutdown_event

    orchestrator_log_prefix = "[WFO_Orchestrator]"
    run_start_time = time.time()

    parser = argparse.ArgumentParser(description="Orchestrateur pour l'Optimisation Walk-Forward (WFO).")
    parser.add_argument(
        "--pair", type=str, required=True,
        help="Paire de trading à optimiser (ex: BTCUSDT)."
    )
    parser.add_argument(
        "--tf", "--context", dest="tf", type=str, required=True,
        help="Label de contexte principal pour cette exécution (ex: 5min_rsi_filter)."
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Répertoire racine du projet si le script n'est pas exécuté depuis la racine."
    )
    args = parser.parse_args()

    project_root_arg = args.root if args.root else str(PROJECT_ROOT)
    project_root_path = Path(project_root_arg).resolve()

    app_config: Optional[AppConfig] = None
    try:
        # load_all_configs est censé initialiser les instances de service dans AppConfig
        app_config = load_all_configs(project_root=str(project_root_path))
        logger.info(f"{orchestrator_log_prefix} --- Démarrage de l'Orchestrateur d'Optimisation Walk-Forward ---")
        logger.info(f"{orchestrator_log_prefix} Args: Paire='{args.pair}', Contexte CLI='{args.tf}', Racine='{project_root_path}'")
        logger.info(f"{orchestrator_log_prefix} AppConfig chargée (avec services injectés attendus).")
    except Exception as e_conf:
        logger.critical(f"{orchestrator_log_prefix} Erreur chargement config: {e_conf}", exc_info=True)
        sys.exit(1)

    if not app_config:
        logger.critical(f"{orchestrator_log_prefix} AppConfig non chargée. Abandon.")
        sys.exit(1)

    run_name_prefix_cfg = app_config.global_config.simulation_defaults.run_name_prefix
    sanitized_pair_for_run_id = _sanitize_path_component(args.pair.upper())
    sanitized_context_for_run_id = _sanitize_path_component(args.tf)
    orchestrator_run_id_prefix = f"{run_name_prefix_cfg}_orch_{sanitized_pair_for_run_id}_{sanitized_context_for_run_id}"
    orchestrator_run_id = generate_run_id(prefix=orchestrator_run_id_prefix)
    logger.info(f"{orchestrator_log_prefix} ID de Run Orchestrateur: {orchestrator_run_id}")

    orchestrator_base_log_dir_path = Path(app_config.global_config.paths.logs_backtest_optimization)
    orchestrator_run_log_dir = orchestrator_base_log_dir_path / orchestrator_run_id
    orchestrator_base_results_dir_path = Path(app_config.global_config.paths.results)
    orchestrator_run_results_dir = orchestrator_base_results_dir_path / orchestrator_run_id

    try:
        ensure_dir_exists(orchestrator_run_log_dir)
        ensure_dir_exists(orchestrator_run_results_dir)
        logger.info(f"{orchestrator_log_prefix} Répertoire log run orchestrateur: {orchestrator_run_log_dir}")
        logger.info(f"{orchestrator_log_prefix} Répertoire résultats run orchestrateur: {orchestrator_run_results_dir}")
    except Exception as e_mkdir_orch:
        logger.critical(f"{orchestrator_log_prefix} Échec création répertoires run orchestrateur: {e_mkdir_orch}", exc_info=True)
        sys.exit(1)

    orchestrator_config_save_path = orchestrator_run_log_dir / "run_config_orchestrator.json"
    try:
        # Créer un dictionnaire sérialisable à partir d'AppConfig
        # Exclure les champs qui ne sont pas facilement sérialisables ou non pertinents pour le log de config
        # (ex: instances de cache, locks, etc.)
        # Pour l'instant, on tente avec EnhancedJSONEncoder, mais il faudra peut-être une méthode to_dict() dans AppConfig.
        app_config_dict_to_save = dataclasses.asdict(app_config) # Peut échouer si des champs non sérialisables
        
        # Tentative de nettoyage pour la sérialisation
        # Ceci est une rustine, idéalement AppConfig aurait une méthode to_serializable_dict()
        if 'data_validator_instance' in app_config_dict_to_save: del app_config_dict_to_save['data_validator_instance']
        if 'cache_manager_instance' in app_config_dict_to_save: del app_config_dict_to_save['cache_manager_instance']
        if 'strategy_loader_instance' in app_config_dict_to_save: del app_config_dict_to_save['strategy_loader_instance']
        if 'error_handler_instance' in app_config_dict_to_save: del app_config_dict_to_save['error_handler_instance']
        if 'event_dispatcher_instance' in app_config_dict_to_save: del app_config_dict_to_save['event_dispatcher_instance']


        with open(orchestrator_config_save_path, 'w', encoding='utf-8') as f_cfg_orch:
            json.dump(app_config_dict_to_save, f_cfg_orch, cls=EnhancedJSONEncoder, indent=4)
        logger.info(f"{orchestrator_log_prefix} Configuration AppConfig (partielle) sauvegardée: {orchestrator_config_save_path}")
    except Exception as e_save_cfg_orch:
        logger.error(f"{orchestrator_log_prefix} Échec sauvegarde config orchestrateur: {e_save_cfg_orch}", exc_info=True)

    wfo_tasks_to_run_args: List[Dict[str, Any]] = []
    if app_config.strategies_config and app_config.strategies_config.strategies:
        for strat_name_key, strat_params_config_instance in app_config.strategies_config.strategies.items():
            if strat_params_config_instance.active_for_optimization:
                task_args_for_wrapper = {
                    "project_root_str": str(project_root_path),
                    "orchestrator_run_id": orchestrator_run_id,
                    "strategy_name": strat_name_key,
                    "pair_symbol": args.pair.upper(),
                    "cli_context_label": args.tf
                }
                wfo_tasks_to_run_args.append(task_args_for_wrapper)
    
    if not wfo_tasks_to_run_args:
        logger.warning(f"{orchestrator_log_prefix} Aucune stratégie active pour optimisation. Arrêt.")
        sys.exit(0)

    logger.info(f"{orchestrator_log_prefix} Nb tâches WFO à exécuter: {len(wfo_tasks_to_run_args)}")

    max_parallel_tasks_from_config = app_config.global_config.optuna_settings.n_jobs
    max_workers_for_pool = max_parallel_tasks_from_config if max_parallel_tasks_from_config != -1 else None
    if max_parallel_tasks_from_config == 0:
        logger.warning(f"{orchestrator_log_prefix} n_jobs=0 invalide. Utilisation de 1 worker.")
        max_workers_for_pool = 1
    logger.info(f"{orchestrator_log_prefix} Max workers pour pool: {max_workers_for_pool or 'os.cpu_count()'}")
    
    if hasattr(signal, "SIGINT"): signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        try: signal.signal(signal.SIGTERM, signal_handler)
        except (OSError, AttributeError, ValueError) as e_sigterm_setup:
             logger.warning(f"{orchestrator_log_prefix} Impossible de configurer handler SIGTERM: {e_sigterm_setup}")

    task_execution_results: List[Dict[str, Any]] = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_for_pool) as executor:
            executor_instance = executor
            future_to_task_args_map = {
                executor.submit(run_wfo_task_wrapper, **task_args_item): task_args_item
                for task_args_item in wfo_tasks_to_run_args
            }
            logger.info(f"{orchestrator_log_prefix} {len(future_to_task_args_map)} tâches WFO soumises.")

            for future_item in concurrent.futures.as_completed(future_to_task_args_map):
                if orchestrator_shutdown_event.is_set():
                    logger.info(f"{orchestrator_log_prefix} Arrêt demandé. Annulation tâches restantes...")
                    for f_to_cancel in future_to_task_args_map:
                        if not f_to_cancel.done() and not f_to_cancel.running():
                            f_to_cancel.cancel()
                    break

                task_identifier_args = future_to_task_args_map[future_item]
                task_log_id_str = f"{task_identifier_args['strategy_name']}/{task_identifier_args['pair_symbol']}/{task_identifier_args['cli_context_label']}"
                try:
                    result_from_task = future_item.result()
                    task_execution_results.append(result_from_task)
                    logger.info(f"{orchestrator_log_prefix} Tâche WFO terminée pour {task_log_id_str}. Statut: {result_from_task.get('status')}")
                    # Log plus de détails si succès
                    if result_from_task.get('status', '').startswith("SUCCESS"):
                         logger.info(f"  -> Détails succès: TaskRunID={result_from_task.get('wfo_task_run_id')}, "
                                     f"LogDir={result_from_task.get('wfo_task_log_dir')}, "
                                     f"Résumé={result_from_task.get('wfo_strategy_pair_summary_file')}")
                    elif result_from_task.get('status','').startswith("FAILURE"):
                        logger.error(f"  -> Échec tâche {task_log_id_str}: {result_from_task.get('error_message', 'Erreur inconnue')}")

                except concurrent.futures.CancelledError:
                    logger.warning(f"{orchestrator_log_prefix} Tâche WFO pour {task_log_id_str} annulée.")
                    task_execution_results.append({"status": "CANCELLED_IN_ORCHESTRATOR", **task_identifier_args})
                except Exception as exc_task_future:
                    logger.error(f"{orchestrator_log_prefix} Tâche WFO {task_log_id_str} a levé une exception: {exc_task_future}", exc_info=True)
                    task_execution_results.append({"status": "EXCEPTION_IN_ORCHESTRATOR", "error_message": str(exc_task_future), **task_identifier_args})
                
    except KeyboardInterrupt:
        logger.info(f"{orchestrator_log_prefix} Interruption clavier (Ctrl+C). Arrêt global demandé...")
        orchestrator_shutdown_event.set()
    except Exception as e_executor_main:
        logger.critical(f"{orchestrator_log_prefix} Erreur ProcessPoolExecutor: {e_executor_main}", exc_info=True)
        orchestrator_shutdown_event.set()
    finally:
        if executor_instance:
            logger.info(f"{orchestrator_log_prefix} Bloc finally: Attente de la fin du ProcessPoolExecutor (si pas déjà arrêté).")
            # Le `with` s'occupe du shutdown(wait=True)
        executor_instance = None
        logger.debug(f"{orchestrator_log_prefix} Instance globale ProcessPoolExecutor remise à None.")

    logger.info(f"{orchestrator_log_prefix} --- Résumé Orchestration (Run ID: {orchestrator_run_id}) ---")
    # ... (logging des résultats comme avant) ...
    successful_wfo_tasks = [res for res in task_execution_results if res.get("status", "").startswith("SUCCESS")]
    logger.info(f"{orchestrator_log_prefix} Tâches WFO soumises: {len(wfo_tasks_to_run_args)}, Résultats reçus: {len(task_execution_results)}, Succès: {len(successful_wfo_tasks)}")


    orchestrator_summary_file_path = orchestrator_run_log_dir / "orchestrator_run_summary.json"
    try:
        # ... (sauvegarde du résumé de l'orchestrateur comme avant) ...
        summary_content_orch = {
            "orchestrator_run_id": orchestrator_run_id,
            "cli_arguments_used": vars(args),
            "project_root_used": str(project_root_path),
            # ... autres champs ...
            "wfo_task_execution_results": task_execution_results
        }
        with open(orchestrator_summary_file_path, 'w', encoding='utf-8') as f_summary_orch:
            json.dump(summary_content_orch, f_summary_orch, cls=EnhancedJSONEncoder, indent=4)
        logger.info(f"{orchestrator_log_prefix} Résumé orchestration sauvegardé: {orchestrator_summary_file_path}")
    except Exception as e_save_summary_orch:
        logger.error(f"{orchestrator_log_prefix} Échec sauvegarde résumé orchestration: {e_save_summary_orch}", exc_info=True)

    if successful_wfo_tasks and app_config: # S'assurer qu'app_config est bien défini
        logger.info(f"{orchestrator_log_prefix} Démarrage génération rapports finaux consolidés...")
        try:
            generate_all_reports(
                log_dir=orchestrator_run_log_dir,
                results_dir=orchestrator_run_results_dir,
                app_config=app_config
            )
            logger.info(f"{orchestrator_log_prefix} Génération rapports finaux terminée. Disponibles dans: {orchestrator_run_results_dir.resolve()}")
        except Exception as e_gen_reports_main:
            logger.error(f"{orchestrator_log_prefix} Erreur génération rapports finaux: {e_gen_reports_main}", exc_info=True)
    else:
        logger.warning(f"{orchestrator_log_prefix} Aucune tâche WFO avec résumé réussi ou AppConfig manquante. Génération rapports finaux sautée.")

    run_end_time = time.time()
    total_duration_seconds_script = run_end_time - run_start_time
    logger.info(f"{orchestrator_log_prefix} Temps Exécution Total Orchestrateur: {total_duration_seconds_script:.2f}s ({timedelta(seconds=total_duration_seconds_script)})")
    logger.info(f"{orchestrator_log_prefix} --- Fin Orchestrateur WFO (Run ID: {orchestrator_run_id}) ---")
    logging.shutdown()

if __name__ == "__main__":
    main()
