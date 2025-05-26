# run_optimize_backtest.py
"""
Orchestrateur principal pour le processus d'Optimisation Walk-Forward (WFO).
Ce script charge la configuration, identifie les tâches WFO à exécuter
(basées sur les stratégies actives et les arguments CLI pour la paire et le contexte),
et lance ces tâches en parallèle en utilisant multiprocessing.
"""

import argparse
import logging
import sys
import time
import json
import dataclasses # Pour dataclasses.asdict (si on devait repasser strategy_config_dict)
from pathlib import Path
from datetime import datetime, timedelta # Pour le nom du répertoire de run
import concurrent.futures
import signal
import os # Pour sys.path
from typing import Any, Dict, List, Optional, Tuple, Callable # Callable retiré car non utilisé ici
import threading
import numpy as np # Pour EnhancedJSONEncoder

# Détection de la racine du projet et ajout de src au PYTHONPATH
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(".").resolve() 

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Configuration initiale du logging (sera surchargée par load_all_configs)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Imports des modules de l'application
try:
    from src.config.loader import load_all_configs, AppConfig
    from src.config.definitions import StrategyParamsConfig 
    from src.utils.run_utils import generate_run_id, ensure_dir_exists
    # Importer la fonction wrapper réelle depuis task_executor
    from src.backtesting.wfo.task_executor import run_wfo_task_wrapper
except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE: Impossible d'importer les modules nécessaires: {e}. "
                    f"Vérifiez PYTHONPATH et les installations. CWD: {Path.cwd()}, sys.path: {sys.path}", exc_info=True)
    sys.exit(1)
except Exception as e_imp:
    logger.critical(f"ÉCHEC CRITIQUE: Erreur inattendue lors des imports initiaux: {e_imp}", exc_info=True)
    sys.exit(1)


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Encodeur JSON amélioré pour sérialiser les dataclasses, Path, datetime, et types numpy.
    """
    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o.as_posix()) # Utiliser as_posix pour des chemins standardisés
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
        try:
            return super().default(o)
        except TypeError:
            logger.warning(f"Type non sérialisable rencontré par EnhancedJSONEncoder: {type(o)}. Rendu comme chaîne.")
            return str(o)

# Variable globale pour le ProcessPoolExecutor pour une gestion propre des signaux
executor_instance: Optional[concurrent.futures.ProcessPoolExecutor] = None
orchestrator_shutdown_event = threading.Event() # Événement pour l'arrêt de l'orchestrateur

def signal_handler(signum, frame):
    """Gère SIGINT (Ctrl+C) et SIGTERM pour un arrêt propre."""
    global executor_instance, orchestrator_shutdown_event
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') and isinstance(signum, int) else str(signum)
    
    if orchestrator_shutdown_event.is_set():
        logger.info(f"Signal {signal_name} reçu, mais l'arrêt est déjà en cours.")
        return

    logger.info(f"Signal {signal_name} reçu. Demande d'arrêt global de l'orchestrateur...")
    orchestrator_shutdown_event.set() # Signaler à la boucle principale de s'arrêter

    if executor_instance:
        logger.info("Arrêt du ProcessPoolExecutor depuis le gestionnaire de signaux...")
        # Tenter d'annuler les futures si Python 3.9+
        # Le shutdown est appelé avec wait=False pour ne pas bloquer le signal handler trop longtemps.
        # La boucle principale gérera l'attente de la fin des threads.
        if sys.version_info >= (3, 9):
            executor_instance.shutdown(wait=False, cancel_futures=True) 
        else:
            executor_instance.shutdown(wait=False)
        logger.info("Demande d'arrêt envoyée au ProcessPoolExecutor.")
    # Ne pas appeler sys.exit() ici pour laisser la boucle principale gérer la fin propre.

def main():
    """
    Fonction principale pour orchestrer l'Optimisation Walk-Forward.
    """
    global executor_instance, orchestrator_shutdown_event 

    orchestrator_log_prefix = "[WFO_Orchestrator]" 
    run_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Orchestrateur pour l'Optimisation Walk-Forward (WFO).")
    parser.add_argument(
        "--pair",
        type=str,
        required=True,
        help="Paire de trading à optimiser (ex: BTCUSDT)."
    )
    parser.add_argument(
        "--tf", "--context", dest="tf",
        type=str,
        required=True,
        help="Label de contexte principal pour cette exécution (ex: 5min_rsi_filter)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Répertoire racine du projet si le script n'est pas exécuté depuis la racine."
    )
    args = parser.parse_args()

    project_root_arg = args.root if args.root else str(PROJECT_ROOT)
    project_root_path = Path(project_root_arg).resolve()

    app_config: Optional[AppConfig] = None
    try:
        app_config = load_all_configs(project_root=str(project_root_path))
        logger.info(f"{orchestrator_log_prefix} --- Démarrage de l'Orchestrateur d'Optimisation Walk-Forward ---")
        logger.info(f"{orchestrator_log_prefix} Arguments du script: Paire='{args.pair}', Contexte CLI (tf)='{args.tf}', Racine Projet='{project_root_path}'")
        logger.info(f"{orchestrator_log_prefix} Configuration de l'application chargée avec succès.")
    except Exception as e_conf:
        logger.critical(f"{orchestrator_log_prefix} Erreur critique lors du chargement de la configuration: {e_conf}", exc_info=True)
        sys.exit(1)

    if not app_config:
        logger.critical(f"{orchestrator_log_prefix} AppConfig n'a pas pu être chargée. Abandon.")
        sys.exit(1)

    orchestrator_run_id_prefix = app_config.global_config.simulation_defaults.run_name_prefix + "_orchestrator" \
        if hasattr(app_config.global_config.simulation_defaults, 'run_name_prefix') \
        else "wfo_orchestrator"
    
    orchestrator_run_id = generate_run_id(prefix=orchestrator_run_id_prefix)
    orchestrator_base_log_dir = Path(app_config.global_config.paths.logs_backtest_optimization)
    orchestrator_run_output_dir = orchestrator_base_log_dir / orchestrator_run_id
    
    try:
        ensure_dir_exists(orchestrator_run_output_dir)
        logger.info(f"{orchestrator_log_prefix} Répertoire de run de l'orchestrateur créé : {orchestrator_run_output_dir}")
    except Exception as e_mkdir:
        logger.critical(f"{orchestrator_log_prefix} Échec de la création du répertoire de run de l'orchestrateur {orchestrator_run_output_dir}: {e_mkdir}", exc_info=True)
        sys.exit(1)

    orchestrator_config_save_path = orchestrator_run_output_dir / "run_config_orchestrator.json"
    try:
        with open(orchestrator_config_save_path, 'w', encoding='utf-8') as f_cfg:
            json.dump(app_config, f_cfg, cls=EnhancedJSONEncoder, indent=4)
        logger.info(f"{orchestrator_log_prefix} Configuration AppConfig de l'orchestrateur sauvegardée dans : {orchestrator_config_save_path}")
    except Exception as e_save_cfg:
        logger.error(f"{orchestrator_log_prefix} Échec de la sauvegarde de la configuration de l'orchestrateur : {e_save_cfg}", exc_info=True)

    wfo_tasks_args: List[Dict[str, Any]] = []
    if app_config.strategies_config and app_config.strategies_config.strategies:
        for strat_name, strat_params_config_obj in app_config.strategies_config.strategies.items():
            if strat_params_config_obj.active_for_optimization:
                task_args = {
                    "project_root_str": str(project_root_path),
                    "orchestrator_run_id": orchestrator_run_id,
                    "strategy_name": strat_name,
                    "pair_symbol": args.pair.upper(),
                    "cli_context_label": args.tf
                }
                wfo_tasks_args.append(task_args)
    
    if not wfo_tasks_args:
        logger.warning(f"{orchestrator_log_prefix} Aucune stratégie active pour l'optimisation trouvée. Arrêt.")
        sys.exit(0)

    logger.info(f"{orchestrator_log_prefix} Nombre de tâches WFO à exécuter : {len(wfo_tasks_args)}")

    max_parallel_tasks = app_config.global_config.optuna_settings.n_jobs
    max_workers_val = max_parallel_tasks if max_parallel_tasks != -1 else None
    logger.info(f"{orchestrator_log_prefix} Utilisation de max_workers={max_workers_val if max_workers_val is not None else 'os.cpu_count()'} pour ProcessPoolExecutor.")
    
    if hasattr(signal, "SIGINT"): signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        try: signal.signal(signal.SIGTERM, signal_handler)
        except (OSError, AttributeError, ValueError): logger.warning("Impossible de configurer SIGTERM.")

    task_results: List[Dict[str, Any]] = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_val) as executor:
            executor_instance = executor 
            
            futures_map = {
                executor.submit(run_wfo_task_wrapper, **task_args): task_args
                for task_args in wfo_tasks_args
            }

            for future in concurrent.futures.as_completed(futures_map):
                if orchestrator_shutdown_event.is_set():
                    logger.info(f"{orchestrator_log_prefix} Arrêt demandé. Annulation des tâches restantes non démarrées.")
                    for f_to_cancel in futures_map:
                        if not f_to_cancel.done() and not f_to_cancel.running():
                            f_to_cancel.cancel()
                    break 

                task_desc_args = futures_map[future]
                task_identity_log = f"{task_desc_args['strategy_name']}/{task_desc_args['pair_symbol']}/{task_desc_args['cli_context_label']}"
                try:
                    result = future.result() 
                    task_results.append(result)
                    logger.info(f"{orchestrator_log_prefix} Tâche WFO terminée pour {task_identity_log}. Statut: {result.get('status')}")
                    if result.get('status') == "SUCCESS":
                        logger.info(f"  -> Run ID tâche WFO: {result.get('wfo_task_run_id')}, LogDir: {result.get('wfo_task_log_dir')}")
                    elif result.get('status') != "SUCCESS":
                        logger.error(f"  -> Échec tâche WFO {task_identity_log}: {result.get('error_message', 'Erreur inconnue')}")
                except concurrent.futures.CancelledError:
                    logger.warning(f"{orchestrator_log_prefix} Tâche WFO pour {task_identity_log} a été annulée.")
                    task_results.append({"status": "CANCELLED_IN_ORCHESTRATOR", **task_desc_args})
                except Exception as exc: 
                    logger.error(f"{orchestrator_log_prefix} Tâche WFO {task_identity_log} a généré une exception non gérée : {exc}", exc_info=True)
                    task_results.append({"status": "EXCEPTION_IN_TASK_PROCESS", "error_message": str(exc), **task_desc_args})
                
    except KeyboardInterrupt:
        logger.info(f"{orchestrator_log_prefix} Interruption clavier (Ctrl+C) reçue. Demande d'arrêt global...")
        orchestrator_shutdown_event.set() 
    except Exception as e_executor:
        logger.critical(f"{orchestrator_log_prefix} Erreur inattendue au niveau ProcessPoolExecutor : {e_executor}", exc_info=True)
        orchestrator_shutdown_event.set() 
    finally:
        # Le `with` statement appelle executor.shutdown(wait=True).
        # Le signal_handler appelle executor_instance.shutdown(wait=False, cancel_futures=True).
        # Ce bloc `finally` est surtout pour s'assurer que `executor_instance` est remis à None.
        if executor_instance:
            logger.info(f"{orchestrator_log_prefix} Bloc finally : executor_instance (global) sera remis à None.")
        executor_instance = None 
        logger.info(f"{orchestrator_log_prefix} executor_instance (global) remis à None.")

    logger.info(f"{orchestrator_log_prefix} --- Résumé de l'Orchestration WFO ---")
    successful_tasks = [res for res in task_results if res.get("status") == "SUCCESS"]
    failed_tasks = [res for res in task_results if res.get("status") != "SUCCESS"]

    logger.info(f"{orchestrator_log_prefix} Tâches WFO traitées (résultats reçus) : {len(task_results)}")
    logger.info(f"  Tâches réussies : {len(successful_tasks)}")
    for res_succ in successful_tasks:
        logger.info(f"    - Strat: {res_succ['strategy_name']}, Paire: {res_succ['pair_symbol']}, Ctx: {res_succ['cli_context_label']}, TaskRunID: {res_succ.get('wfo_task_run_id')}")
    
    if failed_tasks:
        logger.error(f"  Tâches échouées/annulées/avec erreurs : {len(failed_tasks)}")
        for res_fail in failed_tasks:
            logger.error(f"    - Strat: {res_fail.get('strategy_name')}, Paire: {res_fail.get('pair_symbol')}, Ctx: {res_fail.get('cli_context_label')}, Statut: {res_fail.get('status')}, Err: {res_fail.get('error_message', 'N/A')}")
    
    orchestrator_summary_path = orchestrator_run_output_dir / "orchestrator_summary.json"
    try:
        with open(orchestrator_summary_path, 'w', encoding='utf-8') as f_summary:
            json.dump({
                "orchestrator_run_id": orchestrator_run_id,
                "cli_args": vars(args),
                "num_tasks_submitted": len(wfo_tasks_args),
                "num_tasks_results_received": len(task_results),
                "num_successful_tasks": len(successful_tasks),
                "num_failed_or_cancelled_tasks": len(failed_tasks),
                "task_results_summary": task_results 
            }, f_summary, cls=EnhancedJSONEncoder, indent=4)
        logger.info(f"{orchestrator_log_prefix} Résumé de l'orchestration sauvegardé dans : {orchestrator_summary_path}")
    except Exception as e_save_summary:
        logger.error(f"{orchestrator_log_prefix} Échec de la sauvegarde du résumé de l'orchestration : {e_save_summary}", exc_info=True)

    run_end_time = time.time()
    total_duration_seconds = run_end_time - run_start_time
    logger.info(f"{orchestrator_log_prefix} Temps d'Exécution Total de l'Orchestrateur : {total_duration_seconds:.2f} secondes ({timedelta(seconds=total_duration_seconds)})")
    logger.info(f"{orchestrator_log_prefix} --- Fin de l'Orchestrateur d'Optimisation Walk-Forward (Run ID: {orchestrator_run_id}) ---")
    logging.shutdown()

if __name__ == "__main__":
    main()
