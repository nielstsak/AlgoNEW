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
import dataclasses # Pour dataclasses.asdict pour sauvegarder AppConfig
from pathlib import Path
from datetime import datetime, timedelta, timezone # timezone ajouté pour run_id
import concurrent.futures
import signal
import os # Pour sys.path
from typing import Any, Dict, List, Optional, Tuple # Callable retiré car non utilisé directement ici
import threading # Pour orchestrator_shutdown_event

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
    # Le logger n'est pas encore configuré ici, donc print pour ce message initial
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
    from src.utils.run_utils import generate_run_id, ensure_dir_exists, _sanitize_path_component # _sanitize_path_component pour le run_id_prefix
    from src.backtesting.wfo.task_executor import run_wfo_task_wrapper
    # Importer generate_all_reports pour l'appel final
    from src.reporting.master_report_generator import generate_all_reports

except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE (run_optimize_backtest.py): Impossible d'importer les modules nécessaires: {e}. "
                    f"Vérifiez PYTHONPATH et les installations. CWD: {Path.cwd()}, sys.path: {sys.path}", exc_info=True)
    sys.exit(1)
except Exception as e_imp: # pylint: disable=broad-except
    logger.critical(f"ÉCHEC CRITIQUE (run_optimize_backtest.py): Erreur inattendue lors des imports initiaux: {e_imp}", exc_info=True)
    sys.exit(1)


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Encodeur JSON amélioré pour sérialiser les dataclasses, Path, datetime, et types numpy.
    """
    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o.as_posix())
        if isinstance(o, (datetime, pd.Timestamp)): 
            return o.isoformat()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            if np.isnan(o): return None # Sérialiser NaN comme null
            if np.isinf(o): return "Infinity" if o > 0 else "-Infinity" # Sérialiser Infini
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        try:
            return super().default(o)
        except TypeError:
            # Pour les types non gérés, tenter une conversion en chaîne
            logger.debug(f"Type non sérialisable rencontré par EnhancedJSONEncoder: {type(o)}. Rendu comme chaîne.")
            return str(o)

# Variable globale pour le ProcessPoolExecutor pour une gestion propre des signaux
executor_instance: Optional[concurrent.futures.ProcessPoolExecutor] = None
orchestrator_shutdown_event = threading.Event() # Événement pour l'arrêt de l'orchestrateur

def signal_handler(signum, frame):
    """Gère SIGINT (Ctrl+C) et SIGTERM pour un arrêt propre."""
    global executor_instance, orchestrator_shutdown_event
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') and isinstance(signum, int) else str(signum)
    
    if orchestrator_shutdown_event.is_set():
        logger.info(f"Signal {signal_name} reçu, mais l'arrêt de l'orchestrateur est déjà en cours.")
        return

    logger.info(f"Signal {signal_name} reçu. Demande d'arrêt global de l'orchestrateur...")
    orchestrator_shutdown_event.set()

    if executor_instance:
        logger.info("Arrêt du ProcessPoolExecutor depuis le gestionnaire de signaux...")
        # Tenter d'annuler les futures si Python 3.9+
        # Le shutdown est appelé avec wait=False pour ne pas bloquer le signal handler.
        if sys.version_info >= (3, 9):
            executor_instance.shutdown(wait=False, cancel_futures=True) 
        else:
            executor_instance.shutdown(wait=False)
        logger.info("Demande d'arrêt envoyée au ProcessPoolExecutor.")
    # La boucle principale gérera l'attente de la fin des threads/processus.

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
        help="Label de contexte principal pour cette exécution (ex: 5min_rsi_filter). "
             "Ce contexte est utilisé pour nommer les répertoires et identifier le run."
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

    # Charger la configuration de l'application. load_all_configs configure aussi le logging.
    app_config: Optional[AppConfig] = None
    try:
        app_config = load_all_configs(project_root=str(project_root_path))
        logger.info(f"{orchestrator_log_prefix} --- Démarrage de l'Orchestrateur d'Optimisation Walk-Forward ---")
        logger.info(f"{orchestrator_log_prefix} Arguments du script: Paire='{args.pair}', Contexte CLI (tf)='{args.tf}', Racine Projet='{project_root_path}'")
        logger.info(f"{orchestrator_log_prefix} Configuration de l'application chargée avec succès.")
    except Exception as e_conf:
        logger.critical(f"{orchestrator_log_prefix} Erreur critique lors du chargement de la configuration: {e_conf}", exc_info=True)
        sys.exit(1)

    if not app_config: # Double vérification, load_all_configs devrait lever une exception avant
        logger.critical(f"{orchestrator_log_prefix} AppConfig n'a pas pu être chargée. Abandon.")
        sys.exit(1)

    # Générer un ID de run unique pour cet orchestrateur
    run_name_prefix_cfg = app_config.global_config.simulation_defaults.run_name_prefix
    sanitized_pair_for_run_id = _sanitize_path_component(args.pair.upper())
    sanitized_context_for_run_id = _sanitize_path_component(args.tf)
    orchestrator_run_id_prefix = f"{run_name_prefix_cfg}_orch_{sanitized_pair_for_run_id}_{sanitized_context_for_run_id}"
    
    orchestrator_run_id = generate_run_id(prefix=orchestrator_run_id_prefix)
    logger.info(f"{orchestrator_log_prefix} ID de Run de l'Orchestrateur : {orchestrator_run_id}")

    # Créer le répertoire de base pour les logs et résultats de cet orchestrateur
    # Le répertoire de log de l'orchestrateur contiendra les logs de l'orchestrateur lui-même
    # et sera le parent des répertoires de log des tâches WFO.
    orchestrator_base_log_dir_path = Path(app_config.global_config.paths.logs_backtest_optimization)
    orchestrator_run_log_dir = orchestrator_base_log_dir_path / orchestrator_run_id
    
    # Le répertoire de résultats de l'orchestrateur contiendra les rapports finaux.
    orchestrator_base_results_dir_path = Path(app_config.global_config.paths.results)
    orchestrator_run_results_dir = orchestrator_base_results_dir_path / orchestrator_run_id

    try:
        ensure_dir_exists(orchestrator_run_log_dir)
        ensure_dir_exists(orchestrator_run_results_dir)
        logger.info(f"{orchestrator_log_prefix} Répertoire de log du run de l'orchestrateur créé : {orchestrator_run_log_dir}")
        logger.info(f"{orchestrator_log_prefix} Répertoire de résultats du run de l'orchestrateur créé : {orchestrator_run_results_dir}")
    except Exception as e_mkdir_orch:
        logger.critical(f"{orchestrator_log_prefix} Échec de la création des répertoires de run de l'orchestrateur: {e_mkdir_orch}", exc_info=True)
        sys.exit(1)

    # Sauvegarder la configuration AppConfig utilisée pour ce run de l'orchestrateur
    orchestrator_config_save_path = orchestrator_run_log_dir / "run_config_orchestrator.json"
    try:
        with open(orchestrator_config_save_path, 'w', encoding='utf-8') as f_cfg_orch:
            json.dump(app_config, f_cfg_orch, cls=EnhancedJSONEncoder, indent=4)
        logger.info(f"{orchestrator_log_prefix} Configuration AppConfig de l'orchestrateur sauvegardée dans : {orchestrator_config_save_path}")
    except Exception as e_save_cfg_orch:
        logger.error(f"{orchestrator_log_prefix} Échec de la sauvegarde de la configuration de l'orchestrateur : {e_save_cfg_orch}", exc_info=True)

    # Identifier les tâches WFO à exécuter
    wfo_tasks_to_run_args: List[Dict[str, Any]] = []
    if app_config.strategies_config and app_config.strategies_config.strategies:
        for strat_name_key, strat_params_config_instance in app_config.strategies_config.strategies.items():
            if strat_params_config_instance.active_for_optimization:
                # Les arguments pour run_wfo_task_wrapper
                task_args_for_wrapper = {
                    "project_root_str": str(project_root_path),
                    "orchestrator_run_id": orchestrator_run_id,
                    "strategy_name": strat_name_key,
                    "pair_symbol": args.pair.upper(), # Utiliser la paire de la CLI
                    "cli_context_label": args.tf # Utiliser le contexte de la CLI
                }
                wfo_tasks_to_run_args.append(task_args_for_wrapper)
    
    if not wfo_tasks_to_run_args:
        logger.warning(f"{orchestrator_log_prefix} Aucune stratégie active pour l'optimisation trouvée pour la paire '{args.pair}' et le contexte '{args.tf}'. Arrêt.")
        sys.exit(0)

    logger.info(f"{orchestrator_log_prefix} Nombre de tâches WFO (stratégie/paire/contexte) à exécuter : {len(wfo_tasks_to_run_args)}")

    # Configurer le nombre de workers pour ProcessPoolExecutor
    max_parallel_tasks_from_config = app_config.global_config.optuna_settings.n_jobs
    # Si n_jobs est -1, ProcessPoolExecutor utilisera os.cpu_count(). Si 0, c'est une erreur.
    max_workers_for_pool = max_parallel_tasks_from_config if max_parallel_tasks_from_config != -1 else None
    if max_parallel_tasks_from_config == 0:
        logger.warning(f"{orchestrator_log_prefix} n_jobs est 0 dans OptunaSettings, ce qui est invalide pour ProcessPoolExecutor. Utilisation de 1 worker.")
        max_workers_for_pool = 1
    logger.info(f"{orchestrator_log_prefix} Utilisation de max_workers={max_workers_for_pool if max_workers_for_pool is not None else 'os.cpu_count()'} pour ProcessPoolExecutor.")
    
    # Configurer les gestionnaires de signaux
    if hasattr(signal, "SIGINT"): signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        try: signal.signal(signal.SIGTERM, signal_handler)
        except (OSError, AttributeError, ValueError) as e_sigterm_setup: # Erreurs possibles sur certaines plateformes
             logger.warning(f"{orchestrator_log_prefix} Impossible de configurer le gestionnaire pour SIGTERM : {e_sigterm_setup}")

    task_execution_results: List[Dict[str, Any]] = []
    try:
        # Utiliser ProcessPoolExecutor pour exécuter les tâches en parallèle
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_for_pool) as executor:
            executor_instance = executor # Stocker l'instance pour le signal_handler
            
            # Soumettre toutes les tâches
            future_to_task_args_map = {
                executor.submit(run_wfo_task_wrapper, **task_args_item): task_args_item
                for task_args_item in wfo_tasks_to_run_args
            }

            logger.info(f"{orchestrator_log_prefix} Toutes les {len(future_to_task_args_map)} tâches WFO ont été soumises au pool de processus.")

            for future_item in concurrent.futures.as_completed(future_to_task_args_map):
                if orchestrator_shutdown_event.is_set():
                    logger.info(f"{orchestrator_log_prefix} Arrêt demandé. Annulation des tâches WFO restantes non démarrées...")
                    # Tenter d'annuler les futures qui n'ont pas encore commencé ou qui ne tournent pas
                    for f_to_cancel in future_to_task_args_map: # Itérer sur les clés (futures)
                        if not f_to_cancel.done() and not f_to_cancel.running():
                            f_to_cancel.cancel()
                    break # Sortir de la boucle as_completed

                task_identifier_args = future_to_task_args_map[future_item]
                task_log_identity_str = f"{task_identifier_args['strategy_name']}/{task_identifier_args['pair_symbol']}/{task_identifier_args['cli_context_label']}"
                try:
                    result_from_task = future_item.result() # Bloque jusqu'à ce que la tâche soit terminée
                    task_execution_results.append(result_from_task)
                    logger.info(f"{orchestrator_log_prefix} Tâche WFO terminée pour {task_log_identity_str}. Statut: {result_from_task.get('status')}")
                    if result_from_task.get('status') == "SUCCESS_WFO_COMPLETED" or "SUCCESS" in result_from_task.get('status','').upper() :
                        logger.info(f"  -> Run ID tâche WFO: {result_from_task.get('wfo_task_run_id')}, "
                                    f"LogDir Tâche: {result_from_task.get('wfo_task_log_dir')}, "
                                    f"Fichier Résumé Tâche: {result_from_task.get('wfo_strategy_pair_summary_file')}")
                    elif result_from_task.get('status','').startswith("FAILURE"):
                        logger.error(f"  -> Échec tâche WFO {task_log_identity_str}: {result_from_task.get('error_message', 'Erreur inconnue dans la tâche')}")
                except concurrent.futures.CancelledError:
                    logger.warning(f"{orchestrator_log_prefix} Tâche WFO pour {task_log_identity_str} a été annulée (probablement pendant l'arrêt).")
                    task_execution_results.append({"status": "CANCELLED_IN_ORCHESTRATOR", **task_identifier_args})
                except Exception as exc_task_future: # Erreur non gérée dans le processus fils, ou erreur de communication
                    logger.error(f"{orchestrator_log_prefix} Tâche WFO {task_log_identity_str} a généré une exception non gérée au niveau de l'orchestrateur : {exc_task_future}", exc_info=True)
                    task_execution_results.append({"status": "EXCEPTION_ORCHESTRATOR_HANDLING_TASK", "error_message": str(exc_task_future), **task_identifier_args})
                
    except KeyboardInterrupt: # Attraper Ctrl+C ici aussi au cas où
        logger.info(f"{orchestrator_log_prefix} Interruption clavier (Ctrl+C) reçue par l'orchestrateur principal. Demande d'arrêt global...")
        orchestrator_shutdown_event.set() 
    except Exception as e_executor_main: # Erreur inattendue au niveau du ProcessPoolExecutor lui-même
        logger.critical(f"{orchestrator_log_prefix} Erreur inattendue au niveau du ProcessPoolExecutor : {e_executor_main}", exc_info=True)
        orchestrator_shutdown_event.set() # Déclencher un arrêt en cas d'erreur grave
    finally:
        # Le `with` statement appelle executor.shutdown(wait=True) par défaut.
        # Le signal_handler appelle executor_instance.shutdown(wait=False, cancel_futures=True).
        # Ce bloc `finally` est surtout pour s'assurer que `executor_instance` est remis à None.
        if executor_instance:
            logger.info(f"{orchestrator_log_prefix} Bloc finally : l'instance globale de ProcessPoolExecutor sera remise à None.")
        executor_instance = None 
        logger.debug(f"{orchestrator_log_prefix} Instance globale de ProcessPoolExecutor remise à None.")

    # --- Agrégation des Résultats et Génération du Rapport Final ---
    logger.info(f"{orchestrator_log_prefix} --- Résumé de l'Orchestration WFO (Run ID: {orchestrator_run_id}) ---")
    successful_wfo_tasks = [res for res in task_execution_results if res.get("status", "").startswith("SUCCESS")]
    failed_or_problematic_tasks = [res for res in task_execution_results if not res.get("status", "").startswith("SUCCESS")]

    logger.info(f"{orchestrator_log_prefix} Nombre total de tâches WFO soumises : {len(wfo_tasks_to_run_args)}")
    logger.info(f"{orchestrator_log_prefix} Nombre de résultats de tâches WFO reçus : {len(task_execution_results)}")
    logger.info(f"  Tâches WFO complétées avec succès (ou partiellement) : {len(successful_wfo_tasks)}")
    for res_succ_log in successful_wfo_tasks:
        logger.info(f"    - Strat: {res_succ_log['strategy_name']}, Paire: {res_succ_log['pair_symbol']}, Ctx: {res_succ_log['cli_context_label']}, TaskRunID: {res_succ_log.get('wfo_task_run_id')}, Statut: {res_succ_log.get('status')}")
    
    if failed_or_problematic_tasks:
        logger.error(f"  Tâches WFO échouées/annulées/avec erreurs : {len(failed_or_problematic_tasks)}")
        for res_fail_log in failed_or_problematic_tasks:
            logger.error(f"    - Strat: {res_fail_log.get('strategy_name')}, Paire: {res_fail_log.get('pair_symbol')}, Ctx: {res_fail_log.get('cli_context_label')}, "
                         f"TaskRunID: {res_fail_log.get('wfo_task_run_id', 'N/A')}, Statut: {res_fail_log.get('status')}, ErrMsg: {res_fail_log.get('error_message', 'N/A')}")
    
    # Sauvegarder un résumé de l'orchestration
    orchestrator_summary_file_path = orchestrator_run_log_dir / "orchestrator_run_summary.json"
    try:
        summary_content_orch = {
            "orchestrator_run_id": orchestrator_run_id,
            "cli_arguments_used": vars(args),
            "project_root_used": str(project_root_path),
            "num_wfo_tasks_submitted": len(wfo_tasks_to_run_args),
            "num_wfo_tasks_results_received": len(task_execution_results),
            "num_successful_wfo_tasks": len(successful_wfo_tasks),
            "num_failed_or_problematic_wfo_tasks": len(failed_or_problematic_tasks),
            "wfo_task_execution_results": task_execution_results # Liste des dictionnaires retournés par les wrappers
        }
        with open(orchestrator_summary_file_path, 'w', encoding='utf-8') as f_summary_orch:
            json.dump(summary_content_orch, f_summary_orch, cls=EnhancedJSONEncoder, indent=4)
        logger.info(f"{orchestrator_log_prefix} Résumé de l'orchestration sauvegardé dans : {orchestrator_summary_file_path}")
    except Exception as e_save_summary_orch:
        logger.error(f"{orchestrator_log_prefix} Échec de la sauvegarde du résumé de l'orchestration : {e_save_summary_orch}", exc_info=True)

    # --- Génération des Rapports Finaux ---
    if successful_wfo_tasks: # Générer les rapports seulement si au moins une tâche a produit un résumé
        logger.info(f"{orchestrator_log_prefix} Démarrage de la génération des rapports finaux consolidés...")
        try:
            # generate_all_reports utilisera les fichiers wfo_strategy_pair_summary.json
            # qui sont dans les sous-répertoires de log des tâches (wfo_task_log_dir).
            # Le répertoire de log de l'orchestrateur (orchestrator_run_log_dir) est le parent de ces derniers.
            # Le répertoire de résultats de l'orchestrateur (orchestrator_run_results_dir) est où les rapports finaux iront.
            generate_all_reports(
                log_dir=orchestrator_run_log_dir, # Répertoire du run de l'orchestrateur contenant les logs des tâches
                results_dir=orchestrator_run_results_dir, # Répertoire où les rapports finaux seront écrits
                app_config=app_config
            )
            logger.info(f"{orchestrator_log_prefix} Génération des rapports finaux terminée. "
                        f"Rapports disponibles dans : {orchestrator_run_results_dir.resolve()}")
        except Exception as e_gen_reports_main:
            logger.error(f"{orchestrator_log_prefix} Erreur lors de la génération des rapports finaux : {e_gen_reports_main}", exc_info=True)
    else:
        logger.warning(f"{orchestrator_log_prefix} Aucune tâche WFO n'a produit de résumé avec succès. "
                       "La génération des rapports finaux consolidés est sautée.")


    run_end_time = time.time()
    total_duration_seconds_script = run_end_time - run_start_time
    logger.info(f"{orchestrator_log_prefix} Temps d'Exécution Total du Script Orchestrateur : {total_duration_seconds_script:.2f} secondes ({timedelta(seconds=total_duration_seconds_script)})")
    logger.info(f"{orchestrator_log_prefix} --- Fin de l'Orchestrateur d'Optimisation Walk-Forward (Run ID: {orchestrator_run_id}) ---")
    logging.shutdown() # S'assurer que tous les messages de log sont écrits avant la sortie du script principal

if __name__ == "__main__":
    # Cette condition est cruciale pour le bon fonctionnement de multiprocessing sur Windows.
    # Elle assure que le code de création de processus n'est exécuté que lorsque le script
    # est le module principal, et non lorsqu'il est importé par un processus fils.
    main()
