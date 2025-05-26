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
import dataclasses
from pathlib import Path
from datetime import datetime, timedelta, timezone
import concurrent.futures
import signal
import os
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np # Pour EnhancedJSONEncoder

# Détection de la racine du projet et ajout de src au PYTHONPATH
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(".").resolve() # Fallback si __file__ n'est pas défini

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
    from src.config.definitions import StrategyParamsConfig # Pour le typage
    from src.utils.run_utils import generate_run_id, ensure_dir_exists # Utilisation de run_utils
    # La fonction run_wfo_task_wrapper sera importée dynamiquement ou définie en placeholder
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
            return str(o)
        if isinstance(o, (datetime, pd.Timestamp)): # Gérer pd.Timestamp aussi
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

# --- Placeholder pour run_wfo_task_wrapper ---
# Cette fonction serait normalement dans src.backtesting.wfo.task_executor
# et appellerait WalkForwardOptimizer().run(...)
def run_wfo_task_wrapper_placeholder(
    project_root_path_str: str,
    orchestrator_run_id: str,
    # orchestrator_run_output_dir_str: str, # Ce n'est plus nécessaire si WFO gère ses propres sorties
    strategy_name: str,
    strategy_config_dict: Dict[str, Any], # Contenu de StrategyParamsConfig
    pair_symbol: str,
    cli_context_label: str
) -> Dict[str, Any]:
    """
    Placeholder pour la fonction qui exécute une tâche WFO pour une combinaison
    stratégie/paire/contexte.
    Dans une implémentation réelle, cette fonction appellerait WalkForwardOptimizer.
    """
    # Recharger AppConfig dans le processus fils pour éviter les problèmes de pickling
    # et pour s'assurer que chaque tâche a une config fraîche (si la config changeait dynamiquement)
    task_log_prefix = f"[WFO_Task_Placeholder][{strategy_name}/{pair_symbol}/{cli_context_label}]"
    logger.info(f"{task_log_prefix} Démarrage de la tâche WFO (placeholder).")
    
    try:
        # Simuler le rechargement de la config
        app_config_task = load_all_configs(project_root=project_root_path_str)
        logger.info(f"{task_log_prefix} AppConfig rechargée dans le processus fils.")

        # Simuler l'exécution de WalkForwardOptimizer
        # Dans la vraie implémentation, on importerait et appellerait WalkForwardOptimizer
        from src.backtesting.wfo import WalkForwardOptimizer # Importation à l'intérieur de la tâche

        # WalkForwardOptimizer crée son propre run_id et répertoire de sortie
        # basé sur app_config_task.global_config.paths.logs_backtest_optimization.
        # Il n'a pas besoin de orchestrator_run_output_dir_str pour ses sorties principales.
        
        wfo_instance = WalkForwardOptimizer(app_config=app_config_task)
        logger.info(f"{task_log_prefix} WalkForwardOptimizer instancié. Son run_id sera: {wfo_instance.run_id}")
        
        # Simuler un appel à wfo_instance.run()
        # wfo_instance.run() prend `pairs` et `context_labels` comme listes.
        # Ici, nous avons une tâche spécifique.
        # Le `WalkForwardOptimizer.run` actuel semble conçu pour boucler sur plusieurs stratégies/paires/contextes.
        # Pour une tâche unique, il faudrait soit l'adapter, soit appeler une méthode plus granulaire.
        # Supposons que WalkForwardOptimizer.run peut gérer une seule combinaison si on lui passe des listes unitaires.
        
        # Le contexte CLI est utilisé pour le nommage des répertoires par WalkForwardOptimizer
        wfo_task_results = wfo_instance.run(pairs=[pair_symbol], context_labels=[cli_context_label])
        
        # Le résultat de wfo_instance.run() est un dictionnaire.
        # On pourrait vouloir extraire le wfo_task_run_id spécifique généré par ce WalkForwardOptimizer.
        # Et un résumé des résultats.
        
        # Pour ce placeholder, on simule un succès.
        time.sleep(5) # Simuler du travail
        
        # Le `wfo_instance.run_id` est l'ID du run spécifique de cette tâche WFO.
        # Les résultats de cette tâche WFO seront dans `logs/backtest_optimization/{wfo_instance.run_id}`
        
        # Extraire des informations pertinentes du résultat de wfo_instance.run() si nécessaire.
        # wfo_task_results est un dict de la forme: {"strat_pair_context": wfo_summary_for_strategy_pair_context}
        summary_key = f"{strategy_name}_{pair_symbol}_{cli_context_label}" # Le contexte utilisé par WFO est celui passé
        task_summary_data = wfo_task_results.get(summary_key)
        
        num_folds_processed = 0
        if task_summary_data and isinstance(task_summary_data.get("folds_data"), list):
            num_folds_processed = len(task_summary_data["folds_data"])

        logger.info(f"{task_log_prefix} Tâche WFO (placeholder) terminée. Run ID de la tâche WFO: {wfo_instance.run_id}. Folds traités: {num_folds_processed}.")
        return {
            "status": "SUCCESS",
            "strategy": strategy_name,
            "pair": pair_symbol,
            "context": cli_context_label,
            "wfo_task_run_id": wfo_instance.run_id, # ID du run généré par WalkForwardOptimizer
            "message": f"Tâche WFO simulée terminée avec succès pour {strategy_name} sur {pair_symbol} ({cli_context_label}). {num_folds_processed} folds traités.",
            "output_path_wfo_task": str(wfo_instance.run_output_dir.resolve())
        }
    except Exception as e:
        logger.error(f"{task_log_prefix} Erreur dans la tâche WFO (placeholder): {e}", exc_info=True)
        return {
            "status": "FAILURE",
            "strategy": strategy_name,
            "pair": pair_symbol,
            "context": cli_context_label,
            "wfo_task_run_id": None,
            "error_message": str(e),
            "output_path_wfo_task": None
        }

# Variable globale pour le ProcessPoolExecutor pour une gestion propre des signaux
executor_instance: Optional[concurrent.futures.ProcessPoolExecutor] = None

def signal_handler(signum, frame):
    """Gère SIGINT (Ctrl+C) et SIGTERM pour un arrêt propre."""
    global executor_instance
    logger.info(f"Signal {signal.Signals(signum).name if hasattr(signal, 'Signals') else signum} reçu. Tentative d'arrêt propre...")
    if executor_instance:
        logger.info("Arrêt du ProcessPoolExecutor...")
        # Tenter d'annuler les futures si Python 3.9+
        if sys.version_info >= (3, 9):
            executor_instance.shutdown(wait=True, cancel_futures=True) # type: ignore
        else:
            executor_instance.shutdown(wait=True)
        logger.info("ProcessPoolExecutor arrêté.")
    sys.exit(0) # Sortir proprement

def main():
    """
    Fonction principale pour orchestrer l'Optimisation Walk-Forward.
    """
    global executor_instance # Pour que signal_handler puisse y accéder

    run_start_time = time.time()
    # Le logging sera configuré par load_all_configs

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
        logger.info("--- Démarrage de l'Orchestrateur d'Optimisation Walk-Forward ---")
        logger.info(f"Arguments du script: Paire='{args.pair}', Contexte CLI (tf)='{args.tf}', Racine Projet='{project_root_path}'")
        logger.info("Configuration de l'application chargée avec succès.")
    except FileNotFoundError as e_fnf:
        logger.critical(f"Fichier de configuration non trouvé: {e_fnf}. "
                        f"Vérifiez les chemins relatifs à la racine du projet: {project_root_path}. Abandon.", exc_info=True)
        sys.exit(1)
    except Exception as e_conf:
        logger.critical(f"Erreur critique lors du chargement de la configuration: {e_conf}", exc_info=True)
        sys.exit(1)

    if not app_config:
        logger.critical("AppConfig n'a pas pu être chargée. Abandon.")
        sys.exit(1)

    # Créer un répertoire de run pour cet orchestrateur
    orchestrator_run_id_prefix = app_config.global_config.simulation_defaults.run_name_prefix + "_orchestrator" \
        if hasattr(app_config.global_config.simulation_defaults, 'run_name_prefix') \
        else "wfo_orchestrator"
    
    orchestrator_run_id = generate_run_id(prefix=orchestrator_run_id_prefix)
    orchestrator_base_log_dir = Path(app_config.global_config.paths.logs_backtest_optimization)
    orchestrator_run_output_dir = orchestrator_base_log_dir / orchestrator_run_id
    
    try:
        ensure_dir_exists(orchestrator_run_output_dir) # Utilise ensure_dir_exists de run_utils
        logger.info(f"Répertoire de run de l'orchestrateur créé : {orchestrator_run_output_dir}")
    except Exception as e_mkdir:
        logger.critical(f"Échec de la création du répertoire de run de l'orchestrateur {orchestrator_run_output_dir}: {e_mkdir}", exc_info=True)
        sys.exit(1)

    # Sauvegarder AppConfig dans le répertoire du run de l'orchestrateur
    orchestrator_config_save_path = orchestrator_run_output_dir / "run_config_orchestrator.json"
    try:
        with open(orchestrator_config_save_path, 'w', encoding='utf-8') as f_cfg:
            json.dump(app_config, f_cfg, cls=EnhancedJSONEncoder, indent=4)
        logger.info(f"Configuration AppConfig de l'orchestrateur sauvegardée dans : {orchestrator_config_save_path}")
    except Exception as e_save_cfg:
        logger.error(f"Échec de la sauvegarde de la configuration de l'orchestrateur : {e_save_cfg}", exc_info=True)

    # Identifier les tâches WFO
    wfo_tasks_args: List[Dict[str, Any]] = []
    if app_config.strategies_config and app_config.strategies_config.strategies:
        for strat_name, strat_params_config in app_config.strategies_config.strategies.items():
            if strat_params_config.active_for_optimization:
                task_args = {
                    "project_root_path_str": str(project_root_path),
                    "orchestrator_run_id": orchestrator_run_id,
                    "strategy_name": strat_name,
                    "strategy_config_dict": dataclasses.asdict(strat_params_config), # Passer en dict
                    "pair_symbol": args.pair.upper(),
                    "cli_context_label": args.tf
                }
                wfo_tasks_args.append(task_args)
    
    if not wfo_tasks_args:
        logger.warning("Aucune stratégie active pour l'optimisation trouvée pour la paire/contexte spécifiée. Arrêt.")
        sys.exit(0)

    logger.info(f"Nombre de tâches WFO à exécuter : {len(wfo_tasks_args)}")

    # Configurer le nombre maximum de workers pour ProcessPoolExecutor
    # Utiliser optuna_settings.n_jobs, mais s'assurer qu'il est positif pour max_workers
    # Si n_jobs est -1 (tous les CPUs), ProcessPoolExecutor le gère en passant None à max_workers.
    max_parallel_tasks = app_config.global_config.optuna_settings.n_jobs
    if max_parallel_tasks == 0: # 0 n'est pas valide pour ProcessPoolExecutor
        logger.warning("optuna_settings.n_jobs est 0, utilisation de 1 worker pour les tâches WFO.")
        max_parallel_tasks = 1
    elif max_parallel_tasks < -1: # Valeur négative autre que -1
        logger.warning(f"optuna_settings.n_jobs est {max_parallel_tasks}, utilisation de os.cpu_count() workers pour les tâches WFO.")
        max_parallel_tasks = None # os.cpu_count()
    
    logger.info(f"Utilisation de max_workers={max_parallel_tasks if max_parallel_tasks != -1 else 'os.cpu_count()'} pour ProcessPoolExecutor.")
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        try:
            signal.signal(signal.SIGTERM, signal_handler)
        except OSError:
            logger.warning("Impossible de configurer le gestionnaire pour SIGTERM sur cette plateforme.")

    task_results: List[Dict[str, Any]] = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel_tasks if max_parallel_tasks != -1 else None) as executor:
            executor_instance = executor # Rendre l'instance accessible au signal_handler
            
            futures_map = {
                executor.submit(run_wfo_task_wrapper_placeholder, **task_args): task_args
                for task_args in wfo_tasks_args
            }

            for future in concurrent.futures.as_completed(futures_map):
                task_desc = futures_map[future]
                try:
                    result = future.result()
                    task_results.append(result)
                    logger.info(f"Tâche WFO terminée pour {task_desc['strategy_name']}/{task_desc['pair_symbol']}/{task_desc['cli_context_label']}. Statut: {result.get('status')}")
                    if result.get('status') == "SUCCESS":
                        logger.info(f"  -> Run ID de la tâche WFO: {result.get('wfo_task_run_id')}, Chemin de sortie: {result.get('output_path_wfo_task')}")
                    elif result.get('status') == "FAILURE":
                        logger.error(f"  -> Échec de la tâche WFO: {result.get('error_message')}")
                except Exception as exc:
                    logger.error(f"Tâche WFO {task_desc['strategy_name']}/{task_desc['pair_symbol']}/{task_desc['cli_context_label']} a généré une exception : {exc}", exc_info=True)
                    task_results.append({
                        "status": "EXCEPTION_IN_ORCHESTRATOR",
                        "strategy": task_desc['strategy_name'],
                        "pair": task_desc['pair_symbol'],
                        "context": task_desc['cli_context_label'],
                        "error_message": str(exc)
                    })
    except KeyboardInterrupt:
        logger.info("Interruption clavier reçue par l'orchestrateur principal. Arrêt demandé.")
        # Le signal_handler devrait déjà avoir été appelé et géré l'arrêt de l'executor.
    except Exception as e_executor:
        logger.critical(f"Erreur inattendue au niveau du ProcessPoolExecutor : {e_executor}", exc_info=True)
    finally:
        if executor_instance and not executor_instance._shutdown: # type: ignore
            logger.info("Assurer l'arrêt final du ProcessPoolExecutor...")
            if sys.version_info >= (3, 9):
                 executor_instance.shutdown(wait=True, cancel_futures=True) # type: ignore
            else:
                executor_instance.shutdown(wait=True)
            logger.info("ProcessPoolExecutor arrêté (depuis finally).")
        executor_instance = None # Effacer la référence globale

    # Résumé final de l'orchestration
    logger.info("--- Résumé de l'Orchestration WFO ---")
    successful_tasks = [res for res in task_results if res.get("status") == "SUCCESS"]
    failed_tasks = [res for res in task_results if res.get("status") != "SUCCESS"]

    logger.info(f"Nombre total de tâches WFO traitées : {len(task_results)}")
    logger.info(f"  Tâches réussies : {len(successful_tasks)}")
    for res_succ in successful_tasks:
        logger.info(f"    - Stratégie: {res_succ['strategy']}, Paire: {res_succ['pair']}, Contexte: {res_succ['context']}, WFO Task Run ID: {res_succ['wfo_task_run_id']}")
    
    if failed_tasks:
        logger.error(f"  Tâches échouées ou avec erreurs : {len(failed_tasks)}")
        for res_fail in failed_tasks:
            logger.error(f"    - Stratégie: {res_fail['strategy']}, Paire: {res_fail['pair']}, Contexte: {res_fail['context']}, Erreur: {res_fail.get('error_message', 'Inconnue')}")
    
    orchestrator_summary_path = orchestrator_run_output_dir / "orchestrator_summary.json"
    try:
        with open(orchestrator_summary_path, 'w', encoding='utf-8') as f_summary:
            json.dump({
                "orchestrator_run_id": orchestrator_run_id,
                "cli_args": vars(args),
                "num_tasks_total": len(wfo_tasks_args),
                "num_tasks_processed_results": len(task_results),
                "num_successful_tasks": len(successful_tasks),
                "num_failed_tasks": len(failed_tasks),
                "task_results_summary": task_results # Contient les détails de chaque tâche
            }, f_summary, cls=EnhancedJSONEncoder, indent=4)
        logger.info(f"Résumé de l'orchestration sauvegardé dans : {orchestrator_summary_path}")
    except Exception as e_save_summary:
        logger.error(f"Échec de la sauvegarde du résumé de l'orchestration : {e_save_summary}", exc_info=True)

    run_end_time = time.time()
    total_duration_seconds = run_end_time - run_start_time
    logger.info(f"Temps d'Exécution Total de l'Orchestrateur : {total_duration_seconds:.2f} secondes ({timedelta(seconds=total_duration_seconds)})")
    logger.info(f"--- Fin de l'Orchestrateur d'Optimisation Walk-Forward (Run ID: {orchestrator_run_id}) ---")
    logging.shutdown()

if __name__ == "__main__":
    # S'assurer que le script peut être exécuté même si `multiprocessing` a des
    # comportements spécifiques sur certaines plateformes (ex: Windows) pour la création de processus.
    # `if __name__ == "__main__":` est crucial pour `multiprocessing` sur Windows.
    main()

