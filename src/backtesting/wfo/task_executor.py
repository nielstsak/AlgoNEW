# src/backtesting/wfo/task_executor.py
"""
Ce module contient la fonction wrapper pour exécuter une tâche d'Optimisation
Walk-Forward (WFO) dans un processus fils. Il gère le rechargement de la
configuration, la mise en place du logging spécifique à la tâche, et l'invocation
du gestionnaire WFO pour une combinaison stratégie/paire/contexte donnée.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, cast
import os # Ajouté pour os.path.abspath

# Imports de l'application
# Ces imports seront tentés après l'ajustement de sys.path
_app_config_loaded_in_task = False
_WFOManager_class = None

# Logger spécifique pour ce module (sera configuré dans run_wfo_task_wrapper)
task_executor_logger = logging.getLogger(__name__)

def _ensure_src_in_sys_path(project_root_str: str):
    """Assure que la racine du projet et src sont dans sys.path."""
    project_root_path = Path(project_root_str).resolve()
    src_path_abs = (project_root_path / "src").resolve()

    # Vérifier et ajouter project_root_path si nécessaire (pour les imports relatifs depuis src)
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))
        task_executor_logger.debug(f"Ajouté project_root '{project_root_path}' à sys.path dans le processus fils.")

    # Vérifier et ajouter src_path_abs si nécessaire
    if str(src_path_abs) not in sys.path:
        sys.path.insert(0, str(src_path_abs)) # Insérer au début pour priorité
        task_executor_logger.debug(f"Ajouté src_path '{src_path_abs}' à sys.path dans le processus fils.")
    
    # Logguer sys.path pour débogage
    # task_executor_logger.debug(f"sys.path actuel dans le processus fils : {sys.path}")


def _load_dependencies_after_sys_path_setup():
    """Charge les dépendances après que sys.path ait été configuré."""
    global _app_config_loaded_in_task, _WFOManager_class, AppConfig, LoggingConfig, \
           setup_logging, generate_run_id, create_wfo_task_dirs, _sanitize_path_component, \
           WFOManager, load_all_configs
    
    if _app_config_loaded_in_task: # Éviter re-imports inutiles
        return

    try:
        from src.config.loader import load_all_configs, AppConfig
        from src.config.definitions import LoggingConfig
        from src.utils.logging_setup import setup_logging
        from src.utils.run_utils import generate_run_id, create_wfo_task_dirs, _sanitize_path_component
        from src.backtesting.wfo.wfo_manager import WFOManager
        _WFOManager_class = WFOManager
        _app_config_loaded_in_task = True
        task_executor_logger.debug("Dépendances de task_executor chargées avec succès après configuration de sys.path.")
    except ImportError as e:
        task_executor_logger.critical(f"ERREUR CRITIQUE (task_executor - _load_dependencies): Impossible d'importer les modules après setup de sys.path : {e}.", exc_info=True)
        # Rendre la fonction principale inutilisable si les imports échouent
        def run_wfo_task_wrapper(*args, **kwargs) -> Dict[str, Any]: # type: ignore
            return {
                "status": "FAILURE_IMPORT_ERROR_IN_CHILD",
                "error_message": f"Importations critiques échouées dans le processus fils après setup sys.path: {e}",
                "strategy_name": kwargs.get("strategy_name", "inconnu"),
                "pair_symbol": kwargs.get("pair_symbol", "inconnu"),
                "cli_context_label": kwargs.get("cli_context_label", "inconnu"),
                "wfo_task_run_id": None, "summary_file_path": None,
                "live_config_file_path": None, "performance_report_file_path": None
            }
        raise # Renvoyer pour que l'appelant sache que le setup a échoué

def run_wfo_task_wrapper(
    project_root_str: str,
    orchestrator_run_id: str,
    strategy_name: str,
    pair_symbol: str,
    cli_context_label: str
) -> Dict[str, Any]:
    """
    Wrapper exécuté dans un processus fils pour une tâche WFO spécifique.
    """
    # Configurer un logging basique immédiatement pour ce processus fils.
    # Ce handler sera retiré et remplacé par la config de tâche plus tard.
    temp_log_handler = logging.StreamHandler(sys.stdout)
    temp_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
    root_logger_process = logging.getLogger()
    for handler in list(root_logger_process.handlers): # Nettoyer les handlers potentiels dupliqués
        root_logger_process.removeHandler(handler)
        handler.close()
    root_logger_process.addHandler(temp_log_handler)
    root_logger_process.setLevel(logging.DEBUG) # Mettre à DEBUG pour voir les messages de sys.path

    # Assurer que src est dans sys.path AVANT tout autre import de src.*
    try:
        _ensure_src_in_sys_path(project_root_str)
        _load_dependencies_after_sys_path_setup() # Charger les dépendances maintenant
    except Exception as e_setup:
        task_executor_logger.critical(f"Erreur critique lors du setup initial (sys.path ou imports) du processus fils : {e_setup}", exc_info=True)
        return {
            "status": "FAILURE_CHILD_PROCESS_SETUP", "error_message": str(e_setup),
            "strategy_name": strategy_name, "pair_symbol": pair_symbol, "cli_context_label": cli_context_label,
            "wfo_task_run_id": "N/A_SETUP_ERROR", "summary_file_path": None,
            "live_config_file_path": None, "performance_report_file_path": None
        }

    # Générer un ID unique pour cette tâche WFO spécifique
    s_strat_task = _sanitize_path_component(strategy_name)
    s_pair_task = _sanitize_path_component(pair_symbol)
    s_context_task = _sanitize_path_component(cli_context_label)
    wfo_task_run_id_prefix = f"wfo_task_{s_strat_task}_{s_pair_task}_{s_context_task}"
    wfo_task_run_id = generate_run_id(prefix=wfo_task_run_id_prefix)

    log_prefix = f"[{s_strat_task}/{s_pair_task}/{s_context_task}][Task:{wfo_task_run_id}]"
    task_executor_logger.info(f"{log_prefix} Démarrage du wrapper de tâche WFO dans un processus fils (post-setup).")
    task_executor_logger.info(f"{log_prefix} Orchestrateur Run ID: {orchestrator_run_id}")

    app_config: Optional[AppConfig] = None
    wfo_task_log_run_dir_path: Optional[Path] = None

    try:
        task_executor_logger.info(f"{log_prefix} Rechargement de AppConfig depuis la racine du projet : {project_root_str}")
        app_config = load_all_configs(project_root=project_root_str)
        task_executor_logger.info(f"{log_prefix} AppConfig rechargée avec succès dans le processus fils.")

        base_log_dir_orchestrator_run = Path(app_config.global_config.paths.logs_backtest_optimization) / orchestrator_run_id
        base_results_dir_orchestrator_run = Path(app_config.global_config.paths.results) / orchestrator_run_id
        num_folds_for_task = app_config.global_config.wfo_settings.n_splits

        task_specific_dirs_dict = create_wfo_task_dirs(
            base_log_dir_orchestrator_run=base_log_dir_orchestrator_run,
            base_results_dir_orchestrator_run=base_results_dir_orchestrator_run,
            strategy_name=strategy_name,
            pair_symbol=pair_symbol,
            cli_context_label=cli_context_label,
            wfo_task_run_id=wfo_task_run_id,
            num_folds=num_folds_for_task
        )
        
        wfo_task_log_run_dir_path = cast(Path, task_specific_dirs_dict["wfo_task_log_run_dir"])
        wfo_task_results_root_dir_path = cast(Path, task_specific_dirs_dict["wfo_task_results_root_dir"])
        
        task_logging_cfg_obj = LoggingConfig(
            level=app_config.global_config.logging.level if app_config.global_config.logging.level in ["DEBUG","INFO"] else "DEBUG",
            format=app_config.global_config.logging.format,
            log_to_file=True,
            log_filename_global="task_wfo_run.log",
            log_filename_live=None,
            log_levels_by_module=app_config.global_config.logging.log_levels_by_module.copy() if app_config.global_config.logging.log_levels_by_module else {}
        )
        
        root_logger_process.removeHandler(temp_log_handler)
        temp_log_handler.close()
        
        setup_logging(
            log_config=task_logging_cfg_obj,
            log_dir=wfo_task_log_run_dir_path,
            log_filename=task_logging_cfg_obj.log_filename_global,
            root_level=getattr(logging, task_logging_cfg_obj.level.upper(), logging.DEBUG)
        )
        task_executor_logger.info(f"{log_prefix} Logging spécifique à la tâche configuré. Fichier de log : {wfo_task_log_run_dir_path / task_logging_cfg_obj.log_filename_global}")

        task_executor_logger.info(f"{log_prefix} Instanciation de WFOManager...")
        if _WFOManager_class is None: # Vérification si l'import a bien eu lieu
            raise ImportError("La classe WFOManager n'a pas été chargée correctement.")

        wfo_manager_instance = _WFOManager_class(
            app_config=app_config,
            strategy_name=strategy_name,
            pair_symbol=pair_symbol,
            cli_context_label=cli_context_label,
            orchestrator_run_id=orchestrator_run_id,
            wfo_task_log_run_dir=wfo_task_log_run_dir_path,
            wfo_task_results_root_dir=wfo_task_results_root_dir_path,
            wfo_task_run_id=wfo_task_run_id
        )
        
        task_executor_logger.info(f"{log_prefix} Exécution de WFOManager.run_single_wfo_task()...")
        task_artifacts_dict = wfo_manager_instance.run_single_wfo_task()
        task_executor_logger.info(f"{log_prefix} Exécution de WFOManager terminée. Statut retourné: {task_artifacts_dict.get('status')}")
        
        return {
            "status": task_artifacts_dict.get("status", "UNKNOWN_WFO_MANAGER_STATUS"),
            "strategy_name": strategy_name,
            "pair_symbol": pair_symbol,
            "cli_context_label": cli_context_label,
            "orchestrator_run_id": orchestrator_run_id,
            "wfo_task_run_id": wfo_task_run_id,
            "wfo_task_log_dir": str(wfo_task_log_run_dir_path.resolve()),
            "wfo_task_results_dir": str(wfo_task_results_root_dir_path.resolve()),
            "wfo_strategy_pair_summary_file": task_artifacts_dict.get("wfo_strategy_pair_summary_file"),
            "expected_live_config_file_location": task_artifacts_dict.get("expected_live_config_file_location"),
            "expected_performance_report_file_location": task_artifacts_dict.get("expected_performance_report_file_location"),
            "message": f"Tâche WFO pour {strategy_name}/{pair_symbol} ({cli_context_label}) terminée avec statut: {task_artifacts_dict.get('status')}."
        }

    except Exception as e_wrapper:
        task_executor_logger.critical(f"{log_prefix} ERREUR CRITIQUE dans run_wfo_task_wrapper : {e_wrapper}", exc_info=True)
        if wfo_task_log_run_dir_path:
            error_file = wfo_task_log_run_dir_path / "TASK_EXECUTION_ERROR.txt"
            try:
                with open(error_file, "w", encoding="utf-8") as f_err:
                    f_err.write(f"Erreur critique dans run_wfo_task_wrapper pour la tâche {wfo_task_run_id}:\n")
                    import traceback
                    f_err.write(traceback.format_exc())
                task_executor_logger.info(f"{log_prefix} Détails de l'erreur sauvegardés dans : {error_file}")
            except Exception as e_save_err_file:
                task_executor_logger.error(f"{log_prefix} Impossible de sauvegarder le fichier d'erreur : {e_save_err_file}")

        return {
            "status": "FAILURE_IN_WRAPPER",
            "strategy_name": strategy_name, "pair_symbol": pair_symbol, "cli_context_label": cli_context_label,
            "orchestrator_run_id": orchestrator_run_id, "wfo_task_run_id": wfo_task_run_id,
            "wfo_task_log_dir": str(wfo_task_log_run_dir_path.resolve()) if wfo_task_log_run_dir_path else None,
            "wfo_task_results_dir": None, "error_message": str(e_wrapper),
            "wfo_strategy_pair_summary_file": None, "live_config_file_path": None,
            "performance_report_file_path": None
        }
    finally:
        logging.shutdown()
