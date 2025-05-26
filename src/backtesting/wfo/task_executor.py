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
import os

# Logger spécifique pour ce module. Sa configuration sera affinée.
task_executor_logger = logging.getLogger(__name__)

# --- Variables globales pour les modules chargés dynamiquement ---
# Ceci évite les NameError si _load_dependencies_after_sys_path_setup échoue
# ou n'est pas appelé avant que les noms soient utilisés.
AppConfig = None
LoggingConfig = None
setup_logging = None
generate_run_id = None
create_wfo_task_dirs = None
_sanitize_path_component = None
WFOManager_class = None # Renommé pour éviter conflit avec une instance
load_all_configs_func = None # Renommé pour éviter conflit

_dependencies_loaded_in_task = False

def _ensure_src_in_sys_path(project_root_str: str, child_logger: logging.Logger):
    """Assure que la racine du projet et src sont dans sys.path."""
    project_root_path = Path(project_root_str).resolve()
    src_path_abs = (project_root_path / "src").resolve()

    # Log sys.path avant modification
    child_logger.debug(f"sys.path AVANT modification dans le processus fils : {sys.path}")

    paths_to_add = [str(project_root_path), str(src_path_abs)]
    
    for p_str in paths_to_add:
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
            child_logger.info(f"Ajouté '{p_str}' à sys.path dans le processus fils.")
        else:
            child_logger.debug(f"Chemin '{p_str}' déjà présent dans sys.path.")
    
    child_logger.debug(f"sys.path APRÈS modification dans le processus fils : {sys.path}")


def _load_dependencies_after_sys_path_setup(child_logger: logging.Logger):
    """Charge les dépendances après que sys.path ait été configuré."""
    global _dependencies_loaded_in_task, AppConfig, LoggingConfig, setup_logging, \
           generate_run_id, create_wfo_task_dirs, _sanitize_path_component, \
           WFOManager_class, load_all_configs_func
    
    if _dependencies_loaded_in_task:
        child_logger.debug("Dépendances déjà chargées, saut du re-import.")
        return

    child_logger.info("Tentative de chargement des dépendances du projet après configuration de sys.path...")
    try:
        from src.config.loader import load_all_configs, AppConfig as AppConfig_imported
        from src.config.definitions import LoggingConfig as LoggingConfig_imported
        from src.utils.logging_setup import setup_logging as setup_logging_imported
        from src.utils.run_utils import generate_run_id as generate_run_id_imported, \
                                        create_wfo_task_dirs as create_wfo_task_dirs_imported, \
                                        _sanitize_path_component as _sanitize_path_component_imported
        from src.backtesting.wfo.wfo_manager import WFOManager as WFOManager_imported
        
        # Assigner aux variables globales
        AppConfig = AppConfig_imported
        LoggingConfig = LoggingConfig_imported
        setup_logging = setup_logging_imported
        generate_run_id = generate_run_id_imported
        create_wfo_task_dirs = create_wfo_task_dirs_imported
        _sanitize_path_component = _sanitize_path_component_imported
        WFOManager_class = WFOManager_imported
        load_all_configs_func = load_all_configs

        _dependencies_loaded_in_task = True
        child_logger.info("Dépendances de task_executor chargées avec succès.")
    except ImportError as e:
        child_logger.critical(f"ERREUR CRITIQUE (task_executor - _load_dependencies): Impossible d'importer les modules après setup de sys.path : {e}. "
                              f"sys.path actuel: {sys.path}", exc_info=True)
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
    # Utiliser le logger spécifique à ce module pour les messages de setup.
    temp_log_handler = logging.StreamHandler(sys.stdout)
    # Format plus détaillé pour le setup
    temp_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
    
    # Configurer le logger de ce module (`task_executor_logger`)
    task_executor_logger.addHandler(temp_log_handler)
    task_executor_logger.setLevel(logging.DEBUG) # Forcer DEBUG pour voir tous les messages de setup
    task_executor_logger.propagate = False # Éviter duplication si le root logger est aussi configuré

    # Nettoyer les handlers du root logger s'ils existent (pour éviter des logs dupliqués par le root)
    # root_logger_process = logging.getLogger()
    # for handler in list(root_logger_process.handlers):
    #     root_logger_process.removeHandler(handler)
    #     handler.close()

    s_strat_task_ph = strategy_name.replace(" ", "_") # Placeholder simple avant _sanitize_path_component
    s_pair_task_ph = pair_symbol.replace("/", "")
    s_context_task_ph = cli_context_label.replace(" ", "_")
    temp_log_prefix = f"[{s_strat_task_ph}/{s_pair_task_ph}/{s_context_task_ph}][Task:PRE_ID_GEN]"

    task_executor_logger.info(f"{temp_log_prefix} Démarrage du wrapper de tâche WFO dans un processus fils (logging temporaire actif).")

    try:
        _ensure_src_in_sys_path(project_root_str, task_executor_logger)
        _load_dependencies_after_sys_path_setup(task_executor_logger)
    except Exception as e_setup:
        task_executor_logger.critical(f"{temp_log_prefix} Erreur critique lors du setup initial (sys.path ou imports) du processus fils : {e_setup}", exc_info=True)
        return {
            "status": "FAILURE_CHILD_PROCESS_SETUP", "error_message": str(e_setup),
            "strategy_name": strategy_name, "pair_symbol": pair_symbol, "cli_context_label": cli_context_label,
            "wfo_task_run_id": "N/A_SETUP_ERROR", "summary_file_path": None,
            "live_config_file_path": None, "performance_report_file_path": None
        }

    # Maintenant que les utilitaires sont chargés, on peut générer l'ID et le log_prefix finaux
    s_strat_task = _sanitize_path_component(strategy_name)
    s_pair_task = _sanitize_path_component(pair_symbol)
    s_context_task = _sanitize_path_component(cli_context_label)
    wfo_task_run_id_prefix = f"wfo_task_{s_strat_task}_{s_pair_task}_{s_context_task}"
    wfo_task_run_id = generate_run_id(prefix=wfo_task_run_id_prefix)

    log_prefix = f"[{s_strat_task}/{s_pair_task}/{s_context_task}][Task:{wfo_task_run_id}]"
    task_executor_logger.info(f"{log_prefix} Setup initial du processus fils terminé. Orchestrator Run ID: {orchestrator_run_id}")

    app_config_loaded: Optional[AppConfig] = None # Renommé pour éviter conflit avec variable globale
    wfo_task_log_run_dir_path: Optional[Path] = None

    try:
        task_executor_logger.info(f"{log_prefix} Rechargement de AppConfig depuis la racine du projet : {project_root_str}")
        app_config_loaded = load_all_configs_func(project_root=project_root_str) # Utiliser la fonction chargée
        task_executor_logger.info(f"{log_prefix} AppConfig rechargée avec succès dans le processus fils.")

        base_log_dir_orchestrator_run = Path(app_config_loaded.global_config.paths.logs_backtest_optimization) / orchestrator_run_id
        base_results_dir_orchestrator_run = Path(app_config_loaded.global_config.paths.results) / orchestrator_run_id
        num_folds_for_task = app_config_loaded.global_config.wfo_settings.n_splits

        task_specific_dirs_dict = create_wfo_task_dirs(
            base_log_dir_orchestrator_run=base_log_dir_orchestrator_run,
            base_results_dir_orchestrator_run=base_results_dir_orchestrator_run,
            strategy_name=strategy_name, pair_symbol=pair_symbol, cli_context_label=cli_context_label,
            wfo_task_run_id=wfo_task_run_id, num_folds=num_folds_for_task
        )
        
        wfo_task_log_run_dir_path = cast(Path, task_specific_dirs_dict["wfo_task_log_run_dir"])
        wfo_task_results_root_dir_path = cast(Path, task_specific_dirs_dict["wfo_task_results_root_dir"])
        
        task_logging_cfg_obj = LoggingConfig(
            level="DEBUG", # Forcer DEBUG pour les logs de tâche
            format=app_config_loaded.global_config.logging.format,
            log_to_file=True, log_filename_global="task_wfo_run.log", log_filename_live=None,
            log_levels_by_module=(app_config_loaded.global_config.logging.log_levels_by_module.copy() 
                                  if app_config_loaded.global_config.logging.log_levels_by_module else {})
        )
        
        # Retirer le handler temporaire et configurer le logging final pour ce processus
        # Le logger `task_executor_logger` est déjà configuré avec le temp_handler.
        # On va reconfigurer le root logger du processus pour le fichier de la tâche.
        root_logger_process_final = logging.getLogger()
        for handler in list(root_logger_process_final.handlers): # Nettoyer tous les handlers du root
            root_logger_process_final.removeHandler(handler)
            handler.close()
        
        setup_logging( # Ceci configure le root logger du processus courant
            log_config=task_logging_cfg_obj,
            log_dir=wfo_task_log_run_dir_path,
            log_filename=task_logging_cfg_obj.log_filename_global,
            root_level=getattr(logging, task_logging_cfg_obj.level.upper(), logging.DEBUG)
        )
        # Les messages suivants iront dans le fichier de log de la tâche.
        task_executor_logger.info(f"{log_prefix} Logging spécifique à la tâche configuré. Fichier de log : {wfo_task_log_run_dir_path / task_logging_cfg_obj.log_filename_global}")

        task_executor_logger.info(f"{log_prefix} Instanciation de WFOManager...")
        if WFOManager_class is None:
            raise ImportError("La classe WFOManager n'a pas été chargée correctement via _load_dependencies_after_sys_path_setup.")

        wfo_manager_instance = WFOManager_class(
            app_config=app_config_loaded, strategy_name=strategy_name, pair_symbol=pair_symbol,
            cli_context_label=cli_context_label, orchestrator_run_id=orchestrator_run_id,
            wfo_task_log_run_dir=wfo_task_log_run_dir_path,
            wfo_task_results_root_dir=wfo_task_results_root_dir_path,
            wfo_task_run_id=wfo_task_run_id
        )
        
        task_executor_logger.info(f"{log_prefix} Exécution de WFOManager.run_single_wfo_task()...")
        task_artifacts_dict = wfo_manager_instance.run_single_wfo_task()
        task_executor_logger.info(f"{log_prefix} Exécution de WFOManager terminée. Statut retourné: {task_artifacts_dict.get('status')}")
        
        return {
            "status": task_artifacts_dict.get("status", "UNKNOWN_WFO_MANAGER_STATUS"),
            "strategy_name": strategy_name, "pair_symbol": pair_symbol, "cli_context_label": cli_context_label,
            "orchestrator_run_id": orchestrator_run_id, "wfo_task_run_id": wfo_task_run_id,
            "wfo_task_log_dir": str(wfo_task_log_run_dir_path.resolve()),
            "wfo_task_results_dir": str(wfo_task_results_root_dir_path.resolve()),
            "wfo_strategy_pair_summary_file": task_artifacts_dict.get("wfo_strategy_pair_summary_file"),
            "expected_live_config_file_location": task_artifacts_dict.get("expected_live_config_file_location"),
            "expected_performance_report_file_location": task_artifacts_dict.get("expected_performance_report_file_location"),
            "message": f"Tâche WFO pour {strategy_name}/{pair_symbol} ({cli_context_label}) terminée avec statut: {task_artifacts_dict.get('status')}."
        }

    except Exception as e_wrapper:
        task_executor_logger.critical(f"{log_prefix if 'log_prefix' in locals() else temp_log_prefix} ERREUR CRITIQUE dans run_wfo_task_wrapper : {e_wrapper}", exc_info=True)
        if wfo_task_log_run_dir_path: # Si le répertoire a pu être créé
            error_file = wfo_task_log_run_dir_path / "TASK_EXECUTION_ERROR.txt"
            try:
                with open(error_file, "w", encoding="utf-8") as f_err:
                    f_err.write(f"Erreur critique dans run_wfo_task_wrapper pour la tâche {wfo_task_run_id if 'wfo_task_run_id' in locals() else 'NON_GENERE'}:\n")
                    import traceback
                    f_err.write(traceback.format_exc())
                task_executor_logger.info(f"{log_prefix if 'log_prefix' in locals() else temp_log_prefix} Détails de l'erreur sauvegardés dans : {error_file}")
            except Exception as e_save_err_file:
                task_executor_logger.error(f"{log_prefix if 'log_prefix' in locals() else temp_log_prefix} Impossible de sauvegarder le fichier d'erreur : {e_save_err_file}")

        return {
            "status": "FAILURE_IN_WRAPPER",
            "strategy_name": strategy_name, "pair_symbol": pair_symbol, "cli_context_label": cli_context_label,
            "orchestrator_run_id": orchestrator_run_id, 
            "wfo_task_run_id": wfo_task_run_id if 'wfo_task_run_id' in locals() else "N/A_WRAPPER_ERROR",
            "wfo_task_log_dir": str(wfo_task_log_run_dir_path.resolve()) if wfo_task_log_run_dir_path else None,
            "wfo_task_results_dir": None, "error_message": str(e_wrapper),
            "wfo_strategy_pair_summary_file": None, "live_config_file_path": None,
            "performance_report_file_path": None
        }
    finally:
        # S'assurer que les logs sont vidés pour ce processus fils avant qu'il ne se termine
        logging.shutdown()
