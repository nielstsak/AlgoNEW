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
from typing import Any, Dict, Optional, cast # Ajout de Optional

# Imports de l'application
# Assurer que le PYTHONPATH est correctement configuré si ce module est exécuté
# dans un processus complètement séparé sans héritage de sys.path.
# Normalement, multiprocessing sur Unix hérite de sys.path.
try:
    from src.config.loader import load_all_configs, AppConfig
    from src.config.definitions import LoggingConfig # Pour le typage et la création d'une config de log ad-hoc
    from src.utils.logging_setup import setup_logging
    from src.utils.run_utils import generate_run_id, create_wfo_task_dirs, _sanitize_path_component
    # WFOManager sera importé conditionnellement ou un placeholder sera utilisé
    # from src.backtesting.wfo.manager import WFOManager # Sera défini plus tard
except ImportError as e:
    # Ce log pourrait ne pas être capturé si le logging principal n'est pas encore configuré
    # dans le processus fils. Imprimer est une solution de repli.
    print(f"ERREUR CRITIQUE (task_executor): Impossible d'importer les modules nécessaires : {e}. Vérifiez PYTHONPATH.", file=sys.stderr)
    # Rendre la fonction principale inutilisable si les imports échouent
    def run_wfo_task_wrapper(*args, **kwargs) -> Dict[str, Any]:
        return {
            "status": "FAILURE",
            "error_message": f"Importations critiques échouées dans le processus fils: {e}",
            "strategy_name": kwargs.get("strategy_name", "inconnu"),
            "pair_symbol": kwargs.get("pair_symbol", "inconnu"),
            "cli_context_label": kwargs.get("cli_context_label", "inconnu"),
            "wfo_task_run_id": None,
            "summary_file_path": None
        }
    # Pas de logger configuré ici, donc on ne peut pas utiliser logger.critical

# Placeholder pour WFOManager si le fichier n'existe pas encore
# Cela permet au module d'être importable même si WFOManager n'est pas encore créé.
try:
    from src.backtesting.wfo.manager import WFOManager
except ImportError:
    logger_placeholder = logging.getLogger(__name__ + ".placeholder")
    logger_placeholder.warning("src.backtesting.wfo.manager.WFOManager non trouvé. Utilisation d'un placeholder.")
    class WFOManager: # type: ignore
        """Placeholder pour WFOManager."""
        def __init__(self,
                     app_config: AppConfig,
                     strategy_name: str,
                     pair_symbol: str,
                     cli_context_label: str,
                     orchestrator_run_id: str,
                     wfo_task_log_run_dir: Path,
                     wfo_task_results_root_dir: Path,
                     wfo_task_run_id: str):
            self.app_config = app_config
            self.strategy_name = strategy_name
            self.pair_symbol = pair_symbol
            self.cli_context_label = cli_context_label
            self.orchestrator_run_id = orchestrator_run_id
            self.wfo_task_log_run_dir = wfo_task_log_run_dir
            self.wfo_task_results_root_dir = wfo_task_results_root_dir
            self.wfo_task_run_id = wfo_task_run_id
            self.log_prefix = f"[{self.strategy_name}/{self.pair_symbol}/{self.cli_context_label}][Task:{self.wfo_task_run_id}][WFOManagerPlaceholder]"
            logger_placeholder.info(f"{self.log_prefix} Initialisé.")

        def run_single_wfo_task(self) -> Dict[str, Any]:
            logger_placeholder.warning(f"{self.log_prefix} run_single_wfo_task appelé sur un placeholder.")
            # Simuler la création d'un fichier de résumé
            summary_placeholder_path = self.wfo_task_log_run_dir / "wfo_strategy_pair_summary_placeholder.json"
            try:
                with open(summary_placeholder_path, "w") as f:
                    f.write('{"status": "placeholder_success"}')
                logger_placeholder.info(f"{self.log_prefix} Fichier de résumé placeholder créé : {summary_placeholder_path}")
            except Exception as e_write:
                logger_placeholder.error(f"{self.log_prefix} Erreur création fichier résumé placeholder : {e_write}")


            return {
                "status": "SUCCESS_PLACEHOLDER",
                "message": "Tâche WFO exécutée avec WFOManager placeholder.",
                "wfo_strategy_pair_summary_file": str(summary_placeholder_path),
                "live_config_file": None, # Placeholder
                "performance_report_file": None # Placeholder
            }

# Logger spécifique pour ce module (sera configuré dans run_wfo_task_wrapper)
task_executor_logger = logging.getLogger(__name__)

def run_wfo_task_wrapper(
    project_root_str: str,
    orchestrator_run_id: str,
    strategy_name: str,
    pair_symbol: str,
    cli_context_label: str
) -> Dict[str, Any]:
    """
    Wrapper exécuté dans un processus fils pour une tâche WFO spécifique.

    Cette fonction recharge la configuration, met en place le logging spécifique
    à cette tâche, instancie et exécute `WFOManager` pour la combinaison
    stratégie/paire/contexte donnée.

    Args:
        project_root_str (str): Chemin vers la racine du projet.
        orchestrator_run_id (str): ID du run global de l'orchestrateur parent.
        strategy_name (str): Nom de la stratégie à optimiser.
        pair_symbol (str): Symbole de la paire de trading.
        cli_context_label (str): Label de contexte fourni par la CLI à l'orchestrateur.

    Returns:
        Dict[str, Any]: Un dictionnaire indiquant le statut de la tâche,
                        les chemins vers les fichiers de résultats principaux,
                        et toute erreur survenue.
    """
    # Générer un ID unique pour cette tâche WFO spécifique
    # Le préfixe aide à identifier le type de run dans les logs/répertoires.
    s_strat_task = _sanitize_path_component(strategy_name)
    s_pair_task = _sanitize_path_component(pair_symbol)
    s_context_task = _sanitize_path_component(cli_context_label)
    wfo_task_run_id_prefix = f"wfo_task_{s_strat_task}_{s_pair_task}_{s_context_task}"
    wfo_task_run_id = generate_run_id(prefix=wfo_task_run_id_prefix)

    # Préfixe de log pour cette tâche spécifique
    log_prefix = f"[{s_strat_task}/{s_pair_task}/{s_context_task}][Task:{wfo_task_run_id}]"

    # --- Configuration initiale du logging pour ce processus fils ---
    # Avant de charger AppConfig, configurer un logging basique pour ce processus.
    # Cela est important car le processus fils n'hérite pas de la config de logging du parent.
    # On va créer un handler temporaire, puis le reconfigurer avec setup_logging.
    temp_handler = logging.StreamHandler(sys.stdout)
    temp_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(temp_handler) # Ajouter au root logger pour tout capturer
    logging.getLogger().setLevel(logging.INFO) # Niveau de base

    task_executor_logger.info(f"{log_prefix} Démarrage du wrapper de tâche WFO dans un processus fils.")
    task_executor_logger.info(f"{log_prefix} Orchestrateur Run ID: {orchestrator_run_id}")

    app_config: Optional[AppConfig] = None
    try:
        # 1. Recharger AppConfig dans le processus fils
        task_executor_logger.info(f"{log_prefix} Rechargement de AppConfig depuis la racine du projet : {project_root_str}")
        app_config = load_all_configs(project_root=project_root_str)
        task_executor_logger.info(f"{log_prefix} AppConfig rechargée avec succès dans le processus fils.")

        # 2. Créer les répertoires spécifiques à cette tâche WFO et configurer le logging de la tâche
        # Les chemins de base sont dérivés de AppConfig et de l'orchestrator_run_id
        base_log_dir_for_orchestrator_run = Path(app_config.global_config.paths.logs_backtest_optimization) / orchestrator_run_id
        base_results_dir_for_orchestrator_run = Path(app_config.global_config.paths.results) / orchestrator_run_id

        num_folds_for_task = app_config.global_config.wfo_settings.n_splits

        task_specific_dirs = create_wfo_task_dirs(
            base_log_dir_orchestrator_run=base_log_dir_for_orchestrator_run,
            base_results_dir_orchestrator_run=base_results_dir_for_orchestrator_run,
            strategy_name=strategy_name,
            pair_symbol=pair_symbol,
            cli_context_label=cli_context_label,
            wfo_task_run_id=wfo_task_run_id,
            num_folds=num_folds_for_task # Obtenir de AppConfig
        )
        
        wfo_task_log_run_dir = cast(Path, task_specific_dirs["wfo_task_log_run_dir"])
        wfo_task_results_root_dir = cast(Path, task_specific_dirs["wfo_task_results_root_dir"])
        
        # Configurer le logging pour qu'il écrive dans le fichier de log de cette tâche
        # Utiliser une LoggingConfig basique ou celle de AppConfig, mais rediriger la sortie.
        task_logging_config = LoggingConfig( # Créer une config ad-hoc pour ce logger de tâche
            level="DEBUG", # Loguer plus de détails pour les tâches individuelles
            format=app_config.global_config.logging.format, # Utiliser le format global
            log_to_file=True,
            log_filename_global="task_wfo.log", # Nom du fichier de log spécifique à la tâche
            log_filename_live=None, # Non pertinent ici
            log_levels_by_module=app_config.global_config.logging.log_levels_by_module
        )
        
        # Retirer le handler temporaire avant de configurer le logging final pour ce processus
        logging.getLogger().removeHandler(temp_handler)
        temp_handler.close()
        
        setup_logging(
            log_config=task_logging_config,
            log_dir=wfo_task_log_run_dir, # Le log ira dans le répertoire de la tâche
            log_filename=task_logging_config.log_filename_global, # "task_wfo.log"
            root_level=logging.DEBUG # Niveau racine pour ce processus
        )
        task_executor_logger.info(f"{log_prefix} Logging spécifique à la tâche configuré. Fichier de log : {wfo_task_log_run_dir / task_logging_config.log_filename_global}")

        # 3. Instancier et exécuter WFOManager
        task_executor_logger.info(f"{log_prefix} Instanciation de WFOManager...")
        wfo_manager = WFOManager(
            app_config=app_config,
            strategy_name=strategy_name,
            pair_symbol=pair_symbol,
            cli_context_label=cli_context_label,
            orchestrator_run_id=orchestrator_run_id,
            wfo_task_log_run_dir=wfo_task_log_run_dir,
            wfo_task_results_root_dir=wfo_task_results_root_dir,
            wfo_task_run_id=wfo_task_run_id # Passer l'ID de cette tâche
        )
        
        task_executor_logger.info(f"{log_prefix} Exécution de WFOManager.run_single_wfo_task()...")
        # run_single_wfo_task est supposé retourner un dict avec les chemins des artefacts principaux
        task_artifacts = wfo_manager.run_single_wfo_task()

        task_executor_logger.info(f"{log_prefix} Exécution de WFOManager terminée.")
        
        return {
            "status": "SUCCESS",
            "strategy_name": strategy_name,
            "pair_symbol": pair_symbol,
            "cli_context_label": cli_context_label,
            "orchestrator_run_id": orchestrator_run_id,
            "wfo_task_run_id": wfo_task_run_id,
            "wfo_task_log_dir": str(wfo_task_log_run_dir.resolve()),
            "wfo_task_results_dir": str(wfo_task_results_root_dir.resolve()),
            "summary_file_path": task_artifacts.get("wfo_strategy_pair_summary_file"),
            "live_config_file_path": task_artifacts.get("live_config_file"),
            "performance_report_file_path": task_artifacts.get("performance_report_file"),
            "message": f"Tâche WFO pour {strategy_name} sur {pair_symbol} ({cli_context_label}) terminée avec succès."
        }

    except Exception as e:
        task_executor_logger.critical(f"{log_prefix} ERREUR CRITIQUE dans run_wfo_task_wrapper : {e}", exc_info=True)
        return {
            "status": "FAILURE",
            "strategy_name": strategy_name,
            "pair_symbol": pair_symbol,
            "cli_context_label": cli_context_label,
            "orchestrator_run_id": orchestrator_run_id,
            "wfo_task_run_id": wfo_task_run_id, # Peut être None si l'erreur survient avant sa génération
            "error_message": str(e),
            "summary_file_path": None,
            "live_config_file_path": None,
            "performance_report_file_path": None
        }
    finally:
        logging.shutdown() # S'assurer que les logs sont vidés pour ce processus fils

