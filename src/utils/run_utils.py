# src/utils/run_utils.py
"""
Fonctions utilitaires pour la gestion des "runs" de l'application,
notamment la génération d'identifiants de run et la création structurée
des arborescences de répertoires pour les logs et les artefacts.
"""
import logging
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Union

# Importation de ensure_dir_exists depuis file_utils
# S'assurer que src.utils.file_utils est accessible.
try:
    from src.utils.file_utils import ensure_dir_exists
except ImportError:
    # Fallback basique si l'import échoue (ne devrait pas arriver dans un projet structuré)
    logging.getLogger(__name__).critical(
        "CRITICAL: Impossible d'importer ensure_dir_exists depuis src.utils.file_utils. "
        "La création de répertoires échouera."
    )
    def ensure_dir_exists(dir_path: Union[str, Path]) -> bool: # type: ignore
        """Fallback dummy function."""
        logging.error(f"Fonction factice ensure_dir_exists appelée pour {dir_path}. L'importation a échoué.")
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

logger = logging.getLogger(__name__)

def _sanitize_path_component(name: str) -> str:
    """
    Nettoie une chaîne de caractères pour qu'elle soit sûre à utiliser
    comme composant de nom de fichier ou de répertoire.
    Remplace les caractères non alphanumériques (sauf '-', '_', '.') par '_'.

    Args:
        name (str): La chaîne à nettoyer.

    Returns:
        str: La chaîne nettoyée.
    """
    if not name:
        return "default_component"
    # Remplace tout ce qui n'est pas un caractère de mot, un tiret, un point ou un underscore par un underscore.
    sanitized_name = re.sub(r'[^\w\-._]', '_', name)
    # Supprime les underscores potentiels au début ou à la fin après le remplacement.
    sanitized_name = sanitized_name.strip('_')
    # Si après nettoyage la chaîne est vide (ex: "!!!"), retourner un nom par défaut.
    return sanitized_name if sanitized_name else "sanitized_default"

def generate_run_id(prefix: Optional[str] = "run") -> str:
    """
    Génère un identifiant de run unique basé sur le timestamp UTC actuel
    et un préfixe optionnel.

    Args:
        prefix (Optional[str]): Un préfixe pour l'identifiant de run.
                                Par défaut "run".

    Returns:
        str: Un identifiant de run unique (ex: "run_YYYYMMDD_HHMMSS").
    """
    now_utc = datetime.now(timezone.utc)
    timestamp_str = now_utc.strftime("%Y%m%d_%H%M%S") # Format compact et triable

    clean_prefix = _sanitize_path_component(prefix.strip()) if prefix and prefix.strip() else ""

    if clean_prefix:
        return f"{clean_prefix}_{timestamp_str}"
    return timestamp_str


def create_orchestrator_run_dir(base_output_dir: Path, orchestrator_run_id: str) -> Path:
    """
    Crée et retourne le répertoire de base pour un run d'orchestrateur WFO.
    Ce répertoire contiendra les logs globaux de l'orchestrateur et sa configuration.

    Args:
        base_output_dir (Path): Le répertoire de base pour les sorties de l'orchestrateur
                                (ex: `logs/backtest_optimization` ou `results`).
        orchestrator_run_id (str): L'ID unique du run de l'orchestrateur.

    Returns:
        Path: Le chemin vers le répertoire du run de l'orchestrateur créé.
              Lève une OSError si la création échoue.
    """
    orchestrator_run_path = base_output_dir / orchestrator_run_id
    if not ensure_dir_exists(orchestrator_run_path):
        raise OSError(f"Échec de la création du répertoire du run de l'orchestrateur : {orchestrator_run_path}")
    logger.debug(f"Répertoire du run de l'orchestrateur créé/assuré : {orchestrator_run_path}")
    return orchestrator_run_path


def create_wfo_task_dirs(
    base_log_dir_orchestrator_run: Path,
    base_results_dir_orchestrator_run: Path,
    strategy_name: str,
    pair_symbol: str,
    cli_context_label: str,
    wfo_task_run_id: str,
    num_folds: int
) -> Dict[str, Union[Path, List[Path]]]:
    """
    Crée l'arborescence complète des répertoires pour une tâche WFO spécifique
    (stratégie/paire/contexte_cli) au sein d'un run d'orchestrateur.
    Cela inclut les répertoires pour les logs de la tâche, les logs de chaque fold,
    et les résultats (config live, rapport, visualisations Optuna).

    Args:
        base_log_dir_orchestrator_run (Path): Répertoire de log du run de l'orchestrateur
            (ex: `logs/backtest_optimization/ORCH_ID`).
        base_results_dir_orchestrator_run (Path): Répertoire de résultats du run de l'orchestrateur
            (ex: `results/ORCH_ID`).
        strategy_name (str): Nom de la stratégie.
        pair_symbol (str): Symbole de la paire.
        cli_context_label (str): Label de contexte fourni par la CLI.
        wfo_task_run_id (str): ID unique pour cette tâche WFO spécifique.
        num_folds (int): Nombre de folds prévus pour cette tâche WFO.

    Returns:
        Dict[str, Union[Path, List[Path]]]: Un dictionnaire contenant les chemins clés créés.
            Lève une OSError si la création d'un répertoire échoue.
    """
    s_strat = _sanitize_path_component(strategy_name)
    s_pair = _sanitize_path_component(pair_symbol)
    s_context = _sanitize_path_component(cli_context_label)

    log_prefix = f"[CreateWFOTaskDirs][{s_strat}/{s_pair}/{s_context}][TaskID:{wfo_task_run_id}]"
    logger.info(f"{log_prefix} Création de l'arborescence des répertoires pour la tâche WFO...")

    paths_created: Dict[str, Union[Path, List[Path]]] = {}

    try:
        # 1. Répertoire racine des logs pour cette tâche WFO (STRAT/PAIRE/CONTEXTE_CLI)
        #    Ce répertoire contiendra le répertoire du wfo_task_run_id.
        wfo_task_base_log_dir = base_log_dir_orchestrator_run / s_strat / s_pair / s_context
        ensure_dir_exists(wfo_task_base_log_dir)
        paths_created["wfo_task_base_log_dir"] = wfo_task_base_log_dir
        logger.debug(f"{log_prefix} Répertoire de base des logs de la tâche WFO : {wfo_task_base_log_dir}")

        # 2. Répertoire spécifique au run de cette tâche WFO (pour task_wfo.log, summary, et folds)
        wfo_task_log_run_dir = wfo_task_base_log_dir / wfo_task_run_id
        ensure_dir_exists(wfo_task_log_run_dir)
        paths_created["wfo_task_log_run_dir"] = wfo_task_log_run_dir
        logger.debug(f"{log_prefix} Répertoire de run des logs de la tâche WFO : {wfo_task_log_run_dir}")

        # 3. Répertoires de log pour chaque fold
        fold_log_dirs_list: List[Path] = []
        if num_folds < 0:
            logger.warning(f"{log_prefix} num_folds ({num_folds}) est négatif. Aucun répertoire de fold de log ne sera créé.")
        for i in range(num_folds):
            fold_log_path = wfo_task_log_run_dir / f"fold_{i}"
            ensure_dir_exists(fold_log_path)
            fold_log_dirs_list.append(fold_log_path)
            logger.debug(f"{log_prefix} Répertoire de log du fold {i} : {fold_log_path}")
        paths_created["fold_log_dirs"] = fold_log_dirs_list

        # 4. Répertoire racine des résultats pour cette tâche WFO (STRAT/PAIRE/CONTEXTE_CLI)
        #    Pour live_config.json, performance_report.md, et le dossier optuna_visualizations.
        wfo_task_results_root_dir = base_results_dir_orchestrator_run / s_strat / s_pair / s_context
        ensure_dir_exists(wfo_task_results_root_dir)
        paths_created["wfo_task_results_root_dir"] = wfo_task_results_root_dir
        logger.debug(f"{log_prefix} Répertoire racine des résultats de la tâche WFO : {wfo_task_results_root_dir}")

        # 5. Répertoires pour les visualisations Optuna de chaque fold (dans les résultats)
        fold_viz_dirs_list: List[Path] = []
        if num_folds > 0 : # Créer le dossier parent "optuna_visualizations" seulement si des folds existent
            optuna_visualizations_base_dir = wfo_task_results_root_dir / "optuna_visualizations"
            ensure_dir_exists(optuna_visualizations_base_dir) # Assurer que le dossier parent existe
            paths_created["optuna_visualizations_base_dir"] = optuna_visualizations_base_dir

            for i in range(num_folds):
                fold_viz_path = optuna_visualizations_base_dir / f"fold_{i}"
                ensure_dir_exists(fold_viz_path)
                fold_viz_dirs_list.append(fold_viz_path)
                logger.debug(f"{log_prefix} Répertoire de visualisations Optuna du fold {i} : {fold_viz_path}")
        paths_created["fold_viz_dirs"] = fold_viz_dirs_list

        logger.info(f"{log_prefix} Arborescence des répertoires de la tâche WFO créée avec succès.")
        return paths_created

    except OSError as e:
        logger.error(f"{log_prefix} Erreur OSError lors de la création des répertoires de la tâche WFO : {e}", exc_info=True)
        raise
    except Exception as e_gen:
        logger.error(f"{log_prefix} Erreur inattendue lors de la création des répertoires de la tâche WFO : {e_gen}", exc_info=True)
        raise


def create_live_session_dirs(
    base_log_dir: Path,
    base_state_dir: Path,
    account_alias: str,
    strategy_deployment_id: str, # ID unique du déploiement de la stratégie
    pair_symbol: str,
    # context_label: str # Le contexte est souvent inclus dans strategy_deployment_id
    live_session_run_id: Optional[str] = None # Optionnel, pour grouper les logs d'une session de run_live.py
) -> Dict[str, Path]:
    """
    Crée les répertoires de base pour une session de trading live spécifique.
    La structure peut inclure un ID de session de run_live.py pour grouper les logs.

    Args:
        base_log_dir (Path): Chemin racine des logs pour le trading live (ex: `logs/live_trading`).
        base_state_dir (Path): Chemin racine pour les fichiers d'état (ex: `data/live_state`).
        account_alias (str): Alias du compte utilisé.
        strategy_deployment_id (str): ID unique du déploiement de la stratégie.
        pair_symbol (str): Symbole de la paire.
        live_session_run_id (Optional[str]): ID optionnel du run global de la session live.

    Returns:
        Dict[str, Path]: Un dictionnaire contenant les chemins clés créés.
                         Lève une OSError si la création échoue.
    """
    s_account = _sanitize_path_component(account_alias)
    s_strategy_deploy_id = _sanitize_path_component(strategy_deployment_id)
    s_pair = _sanitize_path_component(pair_symbol)

    log_prefix = f"[CreateLiveDirs][Acc:{s_account}][DeployID:{s_strategy_deploy_id}][Pair:{s_pair}]"
    logger.info(f"{log_prefix} Création des répertoires de session live...")

    paths_created: Dict[str, Path] = {}

    # Déterminer le répertoire de log de base pour cette session
    current_session_base_log_dir = base_log_dir
    if live_session_run_id:
        s_live_run_id = _sanitize_path_component(live_session_run_id)
        current_session_base_log_dir = base_log_dir / s_live_run_id
    
    # Structure: BASE_LOG_DIR / [LIVE_SESSION_RUN_ID] / ACCOUNT_ALIAS / STRATEGY_DEPLOYMENT_ID / PAIR_SYMBOL /
    # Les fichiers de log spécifiques (ex: session_YYYYMMDD.log) seront créés par le logger lui-même dans ce répertoire.
    specific_session_log_dir = current_session_base_log_dir / s_account / s_strategy_deploy_id / s_pair
    
    # Le fichier d'état est plus spécifique et géré par LiveTradingState,
    # mais son répertoire de base (par compte/déploiement/paire) peut être assuré ici.
    # Structure: BASE_STATE_DIR / ACCOUNT_ALIAS / STRATEGY_DEPLOYMENT_ID / PAIR_SYMBOL / state.json
    specific_session_state_dir = base_state_dir / s_account / s_strategy_deploy_id / s_pair

    try:
        if not ensure_dir_exists(specific_session_log_dir):
            raise OSError(f"Échec de la création du répertoire de log de session : {specific_session_log_dir}")
        paths_created["session_log_dir"] = specific_session_log_dir
        logger.debug(f"{log_prefix} Répertoire de log de session : {specific_session_log_dir}")

        if not ensure_dir_exists(specific_session_state_dir):
            raise OSError(f"Échec de la création du répertoire d'état de session : {specific_session_state_dir}")
        paths_created["session_state_dir"] = specific_session_state_dir # Répertoire où le state.json sera placé
        logger.debug(f"{log_prefix} Répertoire d'état de session assuré : {specific_session_state_dir}")
        
        logger.info(f"{log_prefix} Répertoires de session live créés/assurés avec succès.")
        return paths_created
    except OSError as e:
        logger.error(f"{log_prefix} Erreur OSError lors de la création des répertoires de session live : {e}", exc_info=True)
        raise
    except Exception as e_gen:
        logger.error(f"{log_prefix} Erreur inattendue lors de la création des répertoires de session live : {e_gen}", exc_info=True)
        raise

if __name__ == '__main__':
    # Configuration du logging pour les tests directs de ce module
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Test de generate_run_id ---
    test_run_id_1 = generate_run_id()
    logger.info(f"Test generate_run_id (défaut) : {test_run_id_1}")
    test_run_id_2 = generate_run_id(prefix="my_wfo_orchestrator")
    logger.info(f"Test generate_run_id (avec préfixe) : {test_run_id_2}")
    test_run_id_3 = generate_run_id(prefix="live_sess!!!") # Test avec nettoyage
    logger.info(f"Test generate_run_id (préfixe à nettoyer) : {test_run_id_3}")

    # --- Configuration des chemins de base pour les tests de création de répertoires ---
    # Utiliser un répertoire temporaire pour les tests
    temp_project_root_for_test = Path("./temp_run_utils_test_output").resolve()
    ensure_dir_exists(temp_project_root_for_test)
    logger.info(f"Répertoire racine temporaire pour les tests : {temp_project_root_for_test}")

    test_base_logs = temp_project_root_for_test / "logs_output" / "backtest_optimization"
    test_base_results = temp_project_root_for_test / "results_output"
    test_base_live_logs = temp_project_root_for_test / "logs_output" / "live_trading"
    test_base_live_state = temp_project_root_for_test / "data_output" / "live_state"

    # --- Test de create_orchestrator_run_dir ---
    try:
        orc_run_id = generate_run_id("test_orc")
        orc_run_dir = create_orchestrator_run_dir(test_base_logs, orc_run_id)
        logger.info(f"Test create_orchestrator_run_dir : {orc_run_dir} (Existe: {orc_run_dir.exists()})")
    except Exception as e:
        logger.error(f"Erreur test create_orchestrator_run_dir : {e}", exc_info=True)

    # --- Test de create_wfo_task_dirs ---
    try:
        # Simuler un orchestrator_run_id et un wfo_task_run_id
        orchestrator_id_for_task = generate_run_id("main_wfo_run")
        task_specific_run_id = generate_run_id("task_ema_btc") # ID généré par WalkForwardOptimizer

        # Les répertoires de base pour la tâche sont relatifs au run de l'orchestrateur
        task_base_log_dir = test_base_logs / orchestrator_id_for_task
        task_base_results_dir = test_base_results / orchestrator_id_for_task
        # create_orchestrator_run_dir aurait déjà créé ces bases, mais on s'assure pour le test
        ensure_dir_exists(task_base_log_dir)
        ensure_dir_exists(task_base_results_dir)


        created_wfo_paths = create_wfo_task_dirs(
            base_log_dir_orchestrator_run=task_base_log_dir,
            base_results_dir_orchestrator_run=task_base_results_dir,
            strategy_name="EMA_Crossover_Strategy",
            pair_symbol="BTC/USDT", # Test avec /
            cli_context_label="5min_default_context",
            wfo_task_run_id=task_specific_run_id,
            num_folds=3
        )
        logger.info(f"Test create_wfo_task_dirs - Chemins créés :")
        for key, path_val in created_wfo_paths.items():
            if isinstance(path_val, list):
                logger.info(f"  {key}:")
                for p_item in path_val:
                    logger.info(f"    - {p_item} (Existe: {Path(p_item).exists()})")
            else:
                logger.info(f"  {key}: {path_val} (Existe: {Path(path_val).exists()})")

        # Vérification spécifique d'un chemin de fold
        if created_wfo_paths.get("fold_log_dirs") and isinstance(created_wfo_paths["fold_log_dirs"], list) and created_wfo_paths["fold_log_dirs"]:
            first_fold_log_dir = Path(created_wfo_paths["fold_log_dirs"][0])
            if not first_fold_log_dir.exists():
                logger.error(f"Erreur : Le premier répertoire de log de fold n'a pas été créé : {first_fold_log_dir}")
        else:
             logger.warning("Aucun répertoire de log de fold retourné par create_wfo_task_dirs.")


    except Exception as e:
        logger.error(f"Erreur test create_wfo_task_dirs : {e}", exc_info=True)

    # --- Test de create_live_session_dirs ---
    try:
        live_run_id_main = generate_run_id("live_main_session")
        created_live_paths = create_live_session_dirs(
            base_log_dir=test_base_live_logs,
            base_state_dir=test_base_live_state,
            account_alias="my_binance_margin_live",
            strategy_deployment_id="EMA_Cross_BTCUSDT_5min_live_deploy_001",
            pair_symbol="BTCUSDT",
            live_session_run_id=live_run_id_main
        )
        logger.info(f"Test create_live_session_dirs - Chemins créés :")
        for key, path_val in created_live_paths.items():
            logger.info(f"  {key}: {path_val} (Existe: {Path(path_val).exists()})")

    except Exception as e:
        logger.error(f"Erreur test create_live_session_dirs : {e}", exc_info=True)

    # Optionnel : Nettoyage du répertoire de test (décommenter pour nettoyer après exécution)
    # import shutil
    # if temp_project_root_for_test.exists():
    #     logger.info(f"Nettoyage du répertoire de test : {temp_project_root_for_test}")
    #     shutil.rmtree(temp_project_root_for_test)
    # else:
    #     logger.info(f"Répertoire de test {temp_project_root_for_test} non trouvé pour le nettoyage.")
