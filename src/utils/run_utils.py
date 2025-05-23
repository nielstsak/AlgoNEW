# Fichier: src/utils/run_utils.py
"""
Fonctions utilitaires pour la gestion des "runs" de l'application,
notamment la création d'identifiants de run et de l'arborescence des répertoires.
"""
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# Tentative d'importation de file_utils.
# Si ce module est dans le même package 'utils', l'import relatif devrait fonctionner.
try:
    from .file_utils import ensure_dir_exists
except ImportError:
    # Fallback si l'importation relative échoue (par exemple, si utils n'est pas traité comme un package
    # ou si file_utils.py n'est pas au même niveau)
    try:
        from src.utils.file_utils import ensure_dir_exists
    except ImportError:
        logging.getLogger(__name__).critical(
            "CRITICAL: Impossible d'importer ensure_dir_exists depuis src.utils.file_utils. "
            "La création de répertoires échouera."
        )
        # Définir une fonction factice pour permettre au reste du module de se charger,
        # mais cela ne fonctionnera pas correctement.
        def ensure_dir_exists(dir_path: Path) -> bool: # type: ignore
            logging.error("Fonction factice ensure_dir_exists appelée. L'importation a échoué.")
            return False

logger = logging.getLogger(__name__)

def generate_run_id(prefix: Optional[str] = "run") -> str:
    """
    Génère un identifiant de run unique basé sur le timestamp UTC actuel.

    Args:
        prefix (Optional[str]): Un préfixe pour l'identifiant de run. Par défaut "run".

    Returns:
        str: Un identifiant de run unique (ex: "run_20230523_143055").
    """
    now_utc = datetime.now(timezone.utc)
    timestamp_str = now_utc.strftime("%Y%m%d_%H%M%S")
    
    if prefix and prefix.strip():
        return f"{prefix.strip()}_{timestamp_str}"
    return timestamp_str

def create_optimization_run_dirs(
    base_log_dir: Path,
    base_results_dir: Path,
    run_id: str,
    strategy_name: str,
    pair_symbol: str,
    context_label: str,
    num_folds: int
) -> Dict[str, Any]: # Retourne Any pour inclure List[Path] pour fold_log_dirs
    """
    Crée l'arborescence complète des répertoires pour un nouveau run d'optimisation WFO,
    pour une combinaison stratégie/paire/contexte donnée.

    Args:
        base_log_dir: Chemin racine des logs d'optimisation (ex: Path("logs/backtest_optimization")).
        base_results_dir: Chemin racine des résultats d'optimisation (ex: Path("results")).
        run_id: L'identifiant unique du run WFO global.
        strategy_name: Nom de la stratégie.
        pair_symbol: Symbole de la paire.
        context_label: Label de contexte.
        num_folds: Nombre de folds prévus pour ce WFO.

    Returns:
        Un dictionnaire contenant les chemins clés créés.
        Lève une OSError si la création d'un répertoire échoue.
    """
    log_prefix = f"[CreateOptRunDirs][{run_id}][{strategy_name}/{pair_symbol}/{context_label}]"
    logger.info(f"{log_prefix} Création de l'arborescence des répertoires...")

    paths_created: Dict[str, Any] = {}

    try:
        # 1. Répertoires au niveau du run_id
        run_log_dir = base_log_dir / run_id
        ensure_dir_exists(run_log_dir) # file_utils devrait lever une erreur si échec
        paths_created["run_log_dir"] = run_log_dir
        logger.debug(f"{log_prefix} Répertoire de log du run: {run_log_dir}")

        run_results_dir = base_results_dir / run_id
        ensure_dir_exists(run_results_dir)
        paths_created["run_results_dir"] = run_results_dir
        logger.debug(f"{log_prefix} Répertoire de résultats du run: {run_results_dir}")

        # 2. Répertoires spécifiques à la stratégie/paire/contexte pour les logs
        strategy_context_log_dir = run_log_dir / strategy_name / pair_symbol / context_label
        ensure_dir_exists(strategy_context_log_dir)
        paths_created["strategy_context_log_dir"] = strategy_context_log_dir
        logger.debug(f"{log_prefix} Répertoire de log stratégie/paire/contexte: {strategy_context_log_dir}")

        # 3. Répertoires spécifiques à la stratégie/paire/contexte pour les résultats
        strategy_context_results_dir = run_results_dir / strategy_name / pair_symbol / context_label
        ensure_dir_exists(strategy_context_results_dir)
        paths_created["strategy_context_results_dir"] = strategy_context_results_dir
        logger.debug(f"{log_prefix} Répertoire de résultats stratégie/paire/contexte: {strategy_context_results_dir}")

        # 4. Répertoires pour chaque fold (dans les logs)
        fold_log_dirs: List[Path] = []
        fold_optuna_viz_dirs: List[Path] = [] # Optionnel, pour les visualisations

        if num_folds < 0:
            logger.warning(f"{log_prefix} num_folds ({num_folds}) est négatif. Aucun répertoire de fold ne sera créé.")
        
        for i in range(num_folds):
            fold_dir_name = f"fold_{i}"
            
            # Répertoire de log pour le fold
            fold_log_path = strategy_context_log_dir / fold_dir_name
            ensure_dir_exists(fold_log_path)
            fold_log_dirs.append(fold_log_path)
            logger.debug(f"{log_prefix} Répertoire de log du fold {i}: {fold_log_path}")

            # Optionnel: Répertoire pour les visualisations Optuna de ce fold (dans les résultats)
            # La génération des rapports créera ce chemin si nécessaire, mais on peut le préparer.
            optuna_viz_path = strategy_context_results_dir / "optuna_visualizations" / fold_dir_name
            ensure_dir_exists(optuna_viz_path)
            fold_optuna_viz_dirs.append(optuna_viz_path)
            logger.debug(f"{log_prefix} Répertoire de visualisations Optuna du fold {i}: {optuna_viz_path}")


        paths_created["fold_log_dirs"] = fold_log_dirs
        paths_created["fold_optuna_viz_dirs"] = fold_optuna_viz_dirs # Ajouté

        logger.info(f"{log_prefix} Arborescence des répertoires d'optimisation créée avec succès.")
        return paths_created

    except OSError as e:
        logger.error(f"{log_prefix} Erreur OSError lors de la création des répertoires: {e}", exc_info=True)
        raise # Relaisser l'exception pour que l'appelant puisse la gérer
    except Exception as e_gen:
        logger.error(f"{log_prefix} Erreur inattendue lors de la création des répertoires: {e_gen}", exc_info=True)
        raise


def create_live_session_dirs(
    base_log_dir: Path,          # Ex: Path("logs/live_trading")
    base_state_dir: Path,        # Ex: Path("data/live_state")
    # run_id: str,               # Un ID global pour la session live, si utilisé
    account_alias: str,
    strategy_id_from_deployment: str, # L'ID complet du déploiement
    pair_symbol: str,
    context_label: str
) -> Dict[str, Path]:
    """
    Crée les répertoires de base pour une session de trading live spécifique.
    LiveTradingManager gère son propre fichier d'état. Cette fonction pourrait
    créer un répertoire parent pour un groupe de logs de session si un run_id de session est utilisé.
    Pour l'instant, elle se concentre sur la création de répertoires de logs basés sur les infos fournies.

    Args:
        base_log_dir: Chemin racine des logs pour le trading live.
        base_state_dir: Chemin racine pour les fichiers d'état.
        account_alias: Alias du compte utilisé.
        strategy_id_from_deployment: ID du déploiement de la stratégie.
        pair_symbol: Symbole de la paire.
        context_label: Label de contexte.

    Returns:
        Un dictionnaire contenant les chemins clés créés.
    """
    # Nettoyer les identifiants pour les noms de répertoires
    clean_account_alias = re.sub(r'[^\w\-_\.]', '_', account_alias)
    clean_strategy_id = re.sub(r'[^\w\-_\.]', '_', strategy_id_from_deployment)
    clean_pair = re.sub(r'[^\w\-_\.]', '_', pair_symbol)
    clean_context = re.sub(r'[^\w\-_\.]', '_', context_label)

    # Exemple de structure: logs/live_trading/ACCOUNT_ALIAS/STRATEGY_ID/PAIR_CONTEXT/
    # L'ID de session (basé sur le timestamp) est généralement inclus dans le nom du fichier de log.
    
    session_log_dir = base_log_dir / clean_account_alias / clean_strategy_id / f"{clean_pair}_{clean_context}"
    # Le fichier d'état est plus spécifique et géré par LiveTradingState,
    # mais on peut s'assurer que son répertoire de base existe.
    # state_dir_for_manager = base_state_dir (LiveTradingState ajoutera le nom de fichier)

    paths_created: Dict[str, Path] = {}
    log_prefix = f"[CreateLiveDirs][{account_alias}/{strategy_id_from_deployment}/{pair_symbol}/{context_label}]"
    logger.info(f"{log_prefix} Création des répertoires de session live...")

    try:
        ensure_dir_exists(session_log_dir)
        paths_created["session_log_dir"] = session_log_dir
        logger.debug(f"{log_prefix} Répertoire de log de session: {session_log_dir}")

        # Assurer que le répertoire d'état de base existe (le fichier spécifique est géré par LiveTradingState)
        ensure_dir_exists(base_state_dir)
        paths_created["base_state_dir_ensured"] = base_state_dir
        logger.debug(f"{log_prefix} Répertoire d'état de base assuré: {base_state_dir}")
        
        logger.info(f"{log_prefix} Répertoires de session live créés/assurés.")
        return paths_created
    except OSError as e:
        logger.error(f"{log_prefix} Erreur OSError lors de la création des répertoires de session live: {e}", exc_info=True)
        raise
    except Exception as e_gen:
        logger.error(f"{log_prefix} Erreur inattendue lors de la création des répertoires de session live: {e_gen}", exc_info=True)
        raise

if __name__ == '__main__':
    # Configuration du logging pour les tests directs
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Test de generate_run_id
    current_run_id = generate_run_id()
    logger.info(f"Generated run_id: {current_run_id}")
    current_run_id_custom = generate_run_id(prefix="opt_wfo")
    logger.info(f"Generated custom run_id: {current_run_id_custom}")

    # Test de create_optimization_run_dirs
    # Créer des chemins de base temporaires pour le test
    temp_project_root = Path("./temp_run_utils_test").resolve()
    temp_logs_base = temp_project_root / "logs" / "backtest_optimization"
    temp_results_base = temp_project_root / "results"

    try:
        logger.info(f"Création des répertoires de test dans: {temp_project_root}")
        # S'assurer que ensure_dir_exists est importé et fonctionnel pour le test
        if not ensure_dir_exists(temp_project_root): # type: ignore
             raise OSError(f"Impossible de créer le répertoire de test racine: {temp_project_root}")


        created_opt_paths = create_optimization_run_dirs(
            base_log_dir=temp_logs_base,
            base_results_dir=temp_results_base,
            run_id=current_run_id_custom,
            strategy_name="EMA_Cross",
            pair_symbol="BTCUSDT",
            context_label="5min_rsi_filter",
            num_folds=3
        )
        logger.info(f"Répertoires d'optimisation créés: {json.dumps(created_opt_paths, indent=2, default=str)}")

        # Vérifier si les répertoires existent (exemple)
        if created_opt_paths["strategy_context_log_dir"].exists() and \
           created_opt_paths["fold_log_dirs"][0].exists():
            logger.info("Vérification de l'existence des répertoires d'optimisation réussie.")
        else:
            logger.error("Erreur : Certains répertoires d'optimisation n'ont pas été créés comme prévu.")

        # Test de create_live_session_dirs
        temp_live_logs_base = temp_project_root / "logs" / "live_trading"
        temp_live_state_base = temp_project_root / "data" / "live_state"

        created_live_paths = create_live_session_dirs(
            base_log_dir=temp_live_logs_base,
            base_state_dir=temp_live_state_base,
            account_alias="my_binance_margin",
            strategy_id_from_deployment="EMA_Cross_BTCUSDT_5min_rsi_filter_live_deploy_001",
            pair_symbol="BTCUSDT",
            context_label="5min_rsi_filter"
        )
        logger.info(f"Répertoires de session live créés: {json.dumps(created_live_paths, indent=2, default=str)}")
        if created_live_paths["session_log_dir"].exists():
             logger.info("Vérification de l'existence du répertoire de log de session live réussie.")
        else:
            logger.error("Erreur : Le répertoire de log de session live n'a pas été créé.")


    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des tests de run_utils: {e}", exc_info=True)
    finally:
        # Optionnel: Nettoyer les répertoires de test
        # import shutil
        # if temp_project_root.exists():
        #     logger.info(f"Nettoyage du répertoire de test: {temp_project_root}")
        #     shutil.rmtree(temp_project_root)
        pass
