# run_generate_reports.py
"""
Point d'entrée pour générer les rapports consolidés d'un run
Walk-Forward Optimization (WFO) complet.
Prend en argument l'ID du run WFO à traiter et orchestre la génération
des rapports via le module master_report_generator.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import timedelta, timezone # timezone ajouté pour une éventuelle utilisation
from typing import Optional, TYPE_CHECKING

# Configuration initiale du logging (sera surchargée par load_all_configs)
# Utile si des erreurs surviennent avant que load_all_configs ne s'exécute.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__) # Logger spécifique pour ce script

# --- Ajout de la racine du projet au PYTHONPATH ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    # Fallback si __file__ n'est pas défini (ex: environnement interactif)
    PROJECT_ROOT = Path(".").resolve()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    logger.debug(f"Ajouté {SRC_PATH} au PYTHONPATH par run_generate_reports.py")

# Imports des modules de l'application
if TYPE_CHECKING:
    from src.config.loader import AppConfig

try:
    from src.config.loader import load_all_configs
    # La fonction generate_all_reports sera dans un nouveau module.
    # Pour l'instant, on définit un placeholder si le module n'existe pas encore.
    # from src.reporting.master_report_generator import generate_all_reports
except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE: Impossible d'importer les modules nécessaires: {e}. "
                    f"Vérifiez PYTHONPATH et les installations. CWD: {Path.cwd()}, sys.path: {sys.path}", exc_info=True)
    sys.exit(1)
except Exception as e_imp: # pylint: disable=broad-except
    logger.critical(f"ÉCHEC CRITIQUE: Erreur inattendue lors des imports initiaux: {e_imp}", exc_info=True)
    sys.exit(1)

# --- Placeholder pour master_report_generator ---
# Ce placeholder sera utilisé si le module réel n'est pas encore créé.
try:
    from src.reporting.master_report_generator import generate_all_reports
except ImportError:
    logger.warning(
        "Module src.reporting.master_report_generator non trouvé. "
        "Utilisation d'un placeholder pour generate_all_reports."
    )
    def generate_all_reports(log_dir: Path, results_dir: Path, app_config: 'AppConfig') -> None: # app_config ajouté
        """
        Placeholder pour la fonction de génération de tous les rapports.
        """
        logger.info(f"[Placeholder] La fonction generate_all_reports serait appelée avec :")
        logger.info(f"[Placeholder]   log_dir: {log_dir}")
        logger.info(f"[Placeholder]   results_dir: {results_dir}")
        logger.info(f"[Placeholder]   app_config.project_name: {app_config.global_config.project_name}")
        
        # Simuler la création d'un fichier de statut de rapport
        status_file = results_dir / "report_generation_status_placeholder.txt"
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            with open(status_file, "w", encoding="utf-8") as f:
                f.write(f"Rapports générés (placeholder) pour le run traité par les logs dans : {log_dir}\n")
                f.write(f"Résultats attendus dans : {results_dir}\n")
            logger.info(f"[Placeholder] Fichier de statut de rapport placeholder créé : {status_file}")
        except Exception as e_placeholder_write: # pylint: disable=broad-except
            logger.error(f"[Placeholder] Erreur lors de la création du fichier de statut placeholder : {e_placeholder_write}")
# --- Fin du Placeholder ---


def main():
    """
    Point d'entrée principal pour le script de génération de rapports.
    """
    run_start_time = time.time()
    # Le logging est configuré par load_all_configs au début.

    parser = argparse.ArgumentParser(
        description="Génère les rapports consolidés pour un run Walk-Forward Optimization (WFO) spécifié."
    )
    parser.add_argument(
        "--timestamp", "--run_id", dest="run_id",
        type=str,
        required=True,
        help="L'identifiant du run WFO (nom du répertoire timestamp, ex: 'opt_YYYYMMDD_HHMMSS') à traiter."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Chemin vers la racine du projet si le script n'est pas exécuté depuis cette racine. "
             "Par défaut, la racine est déduite de l'emplacement du script."
    )
    args = parser.parse_args()

    project_root_arg = args.root if args.root else str(PROJECT_ROOT)
    project_root_path = Path(project_root_arg).resolve()

    app_config: Optional['AppConfig'] = None
    try:
        # load_all_configs configure également le logging de l'application
        app_config = load_all_configs(project_root=str(project_root_path))
        logger.info("--- Démarrage du Script de Génération de Rapports WFO ---")
        logger.info(f"Arguments du script: Run ID='{args.run_id}', Racine Projet='{project_root_path}'")
        logger.info("Configuration de l'application chargée avec succès.")

    except FileNotFoundError as e_fnf:
        logger.critical(f"Fichier de configuration non trouvé: {e_fnf}. "
                        f"Vérifiez les chemins relatifs à la racine du projet: {project_root_path}. Abandon.", exc_info=True)
        sys.exit(1)
    except Exception as e_conf: # pylint: disable=broad-except
        logger.critical(f"Erreur critique lors du chargement de la configuration: {e_conf}", exc_info=True)
        sys.exit(1)

    if not app_config: # Vérification supplémentaire, bien que load_all_configs devrait lever une exception avant.
        logger.critical("AppConfig n'a pas pu être chargée (est None). Abandon.")
        sys.exit(1)

    # Construction des chemins d'accès aux répertoires de logs et de résultats pour le run_id
    try:
        # S'assurer que les chemins dans app_config sont des Path objets ou des strings convertibles
        base_log_dir = Path(app_config.global_config.paths.logs_backtest_optimization)
        base_results_dir = Path(app_config.global_config.paths.results)

        # Le run_id est le nom du répertoire sous base_log_dir et base_results_dir
        run_specific_log_dir = base_log_dir / args.run_id
        run_specific_results_dir = base_results_dir / args.run_id
    except AttributeError as e_attr:
        logger.critical(f"Erreur d'attribut lors de l'accès aux chemins de configuration "
                        f"(AppConfig.global_config.paths.*): {e_attr}. "
                        "Vérifiez la structure de config_global.json et definitions.py.", exc_info=True)
        sys.exit(1)
    except TypeError as e_type:
        logger.critical(f"Erreur de type lors de la construction des chemins (vérifiez que les chemins dans "
                        f"AppConfig sont des strings ou Path) : {e_type}", exc_info=True)
        sys.exit(1)


    logger.info(f"Lecture des logs depuis le répertoire du run : {run_specific_log_dir.resolve()}")
    logger.info(f"Écriture des rapports dans le répertoire de résultats du run : {run_specific_results_dir.resolve()}")

    # Vérification de l'existence du répertoire de logs du run
    if not run_specific_log_dir.is_dir():
        logger.error(f"Le répertoire de logs spécifié pour le run ID '{args.run_id}' n'existe pas : {run_specific_log_dir}")
        logger.error("Vérifiez que l'ID du run est correct et que le run WFO (via run_optimize_backtest.py) "
                     "a bien généré des logs dans ce répertoire.")
        sys.exit(1)

    # Appel à la fonction de génération des rapports
    try:
        logger.info(f"Appel de generate_all_reports pour le run ID: {args.run_id}...")
        # La fonction generate_all_reports est attendue dans src.reporting.master_report_generator
        generate_all_reports(
            log_dir=run_specific_log_dir,
            results_dir=run_specific_results_dir,
            app_config=app_config # Passer app_config pour accès aux configurations globales
        )
        logger.info(f"Génération des rapports terminée avec succès pour le run ID: {args.run_id}.")
        logger.info(f"Les rapports devraient être disponibles dans : {run_specific_results_dir.resolve()}")

    except FileNotFoundError as fnf_err_gen:
        logger.error(f"Erreur : Fichier non trouvé pendant la génération des rapports pour le run ID '{args.run_id}'. {fnf_err_gen}", exc_info=True)
    except Exception as e_gen_all: # pylint: disable=broad-except
        logger.error(f"Une erreur s'est produite lors de la génération des rapports pour le run ID '{args.run_id}': {e_gen_all}", exc_info=True)
    finally:
        run_end_time = time.time()
        total_duration_seconds = run_end_time - run_start_time
        logger.info(f"Temps d'Exécution Total du Script de Génération de Rapports: {total_duration_seconds:.2f} secondes "
                    f"({timedelta(seconds=total_duration_seconds)})")
        logger.info("--- Fin du Script de Génération de Rapports ---")
        logging.shutdown() # S'assurer que tous les messages de log sont écrits

if __name__ == "__main__":
    main()
