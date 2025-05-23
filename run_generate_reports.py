import argparse
import logging
import sys
from pathlib import Path
import time # Ajouté pour le temps d'exécution total
from datetime import timedelta # Ajouté pour formater le temps d'exécution

# Configuration initiale du logging (sera surchargée par load_all_configs)
# Cela est utile si des erreurs se produisent avant même que load_all_configs puisse s'exécuter.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger spécifique pour ce script

# --- Ajouter la racine du projet au PYTHONPATH ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    logger.debug(f"Ajouté {SRC_PATH} au PYTHONPATH")

try:
    from src.config.loader import load_all_configs, AppConfig
    from src.reporting.generator import generate_all_reports
    # setup_logging est appelé à l'intérieur de load_all_configs
except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE: Impossible d'importer les modules nécessaires: {e}. "
                    f"Vérifiez PYTHONPATH et les installations. CWD: {Path.cwd()}, sys.path: {sys.path}", exc_info=True)
    sys.exit(1)
except Exception as e_imp:
    logger.critical(f"ÉCHEC CRITIQUE: Erreur inattendue lors des imports initiaux: {e_imp}", exc_info=True)
    sys.exit(1)


def main():
    run_start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Génère les rapports (Markdown, Live Config JSON, etc.) "
                    "à partir des artefacts d'un run WFO spécifique."
    )
    parser.add_argument(
        "--timestamp", "--run_id", dest="run_id", # Accepter --timestamp ou --run_id
        type=str,
        required=True,
        help="Le nom du répertoire timestamp (ID du run) du run WFO à traiter (ex: '20250522_100000')."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None, # Sera déterminé par PROJECT_ROOT si non fourni
        help="Répertoire racine du projet si le script n'est pas exécuté depuis la racine."
    )
    args = parser.parse_args()

    project_root_arg = args.root if args.root else str(PROJECT_ROOT)

    app_config: Optional[AppConfig] = None
    try:
        # Le logging est configuré par load_all_configs.
        app_config = load_all_configs(project_root=project_root_arg)
        logger.info("--- Lancement de la Génération de Rapports pour le Run WFO ---")
        logger.info(f"Arguments du script: Run ID='{args.run_id}', Racine Projet='{project_root_arg}'")
        logger.info("Configuration de l'application chargée avec succès.")
    except FileNotFoundError as e_fnf:
        logger.critical(f"Fichier de configuration non trouvé: {e_fnf}. "
                        f"Vérifiez les chemins relatifs à la racine du projet: {project_root_arg}. Abandon.")
        sys.exit(1)
    except Exception as e_conf:
        logger.critical(f"Erreur critique lors du chargement de la configuration: {e_conf}", exc_info=True)
        sys.exit(1)

    if not app_config: # Double vérification
        logger.critical("AppConfig n'a pas pu être chargée. Abandon.")
        sys.exit(1)

    # Construction des chemins
    try:
        base_log_dir = Path(app_config.global_config.paths.logs_backtest_optimization)
        base_results_dir = Path(app_config.global_config.paths.results)

        run_specific_log_dir = base_log_dir / args.run_id
        run_specific_results_dir = base_results_dir / args.run_id
    except AttributeError as e_attr:
        logger.critical(f"Erreur d'attribut lors de l'accès aux chemins de configuration (AppConfig.global_config.paths): {e_attr}. "
                        "Vérifiez la structure de config_global.json et definitions.py.", exc_info=True)
        sys.exit(1)


    logger.info(f"Lecture des logs depuis : {run_specific_log_dir.resolve()}")
    logger.info(f"Écriture des résultats dans : {run_specific_results_dir.resolve()}")

    # Validation des chemins
    if not run_specific_log_dir.is_dir():
        logger.error(f"Le répertoire de logs spécifié pour le run ID '{args.run_id}' n'existe pas : {run_specific_log_dir}")
        logger.error("Vérifiez que le run ID est correct et que le run WFO a bien généré des logs.")
        sys.exit(1)

    # Appel à la génération des rapports
    try:
        logger.info(f"Appel de generate_all_reports pour le run ID: {args.run_id}")
        generate_all_reports(log_dir=run_specific_log_dir, results_dir=run_specific_results_dir)
        logger.info(f"Génération des rapports terminée avec succès pour le run ID: {args.run_id}.")

    except FileNotFoundError as fnf_err:
        logger.error(f"Erreur : Fichier non trouvé pendant la génération des rapports pour le run ID '{args.run_id}'. {fnf_err}", exc_info=True)
        # Le run_status.json sera mis à jour par generate_all_reports en cas d'erreurs internes.
        # Si l'erreur est ici, c'est probablement un problème de chemin avant d'appeler generate_all_reports.
    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de la génération des rapports pour le run ID '{args.run_id}': {e}", exc_info=True)
    finally:
        run_end_time = time.time()
        total_duration_seconds = run_end_time - run_start_time
        logger.info(f"Temps d'Exécution Total du Script de Génération de Rapports: {total_duration_seconds:.2f} secondes ({timedelta(seconds=total_duration_seconds)})")
        logger.info("--- Fin du Script de Génération de Rapports ---")
        logging.shutdown()

if __name__ == "__main__":
    main()
