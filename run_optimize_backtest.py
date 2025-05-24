import argparse
import logging
import sys
import time
import json # Pour sauvegarder run_config.json
import dataclasses # Pour dataclasses.is_dataclass et dataclasses.asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union # Union ajouté pour la compatibilité
from datetime import datetime, timedelta # <<< IMPORT timedelta AJOUTÉ ICI
import numpy as np # <<< IMPORT numpy AJOUTÉ ICI (pour EnhancedJSONEncoder)


# Détection de la racine du projet
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    # Fallback si __file__ n'est pas défini (ex: environnement interactif)
    PROJECT_ROOT = Path(".").resolve()

# Ajout de src au PYTHONPATH si nécessaire
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    # print(f"Added {str(SRC_PATH)} to sys.path for module imports.")


# Configuration initiale du logging (sera surchargée par load_all_configs)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger spécifique pour ce script

try:
    from src.config.loader import load_all_configs, AppConfig
    from src.backtesting.wfo import WalkForwardOptimizer
    # setup_logging est appelé à l'intérieur de load_all_configs
except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE: Impossible d'importer les modules nécessaires: {e}. "
                    f"Vérifiez PYTHONPATH et les installations. CWD: {Path.cwd()}, sys.path: {sys.path}", exc_info=True)
    sys.exit(1)
except Exception as e_imp:
    logger.critical(f"ÉCHEC CRITIQUE: Erreur inattendue lors des imports initiaux: {e_imp}", exc_info=True)
    sys.exit(1)


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Encodeur JSON amélioré pour sérialiser les dataclasses, Path, et datetime.
    """
    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, datetime): # pd.Timestamp hérite de datetime
            return o.isoformat()
        # Gérer d'autres types non sérialisables nativement si nécessaire
        # Par exemple, les types numpy :
        if isinstance(o, (np.integer, np.int64)): # type: ignore
            return int(o)
        if isinstance(o, (np.floating, np.float64)): # type: ignore
            # Gérer NaN et Inf pour une sérialisation JSON valide
            if np.isnan(o): # type: ignore
                return None # ou "NaN" si vous préférez une chaîne
            if np.isinf(o): # type: ignore
                return "Infinity" if o > 0 else "-Infinity"
            return float(o)
        if isinstance(o, np.ndarray): # type: ignore
            return o.tolist()
        
        try:
            return super().default(o)
        except TypeError:
            logger.warning(f"Type non sérialisable rencontré par EnhancedJSONEncoder: {type(o)}. Rendu comme string.")
            return str(o)


def main():
    run_start_time = time.time()

    parser = argparse.ArgumentParser(description="Exécute l'Optimisation Walk-Forward (WFO).")
    parser.add_argument(
        "--pair",
        type=str,
        required=True,
        help="Paire de trading à optimiser (ex: BTCUSDT)."
    )
    parser.add_argument(
        "--tf", "--context", dest="tf", # Accepter --tf ou --context
        type=str,
        required=True,
        help="Label de contexte principal pour cette exécution (ex: 5min_rsi_context)."
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
        app_config = load_all_configs(project_root=project_root_arg)
        logger.info("--- Démarrage du Script d'Optimisation Walk-Forward ---")
        logger.info(f"Arguments du script: Paire='{args.pair}', Contexte (tf)='{args.tf}', Racine Projet='{project_root_arg}'")
        logger.info("Configuration de l'application chargée avec succès.")

    except FileNotFoundError as e_fnf:
        logger.critical(f"Fichier de configuration non trouvé: {e_fnf}. "
                        f"Vérifiez les chemins relatifs à la racine du projet: {project_root_arg}. Abandon.")
        sys.exit(1)
    except Exception as e_conf:
        logger.critical(f"Erreur critique lors du chargement de la configuration: {e_conf}", exc_info=True)
        sys.exit(1)

    if not app_config: 
        logger.critical("AppConfig n'a pas pu être chargée. Abandon.")
        sys.exit(1)

    try:
        logger.info(f"Initialisation de WalkForwardOptimizer...")
        wfo_runner = WalkForwardOptimizer(app_config=app_config)
        logger.info(f"WalkForwardOptimizer initialisé. Répertoire de sortie du run: {wfo_runner.run_output_dir}")

        run_config_path = wfo_runner.run_output_dir / "run_config.json"
        try:
            logger.info(f"Sauvegarde de la configuration du run dans: {run_config_path}")
            with open(run_config_path, 'w', encoding='utf-8') as f_cfg:
                json.dump(app_config, f_cfg, cls=EnhancedJSONEncoder, indent=4)
            logger.info(f"Configuration du run sauvegardée avec succès.")
        except Exception as e_save_cfg:
            logger.error(f"Échec de la sauvegarde de run_config.json: {e_save_cfg}", exc_info=True)

        pairs_to_run = [args.pair.upper()]
        contexts_to_run = [args.tf] # Le nom de contexte brut est passé ici

        logger.info(f"Démarrage de wfo_runner.run pour Paire(s): {pairs_to_run}, Contexte(s): {contexts_to_run}...")
        wfo_results = wfo_runner.run(pairs=pairs_to_run, context_labels=contexts_to_run)
        logger.info(f"wfo_runner.run terminé.")
        
        if wfo_results:
            logger.info(f"Optimisation WFO terminée. Nombre de résultats de combinaisons (stratégie/paire/contexte) traités: {len(wfo_results)}.")
            for combo_key, summary_data in wfo_results.items():
                num_folds_in_summary = len(summary_data.get("folds_data", []))
                # Utiliser le context_label_sanitized pour construire le chemin du log, car c'est celui utilisé pour créer le répertoire
                sanitized_ctx_for_path = summary_data.get('context_label', summary_data.get('raw_context_label_input', 'unknown_context'))
                logger.info(f"  - Résumé pour '{combo_key}': {num_folds_in_summary} fold(s) traité(s). "
                            f"Voir {wfo_runner.run_output_dir / summary_data.get('strategy_name', '') / summary_data.get('pair_symbol', '') / sanitized_ctx_for_path / 'wfo_strategy_pair_summary.json'}")
        else:
            logger.warning("Optimisation WFO terminée, mais aucun résultat n'a été retourné par wfo_runner.run.")

    except FileNotFoundError as e_fnf_wfo:
        logger.critical(f"Un fichier nécessaire n'a pas été trouvé pendant l'exécution de WFO: {e_fnf_wfo}. Abandon.", exc_info=True)
    except ImportError as e_imp_wfo:
        logger.critical(f"Erreur d'importation de module pendant l'exécution de WFO: {e_imp_wfo}. Abandon.", exc_info=True)
    except Exception as e_wfo:
        logger.critical(f"Une erreur non interceptée s'est produite pendant l'exécution de WFO: {e_wfo}", exc_info=True)
    finally:
        run_end_time = time.time()
        total_duration_seconds = run_end_time - run_start_time
        logger.info(f"Temps d'Exécution Total du Script: {total_duration_seconds:.2f} secondes ({timedelta(seconds=total_duration_seconds)})")
        logger.info("--- Fin du Script d'Optimisation Walk-Forward ---")
        logging.shutdown() 

if __name__ == "__main__":
    main()
