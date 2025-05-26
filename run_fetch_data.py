# run_fetch_data.py
"""
Ce script orchestre la récupération, le nettoyage initial des données 1-minute,
l'agrégation en timeframes supérieurs, et le calcul d'indicateurs de base
(comme les ATRs sur différentes périodes) pour les données historiques.
"""
import logging
import os 
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
import re 
from typing import Optional, List, Dict, Any, TYPE_CHECKING, cast
from datetime import timedelta 

import pandas as pd
import numpy as np 

# --- Configuration initiale du logging (sera surchargée par load_all_configs) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__) 

# --- Ajout de la racine du projet au PYTHONPATH ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    logger.debug(f"Ajouté {SRC_PATH} au PYTHONPATH par run_fetch_data.py")

# --- Imports des modules de l'application ---
if TYPE_CHECKING:
    from src.config.loader import AppConfig

try:
    from src.config.loader import load_all_configs
    from src.data.acquisition import fetch_all_historical_data
    from src.data import data_utils 
except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE (run_fetch_data.py): Impossible d'importer les modules nécessaires: {e}. "
                    f"Vérifiez PYTHONPATH et les installations. CWD: {Path.cwd()}, sys.path: {sys.path}", exc_info=True)
    sys.exit(1)
except Exception as e_imp: # pylint: disable=broad-except
    logger.critical(f"ÉCHEC CRITIQUE (run_fetch_data.py): Erreur inattendue lors des imports initiaux: {e_imp}", exc_info=True)
    sys.exit(1)

# --- Définition des constantes utilisées dans ce script ---
BASE_OHLCV_COLS: List[str] = ['open', 'high', 'low', 'close', 'volume']

ATR_ENRICHMENT_CONFIG: Dict[str, int] = {
    "atr_low": 10,
    "atr_high": 21,
    "atr_step": 1
}

def _parse_timeframe_string_to_minutes(tf_string: str) -> Optional[int]:
    """
    Convertit une chaîne de timeframe (ex: "1m", "5min", "1h", "1d")
    en un nombre total de minutes.
    Retourne None si le format est inconnu ou invalide.
    """
    if not isinstance(tf_string, str) or not tf_string.strip():
        return None
    
    tf_lower = tf_string.lower().strip()
    match = re.fullmatch(r"(\d+)\s*(s|m|min|h|d|w)", tf_lower)
    if not match:
        logger.warning(f"_parse_timeframe_string_to_minutes: Format de timeframe non reconnu '{tf_string}'.")
        return None
    
    try:
        value = int(match.group(1))
        unit = match.group(2)
        if value <= 0: return None
    except ValueError:
        return None

    if unit == "s":
        if value % 60 == 0: return value // 60
        logger.debug(f"_parse_timeframe_string_to_minutes: Timeframe en secondes '{tf_string}' non multiple de 60s.")
        return None
    elif unit in ("m", "min"):
        return value
    elif unit == "h":
        return value * 60
    elif unit == "d":
        return value * 24 * 60
    elif unit == "w":
        return value * 7 * 24 * 60
    return None


def main():
    """
    Point d'entrée principal pour le script de récupération et d'enrichissement
    des données historiques.
    """
    script_start_time = time.time()

    parser = ArgumentParser(description="Récupère, nettoie et enrichit les données de marché historiques.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Spécifiez le répertoire racine du projet si le script n'est pas lancé depuis la racine du projet.",
    )
    args = parser.parse_args()

    project_root_arg = args.root if args.root else str(PROJECT_ROOT)
    project_root_path = Path(project_root_arg).resolve()

    logger.info(f"--- Démarrage du Script de Récupération et Enrichissement des Données (Racine Projet: {project_root_path}) ---")

    app_config: Optional['AppConfig'] = None
    try:
        app_config = load_all_configs(project_root=str(project_root_path))
        logger.info("Configuration de l'application chargée avec succès.")
    except Exception as e_conf: # pylint: disable=broad-except
        logger.critical(f"Échec du chargement de la configuration de l'application : {e_conf}", exc_info=True)
        sys.exit(1)

    if not app_config:
        logger.critical("AppConfig n'a pas pu être chargée (est None). Abandon.")
        sys.exit(1)

    # --- Étape 1 : Récupération et Nettoyage des Données 1-Minute ---
    logger.info("=== ÉTAPE 1: Démarrage de la récupération et du nettoyage initial des données historiques 1-minute ===")
    raw_data_run_dir_path_str: Optional[str] = None
    try:
        raw_data_run_dir_path_str = fetch_all_historical_data(app_config)
        
        if not raw_data_run_dir_path_str or not Path(raw_data_run_dir_path_str).is_dir():
            logger.error("La récupération des données historiques n'a pas retourné un chemin de répertoire valide pour les données brutes. "
                         "L'enrichissement ne peut pas continuer.")
            sys.exit(1)
        
        cleaned_1min_data_dir = Path(app_config.project_root) / app_config.global_config.paths.data_historical_processed_cleaned # type: ignore
        logger.info("Récupération et nettoyage initial des données historiques 1-minute terminés.")
        logger.info(f"  CSVs bruts (nettoyés) spécifiques à ce run sauvegardés dans : {raw_data_run_dir_path_str}")
        logger.info(f"  Fichiers Parquet 1-minute nettoyés disponibles dans : {cleaned_1min_data_dir}")

    except Exception as e_fetch: # pylint: disable=broad-except
        logger.error("Une erreur s'est produite lors de la récupération et du nettoyage des données historiques 1-minute.", exc_info=True)
        logger.error("Abandon du script en raison d'une erreur lors de l'étape de récupération/nettoyage.")
        sys.exit(1)

    # --- Étape 2 : Enrichissement des Données (Agrégation et Indicateurs de Base) ---
    logger.info("=== ÉTAPE 2: Démarrage du processus d'enrichissement des données (agrégation et calcul d'indicateurs de base) ===")
    
    enriched_data_output_dir = Path(app_config.project_root) / app_config.global_config.paths.data_historical_processed_enriched # type: ignore
    try:
        enriched_data_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Les données enrichies seront sauvegardées dans : {enriched_data_output_dir}")
    except OSError as e_mkdir_enriched:
        logger.error(f"Échec de la création du répertoire des données enrichies {enriched_data_output_dir}: {e_mkdir_enriched}. Abandon.")
        sys.exit(1)

    pairs_to_process: List[str] = app_config.data_config.assets_and_timeframes.pairs
    config_timeframes_str: List[str] = app_config.data_config.assets_and_timeframes.timeframes
    
    timeframes_for_aggregation_minutes: List[int] = []
    for tf_str_from_config in config_timeframes_str:
        minutes = _parse_timeframe_string_to_minutes(tf_str_from_config)
        if minutes is not None and minutes > 1:
            if minutes not in timeframes_for_aggregation_minutes:
                 timeframes_for_aggregation_minutes.append(minutes)
        elif minutes == 1:
            logger.debug(f"Timeframe '1min' trouvé dans la config, ne sera pas agrégé davantage ici.")
    
    timeframes_for_aggregation_minutes.sort()
    logger.info(f"Timeframes cibles pour l'agrégation (en minutes) : {timeframes_for_aggregation_minutes}")

    if not timeframes_for_aggregation_minutes and not ("1min" in config_timeframes_str or "1m" in config_timeframes_str) :
        logger.info("Aucun timeframe (y compris 1-min pour ATR direct) configuré pour l'enrichissement. "
                    "L'enrichissement se limitera aux données 1-minute de base.")


    for pair_symbol_current in pairs_to_process:
        pair_log_prefix = f"[{pair_symbol_current.upper()}]"
        logger.info(f"{pair_log_prefix} Démarrage du processus d'enrichissement...")
        
        path_to_cleaned_1min_parquet = cleaned_1min_data_dir / f"{pair_symbol_current.upper()}_1min_cleaned_with_taker.parquet"
        if not path_to_cleaned_1min_parquet.is_file():
            logger.warning(f"{pair_log_prefix} Fichier Parquet 1-minute nettoyé non trouvé : {path_to_cleaned_1min_parquet}. "
                           "Enrichissement sauté pour cette paire.")
            continue

        try:
            df_1min_current_pair = pd.read_parquet(path_to_cleaned_1min_parquet)
            logger.debug(f"{pair_log_prefix} Chargé {path_to_cleaned_1min_parquet.name}, shape: {df_1min_current_pair.shape}")
        except Exception as e_load_pq_enrich: # pylint: disable=broad-except
            logger.error(f"{pair_log_prefix} Erreur lors du chargement du fichier Parquet {path_to_cleaned_1min_parquet} pour enrichissement : {e_load_pq_enrich}", exc_info=True)
            continue
        
        if df_1min_current_pair.empty:
            logger.warning(f"{pair_log_prefix} DataFrame chargé depuis {path_to_cleaned_1min_parquet.name} est vide. Enrichissement sauté.")
            continue

        if 'timestamp' not in df_1min_current_pair.columns:
            logger.error(f"{pair_log_prefix} Colonne 'timestamp' manquante dans {path_to_cleaned_1min_parquet.name}. Enrichissement impossible.")
            continue
        
        try:
            df_1min_current_pair['timestamp'] = pd.to_datetime(df_1min_current_pair['timestamp'], errors='coerce', utc=True)
            df_1min_current_pair.dropna(subset=['timestamp'], inplace=True)
            # df_1min_current_pair_indexed est utilisé pour le resampling, il a besoin de 'timestamp' comme index
            df_1min_current_pair_indexed = df_1min_current_pair.set_index('timestamp') # drop=True par défaut
            if not df_1min_current_pair_indexed.index.is_monotonic_increasing:
                df_1min_current_pair_indexed = df_1min_current_pair_indexed.sort_index()
            if not df_1min_current_pair_indexed.index.is_unique:
                df_1min_current_pair_indexed = df_1min_current_pair_indexed[~df_1min_current_pair_indexed.index.duplicated(keep='first')]
            logger.debug(f"{pair_log_prefix} Index 'timestamp' préparé pour resampling. Shape : {df_1min_current_pair_indexed.shape}")
        except Exception as e_idx_prep_enrich: # pylint: disable=broad-except
            logger.error(f"{pair_log_prefix} Erreur lors de la préparation de l'index 'timestamp' pour {path_to_cleaned_1min_parquet.name}: {e_idx_prep_enrich}", exc_info=True)
            continue
            
        if df_1min_current_pair_indexed.empty:
            logger.warning(f"{pair_log_prefix} DataFrame devenu vide après préparation de l'index 'timestamp'. Enrichissement sauté.")
            continue

        # final_df_for_pair commence avec les données 1-min (avec 'timestamp' comme colonne)
        final_df_for_pair = df_1min_current_pair.copy() 
        if not all(col in final_df_for_pair.columns for col in BASE_OHLCV_COLS):
            logger.error(f"{pair_log_prefix} Colonnes OHLCV de base manquantes dans df_1min après chargement. Enrichissement impossible.")
            continue

        for tf_mins in timeframes_for_aggregation_minutes:
            tf_log_prefix_loop = f"{pair_log_prefix}[TF:{tf_mins}min]"
            logger.info(f"{tf_log_prefix_loop} Agrégation des K-lines et calcul des ATRs...")

            df_aggregated_tf = data_utils.aggregate_klines_to_dataframe(
                df_1min_current_pair_indexed, 
                tf_mins
            )

            if df_aggregated_tf.empty:
                logger.warning(f"{tf_log_prefix_loop} L'agrégation a résulté en un DataFrame vide. Calcul ATR sauté pour ce timeframe.")
                continue
            
            df_aggregated_tf_with_atr = data_utils.calculate_atr_for_dataframe(
                df_aggregated_tf,
                atr_low=ATR_ENRICHMENT_CONFIG['atr_low'],
                atr_high=ATR_ENRICHMENT_CONFIG['atr_high'],
                atr_step=ATR_ENRICHMENT_CONFIG['atr_step']
            )

            tf_string_for_prefix = f"{tf_mins}min" 
            kline_col_prefix = data_utils.get_kline_prefix_effective(tf_string_for_prefix)
            
            renamed_cols_for_merge: Dict[str, str] = {}
            for col_agg in df_aggregated_tf_with_atr.columns:
                if col_agg.startswith('ATR_'):
                    renamed_cols_for_merge[col_agg] = f"{kline_col_prefix}_{col_agg}"
                else: 
                    renamed_cols_for_merge[col_agg] = f"{kline_col_prefix}_{col_agg}"
            
            df_to_merge = df_aggregated_tf_with_atr.rename(columns=renamed_cols_for_merge)
            df_to_merge = df_to_merge.reset_index() # 'timestamp' (fin de période agrégée) devient une colonne

            if not df_to_merge.empty:
                final_df_for_pair = pd.merge(final_df_for_pair, df_to_merge, on='timestamp', how='left')
                cols_just_merged = [col for col in final_df_for_pair.columns if col.startswith(kline_col_prefix)]
                if cols_just_merged:
                    final_df_for_pair[cols_just_merged] = final_df_for_pair[cols_just_merged].ffill()
                logger.debug(f"{tf_log_prefix_loop} Données agrégées et ATRs fusionnées. Shape de final_df_for_pair : {final_df_for_pair.shape}")
            else:
                logger.warning(f"{tf_log_prefix_loop} DataFrame à fusionner (df_to_merge) est vide après renommage/reset_index.")
        
        if "1min" in config_timeframes_str or "1m" in config_timeframes_str:
            logger.info(f"{pair_log_prefix} Calcul des ATRs sur les données 1-minute de base...")
            # df_1min_current_pair_indexed a 'timestamp' comme index.
            # calculate_atr_for_dataframe prend un df et retourne une copie avec les colonnes ATR.
            # Il ne modifie pas l'index.
            df_1min_with_atr_direct = data_utils.calculate_atr_for_dataframe(
                df_1min_current_pair_indexed.copy(), # Utiliser une copie
                atr_low=ATR_ENRICHMENT_CONFIG['atr_low'],
                atr_high=ATR_ENRICHMENT_CONFIG['atr_high'],
                atr_step=ATR_ENRICHMENT_CONFIG['atr_step']
            )
            atr_cols_1min = [col for col in df_1min_with_atr_direct.columns if col.startswith('ATR_')]
            if atr_cols_1min and not df_1min_with_atr_direct.empty:
                # Pour fusionner, nous avons besoin de 'timestamp' comme colonne dans df_1min_with_atr_direct.
                # Puisque df_1min_current_pair_indexed avait 'timestamp' comme index (drop=True implicite),
                # df_1min_with_atr_direct l'aura aussi comme index. reset_index() le mettra en colonne.
                df_atrs_to_merge_1min = df_1min_with_atr_direct.reset_index() # 'timestamp' devient une colonne
                
                cols_for_merge_1min_atr = ['timestamp'] + atr_cols_1min
                if all(c in df_atrs_to_merge_1min.columns for c in cols_for_merge_1min_atr):
                    final_df_for_pair = pd.merge(
                        final_df_for_pair, 
                        df_atrs_to_merge_1min[cols_for_merge_1min_atr], 
                        on='timestamp', 
                        how='left', 
                        suffixes=('', '_1min_atr_direct_dup') 
                    )
                    logger.debug(f"{pair_log_prefix} ATRs 1-minute fusionnés. Shape de final_df_for_pair : {final_df_for_pair.shape}")
                else:
                    logger.warning(f"{pair_log_prefix} Colonnes manquantes pour la fusion des ATRs 1-minute directs. Attendu: {cols_for_merge_1min_atr}, Trouvé: {df_atrs_to_merge_1min.columns.tolist()}")
            else:
                logger.warning(f"{pair_log_prefix} Le calcul direct des ATRs 1-minute n'a pas produit de colonnes ATR ou un DataFrame vide.")


        output_file_enriched_pair = enriched_data_output_dir / f"{pair_symbol_current.upper()}_enriched.parquet"
        try:
            if 'timestamp' in final_df_for_pair.columns:
                cols_ordered = ['timestamp'] + [col for col in final_df_for_pair.columns if col != 'timestamp']
                final_df_for_pair = final_df_for_pair[cols_ordered]
            
            final_df_for_pair.to_parquet(output_file_enriched_pair, index=False, engine='pyarrow')
            logger.info(f"{pair_log_prefix} Données enrichies sauvegardées dans {output_file_enriched_pair} (Shape: {final_df_for_pair.shape})")
        except Exception as e_save_final_enriched: # pylint: disable=broad-except
            logger.error(f"{pair_log_prefix} Erreur lors de la sauvegarde du fichier Parquet enrichi {output_file_enriched_pair}: {e_save_final_enriched}", exc_info=True)

    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    logger.info(f"--- Script de Récupération et Enrichissement des Données Terminé ---")
    logger.info(f"Temps d'exécution total : {total_duration:.2f} secondes ({timedelta(seconds=total_duration)})")

if __name__ == "__main__":
    main()
