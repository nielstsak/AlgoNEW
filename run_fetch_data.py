import logging
import os # Peut être remplacé par pathlib pour la plupart des usages
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
import re # Pour convertir les minutes en label de fréquence

import pandas as pd
import numpy as np # Pour la gestion des NaN si besoin

# Imports depuis le projet src
try:
    # Ajuster le chemin si src n'est pas directement dans le PYTHONPATH
    # Cela suppose que le script est lancé depuis la racine du projet ou que src y est.
    if str(Path.cwd() / 'src') not in sys.path and str(Path(__file__).resolve().parent / 'src') not in sys.path :
         # Si src n'est pas dans le CWD, et que ce script n'est pas DANS src, on ajoute le parent/src
         # Cela suppose que ce script est à la racine, et src est un sous-dossier.
        sys.path.insert(0, str(Path(__file__).resolve().parent)) # Ajouter la racine du projet
        # print(f"Added to sys.path: {str(Path(__file__).resolve().parent)}")


    from src.config.loader import load_all_configs, AppConfig
    from src.data.acquisition import fetch_all_historical_data # Pour le fetch et nettoyage initial 1-min
    from src.data import data_utils # Pour l'agrégation et le calcul d'ATR sur les timeframes supérieurs
    # setup_logging est appelé dans load_all_configs
except ImportError as e:
    # Fallback logger si les imports ci-dessus échouent avant que le logging principal soit configuré
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.critical(f"CRITICAL ERROR: Failed to import necessary modules: {e}. "
                     f"PYTHONPATH: {sys.path}. CWD: {Path.cwd()}", exc_info=True)
    sys.exit(1)


logger = logging.getLogger(__name__)

# Configuration pour les périodes ATR (peut être externalisée dans config_data.json à l'avenir)
ATR_CONFIG = {
    "atr_low": 10,
    "atr_high": 21, # Inclusif
    "atr_step": 1   # Pour générer ATR_10, ATR_11, ..., ATR_21
    # Alternative: une liste fixe de périodes, ex: [14, 20]
    # "atr_periods": [14, 20]
}

def _parse_timeframe_string_to_minutes(tf_string: str) -> Optional[int]:
    """
    Convertit une chaîne de timeframe (ex: "1m", "5min", "1h", "1d") en un nombre total de minutes.
    Retourne None si le format est inconnu.
    """
    if not isinstance(tf_string, str) or not tf_string.strip():
        return None
    
    tf_lower = tf_string.lower().strip()
    match = re.fullmatch(r"(\d+)\s*(m|min|h|d)", tf_lower)
    if not match:
        return None
    
    value = int(match.group(1))
    unit = match.group(2)

    if unit in ("m", "min"):
        return value
    elif unit == "h":
        return value * 60
    elif unit == "d":
        return value * 60 * 24
    return None

def _get_frequency_label_from_minutes(tf_minutes: int) -> str:
    """
    Convertit un nombre de minutes en un label de fréquence string (ex: 5 -> "5min", 60 -> "60min").
    Utilisé pour le nommage des colonnes.
    """
    if tf_minutes < 60:
        return f"{tf_minutes}min"
    elif tf_minutes % 60 == 0 and tf_minutes < (24*60) :
        hours = tf_minutes // 60
        return f"{hours}h" # Ou f"{tf_minutes}min" pour la cohérence avec get_kline_prefix_effective
    elif tf_minutes % (24*60) == 0:
        days = tf_minutes // (24*60)
        return f"{days}d" # Ou f"{tf_minutes}min"
    return f"{tf_minutes}min" # Fallback


def main():
    script_start_time = time.time()
    # Le logging est configuré par load_all_configs

    parser = ArgumentParser(description="Fetch, clean, and enrich historical market data.")
    parser.add_argument(
        "--root",
        type=str,
        default=None, # Sera déterminé si non fourni
        help="Specify the project root directory if the script is not run from the project root.",
    )
    args = parser.parse_args()

    project_root_arg = args.root
    if project_root_arg is None:
        # Si --root n'est pas fourni, on suppose que ce script est à la racine du projet.
        project_root_arg = str(Path(__file__).resolve().parent)
        
    logger.info(f"--- Starting Data Fetch, Clean, and Enrichment Script (Project Root: {project_root_arg}) ---")

    try:
        config: AppConfig = load_all_configs(project_root=project_root_arg)
        logger.info("Application configuration loaded successfully.")
    except Exception as e_conf:
        logger.critical(f"Failed to load application configuration: {e_conf}", exc_info=True)
        sys.exit(1)

    # --- Étape 1 : Fetch et Nettoyage des Données 1-Minute ---
    logger.info("Step 1: Starting historical 1-minute data fetching and initial cleaning...")
    try:
        # fetch_all_historical_data sauvegarde les .parquet nettoyés dans paths.data_historical_processed_cleaned
        # et retourne le chemin du répertoire de run pour les CSV "bruts-nettoyés".
        raw_data_run_dir_str = fetch_all_historical_data(config) # type: ignore
        
        if not raw_data_run_dir_str or not Path(raw_data_run_dir_str).is_dir():
            logger.error("Historical data fetching and cleaning did not return a valid raw data directory path. Aborting enrichment.")
            return
        
        cleaned_1min_data_dir = Path(config.global_config.paths.data_historical_processed_cleaned)
        logger.info(f"Historical 1-minute data fetching and cleaning complete.")
        logger.info(f"Run-specific raw (cleaned) CSVs saved in: {raw_data_run_dir_str}")
        logger.info(f"Cleaned 1-minute Parquet files available in: {cleaned_1min_data_dir}")

    except Exception as fetch_exc:
        logger.error("An error occurred during historical data fetching and cleaning.", exc_info=True)
        logger.error("Aborting script due to fetching/cleaning error.")
        return

    # --- Étape 2 : Enrichissement des Données (Agrégation et Calcul d'Indicateurs de Base) ---
    logger.info("Step 2: Starting data enrichment process (aggregation and base indicator calculation)...")
    
    enriched_data_dir = Path(config.global_config.paths.data_historical_processed_enriched)
    try:
        enriched_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Enriched data will be saved to: {enriched_data_dir}")
    except OSError as e_mkdir:
        logger.error(f"Failed to create enriched data directory {enriched_data_dir}: {e_mkdir}. Aborting.")
        return

    pairs_to_process = config.data_config.assets_and_timeframes.pairs
    timeframes_config_strings = config.data_config.assets_and_timeframes.timeframes
    
    timeframes_to_aggregate_minutes: List[int] = []
    for tf_str in timeframes_config_strings:
        minutes = _parse_timeframe_string_to_minutes(tf_str)
        if minutes is not None and minutes > 1: # On n'agrège pas le 1-min sur lui-même
            if minutes not in timeframes_to_aggregate_minutes: # Eviter les doublons
                 timeframes_to_aggregate_minutes.append(minutes)
        elif minutes == 1:
            logger.debug(f"Timeframe '1min' or equivalent found in config, will not be aggregated further in this step.")
        else:
            logger.warning(f"Could not parse timeframe string '{tf_str}' to minutes. It will be ignored for aggregation.")
    
    timeframes_to_aggregate_minutes.sort() # Traiter par ordre croissant de timeframe
    logger.info(f"Timeframes to aggregate (in minutes): {timeframes_to_aggregate_minutes}")

    if not timeframes_to_aggregate_minutes:
        logger.info("No timeframes > 1min configured for aggregation. Enrichment will only consist of base 1-min data.")


    for pair_symbol in pairs_to_process:
        pair_log_prefix = f"[{pair_symbol}]"
        logger.info(f"{pair_log_prefix} Starting enrichment process...")
        
        cleaned_1min_file = cleaned_1min_data_dir / f"{pair_symbol}_1min_cleaned_with_taker.parquet"
        if not cleaned_1min_file.is_file(): # Utiliser is_file() pour plus de robustesse
            logger.warning(f"{pair_log_prefix} Cleaned 1-minute data file not found: {cleaned_1min_file}. Skipping enrichment for this pair.")
            continue

        try:
            df_1min = pd.read_parquet(cleaned_1min_file)
            logger.debug(f"{pair_log_prefix} Loaded {cleaned_1min_file.name}, shape: {df_1min.shape}")
        except Exception as e_load_parquet:
            logger.error(f"{pair_log_prefix} Error loading Parquet file {cleaned_1min_file}: {e_load_parquet}", exc_info=True)
            continue
        
        if df_1min.empty:
            logger.warning(f"{pair_log_prefix} DataFrame from {cleaned_1min_file.name} is empty. Skipping.")
            continue

        if 'timestamp' not in df_1min.columns:
            logger.error(f"{pair_log_prefix} 'timestamp' column missing in {cleaned_1min_file.name}. Cannot proceed with enrichment.")
            continue
        
        try:
            df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'], errors='coerce', utc=True)
            df_1min.dropna(subset=['timestamp'], inplace=True) # Important si des conversions échouent
            df_1min = df_1min.set_index('timestamp', drop=False) # Garder 'timestamp' comme colonne aussi
            if not df_1min.index.is_monotonic_increasing:
                df_1min = df_1min.sort_index()
            if not df_1min.index.is_unique:
                df_1min = df_1min[~df_1min.index.duplicated(keep='first')]
            logger.debug(f"{pair_log_prefix} Timestamp index prepared. Shape after prep: {df_1min.shape}")
        except Exception as e_idx_prep:
            logger.error(f"{pair_log_prefix} Error preparing timestamp index for {cleaned_1min_file.name}: {e_idx_prep}", exc_info=True)
            continue
            
        if df_1min.empty: # Vérifier à nouveau après dropna et déduplication
            logger.warning(f"{pair_log_prefix} DataFrame became empty after timestamp preparation. Skipping.")
            continue

        # final_df_for_pair commence avec les données 1-min (avec 'timestamp' comme colonne)
        final_df_for_pair = df_1min.reset_index(drop=True).copy()
        # S'assurer que les colonnes OHLCV de base sont présentes
        base_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in final_df_for_pair.columns for col in base_ohlcv_cols):
            logger.error(f"{pair_log_prefix} Base OHLCV columns missing in df_1min after loading. Cannot proceed.")
            continue


        for tf_minutes in timeframes_to_aggregate_minutes:
            tf_log_prefix = f"{pair_log_prefix}[TF:{tf_minutes}min]"
            logger.info(f"{tf_log_prefix} Aggregating K-lines...")

            # Pour resample, l'index doit être DatetimeIndex. df_1min a déjà 'timestamp' comme index.
            df_1min_for_agg = df_1min # Utiliser df_1min qui a déjà le bon index

            # Règles d'agrégation pour les volumes Taker et autres métadonnées
            extra_agg_rules = {
                'quote_asset_volume': 'sum',
                'number_of_trades': 'sum',
                'taker_buy_base_asset_volume': 'sum',
                'taker_sell_base_asset_volume': 'sum',
                'taker_buy_quote_asset_volume': 'sum',
                'taker_sell_quote_asset_volume': 'sum'
            }
            # S'assurer que seules les colonnes existantes sont dans les règles
            actual_extra_agg_rules = {k: v for k, v in extra_agg_rules.items() if k in df_1min_for_agg.columns}

            df_aggregated = data_utils.aggregate_klines_to_dataframe(
                df_1min_for_agg, tf_minutes, extra_agg_rules=actual_extra_agg_rules
            )

            if df_aggregated.empty:
                logger.warning(f"{tf_log_prefix} Aggregation resulted in an empty DataFrame. Skipping ATR calculation for this timeframe.")
                continue
            
            logger.info(f"{tf_log_prefix} Calculating ATRs (Periods: {ATR_CONFIG['atr_low']}-{ATR_CONFIG['atr_high']}, Step: {ATR_CONFIG['atr_step']})...")
            df_aggregated_with_atr = data_utils.calculate_atr_for_dataframe(
                df_aggregated,
                atr_low=ATR_CONFIG['atr_low'],
                atr_high=ATR_CONFIG['atr_high'],
                atr_step=ATR_CONFIG['atr_step']
            )

            # Renommage des colonnes agrégées et d'ATR
            # Utiliser le label de fréquence original de la config si possible, sinon générer à partir de tf_minutes
            # data_utils.get_kline_prefix_effective attend une string comme "5min", "1h"
            # Nous avons tf_minutes (int). Il faut le reconvertir en string de fréquence.
            freq_label_str = _get_frequency_label_from_minutes(tf_minutes) # ex: 60 -> "60min" ou "1h"
            kline_prefix = data_utils.get_kline_prefix_effective(freq_label_str) # ex: "60min" -> "Klines_60min"
            
            renamed_columns = {}
            for col in df_aggregated_with_atr.columns:
                if col.startswith('ATR_'):
                    renamed_columns[col] = f"{kline_prefix}_{col}" # Ex: Klines_5min_ATR_14
                else: # OHLCV et autres colonnes agrégées
                    renamed_columns[col] = f"{kline_prefix}_{col}" # Ex: Klines_5min_open
            
            df_aggregated_with_atr.rename(columns=renamed_columns, inplace=True)
            logger.debug(f"{tf_log_prefix} Colonnes renommées: {list(df_aggregated_with_atr.columns)}")

            # Fusionner avec final_df_for_pair
            # L'index de df_aggregated_with_atr est le timestamp de fin de chaque barre agrégée.
            # L'index de final_df_for_pair est un RangeIndex à ce stade (après reset_index).
            # On doit fusionner sur la colonne 'timestamp'.
            if not df_aggregated_with_atr.empty:
                df_aggregated_with_atr_ts_col = df_aggregated_with_atr.reset_index() # 'timestamp' devient une colonne
                final_df_for_pair = pd.merge(final_df_for_pair, df_aggregated_with_atr_ts_col, on='timestamp', how='left', suffixes=('', f'_{freq_label_str}_dup'))
                # Remplir les valeurs NaN introduites par le merge (pour les lignes 1-min qui ne correspondent pas à une fin de barre agrégée)
                # avec la dernière valeur valide de la colonne agrégée/ATR.
                cols_to_ffill = [col for col in final_df_for_pair.columns if col.startswith("Klines_")]
                if cols_to_ffill:
                    final_df_for_pair[cols_to_ffill] = final_df_for_pair[cols_to_ffill].ffill()
                logger.debug(f"{tf_log_prefix} Données agrégées et ATR fusionnées. Shape final_df_for_pair: {final_df_for_pair.shape}")

        # Sauvegarde du fichier enrichi pour la paire
        output_file_enriched = enriched_data_dir / f"{pair_symbol}_enriched.parquet"
        try:
            # S'assurer que 'timestamp' est bien la première colonne si elle existe
            if 'timestamp' in final_df_for_pair.columns:
                cols = ['timestamp'] + [col for col in final_df_for_pair.columns if col != 'timestamp']
                final_df_for_pair = final_df_for_pair[cols]
            
            final_df_for_pair.to_parquet(output_file_enriched, index=False, engine='pyarrow')
            logger.info(f"{pair_log_prefix} Enriched data saved to {output_file_enriched} (Shape: {final_df_for_pair.shape})")
        except Exception as e_save_enriched:
            logger.error(f"{pair_log_prefix} Error saving enriched Parquet file {output_file_enriched}: {e_save_enriched}", exc_info=True)

    script_end_time = time.time()
    logger.info(f"--- Data Fetch, Clean, and Enrichment Script Finished ---")
    logger.info(f"Total execution time: {script_end_time - script_start_time:.2f} seconds")

if __name__ == "__main__":
    # Le logging est configuré par load_all_configs appelé dans main()
    main()
