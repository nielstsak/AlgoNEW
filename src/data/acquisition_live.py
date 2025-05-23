import json
import logging
import os # Maintenu pour la compatibilité potentielle, bien que pathlib soit préféré
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING

import pandas as pd
import numpy as np
import requests # Pour les appels REST directs

if TYPE_CHECKING:
    # Utilisation de AppConfig pour une configuration plus complète si disponible
    from src.config.definitions import AppConfig, LiveFetchConfig, PathsConfig, GlobalLiveSettings
    # Autrement, LiveConfig pourrait être un sous-ensemble ou un dict
    # from src.config.loader import LiveConfig

# KLINE_INTERVAL_1MINUTE peut être défini ici ou importé depuis binance.client
try:
    from binance.client import Client as BinanceClient # Uniquement pour la constante
    KLINE_INTERVAL_1MINUTE = BinanceClient.KLINE_INTERVAL_1MINUTE
    BINANCE_LIB_AVAILABLE = True
except ImportError:
    KLINE_INTERVAL_1MINUTE = "1m" # Fallback si python-binance n'est pas installé
    BINANCE_LIB_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "python-binance library not found. KLINE_INTERVAL_1MINUTE set to '1m'. "
        "Actual API calls in other modules might fail if they depend on the library."
    )

logger = logging.getLogger(__name__)

# Colonnes telles que reçues de l'API Binance pour les klines
BINANCE_KLINE_COLUMNS = [
    'kline_open_time', 'open', 'high', 'low', 'close', 'volume',
    'kline_close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

# Colonnes finales dans les fichiers de sortie, incluant le timestamp et les données Taker
# Ce sont les colonnes standardisées pour les fichiers bruts (qu'ils soient historiques ou live)
BASE_COLUMNS_RAW = [
    'timestamp', 'kline_close_time', 'open', 'high', 'low', 'close', 'volume',
    'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'taker_sell_base_asset_volume', 'taker_sell_quote_asset_volume'
]

# Limites de l'API Binance pour les klines par requête
API_BATCH_LIMIT_SPOT_MARGIN = 1000
API_BATCH_LIMIT_FUTURES = 1500


def get_binance_klines_rest(
    symbol: str,
    # config_interval_context: str, # Moins pertinent ici car on fetch toujours du 1m pour le brut live
    limit: int = 100,
    account_type: str = "MARGIN", # SPOT, MARGIN, FUTURES
    end_timestamp_ms: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Récupère les klines 1-minute via l'API REST de Binance.

    Args:
        symbol: Symbole de la paire (ex: BTCUSDT).
        limit: Nombre de klines à récupérer.
        account_type: Type de compte/marché ("SPOT", "MARGIN", "FUTURES").
        end_timestamp_ms: Timestamp de fin (en ms) pour récupérer les klines *avant* ce temps.
                          Si None, récupère les plus récentes.

    Returns:
        DataFrame pandas avec les klines ou None en cas d'erreur.
    """
    actual_fetch_interval = KLINE_INTERVAL_1MINUTE
    log_ctx = f"[{symbol}][{actual_fetch_interval}][{account_type.upper()}]"

    if end_timestamp_ms:
        logger.debug(f"{log_ctx} Requesting {limit} klines ending before {pd.to_datetime(end_timestamp_ms, unit='ms', utc=True)} via REST API")
    else:
        logger.debug(f"{log_ctx} Requesting latest {limit} klines via REST API")

    account_type_upper = account_type.upper()
    api_batch_limit: int

    if account_type_upper in ["SPOT", "MARGIN", "BINANCE_MARGIN"]: # BINANCE_MARGIN pour rétrocompatibilité
        base_url = "https://api.binance.com"
        endpoint = "/api/v3/klines"
        api_batch_limit = API_BATCH_LIMIT_SPOT_MARGIN
    elif account_type_upper == "FUTURES": # USD-M Futures
        base_url = "https://fapi.binance.com"
        endpoint = "/fapi/v1/klines"
        api_batch_limit = API_BATCH_LIMIT_FUTURES
    # TODO: Ajouter un cas pour COIN-M Futures (https://dapi.binance.com, /dapi/v1/klines) si nécessaire
    else:
        logger.error(f"{log_ctx} Unsupported account_type for kline fetching: {account_type}")
        return None

    current_fetch_limit = min(limit, api_batch_limit)
    url = base_url + endpoint
    params: Dict[str, Any] = {"symbol": symbol.upper(), "interval": actual_fetch_interval, "limit": current_fetch_limit}
    if end_timestamp_ms:
        params["endTime"] = end_timestamp_ms

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            logger.warning(f"{log_ctx} No kline data received from Binance API.")
            return pd.DataFrame(columns=BASE_COLUMNS_RAW) # Retourner un DF vide avec les bonnes colonnes

        df = pd.DataFrame(data, columns=BINANCE_KLINE_COLUMNS)
        df.rename(columns={'kline_open_time': 'timestamp'}, inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df['kline_close_time'] = pd.to_datetime(df['kline_close_time'], unit='ms', utc=True, errors='coerce')

        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                        'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = np.nan # Assurer que la colonne existe

        # Calcul des volumes Taker vendeurs
        if 'volume' in df.columns and 'taker_buy_base_asset_volume' in df.columns:
            df['taker_sell_base_asset_volume'] = df['volume'].sub(df['taker_buy_base_asset_volume'], fill_value=0)
        else:
            df['taker_sell_base_asset_volume'] = np.nan
        if 'quote_asset_volume' in df.columns and 'taker_buy_quote_asset_volume' in df.columns:
            df['taker_sell_quote_asset_volume'] = df['quote_asset_volume'].sub(df['taker_buy_quote_asset_volume'], fill_value=0)
        else:
            df['taker_sell_quote_asset_volume'] = np.nan

        # S'assurer que toutes les colonnes de BASE_COLUMNS_RAW sont présentes
        for col in BASE_COLUMNS_RAW:
            if col not in df.columns:
                df[col] = np.nan
        
        df = df[BASE_COLUMNS_RAW] # Sélectionner et ordonner

        essential_cols_for_dropna = ['timestamp', 'open', 'high', 'low', 'close', 'kline_close_time']
        df.dropna(subset=essential_cols_for_dropna, how='any', inplace=True)
        
        logger.info(f"{log_ctx} Fetched and processed {len(df)} klines via REST API.")
        return df

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"{log_ctx} HTTP error fetching klines: {http_err}. Response: {http_err.response.text if http_err.response else 'No response body'}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"{log_ctx} Request error fetching klines: {req_err}")
    except json.JSONDecodeError as json_err:
        logger.error(f"{log_ctx} JSON decode error processing klines response: {json_err}")
    except Exception as e:
        logger.error(f"{log_ctx} Unexpected error fetching/processing klines: {e}", exc_info=True)
    return None


def initialize_pair_data(
    pair: str,
    raw_path: Path, # Chemin vers le fichier {PAIR}_1min_live_raw.csv
    total_klines_to_fetch: int,
    account_type: str
):
    """
    Initializes the raw 1-minute data file for a pair, fetching historical data in batches if needed.
    The `config_interval_context` is implicitly "1min" as this function deals with the base 1-min raw data.
    """
    log_ctx = f"[{pair}][1min][{account_type.upper()}]" # Contexte pour le logging
    logger.info(f"{log_ctx} Initializing live raw data file: {raw_path}")

    try:
        df_existing = pd.DataFrame(columns=BASE_COLUMNS_RAW)
        if raw_path.exists() and raw_path.stat().st_size > 0:
            try:
                df_existing = pd.read_csv(raw_path)
                # Convert timestamp column immediately after loading
                if 'timestamp' in df_existing.columns:
                    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'], utc=True, errors='coerce')
                    df_existing.dropna(subset=['timestamp'], inplace=True) # Remove rows where timestamp conversion failed
                else: # If no timestamp column, treat as empty for safety
                    logger.warning(f"{log_ctx} Existing file {raw_path} lacks 'timestamp' column. Treating as empty.")
                    df_existing = pd.DataFrame(columns=BASE_COLUMNS_RAW)

            except pd.errors.EmptyDataError:
                logger.warning(f"{log_ctx} Raw file {raw_path} is empty. Will fetch full history.")
            except Exception as e_read:
                logger.error(f"{log_ctx} Error reading existing raw file {raw_path}: {e_read}. Attempting to re-fetch history.", exc_info=True)
        else:
            logger.info(f"{log_ctx} Raw file {raw_path} not found or empty. Will fetch history.")
            # Ensure the file exists with headers if it's completely new
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=BASE_COLUMNS_RAW).to_csv(raw_path, index=False)


        num_lines_existing = len(df_existing)
        klines_needed = total_klines_to_fetch - num_lines_existing
        logger.info(f"{log_ctx} Existing rows: {num_lines_existing}. Target total: {total_klines_to_fetch}. Klines needed: {klines_needed}")

        if klines_needed > 0:
            logger.info(f"{log_ctx} Fetching {klines_needed} historical 1-minute klines to complete initial history...")
            all_new_klines_list: List[pd.DataFrame] = []
            current_end_timestamp_ms: Optional[int] = None # Start with most recent and go backwards

            # Determine API batch limit
            api_batch_limit = API_BATCH_LIMIT_SPOT_MARGIN if account_type.upper() in ["SPOT", "MARGIN"] else API_BATCH_LIMIT_FUTURES

            retries_for_full_history = 3
            fetch_attempt = 0

            while klines_needed > 0 and fetch_attempt < retries_for_full_history :
                fetch_this_batch = min(klines_needed, api_batch_limit)
                logger.info(f"{log_ctx} Fetching batch of {fetch_this_batch} klines (remaining needed: {klines_needed}). EndTime ms for API: {current_end_timestamp_ms}")

                df_batch = get_binance_klines_rest(
                    symbol=pair,
                    limit=fetch_this_batch,
                    account_type=account_type,
                    end_timestamp_ms=current_end_timestamp_ms
                )

                if df_batch is not None and not df_batch.empty:
                    all_new_klines_list.append(df_batch)
                    klines_needed -= len(df_batch)
                    
                    # Prepare for next older batch: oldest timestamp from current batch - 1ms
                    # Ensure 'timestamp' is datetime before trying to access .iloc[0]
                    if not pd.api.types.is_datetime64_any_dtype(df_batch['timestamp']):
                        df_batch['timestamp'] = pd.to_datetime(df_batch['timestamp'], utc=True, errors='coerce')
                        df_batch.dropna(subset=['timestamp'], inplace=True) # Reclean if conversion failed for some
                    
                    if df_batch.empty: # If all rows dropped after timestamp conversion
                        logger.warning(f"{log_ctx} Batch became empty after timestamp conversion/dropna. Stopping history fetch for this cycle.")
                        break


                    oldest_ts_in_batch: pd.Timestamp = df_batch['timestamp'].iloc[0]
                    current_end_timestamp_ms = int(oldest_ts_in_batch.timestamp() * 1000) -1 # Go back from this point

                    if len(df_batch) < fetch_this_batch:
                        logger.info(f"{log_ctx} API returned fewer klines ({len(df_batch)}) than requested ({fetch_this_batch}). Assuming end of available history.")
                        break
                    time.sleep(0.3) # API rate limit respect
                else:
                    logger.warning(f"{log_ctx} Failed to fetch a batch or batch empty. Stopping history fetch for this cycle.")
                    fetch_attempt +=1
                    if fetch_attempt < retries_for_full_history:
                        logger.info(f"{log_ctx} Retrying history fetch in 5 seconds (attempt {fetch_attempt}/{retries_for_full_history})")
                        time.sleep(5)
                    else:
                        logger.error(f"{log_ctx} Max retries reached for fetching historical batches.")
                        break
            
            if all_new_klines_list:
                df_new_total = pd.concat(all_new_klines_list, ignore_index=True)
                
                # Combine with existing, sort, and remove duplicates
                df_combined = pd.concat([df_existing, df_new_total], ignore_index=True)
                if 'timestamp' in df_combined.columns: # Should always be true
                    if not pd.api.types.is_datetime64_any_dtype(df_combined['timestamp']):
                        df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'], utc=True, errors='coerce')
                    df_combined.dropna(subset=['timestamp'], inplace=True)
                    df_combined.sort_values(by='timestamp', ascending=True, inplace=True)
                    df_combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                
                # Ensure all base columns are present before saving
                for col in BASE_COLUMNS_RAW:
                    if col not in df_combined.columns:
                        df_combined[col] = np.nan
                
                df_combined[BASE_COLUMNS_RAW].to_csv(raw_path, index=False)
                logger.info(f"{log_ctx} Saved initial/updated {len(df_combined)} rows to {raw_path}")
            elif num_lines_existing == 0: # Fetch failed and file was initially empty
                logger.warning(f"{log_ctx} Initial history fetch failed and file was empty. {raw_path} remains empty (with headers).")
                # File with headers was already created if it didn't exist.
            else: # Fetch failed but there was existing data
                 logger.warning(f"{log_ctx} Initial history fetch failed. Existing {num_lines_existing} rows in {raw_path} are preserved.")
        else:
            logger.info(f"{log_ctx} Sufficient data ({num_lines_existing} rows >= target {total_klines_to_fetch}) already present in {raw_path}.")

    except Exception as e:
        logger.error(f"{log_ctx} Error during live data initialization for {raw_path}: {e}", exc_info=True)


def run_initialization(app_config_obj_or_dict: Union['AppConfig', Dict[str, Any]]):
    """
    Orchestrates the initialization of raw 1-minute live data files for all configured pairs.
    """
    logger.info("--- Running Live Data Initialization (REST API for 1-minute history) ---")

    # Extract necessary configs
    # This part needs to be robust whether a full AppConfig object or a dict (e.g. from JSON) is passed
    if hasattr(app_config_obj_or_dict, 'live_config') and hasattr(app_config_obj_or_dict, 'global_config'): # Likely AppConfig object
        live_fetch_config = app_config_obj_or_dict.live_config.live_fetch # type: ignore
        global_settings = app_config_obj_or_dict.live_config.global_live_settings # type: ignore
        paths_config = app_config_obj_or_dict.global_config.paths # type: ignore
    elif isinstance(app_config_obj_or_dict, dict): # If it's a dict (e.g. loaded from JSON directly)
        live_fetch_config_dict = app_config_obj_or_dict.get('live_fetch', {})
        global_settings_dict = app_config_obj_or_dict.get('global_live_settings', {})
        # Need to access paths from a nested global_config if AppConfig structure is mirrored in dict
        paths_config_dict = app_config_obj_or_dict.get('global_config', {}).get('paths', {})
        
        # Create simple objects or access dict keys for attributes
        class SimpleLiveFetch:
            crypto_pairs = live_fetch_config_dict.get('crypto_pairs', [])
            limit_init_history = live_fetch_config_dict.get('limit_init_history', 1000)
        live_fetch_config = SimpleLiveFetch() # type: ignore

        class SimpleGlobalSettings:
            account_type = global_settings_dict.get('account_type', 'MARGIN')
        global_settings = SimpleGlobalSettings() # type: ignore

        class SimplePaths:
            data_live_raw = paths_config_dict.get('data_live_raw', 'data/live/raw') # Default path
        paths_config = SimplePaths() # type: ignore
    else:
        logger.critical("Invalid configuration object passed to run_initialization.")
        return

    pairs = getattr(live_fetch_config, 'crypto_pairs', [])
    total_klines_to_fetch_init = getattr(live_fetch_config, 'limit_init_history', 1000)
    account_type = getattr(global_settings, 'account_type', 'MARGIN')
    
    # Determine project root to correctly resolve data_live_raw path
    # This assumes that if AppConfig is passed, its project_root is set.
    # If a dict is passed, we need a way to get project_root.
    project_root_path: Path
    if hasattr(app_config_obj_or_dict, 'project_root'):
        project_root_path = Path(app_config_obj_or_dict.project_root) # type: ignore
    else: # Fallback if project_root is not in the passed config (e.g. simple dict)
        try:
            project_root_path = Path(__file__).resolve().parent.parent.parent # Assuming src/data/acquisition_live.py
        except NameError:
            project_root_path = Path(".").resolve() # Current working directory
        logger.warning(f"project_root not found in config, using detected: {project_root_path}")


    raw_dir_str = getattr(paths_config, 'data_live_raw', 'data/live/raw') # Default if not in paths_config
    raw_dir = project_root_path / raw_dir_str # Resolve raw_dir relative to project_root

    if not pairs:
        logger.error("No crypto_pairs defined in live_fetch configuration. Cannot initialize.")
        return

    for pair in pairs:
        if not pair or not isinstance(pair, str):
            logger.warning(f"Invalid pair found in configuration: {pair}. Skipping.")
            continue
            
        # Standardized raw live data filename
        raw_file_name = f"{pair.upper()}_1min_live_raw.csv"
        raw_path_for_pair = raw_dir / raw_file_name
        
        logger.info(f"Initializing 1-minute live raw data for {pair.upper()} into {raw_path_for_pair}...")
        try:
            # The 'config_interval_context' for initialize_pair_data is implicitly "1min"
            # as it deals with the fundamental 1-minute raw data file.
            initialize_pair_data(
                pair=pair.upper(),
                raw_path=raw_path_for_pair,
                total_klines_to_fetch=total_klines_to_fetch_init,
                account_type=account_type
            )
        except Exception as e:
            logger.error(f"Failed to initialize live raw data for {pair.upper()} (file: {raw_path_for_pair}): {e}", exc_info=True)

    logger.info("--- Live Data Initialization (REST fetch) Complete ---")


if __name__ == '__main__':
    # Example for direct testing of this module.
    # In a real application, run_fetch_data_live.py would call run_initialization.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
    logger.info("Running acquisition_live.py directly for testing (REST API only)...")

    # Create a mock configuration dictionary similar to what AppConfig would provide
    # This requires paths to be relative to where this script might be run from if not project root.
    # For robust testing, ensure paths are correctly set or use absolute paths.
    
    # Determine a base path for test data (e.g., in a temporary directory or a test_data subdir)
    try:
        _test_project_root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        _test_project_root = Path(".").resolve()

    _test_data_live_raw_path = _test_project_root / "test_temp_data" / "live" / "raw"
    _test_data_live_raw_path.mkdir(parents=True, exist_ok=True)

    mock_app_config_dict = {
        "global_config": {
            "paths": {
                "data_live_raw": str(_test_data_live_raw_path.relative_to(_test_project_root)) # Path relative to project root
            },
            # Other global_config fields if needed by other parts, but not directly by run_initialization
        },
        "live_config": {
            "live_fetch": {
                "crypto_pairs": ["BTCUSDT", "ETHUSDT"], # Example pairs
                "intervals": ["1m"], # This is mainly for context in other parts, fetch is always 1m
                "limit_init_history": 500, # Smaller number for quick test
                # "limit_per_fetch": 100, # Not used by run_initialization directly
                # "max_retries": 3,
                # "retry_backoff": 1.5
            },
            "global_live_settings": {
                "account_type": "SPOT" # Or MARGIN, FUTURES
            }
        },
        "project_root": str(_test_project_root) # Crucial for resolving relative paths
        # Other AppConfig fields like data_config, strategies_config, api_keys might be needed
        # if the functions called by run_initialization depend on them.
        # For this specific test of acquisition_live, these might be minimal.
    }

    logger.info(f"Using test project root: {_test_project_root}")
    logger.info(f"Test live raw data will be stored in: {_test_data_live_raw_path}")

    try:
        run_initialization(mock_app_config_dict) # type: ignore
        logger.info("Test execution of run_initialization finished.")
    except ImportError as e_imp:
         logger.critical(f"Missing library for test: {e_imp}. Please install required packages (e.g., python-binance, requests).")
    except Exception as main_err:
         logger.critical(f"Critical error in test main execution: {main_err}", exc_info=True)

