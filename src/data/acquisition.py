import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, List, Union, TYPE_CHECKING
from pathlib import Path # <<< IMPORT AJOUTÉ/ASSURÉ ICI

import pandas as pd
import numpy as np

# Attempt to import Binance client and exceptions
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    BINANCE_CLIENT_AVAILABLE = True
    KLINE_INTERVAL_1MINUTE = Client.KLINE_INTERVAL_1MINUTE
except ImportError:
    BINANCE_CLIENT_AVAILABLE = False
    KLINE_INTERVAL_1MINUTE = "1m" # Fallback string
    # Define dummy exceptions if binance library is not available, for type hinting and basic error handling
    class BinanceAPIException(Exception): pass
    class BinanceRequestException(Exception): pass
    class Client: # Dummy client for type hinting
        def __init__(self, api_key=None, api_secret=None, tld='com', testnet=False): pass
        def get_historical_klines(self, symbol, interval, start_str, end_str=None, limit=500): return [] # type: ignore

    logging.getLogger(__name__).warning(
        "python-binance library not found or failed to import. "
        "Actual data fetching will not work. Using dummy Client and exceptions."
    )


if TYPE_CHECKING:
    from src.config.definitions import AppConfig # Ou from src.config.loader import AppConfig

logger = logging.getLogger(__name__)

# Columns as received from the Binance API for klines
BINANCE_KLINES_COLS = [
    'kline_open_time', 'open', 'high', 'low', 'close', 'volume',
    'kline_close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

# Base OHLCV columns we want to ensure are present and correctly typed
OUTPUT_OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

# Final columns in the output files, including timestamp and Taker data
FINAL_OUTPUT_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'taker_sell_base_asset_volume', 'taker_sell_quote_asset_volume'
]


def _parse_and_clean_binance_klines(klines_data: List[List[Any]], pair_symbol: str) -> pd.DataFrame:
    """
    Parses raw Binance kline data, converts types, calculates seller taker volumes,
    and selects/orders the final columns.

    Args:
        klines_data: Raw data from the Binance API (list of lists).
        pair_symbol: The trading pair symbol (for logging).

    Returns:
        A cleaned pandas DataFrame with columns as defined in FINAL_OUTPUT_COLS.
    """
    log_prefix = f"[{pair_symbol}]"
    if not klines_data:
        logger.warning(f"{log_prefix} No kline data provided to _parse_and_clean_binance_klines.")
        return pd.DataFrame(columns=FINAL_OUTPUT_COLS)

    df = pd.DataFrame(klines_data, columns=BINANCE_KLINES_COLS)
    logger.debug(f"{log_prefix} Initial DataFrame shape from raw klines: {df.shape}")

    # Convert kline_open_time to 'timestamp' (datetime UTC)
    df['timestamp'] = pd.to_datetime(df['kline_open_time'], unit='ms', utc=True, errors='coerce')

    # Convert numeric columns
    numeric_cols_to_convert = OUTPUT_OHLCV_COLS + [
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
    ]
    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.warning(f"{log_prefix} Expected numeric column '{col}' not found in raw data. Will be NaN.")
            df[col] = np.nan

    # Calculate seller taker volumes
    if 'volume' in df.columns and 'taker_buy_base_asset_volume' in df.columns:
        df['taker_sell_base_asset_volume'] = df['volume'].sub(df['taker_buy_base_asset_volume'], fill_value=0)
    else:
        logger.warning(f"{log_prefix} Cannot calculate 'taker_sell_base_asset_volume' due to missing source columns ('volume' or 'taker_buy_base_asset_volume').")
        df['taker_sell_base_asset_volume'] = np.nan

    if 'quote_asset_volume' in df.columns and 'taker_buy_quote_asset_volume' in df.columns:
        df['taker_sell_quote_asset_volume'] = df['quote_asset_volume'].sub(df['taker_buy_quote_asset_volume'], fill_value=0)
    else:
        logger.warning(f"{log_prefix} Cannot calculate 'taker_sell_quote_asset_volume' due to missing source columns ('quote_asset_volume' or 'taker_buy_quote_asset_volume').")
        df['taker_sell_quote_asset_volume'] = np.nan

    # Ensure all FINAL_OUTPUT_COLS are present
    for col in FINAL_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan
            logger.debug(f"{log_prefix} Final output column '{col}' added with NaNs as it was not present.")

    df = df[FINAL_OUTPUT_COLS] # Select and order columns

    # Drop rows with NaN for essential columns (timestamp + OHLCV)
    essential_cols_for_dropna = ['timestamp'] + OUTPUT_OHLCV_COLS
    rows_before_dropna = len(df)
    df.dropna(subset=essential_cols_for_dropna, how='any', inplace=True)
    if rows_before_dropna > len(df):
        logger.debug(f"{log_prefix} Dropped {rows_before_dropna - len(df)} rows due to NaNs in essential columns.")

    if df.empty:
        logger.warning(f"{log_prefix} DataFrame became empty after dropping NaNs in essential columns.")
        return df

    # Sort by timestamp and remove duplicates, keeping the last occurrence
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    rows_before_drop_duplicates = len(df)
    df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    if rows_before_drop_duplicates > len(df):
        logger.debug(f"{log_prefix} Removed {rows_before_drop_duplicates - len(df)} duplicate timestamp entries, keeping last.")

    # Final check and fill for OHLCV columns if any NaNs remain (should be rare after dropna)
    if df[OUTPUT_OHLCV_COLS].isnull().any().any():
        logger.warning(f"{log_prefix} NaNs still present in OHLCV columns after initial cleaning. Applying ffill and bfill.")
        for col in OUTPUT_OHLCV_COLS: # Apply only to OHLCV
            df[col] = df[col].ffill().bfill()
        # Re-drop if any OHLCV are still NaN (e.g., if entire column was NaN)
        df.dropna(subset=OUTPUT_OHLCV_COLS, how='any', inplace=True)

    df = df.reset_index(drop=True)
    logger.debug(f"{log_prefix} Data parsed and cleaned. Final shape: {df.shape}")
    return df

def _fetch_single_pair_1min_history_and_clean(
    client: Client,
    pair: str,
    start_str: str,
    end_str: Optional[str] = None,
    asset_type: str = "MARGIN", # Default to MARGIN as per common use case
    limit: int = 1000 # Default API limit for SPOT/MARGIN
) -> Optional[pd.DataFrame]:
    """
    Fetches complete 1-minute historical klines for a single pair using pagination
    and then cleans the data.

    Args:
        client: Initialized Binance API client.
        pair: Trading pair symbol (e.g., "BTCUSDT").
        start_str: Start date string (e.g., "1 Jan, 2020").
        end_str: Optional end date string. If None, fetches up to the current time.
        asset_type: Type of asset/market (e.g., "MARGIN", "SPOT", "FUTURES").
        limit: Number of klines to fetch per API request.

    Returns:
        A cleaned pandas DataFrame or None if fetching fails.
    """
    log_prefix = f"[{pair}][{asset_type}]"
    logger.info(f"{log_prefix} Fetching 1-min history from {start_str} to {end_str or 'now'}...")

    all_klines_raw: List[List[Any]] = []
    current_start_str = start_str
    max_retries = 5 # Increased retries for robustness
    base_retry_delay_seconds = 5 # Base delay, will be increased with exponential backoff

    while True:
        klines_batch: List[List[Any]] = []
        fetch_successful = False
        for attempt in range(max_retries):
            try:
                logger.debug(f"{log_prefix} Requesting klines (limit: {limit}) starting from: {current_start_str}, attempt {attempt + 1}/{max_retries}")
                
                klines_batch = client.get_historical_klines( # type: ignore
                    pair, KLINE_INTERVAL_1MINUTE, current_start_str, end_str=end_str, limit=limit
                )
                logger.debug(f"{log_prefix} Received {len(klines_batch)} klines in batch starting {current_start_str}.")
                fetch_successful = True
                break
            except BinanceAPIException as e:
                logger.error(f"{log_prefix} Binance API Exception (attempt {attempt + 1}/{max_retries}) for start_str {current_start_str}: {e}")
                if e.code == -1121 and "Invalid symbol" in e.message:
                    logger.error(f"{log_prefix} Symbol '{pair}' seems invalid for the connected exchange endpoint. Aborting fetch for this pair.")
                    return pd.DataFrame(columns=FINAL_OUTPUT_COLS)

                if e.status_code in [429, 418] or e.code == -1003:
                    if attempt < max_retries - 1:
                        sleep_time = base_retry_delay_seconds * (2**attempt)
                        logger.warning(f"{log_prefix} Rate limit hit (HTTP {e.status_code}, Code {e.code}). Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"{log_prefix} Max retries reached after rate limit for start_str {current_start_str}.")
                        break
                elif attempt < max_retries - 1:
                    time.sleep(base_retry_delay_seconds)
                else:
                    logger.error(f"{log_prefix} Max retries reached for API error for start_str {current_start_str}.")
                    break
            except BinanceRequestException as e:
                logger.error(f"{log_prefix} Binance Request Exception for start_str {current_start_str}: {e}. Aborting for this pair.")
                return pd.DataFrame(columns=FINAL_OUTPUT_COLS)
            except Exception as e:
                logger.error(f"{log_prefix} Unexpected error (attempt {attempt + 1}/{max_retries}) for start_str {current_start_str}: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(base_retry_delay_seconds * (2**attempt))
                else:
                    logger.error(f"{log_prefix} Max retries reached after unexpected error for start_str {current_start_str}.")
                    break

        if not fetch_successful or not klines_batch:
            logger.info(f"{log_prefix} No more klines fetched or fetch failed for start_str {current_start_str}. Ending pagination.")
            break

        all_klines_raw.extend(klines_batch)
        
        try:
            last_kline_open_time_ms = int(klines_batch[-1][0])
            next_start_time_ms = last_kline_open_time_ms + 60000
            current_start_str = str(next_start_time_ms)
        except (IndexError, TypeError, ValueError) as e_time:
            logger.error(f"{log_prefix} Error processing timestamp for next batch: {e_time}. Batch: {klines_batch[-1] if klines_batch else 'empty'}. Ending pagination.")
            break

        if end_str:
            try:
                end_dt_ms = int(pd.to_datetime(end_str, utc=True).timestamp() * 1000)
                if last_kline_open_time_ms >= end_dt_ms:
                    logger.info(f"{log_prefix} End date {end_str} reached or passed. Last kline time: {pd.to_datetime(last_kline_open_time_ms, unit='ms', utc=True)}.")
                    break
            except Exception as e_parse_end:
                 logger.error(f"{log_prefix} Could not parse end_str '{end_str}' or compare timestamps: {e_parse_end}. Pagination might continue if not handled.")
                 break

        time.sleep(0.25)

    if not all_klines_raw:
        logger.warning(f"{log_prefix} No historical 1-minute klines were retrieved in total.")
        return pd.DataFrame(columns=FINAL_OUTPUT_COLS)

    logger.info(f"{log_prefix} Successfully fetched {len(all_klines_raw)} raw 1-minute klines. Parsing and cleaning...")
    df_cleaned = _parse_and_clean_binance_klines(all_klines_raw, pair)

    if end_str and not df_cleaned.empty:
        try:
            end_dt_filter = pd.to_datetime(end_str, utc=True)
            if not pd.api.types.is_datetime64_any_dtype(df_cleaned['timestamp']):
                 df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'], utc=True, errors='coerce')
            
            original_rows = len(df_cleaned)
            df_cleaned = df_cleaned[df_cleaned['timestamp'] < end_dt_filter]
            if len(df_cleaned) < original_rows:
                logger.debug(f"{log_prefix} Filtered out {original_rows - len(df_cleaned)} rows after end_date {end_str}.")
        except Exception as e_filter_end:
            logger.error(f"{log_prefix} Could not parse end_str '{end_str}' for final filtering: {e_filter_end}.")

    logger.info(f"{log_prefix} Fetching and cleaning for 1-minute data finished. Final rows: {len(df_cleaned)}")
    return df_cleaned


def fetch_all_historical_data(config: 'AppConfig') -> str:
    """
    Orchestrates fetching 1-minute historical data for all configured pairs.
    Saves cleaned data as CSV (for audit/backup) and Parquet (for efficient use).

    Args:
        config: The application's configuration object (AppConfig).

    Returns:
        str: Path to the directory where raw (but cleaned) CSV data for this run was saved.
    """
    logger.info("--- Starting Historical Data Fetching and Cleaning Process (1-minute klines) ---")

    if not BINANCE_CLIENT_AVAILABLE:
        logger.critical("Binance client library is not available. Cannot fetch historical data.")
        raise ImportError("Binance client library not found. Please install python-binance.")

    api_key = config.api_keys.credentials.get(config.accounts_config[0].account_alias, (None,None))[0] if config.accounts_config and config.api_keys.credentials else os.getenv("BINANCE_API_KEY")
    api_secret = config.api_keys.credentials.get(config.accounts_config[0].account_alias, (None,None))[1] if config.accounts_config and config.api_keys.credentials else os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not api_secret:
        logger.error("Binance API key or secret missing. Ensure they are in .env and AccountConfig is set up.")
        raise ValueError("Binance API credentials not found.")

    try:
        client = Client(api_key, api_secret)
        logger.info("Binance API client initialized.")
        client.ping()
        logger.info("Binance API connection successful (ping).")
    except Exception as client_err:
        logger.error(f"Failed to initialize or connect Binance client: {client_err}", exc_info=True)
        raise ConnectionError(f"Binance client initialization/connection failed: {client_err}")


    pairs_to_fetch = config.data_config.assets_and_timeframes.pairs
    start_date_str = config.data_config.historical_period.start_date
    end_date_config_str = config.data_config.historical_period.end_date

    end_date_to_use_str: Optional[str]
    if end_date_config_str is None or end_date_config_str.lower() == "now":
        end_date_to_use_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"End date not specified or 'now', using current UTC time: {end_date_to_use_str}")
    else:
        end_date_to_use_str = end_date_config_str

    asset_type = config.data_config.source_details.asset_type
    max_workers = config.data_config.fetching_options.max_workers
    api_batch_limit = config.data_config.fetching_options.batch_size

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    raw_data_base_path = Path(config.global_config.paths.data_historical_raw)
    raw_run_output_dir = raw_data_base_path / run_timestamp
    
    cleaned_data_output_path = Path(config.global_config.paths.data_historical_processed_cleaned)

    try:
        raw_run_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run-specific directory for raw (cleaned) CSVs: {raw_run_output_dir}")
        cleaned_data_output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory for cleaned Parquet files: {cleaned_data_output_path}")
    except OSError as e:
        logger.error(f"Failed to create output directories: {e}")
        raise

    tasks: List[str] = [pair for pair in pairs_to_fetch if pair]
    results_dfs: Dict[str, Optional[pd.DataFrame]] = {}

    logger.info(f"Starting parallel download and cleaning for {len(tasks)} pairs using {max_workers} workers.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {
            executor.submit(
                _fetch_single_pair_1min_history_and_clean,
                client, pair, start_date_str, end_date_to_use_str, asset_type, api_batch_limit
            ): pair
            for pair in tasks
        }

        for future in as_completed(future_to_pair):
            pair_symbol = future_to_pair[future]
            try:
                result_df = future.result()
                results_dfs[pair_symbol] = result_df

                if result_df is not None and not result_df.empty:
                    raw_csv_file_name = f"{pair_symbol}_1min_raw_with_taker.csv"
                    raw_csv_file_path = raw_run_output_dir / raw_csv_file_name
                    
                    cleaned_parquet_file_name = f"{pair_symbol}_1min_cleaned_with_taker.parquet"
                    cleaned_parquet_file_path = cleaned_data_output_path / cleaned_parquet_file_name
                    
                    try:
                        result_df.to_csv(raw_csv_file_path, index=False)
                        logger.info(f"[{pair_symbol}] Cleaned 1-min data (with taker) saved as CSV to: {raw_csv_file_path}")
                        
                        result_df.to_parquet(cleaned_parquet_file_path, index=False, engine='pyarrow')
                        logger.info(f"[{pair_symbol}] Cleaned 1-min data (with taker) saved as Parquet to: {cleaned_parquet_file_path}")
                    except IOError as e_io:
                        logger.error(f"[{pair_symbol}] Failed to save data file(s): {e_io}")
                    except Exception as e_save:
                        logger.error(f"[{pair_symbol}] Unexpected error saving data file(s): {e_save}", exc_info=True)

                elif result_df is not None and result_df.empty:
                    logger.warning(f"[{pair_symbol}] No 1-minute data returned/processed. Files not saved.")
                else:
                    logger.error(f"[{pair_symbol}] Fetching and cleaning task failed. Files not saved.")
            except Exception as exc:
                logger.error(f"Task for {pair_symbol} generated an exception: {exc}", exc_info=True)
                results_dfs[pair_symbol] = None

    successful_fetches = sum(1 for df in results_dfs.values() if df is not None and not df.empty)
    failed_fetches = len(tasks) - successful_fetches
    logger.info(f"Historical 1-minute data fetching and cleaning finished. "
                f"Successful pairs: {successful_fetches}, Failed/Empty pairs: {failed_fetches}.")

    if successful_fetches == 0 and tasks:
        logger.error("CRITICAL: No historical 1-minute data could be successfully fetched and cleaned for any configured pair.")

    return str(raw_run_output_dir)


if __name__ == '__main__':
    # This basic setup is for direct script execution.
    # In a full app, logging is configured by load_all_configs.
    # This part is for testing this specific file (acquisition.py) in isolation.
    # When run_fetch_data.py calls fetch_all_historical_data, the logging
    # will have already been set up by load_all_configs in run_fetch_data.py.
    
    # The logger messages "Running src/data/acquisition.py directly for testing..." etc.
    # that you saw in the console output are from the `if __name__ == "__main__":` block
    # of `run_fetch_data.py` (artifact id: `run_fetch_data_py_fix_sys`), not from this file.
    # This `if __name__ == "__main__":` block in `acquisition.py` would only run
    # if you executed `python src/data/acquisition.py` directly.
    
    # For clarity, if this file (acquisition.py) were run directly, it would need its own
    # logging setup and a way to get an AppConfig instance.
    
    # Example of how it might be called for testing (requires a valid AppConfig instance):
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #                     handlers=[logging.StreamHandler(sys.stdout)]) # Need sys import here too for direct run
    # logger.info("Running src/data/acquisition.py directly for testing (if __name__ == '__main__')...")
    # try:
    #     # mock_config = ... create or load a minimal AppConfig for testing ...
    #     # fetch_all_historical_data(mock_config)
    #     logger.info("Test execution of acquisition.py finished (if AppConfig was provided).")
    # except Exception as e:
    #     logger.error(f"Error during direct test execution of acquisition.py: {e}", exc_info=True)
    pass
