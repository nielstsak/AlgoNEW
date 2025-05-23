import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
import re
import pandas as pd
import numpy as np
import pandas_ta as ta # type: ignore

# Tentative d'importation depuis src.data. Si ce script est dans src/data, cela pourrait être data_utils
try:
    from src.data import data_utils
except ImportError:
    # Fallback si l'importation directe échoue (par exemple, lors de tests unitaires ou exécution isolée)
    # Cela suppose que data_utils.py est dans le même répertoire ou que le PYTHONPATH est configuré.
    try:
        import data_utils # type: ignore
    except ImportError:
        logging.getLogger(__name__).critical(
            "CRITICAL: Failed to import data_utils. Ensure it's accessible via PYTHONPATH "
            "or in the same directory if running standalone."
        )
        # Définir des fonctions factices pour permettre au reste du module de se charger pour l'analyse
        # mais cela ne fonctionnera pas réellement.
        class data_utils: # type: ignore
            @staticmethod
            def get_kline_prefix_effective(freq_param_value: Optional[str]) -> str: return ""
            @staticmethod
            def calculate_taker_pressure_ratio(df, buy_col, sell_col, out_col): return df
            @staticmethod
            def calculate_taker_pressure_delta(df, buy_col, sell_col, out_col): return df

logger = logging.getLogger(__name__)

# Colonnes OHLCV de base attendues dans le fichier brut 1-minute après chargement et nettoyage initial
BASE_OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

# Colonnes Taker de base attendues dans le fichier brut 1-minute (après calcul si nécessaire)
BASE_TAKER_COLS = [
    'taker_buy_base_asset_volume', 'taker_sell_base_asset_volume',
    'taker_buy_quote_asset_volume', 'taker_sell_quote_asset_volume',
    'quote_asset_volume', 'number_of_trades' # quote_asset_volume et number_of_trades sont aussi agrégés
]


def _ensure_required_columns(df: pd.DataFrame, required_cols: List[str], df_name: str = "DataFrame") -> bool:
    """Vérifie si toutes les colonnes requises sont présentes dans le DataFrame."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"{df_name} is missing required columns: {missing_cols}. Available: {df.columns.tolist()}")
        return False
    return True

# --- Fonctions de calcul d'indicateurs (peuvent être migrées vers data_utils.py) ---
# Note: pandas_ta.ema et pandas_ta.atr sont généralement préférés pour la robustesse et les options.
# Ces versions personnalisées sont conservées pour correspondre à votre code existant.

def _calculate_ema_rolling(series: pd.Series, period: int, 
                           series_name_debug: str = "Series") -> pd.Series:
    """Calcule l'EMA en utilisant ewm.mean()."""
    log_prefix_calc = f"[_calculate_ema_rolling for {series_name_debug}({period})]"
    if not isinstance(series, pd.Series):
        logger.warning(f"{log_prefix_calc} Input is not a pandas Series. Type: {type(series)}")
        return pd.Series(np.nan, name=f"EMA_{period}", index=getattr(series, 'index', None))
    if series.empty:
        logger.debug(f"{log_prefix_calc} Input series is empty.")
        return pd.Series(np.nan, index=series.index, name=f"EMA_{period}")
    if series.isnull().all():
        logger.debug(f"{log_prefix_calc} Input series is all NaN.")
        return pd.Series(np.nan, index=series.index, name=f"EMA_{period}")
    if not isinstance(period, int) or period <= 0:
        logger.warning(f"{log_prefix_calc} Invalid period: {period}. Must be a positive integer.")
        return pd.Series(np.nan, index=series.index, name=f"EMA_{period}")
    
    # min_periods=period assure que l'EMA n'est calculée que lorsque suffisamment de données sont disponibles.
    # adjust=False est commun pour les EMA financières pour correspondre à certaines plateformes.
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def _calculate_atr_rolling(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, 
                           period: int, series_name_debug: str = "SeriesATR") -> pd.Series:
    """Calcule l'ATR."""
    log_prefix_calc = f"[_calculate_atr_rolling for {series_name_debug}({period})]"
    valid_index = getattr(close_series, 'index', None) # Default index

    if not all(isinstance(s, pd.Series) for s in [high_series, low_series, close_series]):
        logger.warning(f"{log_prefix_calc} One or more inputs are not pandas Series.")
        return pd.Series(np.nan, name=f"ATR_{period}", index=valid_index)
    if high_series.empty or low_series.empty or close_series.empty:
        logger.debug(f"{log_prefix_calc} One or more input HLC series is empty.")
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}")
    if high_series.isnull().all() or low_series.isnull().all() or close_series.isnull().all():
        logger.debug(f"{log_prefix_calc} One or more input HLC series is all NaN.")
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}")
    if not isinstance(period, int) or period <= 0:
        logger.warning(f"{log_prefix_calc} Invalid period: {period}. Must be positive integer.")
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}")

    # Utilisation de pandas_ta.atr pour une implémentation standard et robuste
    try:
        atr_series = ta.atr(high=high_series.astype(float), low=low_series.astype(float), close=close_series.astype(float), length=period, append=False)
        if isinstance(atr_series, pd.Series):
            return atr_series
        else: # Should not happen with ta.atr if inputs are valid Series
            logger.warning(f"{log_prefix_calc} pandas_ta.atr did not return a Series. Result type: {type(atr_series)}")
            return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}")
    except Exception as e:
        logger.error(f"{log_prefix_calc} Error calculating ATR using pandas_ta: {e}", exc_info=True)
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}")

# --- Fin des fonctions de calcul d'indicateurs ---

def preprocess_live_data_for_strategy(
    raw_data_path: Path,
    processed_output_path: Path,
    strategy_params: Dict[str, Any],
    strategy_name: str
) -> Optional[pd.DataFrame]:
    """
    Preprocesses raw 1-minute live data for a specific strategy by calculating required indicators.

    Args:
        raw_data_path: Path to the raw 1-minute data CSV file (e.g., PAIR_1min_live_raw.csv).
        processed_output_path: Path to save the processed DataFrame (1-min OHLCV + _strat indicators).
        strategy_params: Dictionary of optimized parameters for the strategy.
        strategy_name: Name of the strategy to determine which indicators to calculate.

    Returns:
        A pandas DataFrame containing the latest row of processed data (1-min OHLCV + indicators),
        or None if processing fails.
    """
    pair_symbol_log = raw_data_path.name.split('_')[0] # Extract pair from filename
    log_prefix = f"[{pair_symbol_log}][{strategy_name}]"
    logger.info(f"{log_prefix} Starting live data preprocessing for strategy. Output: {processed_output_path.name}")
    logger.debug(f"{log_prefix} Strategy params: {strategy_params}")
    processing_start_time = time.time()

    # 1. Load and Clean Raw 1-minute Data
    if not raw_data_path.exists() or raw_data_path.stat().st_size == 0:
        logger.warning(f"{log_prefix} Raw 1-minute data file not found or empty: {raw_data_path}.")
        return None
    
    df_1m_raw: pd.DataFrame
    try:
        df_1m_raw = pd.read_csv(raw_data_path, low_memory=False)
        if df_1m_raw.empty:
            logger.warning(f"{log_prefix} Raw data file {raw_data_path} is empty after loading.")
            return None

        if 'timestamp' not in df_1m_raw.columns:
            logger.error(f"{log_prefix} 'timestamp' column missing in {raw_data_path.name}.")
            return None
        
        df_1m_raw['timestamp'] = pd.to_datetime(df_1m_raw['timestamp'], errors='coerce', utc=True)
        df_1m_raw.dropna(subset=['timestamp'], inplace=True)
        if df_1m_raw.empty:
            logger.warning(f"{log_prefix} DataFrame empty after timestamp conversion/dropna for {raw_data_path.name}.")
            return None
        
        df_1m_raw = df_1m_raw.set_index('timestamp').sort_index()
        if not df_1m_raw.index.is_unique:
            df_1m_raw = df_1m_raw[~df_1m_raw.index.duplicated(keep='last')]
        
        # Ensure base OHLCV and Taker columns are numeric and handle NaNs
        cols_to_convert_and_clean = BASE_OHLCV_COLS + BASE_TAKER_COLS
        if not _ensure_required_columns(df_1m_raw, cols_to_convert_and_clean, f"{log_prefix} Raw Data"):
             logger.error(f"{log_prefix} Essential base OHLCV or Taker columns missing in raw data. Cannot proceed.")
             return None

        for col in cols_to_convert_and_clean:
            df_1m_raw[col] = pd.to_numeric(df_1m_raw[col], errors='coerce')
        
        # Fill NaNs in OHLCV first, then drop rows if still NaN (critical)
        df_1m_raw[BASE_OHLCV_COLS] = df_1m_raw[BASE_OHLCV_COLS].ffill().bfill()
        df_1m_raw.dropna(subset=BASE_OHLCV_COLS, inplace=True)
        
        if df_1m_raw.empty:
            logger.warning(f"{log_prefix} DataFrame 1-minute empty after OHLCV NaN cleaning.")
            return None
        
        # For Taker columns, ffill might be acceptable if some intermediate NaNs occur
        df_1m_raw[BASE_TAKER_COLS] = df_1m_raw[BASE_TAKER_COLS].ffill()


        # Initialize df_processed with base 1-min OHLCV
        df_processed = df_1m_raw[BASE_OHLCV_COLS].copy()
        logger.info(f"{log_prefix} Loaded and cleaned raw 1-min data. Shape: {df_1m_raw.shape}")

    except Exception as e_load:
        logger.error(f"{log_prefix} Error loading/cleaning raw 1-min data from {raw_data_path}: {e_load}", exc_info=True)
        return None

    # 2. Build Frequency Map and Determine Rolling Windows for K-line Simulation
    freq_map_for_rolling: Dict[str, List[str]] = {} # Maps config frequency string to list of indicator contexts
    # Example: {"5min": ["EMA", "MACD_FREQ"], "1h": ["ATR_SLTP_FREQ"]}
    
    # Populate freq_map_for_rolling based on strategy_params
    # This logic needs to scan strategy_params for all 'indicateur_frequence_XYZ' keys
    for param_key, param_value in strategy_params.items():
        if "indicateur_frequence" in param_key and isinstance(param_value, str) and param_value:
            # Extract a context name (e.g., "EMA", "MACD", "PSAR")
            context_name_match = re.search(r"indicateur_frequence_([a-zA-Z0-9_]+)", param_key)
            context_name = context_name_match.group(1).upper() if context_name_match else param_key.upper()
            
            if param_value not in freq_map_for_rolling:
                freq_map_for_rolling[param_value] = []
            if context_name not in freq_map_for_rolling[param_value]:
                freq_map_for_rolling[param_value].append(context_name)

    # Add ATR base frequency for SL/TP explicitly
    atr_freq_sl_tp_raw = strategy_params.get('atr_base_frequency_sl_tp', strategy_params.get('atr_base_frequency'))
    if isinstance(atr_freq_sl_tp_raw, str) and atr_freq_sl_tp_raw:
        if atr_freq_sl_tp_raw not in freq_map_for_rolling:
            freq_map_for_rolling[atr_freq_sl_tp_raw] = []
        if "ATR_SL_TP_FREQ" not in freq_map_for_rolling[atr_freq_sl_tp_raw]: # Unique context
            freq_map_for_rolling[atr_freq_sl_tp_raw].append("ATR_SL_TP_FREQ")
    
    logger.debug(f"{log_prefix} Frequency map for rolling aggregations: {freq_map_for_rolling}")

    rolling_windows_to_calculate: Dict[int, str] = {} # Maps window_size_minutes to config_freq_label
    for freq_str_config, contexts in freq_map_for_rolling.items():
        if freq_str_config.lower() == "1min": # Skip 1min as it's the base
            continue
        match = re.fullmatch(r"(\d+)(min|m|h|d)", freq_str_config.lower().strip())
        if match:
            num, unit = int(match.group(1)), match.group(2)
            window_minutes = 0
            if unit in ["min", "m"]: window_minutes = num
            elif unit == "h": window_minutes = num * 60
            elif unit == "d": window_minutes = num * 24 * 60
            
            if window_minutes > 0:
                if window_minutes not in rolling_windows_to_calculate:
                    rolling_windows_to_calculate[window_minutes] = freq_str_config
                # If already mapped, ensure consistency or log warning (first one wins for label)
            else:
                logger.warning(f"{log_prefix} Invalid window_minutes ({window_minutes}) for freq '{freq_str_config}'.")
        else:
            logger.warning(f"{log_prefix} Could not parse window size from freq_str: '{freq_str_config}' (contexts: {contexts})")
    
    logger.info(f"{log_prefix} Rolling windows to simulate (minutes: label): {rolling_windows_to_calculate}")

    # 3. Calculate Aggregated K-lines using Rolling Windows on 1-min data
    for window_mins, period_label_cfg in sorted(rolling_windows_to_calculate.items()):
        if window_mins <= 0: continue # Should be caught by previous logic
        logger.debug(f"{log_prefix} Simulating {period_label_cfg} K-lines (window: {window_mins} mins)...")
        
        # Ensure min_periods equals window_mins for complete K-lines
        min_p = window_mins 
        
        df_processed[f"Kline_{period_label_cfg}_open"] = df_1m_raw['open'].rolling(window=window_mins, min_periods=min_p).apply(lambda x: x[0] if len(x) >= min_p else np.nan, raw=True)
        df_processed[f"Kline_{period_label_cfg}_high"] = df_1m_raw['high'].rolling(window=window_mins, min_periods=min_p).max()
        df_processed[f"Kline_{period_label_cfg}_low"] = df_1m_raw['low'].rolling(window=window_mins, min_periods=min_p).min()
        df_processed[f"Kline_{period_label_cfg}_close"] = df_1m_raw['close'].rolling(window=window_mins, min_periods=min_p).apply(lambda x: x[-1] if len(x) >= min_p else np.nan, raw=True)
        df_processed[f"Kline_{period_label_cfg}_volume"] = df_1m_raw['volume'].rolling(window=window_mins, min_periods=min_p).sum()

        for taker_col in BASE_TAKER_COLS: # Aggregate all base taker columns
            if taker_col in df_1m_raw.columns:
                df_processed[f"Kline_{period_label_cfg}_{taker_col}"] = df_1m_raw[taker_col].rolling(window=window_mins, min_periods=min_p).sum()
            else: # Should not happen if _ensure_required_columns passed
                df_processed[f"Kline_{period_label_cfg}_{taker_col}"] = np.nan
    
    logger.debug(f"{log_prefix} df_processed columns after K-line simulation: {df_processed.columns.tolist()}")

    # 4. Calculate ATR_strat (for SL/TP)
    atr_period_sl_tp_key = 'atr_period_sl_tp' if 'atr_period_sl_tp' in strategy_params else 'atr_period'
    atr_freq_sl_tp_key = 'atr_base_frequency_sl_tp' if 'atr_base_frequency_sl_tp' in strategy_params else 'atr_base_frequency'
    
    atr_period_val = strategy_params.get(atr_period_sl_tp_key)
    atr_freq_raw_val = strategy_params.get(atr_freq_sl_tp_key)

    if atr_period_val is not None and atr_freq_raw_val is not None:
        atr_period_int = int(atr_period_val)
        # Get the effective label (e.g., "5min", "60min") for constructing column names
        # data_utils.get_kline_prefix_effective returns "Klines_5min" or "" for 1min
        # We need the "5min" part or handle "" for 1min.
        kline_prefix_for_atr_source = data_utils.get_kline_prefix_effective(str(atr_freq_raw_val))
        
        atr_high_col_src = f"{kline_prefix_for_atr_source}_high" if kline_prefix_for_atr_source else "high"
        atr_low_col_src = f"{kline_prefix_for_atr_source}_low" if kline_prefix_for_atr_source else "low"
        atr_close_col_src = f"{kline_prefix_for_atr_source}_close" if kline_prefix_for_atr_source else "close"
        
        logger.debug(f"{log_prefix} ATR_strat: Period={atr_period_int}, FreqRaw='{atr_freq_raw_val}', PrefixSrc='{kline_prefix_for_atr_source}'. Source HLC cols: {atr_high_col_src}, {atr_low_col_src}, {atr_close_col_src}")

        if all(col in df_processed.columns for col in [atr_high_col_src, atr_low_col_src, atr_close_col_src]):
            df_processed['ATR_strat'] = _calculate_atr_rolling(
                df_processed[atr_high_col_src], df_processed[atr_low_col_src], df_processed[atr_close_col_src],
                period=atr_period_int, series_name_debug=f"ATR_SLTP_Src({atr_freq_raw_val})"
            )
            if 'ATR_strat' in df_processed: logger.info(f"{log_prefix} ATR_strat calculated. NaNs: {df_processed['ATR_strat'].isnull().sum()}/{len(df_processed)}")
        else:
            logger.warning(f"{log_prefix} Source HLC columns for ATR_strat not found in df_processed (needed: {atr_high_col_src}, {atr_low_col_src}, {atr_close_col_src}). ATR_strat will be NaN.")
            df_processed['ATR_strat'] = np.nan
    else:
        logger.warning(f"{log_prefix} ATR_strat parameters ('{atr_period_sl_tp_key}' or '{atr_freq_sl_tp_key}') missing. ATR_strat will be NaN.")
        df_processed['ATR_strat'] = np.nan

    # 5. Calculate Strategy-Specific Indicators
    # This section needs to be customized for each strategy, similar to ObjectiveEvaluator
    # For brevity, only a placeholder structure is shown.
    # The key is to use data_utils.get_kline_prefix_effective and select the correct
    # Kline_{TF}_<ohlcv> columns from df_processed as input to ta functions.

    strategy_name_lower = strategy_name.lower()
    # Example for a strategy that uses EMA
    if "ema" in strategy_name_lower: # Generic check, refine for actual strategy names
        ema_short_p = strategy_params.get('ema_short_period')
        ema_freq_raw = strategy_params.get('indicateur_frequence_ema') # Example param name
        if ema_short_p is not None and ema_freq_raw is not None:
            kline_prefix_ema_src = data_utils.get_kline_prefix_effective(str(ema_freq_raw))
            ema_close_col_src = f"{kline_prefix_ema_src}_close" if kline_prefix_ema_src else "close"
            if ema_close_col_src in df_processed.columns:
                df_processed['EMA_short_strat'] = _calculate_ema_rolling(df_processed[ema_close_col_src], period=int(ema_short_p), series_name_debug=f"EMA_short_Src({ema_freq_raw})")
            else:
                df_processed['EMA_short_strat'] = np.nan
                logger.warning(f"{log_prefix} Source close column '{ema_close_col_src}' for EMA_short_strat not found.")
    
    # Example for MACD
    if "macd" in strategy_name_lower:
        macd_fast = strategy_params.get('macd_fast_period')
        macd_slow = strategy_params.get('macd_slow_period')
        macd_signal = strategy_params.get('macd_signal_period')
        macd_freq_raw = strategy_params.get('indicateur_frequence_macd')
        if all(p is not None for p in [macd_fast, macd_slow, macd_signal, macd_freq_raw]):
            kline_prefix_macd_src = data_utils.get_kline_prefix_effective(str(macd_freq_raw))
            macd_close_col_src = f"{kline_prefix_macd_src}_close" if kline_prefix_macd_src else "close"
            if macd_close_col_src in df_processed.columns:
                macd_df = ta.macd(close=df_processed[macd_close_col_src].astype(float), 
                                  fast=int(macd_fast), slow=int(macd_slow), signal=int(macd_signal), 
                                  append=False)
                if macd_df is not None and not macd_df.empty:
                    # Column names from pandas_ta.macd are typically like MACD_F_S_signal, MACDh_F_S_signal, MACDs_F_S_signal
                    df_processed['MACD_line_strat'] = macd_df.iloc[:,0] # MACD Line
                    df_processed['MACD_hist_strat'] = macd_df.iloc[:,1] # MACD Histogram
                    df_processed['MACD_signal_strat'] = macd_df.iloc[:,2] # MACD Signal Line
                else:
                    df_processed['MACD_line_strat'] = np.nan; df_processed['MACD_hist_strat'] = np.nan; df_processed['MACD_signal_strat'] = np.nan
            else:
                logger.warning(f"{log_prefix} Source close column '{macd_close_col_src}' for MACD not found.")
                df_processed['MACD_line_strat'] = np.nan; df_processed['MACD_hist_strat'] = np.nan; df_processed['MACD_signal_strat'] = np.nan


    # 6. Calculate Taker Pressure Indicators (if applicable)
    taker_ma_period_param = strategy_params.get('taker_pressure_indicator_period')
    taker_freq_raw_param = strategy_params.get('indicateur_frequence_taker_pressure')
    if taker_ma_period_param is not None and taker_freq_raw_param is not None:
        taker_ma_period = int(taker_ma_period_param)
        kline_prefix_taker_src = data_utils.get_kline_prefix_effective(str(taker_freq_raw_param))
        
        buy_vol_col_name = f"{kline_prefix_taker_src}_taker_buy_base_asset_volume" if kline_prefix_taker_src else "taker_buy_base_asset_volume"
        sell_vol_col_name = f"{kline_prefix_taker_src}_taker_sell_base_asset_volume" if kline_prefix_taker_src else "taker_sell_base_asset_volume"

        if buy_vol_col_name in df_processed.columns and sell_vol_col_name in df_processed.columns:
            # Use the utility from data_utils
            df_processed = data_utils.calculate_taker_pressure_ratio(
                df_processed, buy_vol_col_name, sell_vol_col_name, "TakerPressureRatio_Raw_Temp"
            )
            if "TakerPressureRatio_Raw_Temp" in df_processed.columns and df_processed["TakerPressureRatio_Raw_Temp"].notna().any():
                df_processed["TakerPressureRatio_MA_strat"] = _calculate_ema_rolling(
                    df_processed["TakerPressureRatio_Raw_Temp"], period=taker_ma_period, 
                    series_name_debug=f"TakerRatioMARaw_Src({taker_freq_raw_param})"
                )
                df_processed.drop(columns=["TakerPressureRatio_Raw_Temp"], inplace=True, errors='ignore')
            else:
                df_processed["TakerPressureRatio_MA_strat"] = np.nan
        else:
            logger.warning(f"{log_prefix} Source Taker columns ('{buy_vol_col_name}', '{sell_vol_col_name}') not found in df_processed. TakerPressureRatio_MA_strat will be NaN.")
            df_processed["TakerPressureRatio_MA_strat"] = np.nan

    # 7. Finalization and Save
    strat_cols = [col for col in df_processed.columns if col.endswith('_strat')]
    if strat_cols:
        df_processed[strat_cols] = df_processed[strat_cols].ffill()
        logger.debug(f"{log_prefix} Applied ffill to _strat columns: {strat_cols}")

    df_final_to_save = df_processed.reset_index() # timestamp becomes a column

    # Define desired column order
    ordered_cols = ['timestamp'] + BASE_OHLCV_COLS
    # Add aggregated Kline columns, sorted by timeframe then type
    for window_mins, period_label_cfg in sorted(rolling_windows_to_calculate.items()):
        for ohlcv_part in BASE_OHLCV_COLS:
            ordered_cols.append(f"Kline_{period_label_cfg}_{ohlcv_part}")
        for taker_part in BASE_TAKER_COLS:
             ordered_cols.append(f"Kline_{period_label_cfg}_{taker_part}")
    # Add _strat columns
    ordered_cols.extend(sorted(strat_cols))
    
    # Reindex, adding missing columns with NaN if any (should not happen if logic is correct)
    current_cols_set = set(df_final_to_save.columns)
    final_ordered_cols_present = [col for col in ordered_cols if col in current_cols_set]
    # Add any other columns that might have been created but are not in the defined order (should be rare)
    other_cols = [col for col in df_final_to_save.columns if col not in final_ordered_cols_present]
    df_final_to_save = df_final_to_save[final_ordered_cols_present + other_cols]

    try:
        processed_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final_to_save.to_csv(processed_output_path, index=False, float_format='%.8f')
        processing_time = time.time() - processing_start_time
        logger.info(f"{log_prefix} Preprocessed live data saved to: {processed_output_path} "
                    f"(Shape: {df_final_to_save.shape}, Time: {processing_time:.3f}s)")
    except Exception as e_save:
        logger.error(f"{log_prefix} Error saving processed live data to {processed_output_path}: {e_save}", exc_info=True)
        return None # Return None if save fails

    # Return the latest row of the processed data
    return df_final_to_save if not df_final_to_save.empty else None

