import logging
import time
import importlib
from typing import Any, Dict, Optional, Tuple, List, Type, Union
from datetime import timezone
import uuid

import numpy as np
import pandas as pd
import pandas_ta as ta # type: ignore
import optuna

try:
    from src.backtesting.simulator import BacktestSimulator
    from src.data import data_utils # Pour get_kline_prefix_effective
    from src.strategies.base import BaseStrategy
    from src.config.definitions import ParamDetail, AppConfig # Importer AppConfig
except ImportError as e:
    # Fallback logger if main logging isn't set up yet
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger(__name__).critical(f"ObjectiveEvaluator: Critical import error: {e}. Ensure PYTHONPATH is correct.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class ObjectiveEvaluator:
    def __init__(self,
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any],
                 df_enriched_slice: pd.DataFrame, # Renamed from data_1min_cleaned_slice
                 simulation_settings: Dict[str, Any],
                 optuna_objectives_config: Dict[str, Any],
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any],
                 app_config: AppConfig, # Added app_config
                 is_oos_eval: bool = False,
                 is_trial_number_for_oos_log: Optional[int] = None):

        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict
        self.df_enriched_slice = df_enriched_slice.copy() # Source unique de données
        self.simulation_settings_global_defaults = simulation_settings
        self.optuna_objectives_config = optuna_objectives_config
        self.pair_symbol = pair_symbol
        self.symbol_info_data = symbol_info_data
        self.is_oos_eval = is_oos_eval
        self.app_config = app_config # Store app_config
        self.is_trial_number_for_oos_log = is_trial_number_for_oos_log

        self.strategy_script_ref = self.strategy_config_dict.get('script_reference')
        self.strategy_class_name = self.strategy_config_dict.get('class_name')

        if not self.strategy_script_ref or not self.strategy_class_name:
            raise ValueError("ObjectiveEvaluator: script_reference or class_name missing in strategy_config_dict")

        self.params_space_details: Dict[str, ParamDetail] = {}
        raw_params_space = self.strategy_config_dict.get('params_space', {})
        if isinstance(raw_params_space, dict):
            for param_key, param_value_obj in raw_params_space.items():
                if isinstance(param_value_obj, ParamDetail):
                    self.params_space_details[param_key] = param_value_obj
                elif isinstance(param_value_obj, dict):
                    try:
                        self.params_space_details[param_key] = ParamDetail(**param_value_obj)
                    except Exception as e_pd_init:
                        logger.error(f"Error creating ParamDetail for {param_key} from dict: {e_pd_init}")
                else:
                    logger.warning(f"Item '{param_key}' in params_space for strategy '{self.strategy_name}' is not ParamDetail or dict. Type: {type(param_value_obj)}")
        else:
            logger.error(f"params_space for strategy '{self.strategy_name}' is not a dict. Type: {type(raw_params_space)}")

        if not self.params_space_details:
            logger.critical(f"CRITICAL: self.params_space_details is EMPTY for strategy {self.strategy_name}. This will likely lead to pruning. Raw params_space: {raw_params_space}")


        if not isinstance(self.df_enriched_slice.index, pd.DatetimeIndex):
            if 'timestamp' in self.df_enriched_slice.columns:
                self.df_enriched_slice['timestamp'] = pd.to_datetime(self.df_enriched_slice['timestamp'], utc=True, errors='coerce')
                self.df_enriched_slice.dropna(subset=['timestamp'], inplace=True)
                self.df_enriched_slice = self.df_enriched_slice.set_index('timestamp')
            else:
                raise ValueError("ObjectiveEvaluator: df_enriched_slice must have a DatetimeIndex or a 'timestamp' column.")

        if self.df_enriched_slice.index.tz is None:
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_localize('UTC')
        elif self.df_enriched_slice.index.tz.utcoffset(self.df_enriched_slice.index[0]) != timezone.utc.utcoffset(self.df_enriched_slice.index[0]):
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_convert('UTC')
        
        if not self.df_enriched_slice.index.is_monotonic_increasing:
            self.df_enriched_slice.sort_index(inplace=True)
        if not self.df_enriched_slice.index.is_unique:
            self.df_enriched_slice = self.df_enriched_slice[~self.df_enriched_slice.index.duplicated(keep='first')]

        logger.debug(f"ObjectiveEvaluator for {self.strategy_name} on {self.pair_symbol} initialized. Enriched data shape: {self.df_enriched_slice.shape}")

    def _calculate_indicator_on_selected_klines(self,
                                                df_source_enriched: pd.DataFrame,
                                                indicator_type: str,
                                                indicator_params: Dict[str, Any],
                                                kline_ohlc_prefix: str
                                                ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        Calculates a technical indicator using pandas_ta on specified OHLCV columns from the source DataFrame.

        Args:
            df_source_enriched: The DataFrame containing source data (e.g., self.df_enriched_slice).
            indicator_type: The name of the indicator function in pandas_ta (e.g., "ema", "rsi").
            indicator_params: Dictionary of parameters for the indicator function (e.g., {"length": 10}).
            kline_ohlc_prefix: Prefix for OHLCV columns (e.g., "Klines_5min", "" for 1-min base).

        Returns:
            A pandas Series or DataFrame with the indicator result(s), or None on failure.
        """
        if df_source_enriched.empty:
            logger.warning(f"df_source_enriched is empty for {indicator_type} with prefix '{kline_ohlc_prefix}'.")
            return None

        # Construct source column names
        open_col = f"{kline_ohlc_prefix}_open" if kline_ohlc_prefix else "open"
        high_col = f"{kline_ohlc_prefix}_high" if kline_ohlc_prefix else "high"
        low_col = f"{kline_ohlc_prefix}_low" if kline_ohlc_prefix else "low"
        close_col = f"{kline_ohlc_prefix}_close" if kline_ohlc_prefix else "close"
        volume_col = f"{kline_ohlc_prefix}_volume" if kline_ohlc_prefix else "volume"

        # Determine required columns for pandas_ta based on indicator_type (simplified)
        required_ta_cols_map: Dict[str, str] = {}
        indicator_type_lower = indicator_type.lower()

        # Default to needing 'close'
        required_ta_cols_map['close'] = close_col

        # Indicators needing H, L, C
        if indicator_type_lower in ['psar', 'adx', 'atr', 'cci', 'donchian', 'ichimoku', 'supertrend', 'kama', 'bbands', 'kc', 'stoch', 'roc', 'mom', 'ao', 'apo', 'aroon', 'chop', 'coppock', 'dm', 'fisher', 'kst', 'massi', 'natr', 'ppo', 'qstick', 'stc', 'trix', 'tsi', 'uo', 'vhf', 'vortex', 'willr', 'alma', 'dema', 'ema', 'fwma', 'hma', 'linreg', 'midpoint', 'midprice', 'rma', 'sinwma', 'sma', 'smma', 'ssf', 'tema', 'trima', 'vidya', 'vwma', 'wcp', 'wma', 'zlma']:
            required_ta_cols_map['high'] = high_col
            required_ta_cols_map['low'] = low_col
        
        # Indicators also needing O (in addition to H, L, C for some)
        if indicator_type_lower in ['ichimoku', 'ao', 'ha', 'ohlc4']: # Example, list might need refinement
             required_ta_cols_map['open'] = open_col


        # Indicators needing V (in addition to C or HLC)
        if indicator_type_lower in ['obv', 'vwap', 'ad', 'adosc', 'cmf', 'efi', 'mfi', 'nvi', 'pvi', 'pvol', 'pvr', 'pvt', 'vwma', 'adx']: # ADX also uses volume for some calculations in some libraries, pandas_ta's adx does not explicitly list volume but it's good practice to provide if available
            required_ta_cols_map['volume'] = volume_col


        ta_inputs: Dict[str, pd.Series] = {}
        all_required_cols_present = True
        for ta_key, source_col_name in required_ta_cols_map.items():
            if source_col_name not in df_source_enriched.columns:
                # Only log error if the column was actually needed by the indicator type
                # This simple check might not be perfect for all pandas_ta indicators' optional args
                logger.warning(f"Source column '{source_col_name}' (for TA key '{ta_key}') for indicator '{indicator_type}' not found in df_source_enriched (prefix: '{kline_ohlc_prefix}').")
                if ta_key == 'close': # Close is almost always critical
                    all_required_cols_present = False; break
                continue # Allow calculation if other critical columns are present

            series_for_ta = df_source_enriched[source_col_name]
            if series_for_ta.isnull().all():
                logger.warning(f"Source column '{source_col_name}' (for TA key '{ta_key}') for indicator '{indicator_type}' is entirely NaN (prefix: '{kline_ohlc_prefix}').")
                if ta_key == 'close':
                     all_required_cols_present = False; break
            ta_inputs[ta_key] = series_for_ta.astype(float) # Ensure float type for pandas_ta

        if not all_required_cols_present or not ta_inputs.get('close', pd.Series(dtype=float)).notna().any():
            logger.error(f"Critical source data (e.g., close) missing or all NaN for indicator '{indicator_type}' with prefix '{kline_ohlc_prefix}'. Cannot calculate.")
            return pd.Series(np.nan, index=df_source_enriched.index) if 'close' not in ta_inputs or ta_inputs['close'].empty else None


        try:
            # Dynamically find the indicator function in pandas_ta
            indicator_function = getattr(ta, indicator_type_lower, None)
            if indicator_function is None:
                # Try to find in submodules like ta.trend, ta.momentum, etc.
                for category in [ta.trend, ta.momentum, ta.overlap, ta.volume, ta.volatility, ta.cycles, ta.statistics, ta.transform, ta.utils]:
                    if hasattr(category, indicator_type_lower):
                        indicator_function = getattr(category, indicator_type_lower)
                        break
            if indicator_function is None:
                logger.error(f"Indicator function '{indicator_type_lower}' not found in pandas_ta or its submodules.")
                return None
            
            # Clean indicator_params: pandas_ta expects specific keyword arguments.
            # We assume indicator_params only contains valid arguments for the function.
            # Example: {'length': 10, 'fast': 5, 'slow': 20}
            # Some pandas_ta functions might not accept all OHLCV if not needed.
            # We pass what we prepared in ta_inputs.
            logger.debug(f"Calculating {indicator_type_lower} with params: {indicator_params} on columns with prefix '{kline_ohlc_prefix}'. Input keys for TA: {list(ta_inputs.keys())}")
            
            # Ensure only necessary inputs are passed if indicator_function is picky
            # For simplicity, passing all available in ta_inputs. Most TA-Lib functions handle extra kwargs.
            result = indicator_function(**ta_inputs, **indicator_params, append=False)
            
            # If result is a DataFrame (e.g. MACD, PSAR, BBANDS), return it. If Series, also fine.
            if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
                return result
            else:
                logger.warning(f"Indicator {indicator_type_lower} did not return a Series or DataFrame. Got: {type(result)}")
                return None

        except Exception as e:
            logger.error(f"Error calculating {indicator_type} with params {indicator_params} using prefix '{kline_ohlc_prefix}': {e}", exc_info=True)
            return None

    def _prepare_data_with_dynamic_indicators(self, trial_params: Dict[str, Any], trial_number_for_log: Optional[Union[int, str]] = None) -> pd.DataFrame:
        """
        Prepares the DataFrame for the BacktestSimulator by:
        1. Copying base 1-min OHLCV from self.df_enriched_slice.
        2. Calculating/assigning ATR_strat based on trial parameters and pre-calculated ATRs in self.df_enriched_slice.
        3. Calculating other strategy-specific indicators (EMA_short_strat, MACD_line_strat, etc.)
           using trial parameters and appropriate source K-lines (1-min or aggregated) from self.df_enriched_slice.
        4. Calculating Taker Pressure indicators if specified by trial_params.
        5. Filling forward NaN values in all calculated _strat columns.
        """
        df_for_simulation = self.df_enriched_slice[['open', 'high', 'low', 'close', 'volume']].copy()

        current_trial_num_str = str(trial_number_for_log) if trial_number_for_log is not None else "N/A_IS_Prep"
        if self.is_oos_eval and self.is_trial_number_for_oos_log is not None:
            current_trial_num_str = f"OOS_for_IS_{self.is_trial_number_for_oos_log}"
        log_prefix = f"[{self.strategy_name}/{self.pair_symbol}/Trial-{current_trial_num_str}]"

        logger.info(f"{log_prefix} Preparing data with dynamic indicators. Enriched slice shape: {self.df_enriched_slice.shape}")
        logger.debug(f"{log_prefix} Trial params: {trial_params}")

        # 1. ATR_strat (for SL/TP)
        atr_period_key = 'atr_period_sl_tp' if 'atr_period_sl_tp' in trial_params else 'atr_period'
        atr_freq_key = 'atr_base_frequency_sl_tp' if 'atr_base_frequency_sl_tp' in trial_params else 'atr_base_frequency'

        atr_period_param = trial_params.get(atr_period_key)
        atr_freq_param_raw = trial_params.get(atr_freq_key)

        if atr_period_param is not None and atr_freq_param_raw is not None:
            atr_period_val = int(atr_period_param)
            kline_prefix_atr_source = data_utils.get_kline_prefix_effective(str(atr_freq_param_raw))
            
            # Construct the name of the pre-calculated ATR column in df_enriched_slice
            # Example: Klines_5min_ATR_14 or ATR_14 (if 1min)
            atr_source_col_name = f"{kline_prefix_atr_source}_ATR_{atr_period_val}" if kline_prefix_atr_source else f"ATR_{atr_period_val}"
            
            logger.debug(f"{log_prefix} ATR_strat: Trying to use pre-calculated column '{atr_source_col_name}' from enriched data.")

            if atr_source_col_name in self.df_enriched_slice.columns:
                df_for_simulation['ATR_strat'] = self.df_enriched_slice[atr_source_col_name].reindex(df_for_simulation.index, method='ffill')
                logger.info(f"{log_prefix} ATR_strat loaded from pre-calculated '{atr_source_col_name}'. NaNs: {df_for_simulation['ATR_strat'].isnull().sum()}/{len(df_for_simulation)}")
            elif kline_prefix_atr_source == "": # atr_freq_param_raw was '1min' or similar
                logger.info(f"{log_prefix} ATR_strat: Pre-calculated ATR for 1-min not found ('{atr_source_col_name}'). Calculating ATR({atr_period_val}) on 1-min base data.")
                atr_series_1min = self._calculate_indicator_on_selected_klines(
                    self.df_enriched_slice, 'atr', {'length': atr_period_val}, "" # Empty prefix for 1-min base
                )
                df_for_simulation['ATR_strat'] = atr_series_1min.reindex(df_for_simulation.index, method='ffill') if isinstance(atr_series_1min, pd.Series) else np.nan
                if isinstance(df_for_simulation.get('ATR_strat'), pd.Series):
                     logger.info(f"{log_prefix} ATR_strat (1-min dynamically calculated) NaNs: {df_for_simulation['ATR_strat'].isnull().sum()}/{len(df_for_simulation)}")
            else:
                logger.warning(f"{log_prefix} ATR_strat: Pre-calculated ATR column '{atr_source_col_name}' NOT FOUND in enriched data for frequency '{atr_freq_param_raw}'. "
                               f"ATR_strat will be NaN. This might indicate missing data in _enriched.parquet or misconfiguration.")
                df_for_simulation['ATR_strat'] = np.nan
        else:
            logger.warning(f"{log_prefix} ATR_strat: Parameters '{atr_period_key}' or '{atr_freq_key}' missing in trial_params. ATR_strat will be NaN.")
            df_for_simulation['ATR_strat'] = np.nan

        # 2. Strategy-Specific Indicators
        # This part needs to be adapted for each strategy's specific parameters and indicator calculations.
        # The example below is generic; specific strategies will have their own logic.
        
        # Example for a strategy needing 'EMA_short_strat' and 'EMA_long_strat'
        if 'ema_short_period' in trial_params and 'indicateur_frequence_ema' in trial_params:
            ema_s_p = int(trial_params['ema_short_period'])
            ema_freq_raw = str(trial_params['indicateur_frequence_ema'])
            kline_prefix_ema_src = data_utils.get_kline_prefix_effective(ema_freq_raw)
            
            ema_short_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'ema', {'length': ema_s_p}, kline_prefix_ema_src)
            df_for_simulation['EMA_short_strat'] = ema_short_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ema_short_series, pd.Series) else np.nan
            if isinstance(df_for_simulation.get('EMA_short_strat'), pd.Series):
                 logger.debug(f"{log_prefix} EMA_short_strat (Source: {kline_prefix_ema_src}_close, Period: {ema_s_p}) NaNs: {df_for_simulation['EMA_short_strat'].isnull().sum()}/{len(df_for_simulation)}")

        if 'ema_long_period' in trial_params and 'indicateur_frequence_ema' in trial_params: # Assuming same freq for long
            ema_l_p = int(trial_params['ema_long_period'])
            ema_freq_raw = str(trial_params['indicateur_frequence_ema']) # Re-fetch in case it's different, though unlikely for EMA short/long
            kline_prefix_ema_src = data_utils.get_kline_prefix_effective(ema_freq_raw)

            ema_long_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'ema', {'length': ema_l_p}, kline_prefix_ema_src)
            df_for_simulation['EMA_long_strat'] = ema_long_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ema_long_series, pd.Series) else np.nan
            if isinstance(df_for_simulation.get('EMA_long_strat'), pd.Series):
                logger.debug(f"{log_prefix} EMA_long_strat (Source: {kline_prefix_ema_src}_close, Period: {ema_l_p}) NaNs: {df_for_simulation['EMA_long_strat'].isnull().sum()}/{len(df_for_simulation)}")

        # Example for MACD
        if 'macd_fast_period' in trial_params and 'indicateur_frequence_macd' in trial_params:
            macd_params = {
                'fast': int(trial_params['macd_fast_period']),
                'slow': int(trial_params['macd_slow_period']),
                'signal': int(trial_params['macd_signal_period'])
            }
            macd_freq_raw = str(trial_params['indicateur_frequence_macd'])
            kline_prefix_macd_src = data_utils.get_kline_prefix_effective(macd_freq_raw)
            macd_df_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'macd', macd_params, kline_prefix_macd_src)
            if macd_df_result is not None and isinstance(macd_df_result, pd.DataFrame) and not macd_df_result.empty:
                # MACD typically returns 3 columns: MACD_line, MACD_histogram, MACD_signal_line
                # Column names from pandas_ta are like 'MACD_fast_slow_signal', 'MACDh_fast_slow_signal', 'MACDs_fast_slow_signal'
                df_for_simulation['MACD_line_strat'] = macd_df_result.iloc[:,0].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['MACD_hist_strat'] = macd_df_result.iloc[:,1].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['MACD_signal_strat'] = macd_df_result.iloc[:,2].reindex(df_for_simulation.index, method='ffill')
                if isinstance(df_for_simulation.get('MACD_line_strat'), pd.Series):
                    logger.debug(f"{log_prefix} MACD indicators calculated. MACD_line_strat NaNs: {df_for_simulation['MACD_line_strat'].isnull().sum()}/{len(df_for_simulation)}")
            else:
                df_for_simulation['MACD_line_strat'] = np.nan
                df_for_simulation['MACD_hist_strat'] = np.nan
                df_for_simulation['MACD_signal_strat'] = np.nan
                logger.warning(f"{log_prefix} MACD calculation failed or returned empty. Indicators will be NaN.")
        
        # Add other strategy-specific indicator calculations here based on self.strategy_name
        # and trial_params, similar to the EMA/MACD examples.
        # Ensure to use data_utils.get_kline_prefix_effective for source column selection.

        # 3. Taker Pressure Indicators (if applicable)
        if 'taker_pressure_indicator_period' in trial_params and 'indicateur_frequence_taker_pressure' in trial_params:
            taker_ma_period = int(trial_params['taker_pressure_indicator_period'])
            taker_freq_raw = str(trial_params['indicateur_frequence_taker_pressure'])
            kline_prefix_taker_src = data_utils.get_kline_prefix_effective(taker_freq_raw)

            # Determine source taker volume columns from self.df_enriched_slice
            buy_vol_col_src = f"{kline_prefix_taker_src}_taker_buy_base_asset_volume" if kline_prefix_taker_src else "taker_buy_base_asset_volume"
            sell_vol_col_src = f"{kline_prefix_taker_src}_taker_sell_base_asset_volume" if kline_prefix_taker_src else "taker_sell_base_asset_volume"

            if buy_vol_col_src in self.df_enriched_slice.columns and sell_vol_col_src in self.df_enriched_slice.columns:
                # Create a temporary DataFrame with only the necessary source columns for ratio calculation
                temp_taker_df = self.df_enriched_slice[[buy_vol_col_src, sell_vol_col_src]].copy()
                
                # Calculate raw ratio using the utility function
                temp_taker_df = data_utils.calculate_taker_pressure_ratio(
                    temp_taker_df,
                    taker_buy_volume_col=buy_vol_col_src, # Pass the actual column names from temp_taker_df
                    taker_sell_volume_col=sell_vol_col_src,
                    output_col_name="TakerPressureRatio_Raw"
                )
                if "TakerPressureRatio_Raw" in temp_taker_df.columns and temp_taker_df["TakerPressureRatio_Raw"].notna().any():
                    # Calculate MA on the raw ratio
                    taker_ratio_ma_series = ta.ema(temp_taker_df["TakerPressureRatio_Raw"].dropna(), length=taker_ma_period, append=False)
                    df_for_simulation['TakerPressureRatio_MA_strat'] = taker_ratio_ma_series.reindex(df_for_simulation.index, method='ffill')
                    if isinstance(df_for_simulation.get('TakerPressureRatio_MA_strat'), pd.Series):
                        logger.debug(f"{log_prefix} TakerPressureRatio_MA_strat calculated. NaNs: {df_for_simulation['TakerPressureRatio_MA_strat'].isnull().sum()}/{len(df_for_simulation)}")
                else:
                    df_for_simulation['TakerPressureRatio_MA_strat'] = np.nan
                    logger.warning(f"{log_prefix} TakerPressureRatio_Raw calculation failed or all NaN.")
            else:
                df_for_simulation['TakerPressureRatio_MA_strat'] = np.nan
                logger.warning(f"{log_prefix} Source columns for Taker Pressure ('{buy_vol_col_src}', '{sell_vol_col_src}') not found in enriched data.")


        # 4. Finalization
        indicator_strat_cols = [col for col in df_for_simulation.columns if col.endswith('_strat')]
        if indicator_strat_cols:
            df_for_simulation[indicator_strat_cols] = df_for_simulation[indicator_strat_cols].ffill()
            logger.debug(f"{log_prefix} Applied ffill to _strat columns: {indicator_strat_cols}")
        
        # Ensure all essential OHLCV columns are present and mostly non-NaN
        essential_ohlcv = ['open', 'high', 'low', 'close']
        if df_for_simulation[essential_ohlcv].isnull().all().any():
            logger.error(f"{log_prefix} One or more essential OHLCV columns are entirely NaN in df_for_simulation. This will likely cause simulation failure.")
            # Consider pruning here if data is unusable: raise optuna.exceptions.TrialPruned("Essential OHLCV data missing after indicator preparation.")

        logger.info(f"{log_prefix} Data preparation complete. df_for_simulation shape: {df_for_simulation.shape}. Columns: {df_for_simulation.columns.tolist()}")
        return df_for_simulation


    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggests parameters for a trial based on the defined parameter space."""
        params_for_trial: Dict[str, Any] = {}
        if not self.params_space_details:
            logger.error(f"params_space_details not initialized for strategy '{self.strategy_name}'. Cannot suggest parameters.")
            # This situation should ideally lead to pruning if it occurs during an Optuna run.
            raise optuna.exceptions.TrialPruned("Parameter space not defined for the strategy.")

        # Example of conditional parameter suggestion:
        # This needs to be customized based on actual strategy parameters.
        # For instance, if 'ma_type' determines other MA parameters:

        # ma_type_param_detail = self.params_space_details.get('ma_type')
        # if ma_type_param_detail and ma_type_param_detail.type == 'categorical' and ma_type_param_detail.choices:
        #     suggested_ma_type = trial.suggest_categorical('ma_type', ma_type_param_detail.choices)
        #     params_for_trial['ma_type'] = suggested_ma_type
        #
        #     if suggested_ma_type == 'ema':
        #         ema_period_detail = self.params_space_details.get('ema_period') # Assuming 'ema_period' is defined for this case
        #         if ema_period_detail and ema_period_detail.type == 'int':
        #             params_for_trial['ema_period'] = trial.suggest_int('ema_period_conditional', int(ema_period_detail.low), int(ema_period_detail.high), step=int(ema_period_detail.step or 1))
        #     elif suggested_ma_type == 'sma':
        #         sma_period_detail = self.params_space_details.get('sma_period')
        #         if sma_period_detail and sma_period_detail.type == 'int':
        #             params_for_trial['sma_period'] = trial.suggest_int('sma_period_conditional', int(sma_period_detail.low), int(sma_period_detail.high), step=int(sma_period_detail.step or 1))
        # else: # If ma_type is not part of this strategy's params_space, or not categorical
        #     pass # Proceed to suggest other parameters normally


        for param_name, p_detail in self.params_space_details.items():
            # Skip if already handled by conditional logic above
            if param_name in params_for_trial:
                continue

            try:
                if p_detail.type == 'int':
                    params_for_trial[param_name] = trial.suggest_int(param_name, int(p_detail.low), int(p_detail.high), step=int(p_detail.step or 1))
                elif p_detail.type == 'float':
                    params_for_trial[param_name] = trial.suggest_float(param_name, float(p_detail.low), float(p_detail.high), step=p_detail.step)
                elif p_detail.type == 'categorical' and p_detail.choices:
                    params_for_trial[param_name] = trial.suggest_categorical(param_name, p_detail.choices)
            except Exception as e_suggest:
                logger.error(f"Error suggesting param '{param_name}' (type: {p_detail.type}): {e_suggest}", exc_info=True)
                # Fallback to a default if suggestion fails, or prune
                if p_detail.choices: params_for_trial[param_name] = p_detail.choices[0]
                elif p_detail.low is not None: params_for_trial[param_name] = p_detail.low
                else:
                    logger.warning(f"Cannot set fallback for param '{param_name}'. Trial might be unstable.")
                    raise optuna.exceptions.TrialPruned(f"Failed to suggest or fallback for param {param_name}")
        return params_for_trial

    def __call__(self, trial: optuna.Trial) -> Tuple[float, ...]:
        """
        Objective function for Optuna. Called for each trial.
        """
        start_time_trial = time.time()
        
        trial_id_for_log: str
        trial_number_for_prepare_log: Optional[int] = None

        if self.is_oos_eval:
            is_trial_num_oos = self.is_trial_number_for_oos_log
            if is_trial_num_oos is not None:
                trial_id_for_log = f"OOS_for_IS_Trial_{is_trial_num_oos}"
                trial_number_for_prepare_log = is_trial_num_oos
            else:
                trial_id_for_log = f"OOS_Trial_UnknownOrigin_{trial.number if hasattr(trial, 'number') else uuid.uuid4().hex[:6]}"
                if hasattr(trial, 'number'): trial_number_for_prepare_log = trial.number
        else:
            trial_id_for_log = str(trial.number) if hasattr(trial, 'number') and trial.number is not None else f"IS_Trial_Unknown_{uuid.uuid4().hex[:6]}"
            if hasattr(trial, 'number'): trial_number_for_prepare_log = trial.number

        log_prefix = f"[{self.strategy_name}/{self.pair_symbol}/Trial-{trial_id_for_log}]"
        logger.info(f"{log_prefix} Starting evaluation...")

        StrategyClass_local: Optional[Type[BaseStrategy]] = None
        try:
            module_path = self.strategy_script_ref.replace('.py', '').replace('/', '.')
            module = importlib.import_module(module_path)
            StrategyClass_local = getattr(module, self.strategy_class_name)
            if not StrategyClass_local or not issubclass(StrategyClass_local, BaseStrategy):
                raise ImportError(f"Could not load a valid BaseStrategy subclass: {self.strategy_class_name}")
        except Exception as e_load_strat:
            logger.error(f"{log_prefix} Failed to load strategy class {self.strategy_class_name}: {e_load_strat}", exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Strategy class load failure: {e_load_strat}")

        current_trial_params: Dict[str, Any]
        if self.is_oos_eval:
            # For OOS, params are fixed, passed from the IS trial
            current_trial_params = trial.params.copy() # trial.params should hold the IS trial's params
            # Clean up any internal Optuna params if they were somehow passed
            current_trial_params.pop("_trial_id_for_oos", None) # Example internal param
            logger.info(f"{log_prefix} OOS evaluation using fixed params: {current_trial_params}")
        else:
            # For IS, suggest parameters
            try:
                current_trial_params = self._suggest_params(trial)
                if not current_trial_params:
                    logger.error(f"{log_prefix} No parameters were suggested. Pruning trial.")
                    raise optuna.exceptions.TrialPruned("No parameters suggested.")
                logger.info(f"{log_prefix} IS evaluation with suggested params: {current_trial_params}")
            except optuna.exceptions.TrialPruned: # Re-raise if _suggest_params prunes
                raise
            except Exception as e_suggest:
                logger.error(f"{log_prefix} Error during parameter suggestion: {e_suggest}", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Parameter suggestion failed: {e_suggest}")


        try:
            data_for_simulation = self._prepare_data_with_dynamic_indicators(current_trial_params, trial_number_for_log=trial_number_for_prepare_log)
            if data_for_simulation.empty or data_for_simulation[['open', 'high', 'low', 'close']].isnull().all().all():
                logger.error(f"{log_prefix} Data for simulation is empty or OHLC fully NaN after preparation. Pruning.")
                raise optuna.exceptions.TrialPruned("Data preparation resulted in unusable data.")
        except optuna.exceptions.TrialPruned: # Re-raise if _prepare_data prunes
            raise
        except Exception as e_prepare:
            logger.error(f"{log_prefix} Error preparing data with dynamic indicators: {e_prepare}", exc_info=True)
            # Return worst possible values for all objectives
            return tuple([-float('inf') if direction == "maximize" else float('inf')
                          for direction in self.optuna_objectives_config['objectives_directions']])

        strategy_instance = StrategyClass_local(params=current_trial_params)

        # Prepare simulation settings for this specific trial
        sim_settings_for_trial = self.app_config.global_config.simulation_defaults.__dict__.copy()
        sim_settings_for_trial['symbol'] = self.pair_symbol
        sim_settings_for_trial['symbol_info'] = self.symbol_info_data
        # Potentially override from app_config.live_config if certain live settings should influence backtest behavior
        # For example, if capital_allocation_pct or margin_leverage from strategy params should be used:
        sim_settings_for_trial['capital_allocation_pct'] = current_trial_params.get('capital_allocation_pct', sim_settings_for_trial.get('capital_allocation_pct'))
        sim_settings_for_trial['margin_leverage'] = current_trial_params.get('margin_leverage', sim_settings_for_trial.get('margin_leverage'))


        # Determine output directory for this specific trial's artifacts (if any)
        # For OOS, log to a subfolder of the IS trial that generated these params
        # For IS, log to trial.number subfolder
        trial_artifact_dir: Optional[Path] = None
        if self.app_config.global_config.paths.logs_backtest_optimization:
            base_log_dir = Path(self.app_config.global_config.paths.logs_backtest_optimization)
            # Construct path: RUN_ID / STRATEGY / PAIR / CONTEXT / FOLD_X / trial_Y_IS or trial_Y_OOS_for_IS_Z
            # This requires knowledge of the current WFO run_id and fold_id, which are not directly in ObjectiveEvaluator.
            # The `output_dir_fold` in StudyManager's `run_optimization_for_fold` provides up to FOLD_X.
            # We can create a sub-directory for the trial there.
            # This is more relevant if BacktestSimulator saves detailed logs per trial.
            # For now, BacktestSimulator's output_dir is set to None by default in the orchestrator.
            # If detailed trial logs are needed, this path logic needs to be robust.
            pass # Simulator output_dir is None for Optuna runs to avoid excessive disk I/O by default

        simulator = BacktestSimulator(
            historical_data_with_indicators=data_for_simulation,
            strategy_instance=strategy_instance,
            simulation_settings=sim_settings_for_trial,
            output_dir=None # Set to None to avoid saving artifacts for each Optuna trial by default
        )

        try:
            backtest_results = simulator.run_simulation()
        except Exception as e_sim:
            logger.error(f"{log_prefix} Error during BacktestSimulator execution: {e_sim}", exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Simulation failed: {e_sim}")


        metrics = backtest_results.get("metrics", {})
        if not metrics:
            logger.warning(f"{log_prefix} No metrics returned from simulation. Pruning.")
            raise optuna.exceptions.TrialPruned("Simulation returned no metrics.")

        objective_values: List[float] = []
        for i, metric_name in enumerate(self.optuna_objectives_config['objectives_names']):
            value = metrics.get(metric_name)
            direction = self.optuna_objectives_config['objectives_directions'][i]

            if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
                logger.warning(f"{log_prefix} Objective '{metric_name}' is None, NaN, or Inf (value: {value}). Assigning default worst value.")
                value = -float('inf') if direction == "maximize" else float('inf')
            objective_values.append(float(value))
        
        # Log all metrics for this trial for easier debugging/analysis from logs
        logger.debug(f"{log_prefix} All metrics for trial: {metrics}")

        end_time_trial = time.time()
        logger.info(f"{log_prefix} Evaluation finished in {end_time_trial - start_time_trial:.2f}s. Objectives: {objective_values}")

        return tuple(objective_values)

