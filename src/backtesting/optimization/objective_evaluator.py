# src/backtesting/optimizer/objective_evaluator.py
import logging
import time
import importlib
from typing import Any, Dict, Optional, Tuple, List, Type, Union
from datetime import timezone
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta # type: ignore
import optuna

try:
    from src.backtesting.simulator import BacktestSimulator
    from src.data import data_utils
    from src.strategies.base import BaseStrategy
    from src.config.definitions import ParamDetail, AppConfig, GlobalConfig, ExchangeSettings # GlobalConfig au lieu de GlobalSettings
    from src.backtesting.performance_analyzer import calculate_performance_metrics_from_inputs # Importer la fonction renommée
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger(__name__).critical(f"ObjectiveEvaluator: Critical import error: {e}. Ensure PYTHONPATH is correct.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class ObjectiveEvaluator:
    def __init__(self,
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any], 
                 df_enriched_slice: pd.DataFrame,
                 optuna_objectives_config: Dict[str, Any], 
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any], # C'est le pair_config
                 app_config: AppConfig, 
                 run_id: str, 
                 is_oos_eval: bool = False,
                 is_trial_number_for_oos_log: Optional[int] = None):

        self.strategy_name_key = strategy_name
        self.strategy_config_dict = strategy_config_dict
        self.df_enriched_slice = df_enriched_slice.copy()
        self.optuna_objectives_config = optuna_objectives_config
        self.pair_symbol = pair_symbol
        self.symbol_info_data = symbol_info_data # Stocker le pair_config
        self.is_oos_eval = is_oos_eval
        self.app_config = app_config
        self.run_id = run_id
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
                    logger.warning(f"Item '{param_key}' in params_space for strategy '{self.strategy_name_key}' is not ParamDetail or dict. Type: {type(param_value_obj)}")
        else:
            logger.error(f"params_space for strategy '{self.strategy_name_key}' is not a dict. Type: {type(raw_params_space)}")

        if not self.params_space_details and not self.is_oos_eval:
            logger.warning(f"Warning: self.params_space_details is EMPTY for IS strategy {self.strategy_name_key}. This might lead to pruning if no params are suggested. Raw params_space: {raw_params_space}")

        if not isinstance(self.df_enriched_slice.index, pd.DatetimeIndex):
            if 'timestamp' in self.df_enriched_slice.columns:
                self.df_enriched_slice['timestamp'] = pd.to_datetime(self.df_enriched_slice['timestamp'], utc=True, errors='coerce')
                self.df_enriched_slice.dropna(subset=['timestamp'], inplace=True)
                self.df_enriched_slice = self.df_enriched_slice.set_index('timestamp')
            else:
                raise ValueError("ObjectiveEvaluator: df_enriched_slice must have a DatetimeIndex or a 'timestamp' column.")

        if self.df_enriched_slice.index.tz is None:
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_localize('UTC')
        elif self.df_enriched_slice.index.tz.utcoffset(self.df_enriched_slice.index[0]) != timezone.utc.utcoffset(self.df_enriched_slice.index[0]): # type: ignore
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_convert('UTC')
        
        if not self.df_enriched_slice.index.is_monotonic_increasing:
            self.df_enriched_slice.sort_index(inplace=True)
        if not self.df_enriched_slice.index.is_unique:
            self.df_enriched_slice = self.df_enriched_slice[~self.df_enriched_slice.index.duplicated(keep='first')]
        
        self.last_backtest_results: Optional[Dict[str, Any]] = None 

        logger.debug(f"ObjectiveEvaluator for {self.strategy_class_name} (key: {self.strategy_name_key}) on {self.pair_symbol} initialized. Enriched data shape: {self.df_enriched_slice.shape}. Run ID: {self.run_id}")

    def _calculate_indicator_on_selected_klines(self,
                                                df_source_enriched: pd.DataFrame,
                                                indicator_type: str,
                                                indicator_params: Dict[str, Any],
                                                kline_ohlc_prefix: str
                                                ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        if df_source_enriched.empty:
            logger.warning(f"df_source_enriched is empty for {indicator_type} with prefix '{kline_ohlc_prefix}'.")
            return None

        open_col = f"{kline_ohlc_prefix}_open" if kline_ohlc_prefix else "open"
        high_col = f"{kline_ohlc_prefix}_high" if kline_ohlc_prefix else "high"
        low_col = f"{kline_ohlc_prefix}_low" if kline_ohlc_prefix else "low"
        close_col = f"{kline_ohlc_prefix}_close" if kline_ohlc_prefix else "close"
        volume_col = f"{kline_ohlc_prefix}_volume" if kline_ohlc_prefix else "volume"

        required_ta_cols_map: Dict[str, str] = {'close': close_col}
        indicator_type_lower = indicator_type.lower()

        if indicator_type_lower in ['psar', 'adx', 'atr', 'cci', 'donchian', 'ichimoku', 'supertrend', 'kama', 'bbands', 'kc', 'stoch', 'roc', 'mom', 'ao', 'apo', 'aroon', 'chop', 'coppock', 'dm', 'fisher', 'kst', 'massi', 'natr', 'ppo', 'qstick', 'stc', 'trix', 'tsi', 'uo', 'vhf', 'vortex', 'willr', 'alma', 'dema', 'ema', 'fwma', 'hma', 'linreg', 'midpoint', 'midprice', 'rma', 'sinwma', 'sma', 'smma', 'ssf', 'tema', 'trima', 'vidya', 'vwma', 'wcp', 'wma', 'zlma', 'slope']:
            required_ta_cols_map['high'] = high_col
            required_ta_cols_map['low'] = low_col
        
        if indicator_type_lower in ['ichimoku', 'ao', 'ha', 'ohlc4']:
             required_ta_cols_map['open'] = open_col

        if indicator_type_lower in ['obv', 'vwap', 'ad', 'adosc', 'cmf', 'efi', 'mfi', 'nvi', 'pvi', 'pvol', 'pvr', 'pvt', 'vwma', 'adx']:
            required_ta_cols_map['volume'] = volume_col

        ta_inputs: Dict[str, pd.Series] = {}
        all_required_cols_present = True
        for ta_key, source_col_name in required_ta_cols_map.items():
            if source_col_name not in df_source_enriched.columns:
                logger.warning(f"Source column '{source_col_name}' for indicator '{indicator_type}' not found (prefix: '{kline_ohlc_prefix}').")
                if ta_key == 'close': all_required_cols_present = False; break
                continue
            series_for_ta = df_source_enriched[source_col_name]
            if series_for_ta.isnull().all():
                logger.warning(f"Source column '{source_col_name}' for indicator '{indicator_type}' is all NaN (prefix: '{kline_ohlc_prefix}').")
                if ta_key == 'close': all_required_cols_present = False; break
            ta_inputs[ta_key] = series_for_ta.astype(float)

        if not all_required_cols_present or not ta_inputs.get('close', pd.Series(dtype=float)).notna().any(): # type: ignore
            logger.error(f"Critical source data (especially 'close') missing or all NaN for indicator '{indicator_type}' with prefix '{kline_ohlc_prefix}'.")
            return None

        try:
            indicator_function = getattr(ta, indicator_type_lower, None)
            if indicator_function is None:
                for category in [ta.trend, ta.momentum, ta.overlap, ta.volume, ta.volatility, ta.cycles, ta.statistics, ta.transform, ta.utils]:
                    if hasattr(category, indicator_type_lower):
                        indicator_function = getattr(category, indicator_type_lower)
                        break
            if indicator_function is None:
                logger.error(f"Indicator function '{indicator_type_lower}' not found in pandas_ta.")
                return None
            
            logger.debug(f"Calculating {indicator_type_lower} with params: {indicator_params} on columns with prefix '{kline_ohlc_prefix}'. Inputs: {list(ta_inputs.keys())}")
            result = indicator_function(**ta_inputs, **indicator_params, append=False) # type: ignore
            
            if isinstance(result, (pd.DataFrame, pd.Series)):
                return result
            else:
                logger.warning(f"Indicator {indicator_type_lower} did not return a Series/DataFrame. Got: {type(result)}")
                return None
        except Exception as e:
            logger.error(f"Error calculating {indicator_type} with params {indicator_params} (prefix '{kline_ohlc_prefix}'): {e}", exc_info=True)
            return None

    def _prepare_data_with_dynamic_indicators(self, trial_params: Dict[str, Any], trial_number_for_log: Optional[Union[int, str]] = None) -> pd.DataFrame:
        df_for_simulation = self.df_enriched_slice[['open', 'high', 'low', 'close', 'volume']].copy()
        current_trial_num_str = str(trial_number_for_log) if trial_number_for_log is not None else "N/A_IS_Prep"
        if self.is_oos_eval and self.is_trial_number_for_oos_log is not None:
            current_trial_num_str = f"OOS_for_IS_{self.is_trial_number_for_oos_log}"
        log_prefix = f"[{self.strategy_class_name}/{self.pair_symbol}/Trial-{current_trial_num_str}]"

        logger.info(f"{log_prefix} Preparing data with dynamic indicators. Enriched slice shape: {self.df_enriched_slice.shape}")
        logger.debug(f"{log_prefix} Trial params: {trial_params}")

        atr_period_key = 'atr_period_sl_tp' if 'atr_period_sl_tp' in trial_params else 'atr_period'
        atr_freq_key = 'atr_base_frequency_sl_tp' if 'atr_base_frequency_sl_tp' in trial_params else 'atr_base_frequency'
        atr_period_param = trial_params.get(atr_period_key)
        atr_freq_param_raw = trial_params.get(atr_freq_key)

        if atr_period_param is not None and atr_freq_param_raw is not None:
            atr_period_val = int(atr_period_param)
            kline_prefix_atr_source = data_utils.get_kline_prefix_effective(str(atr_freq_param_raw))
            atr_source_col_name = f"{kline_prefix_atr_source}_ATR_{atr_period_val}" if kline_prefix_atr_source else f"ATR_{atr_period_val}"
            if atr_source_col_name in self.df_enriched_slice.columns:
                df_for_simulation['ATR_strat'] = self.df_enriched_slice[atr_source_col_name].reindex(df_for_simulation.index, method='ffill')
                logger.info(f"{log_prefix} ATR_strat loaded from pre-calculated '{atr_source_col_name}'.")
            elif kline_prefix_atr_source == "": 
                atr_series_1min = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'atr', {'length': atr_period_val}, "")
                df_for_simulation['ATR_strat'] = atr_series_1min.reindex(df_for_simulation.index, method='ffill') if isinstance(atr_series_1min, pd.Series) else np.nan
                if isinstance(df_for_simulation.get('ATR_strat'), pd.Series) and df_for_simulation['ATR_strat'].notna().any(): logger.info(f"{log_prefix} ATR_strat (1-min dynamically calculated).")
                else: logger.warning(f"{log_prefix} ATR_strat (1-min dynamic calculation) FAILED or all NaN.")
            else:
                logger.warning(f"{log_prefix} ATR_strat: Pre-calculated ATR column '{atr_source_col_name}' NOT FOUND for freq '{atr_freq_param_raw}'. ATR_strat will be NaN.")
                df_for_simulation['ATR_strat'] = np.nan
        else:
            logger.warning(f"{log_prefix} ATR_strat: Parameters missing. ATR_strat will be NaN.")
            df_for_simulation['ATR_strat'] = np.nan
        
        if self.strategy_class_name == "MaCrossoverStrategy":
            logger.info(f"{log_prefix} ENTERING MaCrossoverStrategy indicator calculation block.")
            fast_ma_period = trial_params.get('fast_ma_period')
            slow_ma_period = trial_params.get('slow_ma_period')
            ma_type = str(trial_params.get('ma_type', 'sma')).lower()
            freq_fast_ma_raw = str(trial_params.get('indicateur_frequence_ma_rapide'))
            freq_slow_ma_raw = str(trial_params.get('indicateur_frequence_ma_lente'))

            logger.debug(f"{log_prefix} MaCrossover - Fast MA Params: period={fast_ma_period}, type={ma_type}, freq={freq_fast_ma_raw}")
            if fast_ma_period is not None and freq_fast_ma_raw and freq_fast_ma_raw.lower() != 'none':
                kline_prefix_fast = data_utils.get_kline_prefix_effective(freq_fast_ma_raw)
                logger.debug(f"{log_prefix} MaCrossover - Fast MA Kline Prefix: {kline_prefix_fast}")
                fast_ma_series = self._calculate_indicator_on_selected_klines(
                    self.df_enriched_slice, ma_type, {'length': int(fast_ma_period)}, kline_prefix_fast
                )
                if isinstance(fast_ma_series, pd.Series):
                    df_for_simulation['MA_FAST_strat'] = fast_ma_series.reindex(df_for_simulation.index, method='ffill')
                    logger.debug(f"{log_prefix} MA_FAST_strat assigned. Length: {len(df_for_simulation['MA_FAST_strat'])}, NaNs: {df_for_simulation['MA_FAST_strat'].isnull().sum()}")
                else:
                    df_for_simulation['MA_FAST_strat'] = np.nan
                    logger.warning(f"{log_prefix} fast_ma_series was None or not a Series. MA_FAST_strat set to NaN.")
            else:
                df_for_simulation['MA_FAST_strat'] = np.nan
                logger.warning(f"{log_prefix} fast_ma_period or freq_fast_ma_raw is None/invalid. MA_FAST_strat set to NaN. Period: {fast_ma_period}, Freq: {freq_fast_ma_raw}")

            logger.debug(f"{log_prefix} MaCrossover - Slow MA Params: period={slow_ma_period}, type={ma_type}, freq={freq_slow_ma_raw}")
            if slow_ma_period is not None and freq_slow_ma_raw and freq_slow_ma_raw.lower() != 'none':
                kline_prefix_slow = data_utils.get_kline_prefix_effective(freq_slow_ma_raw)
                logger.debug(f"{log_prefix} MaCrossover - Slow MA Kline Prefix: {kline_prefix_slow}")
                slow_ma_series = self._calculate_indicator_on_selected_klines(
                    self.df_enriched_slice, ma_type, {'length': int(slow_ma_period)}, kline_prefix_slow
                )
                if isinstance(slow_ma_series, pd.Series):
                    df_for_simulation['MA_SLOW_strat'] = slow_ma_series.reindex(df_for_simulation.index, method='ffill')
                    logger.debug(f"{log_prefix} MA_SLOW_strat assigned. Length: {len(df_for_simulation['MA_SLOW_strat'])}, NaNs: {df_for_simulation['MA_SLOW_strat'].isnull().sum()}")
                else:
                    df_for_simulation['MA_SLOW_strat'] = np.nan
                    logger.warning(f"{log_prefix} slow_ma_series was None or not a Series. MA_SLOW_strat set to NaN.")
            else:
                df_for_simulation['MA_SLOW_strat'] = np.nan
                logger.warning(f"{log_prefix} slow_ma_period or freq_slow_ma_raw is None/invalid. MA_SLOW_strat set to NaN. Period: {slow_ma_period}, Freq: {freq_slow_ma_raw}")
            logger.info(f"{log_prefix} EXITING MaCrossoverStrategy block. df_for_simulation columns: {df_for_simulation.columns.tolist()}")

        elif self.strategy_class_name == "PsarReversalOtocoStrategy":
            logger.info(f"{log_prefix} ENTERING PsarReversalOtocoStrategy indicator calculation block.")
            psar_step = trial_params.get('psar_step')
            psar_max_step = trial_params.get('psar_max_step')
            psar_freq_raw = str(trial_params.get('indicateur_frequence_psar'))
            
            if psar_step is not None and psar_max_step is not None and psar_freq_raw and psar_freq_raw.lower() != 'none':
                kline_prefix_psar = data_utils.get_kline_prefix_effective(psar_freq_raw)
                psar_params_ta = {'af': float(psar_step), 'max_af': float(psar_max_step)} 
                psar_df_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'psar', psar_params_ta, kline_prefix_psar)
                if psar_df_result is not None and isinstance(psar_df_result, pd.DataFrame) and not psar_df_result.empty:
                    long_col_name = next((col for col in psar_df_result.columns if 'psarl' in col.lower()), None)
                    short_col_name = next((col for col in psar_df_result.columns if 'psars' in col.lower()), None)

                    if long_col_name: df_for_simulation['PSARl_strat'] = psar_df_result[long_col_name].reindex(df_for_simulation.index, method='ffill')
                    else: df_for_simulation['PSARl_strat'] = np.nan
                    
                    if short_col_name: df_for_simulation['PSARs_strat'] = psar_df_result[short_col_name].reindex(df_for_simulation.index, method='ffill')
                    else: df_for_simulation['PSARs_strat'] = np.nan
                else:
                    df_for_simulation['PSARl_strat'] = np.nan
                    df_for_simulation['PSARs_strat'] = np.nan
                    logger.warning(f"{log_prefix} PSAR calculation failed or returned empty.")
            else:
                df_for_simulation['PSARl_strat'] = np.nan
                df_for_simulation['PSARs_strat'] = np.nan
                logger.warning(f"{log_prefix} psar parameters or frequency missing/invalid for PsarReversalOtocoStrategy.")
            logger.info(f"{log_prefix} EXITING PsarReversalOtocoStrategy block. df_for_simulation columns: {df_for_simulation.columns.tolist()}")

        elif self.strategy_class_name == "TripleMAAnticipationStrategy":
            logger.info(f"{log_prefix} ENTERING TripleMAAnticipationStrategy indicator calculation block.")
            ma_short_p = trial_params.get('ma_short_period')
            ma_medium_p = trial_params.get('ma_medium_period')
            ma_long_p = trial_params.get('ma_long_period')
            ma_type_triple = str(trial_params.get('ma_type', 'ema')).lower() 
            
            freq_mms_raw = str(trial_params.get('indicateur_frequence_mms'))
            freq_mmm_raw = str(trial_params.get('indicateur_frequence_mmm'))
            freq_mml_raw = str(trial_params.get('indicateur_frequence_mml'))

            if ma_short_p is not None and freq_mms_raw and freq_mms_raw.lower() != 'none':
                kline_prefix_s = data_utils.get_kline_prefix_effective(freq_mms_raw)
                series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, ma_type_triple, {'length': int(ma_short_p)}, kline_prefix_s)
                df_for_simulation['MA_SHORT_strat'] = series.reindex(df_for_simulation.index, method='ffill') if isinstance(series, pd.Series) else np.nan
            else: df_for_simulation['MA_SHORT_strat'] = np.nan
            
            if ma_medium_p is not None and freq_mmm_raw and freq_mmm_raw.lower() != 'none':
                kline_prefix_m = data_utils.get_kline_prefix_effective(freq_mmm_raw)
                series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, ma_type_triple, {'length': int(ma_medium_p)}, kline_prefix_m)
                df_for_simulation['MA_MEDIUM_strat'] = series.reindex(df_for_simulation.index, method='ffill') if isinstance(series, pd.Series) else np.nan
            else: df_for_simulation['MA_MEDIUM_strat'] = np.nan

            if ma_long_p is not None and freq_mml_raw and freq_mml_raw.lower() != 'none':
                kline_prefix_l = data_utils.get_kline_prefix_effective(freq_mml_raw)
                series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, ma_type_triple, {'length': int(ma_long_p)}, kline_prefix_l)
                df_for_simulation['MA_LONG_strat'] = series.reindex(df_for_simulation.index, method='ffill') if isinstance(series, pd.Series) else np.nan
            else: df_for_simulation['MA_LONG_strat'] = np.nan
            
            if trial_params.get('anticipate_crossovers', False):
                slope_period = int(trial_params.get('anticipation_slope_period', 3))
                if slope_period < 2: slope_period = 2 
                
                if 'MA_SHORT_strat' in df_for_simulation and df_for_simulation['MA_SHORT_strat'].notna().any():
                    try:
                        ma_short_clean = df_for_simulation['MA_SHORT_strat'].dropna()
                        if len(ma_short_clean) >= slope_period:
                            slope_short_series = ta.slope(ma_short_clean, length=slope_period, append=False) # type: ignore
                            df_for_simulation['SLOPE_MA_SHORT_strat'] = slope_short_series.reindex(df_for_simulation.index, method='ffill') if isinstance(slope_short_series, pd.Series) else np.nan
                        else:
                            df_for_simulation['SLOPE_MA_SHORT_strat'] = np.nan
                    except Exception as e_slope_s:
                        logger.warning(f"{log_prefix} Error calculating slope for MA_SHORT_strat: {e_slope_s}")
                        df_for_simulation['SLOPE_MA_SHORT_strat'] = np.nan
                else: df_for_simulation['SLOPE_MA_SHORT_strat'] = np.nan

                if 'MA_MEDIUM_strat' in df_for_simulation and df_for_simulation['MA_MEDIUM_strat'].notna().any():
                    try:
                        ma_medium_clean = df_for_simulation['MA_MEDIUM_strat'].dropna()
                        if len(ma_medium_clean) >= slope_period:
                            slope_medium_series = ta.slope(ma_medium_clean, length=slope_period, append=False) # type: ignore
                            df_for_simulation['SLOPE_MA_MEDIUM_strat'] = slope_medium_series.reindex(df_for_simulation.index, method='ffill') if isinstance(slope_medium_series, pd.Series) else np.nan
                        else:
                            df_for_simulation['SLOPE_MA_MEDIUM_strat'] = np.nan
                    except Exception as e_slope_m:
                        logger.warning(f"{log_prefix} Error calculating slope for MA_MEDIUM_strat: {e_slope_m}")
                        df_for_simulation['SLOPE_MA_MEDIUM_strat'] = np.nan
                else: df_for_simulation['SLOPE_MA_MEDIUM_strat'] = np.nan
            logger.info(f"{log_prefix} EXITING TripleMAAnticipationStrategy block. df_for_simulation columns: {df_for_simulation.columns.tolist()}")

        elif self.strategy_class_name == "BbandsVolumeRsiStrategy":
            logger.info(f"{log_prefix} ENTERING BbandsVolumeRsiStrategy indicator calculation block.")
            bb_period = trial_params.get('bbands_period')
            bb_std = trial_params.get('bbands_std_dev')
            bb_freq_raw = str(trial_params.get('indicateur_frequence_bbands'))
            vol_ma_period = trial_params.get('volume_ma_period')
            vol_freq_raw = str(trial_params.get('indicateur_frequence_volume'))
            rsi_period = trial_params.get('rsi_period')
            rsi_freq_raw = str(trial_params.get('indicateur_frequence_rsi'))

            if bb_period is not None and bb_std is not None and bb_freq_raw and bb_freq_raw.lower() != 'none':
                kline_prefix_bb = data_utils.get_kline_prefix_effective(bb_freq_raw)
                bb_params = {'length': int(bb_period), 'std': float(bb_std)}
                bb_df_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'bbands', bb_params, kline_prefix_bb)
                if bb_df_result is not None and isinstance(bb_df_result, pd.DataFrame) and not bb_df_result.empty and len(bb_df_result.columns) >= 4:
                    bbl_col = next((col for col in bb_df_result.columns if 'bbl' in col.lower()), bb_df_result.columns[0])
                    bbm_col = next((col for col in bb_df_result.columns if 'bbm' in col.lower()), bb_df_result.columns[1])
                    bbu_col = next((col for col in bb_df_result.columns if 'bbu' in col.lower()), bb_df_result.columns[2])
                    bbb_col = next((col for col in bb_df_result.columns if 'bbb' in col.lower()), bb_df_result.columns[3]) 

                    df_for_simulation['BB_LOWER_strat'] = bb_df_result[bbl_col].reindex(df_for_simulation.index, method='ffill')
                    df_for_simulation['BB_MIDDLE_strat'] = bb_df_result[bbm_col].reindex(df_for_simulation.index, method='ffill')
                    df_for_simulation['BB_UPPER_strat'] = bb_df_result[bbu_col].reindex(df_for_simulation.index, method='ffill')
                    df_for_simulation['BB_BANDWIDTH_strat'] = bb_df_result[bbb_col].reindex(df_for_simulation.index, method='ffill')
                else:
                    for col in ['BB_LOWER_strat', 'BB_MIDDLE_strat', 'BB_UPPER_strat', 'BB_BANDWIDTH_strat']: df_for_simulation[col] = np.nan
            else: 
                for col in ['BB_LOWER_strat', 'BB_MIDDLE_strat', 'BB_UPPER_strat', 'BB_BANDWIDTH_strat']: df_for_simulation[col] = np.nan
            
            if vol_ma_period is not None and vol_freq_raw and vol_freq_raw.lower() != 'none':
                kline_prefix_vol = data_utils.get_kline_prefix_effective(vol_freq_raw)
                vol_source_col = f"{kline_prefix_vol}_volume" if kline_prefix_vol else "volume"
                if vol_source_col in self.df_enriched_slice.columns:
                    volume_series_for_ma = self.df_enriched_slice[vol_source_col].astype(float)
                    if volume_series_for_ma.notna().any():
                        vol_ma_series = ta.sma(volume_series_for_ma.dropna(), length=int(vol_ma_period), append=False) # type: ignore
                        df_for_simulation['Volume_MA_strat'] = vol_ma_series.reindex(df_for_simulation.index, method='ffill') if isinstance(vol_ma_series, pd.Series) else np.nan
                    else: df_for_simulation['Volume_MA_strat'] = np.nan
                else: df_for_simulation['Volume_MA_strat'] = np.nan
            else: df_for_simulation['Volume_MA_strat'] = np.nan

            if rsi_period is not None and rsi_freq_raw and rsi_freq_raw.lower() != 'none':
                kline_prefix_rsi = data_utils.get_kline_prefix_effective(rsi_freq_raw)
                rsi_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'rsi', {'length': int(rsi_period)}, kline_prefix_rsi)
                df_for_simulation['RSI_strat'] = rsi_series.reindex(df_for_simulation.index, method='ffill') if isinstance(rsi_series, pd.Series) else np.nan
            else: df_for_simulation['RSI_strat'] = np.nan
            
            vol_kline_col_strat_expected = f"{data_utils.get_kline_prefix_effective(vol_freq_raw)}_volume" if vol_freq_raw and vol_freq_raw.lower() not in ["1m", "1min", "none"] else "volume"
            if vol_kline_col_strat_expected in self.df_enriched_slice.columns:
                if vol_kline_col_strat_expected not in df_for_simulation.columns: 
                     df_for_simulation[vol_kline_col_strat_expected] = self.df_enriched_slice[vol_kline_col_strat_expected].reindex(df_for_simulation.index, method='ffill')
            else:
                 logger.warning(f"{log_prefix} Expected volume source column '{vol_kline_col_strat_expected}' for BbandsVolumeRsiStrategy not in enriched data.")
                 if vol_kline_col_strat_expected not in df_for_simulation.columns:
                     df_for_simulation[vol_kline_col_strat_expected] = np.nan 
            logger.info(f"{log_prefix} EXITING BbandsVolumeRsiStrategy block. df_for_simulation columns: {df_for_simulation.columns.tolist()}")

        if 'taker_pressure_indicator_period' in trial_params and 'indicateur_frequence_taker_pressure' in trial_params:
            taker_ma_period = int(trial_params['taker_pressure_indicator_period'])
            taker_freq_raw = str(trial_params.get('indicateur_frequence_taker_pressure')) 
            if taker_freq_raw and taker_freq_raw.lower() != 'none':
                kline_prefix_taker_src = data_utils.get_kline_prefix_effective(taker_freq_raw)
                buy_vol_col_src = f"{kline_prefix_taker_src}_taker_buy_base_asset_volume" if kline_prefix_taker_src else "taker_buy_base_asset_volume"
                sell_vol_col_src = f"{kline_prefix_taker_src}_taker_sell_base_asset_volume" if kline_prefix_taker_src else "taker_sell_base_asset_volume"

                if buy_vol_col_src in self.df_enriched_slice.columns and sell_vol_col_src in self.df_enriched_slice.columns:
                    temp_taker_df = self.df_enriched_slice[[buy_vol_col_src, sell_vol_col_src]].copy()
                    temp_taker_df = data_utils.calculate_taker_pressure_ratio(temp_taker_df, buy_vol_col_src, sell_vol_col_src, "TakerPressureRatio_Raw")
                    if "TakerPressureRatio_Raw" in temp_taker_df.columns and temp_taker_df["TakerPressureRatio_Raw"].notna().any():
                        taker_ratio_ma_series = ta.ema(temp_taker_df["TakerPressureRatio_Raw"].dropna(), length=taker_ma_period, append=False) # type: ignore
                        df_for_simulation['TakerPressureRatio_MA_strat'] = taker_ratio_ma_series.reindex(df_for_simulation.index, method='ffill') if isinstance(taker_ratio_ma_series, pd.Series) else np.nan
                    else: df_for_simulation['TakerPressureRatio_MA_strat'] = np.nan
                else: df_for_simulation['TakerPressureRatio_MA_strat'] = np.nan
            else: df_for_simulation['TakerPressureRatio_MA_strat'] = np.nan

        indicator_strat_cols = [col for col in df_for_simulation.columns if col.endswith('_strat')]
        if indicator_strat_cols:
            df_for_simulation[indicator_strat_cols] = df_for_simulation[indicator_strat_cols].ffill()
            logger.debug(f"{log_prefix} Applied ffill to _strat columns: {indicator_strat_cols}")
        
        essential_ohlcv = ['open', 'high', 'low', 'close']
        if df_for_simulation[essential_ohlcv].isnull().all().any(): 
            logger.error(f"{log_prefix} One or more essential OHLCV columns are entirely NaN in df_for_simulation after indicator prep.")
            raise optuna.exceptions.TrialPruned("Essential OHLCV data missing after indicator preparation.")

        logger.info(f"{log_prefix} Data preparation complete. df_for_simulation shape: {df_for_simulation.shape}. Columns: {df_for_simulation.columns.tolist()}")
        return df_for_simulation

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params_for_trial: Dict[str, Any] = {}
        if not self.params_space_details:
            logger.warning(f"Parameter space (params_space_details) is empty for strategy {self.strategy_name_key}. Will use default_params if available, or prune.")
            default_params = self.strategy_config_dict.get('default_params', {})
            if default_params and isinstance(default_params, dict):
                 logger.info(f"Using default_params for strategy {self.strategy_name_key} as params_space is empty: {default_params}")
                 return default_params.copy() 
            raise optuna.exceptions.TrialPruned("Parameter space not defined and no default_params available for the strategy.")

        for param_name, p_detail in self.params_space_details.items():
            if param_name in params_for_trial: continue 
            try:
                if p_detail.type == 'int':
                    params_for_trial[param_name] = trial.suggest_int(param_name, int(p_detail.low), int(p_detail.high), step=int(p_detail.step or 1)) # type: ignore
                elif p_detail.type == 'float':
                    step_val = float(p_detail.step) if p_detail.step is not None else None
                    params_for_trial[param_name] = trial.suggest_float(param_name, float(p_detail.low), float(p_detail.high), step=step_val, log=p_detail.log_scale) # type: ignore
                elif p_detail.type == 'categorical' and p_detail.choices:
                    params_for_trial[param_name] = trial.suggest_categorical(param_name, p_detail.choices)
                else: 
                    logger.warning(f"Unsupported ParamDetail type '{p_detail.type}' or missing choices for '{param_name}'. Using default or low value.")
                    params_for_trial[param_name] = p_detail.default if p_detail.default is not None else p_detail.low

            except Exception as e_suggest: 
                logger.error(f"Error suggesting param '{param_name}' (type: {p_detail.type}, low: {p_detail.low}, high: {p_detail.high}): {e_suggest}", exc_info=True)
                if p_detail.default is not None:
                    params_for_trial[param_name] = p_detail.default
                elif p_detail.low is not None:
                     params_for_trial[param_name] = p_detail.low
                elif p_detail.choices and len(p_detail.choices) > 0:
                    params_for_trial[param_name] = p_detail.choices[0]
                else:
                    logger.critical(f"Cannot fallback for param {param_name}, pruning trial.")
                    raise optuna.exceptions.TrialPruned(f"Failed to suggest or fallback for param {param_name}")
        return params_for_trial

    def __call__(self, trial: optuna.Trial) -> Tuple[float, ...]:
        start_time_trial = time.time()
        trial_id_for_log: str
        trial_number_for_prepare_log: Optional[int] = None
        self.last_backtest_results = None 

        if self.is_oos_eval: 
            is_trial_num_oos = self.is_trial_number_for_oos_log 
            trial_id_for_log = f"OOS_for_IS_Trial_{is_trial_num_oos}" if is_trial_num_oos is not None else f"OOS_Trial_UnknownOrigin_{uuid.uuid4().hex[:6]}"
            trial_number_for_prepare_log = is_trial_num_oos 
        else: 
            trial_id_for_log = str(trial.number) if hasattr(trial, 'number') and trial.number is not None else f"IS_Trial_Unknown_{uuid.uuid4().hex[:6]}"
            if hasattr(trial, 'number'): trial_number_for_prepare_log = trial.number

        log_prefix = f"[{self.strategy_class_name}/{self.pair_symbol}/Trial-{trial_id_for_log}]" 
        logger.info(f"{log_prefix} Starting evaluation...")

        StrategyClass_local: Optional[Type[BaseStrategy]] = None
        try:
            module_path = self.strategy_script_ref 
            if module_path.endswith(".py"): 
                 module_path = module_path.replace('.py', '').replace('/', '.')

            module = importlib.import_module(module_path)
            StrategyClass_local = getattr(module, self.strategy_class_name) # type: ignore
            if not StrategyClass_local or not issubclass(StrategyClass_local, BaseStrategy):
                raise ImportError(f"Could not load a valid BaseStrategy subclass: {self.strategy_class_name} from {module_path}")
        except Exception as e_load_strat:
            logger.error(f"{log_prefix} Failed to load strategy class {self.strategy_class_name} from {self.strategy_script_ref}: {e_load_strat}", exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Strategy class load failure: {e_load_strat}")

        current_trial_params: Dict[str, Any]
        if self.is_oos_eval: 
            current_trial_params = trial.params.copy() 
            logger.info(f"{log_prefix} OOS evaluation using fixed params from IS trial {self.is_trial_number_for_oos_log}: {current_trial_params}")
        else: 
            try:
                current_trial_params = self._suggest_params(trial)
                if not current_trial_params: raise optuna.exceptions.TrialPruned("No parameters suggested.")
                logger.info(f"{log_prefix} IS evaluation with suggested params: {current_trial_params}")
            except optuna.exceptions.TrialPruned: raise
            except Exception as e_suggest:
                logger.error(f"{log_prefix} Error during parameter suggestion: {e_suggest}", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Parameter suggestion failed: {e_suggest}")

        try:
            data_for_simulation = self._prepare_data_with_dynamic_indicators(current_trial_params, trial_number_for_log=trial_number_for_prepare_log)
            if data_for_simulation.empty or data_for_simulation[['open', 'high', 'low', 'close']].isnull().all().all():
                logger.error(f"{log_prefix} Data preparation resulted in unusable (empty or all NaN OHLC) data.")
                raise optuna.exceptions.TrialPruned("Data preparation resulted in unusable data.")
        except optuna.exceptions.TrialPruned: raise
        except Exception as e_prepare:
            logger.error(f"{log_prefix} Error preparing data: {e_prepare}", exc_info=True)
            return tuple([-float('inf') if d == "maximize" else float('inf') for d in self.optuna_objectives_config.get('objectives_directions', ['maximize'])])

        strategy_instance = StrategyClass_local(
            strategy_name=self.strategy_name_key, 
            symbol=self.pair_symbol,
            params=current_trial_params
        )
        strategy_instance.set_backtest_context(
            pair_config=self.symbol_info_data, 
            is_futures=self.app_config.global_config.simulation_defaults.is_futures_trading,
            leverage=current_trial_params.get('margin_leverage', self.app_config.global_config.simulation_defaults.margin_leverage),
            initial_equity=self.app_config.global_config.simulation_defaults.initial_capital
        )
        
        sim_defaults = self.app_config.global_config.simulation_defaults
        
        slippage_config_sim = sim_defaults.slippage_config # Utiliser directement le dict de la config

        simulator = BacktestSimulator(
            df_ohlcv=data_for_simulation,
            strategy_instance=strategy_instance,
            initial_equity=sim_defaults.initial_capital,
            leverage=current_trial_params.get('margin_leverage', sim_defaults.margin_leverage), 
            symbol=self.pair_symbol,
            pair_config=self.symbol_info_data, # <<< PASSÉ ICI
            trading_fee_bps=sim_defaults.trading_fee_bps,
            slippage_config=slippage_config_sim,
            is_futures=sim_defaults.is_futures_trading,
            run_id=self.run_id,             
            is_oos_simulation=self.is_oos_eval, 
            verbosity=0 if not self.is_oos_eval else sim_defaults.backtest_verbosity 
        )

        try:
            trades, equity_curve_df, daily_equity, oos_detailed_log_from_sim = simulator.run_simulation()
            self.last_backtest_results = {
                "trades": trades, "equity_curve_df": equity_curve_df, 
                "daily_equity": daily_equity, "oos_detailed_trades_log": oos_detailed_log_from_sim,
                "metrics": {} 
            }
        except optuna.exceptions.TrialPruned:
            logger.info(f"{log_prefix} Trial pruned during simulation.")
            raise
        except Exception as e_sim:
            logger.error(f"{log_prefix} Error during BacktestSimulator execution: {e_sim}", exc_info=True)
            return tuple([-float('inf') if d == "maximize" else float('inf') for d in self.optuna_objectives_config.get('objectives_directions', ['maximize'])])

        equity_series_for_calc = pd.Series(dtype=float)
        if not equity_curve_df.empty and 'timestamp' in equity_curve_df.columns and 'equity' in equity_curve_df.columns:
            temp_ec_df = equity_curve_df.copy()
            temp_ec_df['timestamp'] = pd.to_datetime(temp_ec_df['timestamp'], errors='coerce', utc=True)
            temp_ec_df.dropna(subset=['timestamp','equity'], inplace=True)
            if not temp_ec_df.empty:
                equity_series_for_calc = temp_ec_df.set_index('timestamp')['equity'].sort_index()
        
        if equity_series_for_calc.empty: 
            logger.warning(f"{log_prefix} Equity series for performance calculation is empty. Using initial capital.")
            equity_series_for_calc = pd.Series([sim_defaults.initial_capital], index=[data_for_simulation.index.min() if not data_for_simulation.empty else pd.Timestamp.now(tz='UTC')])

        metrics: Dict[str, Any]
        if not trades:
            logger.info(f"{log_prefix} No trades made with params {current_trial_params}. Pruning.")
            if not self.is_oos_eval:
                trial.set_user_attr("total_trades", 0)
                trial.set_user_attr("final_equity", sim_defaults.initial_capital)
            metrics = {"Total Trades": 0, "Final Equity USDC": sim_defaults.initial_capital}
            for obj_name in self.optuna_objectives_config.get('objectives_names', []):
                 if obj_name not in metrics: metrics[obj_name] = 0 if "PnL" in obj_name or "Trades" in obj_name else (-10.0 if "Ratio" in obj_name else 0.0)
        else:
            metrics = calculate_performance_metrics_from_inputs(
                trades_df=pd.DataFrame(trades),
                equity_curve_series=equity_series_for_calc,
                initial_capital=sim_defaults.initial_capital,
                risk_free_rate_daily=(1 + self.app_config.global_config.simulation_defaults.risk_free_rate)**(1/252) - 1, 
                periods_per_year=252 
            )
            metrics['Total Trades'] = len(trades)
            metrics['Final Equity USDC'] = equity_series_for_calc.iloc[-1] if not equity_series_for_calc.empty else sim_defaults.initial_capital

        if self.last_backtest_results: 
            self.last_backtest_results["metrics"] = metrics.copy()

        if not self.is_oos_eval: 
            for key, value in metrics.items():
                try:
                    trial.set_user_attr(key, value if pd.notna(value) and not (isinstance(value, float) and (np.isinf(value) or np.isnan(value))) else None)
                except TypeError:
                    trial.set_user_attr(key, str(value) if pd.notna(value) else None) 

        objective_values: List[float] = []
        obj_names = self.optuna_objectives_config.get('objectives_names', [])
        obj_dirs = self.optuna_objectives_config.get('objectives_directions', [])

        if not obj_names or not obj_dirs or len(obj_names) != len(obj_dirs):
            logger.error(f"{log_prefix} objectives_names or objectives_directions misconfigured. Optuna objectives: {self.optuna_objectives_config}")
            default_metric_name = "Sharpe Ratio" 
            if default_metric_name not in metrics: default_metric_name = "Total Net PnL USDC" 
            if default_metric_name not in metrics and obj_names: default_metric_name = obj_names[0] 

            if default_metric_name in metrics:
                 obj_names = [default_metric_name]
                 obj_dirs = ['maximize'] 
                 logger.warning(f"{log_prefix} Falling back to single objective: {default_metric_name} (maximize)")
            else:
                 logger.error(f"{log_prefix} Cannot determine a fallback objective. Pruning.")
                 raise optuna.exceptions.TrialPruned("Objectives configuration error and no fallback objective found.")

        for i, metric_name in enumerate(obj_names):
            value = metrics.get(metric_name)
            direction = obj_dirs[i]
            if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
                logger.warning(f"{log_prefix} Objective '{metric_name}' is invalid (value: {value}). Assigning worst value.")
                if metric_name == "Total Trades" and not trades: value = 0.0
                elif "Ratio" in metric_name and not trades : value = -10.0
                elif ("PnL" in metric_name or "Profit" in metric_name) and not trades: value = 0.0
                else: value = -1e9 if direction == "maximize" else 1e9 
            objective_values.append(float(value))
        
        logger.debug(f"{log_prefix} All metrics for trial: { {k: (f'{v:.4f}' if isinstance(v, float) else v) for k,v in metrics.items()} }")
        end_time_trial = time.time()
        logger.info(f"{log_prefix} Evaluation finished in {end_time_trial - start_time_trial:.2f}s. Objectives ({obj_names}): {objective_values}")
        return tuple(objective_values)

