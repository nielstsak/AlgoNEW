# src/backtesting/wfo.py
import json
import logging
import importlib # Not strictly needed here if classes are imported directly
import time
import traceback # For more detailed error logging if needed
import sys
import argparse # Not used in this module directly, but good for context
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, TYPE_CHECKING

import numpy as np
import pandas as pd

# Conditional imports for type hinting
if TYPE_CHECKING:
    from src.config.definitions import AppConfig, WfoSettings, GlobalConfig # Corrected case for WfoSettings
    # Function import for run_optimization_for_fold
    from src.backtesting.optimizer.optimization_orchestrator import run_optimization_for_fold
    # Class imports for type hints
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator
    from src.backtesting.optimizer.study_manager import StudyManager
    from src.backtesting.optimizer.results_analyzer import ResultsAnalyzer
    from src.live.execution import OrderExecutionClient


# Actual imports needed at runtime
from src.config.definitions import WfoSettings # CORRECTED IMPORT CASE
from src.backtesting.optimizer.optimization_orchestrator import run_optimization_for_fold
from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator
from src.backtesting.optimizer.study_manager import StudyManager
from src.backtesting.optimizer.results_analyzer import ResultsAnalyzer
from src.live.execution import OrderExecutionClient # For symbol_info
# from src.strategies.base import BaseStrategy # Not directly instantiated here
# from src.config.loader import load_all_configs # AppConfig is passed in

logger = logging.getLogger(__name__)

def _sanitize_filename_component(name: str) -> str:
    """
    Sanitizes a string component to be safe for use in file or directory names.
    Removes or replaces characters that are typically invalid in file paths.
    """
    if not name:
        return "default_component"
    sanitized_name = re.sub(r'[^\w\-.]', '_', name)
    sanitized_name = sanitized_name.strip('_')
    return sanitized_name if sanitized_name else "sanitized_default"


class WFOGenerator:
    """
    Generates In-Sample (IS) and Out-of-Sample (OOS) data folds for Walk-Forward Optimization.
    """
    def __init__(self, wfo_settings: WfoSettings): # Corrected type hint case
        """
        Initializes the WFOGenerator.

        Args:
            wfo_settings (WfoSettings): Configuration for Walk-Forward Optimization.
        """
        self.wfo_settings = wfo_settings

    def _validate_data_and_settings(self,
                                    df_enriched_data: pd.DataFrame,
                                    is_total_start_ts: pd.Timestamp,
                                    oos_total_end_ts: pd.Timestamp):
        """Validates input data and WFO settings."""
        if not isinstance(df_enriched_data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        
        if is_total_start_ts >= oos_total_end_ts:
            raise ValueError(f"Total start timestamp {is_total_start_ts} must be before total end timestamp {oos_total_end_ts}.")

        if is_total_start_ts > df_enriched_data.index.max() or oos_total_end_ts < df_enriched_data.index.min():
             logger.warning(f"Specified WFO range {is_total_start_ts} to {oos_total_end_ts} might be outside data range {df_enriched_data.index.min()} to {df_enriched_data.index.max()}.")

        # Check for oos_period_days and min_is_period_days in wfo_settings
        if not hasattr(self.wfo_settings, 'oos_period_days') or not hasattr(self.wfo_settings, 'min_is_period_days'):
            raise AttributeError("WfoSettings object is missing 'oos_period_days' or 'min_is_period_days'. Check config/definitions.py.")

        oos_duration_td = pd.Timedelta(self.wfo_settings.oos_period_days, unit='D')
        min_is_duration_td = pd.Timedelta(self.wfo_settings.min_is_period_days, unit='D')

        if (oos_total_end_ts - is_total_start_ts) < (oos_duration_td + min_is_duration_td):
            raise ValueError(f"Total data duration from {is_total_start_ts} to {oos_total_end_ts} is too short for the specified OOS ({self.wfo_settings.oos_period_days} days) and minimum IS ({self.wfo_settings.min_is_period_days} days) periods.")
        
        if self.wfo_settings.n_splits <= 0:
            raise ValueError("Number of splits (n_splits) must be positive.")


    def _get_expanding_folds(self,
                             df_enriched_data: pd.DataFrame,
                             is_total_start_ts: pd.Timestamp, 
                             oos_total_end_ts: pd.Timestamp,  
                             n_splits: int
                            ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generates expanding In-Sample folds with a fixed Out-of-Sample period at the end.
        The IS window grows by adding earlier data, keeping its end date fixed before the OOS period.
        """
        folds = []
        oos_duration_td = pd.Timedelta(self.wfo_settings.oos_period_days, unit='D')

        approx_oos_start_ts = oos_total_end_ts - oos_duration_td
        
        oos_start_pos = df_enriched_data.index.searchsorted(approx_oos_start_ts, side='left')
        if oos_start_pos == len(df_enriched_data.index) or df_enriched_data.index[oos_start_pos] > oos_total_end_ts:
            logger.error(f"Cannot determine a valid OOS start. Approx OOS start: {approx_oos_start_ts}, Data End: {oos_total_end_ts}, Data last ts: {df_enriched_data.index[-1] if not df_enriched_data.empty else 'N/A'}")
            return []
        actual_oos_start_ts = df_enriched_data.index[oos_start_pos]
        
        if oos_start_pos == 0:
            logger.error(f"OOS period starts at the beginning of the dataset ({actual_oos_start_ts}). No space for IS data.")
            return []
        
        d_timestamp_is_actual_end = df_enriched_data.index[oos_start_pos - 1]

        if d_timestamp_is_actual_end < is_total_start_ts:
            logger.error(f"Calculated IS end {d_timestamp_is_actual_end} is before WFO total start {is_total_start_ts}.")
            return []

        df_oos_fixed_enriched = df_enriched_data.loc[actual_oos_start_ts : oos_total_end_ts]
        if df_oos_fixed_enriched.empty:
            logger.error(f"Fixed OOS dataframe is empty. OOS Start: {actual_oos_start_ts}, OOS End: {oos_total_end_ts}")
            return []
        actual_oos_end_ts_from_df = df_oos_fixed_enriched.index[-1]

        df_is_total_enriched = df_enriched_data.loc[is_total_start_ts : d_timestamp_is_actual_end]
        if df_is_total_enriched.empty:
            logger.error(f"Total IS dataframe is empty. WFO IS Total Start: {is_total_start_ts}, Actual IS End (Tf'): {d_timestamp_is_actual_end}")
            return []
        
        actual_is_total_start_ts_from_df = df_is_total_enriched.index[0] 
        is_total_duration_td = d_timestamp_is_actual_end - actual_is_total_start_ts_from_df
        
        min_is_duration_from_settings_td = pd.Timedelta(self.wfo_settings.min_is_period_days, unit='D')

        if is_total_duration_td < min_is_duration_from_settings_td:
            logger.error(f"Total available IS duration ({is_total_duration_td}) from {actual_is_total_start_ts_from_df} to {d_timestamp_is_actual_end} is less than min_is_period_days ({self.wfo_settings.min_is_period_days} days).")
            return []
        if is_total_duration_td.total_seconds() <= 0:
             logger.error(f"Total available IS duration ({is_total_duration_td}) is zero or negative.")
             return []

        segment_duration_td = is_total_duration_td / n_splits if n_splits > 0 else is_total_duration_td

        for i in range(n_splits):
            fold_idx = i 
            num_segments_current_fold = i + 1
            current_fold_is_end_ts_actual = d_timestamp_is_actual_end 

            current_fold_duration_td = num_segments_current_fold * segment_duration_td
            if current_fold_duration_td > is_total_duration_td: 
                current_fold_duration_td = is_total_duration_td
            
            current_fold_is_start_ts_approx = d_timestamp_is_actual_end - current_fold_duration_td
            
            start_pos = df_is_total_enriched.index.searchsorted(current_fold_is_start_ts_approx, side='right')
            if start_pos == 0: 
                current_fold_is_start_ts_actual = actual_is_total_start_ts_from_df
            else:
                current_fold_is_start_ts_actual = df_is_total_enriched.index[start_pos -1]

            if current_fold_is_start_ts_actual < actual_is_total_start_ts_from_df:
                 current_fold_is_start_ts_actual = actual_is_total_start_ts_from_df
            
            if current_fold_is_start_ts_actual > current_fold_is_end_ts_actual:
                logger.warning(f"Fold {fold_idx}: Calculated IS start {current_fold_is_start_ts_actual} is after IS end {current_fold_is_end_ts_actual}. Adjusting start to end.")
                current_fold_is_start_ts_actual = current_fold_is_end_ts_actual

            df_is_enriched_fold = df_is_total_enriched.loc[current_fold_is_start_ts_actual : current_fold_is_end_ts_actual]

            if df_is_enriched_fold.empty:
                logger.warning(f"Fold {fold_idx}: In-Sample data is empty. Start: {current_fold_is_start_ts_actual}, End: {current_fold_is_end_ts_actual}. Skipping this fold.")
                continue
            
            current_is_fold_duration_actual = df_is_enriched_fold.index[-1] - df_is_enriched_fold.index[0]
            if current_is_fold_duration_actual < min_is_duration_from_settings_td and n_splits > 1 :
                 logger.warning(f"Fold {fold_idx}: Actual IS duration {current_is_fold_duration_actual} is less than min_is_period_days ({self.wfo_settings.min_is_period_days} days). This fold might be too short.")

            folds.append((
                df_is_enriched_fold.copy(),
                df_oos_fixed_enriched.copy(),
                fold_idx,
                df_is_enriched_fold.index[0], 
                df_is_enriched_fold.index[-1],
                df_oos_fixed_enriched.index[0],
                actual_oos_end_ts_from_df 
            ))
            logger.debug(f"Fold {fold_idx}: IS [{folds[-1][3]} to {folds[-1][4]}], OOS [{folds[-1][5]} to {folds[-1][6]}]")

        if not folds and n_splits > 0:
            logger.error("No valid folds were generated. Check data range, n_splits, and period settings.")
        return folds

    def generate_folds(self,
                       df_enriched_data: pd.DataFrame,
                       is_total_start_ts_config: Optional[pd.Timestamp] = None, 
                       oos_total_end_ts_config: Optional[pd.Timestamp] = None    
                      ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        if df_enriched_data.empty:
            logger.error("Input DataFrame df_enriched_data is empty. Cannot generate folds.")
            return []
        if not isinstance(df_enriched_data.index, pd.DatetimeIndex):
            logger.error("df_enriched_data must have a DatetimeIndex.")
            return []
        
        data_min_ts = df_enriched_data.index.min()
        data_max_ts = df_enriched_data.index.max()

        _is_total_start_ts = is_total_start_ts_config if is_total_start_ts_config else data_min_ts
        _oos_total_end_ts = oos_total_end_ts_config if oos_total_end_ts_config else data_max_ts
        
        data_tz = df_enriched_data.index.tz
        if data_tz: 
            if _is_total_start_ts.tzinfo is None:
                _is_total_start_ts = _is_total_start_ts.tz_localize(data_tz)
            elif _is_total_start_ts.tzinfo != data_tz:
                _is_total_start_ts = _is_total_start_ts.tz_convert(data_tz)

            if _oos_total_end_ts.tzinfo is None:
                _oos_total_end_ts = _oos_total_end_ts.tz_localize(data_tz)
            elif _oos_total_end_ts.tzinfo != data_tz:
                _oos_total_end_ts = _oos_total_end_ts.tz_convert(data_tz)
        else: 
            if _is_total_start_ts.tzinfo is not None:
                _is_total_start_ts = _is_total_start_ts.tz_localize(None)
            if _oos_total_end_ts.tzinfo is not None:
                _oos_total_end_ts = _oos_total_end_ts.tz_localize(None)
        
        effective_wfo_start_ts = max(_is_total_start_ts, data_min_ts)
        effective_wfo_end_ts = min(_oos_total_end_ts, data_max_ts)

        if effective_wfo_start_ts >= effective_wfo_end_ts:
            logger.error(f"Effective WFO period is invalid after clipping to data range. Effective Start: {effective_wfo_start_ts}, Effective End: {effective_wfo_end_ts}")
            return []
            
        df_for_wfo = df_enriched_data.loc[effective_wfo_start_ts:effective_wfo_end_ts]
        if df_for_wfo.empty:
            logger.error(f"DataFrame for WFO is empty after slicing from {effective_wfo_start_ts} to {effective_wfo_end_ts}.")
            return []

        self._validate_data_and_settings(df_for_wfo, df_for_wfo.index.min(), df_for_wfo.index.max())

        n_splits = self.wfo_settings.n_splits
        fold_type = self.wfo_settings.fold_type if hasattr(self.wfo_settings, 'fold_type') else "expanding" # Default to expanding

        logger.info(f"Generating {n_splits} {fold_type} WFO folds using data from {df_for_wfo.index.min()} to {df_for_wfo.index.max()}.")
        logger.info(f"OOS period: {self.wfo_settings.oos_period_days} days. Min IS period: {self.wfo_settings.min_is_period_days} days.")

        if fold_type == "expanding": 
            return self._get_expanding_folds(df_for_wfo, df_for_wfo.index.min(), df_for_wfo.index.max(), n_splits)
        else:
            raise ValueError(f"Unsupported fold_type: {fold_type}. Currently, only 'expanding' (with fixed OOS end) is implemented with the new logic.")


class WalkForwardOptimizer:
    def __init__(self, app_config: 'AppConfig'):
        self.app_config: 'AppConfig' = app_config
        self.global_config_obj: 'GlobalConfig' = self.app_config.global_config
        self.wfo_settings: 'WfoSettings' = self.global_config_obj.wfo_settings # Corrected case
        
        self.strategies_config_dict: Dict[str, Dict[str, Any]] = {}
        if hasattr(self.app_config.strategies_config, 'strategies') and \
           isinstance(self.app_config.strategies_config.strategies, dict):
            for name, strategy_params_obj in self.app_config.strategies_config.strategies.items():
                # Assuming StrategyParams is used here
                if hasattr(strategy_params_obj, '__dict__'): # If it's a Pydantic model instance
                    self.strategies_config_dict[name] = strategy_params_obj.__dict__
                elif isinstance(strategy_params_obj, dict): # If it's already a dict
                    self.strategies_config_dict[name] = strategy_params_obj
        else:
            logger.error("AppConfig.strategies_config.strategies is not a dictionary or is missing.")

        self.paths_config: Dict[str, Any] = self.global_config_obj.paths.__dict__
        # self.simulation_defaults: Dict[str, Any] = self.global_config_obj.simulation_defaults.__dict__ # Not directly used here now

        run_timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        logs_opt_path_str = self.paths_config.get('logs_backtest_optimization', 'logs/backtest_optimization')
        
        run_name_prefix = self.global_config_obj.simulation_defaults.run_name_prefix \
            if hasattr(self.global_config_obj.simulation_defaults, 'run_name_prefix') else "opt"
            
        self.run_id = f"{run_name_prefix}_{run_timestamp_str}"
        self.run_output_dir = Path(logs_opt_path_str) / self.run_id
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"WFO run output directory created: {self.run_output_dir} (Run ID: {self.run_id})")

        selected_account_for_client: Optional[Any] = None
        if self.app_config.accounts_config and self.app_config.api_keys:
            try:
                selected_account_for_client = next(
                    acc for acc in self.app_config.accounts_config 
                    if acc.exchange.lower() == 'binance' and not acc.is_testnet
                )
            except StopIteration:
                try: 
                    selected_account_for_client = next(
                        acc for acc in self.app_config.accounts_config if acc.exchange.lower() == 'binance'
                    )
                except StopIteration:
                    selected_account_for_client = None
        
        self.execution_client: Optional[OrderExecutionClient] = None
        if selected_account_for_client:
            credentials = self.app_config.api_keys.credentials.get(selected_account_for_client.account_alias)
            if credentials and len(credentials) == 2:
                api_key_val, api_secret_val = credentials
                self.execution_client = OrderExecutionClient(
                    api_key=api_key_val,
                    api_secret=api_secret_val,
                    account_type=selected_account_for_client.account_type,
                    is_testnet=selected_account_for_client.is_testnet
                )
                if not self.execution_client.test_connection():
                    logger.warning("Failed Binance REST API connection during WFO init. Symbol info might fail.")
            else:
                logger.warning(f"Credentials not found or incomplete for account alias {selected_account_for_client.account_alias}.")
        else:
            logger.warning("No suitable 'binance' account found or configured for OrderExecutionClient initialization. Symbol info might fail.")


    def run(self, pairs: List[str], context_labels: List[str]) -> Dict[str, Any]:
        all_wfo_run_results: Dict[str, Any] = {}
        main_run_log_prefix = f"[WFO Run ID: {self.run_id}]"
        logger.info(f"{main_run_log_prefix} Starting WFO for pairs: {pairs}, context_labels: {context_labels}")

        wfo_generator = WFOGenerator(self.wfo_settings) 

        for pair_symbol in pairs:
            current_context_label_raw = context_labels[0] if context_labels else "default_wfo_context"
            current_context_label_sanitized = _sanitize_filename_component(current_context_label_raw)
            
            pair_log_prefix = f"{main_run_log_prefix}[Pair: {pair_symbol}][Ctx: {current_context_label_sanitized}]"
            logger.info(f"{pair_log_prefix} Processing pair. Raw context: '{current_context_label_raw}', Sanitized: '{current_context_label_sanitized}'")

            symbol_info_data: Optional[Dict[str, Any]] = None
            # Try to get symbol_info_data (pair_config)
            if self.execution_client:
                try:
                    symbol_info_data = self.execution_client.get_symbol_info(pair_symbol)
                except Exception as e_sym_info:
                    logger.error(f"{pair_log_prefix} Error getting symbol_info via API: {e_sym_info}.")
            
            if not symbol_info_data: # Fallback to local config if API fails or not available
                logger.warning(f"{pair_log_prefix} ExecutionClient failed or not initialized for symbol_info. Attempting local config.")
                from src.config.loader import load_exchange_config # Ensure this can load your exchange config structure
                from src.utils.exchange_utils import get_pair_config_for_symbol as get_pair_cfg
                try:
                    # Assuming exchange_settings are part of app_config
                    exchange_settings = self.app_config.exchange_settings
                    if exchange_settings and hasattr(exchange_settings, 'exchange_info_file_path'):
                         symbol_info_data = get_pair_cfg(pair_symbol, exchange_settings.exchange_info_file_path) 
                    if not symbol_info_data:
                         logger.error(f"{pair_log_prefix} Could not get symbol_info from local config for {pair_symbol}. Skipping.")
                         continue
                except Exception as e_local_sym:
                    logger.error(f"{pair_log_prefix} Error getting symbol_info from local config: {e_local_sym}. Skipping.")
                    continue
            
            enriched_data_dir_path_str = self.app_config.global_config.paths.data_historical_processed_enriched
            if not enriched_data_dir_path_str:
                logger.error(f"{pair_log_prefix} Path to enriched data not configured. Skipping pair.")
                continue
            
            enriched_data_dir_path = Path(enriched_data_dir_path_str)
            enriched_filename = f"{pair_symbol}_enriched.parquet" 
            enriched_filepath = enriched_data_dir_path / enriched_filename

            if not enriched_filepath.exists():
                logger.error(f"{pair_log_prefix} Enriched data file not found: {enriched_filepath}. Skipping pair.")
                continue
            
            try:
                logger.debug(f"{pair_log_prefix} Loading enriched data from: {enriched_filepath}")
                data_enriched_full = pd.read_parquet(enriched_filepath)
                if 'timestamp' not in data_enriched_full.columns and not isinstance(data_enriched_full.index, pd.DatetimeIndex):
                    logger.error(f"{pair_log_prefix} Loaded data from {enriched_filepath} needs a 'timestamp' column or DatetimeIndex.")
                    continue
                if 'timestamp' in data_enriched_full.columns:
                    data_enriched_full['timestamp'] = pd.to_datetime(data_enriched_full['timestamp'], utc=True, errors='coerce')
                    data_enriched_full.dropna(subset=['timestamp'], inplace=True)
                    data_enriched_full = data_enriched_full.set_index('timestamp')
                
                if data_enriched_full.index.tz is None: 
                    data_enriched_full.index = data_enriched_full.index.tz_localize('UTC')
                elif data_enriched_full.index.tz.utcoffset(data_enriched_full.index[0]) != timezone.utc.utcoffset(data_enriched_full.index[0]): # type: ignore
                       data_enriched_full.index = data_enriched_full.index.tz_convert('UTC')
                
                data_enriched_full.sort_index(inplace=True)
                if not data_enriched_full.index.is_unique:
                    logger.warning(f"{pair_log_prefix} Duplicate timestamps found in {enriched_filepath}. Keeping first.")
                    data_enriched_full = data_enriched_full[~data_enriched_full.index.duplicated(keep='first')]
                logger.info(f"{pair_log_prefix} Loaded and preprocessed enriched data. Shape: {data_enriched_full.shape}, Range: {data_enriched_full.index.min()} to {data_enriched_full.index.max()}")

            except Exception as e_load:
                logger.error(f"{pair_log_prefix} Failed to load or process enriched file {enriched_filepath}: {e_load}", exc_info=True)
                continue
            
            active_strategies = {
                name: cfg_dict for name, cfg_dict in self.strategies_config_dict.items()
                if isinstance(cfg_dict, dict) and cfg_dict.get('active_for_optimization', False)
            }

            if not active_strategies:
                logger.warning(f"{pair_log_prefix} No strategies marked 'active_for_optimization'. Skipping pair.")
                continue

            for strat_name, strat_config_dict_for_opt in active_strategies.items():
                strat_log_prefix = f"{pair_log_prefix}[Strategy: {strat_name}]"
                logger.info(f"{strat_log_prefix} Processing strategy.")

                strategy_pair_context_output_dir = self.run_output_dir / strat_name / pair_symbol / current_context_label_sanitized
                strategy_pair_context_output_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"{strat_log_prefix} Output directory for this combo: {strategy_pair_context_output_dir}")
                
                fold_summaries: List[Dict[str, Any]] = []
                
                try:
                    folds_generator = wfo_generator.generate_folds(
                        df_enriched_data=data_enriched_full
                        # Pass is_total_start_ts_config and oos_total_end_ts_config if defined in WFO settings
                        # e.g., is_total_start_ts_config=pd.Timestamp(self.wfo_settings.global_is_start_date, tz='UTC') if hasattr(self.wfo_settings, 'global_is_start_date') else None
                    )
                    
                    for df_is_enriched_fold, df_oos_fixed_enriched_fold, fold_idx, start_is, end_is, start_oos, end_oos in folds_generator:
                        fold_specific_log_prefix = f"{strat_log_prefix}[Fold_{fold_idx}]"
                        logger.info(f"{fold_specific_log_prefix} Processing fold.")
                        
                        fold_artifacts_path = strategy_pair_context_output_dir / f"fold_{fold_idx}"
                        fold_artifacts_path.mkdir(parents=True, exist_ok=True)
                        
                        final_params_for_fold: Optional[Dict[str, Any]] = None
                        representative_oos_metrics_fold: Optional[Dict[str, Any]] = None
                        fold_status = "OPTIMIZATION_ATTEMPTED"
                        
                        try:
                            # Ensure strategy_config_dict_for_opt contains necessary fields like 'script_reference', 'class_name', 'params_space'
                            if not all(k in strat_config_dict_for_opt for k in ['script_reference', 'class_name']):
                                logger.error(f"{fold_specific_log_prefix} strategy_config_dict for {strat_name} is missing 'script_reference' or 'class_name'. Skipping fold.")
                                fold_status = "CONFIG_ERROR_MISSING_REF_CLASS"
                                continue # Skip to next fold

                            final_params_for_fold, representative_oos_metrics_fold = run_optimization_for_fold(
                                strategy_name=strat_name, # This is the key from strategies_config
                                strategy_config_dict=strat_config_dict_for_opt, 
                                data_1min_cleaned_is_slice=df_is_enriched_fold,
                                data_1min_cleaned_oos_slice=df_oos_fixed_enriched_fold,
                                app_config=self.app_config, 
                                output_dir_fold=fold_artifacts_path, 
                                pair_symbol=pair_symbol,
                                symbol_info_data=symbol_info_data, # This is the pair_config
                                objective_evaluator_class=ObjectiveEvaluator,
                                study_manager_class=StudyManager,
                                results_analyzer_class=ResultsAnalyzer,
                                run_id=self.run_id 
                            )
                            if final_params_for_fold:
                                fold_status = "COMPLETED_WITH_PARAMS"
                            else:
                                fold_status = "COMPLETED_NO_PARAMS"
                            logger.info(f"{fold_specific_log_prefix} Optimization for fold finished. Status: {fold_status}")
                        except Exception as e_opt_fold:
                            logger.error(f"{fold_specific_log_prefix} Error during run_optimization_for_fold: {e_opt_fold}", exc_info=True)
                            fold_status = "OPTIMIZATION_FAILED_EXCEPTION"

                        fold_summary_entry = {
                            "fold_index": fold_idx,
                            "status": fold_status,
                            "is_period_start": start_is.isoformat() if pd.notna(start_is) else None,
                            "is_period_end": end_is.isoformat() if pd.notna(end_is) else None,
                            "oos_period_start": start_oos.isoformat() if pd.notna(start_oos) else None,
                            "oos_period_end": end_oos.isoformat() if pd.notna(end_oos) else None,
                            "selected_params_for_fold": final_params_for_fold, 
                            "representative_oos_metrics": representative_oos_metrics_fold
                        }
                        fold_summaries.append(fold_summary_entry)
                
                except Exception as e_fold_loop:
                    logger.error(f"{strat_log_prefix} Error in WFO folds loop: {e_fold_loop}", exc_info=True)
                
                wfo_summary_for_strategy_pair_context = {
                    "strategy_name": strat_name,
                    "pair_symbol": pair_symbol,
                    "context_label": current_context_label_sanitized,
                    "raw_context_label_input": current_context_label_raw,
                    "wfo_run_id": self.run_id, 
                    "folds_data": fold_summaries
                }
                
                summary_file_path = strategy_pair_context_output_dir / "wfo_strategy_pair_summary.json"
                try:
                    with open(summary_file_path, 'w', encoding='utf-8') as f_sum:
                        json.dump(wfo_summary_for_strategy_pair_context, f_sum, indent=4, default=str)
                    logger.info(f"{strat_log_prefix} WFO summary for strategy/pair/context saved to: {summary_file_path}")
                except Exception as e_save_final:
                    logger.error(f"{strat_log_prefix} Failed to save final WFO summary: {e_save_final}", exc_info=True)
                
                all_wfo_run_results[f"{strat_name}_{pair_symbol}_{current_context_label_sanitized}"] = wfo_summary_for_strategy_pair_context
        
        logger.info(f"{main_run_log_prefix} WFO processing finished for all configured pairs and strategies.")
        return all_wfo_run_results

