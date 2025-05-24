import json
import logging
import importlib # Not strictly needed here if classes are imported directly
import time
import traceback # For more detailed error logging if needed
import sys
import argparse # Not used in this module directly, but good for context
import math
import re # <<< IMPORT AJOUTÉ ICI
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, TYPE_CHECKING

import numpy as np
import pandas as pd

# Conditional imports for type hinting
if TYPE_CHECKING:
    from src.config.definitions import AppConfig, WfoSettings, GlobalConfig
    # Function import for run_optimization_for_fold
    from src.backtesting.optimizer.optimization_orchestrator import run_optimization_for_fold
    # Class imports for type hints
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator
    from src.backtesting.optimizer.study_manager import StudyManager
    from src.backtesting.optimizer.results_analyzer import ResultsAnalyzer
    from src.live.execution import OrderExecutionClient


# Actual imports needed at runtime
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
    # Replace common invalid characters with an underscore
    # Windows invalid chars: < > : " / \ | ? *
    # Linux/macOS invalid chars: / (and null byte)
    # Keep it simple: replace non-alphanumeric (excluding underscore and hyphen) with underscore
    sanitized_name = re.sub(r'[^\w\-.]', '_', name)
    # Remove leading/trailing underscores that might result from replacement
    sanitized_name = sanitized_name.strip('_')
    # Ensure it's not empty after sanitization
    return sanitized_name if sanitized_name else "sanitized_default"


def _get_expanding_folds(
    df_enriched: pd.DataFrame,
    n_splits: int,
    oos_percent: float
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], None, None]:
    """
    Generates expanding In-Sample (IS) folds and a fixed Out-of-Sample (OOS) period.

    The OOS period is fixed at the end of the dataset.
    IS periods start at the beginning of the dataset and expand towards the start of the OOS period.

    Args:
        df_enriched: DataFrame with DatetimeIndex (UTC, sorted, unique), containing all historical data.
        n_splits: Number of segments to divide the total In-Sample period into.
                  The WFO will run `n_splits` times, with IS periods of 1/n, 2/n, ..., n/n of the total IS data.
        oos_percent: Percentage of the total data to be used as the fixed OOS period (e.g., 30 for 30%).

    Yields:
        Tuple containing:
            - df_is_fold: DataFrame for the current In-Sample fold.
            - df_oos_fixed: DataFrame for the fixed Out-of-Sample period.
            - fold_index: Index of the current fold (0 to n_splits-1).
            - is_start_ts: Start timestamp of the current IS fold.
            - is_end_ts: End timestamp of the current IS fold (this is fixed for all IS folds).
            - oos_start_ts: Start timestamp of the fixed OOS period.
            - oos_end_ts: End timestamp of the fixed OOS period.
    """
    log_prefix = "[_get_expanding_folds]"
    logger.info(f"{log_prefix} Generating expanding folds. n_splits={n_splits}, oos_percent={oos_percent}%.")

    # --- 1. Validations Initiales ---
    if df_enriched.empty:
        logger.error(f"{log_prefix} Input DataFrame df_enriched is empty. Cannot generate folds.")
        return
    if not isinstance(df_enriched.index, pd.DatetimeIndex):
        logger.error(f"{log_prefix} df_enriched must have a DatetimeIndex.")
        return
    if df_enriched.index.tz is None or df_enriched.index.tz.utcoffset(df_enriched.index[0]) != timezone.utc.utcoffset(df_enriched.index[0]): # type: ignore
        logger.error(f"{log_prefix} df_enriched index must be timezone-aware and UTC.")
        # Attempt to convert, but this should ideally be handled upstream
        try:
            if df_enriched.index.tz is None:
                df_enriched.index = df_enriched.index.tz_localize('UTC')
            else:
                df_enriched.index = df_enriched.index.tz_convert('UTC')
            logger.warning(f"{log_prefix} Attempted to convert df_enriched index to UTC.")
        except Exception as e_tz:
            logger.error(f"{log_prefix} Failed to convert df_enriched index to UTC: {e_tz}")
            return
            
    if not df_enriched.index.is_monotonic_increasing:
        logger.warning(f"{log_prefix} df_enriched index is not monotonic increasing. Sorting...")
        df_enriched = df_enriched.sort_index()
    if not df_enriched.index.is_unique:
        logger.warning(f"{log_prefix} df_enriched index has duplicate timestamps. Keeping first occurrence...")
        df_enriched = df_enriched[~df_enriched.index.duplicated(keep='first')]

    if not (0 < oos_percent < 100):
        logger.error(f"{log_prefix} oos_percent ({oos_percent}) must be between 0 and 100 (exclusive).")
        return
    if n_splits < 1:
        logger.error(f"{log_prefix} n_splits ({n_splits}) must be at least 1.")
        return

    n_total_points = len(df_enriched)
    min_points_for_split = n_splits + 1 # Need at least one point per segment + OOS
    if n_total_points < min_points_for_split :
        logger.error(f"{log_prefix} Not enough data ({n_total_points} points) for {n_splits} splits and OOS period. Need at least {min_points_for_split}.")
        return

    # --- 2. Définition de la Période OOS Fixe ---
    a_timestamp = df_enriched.index.min()
    e_timestamp = df_enriched.index.max()
    total_duration_seconds = (e_timestamp - a_timestamp).total_seconds()

    if total_duration_seconds <= 0:
        logger.error(f"{log_prefix} Total duration of the dataset is not positive ({total_duration_seconds}s). Cannot split.")
        return

    oos_duration_seconds = total_duration_seconds * (oos_percent / 100.0)
    
    approx_is_total_end_ts = e_timestamp - timedelta(seconds=oos_duration_seconds)
    
    possible_is_end_timestamps = df_enriched.index[df_enriched.index <= approx_is_total_end_ts]
    if possible_is_end_timestamps.empty:
        logger.error(f"{log_prefix} OOS percentage ({oos_percent}%) is too high or dataset too short. "
                     f"No data points available for the In-Sample period before OOS cutoff ({approx_is_total_end_ts}).")
        return
    d_timestamp_is_actual_end = possible_is_end_timestamps.max()
    logger.debug(f"{log_prefix} Total IS period ends at: {d_timestamp_is_actual_end}")

    oos_start_candidates = df_enriched.index[df_enriched.index > d_timestamp_is_actual_end]
    if oos_start_candidates.empty:
        logger.warning(f"{log_prefix} No data available for OOS period after {d_timestamp_is_actual_end}. OOS period will be empty.")
        df_oos_fixed_enriched = pd.DataFrame(columns=df_enriched.columns, index=pd.DatetimeIndex([], tz='UTC', name='timestamp'))
    else:
        actual_start_oos_ts = oos_start_candidates.min()
        df_oos_fixed_enriched = df_enriched.loc[actual_start_oos_ts:].copy() 
    
    start_oos_fixed_ts = df_oos_fixed_enriched.index.min() if not df_oos_fixed_enriched.empty else pd.NaT
    end_oos_fixed_ts = df_oos_fixed_enriched.index.max() if not df_oos_fixed_enriched.empty else pd.NaT
    logger.info(f"{log_prefix} Fixed OOS period defined from {start_oos_fixed_ts} to {end_oos_fixed_ts} ({len(df_oos_fixed_enriched)} rows).")

    # --- 3. Définition de la Période IS Totale ---
    df_is_total_enriched = df_enriched.loc[:d_timestamp_is_actual_end]
    if df_is_total_enriched.empty:
        logger.error(f"{log_prefix} Total In-Sample period is empty after OOS split (IS end: {d_timestamp_is_actual_end}). Cannot generate IS folds.")
        return
    
    is_total_start_ts = df_is_total_enriched.index.min()
    is_total_duration_td = d_timestamp_is_actual_end - is_total_start_ts
    logger.debug(f"{log_prefix} Total IS period from {is_total_start_ts} to {d_timestamp_is_actual_end} (Duration: {is_total_duration_td}).")

    if is_total_duration_td.total_seconds() <= 0 and n_splits > 0:
        logger.error(f"{log_prefix} Total IS duration is not positive ({is_total_duration_td}). Cannot create {n_splits} IS segments.")
        return
    
    segment_duration_td = is_total_duration_td / n_splits if n_splits > 0 else is_total_duration_td
    if n_splits == 0: 
        logger.error(f"{log_prefix} n_splits is 0, which is invalid for fold generation.")
        return

    for k_segments_in_fold in range(1, n_splits + 1):
        fold_idx_wfo_convention = k_segments_in_fold - 1 
        actual_current_is_fold_start_ts = is_total_start_ts 
        approx_current_is_fold_end_ts = is_total_start_ts + (k_segments_in_fold * segment_duration_td)
        
        if approx_current_is_fold_end_ts > d_timestamp_is_actual_end:
            approx_current_is_fold_end_ts = d_timestamp_is_actual_end
        
        idx_pos_end = df_is_total_enriched.index.searchsorted(approx_current_is_fold_end_ts, side='right') -1
        if idx_pos_end < 0 : 
            logger.warning(f"{log_prefix} Fold {fold_idx_wfo_convention}: Approx IS end {approx_current_is_fold_end_ts} is before any IS data. Skipping.")
            continue
        actual_current_is_fold_end_ts = df_is_total_enriched.index[idx_pos_end]

        if actual_current_is_fold_start_ts > actual_current_is_fold_end_ts:
            logger.warning(f"{log_prefix} Fold {fold_idx_wfo_convention}: Calculated IS start {actual_current_is_fold_start_ts} is after IS end {actual_current_is_fold_end_ts}. Skipping fold.")
            continue

        df_is_enriched_fold = df_is_total_enriched.loc[actual_current_is_fold_start_ts : actual_current_is_fold_end_ts]

        if df_is_enriched_fold.empty:
            logger.warning(f"{log_prefix} Fold {fold_idx_wfo_convention}: IS data slice is empty for period {actual_current_is_fold_start_ts} to {actual_current_is_fold_end_ts}. Skipping.")
            continue
        
        logger.info(f"{log_prefix} Yielding Fold {fold_idx_wfo_convention}: "
                    f"IS [{actual_current_is_fold_start_ts} to {actual_current_is_fold_end_ts}] ({len(df_is_enriched_fold)} rows), "
                    f"OOS [{start_oos_fixed_ts} to {end_oos_fixed_ts}] ({len(df_oos_fixed_enriched)} rows).")
        
        yield (df_is_enriched_fold.copy(), 
               df_oos_fixed_enriched.copy(), 
               fold_idx_wfo_convention, 
               actual_current_is_fold_start_ts, 
               actual_current_is_fold_end_ts, 
               start_oos_fixed_ts, 
               end_oos_fixed_ts)


class WalkForwardOptimizer:
    def __init__(self, app_config: 'AppConfig'):
        self.app_config: 'AppConfig' = app_config
        self.global_config_obj: 'GlobalConfig' = self.app_config.global_config
        self.wfo_settings: 'WfoSettings' = self.global_config_obj.wfo_settings
        
        self.strategies_config_dict: Dict[str, Dict[str, Any]] = {}
        if hasattr(self.app_config.strategies_config, 'strategies') and \
           isinstance(self.app_config.strategies_config.strategies, dict):
            for name, strategy_params_obj in self.app_config.strategies_config.strategies.items():
                if hasattr(strategy_params_obj, '__dict__'):
                     self.strategies_config_dict[name] = strategy_params_obj.__dict__
                elif isinstance(strategy_params_obj, dict): 
                     self.strategies_config_dict[name] = strategy_params_obj
        else:
            logger.error("AppConfig.strategies_config.strategies is not a dictionary or is missing.")

        self.paths_config: Dict[str, Any] = self.global_config_obj.paths.__dict__
        self.simulation_defaults: Dict[str, Any] = self.global_config_obj.simulation_defaults.__dict__

        if not (0 < self.wfo_settings.oos_percent < 100):
            logger.error(f"'oos_percent' ({self.wfo_settings.oos_percent}) is invalid. Correcting to 30%.")
            self.wfo_settings.oos_percent = 30

        run_timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        logs_opt_path_str = self.paths_config.get('logs_backtest_optimization', 'logs/backtest_optimization')
        self.run_output_dir = Path(logs_opt_path_str) / run_timestamp_str 
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"WFO run output directory created: {self.run_output_dir}")

        selected_account_for_client: Optional[Any] = None
        if self.app_config.accounts_config:
            selected_account_for_client = next((acc for acc in self.app_config.accounts_config if acc.exchange.lower() == 'binance' and not acc.is_testnet), None)
            if not selected_account_for_client:
                selected_account_for_client = next((acc for acc in self.app_config.accounts_config if acc.exchange.lower() == 'binance'), None)

        if selected_account_for_client:
            api_key_val = self.app_config.api_keys.credentials.get(selected_account_for_client.account_alias, (None, None))[0]
            api_secret_val = self.app_config.api_keys.credentials.get(selected_account_for_client.account_alias, (None, None))[1]
            self.execution_client: Optional[OrderExecutionClient] = OrderExecutionClient(
                api_key=api_key_val,
                api_secret=api_secret_val,
                account_type=selected_account_for_client.account_type,
                is_testnet=selected_account_for_client.is_testnet
            )
            if not self.execution_client.test_connection():
                logger.warning("Failed Binance REST API connection during WFO init. Symbol info might fail.")
        else:
            logger.warning("No suitable 'binance' account found in accounts_config for OrderExecutionClient initialization. Symbol info might fail.")
            self.execution_client = None


    def run(self, pairs: List[str], context_labels: List[str]) -> Dict[str, Any]:
        all_wfo_run_results: Dict[str, Any] = {}
        main_run_log_prefix = f"[WFO Run: {self.run_output_dir.name}]"
        logger.info(f"{main_run_log_prefix} Starting WFO for pairs: {pairs}, context_labels: {context_labels}")

        for pair_symbol in pairs:
            current_context_label_raw = context_labels[0] if context_labels else "default_wfo_context"
            # Sanitize the context label for use in directory paths
            current_context_label_sanitized = _sanitize_filename_component(current_context_label_raw)
            
            pair_log_prefix = f"{main_run_log_prefix}[Pair: {pair_symbol}][Ctx: {current_context_label_sanitized}]" # Use sanitized here
            
            logger.info(f"{pair_log_prefix} Processing pair. Raw context: '{current_context_label_raw}', Sanitized: '{current_context_label_sanitized}'")

            symbol_info_data: Optional[Dict[str, Any]] = None
            if self.execution_client:
                try:
                    symbol_info_data = self.execution_client.get_symbol_info(pair_symbol)
                    if not symbol_info_data:
                        logger.error(f"{pair_log_prefix} Failed to get symbol_info. Skipping this pair.")
                        continue
                except Exception as e_sym_info:
                    logger.error(f"{pair_log_prefix} Error getting symbol_info: {e_sym_info}. Skipping this pair.")
                    continue
            else:
                logger.error(f"{pair_log_prefix} ExecutionClient not initialized. Cannot get symbol_info. Skipping this pair.")
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
                if 'timestamp' not in data_enriched_full.columns:
                    logger.error(f"{pair_log_prefix} Loaded enriched data from {enriched_filepath} is missing 'timestamp' column.")
                    continue
                
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
                logger.info(f"{pair_log_prefix} Loaded and preprocessed enriched data. Shape: {data_enriched_full.shape}")

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

                # Use sanitized context label for directory creation
                strategy_pair_context_output_dir = self.run_output_dir / strat_name / pair_symbol / current_context_label_sanitized
                strategy_pair_context_output_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"{strat_log_prefix} Output directory for this combo: {strategy_pair_context_output_dir}")
                
                fold_summaries: List[Dict[str, Any]] = []
                
                try:
                    folds_generator = _get_expanding_folds(
                        df_enriched=data_enriched_full,
                        n_splits=self.wfo_settings.n_splits,
                        oos_percent=self.wfo_settings.oos_percent
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
                            final_params_for_fold, representative_oos_metrics_fold = run_optimization_for_fold(
                                strategy_name=strat_name,
                                strategy_config_dict=strat_config_dict_for_opt,
                                data_1min_cleaned_is_slice=df_is_enriched_fold,
                                data_1min_cleaned_oos_slice=df_oos_fixed_enriched_fold,
                                app_config=self.app_config,
                                output_dir_fold=fold_artifacts_path, 
                                pair_symbol=pair_symbol,
                                symbol_info_data=symbol_info_data, # type: ignore
                                objective_evaluator_class=ObjectiveEvaluator,
                                study_manager_class=StudyManager,
                                results_analyzer_class=ResultsAnalyzer
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
                    "context_label": current_context_label_sanitized, # Save sanitized version
                    "raw_context_label_input": current_context_label_raw, # Also save raw input for reference
                    "wfo_run_timestamp": self.run_output_dir.name,
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
