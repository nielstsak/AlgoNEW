# src/backtesting/optimizer/results_analyzer.py
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import optuna

# CORRECTED IMPORTS: Config definitions should come from .definitions
from src.config.definitions import GlobalConfig, StrategyParamsConfig, ExchangeSettings, WfoSettings 
from src.config.loader import load_exchange_config # Functions for loading instances are fine from loader
from src.utils import file_utils
from src.strategies.base import BaseStrategy
from src.backtesting.simulator import BacktestSimulator
from src.backtesting.performance import PerformanceCalculator
# from src.data.data_utils import load_processed_data_for_symbol_or_pair # Not used directly here
from src.utils.exchange_utils import get_pair_config_for_symbol


logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    """
    Analyzes the results of Optuna optimization studies, performs Out-of-Sample (OOS) validation,
    and saves relevant summaries and artifacts.
    """
    def __init__(self,
                 run_id: str,
                 strategy_name: str,
                 pair_symbol: str,
                 context_label: str,
                 study: optuna.Study,
                 global_settings: GlobalConfig, # CORRECTED TYPE HINT
                 strategy_config: StrategyParamsConfig, # CORRECTED TYPE HINT
                 exchange_settings: ExchangeSettings, 
                 wfo_settings: WfoSettings, # CORRECTED TYPE HINT
                 fold_data_map: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]],
                 fold_dirs_map: Dict[int, Path]):
        """
        Initializes the ResultsAnalyzer.

        Args:
            run_id (str): Unique identifier for the optimization run.
            strategy_name (str): Name of the strategy being optimized.
            pair_symbol (str): Trading pair symbol (e.g., "BTCUSDT").
            context_label (str): Context label for the optimization (e.g., timeframe).
            study (optuna.Study): The completed Optuna study object.
            global_settings (GlobalConfig): Global configuration settings instance.
            strategy_config (StrategyParamsConfig): Strategy-specific configuration instance.
            exchange_settings (ExchangeSettings): Exchange-specific configuration instance.
            wfo_settings (WfoSettings): Walk-Forward Optimization settings instance.
            fold_data_map (Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]):
                Map of fold_id to (df_is_fold, df_oos_fold).
            fold_dirs_map (Dict[int, Path]): Map of fold_id to its specific output directory.
        """
        self.run_id = run_id
        self.strategy_name = strategy_name
        self.pair_symbol = pair_symbol
        self.context_label = context_label
        self.study = study
        self.global_settings = global_settings
        self.strategy_config = strategy_config 
        self.exchange_settings = exchange_settings
        self.wfo_settings = wfo_settings
        self.fold_data_map = fold_data_map
        self.fold_dirs_map = fold_dirs_map

        # Assuming LOGS_DIR is globally available or defined in src.config.definitions
        # If not, it needs to be passed or loaded from global_settings.paths
        # Check if global_settings.paths.logs_backtest_optimization exists
        if hasattr(self.global_settings, 'paths') and hasattr(self.global_settings.paths, 'logs_backtest_optimization'):
            base_logs_path = Path(self.global_settings.paths.logs_backtest_optimization)
        else:
            # Fallback if paths not in global_settings, though it should be based on GlobalConfig definition
            from src.config.definitions import LOGS_DIR # Make sure LOGS_DIR is defined or handle error
            base_logs_path = Path(LOGS_DIR) / "backtest_optimization"
            logger.warning(f"Could not find 'logs_backtest_optimization' in global_settings.paths. Using default: {base_logs_path}")

        self.base_log_dir = base_logs_path / self.run_id / \
                            self.strategy_name / self.pair_symbol / self.context_label
        
        self.pair_config = get_pair_config_for_symbol(
            self.pair_symbol, 
            self.exchange_settings.exchange_info_file_path 
        )
        if not self.pair_config:
            try:
                # Attempt to load exchange_info directly if path is available and pair_config failed
                # Assuming load_exchange_config returns the full exchange info dict
                full_exchange_info = load_exchange_config(Path(self.exchange_settings.exchange_info_file_path))
                if isinstance(full_exchange_info, dict) and 'symbols' in full_exchange_info: # Check structure
                    self.pair_config = next((s for s in full_exchange_info['symbols'] if s['symbol'] == self.pair_symbol), None)
                else: # If load_exchange_config returns something else, adapt or use get_pair_config_for_symbol
                    self.pair_config = get_pair_config_for_symbol(self.pair_symbol, full_exchange_info) # type: ignore
            except Exception as e_load_exc:
                logger.error(f"Error trying to load exchange info for pair_config fallback: {e_load_exc}")


            if not self.pair_config: # Final check
                raise ValueError(f"Pair configuration not found for {self.pair_symbol} using path {self.exchange_settings.exchange_info_file_path}")


    def analyze_and_save_results(self, fold_id: int):
        logger.info(f"Analyzing results for fold {fold_id}...")
        fold_dir = self.fold_dirs_map[fold_id]
        fold_dir.mkdir(parents=True, exist_ok=True)

        self._save_optuna_study_summary(fold_id, fold_dir)

        best_is_trials_for_oos = self._get_best_is_trials_for_oos(fold_id)
        if not best_is_trials_for_oos:
            logger.warning(f"No best IS trials found for OOS validation for fold {fold_id}. Skipping OOS.")
            return

        # strategy_config (StrategyParamsConfig) should have script_reference and class_name
        if not hasattr(self.strategy_config, 'script_reference') or not self.strategy_config.script_reference or \
           not hasattr(self.strategy_config, 'class_name') or not self.strategy_config.class_name:
             logger.error(f"StrategyParamsConfig for {self.strategy_name} is missing 'script_reference' or 'class_name'. Cannot run OOS validation.")
             return

        oos_validation_results = self.run_oos_validation_for_best_is_trials(
            fold_id,
            best_is_trials_for_oos,
            self.strategy_config.script_reference, 
            self.strategy_config.class_name      
        )

        if oos_validation_results:
            top_n_to_report = self.wfo_settings.top_n_trials_to_report_oos
            
            summary_filename = f"oos_validation_summary_TOP_{top_n_to_report}_TRIALS.json"
            summary_filepath = fold_dir / summary_filename
            file_utils.save_json(summary_filepath, oos_validation_results) 
            logger.info(f"Saved OOS validation summary to {summary_filepath}")

            for oos_run_summary in oos_validation_results: 
                is_trial_no = oos_run_summary.get("is_trial_number")
                detailed_log_data = oos_run_summary.get("oos_detailed_trades_log")
                
                if detailed_log_data is not None and is_trial_no is not None:
                    detailed_log_filename = f"oos_best_trial_trades_is_trial_{is_trial_no}_fold_{fold_id}.json"
                    detailed_log_filepath = fold_dir / detailed_log_filename
                    file_utils.save_json(detailed_log_filepath, detailed_log_data)
                    logger.info(f"Saved detailed OOS trades for IS trial {is_trial_no} to {detailed_log_filepath}")
                elif is_trial_no is None:
                    logger.warning(f"Missing 'is_trial_number' in OOS run summary for fold {fold_id}, cannot save detailed log.")
        else:
            logger.warning(f"OOS validation did not yield any results for fold {fold_id}.")


    def _save_optuna_study_summary(self, fold_id: int, fold_dir: Path):
        study_summary = {
            "study_name": self.study.study_name,
            "direction": str(self.study.direction), 
            "best_trial_number_is": self.study.best_trial.number if self.study.best_trial else None,
            "best_value_is": self.study.best_value if self.study.best_trial else None,
            "best_params_is": self.study.best_params if self.study.best_trial else None,
            "n_trials_completed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }
        filepath = fold_dir / f"optuna_study_summary_fold_{fold_id}.json"
        file_utils.save_json(filepath, study_summary)
        logger.info(f"Saved Optuna study summary for fold {fold_id} to {filepath}")

        try:
            trials_df = self.study.trials_dataframe()
            trials_filepath = fold_dir / f"optuna_trials_dataframe_fold_{fold_id}.csv"
            trials_df.to_csv(trials_filepath, index=False)
            logger.info(f"Saved Optuna trials dataframe for fold {fold_id} to {trials_filepath}")
        except Exception as e_df:
            logger.error(f"Could not save Optuna trials dataframe for fold {fold_id}: {e_df}")


    def _get_best_is_trials_for_oos(self, fold_id: int) -> List[Dict[str, Any]]:
        if not self.study.trials:
            logger.warning(f"No trials found in the study for fold {fold_id}.")
            return []

        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and pd.notna(t.value)]
        if not completed_trials:
            logger.warning(f"No completed trials with valid (non-NaN) values found for fold {fold_id}.")
            return []
        
        is_maximize = True # Default
        # Optuna direction can be a single StudyDirection or a list for multi-objective
        if isinstance(self.study.direction, optuna.study.StudyDirection): # Single objective
            is_maximize = self.study.direction == optuna.study.StudyDirection.MAXIMIZE
        elif isinstance(self.study.directions, list) and self.study.directions: # Multi-objective
            is_maximize = self.study.directions[0] == optuna.study.StudyDirection.MAXIMIZE
            logger.warning(f"Multi-objective study detected for fold {fold_id}. Using direction of the first objective for sorting IS trials.")
        else: # Fallback or unknown direction type
            logger.warning(f"Could not determine optimization direction for study of fold {fold_id}. Defaulting to 'maximize'. Study direction: {self.study.direction}")


        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=is_maximize)
        
        top_n = self.wfo_settings.top_n_trials_for_oos_validation
        best_trials_for_oos = []
        for trial in sorted_trials[:top_n]:
            best_trials_for_oos.append({
                "trial_number": trial.number,
                "params": trial.params,
                "is_value": trial.value 
            })
        
        logger.info(f"Selected Top {len(best_trials_for_oos)} IS trials for OOS validation for fold {fold_id}.")
        return best_trials_for_oos


    def run_oos_validation_for_best_is_trials(self,
                                              fold_id: int,
                                              best_is_trials_for_oos: List[Dict[str, Any]],
                                              strategy_script_reference: str, 
                                              strategy_class_name: str      
                                             ) -> List[Dict[str, Any]]:
        _, df_oos_fold = self.fold_data_map[fold_id]
        if df_oos_fold.empty:
            logger.warning(f"OOS data for fold {fold_id} is empty. Skipping OOS validation.")
            return []

        oos_results_list = []

        try:
            module_path = strategy_script_reference
            if module_path.endswith(".py"): # Convert file path to module path if needed
                 module_path = module_path.replace('.py', '').replace('/', '.') 
            
            strategy_module = file_utils.import_module_from_path(module_path) 
            StrategyClass = getattr(strategy_module, strategy_class_name)
        except Exception as e:
            logger.error(f"Failed to load strategy class {strategy_class_name} from {strategy_script_reference} for OOS: {e}", exc_info=True)
            return []

        sim_defaults = self.global_settings.simulation_defaults

        for is_trial_info in best_is_trials_for_oos:
            is_trial_number = is_trial_info["trial_number"]
            is_trial_params = is_trial_info["params"]
            is_value = is_trial_info["is_value"]

            logger.info(f"Running OOS validation for IS trial {is_trial_number} (params: {is_trial_params}) on fold {fold_id}...")

            strategy_instance = StrategyClass(
                strategy_name=self.strategy_name, 
                symbol=self.pair_symbol,
                params=is_trial_params
            )
            strategy_instance.set_backtest_context(
                pair_config=self.pair_config,
                is_futures=sim_defaults.is_futures_trading,
                leverage=is_trial_params.get('margin_leverage', sim_defaults.margin_leverage), # Allow override from trial params
                initial_equity=sim_defaults.initial_capital
            )

            simulator = BacktestSimulator(
                df_ohlcv=df_oos_fold.copy(), 
                strategy_instance=strategy_instance,
                initial_equity=sim_defaults.initial_capital,
                leverage=is_trial_params.get('margin_leverage', sim_defaults.margin_leverage), 
                symbol=self.pair_symbol,
                trading_fee_bps=sim_defaults.trading_fee_bps,
                slippage_config=sim_defaults.slippage_config,
                is_futures=sim_defaults.is_futures_trading,
                run_id=self.run_id, 
                is_oos_simulation=True, 
                verbosity=sim_defaults.backtest_verbosity 
            )

            try:
                oos_trades, oos_equity_curve, oos_daily_equity, oos_detailed_log = simulator.run_simulation()
                
                performance_calculator = PerformanceCalculator(
                    trades=oos_trades,
                    equity_curve=oos_equity_curve,
                    daily_equity_values=list(oos_daily_equity.values()), 
                    initial_capital=sim_defaults.initial_capital,
                    risk_free_rate=self.global_settings.risk_free_rate,
                    benchmark_returns=None 
                )
                oos_metrics_dict = performance_calculator.calculate_all_metrics()
                oos_metrics_dict['final_equity'] = oos_equity_curve['equity'].iloc[-1] if not oos_equity_curve.empty else sim_defaults.initial_capital
                oos_metrics_dict['total_trades'] = len(oos_trades)

                oos_trial_summary = {
                    "is_trial_number": is_trial_number,
                    "is_trial_params": is_trial_params,
                    "is_value_from_study": is_value, 
                    "oos_performance": oos_metrics_dict,
                    "oos_detailed_trades_log": oos_detailed_log 
                }
                oos_results_list.append(oos_trial_summary)
                logger.info(f"OOS validation for IS trial {is_trial_number} completed. Metric '{self.wfo_settings.metric_to_optimize}': {oos_metrics_dict.get(self.wfo_settings.metric_to_optimize)}")

            except Exception as e:
                logger.error(f"Error during OOS simulation for IS trial {is_trial_number} on fold {fold_id}: {e}", exc_info=True)
                oos_results_list.append({
                    "is_trial_number": is_trial_number,
                    "is_trial_params": is_trial_params,
                    "is_value_from_study": is_value,
                    "oos_performance": {"error": str(e)},
                    "oos_detailed_trades_log": [] 
                })
        
        metric_key = self.wfo_settings.metric_to_optimize
        
        is_maximize_oos = True # Default
        if isinstance(self.study.direction, optuna.study.StudyDirection):
            is_maximize_oos = self.study.direction == optuna.study.StudyDirection.MAXIMIZE
        elif isinstance(self.study.directions, list) and self.study.directions:
            is_maximize_oos = self.study.directions[0] == optuna.study.StudyDirection.MAXIMIZE


        def get_metric_for_sorting(res):
            val = res.get("oos_performance", {}).get(metric_key)
            if val is None or not isinstance(val, (int, float)) or not pd.notna(val):
                return -float('inf') if is_maximize_oos else float('inf')
            return val

        sorted_oos_results = sorted(
            oos_results_list,
            key=get_metric_for_sorting,
            reverse=is_maximize_oos 
        )
        
        top_n_oos_to_report = self.wfo_settings.top_n_trials_to_report_oos
        return sorted_oos_results[:top_n_oos_to_report]


    def aggregate_fold_results(self, all_fold_oos_summaries: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        # Ensure LOGS_DIR is defined or accessible
        if hasattr(self.global_settings, 'paths') and hasattr(self.global_settings.paths, 'logs_backtest_optimization'):
            base_logs_path = Path(self.global_settings.paths.logs_backtest_optimization)
        else:
            from src.config.definitions import LOGS_DIR # Fallback
            base_logs_path = Path(LOGS_DIR) / "backtest_optimization"
        
        # Use self.base_log_dir which is already defined relative to LOGS_DIR and run_id
        # However, self.base_log_dir is strategy/pair/context specific. Aggregation should be at run_id level.
        aggregated_results_dir = base_logs_path / self.run_id / "_AGGREGATED_RESULTS"
        aggregated_results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = aggregated_results_dir / f"all_folds_oos_summaries_{self.strategy_name}_{self.pair_symbol}_{self.context_label}.json"
        file_utils.save_json(filepath, all_fold_oos_summaries)
        logger.info(f"Saved aggregated OOS summaries for {self.strategy_name}/{self.pair_symbol}/{self.context_label} to {filepath}")
        
        return {"message": "Aggregated fold results saved.", "path": str(filepath)}

