# src/backtesting/optimizer/results_analyzer.py
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Type 
import optuna
import json 

from src.config.definitions import GlobalConfig, StrategyParamsConfig, ExchangeSettings, WfoSettings, AppConfig
from src.utils import file_utils 
from src.strategies.base import BaseStrategy
from src.backtesting.simulator import BacktestSimulator
from src.backtesting.performance import calculate_performance_metrics_from_inputs

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    def __init__(self,
                 run_id: str,
                 strategy_name: str,
                 pair_symbol: str,
                 context_label: str,
                 study: optuna.Study,
                 app_config: AppConfig, 
                 pair_config: Dict[str, Any], 
                 fold_data_map: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]],
                 fold_dirs_map: Dict[int, Path]):
        self.run_id = run_id
        self.strategy_name = strategy_name
        self.pair_symbol = pair_symbol
        self.context_label = context_label
        self.study = study
        self.app_config = app_config
        
        self.global_settings: GlobalConfig = app_config.global_config # type: ignore
        self.strategy_config_obj: Optional[StrategyParamsConfig] = app_config.strategies_config.strategies.get(strategy_name)
        if not self.strategy_config_obj:
            raise ValueError(f"Strategy configuration for '{strategy_name}' not found in AppConfig.")
        # self.exchange_settings: ExchangeSettings = app_config.exchange_settings # Moins pertinent maintenant
        self.wfo_settings: WfoSettings = app_config.global_config.wfo_settings
        
        self.pair_config = pair_config 
        if not self.pair_config:
            raise ValueError(f"Pair configuration for {self.pair_symbol} was not provided to ResultsAnalyzer directly.")

        self.fold_data_map = fold_data_map
        self.fold_dirs_map = fold_dirs_map
        
        self.log_prefix = f"[{self.strategy_name}][{self.pair_symbol}][{self.context_label}][ResultsAnalyzer]"
        logger.info(f"{self.log_prefix} Initialized.")


    def _save_optuna_study_summary(self, fold_id: int, fold_dir: Path):
        fold_dir.mkdir(parents=True, exist_ok=True)
        study_summary = {
            "study_name": self.study.study_name,
            "directions": [str(d) for d in self.study.directions], 
            "best_trials_numbers_is": [t.number for t in self.study.best_trials] if self.study.best_trials else None,
            "best_values_is": [t.values for t in self.study.best_trials] if self.study.best_trials else None,
            "best_params_is_first_objective": self.study.best_params if self.study.best_trial else None, 
            "n_trials_completed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }
        filepath = fold_dir / f"optuna_study_summary_fold_{fold_id}.json"
        
        if hasattr(file_utils, 'save_json') and callable(getattr(file_utils, 'save_json')):
            file_utils.save_json(filepath, study_summary)
        else: 
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(study_summary, f, indent=4, default=str)
        logger.info(f"{self.log_prefix} Saved Optuna study summary for fold {fold_id} to {filepath}")

        try:
            trials_df = self.study.trials_dataframe()
            trials_filepath = fold_dir / f"optuna_trials_dataframe_fold_{fold_id}.csv"
            trials_df.to_csv(trials_filepath, index=False)
            logger.info(f"{self.log_prefix} Saved Optuna trials dataframe for fold {fold_id} to {trials_filepath}")
        except Exception as e_df:
            logger.error(f"{self.log_prefix} Could not save Optuna trials dataframe for fold {fold_id}: {e_df}")


    def _get_best_is_trials_for_oos(self, fold_id: int) -> List[Dict[str, Any]]:
        if not self.study.trials:
            logger.warning(f"{self.log_prefix} No trials found in the study for fold {fold_id}.")
            return []

        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None and all(pd.notna(v) for v in t.values)]
        if not completed_trials:
            logger.warning(f"{self.log_prefix} No completed trials with valid (non-NaN) values found for fold {fold_id}.")
            return []
        
        sorted_pareto_trials = self.study.best_trials
        
        top_n = self.wfo_settings.top_n_trials_for_oos_validation
        best_trials_for_oos = []
        
        for trial in sorted_pareto_trials[:top_n]:
            best_trials_for_oos.append({
                "trial_number": trial.number,
                "params": trial.params,
                "is_values": trial.values 
            })
        
        logger.info(f"{self.log_prefix} Selected Top {len(best_trials_for_oos)} IS trials (from Pareto front if multi-obj) for OOS validation for fold {fold_id}.")
        return best_trials_for_oos


    def run_oos_validation_for_best_is_trials(self,
                                              fold_id: int,
                                              best_is_trials_for_oos: List[Dict[str, Any]],
                                              strategy_script_reference: str,
                                              strategy_class_name: str
                                             ) -> List[Dict[str, Any]]:
        if fold_id not in self.fold_data_map:
            logger.error(f"{self.log_prefix} Data for fold {fold_id} not found in fold_data_map.")
            return []
            
        _, df_oos_fold = self.fold_data_map[fold_id]
        if df_oos_fold.empty:
            logger.warning(f"{self.log_prefix} OOS data for fold {fold_id} is empty. Skipping OOS validation.")
            return []

        oos_results_list = []

        try:
            module_path = strategy_script_reference
            if module_path.endswith(".py"):
                 module_path = module_path.replace('.py', '').replace('/', '.')
            
            strategy_module = importlib.import_module(module_path)
            StrategyClass = getattr(strategy_module, strategy_class_name)
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to load strategy class {strategy_class_name} from {strategy_script_reference} for OOS: {e}", exc_info=True)
            return []

        sim_defaults = self.global_settings.simulation_defaults

        for is_trial_info in best_is_trials_for_oos:
            is_trial_number = is_trial_info["trial_number"]
            is_trial_params = is_trial_info["params"]
            is_values_from_study = is_trial_info["is_values"]

            logger.info(f"{self.log_prefix} Running OOS validation for IS trial {is_trial_number} (params: {is_trial_params}) on fold {fold_id}...")

            strategy_instance = StrategyClass( # type: ignore
                strategy_name=self.strategy_name,
                symbol=self.pair_symbol,
                params=is_trial_params
            )
            strategy_instance.set_backtest_context(
                pair_config=self.pair_config, # Utilise le pair_config stocké
                is_futures=sim_defaults.is_futures_trading,
                leverage=is_trial_params.get('margin_leverage', sim_defaults.margin_leverage),
                initial_equity=sim_defaults.initial_capital
            )

            simulator = BacktestSimulator(
                df_ohlcv=df_oos_fold.copy(),
                strategy_instance=strategy_instance,
                initial_equity=sim_defaults.initial_capital,
                leverage=is_trial_params.get('margin_leverage', sim_defaults.margin_leverage),
                symbol=self.pair_symbol,
                pair_config=self.pair_config, 
                trading_fee_bps=sim_defaults.trading_fee_bps,
                slippage_config=sim_defaults.slippage_config, 
                is_futures=sim_defaults.is_futures_trading,
                run_id=self.run_id,
                is_oos_simulation=True, 
                verbosity=sim_defaults.backtest_verbosity
            )

            try:
                oos_trades, oos_equity_curve_df, oos_daily_equity, oos_detailed_log = simulator.run_simulation()
                
                oos_equity_series = pd.Series(dtype=float)
                if not oos_equity_curve_df.empty and 'timestamp' in oos_equity_curve_df.columns and 'equity' in oos_equity_curve_df.columns:
                    temp_ec_df = oos_equity_curve_df.copy()
                    temp_ec_df['timestamp'] = pd.to_datetime(temp_ec_df['timestamp'], errors='coerce', utc=True)
                    temp_ec_df.dropna(subset=['timestamp','equity'], inplace=True)
                    if not temp_ec_df.empty:
                        oos_equity_series = temp_ec_df.set_index('timestamp')['equity'].sort_index()
                
                if oos_equity_series.empty:
                    oos_equity_series = pd.Series([sim_defaults.initial_capital], index=[df_oos_fold.index.min() if not df_oos_fold.empty else pd.Timestamp.now(tz='UTC')]) # type: ignore

                oos_metrics_dict = calculate_performance_metrics_from_inputs(
                    trades_df=pd.DataFrame(oos_trades),
                    equity_curve_series=oos_equity_series,
                    initial_capital=sim_defaults.initial_capital,
                    risk_free_rate_daily=(1 + self.global_settings.simulation_defaults.risk_free_rate)**(1/252)-1,
                    periods_per_year=252
                )
                oos_metrics_dict['Final Equity USDC'] = oos_equity_series.iloc[-1] if not oos_equity_series.empty else sim_defaults.initial_capital
                oos_metrics_dict['Total Trades'] = len(oos_trades)

                oos_trial_summary = {
                    "is_trial_number": is_trial_number,
                    "is_trial_params": is_trial_params,
                    "is_values_from_study": is_values_from_study,
                    "oos_metrics": oos_metrics_dict,
                    "oos_detailed_trades_log": oos_detailed_log
                }
                oos_results_list.append(oos_trial_summary)
                logger.info(f"{self.log_prefix} OOS validation for IS trial {is_trial_number} completed. Metric '{self.wfo_settings.metric_to_optimize}': {oos_metrics_dict.get(self.wfo_settings.metric_to_optimize)}")

            except Exception as e:
                logger.error(f"{self.log_prefix} Error during OOS simulation for IS trial {is_trial_number} on fold {fold_id}: {e}", exc_info=True)
                oos_results_list.append({
                    "is_trial_number": is_trial_number,
                    "is_trial_params": is_trial_params,
                    "is_values_from_study": is_values_from_study,
                    "oos_metrics": {"error": str(e)},
                    "oos_detailed_trades_log": []
                })
        
        metric_key_oos = self.wfo_settings.metric_to_optimize
        is_maximize_oos = True 
        if self.study.directions: 
            is_maximize_oos = self.study.directions[0] == optuna.study.StudyDirection.MAXIMIZE
        elif isinstance(self.study.direction, optuna.study.StudyDirection): 
            is_maximize_oos = self.study.direction == optuna.study.StudyDirection.MAXIMIZE

        def get_metric_for_sorting_oos(res):
            val = res.get("oos_metrics", {}).get(metric_key_oos)
            if val is None or not isinstance(val, (int, float)) or not pd.notna(val) or np.isinf(val): 
                return -float('inf') if is_maximize_oos else float('inf') 
            return val

        sorted_oos_results = sorted(
            oos_results_list,
            key=get_metric_for_sorting_oos,
            reverse=is_maximize_oos
        )
        
        top_n_oos_to_report = self.wfo_settings.top_n_trials_to_report_oos
        return sorted_oos_results[:top_n_oos_to_report]


    def analyze_and_save_results_for_fold(self, fold_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        logger.info(f"{self.log_prefix} Analyzing results for fold {fold_id}...")
        fold_dir = self.fold_dirs_map.get(fold_id)
        if not fold_dir:
            logger.error(f"{self.log_prefix} Output directory for fold {fold_id} not found. Cannot save results.")
            return None, None 

        fold_dir.mkdir(parents=True, exist_ok=True)

        self._save_optuna_study_summary(fold_id, fold_dir)

        best_is_trials_for_oos = self._get_best_is_trials_for_oos(fold_id)
        if not best_is_trials_for_oos:
            logger.warning(f"{self.log_prefix} No best IS trials found for OOS validation for fold {fold_id}. Skipping OOS.")
            if self.study.best_trial:
                return self.study.best_trial.params, {"IS_ONLY_BEST_VALUE": self.study.best_trial.values if self.study.best_trial.values else self.study.best_value} # type: ignore
            return None, None

        if not self.strategy_config_obj or \
           not self.strategy_config_obj.script_reference or \
           not self.strategy_config_obj.class_name:
             logger.error(f"{self.log_prefix} StrategyParamsConfig for {self.strategy_name} is missing 'script_reference' or 'class_name'. Cannot run OOS validation.")
             return None, None

        oos_validation_results_top_n = self.run_oos_validation_for_best_is_trials(
            fold_id=fold_id,
            best_is_trials_for_oos=best_is_trials_for_oos,
            strategy_script_reference=self.strategy_config_obj.script_reference,
            strategy_class_name=self.strategy_config_obj.class_name
        )

        final_selected_params_for_fold: Optional[Dict[str, Any]] = None
        final_selected_oos_metrics_for_fold: Optional[Dict[str, Any]] = None

        if oos_validation_results_top_n:
            best_oos_run_this_fold = oos_validation_results_top_n[0]
            final_selected_params_for_fold = best_oos_run_this_fold.get("is_trial_params")
            final_selected_oos_metrics_for_fold = best_oos_run_this_fold.get("oos_metrics")
            
            summary_filename = f"oos_validation_summary_TOP_{len(oos_validation_results_top_n)}_TRIALS_fold_{fold_id}.json"
            summary_filepath = fold_dir / summary_filename
            
            if hasattr(file_utils, 'save_json') and callable(getattr(file_utils, 'save_json')):
                 file_utils.save_json(summary_filepath, oos_validation_results_top_n)
            else:
                with open(summary_filepath, 'w', encoding='utf-8') as f:
                    json.dump(oos_validation_results_top_n, f, indent=4, default=str)
            logger.info(f"{self.log_prefix} Saved OOS validation summary to {summary_filepath}")

            for oos_run_summary in oos_validation_results_top_n:
                is_trial_no = oos_run_summary.get("is_trial_number")
                detailed_log_data = oos_run_summary.get("oos_detailed_trades_log")
                
                if detailed_log_data is not None and is_trial_no is not None:
                    detailed_log_filename = f"oos_trades_is_trial_{is_trial_no}_fold_{fold_id}.json"
                    detailed_log_filepath = fold_dir / detailed_log_filename
                    if hasattr(file_utils, 'save_json') and callable(getattr(file_utils, 'save_json')):
                        file_utils.save_json(detailed_log_filepath, detailed_log_data)
                    else:
                        with open(detailed_log_filepath, 'w', encoding='utf-8') as f:
                             json.dump(detailed_log_data, f, indent=4, default=str)
                    logger.info(f"{self.log_prefix} Saved detailed OOS trades for IS trial {is_trial_no} to {detailed_log_filepath}")
        else:
            logger.warning(f"{self.log_prefix} OOS validation did not yield any results for fold {fold_id}. "
                           "Falling back to best IS trial params if available.")
            if self.study.best_trial:
                final_selected_params_for_fold = self.study.best_trial.params
                final_selected_oos_metrics_for_fold = {"IS_ONLY_BEST_VALUE": self.study.best_trial.values if self.study.best_trial.values else self.study.best_value, "message": "No OOS results, used best IS."} # type: ignore
        
        return final_selected_params_for_fold, final_selected_oos_metrics_for_fold

