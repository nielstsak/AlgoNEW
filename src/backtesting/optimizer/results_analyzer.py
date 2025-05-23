import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Type, TYPE_CHECKING, Union
import optuna # type: ignore
import pandas as pd
import numpy as np
import math

if TYPE_CHECKING:
    from src.config.definitions import AppConfig, OptunaSettings # OptunaSettings is part of AppConfig
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    def __init__(self,
                 app_config: 'AppConfig', # Added app_config
                 study: optuna.Study,
                 objective_evaluator_class: Type['ObjectiveEvaluator'], # Class, not instance
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any],
                 output_dir_fold: Path,
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any]
                 ):
        self.app_config = app_config # Store app_config
        self.optuna_settings: 'OptunaSettings' = self.app_config.global_config.optuna_settings
        self.study = study
        self.objective_evaluator_class = objective_evaluator_class
        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict
        self.output_dir_fold = output_dir_fold
        self.pair_symbol = pair_symbol
        self.symbol_info_data = symbol_info_data
        
        self.log_prefix = f"[{self.strategy_name}][{self.pair_symbol}][Fold: {self.output_dir_fold.name}]"
        logger.info(f"{self.log_prefix} ResultsAnalyzer initialized.")

    def _select_n_best_trials_from_pareto(self, n: Optional[int] = None) -> List[optuna.trial.FrozenTrial]:
        """
        Selects the n best trials from the Pareto front of a multi-objective study.
        Handles different selection strategies based on Optuna settings.
        Filters trials with non-finite or missing objective values.
        Applies a PNL threshold if configured.
        Sorts trials based on a composite score or a primary metric.

        Args:
            n (Optional[int]): The number of best trials to select. If None, uses OptunaSettings.n_best_for_oos.

        Returns:
            List[optuna.trial.FrozenTrial]: A list of the selected best trials.
        """
        num_to_select = n if n is not None else self.optuna_settings.n_best_for_oos
        logger.info(f"{self.log_prefix} Selecting top {num_to_select} trials from Pareto front.")

        if not self.study.best_trials:
            logger.warning(f"{self.log_prefix} No best_trials (Pareto front) found in the study. Cannot select trials.")
            return []

        pareto_trials = self.study.best_trials
        logger.debug(f"{self.log_prefix} Initial Pareto front size: {len(pareto_trials)}")

        objectives_names = self.optuna_settings.objectives_names
        objectives_directions = self.optuna_settings.objectives_directions
        selection_strategy = self.optuna_settings.pareto_selection_strategy
        selection_weights = self.optuna_settings.pareto_selection_weights
        pnl_threshold = self.optuna_settings.pareto_selection_pnl_threshold

        # 1. Filter out trials with None values or non-finite values (inf, -inf, nan)
        valid_trials: List[optuna.trial.FrozenTrial] = []
        for t in pareto_trials:
            if t.state == optuna.trial.TrialState.COMPLETE and \
               t.values is not None and \
               all(isinstance(v, (float, int)) and np.isfinite(v) for v in t.values):
                valid_trials.append(t)
            else:
                logger.debug(f"{self.log_prefix} Trial {t.number} (State: {t.state}) excluded from Pareto selection due to invalid/incomplete objective values: {t.values}")
        
        if not valid_trials:
            logger.warning(f"{self.log_prefix} No trials with valid (finite, non-null, COMPLETE) objective values found in Pareto front.")
            return []
        logger.debug(f"{self.log_prefix} Number of trials after initial validity filter: {len(valid_trials)}")

        # 2. Apply PNL threshold if configured
        if pnl_threshold is not None:
            pnl_metric_name = "Total Net PnL USDC" # Assuming this is the standard PnL metric
            try:
                pnl_metric_index = objectives_names.index(pnl_metric_name)
                trials_above_threshold = [
                    t for t in valid_trials
                    if t.values and len(t.values) > pnl_metric_index and t.values[pnl_metric_index] >= pnl_threshold
                ]
                if not trials_above_threshold and valid_trials: # If threshold filtered all, log but proceed with unfiltered
                    logger.warning(f"{self.log_prefix} No trials met PNL threshold of {pnl_threshold}. Proceeding with all {len(valid_trials)} previously valid Pareto trials for sorting.")
                elif trials_above_threshold:
                    logger.info(f"{self.log_prefix} Applied PNL threshold of {pnl_threshold}. {len(trials_above_threshold)} trials remaining from {len(valid_trials)} valid Pareto trials.")
                    valid_trials = trials_above_threshold
            except (ValueError, IndexError):
                logger.warning(f"{self.log_prefix} PNL metric '{pnl_metric_name}' not found in objectives_names or index issue. PNL threshold not applied.")
        
        logger.debug(f"{self.log_prefix} Number of trials after PNL threshold filter: {len(valid_trials)}")
        if not valid_trials: return []


        # 3. Sort based on selection strategy
        if selection_strategy == "SCORE_COMPOSITE" and selection_weights and objectives_names:
            logger.info(f"{self.log_prefix} Sorting trials by SCORE_COMPOSITE.")
            scored_trials = []
            for trial in valid_trials:
                score = 0.0
                is_valid_for_scoring = True
                for i, obj_name in enumerate(objectives_names):
                    weight = selection_weights.get(obj_name, 0.0)
                    if weight != 0.0: # Only consider objectives with non-zero weights
                        if trial.values and i < len(trial.values) and isinstance(trial.values[i], (float, int)) and np.isfinite(trial.values[i]):
                            # Normalize or scale values before applying weights if objectives have vastly different ranges?
                            # For now, direct weighted sum.
                            score += trial.values[i] * weight
                        else:
                            is_valid_for_scoring = False
                            logger.debug(f"{self.log_prefix} Trial {trial.number} invalid for composite scoring: objective '{obj_name}' value missing or invalid.")
                            break
                if is_valid_for_scoring:
                    scored_trials.append({'trial': trial, 'score': score})
            
            if scored_trials:
                scored_trials.sort(key=lambda x: x['score'], reverse=True) # Higher score is better
                valid_trials = [st['trial'] for st in scored_trials]
            else:
                logger.warning(f"{self.log_prefix} No trials could be scored with SCORE_COMPOSITE. Falling back to primary objective sort.")
                # Fallback implemented below
                selection_strategy = "PNL_MAX" # Force fallback

        if selection_strategy != "SCORE_COMPOSITE": # Handles "PNL_MAX" or other primary metric sorts, and fallback
            sort_metric_name = "Total Net PnL USDC" # Default for PNL_MAX
            if selection_strategy and selection_strategy != "PNL_MAX": # If another primary metric is specified
                if selection_strategy in objectives_names:
                    sort_metric_name = selection_strategy
                else:
                    logger.warning(f"{self.log_prefix} pareto_selection_strategy '{selection_strategy}' not in objectives_names. Defaulting to PNL_MAX or first objective.")
            
            try:
                metric_index_for_sort = objectives_names.index(sort_metric_name)
                direction_for_sort = objectives_directions[metric_index_for_sort]
            except (ValueError, IndexError): # Fallback to the first objective if PNL_MAX or specified metric not found
                logger.debug(f"{self.log_prefix} Metric '{sort_metric_name}' not found for sorting. Using first objective '{objectives_names[0]}'.")
                metric_index_for_sort = 0
                direction_for_sort = objectives_directions[0] if objectives_directions else "maximize"
            
            sort_descending = direction_for_sort == "maximize"
            valid_trials.sort(key=lambda t: t.values[metric_index_for_sort] if (t.values and len(t.values) > metric_index_for_sort and np.isfinite(t.values[metric_index_for_sort])) else (-float('inf') if sort_descending else float('inf')),
                              reverse=sort_descending)
            logger.info(f"{self.log_prefix} Sorted {len(valid_trials)} trials by objective '{objectives_names[metric_index_for_sort]}' ({direction_for_sort}).")

        selected_trials = valid_trials[:num_to_select]
        logger.info(f"{self.log_prefix} Selected {len(selected_trials)} best trials for OOS validation from Pareto front.")
        return selected_trials

    def run_oos_validation_for_trials(self,
                                      selected_is_trials: List[optuna.trial.FrozenTrial],
                                      data_1min_cleaned_oos_slice: pd.DataFrame
                                      ) -> List[Dict[str, Any]]:
        """
        Runs Out-of-Sample (OOS) validation for a list of selected In-Sample (IS) trials.
        """
        oos_results_list: List[Dict[str, Any]] = []
        if not selected_is_trials:
            logger.warning(f"{self.log_prefix} No In-Sample trials provided for OOS validation.")
            return oos_results_list
        if data_1min_cleaned_oos_slice.empty:
            logger.warning(f"{self.log_prefix} OOS data slice is empty. Skipping OOS validation.")
            # Return list of dicts with IS info but empty OOS metrics
            for is_trial in selected_is_trials:
                 oos_results_list.append({
                    "is_trial_number": is_trial.number,
                    "is_trial_params": is_trial.params,
                    "is_trial_values": list(is_trial.values) if is_trial.values else None,
                    "oos_metrics": {name: np.nan for name in self.optuna_settings.objectives_names},
                    "oos_error": "OOS_DATA_EMPTY"
                })
            return oos_results_list

        logger.info(f"{self.log_prefix} Starting OOS validation for {len(selected_is_trials)} IS trials.")

        for i, is_trial in enumerate(selected_is_trials):
            logger.info(f"{self.log_prefix} Running OOS validation for IS Trial {is_trial.number} ({i+1}/{len(selected_is_trials)}). Params: {is_trial.params}")

            oos_objective_instance = self.objective_evaluator_class(
                strategy_name=self.strategy_name,
                strategy_config_dict=self.strategy_config_dict,
                df_enriched_slice=data_1min_cleaned_oos_slice,
                simulation_settings=self.app_config.global_config.simulation_defaults.__dict__,
                optuna_objectives_config={
                    'objectives_names': self.optuna_settings.objectives_names,
                    'objectives_directions': self.optuna_settings.objectives_directions
                },
                pair_symbol=self.pair_symbol,
                symbol_info_data=self.symbol_info_data,
                app_config=self.app_config,
                is_oos_eval=True,
                is_trial_number_for_oos_log=is_trial.number # Pass IS trial number for logging context
            )

            # Create a dummy trial for the OOS evaluation call.
            # The parameters and distributions are fixed from the IS trial.
            dummy_oos_trial_values = [0.0] * len(self.optuna_settings.objectives_directions)
            try:
                dummy_oos_trial = optuna.trial.create_trial(
                    params=is_trial.params, # Fixed parameters from IS trial
                    distributions=is_trial.distributions, # Distributions from IS trial
                    values=dummy_oos_trial_values # Placeholder values
                )
            except Exception as e_create_dummy:
                logger.error(f"{self.log_prefix} Failed to create dummy OOS trial for IS trial {is_trial.number}: {e_create_dummy}", exc_info=True)
                oos_metrics_tuple = tuple([-float('inf') if direction == "maximize" else float('inf')
                                           for direction in self.optuna_settings.objectives_directions])
                oos_error_msg = f"DUMMY_TRIAL_CREATION_ERROR: {e_create_dummy}"
                oos_metrics_dict = {name: val for name, val in zip(self.optuna_settings.objectives_names, oos_metrics_tuple)}
                oos_results_list.append({
                    "is_trial_number": is_trial.number,
                    "is_trial_params": is_trial.params,
                    "is_trial_values": list(is_trial.values) if is_trial.values else None,
                    "oos_metrics": oos_metrics_dict,
                    "oos_error": oos_error_msg
                })
                continue


            oos_metrics_tuple: Tuple[float, ...]
            oos_error_msg: Optional[str] = None
            try:
                oos_metrics_tuple = oos_objective_instance(dummy_oos_trial)
                logger.debug(f"{self.log_prefix} OOS metrics for IS Trial {is_trial.number}: {oos_metrics_tuple}")
            except optuna.exceptions.TrialPruned as e_pruned_oos:
                logger.warning(f"{self.log_prefix} OOS evaluation pruned for IS Trial {is_trial.number}: {e_pruned_oos}")
                oos_metrics_tuple = tuple([-float('inf') if direction == "maximize" else float('inf')
                                           for direction in self.optuna_settings.objectives_directions])
                oos_error_msg = f"PRUNED: {e_pruned_oos}"
            except Exception as e_oos_eval:
                logger.error(f"{self.log_prefix} Error during OOS evaluation for IS Trial {is_trial.number}: {e_oos_eval}", exc_info=True)
                oos_metrics_tuple = tuple([-float('inf') if direction == "maximize" else float('inf')
                                           for direction in self.optuna_settings.objectives_directions])
                oos_error_msg = f"ERROR: {e_oos_eval}"

            oos_metrics_dict = {name: val for name, val in zip(self.optuna_settings.objectives_names, oos_metrics_tuple)}

            result_entry = {
                "is_trial_number": is_trial.number,
                "is_trial_params": is_trial.params,
                "is_trial_values": list(is_trial.values) if is_trial.values else None, # Convert tuple to list for JSON
                "oos_metrics": oos_metrics_dict
            }
            if oos_error_msg:
                result_entry["oos_error"] = oos_error_msg
            oos_results_list.append(result_entry)

        logger.info(f"{self.log_prefix} Completed OOS validation for {len(selected_is_trials)} trials.")
        return oos_results_list

    def save_oos_validation_results(self, oos_results: List[Dict[str, Any]]):
        """Saves the OOS validation summary to a JSON file."""
        if not oos_results:
            logger.info(f"{self.log_prefix} No OOS results to save.")
            return

        file_path = self.output_dir_fold / "oos_validation_summary_TOP_N_TRIALS.json"
        try:
            # Ensure data is JSON serializable (handle NaN, Inf, np types)
            serializable_results = []
            for res_entry in oos_results:
                entry_copy = res_entry.copy()
                if 'is_trial_values' in entry_copy and entry_copy['is_trial_values'] is not None:
                    entry_copy['is_trial_values'] = [
                        (v if isinstance(v, (int, float)) and np.isfinite(v) else (str(v) if not pd.isna(v) else None) )
                        for v in entry_copy['is_trial_values']
                    ]
                if 'oos_metrics' in entry_copy and isinstance(entry_copy['oos_metrics'], dict):
                    entry_copy['oos_metrics'] = {
                        k: (v if isinstance(v, (int, float)) and np.isfinite(v) else (str(v) if not pd.isna(v) else None) )
                        for k, v in entry_copy['oos_metrics'].items()
                    }
                serializable_results.append(entry_copy)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=4, allow_nan=False) # allow_nan=False forces conversion of NaN/Inf
            logger.info(f"{self.log_prefix} OOS validation summary saved to: {file_path}")
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to save OOS validation summary to {file_path}: {e}", exc_info=True)

    def select_final_parameters_for_live(self, oos_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selects the final set of parameters for live trading based on OOS results.
        Uses metric_to_optimize from WFO settings as the primary OOS selection metric.
        """
        if not oos_results:
            logger.warning(f"{self.log_prefix} No OOS results provided to select final parameters.")
            return None

        # Use the metric defined in WFO settings for final selection based on OOS performance
        selection_metric_name = self.app_config.global_config.wfo_settings.metric_to_optimize
        # Determine direction from OptunaSettings if metric_to_optimize is one of the objectives
        selection_direction = "maximize" # Default
        try:
            metric_idx = self.optuna_settings.objectives_names.index(selection_metric_name)
            selection_direction = self.optuna_settings.objectives_directions[metric_idx]
        except (ValueError, IndexError):
            logger.warning(f"{self.log_prefix} Metric '{selection_metric_name}' for final OOS selection not found in Optuna objectives_names. "
                           f"Assuming 'maximize' direction. Or, ensure WFO 'metric_to_optimize' matches an Optuna objective.")
            # If not an Optuna objective, we must assume a direction or have it configured elsewhere.
            # For simplicity, if it's PnL-like, maximize. If Drawdown-like, minimize.
            if "pnl" in selection_metric_name.lower() or "sharpe" in selection_metric_name.lower() or "profit" in selection_metric_name.lower():
                selection_direction = "maximize"
            elif "drawdown" in selection_metric_name.lower():
                selection_direction = "minimize"


        logger.info(f"{self.log_prefix} Selecting final parameters based on OOS metric: '{selection_metric_name}' (Direction: {selection_direction})")

        best_oos_trial_info = None
        best_oos_metric_value = -float('inf') if selection_direction == "maximize" else float('inf')

        for result_entry in oos_results:
            oos_metrics = result_entry.get("oos_metrics", {})
            current_metric_value = oos_metrics.get(selection_metric_name)

            if current_metric_value is None or not isinstance(current_metric_value, (int, float)) or not np.isfinite(current_metric_value):
                logger.debug(f"{self.log_prefix} Skipping OOS result for IS trial {result_entry.get('is_trial_number')} due to invalid OOS selection metric value: {current_metric_value}")
                continue

            if selection_direction == "maximize":
                if current_metric_value > best_oos_metric_value:
                    best_oos_metric_value = current_metric_value
                    best_oos_trial_info = result_entry
            else: # minimize
                if current_metric_value < best_oos_metric_value:
                    best_oos_metric_value = current_metric_value
                    best_oos_trial_info = result_entry
        
        if best_oos_trial_info:
            final_params = best_oos_trial_info.get("is_trial_params")
            logger.info(f"{self.log_prefix} Final parameters selected from IS Trial {best_oos_trial_info.get('is_trial_number')} "
                        f"with OOS '{selection_metric_name}': {best_oos_metric_value:.4f}. Parameters: {final_params}")
            return final_params
        else:
            logger.warning(f"{self.log_prefix} Could not select final parameters. No OOS trial had a valid value for metric '{selection_metric_name}'.")
            return None

