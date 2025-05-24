import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Type, TYPE_CHECKING
import optuna # type: ignore
import pandas as pd
import numpy as np
import math

if TYPE_CHECKING: 
    from src.config.definitions import AppConfig, OptunaSettings
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    def __init__(self,
                 app_config: 'AppConfig',
                 study: optuna.Study,
                 strategy_name: str,
                 output_dir_fold: Path, 
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any], 
                 fold_name_for_log: Optional[str] = None
                 ):
        self.app_config = app_config 
        self.optuna_settings: 'OptunaSettings' = self.app_config.global_config.optuna_settings
        self.study = study
        self.strategy_name = strategy_name
        self.output_dir_fold = output_dir_fold 
        self.pair_symbol = pair_symbol
        self.symbol_info_data = symbol_info_data 
        self.fold_name_for_log = fold_name_for_log if fold_name_for_log else self.output_dir_fold.name
        
        self.log_prefix = f"[{self.strategy_name}][{self.pair_symbol}][Fold: {self.fold_name_for_log}]"
        logger.info(f"{self.log_prefix} ResultsAnalyzer initialized for study '{study.study_name}'.")

    def get_pareto_front_trials(self) -> List[optuna.trial.FrozenTrial]:
        log_ctx = f"{self.log_prefix}[GetPareto]"
        if not self.study.best_trials: 
            logger.warning(f"{log_ctx} No best_trials (Pareto front) available in the study '{self.study.study_name}'.")
            return []

        pareto_trials = self.study.best_trials
        valid_pareto_trials: List[optuna.trial.FrozenTrial] = []
        for trial in pareto_trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and \
               trial.values is not None and \
               all(isinstance(v, (float, int)) and np.isfinite(v) for v in trial.values):
                valid_pareto_trials.append(trial)
            else:
                logger.debug(f"{log_ctx} Trial {trial.number} (State: {trial.state}) from Pareto front excluded due to invalid/incomplete objective values: {trial.values}")
        
        logger.info(f"{log_ctx} Found {len(valid_pareto_trials)} valid trials in Pareto front.")
        return valid_pareto_trials

    def _select_n_best_trials_from_list(self,
                                        trials_list: List[optuna.trial.FrozenTrial],
                                        n: Optional[int] = None
                                        ) -> List[optuna.trial.FrozenTrial]:
        num_to_select = n if n is not None else self.optuna_settings.n_best_for_oos
        log_ctx_select = f"{self.log_prefix}[SelectNBest]"
        logger.info(f"{log_ctx_select} Selecting top {num_to_select} trials from a list of {len(trials_list)} trials.")

        if not trials_list:
            logger.warning(f"{log_ctx_select} Input trials_list is empty.")
            return []

        objectives_names = self.optuna_settings.objectives_names
        objectives_directions = self.optuna_settings.objectives_directions
        selection_strategy = self.optuna_settings.pareto_selection_strategy
        selection_weights = self.optuna_settings.pareto_selection_weights
        pnl_threshold = self.optuna_settings.pareto_selection_pnl_threshold
        
        valid_trials = trials_list 

        if pnl_threshold is not None:
            pnl_metric_name = "Total Net PnL USDC" 
            try:
                pnl_metric_index = objectives_names.index(pnl_metric_name)
                trials_above_threshold = [
                    t for t in valid_trials
                    if t.values and len(t.values) > pnl_metric_index and t.values[pnl_metric_index] >= pnl_threshold # type: ignore
                ]
                if not trials_above_threshold and valid_trials:
                    logger.warning(f"{log_ctx_select} No trials met PNL threshold of {pnl_threshold}. Proceeding with all {len(valid_trials)} previously valid trials for sorting.")
                elif trials_above_threshold:
                    logger.info(f"{log_ctx_select} Applied PNL threshold. {len(trials_above_threshold)} trials remaining from {len(valid_trials)}.")
                    valid_trials = trials_above_threshold
            except (ValueError, IndexError):
                logger.warning(f"{log_ctx_select} PNL metric '{pnl_metric_name}' not found. PNL threshold not applied.")
        
        if not valid_trials: 
            logger.warning(f"{log_ctx_select} No valid trials remaining after PNL threshold filter.")
            return []

        if selection_strategy == "SCORE_COMPOSITE" and selection_weights and objectives_names:
            logger.info(f"{log_ctx_select} Sorting trials by SCORE_COMPOSITE.")
            scored_trials = []
            for trial in valid_trials:
                score = 0.0; is_valid_for_scoring = True
                for i, obj_name in enumerate(objectives_names):
                    weight = selection_weights.get(obj_name, 0.0)
                    if weight != 0.0:
                        if trial.values and i < len(trial.values) and isinstance(trial.values[i], (float, int)) and np.isfinite(trial.values[i]): # type: ignore
                            score += trial.values[i] * weight # type: ignore
                        else: is_valid_for_scoring = False; break
                if is_valid_for_scoring: scored_trials.append({'trial': trial, 'score': score})
            
            if scored_trials:
                scored_trials.sort(key=lambda x: x['score'], reverse=True) 
                valid_trials = [st['trial'] for st in scored_trials]
            else:
                logger.warning(f"{log_ctx_select} No trials scorable. Falling back to primary objective sort."); selection_strategy = "PNL_MAX"

        if selection_strategy != "SCORE_COMPOSITE": 
            sort_metric_name = "Total Net PnL USDC" 
            if selection_strategy and selection_strategy != "PNL_MAX":
                if selection_strategy in objectives_names: sort_metric_name = selection_strategy
                else: logger.warning(f"{log_ctx_select} pareto_selection_strategy '{selection_strategy}' invalid. Defaulting to PNL_MAX.")
            
            try:
                metric_index_for_sort = objectives_names.index(sort_metric_name)
                direction_for_sort = objectives_directions[metric_index_for_sort]
            except (ValueError, IndexError): 
                metric_index_for_sort = 0
                direction_for_sort = objectives_directions[0] if objectives_directions else "maximize"
            
            sort_descending = direction_for_sort == "maximize"
            valid_trials.sort(key=lambda t: t.values[metric_index_for_sort] if (t.values and len(t.values) > metric_index_for_sort and np.isfinite(t.values[metric_index_for_sort])) else (-float('inf') if sort_descending else float('inf')), # type: ignore
                              reverse=sort_descending)
            logger.info(f"{log_ctx_select} Sorted {len(valid_trials)} trials by objective '{objectives_names[metric_index_for_sort]}' ({direction_for_sort}).")

        selected_trials = valid_trials[:num_to_select]
        logger.info(f"{log_ctx_select} Selected {len(selected_trials)} best trials.")
        return selected_trials

    def get_best_trials_for_oos_validation(self) -> List[optuna.trial.FrozenTrial]:
        pareto_trials = self.get_pareto_front_trials()
        if not pareto_trials:
            return []
        return self._select_n_best_trials_from_list(pareto_trials)

    def run_oos_validation_for_best_is_trials(self,
                                      best_is_trials_from_study: List[optuna.trial.FrozenTrial],
                                      data_1min_cleaned_oos_slice: pd.DataFrame,
                                      objective_evaluator_class: Type['ObjectiveEvaluator']
                                      ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        
        log_prefix_oos_val = f"{self.log_prefix}[OOS_Validation]"
        logger.info(f"{log_prefix_oos_val} Starting OOS validation for {len(best_is_trials_from_study)} best IS trials.")

        if data_1min_cleaned_oos_slice.empty:
            logger.warning(f"{log_prefix_oos_val} OOS data slice is empty. Skipping OOS validation.")
            empty_oos_results = []
            for rank_idx, is_trial in enumerate(best_is_trials_from_study):
                 empty_oos_results.append({
                    "is_trial_number_optuna": is_trial.number,
                    "is_trial_rank_in_selection": rank_idx + 1, 
                    "params": is_trial.params,
                    "is_metrics": {name: val for name, val in zip(self.optuna_settings.objectives_names, is_trial.values)} if is_trial.values else {},
                    "oos_metrics": {name: np.nan for name in self.optuna_settings.objectives_names},
                    "oos_trades_df": None, 
                    "oos_error": "OOS_DATA_EMPTY"
                })
            return empty_oos_results, None, None

        oos_validation_results: List[Dict[str, Any]] = []
        
        optuna_objectives_config = {
            'objectives_names': self.optuna_settings.objectives_names,
            'objectives_directions': self.optuna_settings.objectives_directions
        }
        
        strategy_config_for_evaluator_obj = self.app_config.strategies_config.strategies.get(self.strategy_name)
        if not strategy_config_for_evaluator_obj:
            logger.error(f"{log_prefix_oos_val} Strategy config for '{self.strategy_name}' not found in AppConfig. Cannot proceed.")
            return [], None, None
        strategy_config_dict_for_evaluator = strategy_config_for_evaluator_obj.__dict__

        for rank_idx, current_is_trial_obj in enumerate(best_is_trials_from_study):
            is_trial_number_optuna = current_is_trial_obj.number
            trial_params = current_is_trial_obj.params
            
            is_metrics_values = current_is_trial_obj.values
            is_metrics_dict_from_values = {}
            if is_metrics_values:
                 for i, obj_name in enumerate(optuna_objectives_config['objectives_names']):
                    if i < len(is_metrics_values):
                        is_metrics_dict_from_values[obj_name] = is_metrics_values[i]
            
            full_is_metrics_from_attrs = current_is_trial_obj.user_attrs.get('full_is_metrics', {})
            final_is_metrics_dict = {**is_metrics_dict_from_values, **full_is_metrics_from_attrs}

            logger.info(f"{log_prefix_oos_val} Running OOS for IS Trial Optuna #{is_trial_number_optuna} (Rank {rank_idx+1}). Params: {trial_params}")

            oos_evaluator = objective_evaluator_class(
                strategy_name=self.strategy_name, 
                strategy_config_dict=strategy_config_dict_for_evaluator,
                df_enriched_slice=data_1min_cleaned_oos_slice.copy(),
                simulation_settings=self.app_config.global_config.simulation_defaults.__dict__,
                optuna_objectives_config=optuna_objectives_config,
                pair_symbol=self.pair_symbol,
                symbol_info_data=self.symbol_info_data, # Utiliser le symbol_info_data stocké
                app_config=self.app_config,
                is_oos_eval=True,
                is_trial_number_for_oos_log=is_trial_number_optuna
            )
            
            class FixedParamsTrialWrapper:
                def __init__(self, params: Dict[str, Any], number: int, distributions: Dict[str, Any]):
                    self.params = params
                    self.number = number
                    self.distributions = distributions 
                    self.user_attrs: Dict[str, Any] = {} 
                def suggest_float(self, name, low, high, step=None, log=False): return self.params[name]
                def suggest_int(self, name, low, high, step=1, log=False): return self.params[name]
                def suggest_categorical(self, name, choices): return self.params[name]
                def report(self, value: float, step: int) -> None: pass
                def should_prune(self) -> bool: return False
                def set_user_attr(self, key: str, value: Any) -> None: self.user_attrs[key] = value

            trial_wrapper_for_oos = FixedParamsTrialWrapper(trial_params, is_trial_number_optuna, current_is_trial_obj.distributions)

            oos_trades_df_for_trial: Optional[pd.DataFrame] = None
            oos_metrics_for_trial: Dict[str, Any] = {} 
            try:
                oos_objectives_values_tuple = oos_evaluator(trial_wrapper_for_oos) 
                
                for i, obj_name in enumerate(optuna_objectives_config['objectives_names']):
                    if i < len(oos_objectives_values_tuple):
                        oos_metrics_for_trial[obj_name] = oos_objectives_values_tuple[i]
                
                current_oos_backtest_results = oos_evaluator.last_backtest_results
                if current_oos_backtest_results:
                    full_oos_metrics = current_oos_backtest_results.get("metrics", {})
                    oos_metrics_for_trial.update(full_oos_metrics)

                    # Récupérer 'trades' qui est un DataFrame simple des trades
                    oos_trades_df_from_results_raw = current_oos_backtest_results.get("trades") 
                    if isinstance(oos_trades_df_from_results_raw, pd.DataFrame) and not oos_trades_df_from_results_raw.empty:
                        oos_trades_df_for_trial = oos_trades_df_from_results_raw.copy()
                        oos_trades_df_for_trial['is_trial_number_optuna'] = is_trial_number_optuna
                        oos_trades_df_for_trial['is_trial_rank_in_selection'] = rank_idx + 1
                        logger.debug(f"{log_prefix_oos_val} Récupéré {len(oos_trades_df_for_trial)} trades OOS pour IS trial Optuna #{is_trial_number_optuna}.")
                    else:
                        logger.info(f"{log_prefix_oos_val} Aucun trade OOS pour IS trial Optuna #{is_trial_number_optuna}.")
                else:
                    logger.warning(f"{log_prefix_oos_val} last_backtest_results non trouvé pour IS trial Optuna #{is_trial_number_optuna}.")

            except optuna.exceptions.TrialPruned as e_pruned_oos:
                 logger.warning(f"{log_prefix_oos_val} OOS eval pour IS Trial Optuna #{is_trial_number_optuna} pruné: {e_pruned_oos}")
                 for i, obj_name in enumerate(optuna_objectives_config['objectives_names']):
                    direction = optuna_objectives_config['objectives_directions'][i]
                    oos_metrics_for_trial[obj_name] = -float('inf') if direction == "maximize" else float('inf')
            except Exception as e_oos:
                logger.error(f"{log_prefix_oos_val} Erreur OOS eval pour IS Trial Optuna #{is_trial_number_optuna}: {e_oos}", exc_info=True)
                for i, obj_name in enumerate(optuna_objectives_config['objectives_names']):
                    direction = optuna_objectives_config['objectives_directions'][i]
                    oos_metrics_for_trial[obj_name] = -float('inf') if direction == "maximize" else float('inf')

            oos_validation_results.append({
                'is_trial_number_optuna': is_trial_number_optuna,
                'is_trial_rank_in_selection': rank_idx + 1,
                'params': trial_params,
                'is_metrics': final_is_metrics_dict, 
                'oos_metrics': oos_metrics_for_trial,
                'oos_trades_df': oos_trades_df_for_trial 
            })
        
        selected_best_overall_params: Optional[Dict[str, Any]] = None
        best_overall_oos_metrics: Optional[Dict[str, Any]] = None

        if oos_validation_results:
            selection_metric_name = self.app_config.global_config.wfo_settings.metric_to_optimize
            selection_direction = self.app_config.global_config.wfo_settings.optimization_direction
            
            best_oos_run_for_final_selection = None
            current_best_value = -float('inf') if selection_direction == "maximize" else float('inf')

            for oos_run in oos_validation_results:
                metric_val = oos_run.get('oos_metrics', {}).get(selection_metric_name)
                if metric_val is not None and np.isfinite(metric_val): # type: ignore
                    if (selection_direction == "maximize" and metric_val > current_best_value) or \
                       (selection_direction == "minimize" and metric_val < current_best_value): # type: ignore
                        current_best_value = metric_val # type: ignore
                        best_oos_run_for_final_selection = oos_run
            
            if best_oos_run_for_final_selection:
                selected_best_overall_params = best_oos_run_for_final_selection['params']
                best_overall_oos_metrics = best_oos_run_for_final_selection['oos_metrics']
                logger.info(f"{log_prefix_oos_val} Meilleur run OOS global (basé sur '{selection_metric_name}'): "
                            f"IS Trial Optuna #{best_oos_run_for_final_selection['is_trial_number_optuna']}, Métriques OOS: {best_overall_oos_metrics}")
            else:
                logger.warning(f"{log_prefix_oos_val} Aucun résultat OOS valide trouvé pour sélectionner le meilleur run overall selon '{selection_metric_name}'.")
        else:
            logger.warning(f"{log_prefix_oos_val} Aucune validation OOS n'a pu être effectuée.")

        return oos_validation_results, selected_best_overall_params, best_overall_oos_metrics

    def save_oos_validation_summary(self, oos_results: List[Dict[str, Any]]):
        if not oos_results:
            logger.info(f"{self.log_prefix} No OOS results to save in summary JSON.")
            return

        serializable_results = []
        for res_entry in oos_results:
            entry_copy = {k: v for k, v in res_entry.items() if k != 'oos_trades_df'} 

            if 'is_metrics' in entry_copy and isinstance(entry_copy['is_metrics'], dict):
                entry_copy['is_metrics'] = {
                    k_met: (v_met if isinstance(v_met, (int, float, str, bool)) and (not isinstance(v_met, float) or np.isfinite(v_met)) else str(v_met) if v_met is not None else None)
                    for k_met, v_met in entry_copy['is_metrics'].items()
                }
            if 'oos_metrics' in entry_copy and isinstance(entry_copy['oos_metrics'], dict):
                 entry_copy['oos_metrics'] = {
                    k_met: (v_met if isinstance(v_met, (int, float, str, bool)) and (not isinstance(v_met, float) or np.isfinite(v_met)) else str(v_met) if v_met is not None else None)
                    for k_met, v_met in entry_copy['oos_metrics'].items()
                }
            serializable_results.append(entry_copy)

        file_path = self.output_dir_fold / "oos_validation_summary_TOP_N_TRIALS.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=4, allow_nan=False) 
            logger.info(f"{self.log_prefix} OOS validation summary (sans trades DF) saved to: {file_path}")
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to save OOS validation summary JSON to {file_path}: {e}", exc_info=True)

    def select_final_parameters_for_live(self, oos_validation_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not oos_validation_results:
            logger.warning(f"{self.log_prefix} No OOS results provided to select final parameters.")
            return None

        selection_metric_name = self.app_config.global_config.wfo_settings.metric_to_optimize
        selection_direction = self.app_config.global_config.wfo_settings.optimization_direction
        
        logger.info(f"{self.log_prefix} Selecting final parameters based on OOS metric: '{selection_metric_name}' (Direction: {selection_direction})")

        best_oos_trial_info = None
        best_oos_metric_value = -float('inf') if selection_direction == "maximize" else float('inf')

        for result_entry in oos_validation_results:
            oos_metrics = result_entry.get("oos_metrics", {})
            current_metric_value = oos_metrics.get(selection_metric_name)

            if current_metric_value is None or not isinstance(current_metric_value, (int, float)) or not np.isfinite(current_metric_value):
                logger.debug(f"{self.log_prefix} Skipping OOS result for IS trial {result_entry.get('is_trial_number_optuna')} due to invalid OOS selection metric value: {current_metric_value}")
                continue

            if selection_direction == "maximize":
                if current_metric_value > best_oos_metric_value:
                    best_oos_metric_value = current_metric_value
                    best_oos_trial_info = result_entry
            else: 
                if current_metric_value < best_oos_metric_value:
                    best_oos_metric_value = current_metric_value
                    best_oos_trial_info = result_entry
        
        if best_oos_trial_info:
            final_params = best_oos_trial_info.get("params") 
            logger.info(f"{self.log_prefix} Final parameters selected from IS Trial {best_oos_trial_info.get('is_trial_number_optuna')} "
                        f"with OOS '{selection_metric_name}': {best_oos_metric_value:.4f}. Parameters: {final_params}")
            return final_params
        else:
            logger.warning(f"{self.log_prefix} Could not select final parameters. No OOS trial had a valid value for metric '{selection_metric_name}'.")
            return None
