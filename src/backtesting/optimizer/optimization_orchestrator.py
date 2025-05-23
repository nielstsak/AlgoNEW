import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TYPE_CHECKING, List

import pandas as pd
import optuna # For type hint optuna.Study

if TYPE_CHECKING:
    from src.config.definitions import AppConfig # Ou from src.config.loader import AppConfig
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator
    from src.backtesting.optimizer.study_manager import StudyManager
    from src.backtesting.optimizer.results_analyzer import ResultsAnalyzer

logger = logging.getLogger(__name__)

def run_optimization_for_fold(
    strategy_name: str,
    strategy_config_dict: Dict[str, Any],
    data_1min_cleaned_is_slice: pd.DataFrame, # This is df_enriched_slice for IS
    data_1min_cleaned_oos_slice: Optional[pd.DataFrame], # This is df_enriched_slice for OOS
    app_config: 'AppConfig',
    output_dir_fold: Path, # Specific path for this fold (e.g., .../fold_X/)
    pair_symbol: str,
    symbol_info_data: Dict[str, Any],
    # Classes are passed for dependency injection / mock testing
    objective_evaluator_class: Type['ObjectiveEvaluator'],
    study_manager_class: Type['StudyManager'],
    results_analyzer_class: Type['ResultsAnalyzer']
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Orchestrates the complete optimization process for a single fold of a Walk-Forward Optimization.

    Args:
        strategy_name: Name of the strategy being optimized.
        strategy_config_dict: Configuration dictionary for the strategy (params_space, etc.).
        data_1min_cleaned_is_slice: Enriched DataFrame for In-Sample optimization.
        data_1min_cleaned_oos_slice: Optional enriched DataFrame for Out-of-Sample validation.
        app_config: The application's configuration object.
        output_dir_fold: The specific directory for this fold's artifacts (e.g., .../fold_X/).
        pair_symbol: The trading pair symbol.
        symbol_info_data: Exchange-provided information about the symbol.
        objective_evaluator_class: The class for ObjectiveEvaluator.
        study_manager_class: The class for StudyManager.
        results_analyzer_class: The class for ResultsAnalyzer.

    Returns:
        A tuple containing:
            - Optional[Dict[str, Any]]: The final selected parameters for this fold.
            - Optional[Dict[str, Any]]: Representative OOS metrics for the selected parameters,
                                         or IS metrics if OOS was not performed/successful.
    """
    fold_log_prefix = f"[{strategy_name}][{pair_symbol}][Fold: {output_dir_fold.name}]"
    logger.info(f"{fold_log_prefix} Starting optimization orchestrator for fold.")

    # --- Validate Input Data ---
    if data_1min_cleaned_is_slice.empty:
        logger.error(f"{fold_log_prefix} In-Sample data slice is empty. Aborting optimization for this fold.")
        return None, None

    # --- In-Sample (IS) Optimization ---
    logger.info(f"{fold_log_prefix} Initializing StudyManager for IS optimization.")
    study_manager_instance = study_manager_class(
        app_config=app_config,
        strategy_name=strategy_name,
        strategy_config_dict=strategy_config_dict,
        study_output_dir=output_dir_fold, # Specific directory for this fold's Optuna DB
        pair_symbol=pair_symbol,
        symbol_info_data=symbol_info_data
    )

    is_study: Optional[optuna.Study] = None
    try:
        logger.info(f"{fold_log_prefix} Running IS study...")
        is_study = study_manager_instance.run_study(
            data_1min_cleaned_is_slice=data_1min_cleaned_is_slice,
            objective_evaluator_class=objective_evaluator_class
        )
        if is_study:
            logger.info(f"{fold_log_prefix} IS study completed. Study name: {is_study.study_name}, "
                        f"Number of trials: {len(is_study.trials)}, "
                        f"Best trial(s) (Pareto front size): {len(is_study.best_trials) if is_study.best_trials else 0}")
        else:
            logger.error(f"{fold_log_prefix} IS study did not return a valid study object. Aborting fold.")
            return None, None
    except Exception as e_is_study:
        logger.error(f"{fold_log_prefix} Error during IS study execution: {e_is_study}", exc_info=True)
        return None, None

    # --- Results Analysis and Out-of-Sample (OOS) Validation ---
    logger.info(f"{fold_log_prefix} Initializing ResultsAnalyzer.")
    results_analyzer_instance = results_analyzer_class(
        app_config=app_config,
        study=is_study, # Pass the completed IS study
        objective_evaluator_class=objective_evaluator_class,
        strategy_name=strategy_name,
        strategy_config_dict=strategy_config_dict,
        output_dir_fold=output_dir_fold,
        pair_symbol=pair_symbol,
        symbol_info_data=symbol_info_data
    )

    # Select Best IS Trials for OOS Validation
    num_top_trials_to_validate = app_config.global_config.optuna_settings.n_best_for_oos
    logger.info(f"{fold_log_prefix} Selecting top {num_top_trials_to_validate} IS trials for OOS validation.")
    selected_is_trials: List[optuna.trial.FrozenTrial] = results_analyzer_instance._select_n_best_trials_from_pareto(
        n=num_top_trials_to_validate
    )

    oos_validation_results: List[Dict[str, Any]] = []
    final_selected_params: Optional[Dict[str, Any]] = None
    representative_oos_metrics: Optional[Dict[str, Any]] = None

    if not selected_is_trials:
        logger.warning(f"{fold_log_prefix} No IS trials were selected from Pareto front for OOS validation.")
        if is_study.best_trials: # If Pareto front exists but selection failed (e.g., all filtered out)
            best_is_trial_overall = is_study.best_trials[0]
            logger.info(f"{fold_log_prefix} Falling back to the overall best IS trial (Trial {best_is_trial_overall.number}) due to no trials selected for OOS.")
            final_selected_params = best_is_trial_overall.params
            representative_oos_metrics = {
                "IS_METRICS_FALLBACK": best_is_trial_overall.values, # Store as tuple or list
                "message": "Used best IS trial parameters as no trials were selected for OOS or OOS validation was skipped/failed."
            }
            logger.warning(f"{fold_log_prefix} Parameters from best IS trial (Number: {best_is_trial_overall.number}): {final_selected_params}")
            logger.warning(f"{fold_log_prefix} IS Metrics for this fallback: {representative_oos_metrics['IS_METRICS_FALLBACK']}")
        else: # No trials in Pareto front at all
            logger.error(f"{fold_log_prefix} No trials in IS Pareto front. Cannot select any parameters for this fold.")
            return None, None # No IS trials, so no params to select
    else:
        # Proceed with OOS validation if trials were selected
        if data_1min_cleaned_oos_slice is not None and not data_1min_cleaned_oos_slice.empty:
            logger.info(f"{fold_log_prefix} Running OOS validation for {len(selected_is_trials)} selected IS trials.")
            oos_validation_results = results_analyzer_instance.run_oos_validation_for_trials(
                selected_is_trials=selected_is_trials,
                data_1min_cleaned_oos_slice=data_1min_cleaned_oos_slice
            )
            results_analyzer_instance.save_oos_validation_results(oos_validation_results)
        else:
            logger.warning(f"{fold_log_prefix} OOS data slice is empty or not provided. Skipping OOS validation.")
            oos_validation_results = [] # Ensure it's an empty list

        # Select Final Parameters based on OOS (or IS fallback if OOS fails to yield a selection)
        logger.info(f"{fold_log_prefix} Selecting final parameters based on OOS results (if available).")
        final_selected_params = results_analyzer_instance.select_final_parameters_for_live(oos_validation_results)

        if final_selected_params:
            # Find the OOS metrics for the selected parameters
            for result_entry in oos_validation_results:
                if result_entry.get("is_trial_params") == final_selected_params:
                    representative_oos_metrics = result_entry.get("oos_metrics")
                    logger.info(f"{fold_log_prefix} Found OOS metrics for final selected parameters: {representative_oos_metrics}")
                    break
            if not representative_oos_metrics: # Should ideally not happen if final_selected_params came from oos_results
                 logger.warning(f"{fold_log_prefix} OOS metrics for final_selected_params not found in oos_validation_results. This is unexpected.")
                 # Fallback: try to find the best OOS metrics overall if selection was unclear
                 if oos_validation_results:
                     best_pnl_oos = -float('inf')
                     for res in oos_validation_results:
                         pnl = res.get("oos_metrics", {}).get("Total Net PnL USDC", -float('inf'))
                         if isinstance(pnl, (int, float)) and pnl > best_pnl_oos:
                             best_pnl_oos = pnl
                             representative_oos_metrics = res.get("oos_metrics")
                     logger.info(f"{fold_log_prefix} Using overall best OOS metrics as representative: {representative_oos_metrics}")


        elif selected_is_trials: # OOS validation did not yield a selection, but we have good IS trials
            logger.warning(f"{fold_log_prefix} No parameters selected based on OOS validation. Falling back to best IS trial from the selected list for OOS.")
            # selected_is_trials is already sorted by _select_n_best_trials_from_pareto
            best_is_trial_for_fallback = selected_is_trials[0]
            final_selected_params = best_is_trial_for_fallback.params
            representative_oos_metrics = {
                "IS_METRICS_FALLBACK": best_is_trial_for_fallback.values,
                "message": "Used best IS trial from OOS candidates as OOS validation did not yield a clear winner or was skipped."
            }
            logger.info(f"{fold_log_prefix} Fallback to IS Trial {best_is_trial_for_fallback.number}. Params: {final_selected_params}. IS Metrics: {representative_oos_metrics['IS_METRICS_FALLBACK']}")
        else: # Should have been caught earlier if selected_is_trials was empty and no IS best_trials
             logger.error(f"{fold_log_prefix} Critical logic error: No selected_is_trials and no IS fallback path taken previously.")
             return None, None


    # --- Final Logging and Return ---
    if final_selected_params:
        logger.info(f"{fold_log_prefix} Orchestration for fold complete.")
        logger.info(f"{fold_log_prefix} Final Selected Parameters: {final_selected_params}")
        if representative_oos_metrics:
            logger.info(f"{fold_log_prefix} Representative OOS/Fallback Metrics: {representative_oos_metrics}")
        else:
            logger.warning(f"{fold_log_prefix} No representative OOS/Fallback metrics available for the selected parameters.")
    else:
        logger.error(f"{fold_log_prefix} Orchestration for fold complete, but NO final parameters could be selected.")

    return final_selected_params, representative_oos_metrics
