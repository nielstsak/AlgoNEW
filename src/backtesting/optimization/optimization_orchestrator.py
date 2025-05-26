import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, Tuple, List, TYPE_CHECKING

import pandas as pd
import optuna
import json

if TYPE_CHECKING:
    from src.config.definitions import AppConfig, GlobalConfig, StrategyParamsConfig, ExchangeSettings, WfoSettings
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator

from src.backtesting.optimizer.study_manager import StudyManager
from src.backtesting.optimizer.results_analyzer import ResultsAnalyzer
from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator # Importation directe

logger = logging.getLogger(__name__)

def run_optimization_for_fold(
    strategy_name: str,
    strategy_config_dict: Dict[str, Any], 
    data_1min_cleaned_is_slice: pd.DataFrame,
    data_1min_cleaned_oos_slice: pd.DataFrame,
    app_config: 'AppConfig',
    output_dir_fold: Path,
    pair_symbol: str,
    symbol_info_data: Dict[str, Any], # C'est le pair_config
    objective_evaluator_class: Type['ObjectiveEvaluator'], # Peut être retiré si on instancie directement
    study_manager_class: Type[StudyManager], # Peut être retiré
    results_analyzer_class: Type[ResultsAnalyzer], # Peut être retiré
    run_id: str,
    context_label: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:

    log_prefix_orchestrator = f"[{strategy_name}][{pair_symbol}][Fold: {output_dir_fold.name}]"
    logger.info(f"{log_prefix_orchestrator} Starting optimization orchestrator for fold.")

    if data_1min_cleaned_is_slice.empty:
        logger.error(f"{log_prefix_orchestrator} In-Sample data slice is empty. Aborting optimization for this fold.")
        return None, None

    logger.info(f"{log_prefix_orchestrator} Initializing StudyManager for IS optimization.")
    study_manager_instance = StudyManager( # Instanciation directe
        app_config=app_config,
        strategy_name=strategy_name,
        strategy_config_dict=strategy_config_dict,
        study_output_dir=output_dir_fold,
        pair_symbol=pair_symbol,
        symbol_info_data=symbol_info_data, # Passe le pair_config
        run_id=run_id
    )

    optuna_study_is: Optional[optuna.Study] = None
    try:
        logger.info(f"{log_prefix_orchestrator} Running IS study...")
        optuna_study_is = study_manager_instance.run_study(
            data_1min_cleaned_is_slice=data_1min_cleaned_is_slice,
            objective_evaluator_class=ObjectiveEvaluator # Passer la classe directement
        )
        if not optuna_study_is :
            logger.error(f"{log_prefix_orchestrator} IS study did not return a valid study object or has no trials. Aborting fold.")
            return None, None
        logger.info(f"{log_prefix_orchestrator} IS study completed. Study name: {optuna_study_is.study_name}, "
                    f"Number of trials: {len(optuna_study_is.trials)}, "
                    f"Pareto front size: {len(optuna_study_is.best_trials) if optuna_study_is.best_trials else 0}")

    except Exception as e_is_study:
        logger.error(f"{log_prefix_orchestrator} Error during IS study execution: {e_is_study}", exc_info=True)
        return None, None

    logger.info(f"{log_prefix_orchestrator} Initializing ResultsAnalyzer for IS results.")
    
    strategy_params_config_obj = app_config.strategies_config.strategies.get(strategy_name)
    if not strategy_params_config_obj:
        logger.error(f"{log_prefix_orchestrator} Strategy config for '{strategy_name}' not found in AppConfig.strategies_config. Cannot initialize ResultsAnalyzer.")
        return None, None
    
    try:
        current_fold_id_int = int(output_dir_fold.name.split('_')[-1])
    except (ValueError, IndexError):
        logger.error(f"{log_prefix_orchestrator} Could not determine fold_id from output_dir_fold name: {output_dir_fold.name}. Using -1 as placeholder.")
        current_fold_id_int = -1

    fold_data_map_for_analyzer = {current_fold_id_int: (data_1min_cleaned_is_slice, data_1min_cleaned_oos_slice)} if current_fold_id_int != -1 else {}
    fold_dirs_map_for_analyzer = {current_fold_id_int: output_dir_fold} if current_fold_id_int != -1 else {}

    results_analyzer_instance = ResultsAnalyzer( # Instanciation directe
        run_id=run_id,
        strategy_name=strategy_name,
        pair_symbol=pair_symbol,
        context_label=context_label,
        study=optuna_study_is,
        app_config=app_config, # Passer AppConfig complet
        pair_config=symbol_info_data, # <<< PASSER symbol_info_data comme pair_config
        fold_data_map=fold_data_map_for_analyzer,
        fold_dirs_map=fold_dirs_map_for_analyzer
    )

    # analyze_and_save_results_for_fold est la méthode à appeler
    # Elle retourne: final_selected_params_for_fold, final_selected_oos_metrics_for_fold
    representative_params, representative_oos_metrics = results_analyzer_instance.analyze_and_save_results_for_fold(
        fold_id=current_fold_id_int
    )
        
    if representative_params:
        logger.info(f"{log_prefix_orchestrator} Orchestration for fold complete. Representative Params: {representative_params}, OOS/Fallback Metrics: {representative_oos_metrics}")
    else:
        logger.error(f"{log_prefix_orchestrator} Orchestration for fold complete, but NO representative parameters could be selected.")

    return representative_params, representative_oos_metrics
