import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, Tuple, List, TYPE_CHECKING

import pandas as pd # Assurez-vous que pandas est importé
import optuna
import json # Ajout de l'import pour la sauvegarde JSON, bien qu'il puisse être utilisé ailleurs

if TYPE_CHECKING:
    from src.config.definitions import AppConfig
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator
    # Garder les imports directs pour l'instanciation
from src.backtesting.optimizer.study_manager import StudyManager
from src.backtesting.optimizer.results_analyzer import ResultsAnalyzer


logger = logging.getLogger(__name__)

def run_optimization_for_fold(
    strategy_name: str,
    strategy_config_dict: Dict[str, Any],
    data_1min_cleaned_is_slice: pd.DataFrame,
    data_1min_cleaned_oos_slice: pd.DataFrame,
    app_config: 'AppConfig',
    output_dir_fold: Path,
    pair_symbol: str,
    symbol_info_data: Dict[str, Any],
    objective_evaluator_class: Type['ObjectiveEvaluator'],
    study_manager_class: Type[StudyManager],
    results_analyzer_class: Type[ResultsAnalyzer]
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:

    log_prefix_orchestrator = f"[{strategy_name}][{pair_symbol}][Fold: {output_dir_fold.name}]"
    logger.info(f"{log_prefix_orchestrator} Starting optimization orchestrator for fold.")

    if data_1min_cleaned_is_slice.empty:
        logger.error(f"{log_prefix_orchestrator} In-Sample data slice is empty. Aborting optimization for this fold.")
        return None, None

    logger.info(f"{log_prefix_orchestrator} Initializing StudyManager for IS optimization.")
    study_manager_instance = study_manager_class(
        app_config=app_config,
        strategy_name=strategy_name,
        strategy_config_dict=strategy_config_dict,
        study_output_dir=output_dir_fold,
        pair_symbol=pair_symbol,
        symbol_info_data=symbol_info_data
    )

    optuna_study_is: Optional[optuna.Study] = None
    try:
        logger.info(f"{log_prefix_orchestrator} Running IS study...")
        optuna_study_is = study_manager_instance.run_study(
            data_1min_cleaned_is_slice=data_1min_cleaned_is_slice,
            objective_evaluator_class=objective_evaluator_class
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
    results_analyzer_instance = results_analyzer_class(
        study=optuna_study_is,
        app_config=app_config,
        strategy_name=strategy_name,
        output_dir_fold=output_dir_fold,
        pair_symbol=pair_symbol,
        symbol_info_data=symbol_info_data, # <<< PASSER symbol_info_data
        fold_name_for_log=output_dir_fold.name
    )

    logger.info(f"{log_prefix_orchestrator} Analyzing IS results and selecting best trials for OOS validation...")
    best_is_trials: List[optuna.trial.FrozenTrial] = results_analyzer_instance.get_best_trials_for_oos_validation()

    representative_params: Optional[Dict[str, Any]] = None
    representative_oos_metrics: Optional[Dict[str, Any]] = None

    if not best_is_trials:
        logger.warning(f"{log_prefix_orchestrator} No suitable IS trials found for OOS validation after analysis. Skipping OOS for this fold.")
        if optuna_study_is.best_trials:
            best_is_trial_overall = optuna_study_is.best_trials[0]
            logger.info(f"{log_prefix_orchestrator} Falling back to overall best IS trial (Optuna Trial #{best_is_trial_overall.number}) due to no specific trials selected for OOS.")
            representative_params = best_is_trial_overall.params
            is_objectives = best_is_trial_overall.values
            is_metrics_dict = {}
            if is_objectives:
                for i, obj_name in enumerate(app_config.global_config.optuna_settings.objectives_names):
                    if i < len(is_objectives):
                        is_metrics_dict[obj_name] = is_objectives[i]

            representative_oos_metrics = {
                "IS_METRICS_FALLBACK": is_metrics_dict,
                "message": "Used best IS trial parameters as no trials were selected for OOS or OOS validation was skipped/failed."
            }
        else:
            logger.error(f"{log_prefix_orchestrator} No trials in IS Pareto front. Cannot select any parameters for this fold.")
    else:
        logger.info(f"{log_prefix_orchestrator} Running OOS validation for {len(best_is_trials)} selected IS trial(s).")

        oos_validation_results, final_selected_params, final_selected_oos_metrics = \
            results_analyzer_instance.run_oos_validation_for_best_is_trials(
                best_is_trials_from_study=best_is_trials,
                data_1min_cleaned_oos_slice=data_1min_cleaned_oos_slice,
                objective_evaluator_class=objective_evaluator_class
            )
        representative_params = final_selected_params
        representative_oos_metrics = final_selected_oos_metrics

        results_analyzer_instance.save_oos_validation_summary(oos_validation_results)

        # --- MODIFICATION START: Sauvegarde des trades OOS en Parquet ET JSON ---
        all_oos_trades_list: List[pd.DataFrame] = []
        if oos_validation_results:
            for oos_run_result in oos_validation_results:
                trades_df = oos_run_result.get('oos_trades_df')
                if trades_df is not None and not trades_df.empty:
                    all_oos_trades_list.append(trades_df)

        if all_oos_trades_list:
            aggregated_oos_trades_df = pd.concat(all_oos_trades_list, ignore_index=True)

            # 1. Sauvegarde en Parquet (existant)
            oos_trades_parquet_filename = output_dir_fold / "all_oos_trades_from_best_is_trials.parquet"
            try:
                aggregated_oos_trades_df.to_parquet(oos_trades_parquet_filename, index=False)
                logger.info(f"{log_prefix_orchestrator} Sauvegardé {len(aggregated_oos_trades_df)} trades OOS agrégés "
                            f"de {len(all_oos_trades_list)} runs OOS en Parquet: {oos_trades_parquet_filename}")
            except Exception as e_save_oos_trades_parquet:
                logger.error(f"{log_prefix_orchestrator} Erreur lors de la sauvegarde des trades OOS agrégés en Parquet: {e_save_oos_trades_parquet}", exc_info=True)

            # 2. NOUVEAU: Sauvegarde en JSON
            oos_trades_json_filename = output_dir_fold / "oos_trades_log.json"
            try:
                # La méthode to_json de pandas gère bien la conversion des Timestamps en ISO 8601 par défaut avec date_format='iso'
                # Il est préférable d'utiliser les types natifs de pandas autant que possible avant la conversion.
                # Si une colonne est déjà une chaîne formatée ISO, elle restera telle quelle.
                # Si une colonne est un objet datetime ou Timestamp, elle sera convertie.
                
                # Créer une copie pour éviter de modifier le DataFrame original utilisé pour le Parquet si des conversions sont nécessaires
                df_for_json = aggregated_oos_trades_df.copy()
                
                # S'assurer que les colonnes de timestamp sont au format string ISO pour JSON si elles sont de type datetime
                # (to_json avec date_format='iso' devrait s'en charger, mais une vérification/conversion explicite peut être plus robuste)
                for col in df_for_json.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_for_json[col]):
                        # Convertir en string ISO 8601 avec 'Z' pour UTC et millisecondes
                        # '%Y-%m-%dT%H:%M:%S.%f' génère 6 chiffres pour les microsecondes.
                        # '[:-3]' enlève les 3 derniers chiffres pour avoir des millisecondes.
                        df_for_json[col] = df_for_json[col].dt.strftime('%Y-%m-%dT%H:%M:%S.%f').str[:-3] + 'Z'
                    elif pd.api.types.is_timedelta64_any_dtype(df_for_json[col]):
                         # Convertir les Timedeltas en secondes totales (ou autre format string lisible)
                        df_for_json[col] = df_for_json[col].dt.total_seconds().astype(str) + " seconds"


                df_for_json.to_json(oos_trades_json_filename, orient="records", indent=4, date_format="iso", default_handler=str)
                logger.info(f"{log_prefix_orchestrator} Sauvegardé {len(df_for_json)} trades OOS agrégés "
                            f"de {len(all_oos_trades_list)} runs OOS en JSON: {oos_trades_json_filename}")
            except Exception as e_save_oos_trades_json:
                logger.error(f"{log_prefix_orchestrator} Erreur lors de la sauvegarde de oos_trades_log.json : {e_save_oos_trades_json}", exc_info=True)
        else:
            logger.info(f"{log_prefix_orchestrator} Aucun trade OOS n'a été trouvé ou généré par les meilleurs trials IS pour ce fold. "
                        "Aucun fichier de log des trades OOS (Parquet ou JSON) n'a été généré.")
        # --- MODIFICATION END ---
        
    if representative_params:
        logger.info(f"{log_prefix_orchestrator} Orchestration for fold complete. Representative Params: {representative_params}, OOS/Fallback Metrics: {representative_oos_metrics}")
    else:
        logger.error(f"{log_prefix_orchestrator} Orchestration for fold complete, but NO representative parameters could be selected.")

    return representative_params, representative_oos_metrics