# src/backtesting/optimization/fold_orchestrator.py
"""
Ce module est responsable de l'orchestration du processus complet pour un
seul fold d'Optimisation Walk-Forward (WFO). Il gère l'optimisation
In-Sample (IS) avec Optuna, suivie de la validation Out-of-Sample (OOS)
des meilleurs paramètres IS identifiés.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TYPE_CHECKING, cast

import pandas as pd
import optuna # Pour l'objet Study et les exceptions Optuna

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import StrategyParamsConfig # Pour le typage
    from src.backtesting.optimization.optuna_study_manager import OptunaStudyManager
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
    from src.backtesting.optimization.oos_validator import OOSValidator

# Imports des modules de l'application
try:
    from src.backtesting.optimization.optuna_study_manager import OptunaStudyManager
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
    from src.backtesting.optimization.oos_validator import OOSValidator
except ImportError as e:
    # Ce log est un fallback, le logging principal est configuré ailleurs
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"FoldOrchestrator: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH."
    )
    raise

logger = logging.getLogger(__name__)

def run_fold_optimization_and_validation(
    app_config: 'AppConfig',
    strategy_name: str,
    strategy_config_dict: Dict[str, Any], 
    data_1min_cleaned_is_slice: pd.DataFrame,
    data_1min_cleaned_oos_slice: pd.DataFrame,
    output_dir_fold: Path, 
    pair_symbol: str,
    symbol_info_data: Dict[str, Any], 
    run_id: str, 
    context_label: str, 
    fold_id_numeric: int
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Orchestre le processus complet pour un seul fold Walk-Forward Optimization (WFO).
    """
    log_prefix_fold_orch = f"[{strategy_name}/{pair_symbol.upper()}/{context_label}][Run:{run_id}][Fold:{fold_id_numeric}][FoldOrch]"
    logger.info(f"{log_prefix_fold_orch} Démarrage de l'orchestration pour le fold.")
    logger.debug(f"{log_prefix_fold_orch} Répertoire de sortie du fold : {output_dir_fold}")
    output_dir_fold.mkdir(parents=True, exist_ok=True)

    optuna_study_is: Optional[optuna.Study] = None
    try:
        study_manager_is = OptunaStudyManager(
            app_config=app_config,
            strategy_name=strategy_name,
            strategy_config_dict=strategy_config_dict,
            study_output_dir=output_dir_fold,
            pair_symbol=pair_symbol,
            symbol_info_data=symbol_info_data,
            run_id=run_id
        )
        
        logger.info(f"{log_prefix_fold_orch} Exécution de l'étude Optuna IS...")
        optuna_study_is = study_manager_is.run_study(
            data_1min_cleaned_is_slice=data_1min_cleaned_is_slice,
            objective_evaluator_class=ObjectiveFunctionEvaluator
        )

        if not optuna_study_is:
            logger.error(f"{log_prefix_fold_orch} L'étude Optuna IS n'a pas retourné d'objet Study valide. Abandon du fold.")
            return None, None
        
        completed_is_trials = [t for t in optuna_study_is.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_is_trials:
             logger.warning(f"{log_prefix_fold_orch} Étude Optuna IS terminée mais aucun essai complété trouvé. "
                            f"Nombre total d'essais dans l'étude: {len(optuna_study_is.trials)}.")
             return None, None
        elif not optuna_study_is.best_trials: # Check best_trials (plural)
            logger.warning(f"{log_prefix_fold_orch} Étude Optuna IS terminée, {len(completed_is_trials)} essais complétés, "
                           "mais study.best_trials est vide. La validation OOS pourrait ne pas trouver de paramètres.")
        else:
            best_is_values_log = optuna_study_is.best_trials[0].values if optuna_study_is.best_trials else 'N/A'
            logger.info(f"{log_prefix_fold_orch} Optimisation IS terminée. Étude: {optuna_study_is.study_name}. "
                        f"Essais complétés: {len(completed_is_trials)}. "
                        f"Meilleur(s) essai(s) IS (valeurs du premier du front de Pareto): {best_is_values_log}")

    except optuna.exceptions.TrialPruned as e_pruned_is:
        logger.warning(f"{log_prefix_fold_orch} Un essai IS a été élagué pendant l'optimisation : {e_pruned_is}")
    except Exception as e_is:
        logger.error(f"{log_prefix_fold_orch} Erreur critique durant l'optimisation IS : {e_is}", exc_info=True)
        return None, None

    logger.info(f"{log_prefix_fold_orch} Phase de validation Out-of-Sample (OOS)...")
    best_params_for_fold_final: Optional[Dict[str, Any]] = None
    corresponding_metrics_final: Optional[Dict[str, Any]] = None

    # Déterminer le meilleur essai IS pour le fallback
    best_is_trial_for_fallback: Optional[optuna.trial.FrozenTrial] = None
    if optuna_study_is:
        if optuna_study_is.best_trials: # Pour multi-objectif, c'est une liste
            best_is_trial_for_fallback = optuna_study_is.best_trials[0]
        elif optuna_study_is.best_trial: # Pour mono-objectif (si best_trials est vide mais best_trial existe)
            best_is_trial_for_fallback = optuna_study_is.best_trial

    if data_1min_cleaned_oos_slice.empty:
        logger.warning(f"{log_prefix_fold_orch} Données Out-of-Sample vides. Validation OOS sautée. Utilisation des meilleurs paramètres IS si disponibles.")
        if best_is_trial_for_fallback:
            best_params_for_fold_final = best_is_trial_for_fallback.params
            corresponding_metrics_final = {
                "status_oos": "SKIPPED_NO_OOS_DATA_FALLBACK_TO_BEST_IS",
                "selected_params_from_is": True,
                "is_metrics_for_best_trial": best_is_trial_for_fallback.values,
            }
            for attr_name, attr_val in best_is_trial_for_fallback.user_attrs.items():
                corresponding_metrics_final[f"is_attr_{attr_name}"] = attr_val
            logger.info(f"{log_prefix_fold_orch} Retour des meilleurs paramètres IS (du trial {best_is_trial_for_fallback.number}) en raison de l'absence de données OOS : {best_params_for_fold_final}")
        else:
            logger.error(f"{log_prefix_fold_orch} Pas de données OOS et aucun meilleur essai IS trouvé. Impossible de sélectionner des paramètres.")
            return None, None
    elif not optuna_study_is or not completed_is_trials: # Vérifier completed_is_trials au lieu de optuna_study_is.best_trials directement ici
        logger.warning(f"{log_prefix_fold_orch} Aucune étude IS valide ou aucun essai IS complété. Impossible de procéder à la validation OOS.")
        return None, None
    else:
        try:
            oos_validator = OOSValidator(
                run_id=run_id,
                strategy_name=strategy_name,
                pair_symbol=pair_symbol,
                context_label=context_label,
                study_is=optuna_study_is,
                app_config=app_config,
                pair_config=symbol_info_data,
                objective_evaluator_class=ObjectiveFunctionEvaluator
            )
            
            logger.info(f"{log_prefix_fold_orch} Analyse des résultats IS et validation OOS via OOSValidator...")
            best_params_for_fold_final, corresponding_metrics_final = oos_validator.analyze_and_save_results_for_fold(
                fold_id_numeric=fold_id_numeric,
                df_oos_fold_data=data_1min_cleaned_oos_slice,
                fold_output_dir=output_dir_fold
            )

            if best_params_for_fold_final and corresponding_metrics_final:
                oos_pnl_log = corresponding_metrics_final.get('Total Net PnL USDC', 'N/A')
                logger.info(f"{log_prefix_fold_orch} Validation OOS terminée. "
                            f"Meilleurs paramètres (issus de IS, sélectionnés sur OOS) : {best_params_for_fold_final}. "
                            f"Métriques OOS correspondantes (PnL): {oos_pnl_log}")
            elif best_params_for_fold_final and not corresponding_metrics_final:
                 logger.warning(f"{log_prefix_fold_orch} Validation OOS n'a pas retourné de métriques claires, mais des paramètres IS ont été sélectionnés.")
                 corresponding_metrics_final = {"status_oos": "COMPLETED_NO_METRICS", "selected_params_from_is": True}
            else: 
                logger.warning(f"{log_prefix_fold_orch} Validation OOS n'a pas permis de sélectionner de paramètres finaux pour ce fold.")
                if best_is_trial_for_fallback:
                    logger.info(f"{log_prefix_fold_orch} Fallback sur les meilleurs paramètres IS (du trial {best_is_trial_for_fallback.number}) car OOS a échoué à sélectionner.")
                    best_params_for_fold_final = best_is_trial_for_fallback.params
                    corresponding_metrics_final = {
                        "status_oos": f"FAILED_FALLBACK_TO_IS",
                        "selected_params_from_is": True,
                        "is_metrics_for_best_trial": best_is_trial_for_fallback.values,
                    }
                    for attr_name, attr_val in best_is_trial_for_fallback.user_attrs.items():
                        corresponding_metrics_final[f"is_attr_{attr_name}"] = attr_val
                else:
                    best_params_for_fold_final = None
                    corresponding_metrics_final = {"status_oos": "FAILED_NO_IS_BEST_TRIAL_FOR_FALLBACK"}
        except Exception as e_oos:
            logger.error(f"{log_prefix_fold_orch} Erreur critique durant la validation OOS : {e_oos}", exc_info=True)
            if best_is_trial_for_fallback:
                logger.warning(f"{log_prefix_fold_orch} Erreur OOS, retour des meilleurs paramètres IS (du trial {best_is_trial_for_fallback.number}) comme fallback.")
                best_params_for_fold_final = best_is_trial_for_fallback.params
                corresponding_metrics_final = {
                    "status_oos": f"ERROR_FALLBACK_TO_IS: {str(e_oos)[:100]}",
                    "selected_params_from_is": True,
                    "is_metrics_for_best_trial": best_is_trial_for_fallback.values,
                }
                for attr_name, attr_val in best_is_trial_for_fallback.user_attrs.items():
                    corresponding_metrics_final[f"is_attr_{attr_name}"] = attr_val
            else:
                best_params_for_fold_final = None
                corresponding_metrics_final = {"status_oos": f"ERROR_NO_IS_FALLBACK: {str(e_oos)[:100]}"}

    logger.info(f"{log_prefix_fold_orch} Orchestration du fold terminée.")
    return best_params_for_fold_final, corresponding_metrics_final
