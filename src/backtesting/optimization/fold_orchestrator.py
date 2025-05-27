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
    # Les imports suivants sont pour les types des classes qui seront réellement utilisées
    from src.backtesting.optimization.optuna_study_manager import OptunaStudyManager
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
    from src.backtesting.optimization.oos_validator import OOSValidator

# Imports des modules de l'application
try:
    from src.backtesting.optimization.optuna_study_manager import OptunaStudyManager
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
    from src.backtesting.optimization.oos_validator import OOSValidator
    # Les interfaces pour les dépendances injectées sont nécessaires si on les type explicitement ici
    # from src.core.interfaces import IStrategyLoader, ICacheManager, IErrorHandler
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"FoldOrchestrator: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH."
    )
    raise

logger = logging.getLogger(__name__)

def run_fold_optimization_and_validation(
    app_config: 'AppConfig', # AppConfig contient maintenant les instances des services
    strategy_name: str,
    strategy_config_dict: Dict[str, Any], # Ceci est StrategyParamsConfig asdict()
    data_1min_cleaned_is_slice: pd.DataFrame,
    data_1min_cleaned_oos_slice: pd.DataFrame,
    output_dir_fold: Path,
    pair_symbol: str,
    symbol_info_data: Dict[str, Any], # C'est le pair_config de l'exchange
    run_id: str, # ID du WFOManager/Task parent
    context_label: str, # Label de contexte CLI global
    fold_id_numeric: int # ID numérique du fold (0, 1, 2...)
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Orchestre le processus complet pour un seul fold Walk-Forward Optimization (WFO).

    Args:
        app_config (AppConfig): Configuration globale de l'application, incluant
                                les instances des services (cache_manager, strategy_loader, etc.).
        strategy_name (str): Nom de la stratégie.
        strategy_config_dict (Dict[str, Any]): Configuration de la stratégie (default_params, params_space).
        data_1min_cleaned_is_slice (pd.DataFrame): Données In-Sample pour ce fold.
        data_1min_cleaned_oos_slice (pd.DataFrame): Données Out-of-Sample pour ce fold.
        output_dir_fold (Path): Répertoire de sortie spécifique à ce fold (ex: .../TASK_ID/fold_0/).
        pair_symbol (str): Symbole de la paire.
        symbol_info_data (Dict[str, Any]): Informations de l'exchange pour la paire.
        run_id (str): ID du run WFO de la tâche parente.
        context_label (str): Label de contexte CLI.
        fold_id_numeric (int): Identifiant numérique du fold.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
            - Meilleurs paramètres IS sélectionnés (basés sur la performance OOS ou fallback IS).
            - Métriques OOS correspondantes (ou métriques IS si fallback).
              Le dictionnaire de métriques peut inclure une clé 'status_oos' pour indiquer
              comment les résultats ont été obtenus (ex: "SUCCESS_OOS", "FALLBACK_IS").
    """
    log_prefix_fold_orch = f"[{strategy_name}/{pair_symbol.upper()}/{context_label}][Run:{run_id}][Fold:{fold_id_numeric}][FoldOrchV2]"
    logger.info(f"{log_prefix_fold_orch} Démarrage de l'orchestration pour le fold.")
    logger.debug(f"{log_prefix_fold_orch} Répertoire de sortie du fold : {output_dir_fold}")
    output_dir_fold.mkdir(parents=True, exist_ok=True)

    # Vérifier la présence des instances de service dans AppConfig
    if not app_config.strategy_loader_instance or \
       not app_config.cache_manager_instance or \
       not app_config.error_handler_instance:
        msg = "Une ou plusieurs instances de service (StrategyLoader, CacheManager, ErrorHandler) sont manquantes dans AppConfig."
        logger.critical(f"{log_prefix_fold_orch} {msg}")
        raise ValueError(msg)

    optuna_study_is: Optional[optuna.Study] = None
    completed_is_trials: List[optuna.trial.FrozenTrial] = []

    # --- Phase 1: Optimisation In-Sample (IS) ---
    try:
        logger.info(f"{log_prefix_fold_orch} Phase d'optimisation In-Sample (IS)...")
        study_manager_is = OptunaStudyManager(
            app_config=app_config, # Passe l'AppConfig complète
            strategy_name=strategy_name,
            strategy_config_dict=strategy_config_dict,
            study_output_dir=output_dir_fold, # Le manager Optuna créera sa DB ici
            pair_symbol=pair_symbol,
            symbol_info_data=symbol_info_data,
            run_id=run_id # ID de la tâche WFO
        )

        # La classe ObjectiveFunctionEvaluator est passée, elle sera instanciée par OptunaStudyManager
        # avec les dépendances nécessaires tirées de app_config.
        optuna_study_is = study_manager_is.run_study(
            data_1min_cleaned_is_slice=data_1min_cleaned_is_slice,
            objective_evaluator_class=ObjectiveFunctionEvaluator # Passer la classe
        )

        if not optuna_study_is:
            logger.error(f"{log_prefix_fold_orch} L'étude Optuna IS n'a pas retourné d'objet Study valide. Abandon du fold.")
            return None, {"status_oos": "FAILURE_IS_STUDY_INVALID", "message": "IS study object was None."}

        completed_is_trials = [t for t in optuna_study_is.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_is_trials:
             logger.warning(f"{log_prefix_fold_orch} Étude Optuna IS terminée mais aucun essai complété trouvé. "
                            f"Nombre total d'essais: {len(optuna_study_is.trials)}.")
             # Pas besoin de retourner ici, OOSValidator gérera ce cas.
        elif not optuna_study_is.best_trials:
            logger.warning(f"{log_prefix_fold_orch} Étude IS: {len(completed_is_trials)} essais complétés, mais study.best_trials est vide.")
        else:
            best_is_values_log = optuna_study_is.best_trials[0].values if optuna_study_is.best_trials and optuna_study_is.best_trials[0].values else 'N/A'
            logger.info(f"{log_prefix_fold_orch} Optimisation IS terminée. Étude: {optuna_study_is.study_name}. "
                        f"Essais complétés: {len(completed_is_trials)}. "
                        f"Meilleur(s) IS (valeurs 1er front): {best_is_values_log}")

    except optuna.exceptions.TrialPruned as e_pruned_is:
        logger.warning(f"{log_prefix_fold_orch} Un essai IS a été élagué durant l'optimisation : {e_pruned_is}")
        # Continuer si possible, OOSValidator gérera l'absence de best_trials si tous sont élagués
    except Exception as e_is:
        logger.error(f"{log_prefix_fold_orch} Erreur critique durant l'optimisation IS : {e_is}", exc_info=True)
        return None, {"status_oos": "FAILURE_IS_OPTIMIZATION", "message": str(e_is)}


    # --- Phase 2: Validation Out-of-Sample (OOS) ---
    logger.info(f"{log_prefix_fold_orch} Phase de validation Out-of-Sample (OOS)...")
    best_params_for_fold_final: Optional[Dict[str, Any]] = None
    corresponding_metrics_final: Optional[Dict[str, Any]] = None

    # Déterminer le meilleur essai IS pour un fallback si OOS échoue ou n'est pas possible
    best_is_trial_for_fallback: Optional[optuna.trial.FrozenTrial] = None
    if optuna_study_is: # S'assurer que l'étude IS existe
        if optuna_study_is.best_trials:
            best_is_trial_for_fallback = optuna_study_is.best_trials[0]
        elif hasattr(optuna_study_is, 'best_trial') and optuna_study_is.best_trial:
            best_is_trial_for_fallback = optuna_study_is.best_trial

    if data_1min_cleaned_oos_slice.empty:
        logger.warning(f"{log_prefix_fold_orch} Données Out-of-Sample vides. Validation OOS sautée.")
        if best_is_trial_for_fallback and best_is_trial_for_fallback.params:
            best_params_for_fold_final = best_is_trial_for_fallback.params
            corresponding_metrics_final = {
                "status_oos": "SKIPPED_NO_OOS_DATA_FALLBACK_TO_BEST_IS",
                "selected_params_from_is": True,
                "is_metrics_for_best_trial": best_is_trial_for_fallback.values,
                "source_is_trial_number": best_is_trial_for_fallback.number
            }
            # Ajouter les user_attrs du trial IS aux métriques pour plus de contexte
            for attr_name, attr_val in best_is_trial_for_fallback.user_attrs.items():
                corresponding_metrics_final[f"is_attr_{attr_name}"] = attr_val
            logger.info(f"{log_prefix_fold_orch} Retour des meilleurs params IS (trial {best_is_trial_for_fallback.number}) "
                        f"car données OOS vides: {best_params_for_fold_final}")
        else:
            logger.error(f"{log_prefix_fold_orch} Pas de données OOS et aucun meilleur essai IS valide trouvé. "
                         "Impossible de sélectionner des paramètres pour ce fold.")
            return None, {"status_oos": "FAILURE_NO_OOS_DATA_NO_IS_FALLBACK"}

    elif not optuna_study_is or not completed_is_trials:
        logger.warning(f"{log_prefix_fold_orch} Aucune étude IS valide ou aucun essai IS complété. Validation OOS impossible.")
        return None, {"status_oos": "FAILURE_NO_VALID_IS_STUDY_OR_TRIALS"}
    else:
        try:
            oos_validator = OOSValidator(
                run_id=run_id,
                strategy_name=strategy_name,
                pair_symbol=pair_symbol,
                context_label=context_label,
                study_is=optuna_study_is, # Passer l'étude IS complétée
                app_config=app_config,   # Passer AppConfig pour accès aux services
                pair_config=symbol_info_data,
                objective_evaluator_class=ObjectiveFunctionEvaluator # Passer la classe
            )

            logger.info(f"{log_prefix_fold_orch} Analyse des résultats IS et validation OOS via OOSValidator...")
            best_params_for_fold_final, corresponding_metrics_final = oos_validator.analyze_and_save_results_for_fold(
                fold_id_numeric=fold_id_numeric,
                df_oos_fold_data=data_1min_cleaned_oos_slice,
                fold_output_dir=output_dir_fold # Le validateur OOS sauvegarde ses artefacts ici
            )

            if best_params_for_fold_final and corresponding_metrics_final:
                oos_pnl_log = corresponding_metrics_final.get('Total Net PnL USDC', 'N/A')
                oos_status_log = corresponding_metrics_final.get('status_oos', corresponding_metrics_final.get('selection_basis', 'N/A'))
                logger.info(f"{log_prefix_fold_orch} Validation OOS terminée. "
                            f"Params sélectionnés: {str(best_params_for_fold_final)[:100]}... "
                            f"Statut OOS: {oos_status_log}, PnL OOS (si dispo): {oos_pnl_log}")
            elif best_params_for_fold_final and not corresponding_metrics_final:
                 logger.warning(f"{log_prefix_fold_orch} Validation OOS: params IS sélectionnés mais pas de métriques OOS claires.")
                 corresponding_metrics_final = {"status_oos": "COMPLETED_OOS_NO_CLEAR_METRICS", "selected_params_from_is": True}
            else: # Aucun paramètre sélectionné par OOSValidator
                logger.warning(f"{log_prefix_fold_orch} Validation OOS n'a pas sélectionné de paramètres finaux.")
                if best_is_trial_for_fallback and best_is_trial_for_fallback.params:
                    logger.info(f"{log_prefix_fold_orch} Fallback sur meilleurs params IS (trial {best_is_trial_for_fallback.number}).")
                    best_params_for_fold_final = best_is_trial_for_fallback.params
                    corresponding_metrics_final = {
                        "status_oos": "FAILED_OOS_VALIDATION_FALLBACK_TO_BEST_IS",
                        "selected_params_from_is": True,
                        "is_metrics_for_best_trial": best_is_trial_for_fallback.values,
                        "source_is_trial_number": best_is_trial_for_fallback.number
                    }
                    for attr_name, attr_val in best_is_trial_for_fallback.user_attrs.items():
                        corresponding_metrics_final[f"is_attr_{attr_name}"] = attr_val
                else:
                    best_params_for_fold_final = None
                    corresponding_metrics_final = {"status_oos": "FAILED_OOS_AND_NO_IS_FALLBACK"}

        except Exception as e_oos:
            logger.error(f"{log_prefix_fold_orch} Erreur critique durant la validation OOS : {e_oos}", exc_info=True)
            if best_is_trial_for_fallback and best_is_trial_for_fallback.params:
                logger.warning(f"{log_prefix_fold_orch} Erreur OOS, retour des meilleurs params IS (trial {best_is_trial_for_fallback.number}) en fallback.")
                best_params_for_fold_final = best_is_trial_for_fallback.params
                corresponding_metrics_final = {
                    "status_oos": f"ERROR_OOS_FALLBACK_TO_IS: {str(e_oos)[:100]}",
                    "selected_params_from_is": True,
                    "is_metrics_for_best_trial": best_is_trial_for_fallback.values,
                    "source_is_trial_number": best_is_trial_for_fallback.number
                }
                for attr_name, attr_val in best_is_trial_for_fallback.user_attrs.items():
                    corresponding_metrics_final[f"is_attr_{attr_name}"] = attr_val
            else:
                best_params_for_fold_final = None
                corresponding_metrics_final = {"status_oos": f"ERROR_OOS_NO_IS_FALLBACK: {str(e_oos)[:100]}"}

    logger.info(f"{log_prefix_fold_orch} Orchestration du fold terminée.")
    return best_params_for_fold_final, corresponding_metrics_final

