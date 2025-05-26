# src/backtesting/optimization/oos_validator.py
"""
Ce module définit la classe OOSValidator, responsable de la phase de validation
Out-of-Sample (OOS) pour un fold WFO donné. Il analyse l'étude Optuna
In-Sample (IS) terminée, sélectionne les meilleurs essais IS, exécute des
backtests OOS pour ces essais, et sauvegarde les résultats.
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Type, TYPE_CHECKING, cast

import pandas as pd
import numpy as np # Pour np.isfinite
import optuna

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import WfoSettings, StrategyParamsConfig, SimulationDefaults
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
    # BaseStrategy n'est pas directement instanciée ici, mais ObjectiveFunctionEvaluator le fait.

# Imports depuis l'application
try:
    from src.utils import file_utils # Pour save_json
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
except ImportError as e:
    # Ce log est un fallback, le logging principal est configuré ailleurs
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"OOSValidator: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH."
    )
    raise

logger = logging.getLogger(__name__)

class OOSValidator:
    """
    Valide les paramètres In-Sample (IS) sur les données Out-of-Sample (OOS)
    pour un fold WFO spécifique.
    """
    def __init__(self,
                 run_id: str, # ID du run WFO global (celui du WFOManager pour la tâche)
                 strategy_name: str,
                 pair_symbol: str,
                 context_label: str, # Label de contexte (sanitized) pour ce run WFO
                 study_is: optuna.Study, # Étude Optuna de la phase IS pour le fold actuel
                 app_config: 'AppConfig',
                 pair_config: Dict[str, Any], # Infos de l'exchange (anciennement symbol_info_data)
                 objective_evaluator_class: Type['ObjectiveFunctionEvaluator'] # La classe, pas une instance
                ):
        """
        Initialise OOSValidator.
        """
        self.run_id = run_id
        self.strategy_name = strategy_name
        self.pair_symbol = pair_symbol.upper()
        self.context_label = context_label
        self.study_is = study_is
        self.app_config = app_config
        self.pair_config = pair_config

        self.objective_evaluator_class = objective_evaluator_class

        fold_name_from_study = self.study_is.study_name.split('_')[-2] if self.study_is.study_name and '_fold_' in self.study_is.study_name else "unknown_fold"
        self.log_prefix = f"[{self.strategy_name}/{self.pair_symbol}/{self.context_label}][Run:{self.run_id}][{fold_name_from_study}][OOSValidator]"

        self.wfo_settings: 'WfoSettings' = self.app_config.global_config.wfo_settings
        self.strategy_config_obj_dict: Optional[Dict[str, Any]] = None # Sera le dict de StrategyParamsConfig
        
        strategy_params_config_dataclass = self.app_config.strategies_config.strategies.get(self.strategy_name)
        if strategy_params_config_dataclass:
            import dataclasses
            self.strategy_config_obj_dict = dataclasses.asdict(strategy_params_config_dataclass)
        
        if not self.strategy_config_obj_dict:
            msg = f"Configuration pour la stratégie '{self.strategy_name}' non trouvée dans AppConfig ou échec de conversion."
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)

        logger.info(f"{self.log_prefix} OOSValidator initialisé pour l'étude IS: '{self.study_is.study_name}'.")

    def _save_optuna_study_summary(self, fold_output_dir: Path) -> None:
        """Sauvegarde un résumé de l'étude Optuna IS et le DataFrame des essais."""
        fold_name = fold_output_dir.name # ex: "fold_0"
        logger.info(f"{self.log_prefix}[{fold_name}] Sauvegarde du résumé de l'étude Optuna IS...")
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        best_trial_for_summary: Optional[optuna.trial.FrozenTrial] = None
        if self.study_is.best_trials: # Pour multi-objectif, best_trials est une liste non-dominée
            best_trial_for_summary = self.study_is.best_trials[0] # Prendre le premier comme représentatif
            logger.debug(f"{self.log_prefix}[{fold_name}] Étude multi-objectifs, utilisation du premier de 'best_trials' pour le résumé.")
        elif self.study_is.best_trial: # Pour mono-objectif
             best_trial_for_summary = self.study_is.best_trial
             logger.debug(f"{self.log_prefix}[{fold_name}] Étude mono-objectif, utilisation de 'best_trial' pour le résumé.")
        else:
            logger.warning(f"{self.log_prefix}[{fold_name}] Aucun 'best_trial' ou 'best_trials' disponible dans l'étude IS. Résumé partiel.")


        study_summary_data = {
            "study_name": self.study_is.study_name,
            "directions": [str(d) for d in self.study_is.directions],
            "n_trials_completed_is": len([t for t in self.study_is.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "best_trial_is_number": best_trial_for_summary.number if best_trial_for_summary else None,
            "best_trial_is_values": best_trial_for_summary.values if best_trial_for_summary else None,
            "best_trial_is_params": best_trial_for_summary.params if best_trial_for_summary else None,
            "pareto_front_trials_is_count": len(self.study_is.best_trials) # Nombre d'essais sur le front de Pareto
        }
        summary_filepath = fold_output_dir / f"optuna_study_summary_{fold_name}.json"
        
        try:
            file_utils.save_json(summary_filepath, study_summary_data, indent=2, default_serializer=str)
            logger.info(f"{self.log_prefix}[{fold_name}] Résumé de l'étude Optuna IS sauvegardé : {summary_filepath}")
        except Exception as e_save_sum:
            logger.error(f"{self.log_prefix}[{fold_name}] Échec de la sauvegarde du résumé de l'étude Optuna IS : {e_save_sum}", exc_info=True)

        try:
            trials_df = self.study_is.trials_dataframe()
            trials_filepath = fold_output_dir / f"optuna_trials_dataframe_{fold_name}.csv"
            trials_df.to_csv(trials_filepath, index=False)
            logger.info(f"{self.log_prefix}[{fold_name}] DataFrame des essais Optuna IS sauvegardé : {trials_filepath}")
        except Exception as e_df: # Peut échouer si aucun essai, etc.
            logger.error(f"{self.log_prefix}[{fold_name}] Impossible de sauvegarder le DataFrame des essais Optuna IS : {e_df}", exc_info=True)

    def _get_best_is_trials_for_oos(self) -> List[Dict[str, Any]]:
        fold_name = self.study_is.study_name.split('_')[-2] if self.study_is.study_name and '_fold_' in self.study_is.study_name else "unknown_fold"
        logger.info(f"{self.log_prefix}[{fold_name}] Sélection des meilleurs essais IS pour validation OOS...")
        
        if not self.study_is.trials:
            logger.warning(f"{self.log_prefix}[{fold_name}] Aucun essai trouvé dans l'étude IS.")
            return []

        completed_valid_trials = [
            t for t in self.study_is.trials
            if t.state == optuna.trial.TrialState.COMPLETE and
               t.values is not None and
               all(isinstance(v, (int, float)) and np.isfinite(v) for v in t.values) # type: ignore
        ]

        if not completed_valid_trials:
            logger.warning(f"{self.log_prefix}[{fold_name}] Aucun essai IS complété avec des valeurs valides trouvé.")
            return []

        trials_to_consider_for_oos: List[optuna.trial.FrozenTrial]
        if len(self.study_is.directions) > 1:
            trials_to_consider_for_oos = self.study_is.best_trials
            logger.info(f"{self.log_prefix}[{fold_name}] Étude multi-objectifs. {len(trials_to_consider_for_oos)} essais sur le front de Pareto.")
        else:
            direction = self.study_is.directions[0]
            trials_to_consider_for_oos = sorted(
                completed_valid_trials,
                key=lambda t: t.values[0] if t.values else (-float('inf') if direction == optuna.study.StudyDirection.MAXIMIZE else float('inf')), # type: ignore
                reverse=(direction == optuna.study.StudyDirection.MAXIMIZE)
            )
            logger.info(f"{self.log_prefix}[{fold_name}] Étude mono-objectif (Direction: {direction}). {len(trials_to_consider_for_oos)} essais triés.")

        top_n_for_oos_validation = self.wfo_settings.top_n_trials_for_oos_validation
        selected_is_trials_info: List[Dict[str, Any]] = []
        for trial_obj in trials_to_consider_for_oos[:top_n_for_oos_validation]:
            selected_is_trials_info.append({
                "trial_number": trial_obj.number,
                "params": trial_obj.params,
                "is_values": trial_obj.values
            })
        
        logger.info(f"{self.log_prefix}[{fold_name}] {len(selected_is_trials_info)} meilleur(s) essai(s) IS sélectionné(s) pour OOS (cible: {top_n_for_oos_validation}).")
        return selected_is_trials_info

    def _run_oos_simulation_for_trial(self,
                                      is_trial_info: Dict[str, Any],
                                      # df_oos_fold_data est maintenant dans l'instance de l'évaluateur
                                      objective_evaluator_oos_instance: 'ObjectiveFunctionEvaluator'
                                     ) -> Optional[Dict[str, Any]]:
        is_trial_number = is_trial_info["trial_number"]
        is_trial_params = is_trial_info["params"]
        fold_name = self.study_is.study_name.split('_')[-2] if self.study_is.study_name and '_fold_' in self.study_is.study_name else "unknown_fold"
        oos_log_prefix = f"{self.log_prefix}[{fold_name}][OOS_for_IS_Trial:{is_trial_number}]"
        logger.info(f"{oos_log_prefix} Démarrage simulation OOS avec params : {is_trial_params}")

        oos_temp_study_name = f"oos_eval_{self.study_is.study_name}_is_trial_{is_trial_number}"
        oos_study_temp = optuna.create_study(
            study_name=oos_temp_study_name, storage=None, 
            directions=self.study_is.directions, # Utiliser les mêmes directions que l'étude IS
            sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.NopPruner()
        )

        try:
            oos_study_temp.enqueue_trial(
                is_trial_params,
                user_attrs={
                    "is_oos_validation_run": True,
                    "source_is_trial_number": is_trial_number,
                    "source_is_trial_values": is_trial_info.get("is_values")
                }
            )
            
            objective_evaluator_oos_instance.is_trial_number_for_oos_log = is_trial_number
            
            oos_study_temp.optimize(objective_evaluator_oos_instance, n_trials=1, catch=(Exception,))
            
            last_trial_run = oos_study_temp.trials[-1]
            
            if last_trial_run.state != optuna.trial.TrialState.COMPLETE:
                error_msg = f"Simulation OOS (trial temp {last_trial_run.number}) non complétée. État: {last_trial_run.state}."
                if last_trial_run.state == optuna.trial.TrialState.FAIL and last_trial_run.user_attrs.get('failure_reason'):
                    error_msg += f" Raison: {last_trial_run.user_attrs['failure_reason']}"
                logger.error(f"{oos_log_prefix} {error_msg}")
                return {"error": error_msg, "oos_metrics": None, "oos_detailed_log": None}

            if objective_evaluator_oos_instance.last_backtest_results and \
               objective_evaluator_oos_instance.last_backtest_results.get("metrics"):
                oos_metrics = objective_evaluator_oos_instance.last_backtest_results["metrics"]
                oos_detailed_log = objective_evaluator_oos_instance.last_backtest_results.get("oos_detailed_trades_log", [])
                logger.info(f"{oos_log_prefix} Simulation OOS complétée. PnL OOS: {oos_metrics.get('Total Net PnL USDC', 'N/A')}")
                return {"oos_metrics": oos_metrics, "oos_detailed_log": oos_detailed_log}
            else:
                logger.error(f"{oos_log_prefix} Aucune métrique de backtest trouvée après simulation OOS (trial temp {last_trial_run.number}).")
                return {"error": "Aucune métrique de backtest trouvée après OOS.", "oos_metrics": None, "oos_detailed_log": None}

        except optuna.exceptions.TrialPruned as e_pruned:
            logger.warning(f"{oos_log_prefix} Simulation OOS élaguée : {e_pruned}")
            return {"error": f"Élagué: {e_pruned}", "oos_metrics": None, "oos_detailed_log": None}
        except Exception as e_sim_oos:
            logger.error(f"{oos_log_prefix} Erreur durant la simulation OOS : {e_sim_oos}", exc_info=True)
            return {"error": str(e_sim_oos), "oos_metrics": None, "oos_detailed_log": None}

    def analyze_and_save_results_for_fold(self,
                                          fold_id_numeric: int,
                                          df_oos_fold_data: pd.DataFrame,
                                          fold_output_dir: Path
                                         ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        fold_log_prefix = f"{self.log_prefix}[Fold:{fold_id_numeric}]"
        logger.info(f"{fold_log_prefix} Démarrage analyse et sauvegarde des résultats.")
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        self._save_optuna_study_summary(fold_output_dir)
        best_is_trials_list = self._get_best_is_trials_for_oos()

        if not best_is_trials_list: # Aucun essai IS valide à tester en OOS
            logger.warning(f"{fold_log_prefix} Aucun meilleur essai IS trouvé pour validation OOS.")
            # Tenter un fallback sur le "best_trial" global de l'étude IS s'il existe (même si c'est une étude multi-obj, on prend le premier de best_trials)
            best_is_trial_for_fallback: Optional[optuna.trial.FrozenTrial] = None
            if self.study_is.best_trials: # Pour multi-objectif, c'est une liste
                best_is_trial_for_fallback = self.study_is.best_trials[0]
            elif self.study_is.best_trial: # Pour mono-objectif
                best_is_trial_for_fallback = self.study_is.best_trial
            
            if best_is_trial_for_fallback and best_is_trial_for_fallback.state == optuna.trial.TrialState.COMPLETE:
                logger.info(f"{fold_log_prefix} Fallback: Utilisation du meilleur essai IS (numéro {best_is_trial_for_fallback.number}) car aucun essai OOS ne peut être exécuté.")
                return best_is_trial_for_fallback.params, {"is_metrics": best_is_trial_for_fallback.values, "oos_status": "NO_VALID_IS_TRIALS_FOR_OOS_FALLBACK_TO_BEST_IS_ONLY"}
            else:
                logger.error(f"{fold_log_prefix} Aucun 'best_trial' IS non plus. Impossible de sélectionner des paramètres.")
                return None, None

        if df_oos_fold_data.empty:
            logger.warning(f"{fold_log_prefix} Données OOS vides pour le fold. Validation OOS sautée. Fallback sur meilleurs params IS.")
            best_is_trial_fallback_oos_empty = best_is_trials_list[0] # Prendre le premier de la liste des meilleurs IS
            return best_is_trial_fallback_oos_empty["params"], {"is_metrics": best_is_trial_fallback_oos_empty.get("is_values"), "oos_status": "SKIPPED_EMPTY_OOS_DATA_FALLBACK_TO_BEST_IS"}

        if not self.strategy_config_obj_dict or \
           not self.strategy_config_obj_dict.get('script_reference') or \
           not self.strategy_config_obj_dict.get('class_name'):
            logger.error(f"{fold_log_prefix} Configuration de stratégie (script/classe) manquante. Validation OOS impossible.")
            return best_is_trials_list[0]["params"], {"is_metrics": best_is_trials_list[0].get("is_values"), "oos_status": "CONFIG_ERROR_SKIPPED_OOS"}

        objective_evaluator_for_oos_runs = self.objective_evaluator_class(
            strategy_name=self.strategy_name,
            strategy_config_dict=cast(Dict[str,Any], self.strategy_config_obj_dict), # Assurer que c'est un dict
            df_enriched_slice=df_oos_fold_data.copy(),
            optuna_objectives_config={
                'objectives_names': self.app_config.global_config.optuna_settings.objectives_names,
                'objectives_directions': self.app_config.global_config.optuna_settings.objectives_directions
            },
            pair_symbol=self.pair_symbol, symbol_info_data=self.pair_config,
            app_config=self.app_config, run_id=self.run_id,
            is_oos_eval=True,
        )
        
        oos_validation_results_list: List[Dict[str, Any]] = []
        for is_trial_to_validate in best_is_trials_list:
            oos_run_res = self._run_oos_simulation_for_trial(
                is_trial_info=is_trial_to_validate,
                objective_evaluator_oos_instance=objective_evaluator_for_oos_runs
            )
            current_oos_summary = {
                "is_trial_number": is_trial_to_validate["trial_number"],
                "is_trial_params": is_trial_to_validate["params"],
                "is_trial_values_from_study": is_trial_to_validate.get("is_values"),
                "oos_metrics": oos_run_res.get("oos_metrics") if oos_run_res else None,
                "oos_detailed_trades_log": oos_run_res.get("oos_detailed_log") if oos_run_res else None,
                "oos_error": oos_run_res.get("error") if oos_run_res else "Simulation OOS a retourné None"
            }
            oos_validation_results_list.append(current_oos_summary)

        metric_to_sort_oos_by = self.wfo_settings.metric_to_optimize
        sort_direction_is_maximize = self.study_is.directions[0] == optuna.study.StudyDirection.MAXIMIZE

        def get_oos_metric_for_sort(oos_res: Dict[str, Any]) -> Any:
            val = oos_res.get("oos_metrics", {}).get(metric_to_sort_oos_by)
            if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
                return -float('inf') if sort_direction_is_maximize else float('inf')
            return val

        sorted_oos_results = sorted(
            [res for res in oos_validation_results_list if res.get("oos_metrics") is not None],
            key=get_oos_metric_for_sort,
            reverse=sort_direction_is_maximize
        )
        
        top_m_to_report = self.wfo_settings.top_n_trials_to_report_oos
        final_oos_results_to_save = sorted_oos_results[:top_m_to_report]

        final_selected_params: Optional[Dict[str, Any]] = None
        final_representative_metrics: Optional[Dict[str, Any]] = None

        if final_oos_results_to_save:
            oos_summary_filename = f"oos_validation_summary_TOP_{len(final_oos_results_to_save)}_TRIALS_{fold_output_dir.name}.json"
            oos_summary_filepath = fold_output_dir / oos_summary_filename
            try:
                file_utils.save_json(oos_summary_filepath, final_oos_results_to_save, indent=2, default_serializer=str)
                logger.info(f"{fold_log_prefix} Résumé validation OOS sauvegardé : {oos_summary_filepath}")
            except Exception as e_save_oos:
                 logger.error(f"{fold_log_prefix} Échec sauvegarde résumé OOS : {e_save_oos}", exc_info=True)

            for oos_run_summary in final_oos_results_to_save:
                is_trial_num = oos_run_summary.get("is_trial_number")
                detailed_log = oos_run_summary.get("oos_detailed_trades_log")
                if detailed_log is not None and is_trial_num is not None:
                    log_fname = f"oos_trades_log_is_trial_{is_trial_num}_{fold_output_dir.name}.json"
                    log_fpath = fold_output_dir / log_fname
                    try:
                        file_utils.save_json(log_fpath, detailed_log, indent=2, default_serializer=str)
                    except Exception as e_save_detail:
                        logger.error(f"{fold_log_prefix} Échec sauvegarde log OOS détaillé pour IS trial {is_trial_num}: {e_save_detail}", exc_info=True)
            
            best_oos_run_for_fold = final_oos_results_to_save[0]
            final_selected_params = best_oos_run_for_fold.get("is_trial_params")
            final_representative_metrics = best_oos_run_for_fold.get("oos_metrics")
            if final_representative_metrics:
                final_representative_metrics["selection_basis"] = "BEST_OOS_PERFORMANCE"
                final_representative_metrics["source_is_trial_number"] = best_oos_run_for_fold.get("is_trial_number")
            logger.info(f"{fold_log_prefix} Meilleurs params (basés sur OOS) : {final_selected_params}. Métriques OOS: {final_representative_metrics.get(metric_to_sort_oos_by) if final_representative_metrics else 'N/A'}")
        else:
            logger.warning(f"{fold_log_prefix} Validation OOS n'a produit aucun résultat rapportable. Fallback sur meilleurs params IS.")
            if best_is_trials_list:
                fallback_is_trial = best_is_trials_list[0]
                final_selected_params = fallback_is_trial["params"]
                final_representative_metrics = {"status_oos": "NO_VALID_OOS_RESULTS_FALLBACK_TO_BEST_IS", "selected_params_from_is": True, "is_metrics_for_selected_trial": fallback_is_trial.get("is_values"), "source_is_trial_number": fallback_is_trial.get("trial_number")}
            else:
                final_representative_metrics = {"status_oos": "FAILED_NO_IS_OR_OOS_RESULTS"}

        return final_selected_params, final_representative_metrics
