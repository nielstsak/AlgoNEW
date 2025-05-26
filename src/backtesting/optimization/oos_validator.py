# src/backtesting.optimization.oos_validator.py
"""
Ce module définit la classe OOSValidator, responsable de la phase de validation
Out-of-Sample (OOS) pour un fold WFO donné. Il analyse l'étude Optuna
In-Sample (IS) terminée, sélectionne les meilleurs essais IS, exécute des
backtests OOS pour ces essais, et sauvegarde les résultats.
"""
import logging
import json
import importlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Type, TYPE_CHECKING, cast

import pandas as pd
import numpy as np
import optuna

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import WfoSettings, StrategyParamsConfig, SimulationDefaults
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
    from src.strategies.base import BaseStrategy

# Imports depuis l'application
try:
    from src.utils import file_utils # Pour save_json, si disponible
    # La classe ObjectiveFunctionEvaluator sera importée pour l'instanciation
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
except ImportError as e:
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
                 context_label: str, # Label de contexte (sanitized)
                 study_is: optuna.Study, # Étude Optuna de la phase IS pour le fold actuel
                 app_config: 'AppConfig',
                 pair_config: Dict[str, Any], # Anciennement symbol_info_data
                 # Ces maps contiendront typiquement une seule entrée pour l'instance OOSValidator
                 # car elle est appelée par fold_orchestrator pour un fold spécifique.
                 fold_data_map: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], # {fold_id: (df_is, df_oos)}
                 fold_dirs_map: Dict[int, Path] # {fold_id: Path_to_fold_output_dir}
                ):
        """
        Initialise OOSValidator.

        Args:
            run_id (str): ID du run WFO global.
            strategy_name (str): Nom de la stratégie.
            pair_symbol (str): Symbole de la paire.
            context_label (str): Label de contexte (sanitized).
            study_is (optuna.Study): Étude Optuna IS pour le fold actuel.
            app_config (AppConfig): Configuration globale de l'application.
            pair_config (Dict[str, Any]): Informations de l'exchange pour la paire.
            fold_data_map (Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]): Données IS/OOS pour les folds.
            fold_dirs_map (Dict[int, Path]): Répertoires de sortie pour les folds.
        """
        self.run_id = run_id
        self.strategy_name = strategy_name
        self.pair_symbol = pair_symbol
        self.context_label = context_label
        self.study_is = study_is # L'étude IS déjà exécutée pour ce fold
        self.app_config = app_config
        self.pair_config = pair_config # C'est le symbol_info_data

        self.fold_data_map = fold_data_map
        self.fold_dirs_map = fold_dirs_map

        self.log_prefix = f"[{self.strategy_name}/{self.pair_symbol}/{self.context_label}][Run:{self.run_id}][OOSValidator]"

        # Récupération des configurations nécessaires
        self.wfo_settings: 'WfoSettings' = self.app_config.global_config.wfo_settings
        self.strategy_config_obj: Optional['StrategyParamsConfig'] = self.app_config.strategies_config.strategies.get(self.strategy_name)
        self.sim_defaults: 'SimulationDefaults' = self.app_config.global_config.simulation_defaults

        if not self.strategy_config_obj:
            msg = f"Configuration pour la stratégie '{self.strategy_name}' non trouvée dans AppConfig."
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)

        logger.info(f"{self.log_prefix} OOSValidator initialisé.")

    def _save_optuna_study_summary(self, fold_id: int, fold_dir: Path) -> None:
        """
        Sauvegarde un résumé de l'étude Optuna IS et le DataFrame des essais.
        """
        logger.info(f"{self.log_prefix}[Fold:{fold_id}] Sauvegarde du résumé de l'étude Optuna IS...")
        fold_dir.mkdir(parents=True, exist_ok=True) # S'assurer que le répertoire existe

        study_summary_data = {
            "study_name": self.study_is.study_name,
            "directions": [str(d) for d in self.study_is.directions],
            "n_trials_completed_is": len([t for t in self.study_is.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "best_trial_is_number": self.study_is.best_trial.number if self.study_is.best_trial else None,
            "best_trial_is_values": self.study_is.best_trial.values if self.study_is.best_trial else None,
            "best_trial_is_params": self.study_is.best_trial.params if self.study_is.best_trial else None,
            "pareto_front_trials_is_count": len(self.study_is.best_trials)
        }
        summary_filepath = fold_dir / f"optuna_study_summary_fold_{fold_id}.json"
        
        try:
            if hasattr(file_utils, 'save_json') and callable(getattr(file_utils, 'save_json')):
                file_utils.save_json(summary_filepath, study_summary_data) # type: ignore
            else: # Fallback si file_utils ou save_json n'est pas disponible
                with open(summary_filepath, 'w', encoding='utf-8') as f:
                    json.dump(study_summary_data, f, indent=2, default=str)
            logger.info(f"{self.log_prefix}[Fold:{fold_id}] Résumé de l'étude Optuna IS sauvegardé : {summary_filepath}")
        except Exception as e_save_sum:
            logger.error(f"{self.log_prefix}[Fold:{fold_id}] Échec de la sauvegarde du résumé de l'étude Optuna IS : {e_save_sum}", exc_info=True)

        try:
            trials_df = self.study_is.trials_dataframe()
            trials_filepath = fold_dir / f"optuna_trials_dataframe_fold_{fold_id}.csv"
            trials_df.to_csv(trials_filepath, index=False)
            logger.info(f"{self.log_prefix}[Fold:{fold_id}] DataFrame des essais Optuna IS sauvegardé : {trials_filepath}")
        except Exception as e_df:
            logger.error(f"{self.log_prefix}[Fold:{fold_id}] Impossible de sauvegarder le DataFrame des essais Optuna IS : {e_df}", exc_info=True)


    def _get_best_is_trials_for_oos(self, fold_id: int) -> List[Dict[str, Any]]:
        """
        Sélectionne les N meilleurs essais IS pour la validation OOS.
        """
        logger.info(f"{self.log_prefix}[Fold:{fold_id}] Sélection des meilleurs essais IS pour validation OOS...")
        
        if not self.study_is.trials:
            logger.warning(f"{self.log_prefix}[Fold:{fold_id}] Aucun essai trouvé dans l'étude IS. Impossible de sélectionner pour OOS.")
            return []

        # Filtrer les essais complétés avec des valeurs valides
        completed_valid_trials = [
            t for t in self.study_is.trials
            if t.state == optuna.trial.TrialState.COMPLETE and
               t.values is not None and
               all(pd.notna(v) and np.isfinite(v) for v in t.values) # type: ignore
        ]

        if not completed_valid_trials:
            logger.warning(f"{self.log_prefix}[Fold:{fold_id}] Aucun essai IS complété avec des valeurs valides trouvé.")
            return []

        # Sélectionner les essais
        # Si multi-objectifs, study.best_trials retourne le front de Pareto.
        # Si mono-objectif, study.best_trials retourne une liste avec le meilleur essai.
        
        # Optuna trie déjà `best_trials` (le front de Pareto) d'une manière non dominée.
        # Pour sélectionner un sous-ensemble, on peut prendre les premiers de cette liste.
        # Si on a une stratégie de sélection Pareto spécifique (ex: SCORE_COMPOSITE),
        # elle aurait dû être appliquée par OptunaStudyManager ou ici si on veut retrier.
        # Pour l'instant, on prend les N premiers du front de Pareto ou les N meilleurs si mono-objectif.
        
        trials_to_consider_for_oos: List[optuna.trial.FrozenTrial]
        if self.study_is.directions and len(self.study_is.directions) > 1: # Multi-objectifs
            trials_to_consider_for_oos = self.study_is.best_trials # C'est le front de Pareto
            logger.info(f"{self.log_prefix}[Fold:{fold_id}] Étude multi-objectifs. "
                        f"{len(trials_to_consider_for_oos)} essais sur le front de Pareto.")
        else: # Mono-objectif
            # Trier les essais complétés par leur valeur (selon la direction)
            direction = self.study_is.directions[0] if self.study_is.directions else optuna.study.StudyDirection.MAXIMIZE
            trials_to_consider_for_oos = sorted(
                completed_valid_trials,
                key=lambda t: t.values[0] if t.values else (-float('inf') if direction == optuna.study.StudyDirection.MAXIMIZE else float('inf')), # type: ignore
                reverse=(direction == optuna.study.StudyDirection.MAXIMIZE)
            )
            logger.info(f"{self.log_prefix}[Fold:{fold_id}] Étude mono-objectif (Direction: {direction}). "
                        f"{len(trials_to_consider_for_oos)} essais complétés valides triés.")

        top_n_for_oos_validation = self.wfo_settings.top_n_trials_for_oos_validation
        selected_is_trials_info: List[Dict[str, Any]] = []

        for trial_obj in trials_to_consider_for_oos[:top_n_for_oos_validation]:
            selected_is_trials_info.append({
                "trial_number": trial_obj.number,
                "params": trial_obj.params,
                "is_values": trial_obj.values # Tuple des valeurs d'objectifs IS
            })
        
        logger.info(f"{self.log_prefix}[Fold:{fold_id}] {len(selected_is_trials_info)} meilleur(s) essai(s) IS sélectionné(s) pour la validation OOS "
                    f"(cible: {top_n_for_oos_validation}).")
        return selected_is_trials_info

    def _run_oos_simulation_for_trial(self,
                                      fold_id: int,
                                      is_trial_info: Dict[str, Any],
                                      df_oos_fold_data: pd.DataFrame,
                                      objective_evaluator_oos_instance: 'ObjectiveFunctionEvaluator'
                                     ) -> Optional[Dict[str, Any]]:
        """
        Exécute une simulation OOS pour un ensemble de paramètres IS donné.
        Utilise une instance de ObjectiveFunctionEvaluator en mode OOS.
        """
        is_trial_number = is_trial_info["params"]
        is_trial_params = is_trial_info["params"]
        oos_log_prefix = f"{self.log_prefix}[Fold:{fold_id}][OOS_for_IS_Trial:{is_trial_number}]"
        logger.info(f"{oos_log_prefix} Démarrage de la simulation OOS avec params : {is_trial_params}")

        # Pour évaluer un ensemble de paramètres fixes en OOS avec ObjectiveFunctionEvaluator,
        # nous devons simuler un "trial" Optuna avec ces paramètres.
        # Optuna permet d'ajouter un trial avec des paramètres spécifiques à une étude existante
        # puis de l'optimiser pour un seul essai.
        
        # Créer un "faux" trial Optuna ou utiliser enqueue_trial
        # study.enqueue_trial(params) ajoute le trial à la file d'attente.
        # study.optimize(objective, n_trials=1) exécutera alors ce trial enqueued.
        
        # L'ObjectiveFunctionEvaluator est déjà configuré pour OOS (is_oos_eval=True).
        # Son __call__ utilisera trial.params (qui seront les is_trial_params enqueued).
        
        try:
            # Enqueue le trial avec les paramètres IS fixes.
            # Les user_attrs peuvent être utiles pour identifier l'origine de ce trial OOS.
            self.study_is.enqueue_trial(
                is_trial_params,
                user_attrs={
                    "is_oos_validation_run": True,
                    "source_is_trial_number": is_trial_number,
                    "source_is_trial_values": is_trial_info.get("is_values")
                }
            )
            
            # Exécuter ce seul trial enqueued.
            # L'instance d'ObjectiveFunctionEvaluator doit être celle configurée pour OOS.
            self.study_is.optimize(objective_evaluator_oos_instance, n_trials=1, catch=(Exception,)) # catch Exception pour ne pas arrêter tout
            
            # Récupérer le dernier trial exécuté (celui qu'on vient d'enqueuer et d'exécuter)
            last_trial_run = self.study_is.trials[-1]
            
            if last_trial_run.state != optuna.trial.TrialState.COMPLETE:
                error_msg = f"La simulation OOS (trial {last_trial_run.number}) n'a pas complété. État: {last_trial_run.state}."
                if last_trial_run.state == optuna.trial.TrialState.FAIL and last_trial_run.user_attrs.get('failure_reason'):
                    error_msg += f" Raison: {last_trial_run.user_attrs['failure_reason']}"
                logger.error(f"{oos_log_prefix} {error_msg}")
                return {"error": error_msg, "oos_metrics": None, "oos_detailed_log": None}

            # Les métriques OOS sont maintenant dans les user_attrs du last_trial_run,
            # ou dans objective_evaluator_oos_instance.last_backtest_results
            if objective_evaluator_oos_instance.last_backtest_results and \
               objective_evaluator_oos_instance.last_backtest_results.get("metrics"):
                
                oos_metrics = objective_evaluator_oos_instance.last_backtest_results["metrics"]
                oos_detailed_log = objective_evaluator_oos_instance.last_backtest_results.get("oos_detailed_trades_log", [])
                logger.info(f"{oos_log_prefix} Simulation OOS complétée. "
                            f"Métriques OOS (ex: PnL): {oos_metrics.get('Total Net PnL USDC', 'N/A')}")
                return {
                    "oos_metrics": oos_metrics,
                    "oos_detailed_log": oos_detailed_log
                }
            else:
                logger.error(f"{oos_log_prefix} Aucune métrique de backtest trouvée après la simulation OOS (trial {last_trial_run.number}).")
                return {"error": "Aucune métrique de backtest trouvée après OOS.", "oos_metrics": None, "oos_detailed_log": None}

        except optuna.exceptions.TrialPruned as e_pruned:
            logger.warning(f"{oos_log_prefix} Simulation OOS élaguée : {e_pruned}")
            return {"error": f"Élagué: {e_pruned}", "oos_metrics": None, "oos_detailed_log": None}
        except Exception as e_sim_oos:
            logger.error(f"{oos_log_prefix} Erreur durant la simulation OOS : {e_sim_oos}", exc_info=True)
            return {"error": str(e_sim_oos), "oos_metrics": None, "oos_detailed_log": None}


    def _run_oos_validation_for_selected_is_trials(self,
                                                 fold_id: int,
                                                 best_is_trials_for_oos: List[Dict[str, Any]]
                                                ) -> List[Dict[str, Any]]:
        """
        Exécute la validation OOS pour une liste d'essais IS sélectionnés.
        """
        log_prefix_oos_run = f"{self.log_prefix}[Fold:{fold_id}][RunOOSValidation]"
        logger.info(f"{log_prefix_oos_run} Démarrage de la validation OOS pour {len(best_is_trials_for_oos)} essai(s) IS.")

        if fold_id not in self.fold_data_map:
            logger.error(f"{log_prefix_oos_run} Données pour le fold {fold_id} non trouvées dans fold_data_map.")
            return []
            
        _, df_oos_fold_data = self.fold_data_map[fold_id]
        if df_oos_fold_data.empty:
            logger.warning(f"{log_prefix_oos_run} Données OOS pour le fold {fold_id} vides. Validation OOS sautée.")
            return []

        if not self.strategy_config_obj or not self.strategy_config_obj.script_reference or not self.strategy_config_obj.class_name:
            logger.error(f"{log_prefix_oos_run} Configuration de stratégie (script_reference/class_name) manquante. Impossible de valider OOS.")
            return []

        oos_validation_results: List[Dict[str, Any]] = []

        # Créer une seule instance d'ObjectiveFunctionEvaluator pour toutes les simulations OOS de ce fold.
        # Elle sera configurée pour le mode OOS.
        objective_evaluator_oos = ObjectiveFunctionEvaluator(
            strategy_name=self.strategy_name,
            strategy_config_dict=self.app_config.strategies_config.strategies[self.strategy_name].__dict__, # Passer le dict de config
            df_enriched_slice=df_oos_fold_data.copy(), # Lui donner les données OOS
            optuna_objectives_config={ # Les objectifs ne sont pas optimisés en OOS, mais l'évaluateur en a besoin
                'objectives_names': self.app_config.global_config.optuna_settings.objectives_names,
                'objectives_directions': self.app_config.global_config.optuna_settings.objectives_directions
            },
            pair_symbol=self.pair_symbol,
            symbol_info_data=self.pair_config,
            app_config=self.app_config,
            run_id=self.run_id,
            is_oos_eval=True, # Mode OOS
            # is_trial_number_for_oos_log sera mis à jour pour chaque trial IS
        )

        for is_trial_info in best_is_trials_for_oos:
            is_trial_num = is_trial_info["trial_number"]
            objective_evaluator_oos.is_trial_number_for_oos_log = is_trial_num # Mettre à jour pour le logging

            oos_run_result = self._run_oos_simulation_for_trial(
                fold_id=fold_id,
                is_trial_info=is_trial_info,
                df_oos_fold_data=df_oos_fold_data, # Non utilisé directement si l'évaluateur a déjà les données
                objective_evaluator_oos_instance=objective_evaluator_oos
            )
            
            current_oos_summary = {
                "is_trial_number": is_trial_num,
                "is_trial_params": is_trial_info["params"],
                "is_trial_values_from_study": is_trial_info.get("is_values"), # Objectifs IS
                "oos_metrics": None,
                "oos_detailed_trades_log": None,
                "oos_error": None
            }
            if oos_run_result:
                current_oos_summary["oos_metrics"] = oos_run_result.get("oos_metrics")
                current_oos_summary["oos_detailed_trades_log"] = oos_run_result.get("oos_detailed_log")
                current_oos_summary["oos_error"] = oos_run_result.get("error")
            
            oos_validation_results.append(current_oos_summary)

        # Trier les résultats OOS et retourner les M meilleurs
        metric_to_sort_oos_by = self.wfo_settings.metric_to_optimize
        # La direction d'optimisation est celle définie pour l'étude IS (généralement la même pour OOS)
        sort_direction_is_maximize = self.study_is.directions[0] == optuna.study.StudyDirection.MAXIMIZE \
            if self.study_is.directions else True # Fallback à maximiser

        def get_oos_metric_value_for_sort(oos_res_dict: Dict[str, Any]) -> Any:
            val = oos_res_dict.get("oos_metrics", {}).get(metric_to_sort_oos_by)
            if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
                return -float('inf') if sort_direction_is_maximize else float('inf')
            return val

        sorted_oos_results = sorted(
            oos_validation_results,
            key=get_oos_metric_value_for_sort,
            reverse=sort_direction_is_maximize
        )
        
        top_m_to_report = self.wfo_settings.top_n_trials_to_report_oos
        logger.info(f"{log_prefix_oos_run} Validation OOS terminée. {len(sorted_oos_results)} résultats obtenus. "
                    f"Retour des {min(top_m_to_report, len(sorted_oos_results))} meilleurs.")
        return sorted_oos_results[:top_m_to_report]


    def analyze_and_save_results_for_fold(self,
                                          fold_id: int
                                         ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Analyse les résultats de l'étude IS, effectue la validation OOS,
        sauvegarde les artefacts, et retourne les meilleurs paramètres pour le fold.

        Args:
            fold_id (int): L'identifiant numérique du fold actuel.

        Returns:
            Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
                - Meilleurs paramètres (issus de IS, validés ou non par OOS).
                - Métriques OOS (ou IS si fallback) correspondantes.
                Retourne (None, None) si échec complet.
        """
        fold_log_prefix = f"{self.log_prefix}[Fold:{fold_id}]"
        logger.info(f"{fold_log_prefix} Démarrage de l'analyse et sauvegarde des résultats pour le fold.")

        fold_output_dir = self.fold_dirs_map.get(fold_id)
        if not fold_output_dir:
            logger.error(f"{fold_log_prefix} Répertoire de sortie pour le fold {fold_id} non trouvé. Impossible de sauvegarder.")
            return None, None
        fold_output_dir.mkdir(parents=True, exist_ok=True) # S'assurer qu'il existe

        # 1. Sauvegarder le résumé de l'étude Optuna IS
        self._save_optuna_study_summary(fold_id, fold_output_dir)

        # 2. Obtenir les meilleurs essais IS à valider en OOS
        best_is_trials_list = self._get_best_is_trials_for_oos(fold_id)

        if not best_is_trials_list:
            logger.warning(f"{fold_log_prefix} Aucun meilleur essai IS trouvé pour la validation OOS. "
                           "Si l'étude IS a eu des essais, cela peut indiquer des problèmes avec les valeurs d'objectifs.")
            # Tenter un fallback sur le "best_trial" global de l'étude IS si disponible
            if self.study_is.best_trial and self.study_is.best_trial.state == optuna.trial.TrialState.COMPLETE:
                logger.info(f"{fold_log_prefix} Utilisation du 'best_trial' global de l'étude IS comme fallback.")
                return self.study_is.best_trial.params, {"is_metrics": self.study_is.best_trial.values, "oos_status": "NO_VALID_IS_TRIALS_FOR_OOS_FALLBACK_TO_BEST_IS_ONLY"} # type: ignore
            else:
                logger.error(f"{fold_log_prefix} Aucun 'best_trial' global IS non plus. Impossible de sélectionner des paramètres.")
                return None, None

        # 3. Exécuter la validation OOS
        if not self.strategy_config_obj or not self.strategy_config_obj.script_reference or not self.strategy_config_obj.class_name:
            logger.error(f"{fold_log_prefix} script_reference ou class_name manquant dans StrategyParamsConfig pour {self.strategy_name}. Validation OOS impossible.")
            # Retourner les meilleurs params IS sans validation OOS
            return best_is_trials_list[0]["params"], {"is_metrics": best_is_trials_list[0]["is_values"], "oos_status": "CONFIG_ERROR_SKIPPED_OOS"}

        
        top_m_oos_results = self._run_oos_validation_for_selected_is_trials(
            fold_id=fold_id,
            best_is_trials_for_oos=best_is_trials_list
            # strategy_script_reference et class_name sont accessibles via self.strategy_config_obj
        )

        # 4. Sauvegarder les résultats OOS et déterminer les meilleurs paramètres pour le fold
        final_selected_params_for_this_fold: Optional[Dict[str, Any]] = None
        final_representative_metrics_for_this_fold: Optional[Dict[str, Any]] = None

        if top_m_oos_results:
            # Sauvegarder le résumé des M meilleurs essais OOS
            oos_summary_filename = f"oos_validation_summary_TOP_{len(top_m_oos_results)}_TRIALS_fold_{fold_id}.json"
            oos_summary_filepath = fold_output_dir / oos_summary_filename
            try:
                if hasattr(file_utils, 'save_json') and callable(getattr(file_utils, 'save_json')):
                    file_utils.save_json(oos_summary_filepath, top_m_oos_results) # type: ignore
                else:
                    with open(oos_summary_filepath, 'w', encoding='utf-8') as f:
                        json.dump(top_m_oos_results, f, indent=2, default=str)
                logger.info(f"{fold_log_prefix} Résumé de la validation OOS sauvegardé : {oos_summary_filepath}")
            except Exception as e_save_oos_sum:
                 logger.error(f"{fold_log_prefix} Échec de la sauvegarde du résumé OOS : {e_save_oos_sum}", exc_info=True)

            # Sauvegarder les logs de trades détaillés pour ces M essais OOS
            for oos_run_summary in top_m_oos_results:
                is_trial_num_oos = oos_run_summary.get("is_trial_number")
                detailed_log_data_oos = oos_run_summary.get("oos_detailed_trades_log")
                if detailed_log_data_oos is not None and is_trial_num_oos is not None:
                    log_filename = f"oos_trades_log_is_trial_{is_trial_num_oos}_fold_{fold_id}.json"
                    log_filepath = fold_output_dir / log_filename
                    try:
                        if hasattr(file_utils, 'save_json') and callable(getattr(file_utils, 'save_json')):
                             file_utils.save_json(log_filepath, detailed_log_data_oos) # type: ignore
                        else:
                            with open(log_filepath, 'w', encoding='utf-8') as f:
                                json.dump(detailed_log_data_oos, f, indent=2, default=str)
                        logger.debug(f"{fold_log_prefix} Log de trades OOS détaillé sauvegardé pour l'essai IS {is_trial_num_oos} : {log_filepath}")
                    except Exception as e_save_oos_detail:
                        logger.error(f"{fold_log_prefix} Échec de la sauvegarde du log de trades OOS détaillé pour l'essai IS {is_trial_num_oos} : {e_save_oos_detail}", exc_info=True)
            
            # Le "meilleur" est le premier de la liste triée top_m_oos_results
            best_oos_run_for_fold = top_m_oos_results[0]
            final_selected_params_for_this_fold = best_oos_run_for_fold.get("is_trial_params")
            final_representative_metrics_for_this_fold = best_oos_run_for_fold.get("oos_metrics")
            if final_representative_metrics_for_this_fold: # Ajouter une note sur l'origine
                final_representative_metrics_for_this_fold["selection_basis"] = "BEST_OOS_PERFORMANCE"
                final_representative_metrics_for_this_fold["source_is_trial_number"] = best_oos_run_for_fold.get("is_trial_number")
            
            logger.info(f"{fold_log_prefix} Meilleurs paramètres pour le fold (basés sur OOS) : {final_selected_params_for_this_fold}")
            logger.info(f"{fold_log_prefix} Métriques OOS représentatives : {final_representative_metrics_for_this_fold.get('Total Net PnL USDC') if final_representative_metrics_for_this_fold else 'N/A'}")

        else: # Aucun résultat OOS satisfaisant
            logger.warning(f"{fold_log_prefix} La validation OOS n'a produit aucun résultat rapportable. "
                           "Fallback sur les meilleurs paramètres IS (si disponibles).")
            if best_is_trials_list: # S'il y avait des essais IS à valider
                # Prendre le premier de la liste des "meilleurs IS" (qui était déjà triée ou issue du front de Pareto)
                fallback_is_trial = best_is_trials_list[0]
                final_selected_params_for_this_fold = fallback_is_trial["params"]
                final_representative_metrics_for_this_fold = {
                    "status_oos": "NO_VALID_OOS_RESULTS_FALLBACK_TO_BEST_IS",
                    "selected_params_from_is": True,
                    "is_metrics_for_selected_trial": fallback_is_trial.get("is_values"),
                    "source_is_trial_number": fallback_is_trial.get("trial_number")
                    # On pourrait essayer de récupérer les user_attrs du trial IS si besoin
                }
                logger.info(f"{fold_log_prefix} Fallback: Meilleurs params IS : {final_selected_params_for_this_fold}")
            else: # Même pas d'essais IS valides au départ
                logger.error(f"{fold_log_prefix} Aucun essai IS valide n'a été fourni à OOS et OOS a échoué. Impossible de sélectionner des paramètres.")
                final_selected_params_for_this_fold = None
                final_representative_metrics_for_this_fold = {"status_oos": "FAILED_NO_IS_OR_OOS_RESULTS"}

        return final_selected_params_for_this_fold, final_representative_metrics_for_this_fold

