# src/backtesting/optimization/optuna_study_manager.py
"""
Ce module définit OptunaStudyManager, responsable de la création,
de la configuration (sampler, pruner, directions des objectifs), et de
l'exécution d'une étude Optuna pour l'optimisation In-Sample (IS)
des hyperparamètres d'une stratégie.
"""
import logging
import dataclasses # Pour dataclasses.replace si on copie OptunaSettings
from pathlib import Path
from typing import Any, Dict, Optional, Type, TYPE_CHECKING, List, cast

import optuna # type: ignore
import pandas as pd

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import OptunaSettings, SamplerPrunerProfile, StrategyParamsConfig
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator

# Imports depuis l'application
try:
    from src.config.definitions import OptunaSettings, SamplerPrunerProfile # Pour le typage
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
except ImportError as e:
    # Ce log est un fallback, le logging principal est configuré ailleurs
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"OptunaStudyManager: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH."
    )
    raise

logger = logging.getLogger(__name__)

class OptunaStudyManager:
    """
    Gère la création, la configuration et l'exécution d'une étude Optuna
    pour l'optimisation In-Sample (IS).
    """
    def __init__(self,
                 app_config: 'AppConfig',
                 strategy_name: str,
                 # strategy_config_dict est le contenu de StrategyParamsConfig pour la stratégie
                 strategy_config_dict: Dict[str, Any],
                 study_output_dir: Path, # Répertoire spécifique au fold (ex: .../TASK_ID/fold_0/)
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any], # Infos de l'exchange (pair_config)
                 run_id: str # ID du run WFO global de l'orchestrateur (celui du WFOManager pour cette tâche)
                 ):
        """
        Initialise OptunaStudyManager.

        Args:
            app_config (AppConfig): Configuration globale de l'application.
            strategy_name (str): Nom de la stratégie.
            strategy_config_dict (Dict[str, Any]): Dictionnaire de configuration pour la stratégie
                                                   (provenant de AppConfig.strategies_config.strategies[strategy_name]).
            study_output_dir (Path): Répertoire de sortie pour ce fold où la DB Optuna sera stockée.
            pair_symbol (str): Symbole de la paire de trading.
            symbol_info_data (Dict[str, Any]): Informations de l'exchange pour la paire (pair_config).
            run_id (str): ID du run WFO global de l'orchestrateur (ou de la tâche WFO parente).
        """
        self.app_config = app_config
        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict # C'est StrategyParamsConfig converti en dict
        self.study_output_dir = study_output_dir # Ex: .../TASK_ID/fold_0/
        self.pair_symbol = pair_symbol.upper()
        self.symbol_info_data = symbol_info_data
        self.run_id = run_id # ID du WFOManager/Task

        # Le préfixe de log inclut maintenant le nom du répertoire du fold pour plus de clarté
        self.log_prefix = f"[{self.strategy_name}/{self.pair_symbol}][Fold:{self.study_output_dir.name}][OptunaStudyMgr]"

        # Récupérer OptunaSettings et créer une copie mutable pour appliquer les profils
        original_optuna_settings: OptunaSettings = self.app_config.global_config.optuna_settings
        # Utiliser dataclasses.replace pour une copie propre si OptunaSettings est une dataclass
        if dataclasses.is_dataclass(original_optuna_settings):
            self.active_optuna_settings: OptunaSettings = dataclasses.replace(original_optuna_settings)
        else: # Fallback si ce n'est pas une dataclass (ne devrait pas arriver avec la structure actuelle)
            import copy
            self.active_optuna_settings = copy.deepcopy(original_optuna_settings) # type: ignore

        profile_to_activate = self.active_optuna_settings.default_profile_to_activate
        available_profiles = self.active_optuna_settings.sampler_pruner_profiles

        if profile_to_activate and available_profiles and profile_to_activate in available_profiles:
            profile_config = available_profiles[profile_to_activate]
            logger.info(f"{self.log_prefix} Activation du profil Optuna : '{profile_to_activate}' - {profile_config.description}")
            
            self.active_optuna_settings.sampler_name = profile_config.sampler_name
            self.active_optuna_settings.sampler_params = profile_config.sampler_params.copy() if profile_config.sampler_params else {}
            self.active_optuna_settings.pruner_name = profile_config.pruner_name
            self.active_optuna_settings.pruner_params = profile_config.pruner_params.copy() if profile_config.pruner_params else {}
            
            logger.info(f"{self.log_prefix} Profil '{profile_to_activate}' appliqué. "
                        f"Sampler: {self.active_optuna_settings.sampler_name} (Params: {self.active_optuna_settings.sampler_params}), "
                        f"Pruner: {self.active_optuna_settings.pruner_name} (Params: {self.active_optuna_settings.pruner_params})")
        elif profile_to_activate:
            logger.warning(f"{self.log_prefix} Profil Optuna '{profile_to_activate}' non trouvé dans sampler_pruner_profiles. "
                           "Utilisation des paramètres Optuna par défaut (top-level).")
        else:
            logger.info(f"{self.log_prefix} Aucun default_profile_to_activate spécifié. "
                        "Utilisation des paramètres Optuna par défaut (top-level).")

        # Construction du chemin de la DB Optuna et de l'URL de stockage
        # Le nom du fichier DB inclut stratégie, paire, et nom du fold pour unicité.
        db_file_name = f"optuna_is_study_{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}.db"
        self.db_path = self.study_output_dir / db_file_name
        self.storage_url = f"sqlite:///{self.db_path.resolve()}"

        try:
            self.study_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"{self.log_prefix} Répertoire de sortie de l'étude (pour DB) assuré : {self.study_output_dir}")
            logger.info(f"{self.log_prefix} URL de stockage Optuna pour l'étude IS : {self.storage_url}")
        except OSError as e_mkdir:
            logger.critical(f"{self.log_prefix} Échec de la création du répertoire de sortie de l'étude {self.study_output_dir}: {e_mkdir}", exc_info=True)
            raise

        logger.info(f"{self.log_prefix} OptunaStudyManager initialisé.")

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Crée une instance de sampler Optuna basée sur la configuration active."""
        sampler_name = self.active_optuna_settings.sampler_name.lower()
        sampler_params = self.active_optuna_settings.sampler_params or {} # Assurer un dict
        
        logger.info(f"{self.log_prefix} Création du sampler : '{sampler_name}' avec params : {sampler_params}")
        
        try:
            if sampler_name == 'tpesampler':
                return optuna.samplers.TPESampler(**sampler_params)
            elif sampler_name == 'nsgaiisampler':
                if len(self.active_optuna_settings.objectives_directions) <= 1 and self.log_prefix: # Vérifier si log_prefix est initialisé
                    logger.warning(f"{self.log_prefix} NSGAIISampler est pour multi-objectif, mais "
                                   f"{len(self.active_optuna_settings.objectives_directions)} objectif(s) configuré(s).")
                return optuna.samplers.NSGAIISampler(**sampler_params)
            elif sampler_name == 'cmaessampler':
                return optuna.samplers.CmaEsSampler(**sampler_params)
            elif sampler_name == 'randomsampler':
                return optuna.samplers.RandomSampler(**sampler_params)
            # Ajouter d'autres samplers ici si nécessaire
            else:
                logger.warning(f"{self.log_prefix} Sampler inconnu '{sampler_name}'. Utilisation de TPESampler par défaut.")
                return optuna.samplers.TPESampler()
        except Exception as e_sampler:
            logger.error(f"{self.log_prefix} Erreur lors de la création du sampler '{sampler_name}': {e_sampler}. Fallback sur TPESampler.", exc_info=True)
            return optuna.samplers.TPESampler()


    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Crée une instance de pruner Optuna basée sur la configuration active."""
        pruner_name = self.active_optuna_settings.pruner_name.lower()
        pruner_params = self.active_optuna_settings.pruner_params or {} # Assurer un dict

        logger.info(f"{self.log_prefix} Création du pruner : '{pruner_name}' avec params : {pruner_params}")

        try:
            if pruner_name == 'medianpruner':
                return optuna.pruners.MedianPruner(**pruner_params)
            elif pruner_name == 'hyperbandpruner':
                return optuna.pruners.HyperbandPruner(**pruner_params)
            elif pruner_name == 'noppruner':
                return optuna.pruners.NopPruner() # Pas de paramètres pour NopPruner
            elif pruner_name == 'patientpruner':
                return optuna.pruners.PatientPruner(**pruner_params)
            elif pruner_name == 'successivehalvingpruner':
                return optuna.pruners.SuccessiveHalvingPruner(**pruner_params)
            # Ajouter d'autres pruners ici si nécessaire
            else:
                logger.warning(f"{self.log_prefix} Pruner inconnu '{pruner_name}'. Utilisation de MedianPruner par défaut.")
                return optuna.pruners.MedianPruner()
        except Exception as e_pruner:
            logger.error(f"{self.log_prefix} Erreur lors de la création du pruner '{pruner_name}': {e_pruner}. Fallback sur MedianPruner.", exc_info=True)
            return optuna.pruners.MedianPruner()

    def run_study(self,
                  data_1min_cleaned_is_slice: pd.DataFrame, # Données IS pour ce fold
                  objective_evaluator_class: Type['ObjectiveFunctionEvaluator']
                  ) -> optuna.Study:
        """
        Crée (ou charge si existante) et exécute l'étude Optuna pour l'optimisation IS.

        Args:
            data_1min_cleaned_is_slice (pd.DataFrame): Données In-Sample pour l'optimisation.
                                                       L'index doit être un DatetimeIndex UTC.
            objective_evaluator_class (Type[ObjectiveFunctionEvaluator]): La classe de
                l'évaluateur d'objectif à utiliser (la vraie classe, pas un placeholder).

        Returns:
            optuna.Study: L'objet étude Optuna après exécution (ou chargement).
        """
        # Nom de l'étude : unique par stratégie, paire, et nom du répertoire du fold.
        # self.study_output_dir.name est typiquement "fold_N".
        study_name = f"{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}_is_opt"
        logger.info(f"{self.log_prefix} Préparation pour exécuter/charger l'étude Optuna IS : '{study_name}'")

        objectives_names: List[str] = self.active_optuna_settings.objectives_names
        objectives_directions: List[str] = self.active_optuna_settings.objectives_directions

        if not objectives_names or len(objectives_names) != len(objectives_directions):
            msg = (f"Configuration des objectifs Optuna invalide. Noms: {objectives_names}, "
                   f"Directions: {objectives_directions}. Doivent être des listes non vides de même longueur.")
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg) # Erreur bloquante
        
        valid_directions = ["maximize", "minimize"]
        for i, direction in enumerate(objectives_directions):
            if direction.lower() not in valid_directions:
                logger.error(f"{self.log_prefix} Direction d'objectif invalide '{direction}' pour objectif '{objectives_names[i]}'. "
                             f"Doit être 'maximize' ou 'minimize'. Correction vers 'maximize'.")
                objectives_directions[i] = "maximize" # Fallback sûr

        study: optuna.Study
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage_url,
                sampler=self._create_sampler(),
                pruner=self._create_pruner(),
                directions=objectives_directions,
                load_if_exists=True # Charger l'étude si elle existe déjà dans la DB
            )
            logger.info(f"{self.log_prefix} Étude Optuna IS '{study.study_name}' créée/chargée avec succès depuis {self.storage_url}.")
            logger.info(f"{self.log_prefix} Objectifs de l'étude: {objectives_names}, Directions: {study.directions}")
        except Exception as e_create_study:
            logger.critical(f"{self.log_prefix} Échec de la création ou du chargement de l'étude Optuna IS '{study_name}' à '{self.storage_url}': {e_create_study}", exc_info=True)
            raise

        optuna_objectives_config_for_eval = {
            'objectives_names': objectives_names,
            'objectives_directions': objectives_directions # Utiliser les directions potentiellement corrigées
        }

        objective_instance = objective_evaluator_class(
            strategy_name=self.strategy_name,
            strategy_config_dict=self.strategy_config_dict, # C'est StrategyParamsConfig as dict
            df_enriched_slice=data_1min_cleaned_is_slice,
            optuna_objectives_config=optuna_objectives_config_for_eval,
            pair_symbol=self.pair_symbol,
            symbol_info_data=self.symbol_info_data, # C'est le pair_config
            app_config=self.app_config,
            run_id=self.run_id, # ID du WFOManager/Task
            is_oos_eval=False, # C'est une évaluation IS
            is_trial_number_for_oos_log=None # Non pertinent pour l'évaluation IS
        )

        n_trials_target = self.active_optuna_settings.n_trials
        n_jobs_parallel = self.active_optuna_settings.n_jobs
        if n_jobs_parallel == 0:
            logger.warning(f"{self.log_prefix} n_jobs configuré à 0, utilisation de 1 (séquentiel).")
            n_jobs_parallel = 1
        
        completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials_to_run = n_trials_target - completed_trials_count

        if remaining_trials_to_run <= 0:
            logger.info(f"{self.log_prefix} L'étude '{study.study_name}' a déjà {completed_trials_count} essai(s) complété(s) "
                        f"(cible: {n_trials_target}). Aucun nouvel essai ne sera exécuté pour ce fold.")
        else:
            logger.info(f"{self.log_prefix} Démarrage de l'optimisation Optuna IS pour '{study.study_name}'.")
            logger.info(f"{self.log_prefix} Cible totale d'essais: {n_trials_target}. Déjà complétés: {completed_trials_count}. "
                        f"Restants à exécuter: {remaining_trials_to_run}. Utilisation de {n_jobs_parallel} job(s) parallèle(s).")
            try:
                study.optimize(
                    objective_instance, # L'instance de ObjectiveFunctionEvaluator
                    n_trials=remaining_trials_to_run,
                    n_jobs=n_jobs_parallel,
                    gc_after_trial=True,
                    callbacks=[], # Peut être étendu avec des callbacks Optuna personnalisés
                    catch=(Exception,) # Attraper les exceptions dans les trials pour ne pas arrêter toute l'étude
                )
            except optuna.exceptions.TrialPruned as e_pruned_opt:
                logger.info(f"{self.log_prefix} Un ou plusieurs essais ont été élagués pendant study.optimize: {e_pruned_opt}")
            except Exception as e_optimize: # Erreur plus générale durant optimize
                logger.error(f"{self.log_prefix} Erreur durant study.optimize pour '{study.study_name}': {e_optimize}", exc_info=True)
                # L'étude est quand même retournée avec les essais complétés jusqu'à présent.

        final_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        logger.info(f"{self.log_prefix} Optimisation Optuna IS terminée pour '{study.study_name}'. "
                    f"Essais complétés: {final_completed_trials}, Échoués: {failed_trials}, Élagués: {pruned_trials}.")
        
        return study
