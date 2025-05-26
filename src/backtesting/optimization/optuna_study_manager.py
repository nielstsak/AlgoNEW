# src/backtesting/optimization/optuna_study_manager.py
"""
Ce module définit OptunaStudyManager, responsable de la création,
de la configuration (sampler, pruner, directions des objectifs), et de
l'exécution d'une étude Optuna pour l'optimisation In-Sample (IS)
des hyperparamètres d'une stratégie.
"""
import logging
import dataclasses # Pour dataclasses.replace
from pathlib import Path
from typing import Any, Dict, Optional, Type, TYPE_CHECKING, List, cast

import optuna # type: ignore
import pandas as pd

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import OptunaSettings, SamplerPrunerProfile
    # La classe ObjectiveFunctionEvaluator sera définie dans un autre module
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator


logger = logging.getLogger(__name__)

# --- Placeholder pour ObjectiveFunctionEvaluator ---
# Permet au module d'être importable même si le module dépendant n'est pas encore créé.
try:
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
except ImportError:
    logger.warning(
        "OptunaStudyManager: La classe ObjectiveFunctionEvaluator n'a pas pu être importée. "
        "Utilisation d'un placeholder. L'exécution réelle de l'étude échouera."
    )
    class ObjectiveFunctionEvaluator: # type: ignore
        """Placeholder pour ObjectiveFunctionEvaluator."""
        def __init__(self, strategy_name: str, strategy_config_dict: Dict[str, Any],
                     df_enriched_slice: pd.DataFrame, optuna_objectives_config: Dict[str, Any],
                     pair_symbol: str, symbol_info_data: Dict[str, Any],
                     app_config: 'AppConfig', run_id: str,
                     is_oos_eval: bool = False, is_trial_number_for_oos_log: Optional[int] = None):
            self.log_prefix = f"[{strategy_name}/{pair_symbol}][ObjectiveFunctionEvaluatorPlaceholder]"
            logger.info(f"{self.log_prefix} Initialisé (placeholder).")
        def __call__(self, trial: optuna.Trial) -> float: # Ou Tuple[float, ...] pour multi-objectif
            logger.error(f"{self.log_prefix} __call__ appelé sur un placeholder. Cela ne devrait pas arriver en production.")
            raise NotImplementedError("ObjectiveFunctionEvaluator placeholder ne peut pas être appelé.")
# --- Fin du Placeholder ---


class OptunaStudyManager:
    """
    Gère la création, la configuration et l'exécution d'une étude Optuna
    pour l'optimisation In-Sample (IS).
    """
    def __init__(self,
                 app_config: 'AppConfig',
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any], # Contenu de StrategyParamsConfig
                 study_output_dir: Path, # Répertoire spécifique au fold
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any], # Infos de l'exchange (pair_config)
                 run_id: str # ID du run WFO global de l'orchestrateur
                 ):
        """
        Initialise OptunaStudyManager.

        Args:
            app_config (AppConfig): Configuration globale de l'application.
            strategy_name (str): Nom de la stratégie.
            strategy_config_dict (Dict[str, Any]): Dictionnaire de configuration pour la stratégie.
            study_output_dir (Path): Répertoire de sortie pour ce fold (DB Optuna, etc.).
            pair_symbol (str): Symbole de la paire de trading.
            symbol_info_data (Dict[str, Any]): Informations de l'exchange pour la paire.
            run_id (str): ID du run WFO global de l'orchestrateur.
        """
        self.app_config = app_config
        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict
        self.study_output_dir = study_output_dir
        self.pair_symbol = pair_symbol
        self.symbol_info_data = symbol_info_data
        self.run_id = run_id # ID du run WFO global (celui du WFOManager pour cette tâche)

        self.log_prefix = f"[{self.strategy_name}/{self.pair_symbol}][Fold:{self.study_output_dir.name}][OptunaStudyManager]"

        # Récupérer OptunaSettings et créer une copie mutable pour appliquer les profils
        original_optuna_settings: 'OptunaSettings' = self.app_config.global_config.optuna_settings
        self.active_optuna_settings: 'OptunaSettings' = dataclasses.replace(original_optuna_settings)

        profile_to_activate = self.active_optuna_settings.default_profile_to_activate
        available_profiles = self.active_optuna_settings.sampler_pruner_profiles

        if profile_to_activate and available_profiles and profile_to_activate in available_profiles:
            profile_config = available_profiles[profile_to_activate]
            logger.info(f"{self.log_prefix} Activation du profil Optuna : '{profile_to_activate}'")
            
            # Mettre à jour les paramètres actifs avec ceux du profil
            # Utiliser getattr pour accéder aux champs de SamplerPrunerProfile (qui est une dataclass)
            self.active_optuna_settings.sampler_name = getattr(profile_config, 'sampler_name', self.active_optuna_settings.sampler_name)
            self.active_optuna_settings.sampler_params = getattr(profile_config, 'sampler_params', self.active_optuna_settings.sampler_params)
            self.active_optuna_settings.pruner_name = getattr(profile_config, 'pruner_name', self.active_optuna_settings.pruner_name)
            self.active_optuna_settings.pruner_params = getattr(profile_config, 'pruner_params', self.active_optuna_settings.pruner_params)
            
            logger.info(f"{self.log_prefix} Profil '{profile_to_activate}' appliqué. "
                        f"Sampler: {self.active_optuna_settings.sampler_name}, "
                        f"Pruner: {self.active_optuna_settings.pruner_name}")
        elif profile_to_activate:
            logger.warning(f"{self.log_prefix} Profil Optuna '{profile_to_activate}' non trouvé dans sampler_pruner_profiles. "
                           "Utilisation des paramètres Optuna par défaut (top-level).")
        else:
            logger.info(f"{self.log_prefix} Aucun default_profile_to_activate spécifié. "
                        "Utilisation des paramètres Optuna par défaut (top-level).")

        # Construction du chemin de la DB Optuna et de l'URL de stockage
        # Le nom du fichier DB inclut maintenant plus de détails pour l'unicité.
        # Le nom du répertoire du fold (study_output_dir.name) est déjà "fold_N".
        db_file_name = f"optuna_is_study_{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}.db"
        self.db_path = self.study_output_dir / db_file_name
        self.storage_url = f"sqlite:///{self.db_path.resolve()}"

        try:
            self.study_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"{self.log_prefix} Répertoire de sortie de l'étude (pour DB) assuré : {self.study_output_dir}")
            logger.info(f"{self.log_prefix} URL de stockage Optuna pour l'étude IS : {self.storage_url}")
        except OSError as e:
            logger.critical(f"{self.log_prefix} Échec de la création du répertoire de sortie de l'étude {self.study_output_dir}: {e}", exc_info=True)
            raise

        logger.info(f"{self.log_prefix} OptunaStudyManager initialisé.")


    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Crée une instance de sampler Optuna basée sur la configuration active."""
        sampler_name = self.active_optuna_settings.sampler_name.lower()
        # S'assurer que sampler_params est un dict, même s'il est None dans la config
        sampler_params = self.active_optuna_settings.sampler_params or {}
        
        logger.info(f"{self.log_prefix} Création du sampler : {sampler_name} avec params : {sampler_params}")
        
        if sampler_name == 'tpesampler':
            return optuna.samplers.TPESampler(**sampler_params)
        elif sampler_name == 'nsgaiisampler':
            # Nécessaire pour les études multi-objectifs
            if len(self.active_optuna_settings.objectives_directions) <= 1:
                logger.warning(f"{self.log_prefix} NSGAIISampler est généralement pour le multi-objectif, "
                               f"mais seulement {len(self.active_optuna_settings.objectives_directions)} objectif(s) configuré(s).")
            return optuna.samplers.NSGAIISampler(**sampler_params)
        elif sampler_name == 'cmaessampler':
            return optuna.samplers.CmaEsSampler(**sampler_params)
        elif sampler_name == 'randomsampler':
            return optuna.samplers.RandomSampler(**sampler_params)
        else:
            logger.warning(f"{self.log_prefix} Sampler inconnu '{sampler_name}'. Utilisation de TPESampler par défaut.")
            return optuna.samplers.TPESampler()

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Crée une instance de pruner Optuna basée sur la configuration active."""
        pruner_name = self.active_optuna_settings.pruner_name.lower()
        pruner_params = self.active_optuna_settings.pruner_params or {}

        logger.info(f"{self.log_prefix} Création du pruner : {pruner_name} avec params : {pruner_params}")

        if pruner_name == 'medianpruner':
            return optuna.pruners.MedianPruner(**pruner_params)
        elif pruner_name == 'hyperbandpruner':
            return optuna.pruners.HyperbandPruner(**pruner_params)
        elif pruner_name == 'noppruner':
            return optuna.pruners.NopPruner()
        elif pruner_name == 'patientpruner':
            # PatientPruner peut nécessiter un `trial` dans son constructeur pour certaines versions
            # ou si `wrapped_pruner` est utilisé. Ici, on suppose que les params sont directs.
            try:
                return optuna.pruners.PatientPruner(**pruner_params)
            except TypeError as e_patient:
                logger.warning(f"{self.log_prefix} Échec de la création de PatientPruner avec les params {pruner_params} "
                               f"(peut nécessiter un wrapped_pruner ou des params spécifiques à la version) : {e_patient}. "
                               "Retour à MedianPruner.")
                return optuna.pruners.MedianPruner()
        elif pruner_name == 'successivehalvingpruner':
            return optuna.pruners.SuccessiveHalvingPruner(**pruner_params)
        else:
            logger.warning(f"{self.log_prefix} Pruner inconnu '{pruner_name}'. Utilisation de MedianPruner par défaut.")
            return optuna.pruners.MedianPruner()

    def run_study(self,
                  data_1min_cleaned_is_slice: pd.DataFrame,
                  objective_evaluator_class: Type['ObjectiveFunctionEvaluator']
                  ) -> optuna.Study:
        """
        Crée (ou charge si existante) et exécute l'étude Optuna pour l'optimisation IS.

        Args:
            data_1min_cleaned_is_slice (pd.DataFrame): Données In-Sample pour l'optimisation.
            objective_evaluator_class (Type[ObjectiveFunctionEvaluator]): La classe de
                l'évaluateur d'objectif à utiliser.

        Returns:
            optuna.Study: L'objet étude Optuna après exécution (ou chargement).
        """
        # Nom de l'étude : unique par stratégie, paire, et nom du répertoire du fold.
        # study_output_dir.name est typiquement "fold_N".
        study_name = f"{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}_is_opt"
        logger.info(f"{self.log_prefix} Préparation pour exécuter/charger l'étude Optuna IS : '{study_name}'")

        # Validation des objectifs et directions
        objectives_names: List[str] = self.active_optuna_settings.objectives_names
        objectives_directions: List[str] = self.active_optuna_settings.objectives_directions

        if not objectives_names or len(objectives_names) != len(objectives_directions):
            logger.error(f"{self.log_prefix} Configuration des objectifs Optuna invalide. "
                         f"Noms: {objectives_names}, Directions: {objectives_directions}. "
                         "Doivent être des listes non vides de même longueur.")
            # Utiliser un fallback sûr ou lever une exception
            default_objective_name = "Total Net PnL USDC" # Un nom commun
            logger.warning(f"{self.log_prefix} Utilisation d'un objectif unique par défaut : '{default_objective_name}' (maximize).")
            objectives_names = [default_objective_name]
            objectives_directions = ["maximize"]
        
        for direction in objectives_directions:
            if direction.lower() not in ["maximize", "minimize"]:
                logger.error(f"{self.log_prefix} Direction d'objectif invalide '{direction}'. "
                             "Correction vers 'maximize'.")
                objectives_directions = ["maximize" if d.lower() not in ["maximize", "minimize"] else d.lower() for d in objectives_directions]
                break # Corriger une fois et sortir

        study: optuna.Study
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage_url,
                sampler=self._create_sampler(),
                pruner=self._create_pruner(),
                directions=objectives_directions, # Liste des directions
                load_if_exists=True
            )
            logger.info(f"{self.log_prefix} Étude Optuna IS '{study.study_name}' créée/chargée avec succès depuis {self.storage_url}.")
            logger.info(f"{self.log_prefix} Directions de l'étude : {study.directions}")
        except Exception as e_create_study:
            logger.critical(f"{self.log_prefix} Échec de la création ou du chargement de l'étude Optuna IS '{study_name}' à '{self.storage_url}': {e_create_study}", exc_info=True)
            raise # Renvoyer l'exception pour que l'orchestrateur de fold puisse la gérer

        # Configuration des objectifs pour l'évaluateur
        optuna_objectives_config_for_eval = {
            'objectives_names': objectives_names,
            'objectives_directions': objectives_directions
        }

        # Instanciation de l'évaluateur d'objectif
        objective_instance = objective_evaluator_class(
            strategy_name=self.strategy_name,
            strategy_config_dict=self.strategy_config_dict,
            df_enriched_slice=data_1min_cleaned_is_slice,
            optuna_objectives_config=optuna_objectives_config_for_eval,
            pair_symbol=self.pair_symbol,
            symbol_info_data=self.symbol_info_data, # C'est le pair_config
            app_config=self.app_config,
            run_id=self.run_id, # ID du WFOManager/Task
            is_oos_eval=False, # C'est une évaluation IS
            is_trial_number_for_oos_log=None # Non pertinent pour l'évaluation IS
        )

        # Exécution de l'optimisation
        n_trials_target = self.active_optuna_settings.n_trials
        n_jobs_parallel = self.active_optuna_settings.n_jobs
        if n_jobs_parallel == 0: # Optuna n'accepte pas 0
            logger.warning(f"{self.log_prefix} n_jobs configuré à 0, utilisation de 1 (séquentiel).")
            n_jobs_parallel = 1
        
        # Déterminer le nombre d'essais restants
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
                    gc_after_trial=True, # Nettoyage de la mémoire après chaque essai
                    callbacks=[] # Peut être étendu avec des callbacks Optuna personnalisés
                )
            except optuna.exceptions.TrialPruned as e_pruned_during_opt:
                # Ceci est normal si un pruner est actif et élague des essais.
                logger.info(f"{self.log_prefix} Un ou plusieurs essais ont été élagués pendant study.optimize: {e_pruned_during_opt}")
            except Exception as e_optimize:
                logger.error(f"{self.log_prefix} Erreur durant study.optimize pour '{study.study_name}': {e_optimize}", exc_info=True)
                # L'étude est quand même retournée avec les essais complétés jusqu'à présent.

        final_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"{self.log_prefix} Optimisation Optuna IS terminée pour '{study.study_name}'. "
                    f"Nombre total d'essais complétés dans l'étude : {final_completed_trials}.")
        
        return study

