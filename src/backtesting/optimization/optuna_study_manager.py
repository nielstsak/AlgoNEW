# src/backtesting/optimization/optuna_study_manager.py
"""
Ce module définit OptunaStudyManager, responsable de la création,
de la configuration (sampler, pruner, directions des objectifs, callbacks),
et de l'exécution d'une étude Optuna pour l'optimisation In-Sample (IS)
des hyperparamètres d'une stratégie.
Il intègre désormais le checkpointing, le monitoring et des fonctionnalités avancées.
"""
import logging
import dataclasses
import time
import threading # Pour le heartbeat dans un thread séparé si study.optimize est bloquant
from pathlib import Path
from typing import Any, Dict, Optional, Type, TYPE_CHECKING, List, cast, Callable, Union
from datetime import datetime, timedelta # Pour ETA dans analytics
from dataclasses import dataclass, field

import optuna
import pandas as pd
import numpy as np # Pour np.mean, np.std dans analytics

# Imports pour les intégrations (gestion des ImportError)
try:
    from optuna.integration import MLflowCallback, TensorBoardCallback
    OPTUNA_INTEGRATIONS_AVAILABLE = True
except ImportError:
    MLflowCallback = None # type: ignore
    TensorBoardCallback = None # type: ignore
    OPTUNA_INTEGRATIONS_AVAILABLE = False
    logging.getLogger(__name__).info(
        "MLflowCallback ou TensorBoardCallback non disponibles depuis optuna.integration. "
        "Ces fonctionnalités de callback seront désactivées."
    )
try:
    from optuna.integration import BoTorchSampler, SkoptSampler
    OPTUNA_BAYESIAN_SAMPLERS_AVAILABLE = True
except ImportError:
    BoTorchSampler = None # type: ignore
    SkoptSampler = None # type: ignore
    OPTUNA_BAYESIAN_SAMPLERS_AVAILABLE = False
    logging.getLogger(__name__).info(
        "BoTorchSampler ou SkoptSampler non disponibles. "
        "Les options de sampler bayésien avancé seront limitées."
    )


if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import OptunaSettings, SamplerPrunerProfile
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator

# Imports depuis l'application
try:
    from src.config.definitions import OptunaSettings, SamplerPrunerProfile
    from src.backtesting.optimization.objective_function_evaluator import ObjectiveFunctionEvaluator
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"OptunaStudyManager: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH."
    )
    raise

logger = logging.getLogger(__name__)

# --- Callbacks Personnalisés ---
class HeartbeatCallback:
    """Callback pour afficher un message de 'heartbeat' périodiquement."""
    def __init__(self, interval_seconds: int = 30):
        self.interval_seconds = interval_seconds
        self.last_heartbeat_time = time.monotonic()
        self.log_prefix = "[OptunaHeartbeat]"

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        current_time = time.monotonic()
        if current_time - self.last_heartbeat_time >= self.interval_seconds:
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            total_trials_in_study = len(study.trials)
            best_values_str = "N/A"
            if study.best_trials:
                best_values_str = ", ".join([f"{v:.4f}" for v in study.best_trials[0].values]) if study.best_trials[0].values else "N/A"
            
            logger.info(f"{self.log_prefix} Étude '{study.study_name}': {completed_trials}/{total_trials_in_study} essais (complétés/total). "
                        f"Meilleures valeurs actuelles (1er du front): [{best_values_str}]")
            self.last_heartbeat_time = current_time

class EarlyStoppingCallback:
    """Callback pour arrêter l'étude si aucune amélioration n'est observée."""
    def __init__(self, early_stopping_rounds: int, patience: int = 0, delta_threshold: float = 0.0001):
        self.early_stopping_rounds = early_stopping_rounds
        self.patience = patience # Nombre de rounds à attendre avant d'appliquer l'early stopping
        self.delta_threshold = delta_threshold # Différence minimale pour considérer une amélioration
        self._no_improvement_count = 0
        self._best_value_seen: Optional[float] = None
        self.log_prefix = "[OptunaEarlyStopping]"

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if not study.directions: # Ne peut pas fonctionner sans directions
            return
        
        # Pour multi-objectif, on se base sur le premier objectif par défaut, ou une logique plus complexe.
        # Optuna ne stoppe pas l'étude entière sur un seul objectif en multi-objectif via cette méthode simple.
        # Un vrai early stopping multi-objectif est plus complexe.
        # Ici, on se base sur le premier 'best_value' disponible (qui est pour le premier objectif si mono).
        current_value: Optional[float] = None
        if study.best_value is not None: # Mono-objectif
            current_value = study.best_value
        elif study.best_trials and study.best_trials[0].values: # Multi-objectif, prendre la valeur du 1er objectif du 1er trial du front
            current_value = study.best_trials[0].values[0]

        if current_value is None: # Pas encore de valeur de référence
            return

        if trial.number < self.patience: # Attendre la fin de la période de patience
            return

        improved = False
        if self._best_value_seen is None:
            self._best_value_seen = current_value
            improved = True # Premier enregistrement est une "amélioration"
        else:
            # Gérer la direction de l'optimisation
            is_maximize = study.directions[0] == optuna.study.StudyDirection.MAXIMIZE
            if is_maximize:
                if current_value > self._best_value_seen + self.delta_threshold:
                    self._best_value_seen = current_value
                    improved = True
            else: # Minimize
                if current_value < self._best_value_seen - self.delta_threshold:
                    self._best_value_seen = current_value
                    improved = True
        
        if improved:
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.early_stopping_rounds:
            logger.info(f"{self.log_prefix} Arrêt anticipé déclenché pour l'étude '{study.study_name}'. "
                        f"Aucune amélioration significative observée pendant {self.early_stopping_rounds} essais consécutifs "
                        f"(après {self.patience} essais de patience). Meilleure valeur vue: {self._best_value_seen:.4f}")
            study.stop()




@dataclass
class StudyAnalytics:
    """Dataclass pour stocker les analyses d'une étude Optuna."""
    study_name: str
    total_trials_in_study: int
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    start_time_iso: Optional[str] = None
    current_time_iso: Optional[str] = None
    elapsed_time_seconds: Optional[float] = None
    trials_per_second: Optional[float] = None
    estimated_time_remaining_seconds: Optional[float] = None
    target_total_trials: Optional[int] = None
    best_values: Optional[List[float]] = None # Pour multi-objectif, liste des valeurs du premier front
    best_params: Optional[Dict[str, Any]] = None # Params du premier trial du front de Pareto
    param_importances: Optional[Dict[str, float]] = None # Pour le premier objectif
    # On pourrait ajouter l'évolution des métriques ici si besoin


class OptunaStudyManager:
    """
    Gère la création, la configuration et l'exécution d'une étude Optuna
    pour l'optimisation In-Sample (IS), avec checkpointing et monitoring.
    """
    def __init__(self,
                 app_config: 'AppConfig',
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any],
                 study_output_dir: Path,
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any],
                 run_id: str
                 ):
        self.app_config = app_config
        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict
        self.study_output_dir = study_output_dir
        self.pair_symbol = pair_symbol.upper()
        self.symbol_info_data = symbol_info_data
        self.run_id = run_id

        self.log_prefix = f"[{self.strategy_name}/{self.pair_symbol}][Fold:{self.study_output_dir.name}][OptunaStudyMgrV2]"

        original_optuna_settings: OptunaSettings = self.app_config.global_config.optuna_settings
        self.active_optuna_settings: OptunaSettings = dataclasses.replace(original_optuna_settings)

        self._apply_optuna_profile() # Appliquer le profil configuré

        # Checkpointing et Storage
        self.checkpoint_dir: Optional[Path] = None # Sera défini par enable_checkpointing
        self.checkpoint_interval_trials: int = 10 # Valeur par défaut
        self.storage_url: Optional[str] = self.active_optuna_settings.storage # Peut être "sqlite:///...", "redis://...", ou None

        # Callbacks
        self.callbacks: List[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = []
        self.callbacks.append(HeartbeatCallback(interval_seconds=30)) # Heartbeat par défaut

        # Si un chemin de stockage local est implicitement défini par study_output_dir
        # et que self.storage_url est None, on configure un RDBStorage local.
        if self.storage_url is None:
            self.enable_checkpointing(self.study_output_dir) # Active le checkpointing local par défaut

        logger.info(f"{self.log_prefix} OptunaStudyManager V2 initialisé. Storage URL: {self.storage_url}")

    def _apply_optuna_profile(self) -> None:
        """Applique un profil de sampler/pruner si configuré."""
        profile_to_activate = self.active_optuna_settings.default_profile_to_activate
        available_profiles = self.active_optuna_settings.sampler_pruner_profiles

        if profile_to_activate and available_profiles and profile_to_activate in available_profiles:
            profile_config = available_profiles[profile_to_activate]
            logger.info(f"{self.log_prefix} Activation du profil Optuna: '{profile_to_activate}' - {profile_config.description}")
            self.active_optuna_settings.sampler_name = profile_config.sampler_name
            self.active_optuna_settings.sampler_params = profile_config.sampler_params.copy() if profile_config.sampler_params else {}
            self.active_optuna_settings.pruner_name = profile_config.pruner_name
            self.active_optuna_settings.pruner_params = profile_config.pruner_params.copy() if profile_config.pruner_params else {}
        elif profile_to_activate:
            logger.warning(f"{self.log_prefix} Profil Optuna '{profile_to_activate}' non trouvé. Utilisation des paramètres top-level.")
        else:
            logger.info(f"{self.log_prefix} Aucun profil Optuna par défaut activé. Utilisation des paramètres top-level.")


    def enable_checkpointing(self, checkpoint_dir: Union[str, Path], interval_trials: int = 10) -> None:
        """
        Active le checkpointing en s'assurant que le stockage est persistant (RDB).
        Si le stockage actuel est en mémoire, il est changé pour un fichier SQLite.
        """
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_interval_trials = interval_trials
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Si le storage_url actuel est None (mémoire) ou ne pointe pas vers un fichier,
        # on le configure pour utiliser un fichier SQLite dans checkpoint_dir.
        # Si c'est déjà un RDB (sqlite, postgres, mysql) ou Redis, on le conserve.
        is_memory_storage = self.storage_url is None
        is_non_file_rdb = self.storage_url and not self.storage_url.startswith("sqlite:///")
        
        if is_memory_storage or (self.storage_url and not self.storage_url.startswith("redis://") and not Path(self.storage_url.replace("sqlite:///", "")).parent == self.checkpoint_dir):
            db_file_name = f"optuna_study_{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}.db"
            self.db_path = self.checkpoint_dir / db_file_name
            self.storage_url = f"sqlite:///{self.db_path.resolve()}"
            logger.info(f"{self.log_prefix} Checkpointing activé. Storage configuré sur RDB (SQLite): {self.storage_url}")
        else:
            logger.info(f"{self.log_prefix} Checkpointing: Utilisation du storage existant: {self.storage_url} (supposé persistant).")
        
        # Ajouter une callback pour loguer les "checkpoints" (qui sont implicites avec RDB)
        # L'ancienne CheckpointCallback d'Optuna n'est plus recommandée avec RDB.
        # On peut créer une callback custom si on veut loguer tous les N trials.
        # Pour l'instant, la HeartbeatCallback donne déjà un suivi.
        # self.add_study_callbacks([CheckpointCallback(self.checkpoint_dir, interval_trials)])


    def add_study_callbacks(self, callbacks_to_add: List[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]]) -> None:
        """Ajoute des callbacks à l'étude."""
        self.callbacks.extend(callbacks_to_add)
        logger.info(f"{self.log_prefix} {len(callbacks_to_add)} callback(s) ajouté(s). Total callbacks: {len(self.callbacks)}")

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Crée une instance de sampler Optuna basée sur la configuration active."""
        sampler_name = self.active_optuna_settings.sampler_name.lower()
        sampler_params = self.active_optuna_settings.sampler_params or {}
        logger.info(f"{self.log_prefix} Création du sampler: '{sampler_name}' avec params: {sampler_params}")
        # ... (logique existante de _create_sampler) ...
        # Ajout de la gestion des samplers d'intégration
        try:
            if sampler_name == 'tpesampler': return optuna.samplers.TPESampler(**sampler_params)
            elif sampler_name == 'nsgaiisampler': return optuna.samplers.NSGAIISampler(**sampler_params)
            elif sampler_name == 'cmaessampler': return optuna.samplers.CmaEsSampler(**sampler_params)
            elif sampler_name == 'randomsampler': return optuna.samplers.RandomSampler(**sampler_params)
            elif sampler_name == 'botorchsampler' and BoTorchSampler: return BoTorchSampler(**sampler_params) # type: ignore
            elif sampler_name == 'skoptsampler' and SkoptSampler: return SkoptSampler(**sampler_params) # type: ignore
            else:
                logger.warning(f"{self.log_prefix} Sampler '{sampler_name}' inconnu ou non disponible. Fallback sur TPESampler.")
                return optuna.samplers.TPESampler()
        except Exception as e_sampler_create:
            logger.error(f"{self.log_prefix} Erreur création sampler '{sampler_name}': {e_sampler_create}. Fallback sur TPESampler.", exc_info=True)
            return optuna.samplers.TPESampler()


    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Crée une instance de pruner Optuna."""
        pruner_name = self.active_optuna_settings.pruner_name.lower()
        pruner_params = self.active_optuna_settings.pruner_params or {}
        n_trials_target = self.active_optuna_settings.n_trials

        # Pruning agressif si > 1000 trials
        if n_trials_target > 1000 and pruner_name not in ['successivehalvingpruner', 'hyperbandpruner']:
            logger.info(f"{self.log_prefix} Plus de 1000 essais ciblés. Passage à SuccessiveHalvingPruner pour un élagage plus agressif.")
            pruner_name = 'successivehalvingpruner'
            # Les pruner_params par défaut pour SuccessiveHalvingPruner sont généralement bons.
            # On pourrait ajuster min_resource, reduction_factor ici si besoin.
            pruner_params.setdefault('min_resource', 1) # Standard
            pruner_params.setdefault('reduction_factor', 4) # Standard
            pruner_params.setdefault('min_early_stopping_rate', 0) # Standard

        logger.info(f"{self.log_prefix} Création du pruner: '{pruner_name}' avec params: {pruner_params}")
        # ... (logique existante de _create_pruner) ...
        try:
            if pruner_name == 'medianpruner': return optuna.pruners.MedianPruner(**pruner_params)
            elif pruner_name == 'hyperbandpruner': return optuna.pruners.HyperbandPruner(**pruner_params)
            elif pruner_name == 'noppruner': return optuna.pruners.NopPruner()
            elif pruner_name == 'patientpruner': return optuna.pruners.PatientPruner(**pruner_params)
            elif pruner_name == 'successivehalvingpruner': return optuna.pruners.SuccessiveHalvingPruner(**pruner_params)
            else:
                logger.warning(f"{self.log_prefix} Pruner '{pruner_name}' inconnu. Fallback sur MedianPruner.")
                return optuna.pruners.MedianPruner()
        except Exception as e_pruner_create:
            logger.error(f"{self.log_prefix} Erreur création pruner '{pruner_name}': {e_pruner_create}. Fallback sur MedianPruner.", exc_info=True)
            return optuna.pruners.MedianPruner()

    def create_custom_sampler(self, sampler_config: Dict[str, Any]) -> optuna.samplers.BaseSampler:
        """
        Factory pour créer des samplers Optuna personnalisés ou d'intégration.
        """
        name = sampler_config.get("name", "").lower()
        params = sampler_config.get("params", {})
        log_prefix_custom_sampler = f"{self.log_prefix}[CreateCustomSampler:{name}]"
        logger.info(f"{log_prefix_custom_sampler} Tentative de création avec params: {params}")

        if name == "botorch" and BoTorchSampler:
            return BoTorchSampler(**params) # type: ignore
        elif name == "skopt" and SkoptSampler:
            return SkoptSampler(**params) # type: ignore
        # Ajouter d'autres samplers custom/intégration ici
        # elif name == "smac_sampler_custom": # Exemple pour un SMAC custom (nécessiterait une classe wrapper)
        #     from some_custom_smac_wrapper import SmacSamplerWrapper
        #     return SmacSamplerWrapper(**params)
        else:
            # Fallback sur la création de sampler standard si le nom n'est pas reconnu comme "custom"
            logger.warning(f"{log_prefix_custom_sampler} Sampler custom '{name}' non reconnu ou non disponible. "
                           "Tentative de création via la logique standard de _create_sampler.")
            # Temporairement, on utilise les settings actifs pour recréer un sampler standard
            # si le nom custom n'est pas géré. Idéalement, on aurait une distinction claire.
            temp_active_settings = dataclasses.replace(self.active_optuna_settings)
            temp_active_settings.sampler_name = name # Utiliser le nom demandé
            temp_active_settings.sampler_params = params
            
            # Recréer un sampler standard avec ces settings temporaires
            # Ceci est un peu un hack, il faudrait une meilleure structure si on mélange
            # les samplers "standards" et "customs" via cette fonction.
            # Pour l'instant, on assume que cette fonction est pour des samplers *non standards*.
            # Si `name` est "TPESampler", on le recrée.
            if name in ['tpesampler', 'nsgaiisampler', 'cmaessampler', 'randomsampler']:
                 current_sampler_name_backup = self.active_optuna_settings.sampler_name
                 current_sampler_params_backup = self.active_optuna_settings.sampler_params
                 self.active_optuna_settings.sampler_name = name
                 self.active_optuna_settings.sampler_params = params
                 sampler_instance = self._create_sampler() # Utilise les settings actifs modifiés
                 self.active_optuna_settings.sampler_name = current_sampler_name_backup # Restaurer
                 self.active_optuna_settings.sampler_params = current_sampler_params_backup
                 return sampler_instance
            else:
                logger.error(f"{log_prefix_custom_sampler} Sampler custom '{name}' non implémenté ou non disponible. "
                               "Retour de TPESampler par défaut.")
                return optuna.samplers.TPESampler()


    def run_study(self,
                  data_1min_cleaned_is_slice: pd.DataFrame,
                  objective_evaluator_class: Type['ObjectiveFunctionEvaluator']
                  ) -> optuna.Study:
        study_name = f"{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}_is_opt"
        logger.info(f"{self.log_prefix} Préparation étude Optuna IS: '{study_name}', Storage: {self.storage_url}")

        # ... (logique de create_study existante avec directions, sampler, pruner) ...
        objectives_names: List[str] = self.active_optuna_settings.objectives_names
        objectives_directions_str: List[str] = self.active_optuna_settings.objectives_directions
        
        # Convertir les directions string en objets Optuna StudyDirection
        optuna_directions: List[optuna.study.StudyDirection] = []
        for i, direction_str in enumerate(objectives_directions_str):
            try:
                optuna_directions.append(optuna.study.StudyDirection[direction_str.upper()])
            except KeyError:
                logger.error(f"{self.log_prefix} Direction d'objectif invalide '{direction_str}' pour '{objectives_names[i]}'. "
                             "Utilisation de MAXIMIZE par défaut.")
                optuna_directions.append(optuna.study.StudyDirection.MAXIMIZE)

        study: optuna.Study
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage_url, # Utilise RDB si self.storage_url est un DSN sqlite://
                sampler=self._create_sampler(),
                pruner=self._create_pruner(),
                directions=optuna_directions, # Utiliser les objets StudyDirection
                load_if_exists=True # Essentiel pour la reprise (auto-resume)
            )
            logger.info(f"{self.log_prefix} Étude '{study.study_name}' créée/chargée. Objectifs: {objectives_names}, Directions: {study.directions}")
        except Exception as e_create_study:
            logger.critical(f"{self.log_prefix} Échec création/chargement étude '{study_name}': {e_create_study}", exc_info=True)
            raise

        # Ajouter les callbacks configurés (MLflow, TensorBoard si disponibles et configurés)
        # Exemple:
        # if MLflowCallback and self.app_config.monitoring.mlflow_tracking_uri:
        #     mlflc = MLflowCallback(
        #         tracking_uri=self.app_config.monitoring.mlflow_tracking_uri,
        #         metric_name=[name.replace(" ", "_") for name in objectives_names] # MLflow n'aime pas les espaces
        #     )
        #     self.callbacks.append(mlflc)
        
        # Ajouter un callback d'arrêt anticipé si configuré
        if self.active_optuna_settings.pruner_params.get("early_stopping_rounds"): # Supposons un param pour ça
            early_stop_rounds = int(self.active_optuna_settings.pruner_params["early_stopping_rounds"])
            patience_rounds = int(self.active_optuna_settings.pruner_params.get("early_stopping_patience", 0))
            self.callbacks.append(EarlyStoppingCallback(early_stopping_rounds=early_stop_rounds, patience=patience_rounds))


        objective_instance = objective_evaluator_class(
            strategy_name_key=self.strategy_name, # Renommé pour correspondre
            strategy_config_dict=self.strategy_config_dict,
            df_enriched_slice=data_1min_cleaned_is_slice,
            optuna_objectives_config={
                'objectives_names': objectives_names,
                'objectives_directions': objectives_directions_str # Passer les strings originaux
            },
            pair_symbol=self.pair_symbol, symbol_info_data=self.symbol_info_data,
            app_config=self.app_config, run_id=self.run_id,
            strategy_loader=self.app_config.strategy_loader_instance,  # type: ignore # Passer l'instance
            cache_manager=self.app_config.cache_manager_instance,      # type: ignore # Passer l'instance
            error_handler=self.app_config.error_handler_instance,      # type: ignore # Passer l'instance
            is_oos_eval=False
        )

        n_trials_target = self.active_optuna_settings.n_trials
        n_jobs_parallel = self.active_optuna_settings.n_jobs
        if n_jobs_parallel == 0: n_jobs_parallel = 1
        
        completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials_to_run = n_trials_target - completed_trials_count

        if remaining_trials_to_run <= 0:
            logger.info(f"{self.log_prefix} Étude '{study.study_name}' a déjà {completed_trials_count} essai(s) complété(s). Pas de nouveaux essais.")
        else:
            logger.info(f"{self.log_prefix} Démarrage optimisation Optuna IS pour '{study.study_name}'. "
                        f"Cible: {n_trials_target}, Restants: {remaining_trials_to_run}, Jobs: {n_jobs_parallel}.")
            try:
                study.optimize(
                    objective_instance,
                    n_trials=remaining_trials_to_run,
                    n_jobs=n_jobs_parallel,
                    gc_after_trial=True,
                    callbacks=self.callbacks, # Utiliser les callbacks configurés
                    catch=(Exception,)
                )
            except optuna.exceptions.OptunaError as e_opt_err: # Ex: si un callback lève une erreur non gérée
                logger.error(f"{self.log_prefix} Erreur Optuna durant study.optimize: {e_opt_err}", exc_info=True)
            except Exception as e_gen_opt:
                 logger.error(f"{self.log_prefix} Erreur générale durant study.optimize: {e_gen_opt}", exc_info=True)


        # ... (logique de fin de run_study existante) ...
        final_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"{self.log_prefix} Optimisation IS terminée pour '{study.study_name}'. Essais complétés: {final_completed_trials}.")
        return study

    def get_study_analytics(self, study: optuna.Study, target_total_trials: Optional[int] = None) -> StudyAnalytics:
        """Calcule et retourne des analyses sur une étude Optuna."""
        log_prefix_analytics = f"{self.log_prefix}[Analytics:{study.study_name}]"
        logger.info(f"{log_prefix_analytics} Calcul des analyses de l'étude...")

        total_trials_in_study = len(study.trials)
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        start_time_dt = study.user_attrs.get("study_start_datetime_utc") # Supposer qu'on le stocke
        if not start_time_dt and study.trials: # Fallback sur le premier trial
            start_time_dt = study.trials[0].datetime_start.isoformat() if study.trials[0].datetime_start else None
        
        current_time_dt_iso = datetime.now(timezone.utc).isoformat()
        elapsed_seconds: Optional[float] = None
        if start_time_dt:
            try:
                elapsed_seconds = (datetime.fromisoformat(current_time_dt_iso.replace("Z","+00:00")) - datetime.fromisoformat(start_time_dt.replace("Z","+00:00"))).total_seconds()
            except: pass # pylint: disable=bare-except

        trials_per_sec: Optional[float] = None
        if elapsed_seconds and elapsed_seconds > 0 and completed_trials > 0:
            trials_per_sec = completed_trials / elapsed_seconds

        eta_seconds: Optional[float] = None
        actual_target_trials = target_total_trials if target_total_trials is not None else self.active_optuna_settings.n_trials
        if trials_per_sec and trials_per_sec > 0 and actual_target_trials > completed_trials:
            eta_seconds = (actual_target_trials - completed_trials) / trials_per_sec

        best_vals: Optional[List[float]] = None
        best_prms: Optional[Dict[str, Any]] = None
        if study.best_trials: # Pour multi-objectif, c'est une liste. On prend le premier.
            best_trial_for_analytics = study.best_trials[0]
            best_vals = best_trial_for_analytics.values
            best_prms = best_trial_for_analytics.params
        elif hasattr(study, 'best_trial') and study.best_trial: # Pour mono-objectif
            best_vals = [study.best_value] if study.best_value is not None else None # type: ignore
            best_prms = study.best_params

        param_importances_dict: Optional[Dict[str, float]] = None
        if completed_trials > 0: # L'importance ne peut être calculée que s'il y a des essais complétés
            try:
                # Pour multi-objectif, get_param_importances nécessite un 'target'
                if len(study.directions) > 1:
                    # Calculer pour le premier objectif par défaut
                    target_objective_index = 0
                    param_importances_dict = optuna.importance.get_param_importances(
                        study, 
                        target=lambda t: t.values[target_objective_index] if t.values and len(t.values) > target_objective_index else None,
                        target_name=self.active_optuna_settings.objectives_names[target_objective_index] if self.active_optuna_settings.objectives_names else f"objectif_{target_objective_index}"
                    )
                else: # Mono-objectif
                    param_importances_dict = optuna.importance.get_param_importances(study)
            except Exception as e_importance: # Peut échouer si pas assez de diversité ou trop peu d'essais
                logger.warning(f"{log_prefix_analytics} Impossible de calculer l'importance des paramètres: {e_importance}")
        
        analytics_data = StudyAnalytics(
            study_name=study.study_name,
            total_trials_in_study=total_trials_in_study,
            completed_trials=completed_trials,
            pruned_trials=pruned_trials,
            failed_trials=failed_trials,
            start_time_iso=start_time_dt,
            current_time_iso=current_time_dt_iso,
            elapsed_time_seconds=elapsed_seconds,
            trials_per_second=trials_per_sec,
            estimated_time_remaining_seconds=eta_seconds,
            target_total_trials=actual_target_trials,
            best_values=best_vals,
            best_params=best_prms,
            param_importances=param_importances_dict
        )
        logger.info(f"{log_prefix_analytics} Analyses de l'étude calculées.")
        logger.debug(f"{log_prefix_analytics} Données analytiques: {analytics_data}")
        return analytics_data

