import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, TYPE_CHECKING, List

import optuna # type: ignore
import pandas as pd

if TYPE_CHECKING:
    from src.config.definitions import AppConfig, OptunaSettings # OptunaSettings is part of AppConfig
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator

logger = logging.getLogger(__name__)

class StudyManager:
    def __init__(self,
                 app_config: 'AppConfig', # Added app_config
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any],
                 study_output_dir: Path, # Expected to be logs/backtest_optimization/RUN_ID/STRAT/PAIR/CONTEXT/fold_X/
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any]
                 ):
        self.app_config = app_config
        self.optuna_settings: 'OptunaSettings' = self.app_config.global_config.optuna_settings
        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict # This contains params_space
        self.study_output_dir = study_output_dir # This is specific to the fold
        self.pair_symbol = pair_symbol
        self.symbol_info_data = symbol_info_data
        
        self.log_prefix = f"[{self.strategy_name}][{self.pair_symbol}][Fold: {self.study_output_dir.name}]"
        logger.info(f"{self.log_prefix} StudyManager initialized.")

        self.study_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct a specific DB name for this fold's IS optimization study
        db_file_name = f"optuna_is_study_{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}.db"
        self.db_path_str = str((self.study_output_dir / db_file_name).resolve())
        self.storage_url = f"sqlite:///{self.db_path_str}"
        logger.info(f"{self.log_prefix} Optuna study storage URL: {self.storage_url}")


    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Instantiates the Optuna sampler based on configuration."""
        sampler_name = self.optuna_settings.sampler_name.lower()
        sampler_params = self.optuna_settings.sampler_params or {}
        
        logger.info(f"{self.log_prefix} Creating sampler: {sampler_name} with params: {sampler_params}")

        if sampler_name == 'tpesampler':
            return optuna.samplers.TPESampler(**sampler_params)
        elif sampler_name == 'nsgaiisampler':
            # NSGAIISampler might have specific considerations for multi-objective if not handled by default
            return optuna.samplers.NSGAIISampler(**sampler_params)
        elif sampler_name == 'cmaessampler':
            return optuna.samplers.CmaEsSampler(**sampler_params)
        elif sampler_name == 'randomsampler':
            return optuna.samplers.RandomSampler(**sampler_params)
        # Add other supported samplers here
        # Example for BoTorchSampler (requires BoTorch installation)
        # elif sampler_name == 'botorchsampler':
        #     if optuna.integration.BoTorchSampler is None:
        #         logger.warning("BoTorchSampler requested but BoTorch integration not available. Falling back to TPESampler.")
        #         return optuna.samplers.TPESampler()
        #     return optuna.integration.BoTorchSampler(**sampler_params)
        else:
            logger.warning(f"{self.log_prefix} Unknown sampler '{sampler_name}'. Defaulting to TPESampler.")
            return optuna.samplers.TPESampler()

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Instantiates the Optuna pruner based on configuration."""
        pruner_name = self.optuna_settings.pruner_name.lower()
        pruner_params = self.optuna_settings.pruner_params or {}

        logger.info(f"{self.log_prefix} Creating pruner: {pruner_name} with params: {pruner_params}")

        if pruner_name == 'medianpruner':
            return optuna.pruners.MedianPruner(**pruner_params)
        elif pruner_name == 'hyperbandpruner':
            return optuna.pruners.HyperbandPruner(**pruner_params)
        elif pruner_name == 'nopruner':
            return optuna.pruners.NopPruner()
        elif pruner_name == 'patientpruner':
            # PatientPruner constructor expects a trial, which is not available at this stage.
            # It's typically used with specific callbacks or within the objective.
            # If used globally, it might need a default trial or specific setup.
            # For now, if 'PatientPruner' is chosen with params, it might error if it expects a trial.
            # A common way is to not pass trial here and let Optuna handle it, or ensure params are compatible.
            # Let's assume pruner_params are correctly set for global instantiation if possible.
            try:
                return optuna.pruners.PatientPruner(**pruner_params)
            except TypeError as e:
                logger.warning(f"{self.log_prefix} PatientPruner creation failed with params {pruner_params} (may require a trial): {e}. Falling back to MedianPruner.")
                return optuna.pruners.MedianPruner()
        elif pruner_name == 'successivehalvingpruner':
            return optuna.pruners.SuccessiveHalvingPruner(**pruner_params)
        # Add other supported pruners here
        else:
            logger.warning(f"{self.log_prefix} Unknown pruner '{pruner_name}'. Defaulting to MedianPruner.")
            return optuna.pruners.MedianPruner()

    def run_study(self,
                  data_1min_cleaned_is_slice: pd.DataFrame, # This is the df_enriched_slice
                  objective_evaluator_class: Type['ObjectiveEvaluator']
                  ) -> optuna.Study:
        """
        Creates (or loads if existing) and runs the Optuna study for In-Sample optimization.
        """
        # Construct a unique study name for this specific fold
        # self.study_output_dir.name should be "fold_X"
        study_name = f"{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}_is_opt"
        logger.info(f"{self.log_prefix} Preparing to run/load Optuna study: '{study_name}'")

        objectives_names = self.optuna_settings.objectives_names
        objectives_directions = self.optuna_settings.objectives_directions

        if len(objectives_names) != len(objectives_directions):
            logger.error(f"{self.log_prefix} Mismatch between objectives_names ({len(objectives_names)}) and objectives_directions ({len(objectives_directions)}). "
                         "Defaulting to single PNL objective.")
            objectives_names = ["Total Net PnL USDC"]
            objectives_directions = ["maximize"]
        
        # Validate directions
        for direction in objectives_directions:
            if direction not in ["maximize", "minimize"]:
                logger.error(f"{self.log_prefix} Invalid objective direction '{direction}'. Defaulting to 'maximize' for all.")
                objectives_directions = ["maximize"] * len(objectives_names)
                break
        
        study: optuna.Study
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage_url,
                sampler=self._create_sampler(),
                pruner=self._create_pruner(),
                directions=objectives_directions,
                load_if_exists=True # Crucial for resuming studies
            )
            logger.info(f"{self.log_prefix} Optuna study '{study.study_name}' created/loaded successfully from {self.storage_url}.")
        except Exception as e_create:
            logger.critical(f"{self.log_prefix} Failed to create or load Optuna study '{study_name}' at '{self.storage_url}': {e_create}", exc_info=True)
            raise # Re-raise as this is critical

        optuna_objectives_config_for_evaluator = {
            'objectives_names': objectives_names,
            'objectives_directions': objectives_directions
        }

        # Instantiate ObjectiveEvaluator
        objective_instance = objective_evaluator_class(
            strategy_name=self.strategy_name,
            strategy_config_dict=self.strategy_config_dict,
            df_enriched_slice=data_1min_cleaned_is_slice, # Pass the IS enriched data
            simulation_settings=self.app_config.global_config.simulation_defaults.__dict__,
            optuna_objectives_config=optuna_objectives_config_for_evaluator,
            pair_symbol=self.pair_symbol,
            symbol_info_data=self.symbol_info_data,
            app_config=self.app_config, # Pass the full app_config
            is_oos_eval=False, # This is for In-Sample optimization
            is_trial_number_for_oos_log=None # Not applicable for IS run itself
        )

        n_trials_config = self.optuna_settings.n_trials
        n_jobs_config = self.optuna_settings.n_jobs
        
        if n_jobs_config == 0:
            logger.warning(f"{self.log_prefix} Optuna n_jobs is 0, which is invalid. Defaulting to 1 (sequential).")
            n_jobs_config = 1
        # n_jobs = -1 means use all CPUs, which is valid for Optuna.

        # Check completed trials to support resuming
        completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = n_trials_config - completed_trials_count

        if remaining_trials <= 0 and completed_trials_count >= n_trials_config:
            logger.info(f"{self.log_prefix} Study '{study.study_name}' already has {completed_trials_count} completed trials (target: {n_trials_config}). No new trials will be run for this fold.")
        else:
            if remaining_trials < 0 : # Should not happen if logic is correct
                logger.warning(f"{self.log_prefix} remaining_trials is negative ({remaining_trials}). This is unexpected. Setting to 0.")
                remaining_trials = 0
            
            logger.info(f"{self.log_prefix} Starting Optuna IS optimization for '{study.study_name}'. "
                        f"Target total trials: {n_trials_config}. Completed: {completed_trials_count}. Remaining: {remaining_trials}. "
                        f"Using {n_jobs_config} job(s).")
            try:
                study.optimize(
                    objective_instance, # The callable ObjectiveEvaluator instance
                    n_trials=remaining_trials, # Run only the remaining number of trials
                    n_jobs=n_jobs_config,
                    gc_after_trial=True, # Helps manage memory
                    callbacks=[] # Add any Optuna callbacks if needed
                )
            except Exception as e_optimize:
                logger.error(f"{self.log_prefix} Error during Optuna study.optimize for '{study.study_name}': {e_optimize}", exc_info=True)
                # The study object will still be returned, potentially with partial results.
        
        final_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"{self.log_prefix} Optuna IS optimization finished for '{study.study_name}'. Total completed trials in study: {final_completed_trials}.")
        return study

