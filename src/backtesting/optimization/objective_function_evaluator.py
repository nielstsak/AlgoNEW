# src/backtesting/optimization/objective_function_evaluator.py
"""
Ce module définit ObjectiveFunctionEvaluator, la fonction objectif pour Optuna.
Elle évalue un ensemble d'hyperparamètres (un "trial") en exécutant un backtest
et en retournant les métriques de performance qui servent d'objectifs pour
l'optimisation. Elle est utilisée pour l'optimisation In-Sample (IS) et
la validation Out-of-Sample (OOS).
"""
import logging
import time
import importlib
import uuid # Pour les logs OOS détaillés si besoin
from typing import Any, Dict, Optional, Tuple, List, Type, Union, TYPE_CHECKING, cast
from datetime import timezone # Pour s'assurer que les timestamps sont UTC
import os 
from pathlib import Path # Ajouté pour une manipulation de chemin plus robuste

import numpy as np
import pandas as pd
import optuna

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import ParamDetail, SimulationDefaults, StrategyParamsConfig
    from src.strategies.base import BaseStrategy
    from src.backtesting.core_simulator import BacktestRunner

# Imports depuis l'application
try:
    from src.config.definitions import ParamDetail, SimulationDefaults
    from src.backtesting.core_simulator import BacktestRunner
    from src.backtesting.performance_analyzer import calculate_performance_metrics_from_inputs
    from src.strategies.base import BaseStrategy
    from src.backtesting.indicator_calculator import calculate_indicators_for_trial
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"ObjectiveFunctionEvaluator: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH."
    )
    raise

logger = logging.getLogger(__name__)

class ObjectiveFunctionEvaluator:
    """
    Fonction objectif pour Optuna. Évalue un ensemble d'hyperparamètres
    en exécutant un backtest et retourne les métriques de performance.
    """
    def __init__(self,
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any],
                 df_enriched_slice: pd.DataFrame,
                 optuna_objectives_config: Dict[str, Any],
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any],
                 app_config: 'AppConfig',
                 run_id: str,
                 is_oos_eval: bool = False,
                 is_trial_number_for_oos_log: Optional[int] = None
                ):
        self.strategy_name_key = strategy_name
        self.strategy_config_dict = strategy_config_dict
        self.df_enriched_slice = df_enriched_slice.copy()
        self.optuna_objectives_config = optuna_objectives_config
        self.pair_symbol = pair_symbol.upper()
        self.symbol_info_data = symbol_info_data
        self.app_config = app_config
        self.run_id = run_id
        self.is_oos_eval = is_oos_eval
        self.is_trial_number_for_oos_log = is_trial_number_for_oos_log

        self.log_prefix = (
            f"[{self.strategy_name_key}/{self.pair_symbol}]"
            f"[Run:{self.run_id}]"
            f"[{'OOS' if self.is_oos_eval else 'IS_Opt'}]"
            f"{f'[OrigIS_Trial:{self.is_trial_number_for_oos_log}]' if self.is_oos_eval and self.is_trial_number_for_oos_log is not None else ''}"
        )

        self.strategy_script_reference: str = self.strategy_config_dict.get('script_reference', '')
        self.strategy_class_name: str = self.strategy_config_dict.get('class_name', '')
        if not self.strategy_script_reference or not self.strategy_class_name:
            msg = f"'script_reference' ou 'class_name' manquant dans strategy_config_dict pour {self.strategy_name_key}."
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)

        self.params_space_details: Dict[str, ParamDetail] = {}
        raw_params_space = self.strategy_config_dict.get('params_space', {})
        if isinstance(raw_params_space, dict):
            for param_key, param_value_dict in raw_params_space.items():
                if isinstance(param_value_dict, dict):
                    try:
                        self.params_space_details[param_key] = ParamDetail(**param_value_dict)
                    except Exception as e_pd_create:
                        logger.error(f"{self.log_prefix} Erreur lors de la création de ParamDetail pour '{param_key}' avec données {param_value_dict}: {e_pd_create}")
                else:
                    logger.warning(f"{self.log_prefix} Valeur inattendue pour '{param_key}' dans params_space. Attendu un dict, reçu {type(param_value_dict)}.")
        else:
            logger.error(f"{self.log_prefix} 'params_space' pour la stratégie '{self.strategy_name_key}' n'est pas un dictionnaire valide.")
        
        if not self.params_space_details and not self.is_oos_eval:
             logger.warning(f"{self.log_prefix} params_space_details est VIDE pour l'évaluation IS.")

        if not isinstance(self.df_enriched_slice.index, pd.DatetimeIndex):
            msg = "df_enriched_slice doit avoir un DatetimeIndex."
            logger.error(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        if self.df_enriched_slice.index.tz is None:
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_localize('UTC') # type: ignore
        elif self.df_enriched_slice.index.tz.utcoffset(None) != timezone.utc.utcoffset(None):
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_convert('UTC') # type: ignore

        if not self.df_enriched_slice.index.is_monotonic_increasing:
            self.df_enriched_slice.sort_index(inplace=True)
        if self.df_enriched_slice.index.duplicated().any():
            self.df_enriched_slice = self.df_enriched_slice[~self.df_enriched_slice.index.duplicated(keep='first')]

        self.last_backtest_results: Optional[Dict[str, Any]] = None
        logger.info(f"{self.log_prefix} ObjectiveFunctionEvaluator initialisé. Données shape: {self.df_enriched_slice.shape}")

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params_for_trial: Dict[str, Any] = {}
        log_prefix_suggest = f"{self.log_prefix}[Trial:{trial.number}][SuggestParams]"

        if not self.params_space_details:
            default_params = self.strategy_config_dict.get('default_params', {})
            if default_params and isinstance(default_params, dict):
                logger.warning(f"{log_prefix_suggest} params_space vide. Utilisation de default_params : {default_params}")
                return default_params.copy()
            raise optuna.exceptions.TrialPruned("params_space vide et pas de default_params.")

        for param_name, p_detail in self.params_space_details.items():
            try:
                if p_detail.type == 'int':
                    low = int(p_detail.low) if p_detail.low is not None else 0
                    high = int(p_detail.high) if p_detail.high is not None else low + 1
                    step = int(p_detail.step or 1)
                    params_for_trial[param_name] = trial.suggest_int(param_name, low, high, step=step)
                elif p_detail.type == 'float':
                    low_f = float(p_detail.low) if p_detail.low is not None else 0.0
                    high_f = float(p_detail.high) if p_detail.high is not None else low_f + 1.0
                    step_f = float(p_detail.step) if p_detail.step is not None else None
                    params_for_trial[param_name] = trial.suggest_float(param_name, low_f, high_f, step=step_f, log=p_detail.log_scale)
                elif p_detail.type == 'categorical' and p_detail.choices:
                    params_for_trial[param_name] = trial.suggest_categorical(param_name, p_detail.choices)
                else: # Fallback si le type n'est pas géré ou si la config est invalide
                    params_for_trial[param_name] = p_detail.default if p_detail.default is not None else (p_detail.low if p_detail.low is not None else None)
                    if params_for_trial[param_name] is None and p_detail.choices: # Fallback pour catégoriel mal configuré
                        params_for_trial[param_name] = p_detail.choices[0]
                    logger.warning(f"{log_prefix_suggest} Type de paramètre '{p_detail.type}' non géré pour '{param_name}' ou configuration ParamDetail invalide. Utilisation de fallback: {params_for_trial[param_name]}")

            except Exception as e_sug:
                logger.error(f"{log_prefix_suggest} Erreur suggestion paramètre '{param_name}': {e_sug}", exc_info=True)
                # Fallback en cas d'erreur de suggestion
                if p_detail.default is not None: params_for_trial[param_name] = p_detail.default
                elif p_detail.type == 'categorical' and p_detail.choices: params_for_trial[param_name] = p_detail.choices[0]
                else: raise optuna.exceptions.TrialPruned(f"Échec suggestion pour {param_name}")
        
        logger.debug(f"{log_prefix_suggest} Paramètres suggérés : {params_for_trial}")
        return params_for_trial

    def _prepare_data_with_dynamic_indicators(self,
                                              strategy_instance_for_trial: 'BaseStrategy',
                                              trial_number_for_log: Optional[Union[int, str]] = None
                                             ) -> pd.DataFrame:
        current_eval_id = trial_number_for_log if trial_number_for_log is not None else "N/A"
        log_prefix_prep = f"{self.log_prefix}[EvalID:{current_eval_id}][PrepDataIndicators]"
        logger.info(f"{log_prefix_prep} Préparation des données avec indicateurs dynamiques via IndicatorCalculator.")

        try:
            required_configs = strategy_instance_for_trial.get_required_indicator_configs()
            if not required_configs:
                logger.warning(f"{log_prefix_prep} Stratégie n'a retourné aucune config d'indicateur requise.")
                base_cols_to_return = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in self.df_enriched_slice.columns]
                if not base_cols_to_return:
                    raise ValueError("Colonnes OHLCV de base manquantes dans df_enriched_slice.")
                return self.df_enriched_slice[base_cols_to_return].copy()
        except Exception as e_get_configs:
            logger.error(f"{log_prefix_prep} Erreur get_required_indicator_configs(): {e_get_configs}", exc_info=True)
            raise ValueError(f"Erreur get_required_indicator_configs: {e_get_configs}")

        try:
            df_for_simulation = calculate_indicators_for_trial(
                df_source_enriched=self.df_enriched_slice,
                required_indicator_configs=required_configs,
                log_prefix_context=f"[Trial:{current_eval_id}]"
            )
        except Exception as e_calc_indic:
            logger.error(f"{log_prefix_prep} Erreur calculate_indicators_for_trial: {e_calc_indic}", exc_info=True)
            raise ValueError(f"Erreur calculate_indicators_for_trial: {e_calc_indic}")

        if df_for_simulation.empty:
            raise ValueError("calculate_indicators_for_trial a retourné un DataFrame vide.")
        
        try:
            df_final_for_simulation = strategy_instance_for_trial._calculate_indicators(df_for_simulation)
        except Exception as e_strat_calc:
            logger.error(f"{log_prefix_prep} Erreur strategy_instance._calculate_indicators(): {e_strat_calc}", exc_info=True)
            raise ValueError(f"Erreur strategy_instance._calculate_indicators: {e_strat_calc}")
        
        if df_final_for_simulation.empty:
            raise ValueError("strategy_instance._calculate_indicators() a retourné un DataFrame vide.")

        logger.info(f"{log_prefix_prep} Préparation des données avec indicateurs terminée. Shape final: {df_final_for_simulation.shape}")
        return df_final_for_simulation

    def _resolve_module_import_path(self, script_reference_from_config: str) -> str:
        """
        Résout le chemin du script de stratégie en un chemin d'importation Python valide.
        """
        log_prefix_import = f"{self.log_prefix}[ResolveImportPath]"
        
        if not self.app_config.project_root:
            logger.error(f"{log_prefix_import} project_root non défini dans AppConfig. "
                         "Impossible de résoudre le chemin d'import de manière robuste.")
            # Fallback sur l'ancienne méthode, moins robuste
            return script_reference_from_config.replace('.py', '').replace(os.sep, '.')

        project_root_path = Path(self.app_config.project_root).resolve()
        script_path_rel_to_config = Path(script_reference_from_config) # Ex: "src/strategies/my_strat.py"

        # Construire le chemin absolu du script
        # Si script_path_rel_to_config est déjà absolu, project_root_path ne sera pas préfixé.
        script_path_abs = (project_root_path / script_path_rel_to_config).resolve()
        logger.debug(f"{log_prefix_import} Chemin absolu du script de stratégie : {script_path_abs}")

        # Tenter de rendre le chemin relatif à un répertoire parent qui est dans sys.path
        # ou à la racine du projet, pour former un chemin d'import Python.
        
        # Cas 1: Le script est sous 'src' et 'src' est un package racine pour les imports
        try:
            # Trouver le dossier 'src' dans le chemin absolu du script
            src_dir_in_path_parts = [part for part in script_path_abs.parents if part.name == 'src']
            if src_dir_in_path_parts:
                src_dir_base = src_dir_in_path_parts[0] # Le 'src' le plus proche du fichier
                # Le chemin d'import commence à partir du parent de 'src' si 'src' lui-même fait partie du module
                # ou à partir de 'src' si le parent de 'src' est la racine d'importation.
                # Généralement, si on a project_root/src/strategies, l'import est src.strategies...
                # Donc, on rend relatif au parent de 'src' (qui est project_root)
                module_rel_path_obj = script_path_abs.relative_to(src_dir_base.parent).with_suffix('')
                module_import_str = '.'.join(module_rel_path_obj.parts)
                logger.debug(f"{log_prefix_import} Chemin d'import (via 'src' parent) : {module_import_str}")
                return module_import_str
        except ValueError: # Si .relative_to échoue
            logger.debug(f"{log_prefix_import} Échec de la résolution relative au parent de 'src'. Tentative par rapport à project_root.")
            pass # Tenter la méthode suivante

        # Cas 2: Le script est sous project_root (mais peut-être pas directement sous 'src')
        try:
            module_rel_path_obj = script_path_abs.relative_to(project_root_path).with_suffix('')
            module_import_str = '.'.join(module_rel_path_obj.parts)
            logger.debug(f"{log_prefix_import} Chemin d'import (via project_root) : {module_import_str}")
            return module_import_str
        except ValueError:
            logger.warning(f"{log_prefix_import} Échec de la résolution relative à project_root. "
                           f"Script path: {script_path_abs}, Project root: {project_root_path}")
            pass

        # Cas 3: Fallback sur l'ancienne méthode (remplacement de séparateurs)
        # Cela suppose que script_reference_from_config est déjà un chemin de type "src/..."
        logger.warning(f"{log_prefix_import} Utilisation de la méthode de fallback pour le chemin d'import (remplacement de séparateurs).")
        return script_reference_from_config.replace('.py', '').replace(os.sep, '.')


    def __call__(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        start_time_trial = time.time()
        trial_log_id: str
        trial_number_for_prepare_log: Optional[Union[int, str]]
        
        if self.is_oos_eval:
            trial_log_id = f"OOS_for_IS_Trial_{self.is_trial_number_for_oos_log}" if self.is_trial_number_for_oos_log is not None else f"OOS_FixedParams_{trial.number if hasattr(trial, 'number') else uuid.uuid4().hex[:6]}"
            trial_number_for_prepare_log = self.is_trial_number_for_oos_log
        else:
            trial_log_id = str(trial.number)
            trial_number_for_prepare_log = trial.number

        current_log_prefix = f"{self.log_prefix}[Trial:{trial_log_id}]"
        logger.info(f"{current_log_prefix} Démarrage de l'évaluation du trial.")

        current_trial_params: Dict[str, Any]
        if self.is_oos_eval:
            current_trial_params = trial.params # Les params sont fixés par enqueue_trial pour OOS
            logger.info(f"{current_log_prefix} Évaluation OOS avec params fixes (de IS trial {self.is_trial_number_for_oos_log}): {current_trial_params}")
        else: 
            try:
                current_trial_params = self._suggest_params(trial)
                if not current_trial_params:
                    logger.error(f"{current_log_prefix} Aucun paramètre suggéré. Élagueage.")
                    trial.set_user_attr("failure_reason", "Aucun paramètre suggéré")
                    raise optuna.exceptions.TrialPruned("Aucun paramètre suggéré.")
                logger.info(f"{current_log_prefix} Paramètres IS suggérés : {current_trial_params}")
            except optuna.exceptions.TrialPruned: raise
            except Exception as e_suggest:
                logger.error(f"{current_log_prefix} Erreur suggestion params : {e_suggest}", exc_info=True)
                trial.set_user_attr("failure_reason", f"Erreur suggestion params: {str(e_suggest)[:100]}")
                raise optuna.exceptions.TrialPruned(f"Erreur suggestion params: {e_suggest}")

        StrategyClassImpl: Type['BaseStrategy']
        try:
            module_import_path = self._resolve_module_import_path(self.strategy_script_reference)
            logger.debug(f"{current_log_prefix} Tentative d'import du module de stratégie : '{module_import_path}' pour la classe '{self.strategy_class_name}'.")
            
            module = importlib.import_module(module_import_path)
            StrategyClassImpl = getattr(module, self.strategy_class_name)
            if not issubclass(StrategyClassImpl, BaseStrategy):
                raise TypeError(f"{self.strategy_class_name} n'est pas une sous-classe de BaseStrategy.")
            logger.debug(f"{current_log_prefix} Classe de stratégie '{self.strategy_class_name}' chargée depuis '{module_import_path}'.")
        except ModuleNotFoundError as e_mnfe:
            logger.error(f"{current_log_prefix} ModuleNotFoundError lors du chargement de la stratégie '{module_import_path if 'module_import_path' in locals() else self.strategy_script_reference}': {e_mnfe}. Vérifiez PYTHONPATH et la structure du projet.", exc_info=True)
            trial.set_user_attr("failure_reason", f"ModuleNotFoundError: {e_mnfe}")
            return self._get_worst_objective_values(f"ModuleNotFoundError: {e_mnfe}")
        except Exception as e_load_strat:
            logger.error(f"{current_log_prefix} Échec chargement classe stratégie {self.strategy_class_name} depuis {self.strategy_script_reference} (module tenté: {module_import_path if 'module_import_path' in locals() else 'non_defini'}): {e_load_strat}", exc_info=True)
            trial.set_user_attr("failure_reason", f"Échec chargement classe strat: {str(e_load_strat)[:100]}")
            return self._get_worst_objective_values(f"Échec chargement classe strat: {e_load_strat}")

        try:
            strategy_instance_for_eval = StrategyClassImpl(
                strategy_name=self.strategy_name_key,
                symbol=self.pair_symbol,
                params=current_trial_params
            )
        except ValueError as e_strat_init_val:
            logger.warning(f"{current_log_prefix} Erreur validation params instanciation stratégie: {e_strat_init_val}. Élagueage.")
            trial.set_user_attr("failure_reason", f"Validation params strat: {str(e_strat_init_val)[:100]}")
            raise optuna.exceptions.TrialPruned(f"Validation params strat: {e_strat_init_val}")
        except Exception as e_strat_init:
            logger.error(f"{current_log_prefix} Erreur instanciation stratégie : {e_strat_init}", exc_info=True)
            trial.set_user_attr("failure_reason", f"Erreur instanciation strat: {str(e_strat_init)[:100]}")
            return self._get_worst_objective_values(f"Erreur instanciation strat: {e_strat_init}")
        
        try:
            df_for_simulation = self._prepare_data_with_dynamic_indicators(
                strategy_instance_for_eval,
                trial_number_for_log=trial_number_for_prepare_log
            )
            if df_for_simulation.empty or df_for_simulation[['open', 'high', 'low', 'close']].isnull().all().all():
                logger.error(f"{current_log_prefix} Préparation données a résulté en DataFrame inutilisable. Élagueage.")
                trial.set_user_attr("failure_reason", "Données inutilisables post-indicateurs")
                raise optuna.exceptions.TrialPruned("Données inutilisables post-indicateurs.")
        except optuna.exceptions.TrialPruned: raise
        except Exception as e_prepare:
            logger.error(f"{current_log_prefix} Erreur préparation données : {e_prepare}", exc_info=True)
            trial.set_user_attr("failure_reason", f"Erreur préparation données: {str(e_prepare)[:100]}")
            return self._get_worst_objective_values(f"Erreur préparation données: {e_prepare}")

        sim_defaults: SimulationDefaults = self.app_config.global_config.simulation_defaults
        leverage_to_use = int(current_trial_params.get('margin_leverage', sim_defaults.margin_leverage))
        
        strategy_instance_for_eval.set_backtest_context(
            pair_config=self.symbol_info_data,
            is_futures=sim_defaults.is_futures_trading,
            leverage=leverage_to_use,
            initial_equity=sim_defaults.initial_capital,
            account_type=self.app_config.data_config.source_details.asset_type
        )

        simulator = BacktestRunner(
            df_ohlcv_with_indicators=df_for_simulation,
            strategy_instance=strategy_instance_for_eval,
            initial_equity=sim_defaults.initial_capital,
            leverage=leverage_to_use,
            symbol=self.pair_symbol,
            pair_config=self.symbol_info_data,
            trading_fee_bps=sim_defaults.trading_fee_bps,
            slippage_config_dict=sim_defaults.slippage_config.__dict__,
            is_futures=sim_defaults.is_futures_trading,
            run_id=self.run_id,
            is_oos_simulation=self.is_oos_eval,
            verbosity=0 if not self.is_oos_eval else sim_defaults.backtest_verbosity
        )

        trades: List[Dict[str, Any]]
        equity_curve_df: pd.DataFrame
        oos_detailed_log_from_sim: List[Dict[str, Any]]

        try:
            logger.info(f"{current_log_prefix} Démarrage simulation backtest...")
            trades, equity_curve_df, _, oos_detailed_log_from_sim = simulator.run_simulation()
            self.last_backtest_results = {
                "params": current_trial_params, "trades": trades,
                "equity_curve_df": equity_curve_df, "oos_detailed_trades_log": oos_detailed_log_from_sim,
                "metrics": {}
            }
            logger.info(f"{current_log_prefix} Simulation terminée. Trades: {len(trades)}")
        except optuna.exceptions.TrialPruned as e_pruned_sim:
            logger.info(f"{current_log_prefix} Trial élagué pendant simulation : {e_pruned_sim}")
            trial.set_user_attr("failure_reason", f"Élagué pendant sim: {str(e_pruned_sim)[:100]}")
            raise
        except Exception as e_sim:
            logger.error(f"{current_log_prefix} Erreur durant BacktestRunner : {e_sim}", exc_info=True)
            trial.set_user_attr("failure_reason", f"Erreur BacktestRunner: {str(e_sim)[:100]}")
            return self._get_worst_objective_values(f"Erreur BacktestRunner: {e_sim}")

        equity_series_for_metrics = pd.Series(dtype=float)
        if not equity_curve_df.empty and 'timestamp' in equity_curve_df.columns and 'equity' in equity_curve_df.columns:
            ec_df_copy = equity_curve_df.copy()
            ec_df_copy['timestamp'] = pd.to_datetime(ec_df_copy['timestamp'], errors='coerce', utc=True)
            ec_df_copy.dropna(subset=['timestamp', 'equity'], inplace=True)
            if not ec_df_copy.empty:
                equity_series_for_metrics = ec_df_copy.set_index('timestamp')['equity'].sort_index()
        
        if equity_series_for_metrics.empty:
            logger.warning(f"{current_log_prefix} Série d'équité vide. Utilisation capital initial pour métriques.")
            start_ts_data = df_for_simulation.index.min() if not df_for_simulation.empty else pd.Timestamp.now(tz='UTC')
            equity_series_for_metrics = pd.Series([sim_defaults.initial_capital], index=[start_ts_data])

        metrics = calculate_performance_metrics_from_inputs(
            trades_df=pd.DataFrame(trades), equity_curve_series=equity_series_for_metrics,
            initial_capital=sim_defaults.initial_capital,
            risk_free_rate_daily=(1 + sim_defaults.risk_free_rate)**(1/252) - 1,
            periods_per_year=252
        )
        metrics['Total Trades'] = metrics.get('Total Trades', len(trades))
        
        if self.last_backtest_results: self.last_backtest_results["metrics"] = metrics.copy()

        if not self.is_oos_eval:
            for key, value in metrics.items():
                attr_val_optuna: Any = None
                if isinstance(value, (int, float, str, bool)) and pd.notna(value) and not (isinstance(value, float) and (np.isinf(value) or np.isnan(value))):
                    attr_val_optuna = value
                elif pd.notna(value): attr_val_optuna = str(value)
                if attr_val_optuna is not None:
                    try: trial.set_user_attr(key, attr_val_optuna)
                    except TypeError:
                        logger.debug(f"{current_log_prefix} Impossible de définir user_attr '{key}' avec valeur '{attr_val_optuna}' (type: {type(attr_val_optuna)}). Conversion en str.")
                        trial.set_user_attr(key, str(attr_val_optuna))

        objective_values_list: List[float] = []
        obj_names: List[str] = self.optuna_objectives_config.get('objectives_names', ["Total Net PnL USDC"])
        obj_dirs: List[str] = self.optuna_objectives_config.get('objectives_directions', ["maximize"] * len(obj_names))

        for i, metric_name in enumerate(obj_names):
            value = metrics.get(metric_name)
            direction = obj_dirs[i].lower() if i < len(obj_dirs) and isinstance(obj_dirs[i], str) else "maximize"

            if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
                logger.warning(f"{current_log_prefix} Objectif '{metric_name}' valeur invalide: {value}. Assignation valeur très mauvaise.")
                value = -1e12 if direction == "maximize" else 1e12
                if metric_name == "Total Trades" and (not trades or len(trades) == 0): value = 0.0
                elif "Ratio" in metric_name and (not trades or len(trades) == 0): value = -10.0 
                elif ("PnL" in metric_name or "Profit" in metric_name) and (not trades or len(trades) == 0): value = 0.0
            objective_values_list.append(float(value))

        log_metrics_summary = {k: metrics.get(k) for k in ["Total Net PnL USDC", "Sharpe Ratio", "Max Drawdown Pct", "Total Trades", "Win Rate Pct"]}
        logger.info(f"{current_log_prefix} Métriques clés: { {k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in log_metrics_summary.items()} }")
        
        end_time_trial = time.time()
        logger.info(f"{current_log_prefix} Évaluation trial terminée en {end_time_trial - start_time_trial:.2f}s. Objectifs ({obj_names}): {objective_values_list}")
        
        if trial.should_prune():
            logger.info(f"{current_log_prefix} Trial élagué par le pruner Optuna après évaluation.")
            trial.set_user_attr("failure_reason", "Élagué par le pruner Optuna post-évaluation")
            raise optuna.exceptions.TrialPruned()

        return tuple(objective_values_list) if len(objective_values_list) > 1 else objective_values_list[0]

    def _get_worst_objective_values(self, reason_for_worst: str) -> Union[float, Tuple[float, ...]]:
        logger.warning(f"{self.log_prefix} Retour des pires valeurs d'objectif. Raison: {reason_for_worst}")
        obj_dirs: List[str] = self.optuna_objectives_config.get('objectives_directions', ['maximize'])
        num_objectives = len(self.optuna_objectives_config.get('objectives_names', ['']))
        
        if not obj_dirs or len(obj_dirs) != num_objectives:
            obj_dirs = ['maximize'] * num_objectives
            logger.warning(f"{self.log_prefix} Directions d'objectifs incohérentes ou manquantes, utilisation de 'maximize' pour tous.")

        worst_values = tuple([-1e12 if d.lower() == "maximize" else 1e12 for d in obj_dirs])
        return worst_values if len(worst_values) > 1 else worst_values[0]
