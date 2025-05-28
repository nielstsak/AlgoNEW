# src/backtesting/optimization/objective_function_evaluator.py
"""
Ce module défini ObjectiveFunctionEvaluator, la fonction objectif pour Optuna.
elle évalue un ensemble d'hyperparamètres (un "trial") en exécutant un backtest
et en retournant les métriques de performance qui servent d'objectifs pour
l'optimisation.
Refactorisé pour l'injection de dépendances, la mise en cache des évaluations,
et une meilleure gestion des erreurs.
"""
import logging
import time
import importlib
import uuid
import json # Pour hasher les dictionnaires de paramètres de manière stable
import hashlib # Pour générer les clés de cache
import traceback # Pour les traces d'erreur complètes
import contextlib # Pourrait être utilisé pour des context managers si besoin
import functools # Pour @functools.lru_cache ou d'autres décorateurs

from typing import Any, Dict, Optional, Tuple, List, Type, Union, TYPE_CHECKING, Callable, cast, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import optuna

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import ParamDetail, SimulationDefaults, StrategyParamsConfig
    from src.strategies.base import IStrategy # Utiliser l'interface
    from src.backtesting.core_simulator import BacktestRunner
    from src.core.interfaces import ICacheManager # Importer la vraie interface

# Imports depuis l'application
try:
    from src.config.definitions import ParamDetail, SimulationDefaults
    from src.backtesting.core_simulator import BacktestRunner
    from src.backtesting.performance_analyzer import calculate_performance_metrics_from_inputs
    from src.strategies.base import IStrategy # Utiliser l'interface IStrategy
    from src.backtesting.indicator_calculator import calculate_indicators_for_trial
    from src.core.interfaces import ICacheManager # Importer la vraie interface
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"ObjectiveFunctionEvaluator: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH."
    )
    # Définir des placeholders si les imports échouent pour permettre le chargement du module
    class IStrategy: pass # type: ignore
    class ICacheManager: # type: ignore
        def get_or_compute(self, key: str, compute_func: Callable[[], Any], ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> Any: return compute_func()
    raise

logger = logging.getLogger(__name__)

# --- Dataclass pour les résultats d'erreur ---
@dataclass
class ErrorResult:
    """Structure pour retourner les détails d'une erreur d'évaluation."""
    error_type: str
    message: str
    traceback_str: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    suggestions: List[str] = field(default_factory=list) # Suggestions pour résoudre l'erreur

# --- Interfaces Placeholder pour les dépendances injectées (à déplacer dans core.interfaces.py) ---
class IStrategyLoader(Protocol if TYPE_CHECKING else object):
    """Interface pour un chargeur de stratégie."""
    def load_strategy(self,
                      strategy_name_key: str,
                      params_for_strategy: Dict[str, Any],
                      strategy_script_ref: str,
                      strategy_class_name: str,
                      pair_symbol: str # Ajouté pour que le loader puisse passer le symbole
                     ) -> 'IStrategy':
        ...

class IErrorHandler(Protocol if TYPE_CHECKING else object):
    """Interface pour un gestionnaire d'erreurs."""
    def handle_evaluation_error(self,
                                exception: Exception,
                                context: Dict[str, Any],
                                trial: Optional[optuna.Trial] = None
                               ) -> ErrorResult:
        ...

# --- Implémentation Placeholder Simple (pour test si non injecté de l'extérieur) ---
class SimpleStrategyLoader(IStrategyLoader):
    """Chargeur de stratégie simple pour l'ObjectiveFunctionEvaluator."""
    def load_strategy(self,
                      strategy_name_key: str, # Nom de la config de stratégie (ex: "ma_crossover_strategy")
                      params_for_strategy: Dict[str, Any], # Paramètres du trial Optuna
                      strategy_script_ref: str, # Ex: "src/strategies/ma_crossover_strategy.py"
                      strategy_class_name: str, # Ex: "MaCrossoverStrategy"
                      pair_symbol: str
                     ) -> 'IStrategy':
        log_prefix_loader = f"[SimpleStrategyLoader][{strategy_name_key}]"
        logger.debug(f"{log_prefix_loader} Tentative de chargement de la stratégie '{strategy_class_name}' depuis '{strategy_script_ref}' avec params: {params_for_strategy}")
        
        # Convertir le chemin de script en chemin d'import Python
        # Ex: "src/strategies/my_strat.py" -> "src.strategies.my_strat"
        module_path_standardized = strategy_script_ref.replace('\\', '/').removesuffix('.py')
        parts = module_path_standardized.split('/')
        if parts and parts[0] != 'src': # S'assurer que ça commence par src si c'est un chemin relatif depuis la racine
            module_import_str = module_path_standardized.replace('/', '.')
            if not module_import_str.startswith('src.') and 'src.' in module_import_str: # ex: project_root/src/...
                module_import_str = f"src.{module_import_str.split('src.', 1)[-1]}"
            elif not module_import_str.startswith('src.'): # ex: strategies/my_strat
                 logger.warning(f"{log_prefix_loader} Chemin d'import '{module_import_str}' ne commence pas par 'src.'. Tentative d'import direct.")
        else: # ex: src/strategies/my_strat
            module_import_str = '.'.join(parts)
        
        try:
            module = importlib.import_module(module_import_str)
            StrategyClassImpl = getattr(module, strategy_class_name)
            
            # S'assurer que c'est bien une sous-classe de IStrategy (ou BaseStrategy si IStrategy n'est pas complètement défini)
            # from src.strategies.base import BaseStrategy # Import local pour ce check
            # if not issubclass(StrategyClassImpl, (IStrategy, BaseStrategy)): # type: ignore
            #     raise TypeError(f"La classe '{strategy_class_name}' n'hérite pas de IStrategy/BaseStrategy.")

            # strategy_name_key est le nom de la config (ex: "ma_crossover_strategy")
            # pair_symbol est la paire pour cette évaluation
            # params_for_strategy sont les hyperparamètres du trial Optuna
            instance = StrategyClassImpl(
                strategy_name=strategy_name_key, # Utiliser le nom de la config pour l'instance
                symbol=pair_symbol,
                params=params_for_strategy
            )
            logger.info(f"{log_prefix_loader} Stratégie '{strategy_class_name}' instanciée avec succès.")
            return instance
        except ModuleNotFoundError as e_mnfe:
            logger.error(f"{log_prefix_loader} ModuleNotFoundError pour '{module_import_str}': {e_mnfe}. Vérifiez PYTHONPATH et le chemin du script.", exc_info=True)
            raise
        except AttributeError as e_attr:
            logger.error(f"{log_prefix_loader} AttributeError: Classe '{strategy_class_name}' non trouvée dans module '{module_import_str}': {e_attr}", exc_info=True)
            raise
        except Exception as e_inst:
            logger.error(f"{log_prefix_loader} Erreur lors de l'instanciation de '{strategy_class_name}': {e_inst}", exc_info=True)
            raise

class SimpleErrorHandler(IErrorHandler):
    """Gestionnaire d'erreurs simple."""
    def handle_evaluation_error(self,
                                exception: Exception,
                                context: Dict[str, Any],
                                trial: Optional[optuna.Trial] = None
                               ) -> ErrorResult:
        error_type = type(exception).__name__
        message = str(exception)
        tb_str = traceback.format_exc()
        
        err_result = ErrorResult(
            error_type=error_type,
            message=message,
            traceback_str=tb_str,
            context=context
        )
        if isinstance(exception, optuna.exceptions.TrialPruned):
            err_result.suggestions.append("L'essai a été élagué, probablement en raison de performances intermédiaires insuffisantes ou de paramètres invalides.")
        elif "indicateur" in message.lower() or "indicator" in message.lower():
            err_result.suggestions.append("Vérifiez la configuration des indicateurs, les noms de colonnes sources, ou les paramètres des indicateurs.")
        elif "données" in message.lower() or "data" in message.lower():
            err_result.suggestions.append("Vérifiez la qualité et le format des données d'entrée (OHLCV).")
        
        if trial:
            trial.set_user_attr("error_details", err_result.message[:500]) # Limiter la taille pour Optuna
            trial.set_user_attr("error_type", err_result.error_type)
            
        logger.error(f"[SimpleErrorHandler] Erreur gérée: {error_type} - {message}. Contexte: {context}")
        logger.debug(f"[SimpleErrorHandler] Traceback complet:\n{tb_str}")
        return err_result


class ObjectiveFunctionEvaluator:
    """
    Fonction objectif pour Optuna. Évalue un ensemble d'hyperparamètres
    en exécutant un backtest et retourne les métriques de performance.
    Refactorisé pour l'injection de dépendances et la mise en cache.
    """
    def __init__(self,
                 strategy_name_key: str, # Ex: "ma_crossover_strategy"
                 strategy_config_dict: Dict[str, Any], # Contenu de StrategyParamsConfig
                 df_enriched_slice: pd.DataFrame,
                 optuna_objectives_config: Dict[str, Any], # {'names': [], 'directions': []}
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any], # pair_config de l'exchange
                 app_config: 'AppConfig',
                 run_id: str, # ID du WFOManager/Task
                 # Dépendances injectées
                 strategy_loader: IStrategyLoader,
                 cache_manager: ICacheManager,
                 error_handler: IErrorHandler,
                 is_oos_eval: bool = False,
                 is_trial_number_for_oos_log: Optional[int] = None
                ):
        self.strategy_name_key = strategy_name_key
        self.strategy_config_dict = strategy_config_dict
        self.df_enriched_slice = self._prepare_df(df_enriched_slice.copy())
        self.optuna_objectives_config = optuna_objectives_config
        self.pair_symbol = pair_symbol.upper()
        self.symbol_info_data = symbol_info_data
        self.app_config = app_config
        self.run_id = run_id
        self.is_oos_eval = is_oos_eval
        self.is_trial_number_for_oos_log = is_trial_number_for_oos_log

        self.strategy_loader = strategy_loader
        self.cache_manager = cache_manager
        self.error_handler = error_handler

        self.log_prefix = self._build_log_prefix()

        self.strategy_script_reference, self.strategy_class_name = self._validate_strategy_config()
        self.params_space_details = self._load_params_space_details()

        self.last_backtest_results: Optional[Dict[str, Any]] = None
        self._last_indicator_params_signature: Optional[str] = None
        self._last_prepared_df_with_indicators: Optional[pd.DataFrame] = None

        logger.info(f"{self.log_prefix} Initialisé. Données shape: {self.df_enriched_slice.shape}")

    def _build_log_prefix(self) -> str:
        """Construit le préfixe de log pour l'évaluateur."""
        oos_tag = 'OOS' if self.is_oos_eval else 'IS_Opt'
        trial_tag = f'[OrigIS_Trial:{self.is_trial_number_for_oos_log}]' if self.is_oos_eval and self.is_trial_number_for_oos_log is not None else ''
        return (
            f"[{self.strategy_name_key}/{self.pair_symbol}]"
            f"[Run:{self.run_id}]"
            f"[{oos_tag}]"
            f"{trial_tag}"
            f"[ObjFuncEvalV2]"
        )

    def _validate_strategy_config(self) -> Tuple[str, str]:
        """Valide la configuration de la stratégie et retourne la référence du script et le nom de la classe."""
        script_ref = self.strategy_config_dict.get('script_reference', '')
        class_name = self.strategy_config_dict.get('class_name', '')
        if not script_ref or not class_name:
            msg = f"'script_reference' ou 'class_name' manquant dans strategy_config_dict pour {self.strategy_name_key}."
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        return script_ref, class_name

    def _load_params_space_details(self) -> Dict[str, ParamDetail]:
        """Charge les détails de l'espace des paramètres depuis la configuration."""
        params_space: Dict[str, ParamDetail] = {}
        raw_params_space = self.strategy_config_dict.get('params_space', {})
        if isinstance(raw_params_space, dict):
            for param_key, param_value_dict in raw_params_space.items():
                if isinstance(param_value_dict, dict):
                    try:
                        from src.config.definitions import ParamDetail # Import local pour éviter dépendance circulaire au niveau module
                        params_space[param_key] = ParamDetail(**param_value_dict)
                    except Exception as e_pd_create:
                        logger.error(f"{self.log_prefix} Erreur création ParamDetail pour '{param_key}': {e_pd_create}")
        return params_space

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare et standardise le DataFrame d'entrée."""
        if not isinstance(df.index, pd.DatetimeIndex):
            msg = "df_enriched_slice doit avoir un DatetimeIndex."
            logger.error(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif str(df.index.tz).upper() != 'UTC':
            df.index = df.index.tz_convert('UTC')
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
        return df

    def _generate_params_signature(self, params: Dict[str, Any], relevant_keys_prefix: Optional[List[str]] = None) -> str:
        """Génère une signature stable (hash) pour un sous-ensemble de paramètres."""
        if relevant_keys_prefix is None:
            relevant_keys_prefix = ["indicateur_frequence", "ma_", "atr_", "rsi_", "bbands_", "psar_"] # Exemples

        keys_for_signature = sorted([
            k for k in params
            if any(k.startswith(prefix) for prefix in relevant_keys_prefix)
        ])
        
        if not keys_for_signature: # Si aucun paramètre pertinent pour les indicateurs
            return "no_indicator_params"

        params_subset_for_signature = {k: params[k] for k in keys_for_signature}
        # Utiliser json.dumps avec sort_keys pour une représentation stable avant hachage
        params_str = json.dumps(params_subset_for_signature, sort_keys=True, default=str)
        return hashlib.sha256(params_str.encode('utf-8')).hexdigest()

    def _prepare_data_with_dynamic_indicators(self,
                                              strategy_instance: 'IStrategy',
                                              current_trial_params_for_indic_key: Dict[str, Any],
                                              trial_number_for_log: Optional[Union[int, str]] = None
                                             ) -> pd.DataFrame:
        """Prépare les données avec indicateurs, utilisant la mémoïsation si les paramètres d'indicateurs sont identiques."""
        eval_id_log = trial_number_for_log if trial_number_for_log is not None else "N/A"
        log_prefix_prep = f"{self.log_prefix}[EvalID:{eval_id_log}][PrepDataIndicators]"
        
        # Générer une signature pour les paramètres qui affectent les indicateurs
        current_indic_params_signature = self._generate_params_signature(current_trial_params_for_indic_key)

        if current_indic_params_signature == self._last_indicator_params_signature and \
           self._last_prepared_df_with_indicators is not None:
            logger.info(f"{log_prefix_prep} Paramètres d'indicateurs identiques au précédent. Utilisation du DataFrame préparé mémoïsé.")
            return self._last_prepared_df_with_indicators.copy() # Retourner une copie pour éviter modifications inattendues

        logger.info(f"{log_prefix_prep} Nouveaux paramètres d'indicateurs (ou premier run). Calcul des indicateurs...")
        
        # ... (logique existante de calculate_indicators_for_trial et strategy_instance._calculate_indicators) ...
        # Adapté pour utiliser strategy_instance et df_enriched_slice de self
        try:
            required_configs = strategy_instance.get_required_indicator_configs()
        except Exception as e_get_cfg:
            logger.error(f"{log_prefix_prep} Erreur get_required_indicator_configs(): {e_get_cfg}", exc_info=True)
            raise ValueError(f"Erreur get_required_indicator_configs: {e_get_cfg}") from e_get_cfg

        try:
            df_with_ta_indicators = calculate_indicators_for_trial(
                df_source_enriched=self.df_enriched_slice, # Utiliser le slice stocké
                required_indicator_configs=required_configs,
                cache_manager=self.cache_manager, # Passer le cache_manager au calculateur d'indicateurs
                log_prefix_context=f"[Trial:{eval_id_log}]"
            )
        except Exception as e_calc_indic_trial:
            logger.error(f"{log_prefix_prep} Erreur calculate_indicators_for_trial: {e_calc_indic_trial}", exc_info=True)
            raise ValueError(f"Erreur calculate_indicators_for_trial: {e_calc_indic_trial}") from e_calc_indic_trial

        if df_with_ta_indicators.empty:
            raise ValueError("calculate_indicators_for_trial a retourné un DataFrame vide.")
        
        try:
            df_final_for_simulation = strategy_instance._calculate_indicators(df_with_ta_indicators)
        except Exception as e_strat_calc:
            logger.error(f"{log_prefix_prep} Erreur strategy_instance._calculate_indicators(): {e_strat_calc}", exc_info=True)
            raise ValueError(f"Erreur strategy_instance._calculate_indicators: {e_strat_calc}") from e_strat_calc
        
        if df_final_for_simulation.empty:
            raise ValueError("strategy_instance._calculate_indicators() a retourné un DataFrame vide.")

        # Mettre à jour les informations de mémoïsation
        self._last_indicator_params_signature = current_indic_params_signature
        self._last_prepared_df_with_indicators = df_final_for_simulation.copy() # Stocker une copie

        logger.info(f"{log_prefix_prep} Préparation des données avec indicateurs terminée. Shape final: {df_final_for_simulation.shape}")
        return df_final_for_simulation


    def _perform_evaluation_for_trial(self, trial: optuna.Trial, trial_params: Dict[str, Any]) -> Union[float, Tuple[float, ...]]:
        """
        Contient la logique réelle d'évaluation d'un trial (chargement strat, prépa données, simu, métriques).
        Cette fonction est ce qui sera mis en cache par _evaluate_with_cache.
        """
        eval_id_log = self._get_eval_id_log(trial)
        current_log_prefix = f"{self.log_prefix}[Trial:{eval_id_log}][PerformEval]"

        strategy_instance = self._load_strategy(trial_params, current_log_prefix)
        df_for_simulation = self._prepare_simulation_data(strategy_instance, trial_params, eval_id_log, current_log_prefix)
        
        trades_log, equity_curve_df, oos_detailed_log = self._run_simulation(
            df_for_simulation, strategy_instance, trial_params, current_log_prefix
        )
        
        metrics = self._calculate_metrics(
            trades_log, equity_curve_df, df_for_simulation, eval_id_log, current_log_prefix
        )
        
        if self.last_backtest_results: # self.last_backtest_results est mis à jour dans _run_simulation
            self.last_backtest_results["metrics"] = metrics.copy()

        return self._determine_objective_values(metrics)

    def _get_eval_id_log(self, trial: optuna.Trial) -> str:
        """Détermine l'ID de log pour l'évaluation en cours."""
        if self.is_oos_eval:
            return str(self.is_trial_number_for_oos_log) if self.is_trial_number_for_oos_log is not None else "N/A_OOS"
        return str(trial.number) if hasattr(trial, 'number') else "N/A_IS"

    def _load_strategy(self, trial_params: Dict[str, Any], log_prefix: str) -> 'IStrategy':
        """Charge l'instance de la stratégie."""
        time_start = time.perf_counter()
        try:
            strategy_instance = self.strategy_loader.load_strategy(
                strategy_name_key=self.strategy_name_key,
                params_for_strategy=trial_params,
                strategy_script_ref=self.strategy_script_reference,
                strategy_class_name=self.strategy_class_name,
                pair_symbol=self.pair_symbol
            )
        except Exception as e:
            logger.error(f"{log_prefix} Erreur chargement stratégie: {e}", exc_info=True)
            raise
        logger.info(f"{log_prefix} Stratégie chargée en {time.perf_counter() - time_start:.4f}s.")
        return strategy_instance

    def _prepare_simulation_data(self, strategy_instance: 'IStrategy', trial_params: Dict[str, Any], eval_id_log: str, log_prefix: str) -> pd.DataFrame:
        """Prépare les données pour la simulation, incluant les indicateurs."""
        time_start = time.perf_counter()
        try:
            df_for_simulation = self._prepare_data_with_dynamic_indicators(
                strategy_instance,
                trial_params,
                trial_number_for_log=eval_id_log
            )
            if df_for_simulation.empty or df_for_simulation[['open', 'high', 'low', 'close']].isnull().all().all():
                raise ValueError("Préparation des données a résulté en un DataFrame vide ou inutilisable.")
        except Exception as e:
            logger.error(f"{log_prefix} Erreur préparation données: {e}", exc_info=True)
            raise
        logger.info(f"{log_prefix} Données préparées en {time.perf_counter() - time_start:.4f}s.")
        return df_for_simulation

    def _run_simulation(self, df_for_simulation: pd.DataFrame, strategy_instance: 'IStrategy', trial_params: Dict[str, Any], log_prefix: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame, List[Dict[str, Any]]]:
        """Exécute la simulation de backtesting."""
        sim_defaults: SimulationDefaults = self.app_config.global_config.simulation_defaults
        leverage_to_use = int(trial_params.get('margin_leverage', sim_defaults.margin_leverage))

        strategy_instance.set_trading_context(
            pair_config=self.symbol_info_data,
            is_futures=sim_defaults.is_futures_trading,
            leverage=leverage_to_use,
            initial_equity=sim_defaults.initial_capital,
            account_type=self.app_config.data_config.source_details.asset_type
        )

        simulator = BacktestRunner(
            df_ohlcv_with_indicators=df_for_simulation, strategy_instance=strategy_instance,
            initial_equity=sim_defaults.initial_capital, leverage=leverage_to_use,
            symbol=self.pair_symbol, pair_config=self.symbol_info_data,
            trading_fee_bps=sim_defaults.trading_fee_bps,
            slippage_config_dict=sim_defaults.slippage_config.__dict__,
            is_futures=sim_defaults.is_futures_trading, run_id=self.run_id,
            data_validator=self.app_config.data_validator_instance, # type: ignore
            cache_manager=self.cache_manager,
            event_dispatcher=self.app_config.event_dispatcher_instance, # type: ignore
            is_oos_simulation=self.is_oos_eval,
            verbosity=0 if not self.is_oos_eval else sim_defaults.backtest_verbosity
        )
        
        time_start = time.perf_counter()
        try:
            trades_log, equity_curve_df, _, oos_detailed_log = simulator.run_simulation()
            self.last_backtest_results = { # Mise à jour ici
                "params": trial_params, "trades": trades_log,
                "equity_curve_df": equity_curve_df,
                "oos_detailed_trades_log": oos_detailed_log,
                "metrics": {} 
            }
        except optuna.exceptions.TrialPruned as e_pruned:
            logger.info(f"{log_prefix} Trial élagué par BacktestRunner: {e_pruned}")
            raise
        except Exception as e:
            logger.error(f"{log_prefix} Erreur BacktestRunner: {e}", exc_info=True)
            raise
        logger.info(f"{log_prefix} Simulation terminée ({len(trades_log)} trades) en {time.perf_counter() - time_start:.4f}s.")
        return trades_log, equity_curve_df, oos_detailed_log

    def _calculate_metrics(self, trades_log: List[Dict[str, Any]], equity_curve_df: pd.DataFrame, df_for_simulation: pd.DataFrame, eval_id_log: str, log_prefix: str) -> Dict[str, Any]:
        """Calcule les métriques de performance."""
        sim_defaults = self.app_config.global_config.simulation_defaults
        equity_series = equity_curve_df['equity'] if not equity_curve_df.empty and 'equity' in equity_curve_df.columns else \
                        pd.Series([sim_defaults.initial_capital], index=[df_for_simulation.index.min() if not df_for_simulation.empty else pd.Timestamp.now(tz='UTC')])

        time_start = time.perf_counter()
        metrics = calculate_performance_metrics_from_inputs(
            trades_df=pd.DataFrame(trades_log), equity_curve_series=equity_series,
            initial_capital=sim_defaults.initial_capital,
            risk_free_rate_daily=(1 + sim_defaults.risk_free_rate)**(1/252) - 1,
            periods_per_year=252,
            cache_manager=self.cache_manager,
            base_cache_key_prefix=f"{self.log_prefix}_trial_{eval_id_log}_metrics"
        )
        metrics['Total Trades'] = metrics.get('Total Trades', len(trades_log))
        logger.info(f"{log_prefix} Métriques calculées en {time.perf_counter() - time_start:.4f}s.")
        return metrics

    def _determine_objective_values(self, metrics: Dict[str, Any]) -> Union[float, Tuple[float, ...]]:
        """Détermine les valeurs des objectifs pour Optuna à partir des métriques."""
        objective_values_list: List[float] = []
        obj_names = self.optuna_objectives_config.get('objectives_names', ["Total Net PnL USDC"])
        obj_dirs = self.optuna_objectives_config.get('objectives_directions', ["maximize"] * len(obj_names))

        for i, metric_name in enumerate(obj_names):
            value = metrics.get(metric_name)
            direction = obj_dirs[i].lower() if i < len(obj_dirs) and isinstance(obj_dirs[i], str) else "maximize"
            
            if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                objective_values_list.append(float(value))
            else:
                worst_val = -1e12 if direction == "maximize" else 1e12
                objective_values_list.append(worst_val)
        
        return tuple(objective_values_list) if len(objective_values_list) > 1 else objective_values_list[0]
        
    def _evaluate_with_cache(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        """Gère la mise en cache de l'évaluation complète du trial."""
        eval_id_log = (
            self.is_trial_number_for_oos_log
            if self.is_oos_eval
            else trial.number
            if hasattr(trial, 'number')
            else "N/A_OOS_FIXED"
        )
        current_log_prefix = f"{self.log_prefix}[Trial:{eval_id_log}][EvalWithCache]"

        # Utiliser les paramètres du trial pour la clé de cache
        # S'assurer que les params sont triés pour une clé stable
        params_for_key = trial.params.copy() if trial.params else {}
        if not params_for_key and self.is_oos_eval: # Pour OOS, trial.params peut être vide si on enqueued sans params
             # Dans ce cas, on ne peut pas vraiment utiliser trial.params pour la clé de cache.
             # La logique OOS devrait passer les params IS via user_attrs ou une autre méthode.
             # Pour l'instant, on suppose que si is_oos_eval, trial.params CONTIENT les params à évaluer.
             logger.error(f"{current_log_prefix} Évaluation OOS mais trial.params est vide. Impossible de générer une clé de cache ou d'évaluer.")
             raise ValueError("OOS evaluation requires params in trial.params for caching/evaluation.")


        # Ajouter d'autres éléments contextuels à la clé si nécessaire (ex: data slice fingerprint)
        # Pour l'instant, on se base principalement sur les paramètres du trial.
        # Un fingerprint du df_enriched_slice pourrait être ajouté pour plus de robustesse si les données changent.
        # df_fingerprint = hashlib.sha256(pd.util.hash_pandas_object(self.df_enriched_slice, index=True).values).hexdigest()
        
        key_material_dict = {
            "strategy_name": self.strategy_name_key,
            "pair_symbol": self.pair_symbol,
            "params": params_for_key,
            "is_oos": self.is_oos_eval,
            # "data_fingerprint": df_fingerprint # Optionnel
        }
        cache_key_str = json.dumps(key_material_dict, sort_keys=True, default=str)
        cache_key_hash = hashlib.sha256(cache_key_str.encode('utf-8')).hexdigest()
        logger.debug(f"{current_log_prefix} Clé de cache pour l'évaluation complète : {cache_key_hash}")

        # La fonction à exécuter si le cache est manquant ou expiré
        # Elle prend `trial` et `params_for_key`
        compute_evaluation_func = functools.partial(self._perform_evaluation_for_trial, trial=trial, trial_params=params_for_key)

        try:
            # Cache pour 1 heure (3600 secondes)
            cached_objectives = self.cache_manager.get_or_compute(
                key=cache_key_hash,
                compute_func=compute_evaluation_func, # type: ignore
                ttl=3600
            )
            logger.info(f"{current_log_prefix} Évaluation obtenue (cache ou calcul).")
            return cast(Union[float, Tuple[float, ...]], cached_objectives)

        except optuna.exceptions.TrialPruned as e_pruned_in_compute:
            logger.info(f"{current_log_prefix} Trial élagué pendant _perform_evaluation_for_trial (attrapé par _evaluate_with_cache): {e_pruned_in_compute}")
            raise # Renvoyer pour qu'Optuna le gère
        except Exception as e_eval:
            logger.error(f"{current_log_prefix} Erreur non gérée durant _perform_evaluation_for_trial (attrapé par _evaluate_with_cache): {e_eval}", exc_info=True)
            # Gérer l'erreur et retourner des valeurs pénalisantes
            # L'erreur est déjà loguée par _perform_evaluation_for_trial si elle y est attrapée.
            # Ici, c'est une double sécurité ou si _perform_evaluation_for_trial la renvoie.
            error_context = {"trial_params": params_for_key, "cache_key": cache_key_hash, "stage": "perform_evaluation"}
            self.error_handler.handle_evaluation_error(e_eval, error_context, trial)
            return self._get_worst_objective_values(f"Erreur critique durant _perform_evaluation: {e_eval}")


    def __call__(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        """Point d'entrée pour Optuna pour évaluer un trial."""
        time_start_call = time.perf_counter()
        eval_id_log = self._get_eval_id_log(trial)
        current_log_prefix = f"{self.log_prefix}[Trial:{eval_id_log}][__call__]"
        logger.info(f"{current_log_prefix} Démarrage de l'évaluation.")

        params_for_this_trial = self._get_params_for_trial(trial, current_log_prefix)
        if params_for_this_trial is None: # Indique une erreur de configuration ou un élagage précoce
            return self._get_worst_objective_values("Paramètres non déterminés pour le trial.")

        try:
            # `trial.params` est rempli par `_suggest_params_for_optuna_trial` (pour IS)
            # ou par `study.enqueue_trial` (pour OOS).
            # `_evaluate_with_cache` utilise `trial.params` pour la clé de cache.
            objective_values = self._evaluate_with_cache(trial)
            
            duration = time.perf_counter() - time_start_call
            logger.info(f"{current_log_prefix} Évaluation terminée en {duration:.4f}s. Objectifs: {objective_values}")
            return objective_values
        except optuna.exceptions.TrialPruned as e_pruned:
            logger.info(f"{current_log_prefix} Trial élagué: {e_pruned}")
            raise
        except Exception as e_main:
            logger.critical(f"{current_log_prefix} Erreur critique: {e_main}", exc_info=True)
            # Utiliser trial.params si disponible, sinon params_for_this_trial (qui pourrait être None)
            error_params = trial.params.copy() if trial.params else params_for_this_trial if params_for_this_trial is not None else {}
            self.error_handler.handle_evaluation_error(e_main, {"trial_params": error_params, "stage": "__call__"}, trial)
            return self._get_worst_objective_values(f"Erreur critique __call__: {e_main}")

    def _get_params_for_trial(self, trial: optuna.Trial, log_prefix: str) -> Optional[Dict[str, Any]]:
        """
        Détermine les paramètres pour le trial.
        Suggère si IS et non déjà présents, sinon utilise trial.params (pour OOS ou IS enqueued).
        Retourne None si une erreur survient ou si élagage nécessaire.
        """
        if not self.is_oos_eval and not trial.params: # Cas standard IS: Optuna suggère
            try:
                # _suggest_params_for_optuna_trial remplit trial.params et les retourne
                return self._suggest_params_for_optuna_trial(trial)
            except optuna.exceptions.TrialPruned:
                raise # Laisser Optuna gérer l'élagage
            except Exception as e_suggest:
                logger.error(f"{log_prefix} Erreur _suggest_params_for_optuna_trial: {e_suggest}", exc_info=True)
                self.error_handler.handle_evaluation_error(e_suggest, {"stage": "suggest_params_is", "params_space": self.params_space_details}, trial)
                # Pas besoin de retourner _get_worst_objective_values ici, car Optuna attrape l'exception
                # si on la relance, ou on peut retourner None pour indiquer un échec.
                # Mais comme on a déjà géré l'erreur et que Optuna va élaguer, on peut juste relancer.
                raise # Ou retourner None et laisser __call__ gérer la valeur de retour.
                # Pour la robustesse, si on ne relance pas, __call__ doit gérer le None.

        # Cas OOS ou IS avec params déjà fixés (ex: study.enqueue_trial)
        # trial.params devrait déjà être rempli.
        if not trial.params and self.is_oos_eval: # Erreur: OOS mais pas de params
            logger.error(f"{log_prefix} Évaluation OOS mais trial.params est vide. Impossible d'évaluer.")
            self.error_handler.handle_evaluation_error(ValueError("OOS eval: trial.params vide"), {"stage": "oos_params_check"}, trial)
            # Pourrait lever TrialPruned ici ou retourner None pour que __call__ gère.
            # Lever TrialPruned est plus direct pour Optuna.
            raise optuna.exceptions.TrialPruned("OOS eval: trial.params vide")
        
        return trial.params.copy() if trial.params else {}


    def _suggest_params_for_optuna_trial(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggère les paramètres pour un trial Optuna (utilisé en mode IS)."""
        params_for_trial: Dict[str, Any] = {}
        log_prefix_suggest = f"{self.log_prefix}[Trial:{trial.number}][SuggestParamsOptuna]"

        if not self.params_space_details:
            self._handle_empty_params_space(trial, log_prefix_suggest) # Lève TrialPruned

        for param_name, p_detail in self.params_space_details.items():
            try:
                if p_detail.type == 'int':
                    params_for_trial[param_name] = trial.suggest_int(
                        param_name, 
                        int(p_detail.low) if p_detail.low is not None else 0,
                        int(p_detail.high) if p_detail.high is not None else (int(p_detail.low) if p_detail.low is not None else 0) + 1, # type: ignore
                        step=int(p_detail.step or 1)
                    )
                elif p_detail.type == 'float':
                    params_for_trial[param_name] = trial.suggest_float(
                        param_name,
                        float(p_detail.low) if p_detail.low is not None else 0.0,
                        float(p_detail.high) if p_detail.high is not None else (float(p_detail.low) if p_detail.low is not None else 0.0) + 1.0, # type: ignore
                        step=float(p_detail.step) if p_detail.step is not None else None,
                        log=p_detail.log_scale
                    )
                elif p_detail.type == 'categorical' and p_detail.choices:
                    params_for_trial[param_name] = trial.suggest_categorical(param_name, p_detail.choices)
                else:
                    # Gérer autres types ou fallback si nécessaire, ex: utiliser p_detail.default
                    if p_detail.default is not None:
                        params_for_trial[param_name] = p_detail.default # Suggérer comme constant si type inconnu mais défaut existe
                        logger.warning(f"{log_prefix_suggest} Type de paramètre '{p_detail.type}' pour '{param_name}' non géré. "
                                       f"Utilisation de la valeur par défaut: {p_detail.default}")
                    else:
                        logger.error(f"{log_prefix_suggest} Type de paramètre '{p_detail.type}' pour '{param_name}' non géré et pas de défaut.")
                        raise optuna.exceptions.TrialPruned(f"Type param non géré '{p_detail.type}' pour '{param_name}' sans défaut.")
            except Exception as e_sug_item:
                logger.error(f"{log_prefix_suggest} Erreur suggestion paramètre '{param_name}': {e_sug_item}", exc_info=True)
                if p_detail.default is not None:
                    params_for_trial[param_name] = p_detail.default
                    trial.set_user_attr(f"warning_suggest_{param_name}", f"Used ParamDetail default due to error: {e_sug_item}")
                else:
                    raise optuna.exceptions.TrialPruned(f"Échec suggestion pour {param_name}") from e_sug_item
        
        logger.debug(f"{log_prefix_suggest} Paramètres suggérés pour Optuna trial: {params_for_trial}")
        # Optuna stocke ces paramètres dans trial.params automatiquement après les appels à suggest_*.
        return params_for_trial

    def _handle_empty_params_space(self, trial: optuna.Trial, log_prefix_suggest: str):
        """Gère le cas où params_space_details est vide."""
        default_params = self.strategy_config_dict.get('default_params') or {}
        if not isinstance(default_params, dict): # Vérification supplémentaire
            trial.set_user_attr("failure_reason", "params_space vide et default_params n'est pas un dict.")
            raise optuna.exceptions.TrialPruned("params_space vide et default_params n'est pas un dict.")

        logger.warning(f"{log_prefix_suggest} params_space vide. Tentative d'utilisation de default_params: {default_params}")
        if not default_params: # Si default_params est aussi vide
            trial.set_user_attr("failure_reason", "params_space vide et pas de default_params.")
            raise optuna.exceptions.TrialPruned("params_space vide et pas de default_params.")

        # Si default_params existe, on pourrait essayer de les utiliser, mais Optuna attend des appels `trial.suggest_*`.
        # Il est plus sûr d'élaguer si l'espace n'est pas défini pour l'optimisation.
        trial.set_user_attr("failure_reason", "params_space vide. L'optimisation nécessite un params_space défini.")
        raise optuna.exceptions.TrialPruned("params_space vide. L'optimisation nécessite un params_space défini.")

    def _get_worst_objective_values(self, reason_for_worst: str) -> Union[float, Tuple[float, ...]]:
        """Retourne les pires valeurs possibles pour les objectifs configurés."""
        logger.warning(f"{self.log_prefix} Retour des pires valeurs d'objectif. Raison: {reason_for_worst}")
        
        obj_dirs: List[str] = self.optuna_objectives_config.get('objectives_directions', ['maximize'])
        num_objectives = len(self.optuna_objectives_config.get('objectives_names', ['']))
        
        if not obj_dirs or len(obj_dirs) != num_objectives:
            obj_dirs = ['maximize'] * num_objectives
            logger.warning(f"{self.log_prefix} Directions d'objectifs incohérentes ou manquantes. Utilisation de 'maximize' par défaut pour {num_objectives} objectifs.")

        worst_values_list = [
            -1e12 if d.lower() == "maximize" else 1e12 for d in obj_dirs
        ]
        
        # Cas spécial pour "Total Trades" si c'est un objectif de minimisation (peu probable mais géré)
        obj_names_list: List[str] = self.optuna_objectives_config.get('objectives_names', [])
        for i, obj_name in enumerate(obj_names_list):
            if "Total Trades" in obj_name and obj_dirs[i].lower() == "minimize":
                worst_values_list[i] = 1e9 # Un grand nombre de trades est mauvais si on minimise
            elif "Total Trades" in obj_name and obj_dirs[i].lower() == "maximize":
                 worst_values_list[i] = 0 # 0 trades est le pire si on maximise les trades (peu probable aussi)
            elif ("PnL" in obj_name or "Profit" in obj_name or "Equity" in obj_name) and obj_dirs[i].lower() == "maximize":
                 worst_values_list[i] = -1e12 # Très mauvais PnL
            elif ("Drawdown" in obj_name) and obj_dirs[i].lower() == "minimize": # Max Drawdown Pct est souvent minimisé
                 worst_values_list[i] = 100.0 # 100% drawdown est très mauvais
            elif ("Sharpe" in obj_name or "Sortino" in obj_name or "Calmar" in obj_name) and obj_dirs[i].lower() == "maximize":
                 worst_values_list[i] = -10.0 # Très mauvais ratio
                 
        return tuple(worst_values_list) if len(worst_values_list) > 1 else worst_values_list[0]

