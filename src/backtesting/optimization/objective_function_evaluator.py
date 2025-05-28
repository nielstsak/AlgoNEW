# src/backtesting/optimization/objective_function_evaluator.py
"""
Ce module définit ObjectiveFunctionEvaluator, la fonction objectif pour Optuna.
Elle évalue un ensemble d'hyperparamètres (un "trial") en exécutant un backtest
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

from typing import Any, Dict, Optional, Tuple, List, Type, Union, TYPE_CHECKING, Callable, cast
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
        if parts and parts[0] != 'src': # Assurer que ça commence par src si c'est un chemin relatif depuis la racine
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
        self.strategy_config_dict = strategy_config_dict # C'est StrategyParamsConfig as dict
        self.df_enriched_slice = df_enriched_slice.copy() # Travailler sur une copie
        self.optuna_objectives_config = optuna_objectives_config
        self.pair_symbol = pair_symbol.upper()
        self.symbol_info_data = symbol_info_data
        self.app_config = app_config
        self.run_id = run_id
        self.is_oos_eval = is_oos_eval
        self.is_trial_number_for_oos_log = is_trial_number_for_oos_log

        # Injection des dépendances
        self.strategy_loader = strategy_loader
        self.cache_manager = cache_manager
        self.error_handler = error_handler

        self.log_prefix = (
            f"[{self.strategy_name_key}/{self.pair_symbol}]"
            f"[Run:{self.run_id}]"
            f"[{'OOS' if self.is_oos_eval else 'IS_Opt'}]"
            f"{f'[OrigIS_Trial:{self.is_trial_number_for_oos_log}]' if self.is_oos_eval and self.is_trial_number_for_oos_log is not None else ''}"
            f"[ObjFuncEvalV2]"
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
                        # Assumer que ParamDetail est importable depuis definitions
                        from src.config.definitions import ParamDetail
                        self.params_space_details[param_key] = ParamDetail(**param_value_dict)
                    except Exception as e_pd_create:
                         logger.error(f"{self.log_prefix} Erreur création ParamDetail pour '{param_key}': {e_pd_create}")
        
        # Standardisation de l'index du DataFrame
        if not isinstance(self.df_enriched_slice.index, pd.DatetimeIndex):
            msg = "df_enriched_slice doit avoir un DatetimeIndex."
            logger.error(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        if self.df_enriched_slice.index.tz is None:
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_localize('UTC') # type: ignore
        elif str(self.df_enriched_slice.index.tz).upper() != 'UTC':
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_convert('UTC') # type: ignore
        if not self.df_enriched_slice.index.is_monotonic_increasing: self.df_enriched_slice.sort_index(inplace=True)
        if self.df_enriched_slice.index.duplicated().any(): self.df_enriched_slice = self.df_enriched_slice[~self.df_enriched_slice.index.duplicated(keep='first')]

        self.last_backtest_results: Optional[Dict[str, Any]] = None # Pour stocker les résultats du dernier backtest (utile pour OOS)
        
        # Pour la mémoïsation des données préparées
        self._last_indicator_params_signature: Optional[str] = None
        self._last_prepared_df_with_indicators: Optional[pd.DataFrame] = None

        logger.info(f"{self.log_prefix} Initialisé. Données shape: {self.df_enriched_slice.shape}")


    def _generate_params_signature(self, params: Dict[str, Any], relevant_keys_prefix: Optional[List[str]] = None) -> str:
        """Génère une signature stable (hash) pour un sous-ensemble de paramètres."""
        if relevant_keys_prefix is None:
            relevant_keys_prefix = ["indicateur_frequence", "ma_", "atr_", "rsi_", "bbands_", "psar_"] # Exemples

        keys_for_signature = sorted([
            k for k in params.keys()
            if any(k.startswith(prefix) for prefix in relevant_keys_prefix)
        ])
        
        if not keys_for_signature: # Si aucun paramètre pertinent pour les indicateurs
            return "no_indicator_params"

        params_subset_for_signature = {k: params[k] for k in keys_for_signature}
        # Utiliser json.dumps avec sort_keys pour une représentation stable avant hashage
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
            raise ValueError(f"Erreur get_required_indicator_configs: {e_get_cfg}")

        try:
            df_with_ta_indicators = calculate_indicators_for_trial(
                df_source_enriched=self.df_enriched_slice, # Utiliser le slice stocké
                required_indicator_configs=required_configs,
                cache_manager=self.cache_manager, # Passer le cache_manager au calculateur d'indicateurs
                log_prefix_context=f"[Trial:{eval_id_log}]"
            )
        except Exception as e_calc_indic_trial:
            logger.error(f"{log_prefix_prep} Erreur calculate_indicators_for_trial: {e_calc_indic_trial}", exc_info=True)
            raise ValueError(f"Erreur calculate_indicators_for_trial: {e_calc_indic_trial}")

        if df_with_ta_indicators.empty:
            raise ValueError("calculate_indicators_for_trial a retourné un DataFrame vide.")
        
        try:
            df_final_for_simulation = strategy_instance._calculate_indicators(df_with_ta_indicators)
        except Exception as e_strat_calc:
            logger.error(f"{log_prefix_prep} Erreur strategy_instance._calculate_indicators(): {e_strat_calc}", exc_info=True)
            raise ValueError(f"Erreur strategy_instance._calculate_indicators: {e_strat_calc}")
        
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
        eval_id_log = trial.number if hasattr(trial, 'number') and not self.is_oos_eval else \
                      (self.is_trial_number_for_oos_log if self.is_oos_eval else "N/A")
        current_log_prefix = f"{self.log_prefix}[Trial:{eval_id_log}][PerformEval]"
        
        # 1. Charger la Stratégie (chronométré)
        time_start_load_strat = time.perf_counter()
        try:
            strategy_instance = self.strategy_loader.load_strategy(
                strategy_name_key=self.strategy_name_key,
                params_for_strategy=trial_params,
                strategy_script_ref=self.strategy_script_reference,
                strategy_class_name=self.strategy_class_name,
                pair_symbol=self.pair_symbol # Passer la paire
            )
        except Exception as e_load_s:
            logger.error(f"{current_log_prefix} Erreur chargement stratégie: {e_load_s}", exc_info=True)
            raise # Renvoyer pour être attrapé par _evaluate_with_cache ou __call__
        time_end_load_strat = time.perf_counter()
        logger.info(f"{current_log_prefix} Stratégie chargée en {time_end_load_strat - time_start_load_strat:.4f}s.")

        # 2. Préparer les Données avec Indicateurs (chronométré, utilise la mémoïsation interne)
        time_start_prep_data = time.perf_counter()
        try:
            df_for_simulation = self._prepare_data_with_dynamic_indicators(
                strategy_instance,
                trial_params, # Passer les params du trial pour la clé de mémoïsation des indicateurs
                trial_number_for_log=eval_id_log
            )
            if df_for_simulation.empty or df_for_simulation[['open', 'high', 'low', 'close']].isnull().all().all():
                raise ValueError("Préparation des données a résulté en un DataFrame vide ou inutilisable.")
        except Exception as e_prep_d:
            logger.error(f"{current_log_prefix} Erreur préparation données: {e_prep_d}", exc_info=True)
            raise
        time_end_prep_data = time.perf_counter()
        logger.info(f"{current_log_prefix} Données préparées en {time_end_prep_data - time_start_prep_data:.4f}s.")

        # 3. Exécuter la Simulation (chronométré)
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
            data_validator=self.app_config.data_validator_instance, # type: ignore # Passer l'instance
            cache_manager=self.cache_manager, # Passer le cache manager
            event_dispatcher=self.app_config.event_dispatcher_instance, # type: ignore # Passer l'instance
            is_oos_simulation=self.is_oos_eval,
            verbosity=0 if not self.is_oos_eval else sim_defaults.backtest_verbosity
        )
        
        trades_log: List[Dict[str, Any]]
        equity_curve_df: pd.DataFrame
        oos_detailed_log: List[Dict[str, Any]] # Capturer ce log

        time_start_sim = time.perf_counter()
        try:
            trades_log, equity_curve_df, _, oos_detailed_log = simulator.run_simulation()
            # Stocker les résultats bruts pour un accès potentiel (ex: OOSValidator)
            self.last_backtest_results = {
                "params": trial_params, "trades": trades_log,
                "equity_curve_df": equity_curve_df,
                "oos_detailed_trades_log": oos_detailed_log, # Important pour OOS
                "metrics": {} # Sera rempli ensuite
            }
        except optuna.exceptions.TrialPruned as e_pruned_sim_intern: # Si le simulateur élague
            logger.info(f"{current_log_prefix} Trial élagué par BacktestRunner: {e_pruned_sim_intern}")
            raise
        except Exception as e_sim_run:
            logger.error(f"{current_log_prefix} Erreur BacktestRunner: {e_sim_run}", exc_info=True)
            raise
        time_end_sim = time.perf_counter()
        logger.info(f"{current_log_prefix} Simulation terminée ({len(trades_log)} trades) en {time_end_sim - time_start_sim:.4f}s.")

        # 4. Calculer les Métriques (chronométré)
        equity_series_for_metrics = pd.Series(dtype=float)
        if not equity_curve_df.empty and 'equity' in equity_curve_df.columns:
            # L'index de equity_curve_df est déjà un DatetimeIndex UTC
            equity_series_for_metrics = equity_curve_df['equity']
        
        if equity_series_for_metrics.empty:
            start_ts_data = df_for_simulation.index.min() if not df_for_simulation.empty else pd.Timestamp.now(tz='UTC')
            equity_series_for_metrics = pd.Series([sim_defaults.initial_capital], index=[start_ts_data])

        time_start_metrics = time.perf_counter()
        metrics = calculate_performance_metrics_from_inputs(
            trades_df=pd.DataFrame(trades_log), equity_curve_series=equity_series_for_metrics,
            initial_capital=sim_defaults.initial_capital,
            risk_free_rate_daily=(1 + sim_defaults.risk_free_rate)**(1/252) - 1,
            periods_per_year=252,
            cache_manager=self.cache_manager, # Passer le cache aux métriques
            base_cache_key_prefix=f"{self.log_prefix}_trial_{eval_id_log}_metrics"
        )
        metrics['Total Trades'] = metrics.get('Total Trades', len(trades_log))
        if self.last_backtest_results: self.last_backtest_results["metrics"] = metrics.copy()
        time_end_metrics = time.perf_counter()
        logger.info(f"{current_log_prefix} Métriques calculées en {time_end_metrics - time_start_metrics:.4f}s.")

        # 5. Retourner les valeurs des objectifs pour Optuna
        objective_values_list: List[float] = []
        obj_names: List[str] = self.optuna_objectives_config.get('objectives_names', ["Total Net PnL USDC"])
        obj_dirs: List[str] = self.optuna_objectives_config.get('objectives_directions', ["maximize"] * len(obj_names))

        for i, metric_name in enumerate(obj_names):
            value = metrics.get(metric_name)
            direction = obj_dirs[i].lower() if i < len(obj_dirs) and isinstance(obj_dirs[i], str) else "maximize"
            
            if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                objective_values_list.append(float(value))
            else: # Valeur non finie ou None
                worst_val = -1e12 if direction == "maximize" else 1e12
                objective_values_list.append(worst_val)
        
        return tuple(objective_values_list) if len(objective_values_list) > 1 else objective_values_list[0]

    def _evaluate_with_cache(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        """Gère la mise en cache de l'évaluation complète du trial."""
        eval_id_log = trial.number if hasattr(trial, 'number') and not self.is_oos_eval else \
                      (self.is_trial_number_for_oos_log if self.is_oos_eval else "N/A_OOS_FIXED")
        current_log_prefix = f"{self.log_prefix}[Trial:{eval_id_log}][EvalWithCache]"

        # Utiliser les paramètres du trial pour la clé de cache
        # S'assurer que les params sont triés pour une clé stable
        params_for_key = trial.params.copy() if trial.params else {}
        if not params_for_key and self.is_oos_eval: # Pour OOS, trial.params peut être vide si on enqueued sans params
             # Dans ce cas, on ne peut pas vraiment utiliser trial.params pour la clé de cache.
             # La logique OOS devrait passer les params IS via user_attrs ou une autre méthode.
             # Pour l'instant, on assume que si is_oos_eval, trial.params CONTIENT les params à évaluer.
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
        eval_id_log = trial.number if hasattr(trial, 'number') and not self.is_oos_eval else \
                      (self.is_trial_number_for_oos_log if self.is_oos_eval else "N/A_OOS_FIXED")
        current_log_prefix = f"{self.log_prefix}[Trial:{eval_id_log}][__call__]"
        logger.info(f"{current_log_prefix} Démarrage de l'évaluation (via __call__).")
        
        # Pour l'évaluation OOS, les paramètres sont fixés et passés via trial.params
        # (par ex. en utilisant study.enqueue_trial(fixed_params) dans OOSValidator)
        # Pour l'évaluation IS, Optuna suggère les paramètres.
        
        # La logique de suggestion de paramètres pour IS est maintenant dans _perform_evaluation_for_trial,
        # car elle est spécifique au trial et ne doit pas être cachée au niveau de _evaluate_with_cache
        # si la clé de cache est basée uniquement sur les params.
        # Cependant, si _evaluate_with_cache met en cache le résultat de _perform_evaluation_for_trial,
        # alors les params suggérés par Optuna seront utilisés pour la clé de cache.
        
        # Si c'est une évaluation OOS, les params sont déjà dans trial.params (via enqueue_trial)
        # Si c'est IS, _suggest_params sera appelé dans _perform_evaluation_for_trial
        # si `trial.params` est vide.
        # Pour que le cache fonctionne correctement, `trial.params` doit être rempli *avant*
        # la génération de la clé de cache dans `_evaluate_with_cache`.
        # Donc, si IS et `trial.params` est vide, on doit le remplir ici.
        
        params_for_this_trial: Dict[str, Any]
        if not self.is_oos_eval and not trial.params: # IS et pas de params déjà fixés (cas normal)
            try:
                params_for_this_trial = self._suggest_params_for_optuna_trial(trial) # Nouvelle méthode pour juste suggérer
                # Mettre à jour trial.params pour que _evaluate_with_cache puisse l'utiliser pour la clé
                # Optuna ne permet pas de modifier trial.params directement après suggestion.
                # On va passer ces params à _evaluate_with_cache / _perform_evaluation_for_trial
                # et la clé de cache sera générée à partir de ces params.
            except optuna.exceptions.TrialPruned:
                raise # Laisser Optuna gérer l'élagage
            except Exception as e_sug_call:
                logger.error(f"{current_log_prefix} Erreur _suggest_params_for_optuna_trial: {e_sug_call}", exc_info=True)
                error_context = {"stage": "suggest_params_is", "params_space": self.params_space_details}
                self.error_handler.handle_evaluation_error(e_sug_call, error_context, trial)
                return self._get_worst_objective_values(f"Erreur suggestion params IS: {e_sug_call}")
        else: # OOS ou IS avec params déjà fixés (ex: enqueue_trial)
            params_for_this_trial = trial.params.copy() if trial.params else {}
            if not params_for_this_trial and self.is_oos_eval:
                 logger.error(f"{current_log_prefix} Évaluation OOS mais trial.params est vide. Impossible d'évaluer.")
                 self.error_handler.handle_evaluation_error(ValueError("OOS eval: trial.params vide"), {"stage": "oos_params_check"}, trial)
                 return self._get_worst_objective_values("OOS eval: trial.params vide")
        
        # La clé de cache dans _evaluate_with_cache utilisera ces `params_for_this_trial`
        # en les récupérant depuis `trial` (si on les y stocke) ou en les passant.
        # Pour l'instant, _perform_evaluation_for_trial reçoit `trial_params` qui sont ceux-ci.
        
        try:
            # _evaluate_with_cache va maintenant appeler _perform_evaluation_for_trial
            # qui utilisera les `params_for_this_trial`
            # Nous devons nous assurer que `trial.params` est rempli si c'est un nouveau trial IS
            # ou que `_perform_evaluation_for_trial` reçoit les bons params.
            # Optuna gère `trial.params` après `trial.suggest_...`.
            # Si on appelle `_suggest_params_for_optuna_trial`, `trial.params` est rempli.
            
            # Modifier _evaluate_with_cache pour qu'il prenne les params en argument
            # ou s'assurer que trial.params est bien rempli avant de générer la clé.
            # La solution la plus simple est que _evaluate_with_cache lise trial.params.
            
            # Si c'est un nouveau trial IS, _suggest_params_for_optuna_trial a rempli trial.params.
            # Si c'est OOS, trial.params a été rempli par enqueue_trial.
            
            objective_values = self._evaluate_with_cache(trial) # _evaluate_with_cache lira trial.params
            
            time_end_call = time.perf_counter()
            logger.info(f"{current_log_prefix} Évaluation complète (via __call__) terminée en {time_end_call - time_start_call:.4f}s. "
                        f"Objectifs: {objective_values}")
            return objective_values
        except optuna.exceptions.TrialPruned as e_pruned_call:
            logger.info(f"{current_log_prefix} Trial élagué (attrapé par __call__): {e_pruned_call}")
            raise # Laisser Optuna gérer
        except Exception as e_call_main:
            logger.critical(f"{current_log_prefix} Erreur critique non gérée dans __call__: {e_call_main}", exc_info=True)
            error_context = {"trial_params": trial.params if trial.params else params_for_this_trial, "stage": "__call__"}
            self.error_handler.handle_evaluation_error(e_call_main, error_context, trial)
            return self._get_worst_objective_values(f"Erreur critique __call__: {e_call_main}")


    def _suggest_params_for_optuna_trial(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggère les paramètres pour un trial Optuna (utilisé en mode IS)."""
        # Cette méthode est séparée pour que __call__ puisse remplir trial.params
        # avant d'appeler _evaluate_with_cache qui génère la clé de cache.
        params_for_trial: Dict[str, Any] = {}
        log_prefix_suggest = f"{self.log_prefix}[Trial:{trial.number}][SuggestParamsOptuna]"

        if not self.params_space_details: # Devrait être attrapé par __init__ si is_oos_eval=False
            default_params = self.strategy_config_dict.get('default_params', {})
            if default_params and isinstance(default_params, dict):
                logger.warning(f"{log_prefix_suggest} params_space vide. Utilisation de default_params : {default_params}")
                # Optuna ne permet pas de retourner directement sans suggestion, donc on doit suggérer les défauts
                for p_name, p_val in default_params.items():
                    # Ceci est un hack, car on ne connaît pas le type de suggestion à faire.
                    # Idéalement, params_space ne devrait jamais être vide pour l'optimisation.
                    # On va essayer de suggérer comme constant si possible.
                    # Pour l'instant, on retourne juste les défauts, et on espère que Optuna
                    # ne les utilise pas pour générer une clé de cache si on ne les suggère pas.
                    # Mieux: élaguer si params_space est vide.
                    trial.set_user_attr("failure_reason", "params_space vide, utilisation des défauts, mais élagueage.")
                    raise optuna.exceptions.TrialPruned("params_space vide, utilisation des défauts, mais élagueage.")
                # return default_params.copy() # Ne devrait pas être atteint
            trial.set_user_attr("failure_reason", "params_space vide et pas de default_params.")
            raise optuna.exceptions.TrialPruned("params_space vide et pas de default_params.")

        for param_name, p_detail in self.params_space_details.items():
            try:
                if p_detail.type == 'int':
                    low = int(p_detail.low) if p_detail.low is not None else 0
                    high = int(p_detail.high) if p_detail.high is not None else low + 1
                    if low > high: high = low
                    step = int(p_detail.step or 1)
                    params_for_trial[param_name] = trial.suggest_int(param_name, low, high, step=step)
                elif p_detail.type == 'float':
                    low_f = float(p_detail.low) if p_detail.low is not None else 0.0
                    high_f = float(p_detail.high) if p_detail.high is not None else low_f + 1.0
                    if low_f > high_f: high_f = low_f
                    step_f = float(p_detail.step) if p_detail.step is not None else None
                    params_for_trial[param_name] = trial.suggest_float(param_name, low_f, high_f, step=step_f, log=p_detail.log_scale)
                elif p_detail.type == 'categorical' and p_detail.choices:
                    params_for_trial[param_name] = trial.suggest_categorical(param_name, p_detail.choices)
                # ... (gestion des autres types ou fallback comme dans _suggest_params original)
            except Exception as e_sug_item:
                logger.error(f"{log_prefix_suggest} Erreur suggestion paramètre '{param_name}': {e_sug_item}", exc_info=True)
                # Fallback sur la valeur par défaut du ParamDetail si elle existe
                if p_detail.default is not None:
                    params_for_trial[param_name] = p_detail.default
                    trial.set_user_attr(f"warning_suggest_{param_name}", f"Used ParamDetail default due to error: {e_sug_item}")
                else:
                    trial.set_user_attr("failure_reason", f"Échec suggestion pour {param_name}")
                    raise optuna.exceptions.TrialPruned(f"Échec suggestion pour {param_name}")
        
        logger.debug(f"{log_prefix_suggest} Paramètres suggérés pour Optuna trial : {params_for_trial}")
        # Optuna stocke ces paramètres dans trial.params automatiquement.
        return params_for_trial # Retourner pour référence si besoin, mais trial.params est la source de vérité.


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

