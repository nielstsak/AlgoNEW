# src/config/definitions.py
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union, Tuple, Callable # Added Callable
from pathlib import Path # Added Path
import pandas as pd # Added for pd.to_datetime
from decimal import Decimal # Added for Decimal
import enum # Added for Enum
import logging
# --- Imports des interfaces pour AppConfig (pour type hinting) ---
# Ces imports supposent que les interfaces sont définies dans src.core.interfaces
# et que les implémentations de base/simples sont disponibles (ex: dans utils ou core)
try:
    from src.core.interfaces import (
        IDataValidator, ICacheManager, IEventDispatcher
    )
    # Pour IStrategyLoader et IErrorHandler, si elles sont dans interfaces.py
    # from src.core.interfaces import IStrategyLoader, IErrorHandler
    # Si les implémentations simples sont utilisées directement:
    from src.backtesting.optimization.objective_function_evaluator import (
        IStrategyLoader, SimpleStrategyLoader, # IStrategyLoader est un Protocol ici
        IErrorHandler, SimpleErrorHandler     # IErrorHandler est un Protocol ici
    )
    # StrategyFactory est une classe concrète
    from src.strategies.strategy_factory import StrategyFactory
except ImportError:
    # Fallbacks si les imports échouent (pour permettre au module de se charger)
    class IDataValidator: pass # type: ignore
    class ICacheManager: pass # type: ignore
    class IEventDispatcher: pass # type: ignore
    class IStrategyLoader: pass # type: ignore
    class IErrorHandler: pass # type: ignore
    class StrategyFactory: pass # type: ignore
    class SimpleStrategyLoader(IStrategyLoader): pass # type: ignore
    class SimpleErrorHandler(IErrorHandler): pass # type: ignore
    logging.getLogger(__name__).warning(
        "Certaines interfaces ou classes de service n'ont pas pu être importées dans definitions.py. "
        "AppConfig utilisera 'Any' pour ces types."
    )


# --- Enum pour les types de fold ---
class FoldType(str, enum.Enum):
    """Types de génération de folds pour WFO."""
    EXPANDING = "expanding" # Fenêtre IS expansive, OOS fixe ou suivant
    ROLLING = "rolling"     # Fenêtre IS roulante, OOS suivant
    ADAPTIVE = "adaptive"   # Taille/position ajustée par volatilité/régime
    COMBINATORIAL = "combinatorial" # Périodes IS/OOS non-contiguës
    OPTIMIZED_BOUNDARIES = "optimized_boundaries" # Basé sur détection de points de changement

@dataclass
class PathsConfig:
    """
    Configuration pour divers chemins de données et de logs utilisés par l'application.
    """
    data_historical_raw: str
    data_historical_processed_cleaned: str
    data_historical_processed_enriched: str
    logs_backtest_optimization: str
    logs_live: str
    results: str
    data_live_raw: str
    data_live_processed: str
    live_state: str
    # Optionnel: chemin vers les templates Jinja2 pour les rapports
    report_templates_dir: str = "src/reporting/templates"


@dataclass
class LoggingConfig:
    """
    Configuration pour le logging global de l'application.
    """
    level: str
    format: str
    log_to_file: bool
    log_filename_global: str
    log_filename_live: Optional[str] = None
    log_levels_by_module: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"LoggingConfig: level '{self.level}' invalide. Doit être l'un de {valid_levels}.")
        if self.log_levels_by_module:
            for module, level_str in self.log_levels_by_module.items():
                if level_str.upper() not in valid_levels:
                    raise ValueError(f"LoggingConfig: log_levels_by_module contient un niveau invalide ('{level_str}') pour le module '{module}'.")

@dataclass
class SlippageConfig:
    """
    Configuration détaillée pour la simulation du slippage.
    """
    method: str = "percentage"
    percentage_max_bps: Optional[float] = 5.0
    volume_factor: Optional[float] = 0.1
    volatility_factor: Optional[float] = 0.1
    min_slippage_bps: float = 0.0
    max_slippage_bps: float = 20.0

    def __post_init__(self):
        if self.method not in ["percentage", "fixed_bps", "volume_based", "volatility_based", "none"]:
            raise ValueError(f"SlippageConfig: méthode de slippage '{self.method}' non supportée.")
        # ... (autres validations existantes) ...

@dataclass
class SimulationDefaults:
    """
    Paramètres par défaut pour les simulations de backtesting.
    """
    initial_capital: float
    margin_leverage: int
    trading_fee_bps: float
    is_futures_trading: bool = False
    backtest_verbosity: int = 1
    risk_free_rate: float = 0.0
    run_name_prefix: str = "opt"
    slippage_config: SlippageConfig = field(default_factory=SlippageConfig)

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("SimulationDefaults: initial_capital doit être positif.")
        # ... (autres validations existantes) ...

@dataclass
class WfoSettings:
    """
    Paramètres pour le Walk-Forward Optimization (WFO).
    Étendu avec de nouveaux paramètres de génération de folds.
    """
    n_splits: int
    oos_period_days: int
    min_is_period_days: int
    metric_to_optimize: str
    optimization_direction: str
    fold_type: FoldType = FoldType.EXPANDING # Utilise l'Enum
    top_n_trials_for_oos_validation: int = 10
    top_n_trials_to_report_oos: int = 3
    # Nouveaux paramètres pour la génération de folds avancée
    overlap_ratio_is_oos: float = 0.0 # Ratio de chevauchement entre fin IS et début OOS (0.0 = pas de chevauchement)
    purging_period_days: int = 0 # Période de purge entre IS et OOS
    embargo_period_days: int = 0 # Période d'embargo après OOS avant le début du prochain IS (pour rolling)
    # Paramètres pour fold_type = ADAPTIVE
    adaptive_volatility_window: int = 20
    adaptive_volatility_column: str = 'close_returns_std' # Colonne à utiliser pour mesurer la volatilité
    adaptive_n_regimes_target: int = 3 # Nombre de régimes de volatilité à viser
    # Paramètres pour fold_type = COMBINATORIAL
    combinatorial_n_combinations: int = 10
    combinatorial_is_duration_days: int = 90
    combinatorial_oos_duration_days: int = 30
    combinatorial_min_gap_days: int = 5
    combinatorial_random_seed: Optional[int] = None
    # Paramètres pour fold_type = OPTIMIZED_BOUNDARIES
    change_point_model: str = "l2" # Modèle pour ruptures (ex: "l2", "rbf")
    change_point_penalty: Optional[float] = None # Pénalité pour Pelt, si None, Binseg avec n_splits-1 bkps
    change_point_series_column: str = 'close' # Colonne à utiliser pour la détection de points de changement

    def __post_init__(self):
        if self.n_splits <= 0: raise ValueError("n_splits doit être positif.")
        if self.oos_period_days <= 0: raise ValueError("oos_period_days doit être positif.")
        if self.min_is_period_days <= 0: raise ValueError("min_is_period_days doit être positif.")
        if not isinstance(self.fold_type, FoldType):
            try: self.fold_type = FoldType(str(self.fold_type).lower())
            except ValueError: raise ValueError(f"fold_type '{self.fold_type}' invalide. Options: {[ft.value for ft in FoldType]}")
        if self.optimization_direction not in ["maximize", "minimize"]: raise ValueError("optimization_direction doit être 'maximize' ou 'minimize'.")
        if not (0.0 <= self.overlap_ratio_is_oos < 1.0): raise ValueError("overlap_ratio_is_oos doit être entre 0.0 (inclus) et 1.0 (exclus).")
        if self.purging_period_days < 0: raise ValueError("purging_period_days ne peut pas être négatif.")
        if self.embargo_period_days < 0: raise ValueError("embargo_period_days ne peut pas être négatif.")
        # ... (autres validations existantes et nouvelles) ...

@dataclass
class SamplerPrunerProfile:
    """Définit un profil de sampler et pruner pour Optuna."""
    description: str
    sampler_name: str
    pruner_name: str
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    pruner_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptunaSettings:
    """Configuration pour Optuna."""
    n_trials: int
    n_jobs: int = 1
    storage: Optional[str] = None
    objectives_names: List[str] = field(default_factory=lambda: ["Total Net PnL USDC"])
    objectives_directions: List[str] = field(default_factory=lambda: ["maximize"])
    pareto_selection_strategy: Optional[str] = "SCORE_COMPOSITE"
    pareto_selection_weights: Optional[Dict[str, float]] = None
    pareto_selection_pnl_threshold: Optional[float] = 0.0
    default_profile_to_activate: Optional[str] = None
    sampler_name: str = "TPESampler"
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    pruner_name: str = "MedianPruner"
    pruner_params: Dict[str, Any] = field(default_factory=dict)
    sampler_pruner_profiles: Dict[str, SamplerPrunerProfile] = field(default_factory=dict)
    # Nouveau pour le callback d'early stopping
    early_stopping_rounds: Optional[int] = None # Nombre de trials sans amélioration pour arrêter
    early_stopping_patience: int = 0 # Nombre de trials initiaux à ignorer pour l'early stopping
    early_stopping_delta_threshold: float = 0.0001 # Amélioration minimale pour réinitialiser le compteur

    def __post_init__(self):
        # ... (validations existantes) ...
        if self.early_stopping_rounds is not None and self.early_stopping_rounds <= 0:
            raise ValueError("OptunaSettings: early_stopping_rounds doit être positif s'il est défini.")
        if self.early_stopping_patience < 0:
            raise ValueError("OptunaSettings: early_stopping_patience ne peut pas être négatif.")


@dataclass
class GlobalConfig:
    """Configuration globale pour l'application."""
    project_name: str
    paths: PathsConfig
    logging: LoggingConfig
    simulation_defaults: SimulationDefaults
    wfo_settings: WfoSettings
    optuna_settings: OptunaSettings

@dataclass
class SourceDetails:
    """Détails sur la source de données."""
    exchange: str
    asset_type: str

@dataclass
class AssetsAndTimeframes:
    """Configuration pour les actifs et les timeframes."""
    pairs: List[str]
    timeframes: List[str]
    def __post_init__(self):
        if not self.pairs: raise ValueError("AssetsAndTimeframes: 'pairs' ne peut être vide.")
        # timeframes peut être vide si la stratégie n'utilise que du 1min
        # if not self.timeframes: raise ValueError("AssetsAndTimeframes: 'timeframes' ne peut être vide.")


@dataclass
class HistoricalPeriod:
    """Définit la période pour la récupération des données historiques."""
    start_date: str
    end_date: Optional[str] = None
    def __post_init__(self):
        try:
            pd_start_date = pd.Timestamp(self.start_date, tz='UTC')
            if self.end_date and str(self.end_date).lower() != "now":
                pd_end_date = pd.Timestamp(self.end_date, tz='UTC')
                if pd_start_date >= pd_end_date:
                    raise ValueError(f"HistoricalPeriod: start_date ({self.start_date}) doit être antérieure à end_date ({self.end_date}).")
        except Exception as e:
             raise ValueError(f"HistoricalPeriod: date invalide. Erreur: {e}")


@dataclass
class FetchingOptions:
    """Options pour les processus de récupération de données."""
    max_workers: int
    batch_size: int
    def __post_init__(self):
        if self.max_workers <= 0: raise ValueError("max_workers doit être positif.")
        if self.batch_size <= 0: raise ValueError("batch_size doit être positif.")

@dataclass
class DataConfig:
    """Configuration relative aux sources de données, actifs et récupération."""
    source_details: SourceDetails
    assets_and_timeframes: AssetsAndTimeframes
    historical_period: HistoricalPeriod
    fetching_options: FetchingOptions

@dataclass
class ParamDetail:
    """Décrit un paramètre individuel dans l'espace des paramètres d'une stratégie."""
    type: str
    default: Optional[Any] = None
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    step: Optional[Union[float, int]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    # ... (validation __post_init__ existante) ...

@dataclass
class StrategyParamsConfig:
    """Configuration pour une stratégie de trading spécifique."""
    active_for_optimization: bool
    script_reference: str
    class_name: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    params_space: Dict[str, ParamDetail] = field(default_factory=dict)
    # Optionnel: version de la stratégie, pour suivi
    version: str = "1.0.0"
    # Optionnel: tags pour grouper ou filtrer les stratégies
    tags: List[str] = field(default_factory=list)


@dataclass
class StrategiesConfig:
    """Conteneur pour les configurations de multiples stratégies de trading."""
    strategies: Dict[str, StrategyParamsConfig]

@dataclass
class AccountConfig:
    """Configuration pour un seul compte API."""
    account_alias: str
    exchange: str
    account_type: str
    is_testnet: bool
    api_key_env_var: Optional[str] = None
    api_secret_env_var: Optional[str] = None
    def __post_init__(self):
        if not self.account_alias.strip(): raise ValueError("account_alias ne peut être vide.")

@dataclass
class ApiKeys:
    """Conteneur pour les identifiants API."""
    credentials: Dict[str, Tuple[Optional[str], Optional[str]]] = field(default_factory=dict)

@dataclass
class ExchangeSettings:
    """Paramètres relatifs à un exchange spécifique."""
    exchange_name: str = "binance"
    exchange_info_file_path: str = "config/binance_exchange_info.json" # Relatif à project_root

@dataclass
class GlobalLiveSettings:
    """Paramètres globaux pour les opérations de trading en direct."""
    run_live_trading: bool
    max_concurrent_strategies: int
    default_position_sizing_pct_capital: float
    global_risk_limit_pct_capital: float
    session_cycle_interval_seconds: int = 30
    # ... (validation __post_init__ existante) ...

@dataclass
class OverrideRiskSettings:
    """Paramètres de risque optionnels pour surcharger ceux d'un déploiement."""
    position_sizing_pct_capital: Optional[float] = None
    max_loss_per_trade_pct: Optional[float] = None # Non utilisé activement dans le backtester actuel
    # ... (validation __post_init__ existante) ...

@dataclass
class StrategyDeployment:
    """Configuration pour déployer une stratégie optimisée en trading en direct."""
    active: bool
    strategy_id: str # ID unique pour ce déploiement (ex: MaCross_BTCUSDT_5m_OptimRunXYZ)
    results_config_path: str # Chemin vers le live_config.json généré par WFO (relatif à project_root)
    account_alias_to_use: str
    override_risk_settings: Optional[OverrideRiskSettings] = None
    # ... (validation __post_init__ existante) ...

@dataclass
class LiveFetchConfig:
    """Configuration pour la récupération de données pendant le trading en direct."""
    crypto_pairs: List[str]
    intervals: List[str] # Ex: ["1m", "5m"]
    limit_init_history: int # Nombre de klines 1m à charger initialement
    limit_per_fetch: int # Nombre de klines 1m à récupérer par update REST
    max_retries: int = 3
    retry_backoff: float = 1.5
    # ... (validation __post_init__ existante) ...

@dataclass
class LiveLoggingConfig(LoggingConfig): # Hérite de LoggingConfig pour partager la structure
    """Configuration de logging spécifiquement pour les sessions de trading en direct."""
    # log_filename_live est déjà dans LoggingConfig
    # On peut ajouter des champs spécifiques au live si besoin
    pass


@dataclass
class LiveConfig:
    """Configuration agrégée pour les opérations de trading en direct."""
    global_live_settings: GlobalLiveSettings
    strategy_deployments: List[StrategyDeployment]
    live_fetch: LiveFetchConfig
    live_logging: LiveLoggingConfig # Utilise la nouvelle classe LiveLoggingConfig

@dataclass
class AppConfig:
    """
    Dataclass racine pour la configuration complète de l'application.
    Ajout des instances de services.
    """
    global_config: GlobalConfig
    data_config: DataConfig
    strategies_config: StrategiesConfig
    live_config: LiveConfig
    accounts_config: List[AccountConfig]
    api_keys: ApiKeys
    exchange_settings: ExchangeSettings
    project_root: Optional[str] = None

    # Instances des services principaux (initialisées par le loader.py)
    # Utiliser 'Any' pour éviter les problèmes d'import circulaire si les interfaces
    # sont définies dans un module qui importe definitions.py, ou utiliser des strings
    # pour forward references si Python >= 3.7 (mais dataclasses ne le gère pas bien directement).
    # L'idéal est d'importer les Protocoles/ABCs depuis src.core.interfaces.
    data_validator_instance: Optional[Any] = field(default=None, repr=False) # IDataValidator
    cache_manager_instance: Optional[Any] = field(default=None, repr=False)    # ICacheManager
    strategy_factory_instance: Optional[Any] = field(default=None, repr=False) # StrategyFactory
    strategy_loader_instance: Optional[Any] = field(default=None, repr=False)  # IStrategyLoader
    error_handler_instance: Optional[Any] = field(default=None, repr=False)    # IErrorHandler
    event_dispatcher_instance: Optional[Any] = field(default=None, repr=False) # IEventDispatcher

    def __post_init__(self):
        if self.accounts_config:
            aliases = [acc.account_alias for acc in self.accounts_config]
            if len(aliases) != len(set(aliases)):
                raise ValueError("AppConfig: account_alias dans accounts_config doit être unique.")
        # Les instances de service sont initialisées par le loader, pas ici.
