# src/config/definitions.py
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union, Tuple
import pandas as pd # Added for pd.to_datetime

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

@dataclass
class LoggingConfig:
    """
    Configuration pour le logging global de l'application.
    """
    level: str
    format: str
    log_to_file: bool
    log_filename_global: str
    log_filename_live: Optional[str] = None # Nom de fichier spécifique pour les logs live
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
    method: str = "percentage" # Méthodes: "percentage", "fixed_bps", "volume_based", "volatility_based", "none"
    percentage_max_bps: Optional[float] = 5.0 # Pour la méthode 'percentage', slippage max en BPS
    volume_factor: Optional[float] = 0.1 # Pour 'volume_based'
    volatility_factor: Optional[float] = 0.1 # Pour 'volatility_based'
    min_slippage_bps: float = 0.0 # Slippage minimum à appliquer en BPS
    max_slippage_bps: float = 20.0 # Plafond de slippage en BPS

    def __post_init__(self):
        if self.method not in ["percentage", "fixed_bps", "volume_based", "volatility_based", "none"]:
            raise ValueError(f"SlippageConfig: méthode de slippage '{self.method}' non supportée.")
        if self.percentage_max_bps is not None and self.percentage_max_bps < 0:
            raise ValueError("SlippageConfig: percentage_max_bps ne peut pas être négatif.")
        if self.volume_factor is not None and self.volume_factor < 0:
            raise ValueError("SlippageConfig: volume_factor ne peut pas être négatif.")
        if self.volatility_factor is not None and self.volatility_factor < 0:
            raise ValueError("SlippageConfig: volatility_factor ne peut pas être négatif.")
        if self.min_slippage_bps < 0:
            raise ValueError("SlippageConfig: min_slippage_bps ne peut pas être négatif.")
        if self.max_slippage_bps < self.min_slippage_bps:
            raise ValueError(f"SlippageConfig: max_slippage_bps ({self.max_slippage_bps}) ne peut pas être inférieur à min_slippage_bps ({self.min_slippage_bps}).")

@dataclass
class SimulationDefaults:
    """
    Paramètres par défaut pour les simulations de backtesting.
    """
    initial_capital: float
    margin_leverage: int
    trading_fee_bps: float # Frais de trading en points de base (ex: 7 pour 0.07%)
    is_futures_trading: bool = False
    backtest_verbosity: int = 1
    risk_free_rate: float = 0.0 # Taux sans risque annuel pour les calculs de Sharpe, etc.
    run_name_prefix: str = "opt" # Préfixe pour les ID de run WFO
    slippage_config: SlippageConfig = field(default_factory=SlippageConfig) 

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("SimulationDefaults: initial_capital doit être positif.")
        if not isinstance(self.margin_leverage, int) or self.margin_leverage < 1:
            raise ValueError("SimulationDefaults: margin_leverage doit être un entier >= 1.")
        if self.trading_fee_bps < 0:
            raise ValueError("SimulationDefaults: trading_fee_bps ne peut pas être négatif.")
        if not (0 <= self.risk_free_rate <= 1): 
            raise ValueError("SimulationDefaults: risk_free_rate doit être entre 0 et 1.")
        if self.backtest_verbosity not in [0, 1, 2]:
            raise ValueError("SimulationDefaults: backtest_verbosity doit être 0, 1, ou 2.")

@dataclass
class WfoSettings:
    """
    Paramètres pour le Walk-Forward Optimization (WFO).
    """
    n_splits: int
    oos_period_days: int 
    min_is_period_days: int 
    metric_to_optimize: str 
    optimization_direction: str 
    fold_type: str = "expanding" 
    top_n_trials_for_oos_validation: int = 10
    top_n_trials_to_report_oos: int = 3

    def __post_init__(self):
        if self.n_splits <= 0:
            raise ValueError("WfoSettings: n_splits doit être positif.")
        if self.oos_period_days <= 0:
            raise ValueError("WfoSettings: oos_period_days doit être positif.")
        if self.min_is_period_days <= 0:
            raise ValueError("WfoSettings: min_is_period_days doit être positif.")
        if self.fold_type not in ["expanding"]: 
            raise ValueError(f"WfoSettings: fold_type '{self.fold_type}' non supporté.")
        if self.optimization_direction not in ["maximize", "minimize"]:
            raise ValueError("WfoSettings: optimization_direction doit être 'maximize' ou 'minimize'.")
        if self.top_n_trials_for_oos_validation <= 0:
            raise ValueError("WfoSettings: top_n_trials_for_oos_validation doit être positif.")
        if self.top_n_trials_to_report_oos <= 0:
            raise ValueError("WfoSettings: top_n_trials_to_report_oos doit être positif.")
        if self.top_n_trials_to_report_oos > self.top_n_trials_for_oos_validation:
            raise ValueError("WfoSettings: top_n_trials_to_report_oos ne peut pas être supérieur à top_n_trials_for_oos_validation.")

@dataclass
class SamplerPrunerProfile:
    """
    Définit un profil de sampler et pruner pour Optuna.
    """
    description: str
    sampler_name: str
    pruner_name: str
    sampler_params: Dict[str, Any] = field(default_factory=dict) 
    pruner_params: Dict[str, Any] = field(default_factory=dict) 


@dataclass
class OptunaSettings:
    """
    Configuration pour l'optimisation des hyperparamètres avec Optuna.
    """
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


    def __post_init__(self):
        if self.n_trials <= 0:
            raise ValueError("OptunaSettings: n_trials doit être positif.")
        if self.n_jobs < -1 or self.n_jobs == 0: 
             raise ValueError(f"OptunaSettings: n_jobs ({self.n_jobs}) doit être -1 (pour tous les CPUs) ou >= 1.")
        if len(self.objectives_names) != len(self.objectives_directions):
            raise ValueError("OptunaSettings: objectives_names et objectives_directions doivent avoir la même longueur.")
        for direction in self.objectives_directions:
            if direction not in ["maximize", "minimize"]:
                raise ValueError(f"OptunaSettings: La direction d'objectif '{direction}' est invalide. Doit être 'maximize' ou 'minimize'.")
        if self.pareto_selection_strategy == "SCORE_COMPOSITE":
            if not self.pareto_selection_weights:
                raise ValueError("OptunaSettings: pareto_selection_weights est requis pour la stratégie 'SCORE_COMPOSITE'.")
            if not all(name in self.objectives_names for name in self.pareto_selection_weights.keys()): # type: ignore
                raise ValueError("OptunaSettings: Toutes les clés dans pareto_selection_weights doivent correspondre à un nom d'objectif dans objectives_names.")
        if self.default_profile_to_activate and self.default_profile_to_activate not in self.sampler_pruner_profiles:
            raise ValueError(f"OptunaSettings: default_profile_to_activate '{self.default_profile_to_activate}' non trouvé dans sampler_pruner_profiles.")

@dataclass
class GlobalConfig:
    """
    Configuration globale pour l'application, agrégeant d'autres configurations spécifiques.
    """
    project_name: str
    paths: PathsConfig
    logging: LoggingConfig
    simulation_defaults: SimulationDefaults
    wfo_settings: WfoSettings
    optuna_settings: OptunaSettings

@dataclass
class SourceDetails:
    """
    Détails sur la source de données.
    """
    exchange: str
    asset_type: str 

@dataclass
class AssetsAndTimeframes:
    """
    Configuration pour les actifs (paires) et les timeframes à traiter ou utiliser.
    """
    pairs: List[str]
    timeframes: List[str]

    def __post_init__(self):
        if not self.pairs:
            raise ValueError("AssetsAndTimeframes: la liste 'pairs' ne peut pas être vide.")
        if not self.timeframes:
            raise ValueError("AssetsAndTimeframes: la liste 'timeframes' ne peut pas être vide.")

@dataclass
class HistoricalPeriod:
    """
    Définit la période pour la récupération des données historiques.
    """
    start_date: str 
    end_date: Optional[str] = None 

    def __post_init__(self):
        try:
            pd_start_date = pd.Timestamp(self.start_date, tz='UTC')
            if self.end_date:
                pd_end_date = pd.Timestamp(self.end_date, tz='UTC')
                if pd_start_date >= pd_end_date:
                    raise ValueError(f"HistoricalPeriod: start_date ({self.start_date}) doit être antérieure à end_date ({self.end_date}).")
        except Exception as e:
             raise ValueError(f"HistoricalPeriod: start_date '{self.start_date}' et/ou end_date '{self.end_date}' ne sont pas des chaînes de date valides ou ne forment pas une période valide. Erreur: {e}")

@dataclass
class FetchingOptions: 
    """
    Options pour les processus de récupération de données.
    """
    max_workers: int
    batch_size: int

    def __post_init__(self):
        if self.max_workers <= 0:
            raise ValueError("FetchingOptions: max_workers doit être positif.")
        if self.batch_size <= 0:
            raise ValueError("FetchingOptions: batch_size doit être positif.")

@dataclass
class DataConfig:
    """
    Configuration relative aux sources de données, actifs et récupération.
    """
    source_details: SourceDetails
    assets_and_timeframes: AssetsAndTimeframes
    historical_period: HistoricalPeriod
    fetching_options: FetchingOptions

@dataclass
class ParamDetail:
    """
    Décrit un paramètre individuel dans l'espace des paramètres d'une stratégie,
    utilisé pour l'optimisation Optuna.
    """
    type: str 
    default: Optional[Any] = None
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    step: Optional[Union[float, int]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False 

    def __post_init__(self):
        if self.type not in ["int", "float", "categorical"]:
            raise ValueError(f"ParamDetail: Type invalide '{self.type}'. Doit être 'int', 'float', ou 'categorical'.")
        if self.type == "categorical":
            if not isinstance(self.choices, list) or not self.choices:
                raise ValueError("ParamDetail: 'choices' doit être une liste non vide pour les paramètres catégoriels.")
            if self.default is not None and self.default not in self.choices:
                raise ValueError(f"ParamDetail: La valeur par défaut '{self.default}' n'est pas dans les choix {self.choices} pour le paramètre catégoriel.")
        elif self.type in ["int", "float"]:
            if self.low is None or self.high is None:
                raise ValueError(f"ParamDetail: 'low' et 'high' doivent être fournis pour les paramètres de type '{self.type}'.")
            if not isinstance(self.low, (int, float)) or not isinstance(self.high, (int, float)):
                raise ValueError(f"ParamDetail: 'low' et 'high' doivent être numériques pour les paramètres de type '{self.type}'.")
            # CORRECTION ICI: Autoriser low == high
            if self.low > self.high: 
                raise ValueError(f"ParamDetail: 'low' ({self.low}) ne peut pas être supérieur à 'high' ({self.high}).")
            if self.default is not None:
                if not (self.low <= self.default <= self.high): # type: ignore
                    raise ValueError(f"ParamDetail: La valeur par défaut '{self.default}' est hors de la plage [{self.low}, {self.high}].")
            if self.step is not None:
                if not isinstance(self.step, (int, float)):
                    raise ValueError(f"ParamDetail: 'step' doit être numérique s'il est fourni pour les paramètres de type '{self.type}'.")
                if self.step <= 0:
                    raise ValueError(f"ParamDetail: 'step' ({self.step}) doit être positif s'il est fourni.")
                # Vérifier la cohérence de step avec low et high si low != high
                if self.low != self.high and isinstance(self.low, (int,float)) and isinstance(self.high, (int,float)):
                    if self.type == "int" and not isinstance((self.high - self.low) / self.step, int) and ((self.high - self.low) % self.step != 0) : # type: ignore
                         # Permettre si high-low est un multiple de step, ou si step est tel que high est atteignable
                         pass # La validation d'Optuna est plus souple ici
                    # Pour float, Optuna gère aussi la non-atteignabilité exacte de high par step.
                    pass

            if self.type == "int":
                for val_name, val in [('default', self.default), ('low', self.low), ('high', self.high), ('step', self.step)]:
                    if val is not None and not isinstance(val, int):
                        if isinstance(val, float) and val.is_integer():
                            setattr(self, val_name, int(val))
                        else:
                            raise ValueError(f"ParamDetail: '{val_name}' ({val}) doit être un entier pour le type 'int'.")
        if self.type == "float" and self.log_scale and (self.low is None or self.low <= 0): # type: ignore
            raise ValueError("ParamDetail: 'low' doit être > 0 pour les paramètres float avec log_scale=True.")

@dataclass
class StrategyParamsConfig:
    """
    Configuration pour une stratégie de trading spécifique, incluant son espace de paramètres
    pour l'optimisation et les paramètres par défaut.
    """
    active_for_optimization: bool
    script_reference: str 
    class_name: str 
    default_params: Dict[str, Any] = field(default_factory=dict)
    params_space: Dict[str, ParamDetail] = field(default_factory=dict)

@dataclass
class StrategiesConfig:
    """
    Conteneur pour les configurations de multiples stratégies de trading.
    """
    strategies: Dict[str, StrategyParamsConfig]

@dataclass
class AccountConfig:
    """
    Configuration pour un seul compte API.
    """
    account_alias: str
    exchange: str
    account_type: str 
    is_testnet: bool
    api_key_env_var: Optional[str] = None
    api_secret_env_var: Optional[str] = None

    def __post_init__(self):
        if not self.account_alias.strip():
            raise ValueError("AccountConfig: account_alias ne peut pas être vide.")

@dataclass
class ApiKeys:
    """
    Conteneur pour les identifiants API.
    """
    credentials: Dict[str, Tuple[Optional[str], Optional[str]]] = field(default_factory=dict)

@dataclass
class ExchangeSettings:
    """
    Paramètres relatifs à un exchange spécifique.
    """
    exchange_name: str = "binance"
    exchange_info_file_path: str = "config/binance_exchange_info.json"

@dataclass
class GlobalLiveSettings:
    """
    Paramètres globaux pour les opérations de trading en direct.
    """
    run_live_trading: bool
    max_concurrent_strategies: int
    default_position_sizing_pct_capital: float
    global_risk_limit_pct_capital: float
    session_cycle_interval_seconds: int = 30

    def __post_init__(self):
        if self.max_concurrent_strategies <= 0:
            raise ValueError("GlobalLiveSettings: max_concurrent_strategies doit être positif.")
        if not (0 < self.default_position_sizing_pct_capital <= 1):
            raise ValueError("GlobalLiveSettings: default_position_sizing_pct_capital doit être entre 0 (exclusif) et 1 (inclusif).")
        if not (0 < self.global_risk_limit_pct_capital <= 1):
            raise ValueError("GlobalLiveSettings: global_risk_limit_pct_capital doit être entre 0 (exclusif) et 1 (inclusif).")
        if self.session_cycle_interval_seconds <= 0:
            raise ValueError("GlobalLiveSettings: session_cycle_interval_seconds doit être positif.")

@dataclass
class OverrideRiskSettings:
    """
    Paramètres de risque optionnels pour surcharger ceux d'un déploiement de stratégie.
    """
    position_sizing_pct_capital: Optional[float] = None
    max_loss_per_trade_pct: Optional[float] = None

    def __post_init__(self):
        if self.position_sizing_pct_capital is not None and not (0 < self.position_sizing_pct_capital <= 1):
            raise ValueError("OverrideRiskSettings: position_sizing_pct_capital doit être entre 0 (exclusif) et 1 (inclusif) s'il est fourni.")
        if self.max_loss_per_trade_pct is not None and not (0 < self.max_loss_per_trade_pct <= 1):
            raise ValueError("OverrideRiskSettings: max_loss_per_trade_pct doit être entre 0 (exclusif) et 1 (inclusif) s'il est fourni.")

@dataclass
class StrategyDeployment:
    """
    Configuration pour déployer une stratégie optimisée en trading en direct.
    """
    active: bool
    strategy_id: str
    results_config_path: str 
    account_alias_to_use: str
    override_risk_settings: Optional[OverrideRiskSettings] = None

    def __post_init__(self):
        if self.active:
            if not self.strategy_id.strip():
                raise ValueError("StrategyDeployment: strategy_id ne peut pas être vide pour un déploiement actif.")
            if not self.results_config_path.strip():
                raise ValueError(f"StrategyDeployment (ID: {self.strategy_id}): results_config_path ne peut pas être vide pour un déploiement actif.")
            if not self.account_alias_to_use.strip():
                raise ValueError(f"StrategyDeployment (ID: {self.strategy_id}): account_alias_to_use ne peut pas être vide pour un déploiement actif.")

@dataclass
class LiveFetchConfig:
    """
    Configuration pour la récupération de données pendant le trading en direct.
    """
    crypto_pairs: List[str]
    intervals: List[str]
    limit_init_history: int
    limit_per_fetch: int
    max_retries: int = 3
    retry_backoff: float = 1.5

    def __post_init__(self):
        if self.limit_init_history < 0: 
            raise ValueError("LiveFetchConfig: limit_init_history ne peut pas être négatif.")
        if self.limit_per_fetch <= 0:
            raise ValueError("LiveFetchConfig: limit_per_fetch doit être positif.")
        if self.max_retries < 0:
            raise ValueError("LiveFetchConfig: max_retries ne peut pas être négatif.")
        if self.retry_backoff <= 0:
            raise ValueError("LiveFetchConfig: retry_backoff doit être positif.")

@dataclass
class LiveLoggingConfig:
    """
    Configuration de logging spécifiquement pour les sessions de trading en direct.
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    log_to_file: bool = True
    log_filename_live: str = "live_trading_activity.log"
    log_levels_by_module: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"LiveLoggingConfig: level '{self.level}' invalide.")
        if self.log_levels_by_module:
            for module, level_str in self.log_levels_by_module.items():
                if level_str.upper() not in valid_levels:
                    raise ValueError(f"LiveLoggingConfig: Niveau de log invalide ('{level_str}') pour le module '{module}'.")

@dataclass
class LiveConfig:
    """
    Configuration agrégée pour les opérations de trading en direct.
    """
    global_live_settings: GlobalLiveSettings
    strategy_deployments: List[StrategyDeployment]
    live_fetch: LiveFetchConfig
    live_logging: LiveLoggingConfig

@dataclass
class AppConfig:
    """
    Dataclass racine pour la configuration complète de l'application.
    """
    global_config: GlobalConfig
    data_config: DataConfig
    strategies_config: StrategiesConfig
    live_config: LiveConfig
    accounts_config: List[AccountConfig]
    api_keys: ApiKeys
    exchange_settings: ExchangeSettings
    project_root: Optional[str] = None 

    def __post_init__(self):
        if self.accounts_config:
            aliases = [acc.account_alias for acc in self.accounts_config]
            if len(aliases) != len(set(aliases)):
                raise ValueError("AppConfig: account_alias dans accounts_config doit être unique.")
