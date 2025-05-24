import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union, Tuple # Ensure Tuple is imported

@dataclass
class PathsConfig:
    """
    Configuration for various data and logging paths used by the application.
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
    Configuration for global application logging.
    """
    level: str
    format: str
    log_to_file: bool
    log_filename_global: str
    log_filename_live: Optional[str] = None
    log_levels_by_module: Optional[Dict[str, str]] = None

@dataclass
class SimulationDefaults:
    """
    Default settings for backtesting simulations.
    """
    initial_capital: float
    margin_leverage: int # Changed from float to int as leverage is typically whole numbers
    transaction_fee_pct: float
    slippage_pct: float

    def __post_init__(self):
        if not isinstance(self.margin_leverage, int) or self.margin_leverage < 1:
            # Allow float if it's a whole number, then cast to int
            if isinstance(self.margin_leverage, float) and self.margin_leverage.is_integer() and self.margin_leverage >= 1:
                self.margin_leverage = int(self.margin_leverage)
            else:
                raise ValueError("SimulationDefaults: margin_leverage must be an integer >= 1.")


@dataclass
class WfoSettings:
    """
    Settings for Walk-Forward Optimization (WFO).
    """
    n_splits: int
    oos_percent: int
    metric_to_optimize: str
    optimization_direction: str

@dataclass
class OptunaSettings:
    """
    Configuration for Optuna hyperparameter optimization.
    """
    n_trials: int
    sampler_name: str
    pruner_name: str
    sampler_params: Optional[Dict[str, Any]] = None
    pruner_params: Optional[Dict[str, Any]] = None
    storage: Optional[str] = None
    n_jobs: int = 1
    objectives_names: List[str] = field(default_factory=lambda: ["Total Net PnL USDC", "Sharpe Ratio"])
    objectives_directions: List[str] = field(default_factory=lambda: ["maximize", "maximize"])
    pareto_selection_strategy: Optional[str] = "PNL_MAX"
    pareto_selection_weights: Optional[Dict[str, float]] = None
    pareto_selection_pnl_threshold: Optional[float] = None
    n_best_for_oos: int = 10
    # >>> CHAMPS AJOUTÉS POUR LA GESTION DES PROFILS <<<
    default_profile_to_activate: Optional[str] = None
    sampler_pruner_profiles: Optional[Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    # >>> FIN DES CHAMPS AJOUTÉS <<<


    def __post_init__(self):
        if len(self.objectives_names) != len(self.objectives_directions):
            raise ValueError("OptunaSettings: objectives_names and objectives_directions must have the same length.")
        for direction in self.objectives_directions:
            if direction not in ["maximize", "minimize"]:
                raise ValueError(f"OptunaSettings: Invalid objective direction '{direction}'. Must be 'maximize' or 'minimize'.")
        if self.pareto_selection_strategy == "SCORE_COMPOSITE":
            if not self.pareto_selection_weights:
                raise ValueError("OptunaSettings: pareto_selection_weights must be provided if pareto_selection_strategy is 'SCORE_COMPOSITE'.")
            if not all(name in self.objectives_names for name in self.pareto_selection_weights.keys()): # type: ignore
                raise ValueError("OptunaSettings: All keys in pareto_selection_weights must correspond to an objective name in objectives_names.")
        if self.n_best_for_oos <= 0:
            raise ValueError("OptunaSettings: n_best_for_oos must be a positive integer.")
        if self.n_jobs < -1 or self.n_jobs == 0:
             raise ValueError(f"OptunaSettings: n_jobs ({self.n_jobs}) must be -1 or >= 1.")

@dataclass
class GlobalConfig:
    """
    Global configuration for the application, aggregating other specific configurations.
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
    Details about the data source.
    """
    exchange: str
    asset_type: str

@dataclass
class AssetsAndTimeframes:
    """
    Configuration for assets (pairs) and timeframes to be processed or used.
    """
    pairs: List[str]
    timeframes: List[str]

@dataclass
class HistoricalPeriod:
    """
    Defines the period for fetching historical data.
    """
    start_date: str
    end_date: Optional[str] = None

    def __post_init__(self):
        try:
            if self.end_date:
                # Basic check, more robust date parsing would be better if formats vary
                if pd.to_datetime(self.start_date) > pd.to_datetime(self.end_date): # type: ignore
                    raise ValueError(f"HistoricalPeriod: start_date ({self.start_date}) cannot be after end_date ({self.end_date}).")
        except Exception: # Catches parsing errors from pd.to_datetime or other issues
             raise ValueError("HistoricalPeriod: start_date and/or end_date are not valid date strings.")


@dataclass
class FetchingOptions:
    """
    Options for data fetching processes.
    """
    max_workers: int
    batch_size: int

    def __post_init__(self):
        if self.max_workers <= 0:
            raise ValueError("FetchingOptions: max_workers must be positive.")
        if self.batch_size <= 0:
            raise ValueError("FetchingOptions: batch_size must be positive.")

@dataclass
class DataConfig:
    """
    Configuration related to data sources, assets, and fetching.
    """
    source_details: SourceDetails
    assets_and_timeframes: AssetsAndTimeframes
    historical_period: HistoricalPeriod
    fetching_options: FetchingOptions

@dataclass
class ParamDetail:
    """
    Describes an individual parameter within a strategy's parameter space,
    used for Optuna optimization.
    """
    type: str
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    step: Optional[Union[float, int]] = None
    choices: Optional[List[Any]] = None

    def __post_init__(self):
        if self.type not in ["int", "float", "categorical"]:
            raise ValueError(f"ParamDetail: Invalid type '{self.type}'. Must be 'int', 'float', or 'categorical'.")
        if self.type == "categorical" and (self.choices is None or not isinstance(self.choices, list) or not self.choices):
            raise ValueError("ParamDetail: 'choices' must be a non-empty list for categorical parameters.")
        if self.type in ["int", "float"]:
            if self.low is None or self.high is None:
                raise ValueError(f"ParamDetail: 'low' and 'high' must be provided for {self.type} parameters.")
            if not isinstance(self.low, (int, float)) or not isinstance(self.high, (int, float)):
                raise ValueError(f"ParamDetail: 'low' and 'high' must be numeric for {self.type} parameters.")
            if self.low > self.high:
                raise ValueError(f"ParamDetail: 'low' ({self.low}) cannot be greater than 'high' ({self.high}).")
            if self.step is not None:
                if not isinstance(self.step, (int, float)):
                     raise ValueError(f"ParamDetail: 'step' must be numeric if provided for {self.type} parameters.")
                if self.step <= 0:
                    raise ValueError(f"ParamDetail: 'step' ({self.step}) must be positive if provided.")
                if self.type == "int":
                    if isinstance(self.step, float):
                        if self.step.is_integer():
                            self.step = int(self.step) 
                        else:
                            raise ValueError(f"ParamDetail: 'step' ({self.step}) must be a whole number for 'int' type parameters if specified as float.")
                    elif not isinstance(self.step, int):
                         raise ValueError(f"ParamDetail: 'step' ({self.step}, type: {type(self.step)}) must be an integer for 'int' type parameters.")


@dataclass
class StrategyParams:
    """
    Configuration for a specific trading strategy, including its parameter space for optimization.
    """
    active_for_optimization: bool
    script_reference: str
    class_name: str
    params_space: Dict[str, ParamDetail]

@dataclass
class StrategiesConfig:
    """
    Container for configurations of multiple trading strategies.
    """
    strategies: Dict[str, StrategyParams]

@dataclass
class AccountConfig:
    """
    Configuration for a single API account.
    """
    account_alias: str
    exchange: str
    account_type: str
    is_testnet: bool
    api_key_env_var: str
    api_secret_env_var: str 

    def __post_init__(self):
        if not self.account_alias.strip():
            raise ValueError("AccountConfig: account_alias cannot be empty.")
        if not self.exchange.strip():
            raise ValueError(f"AccountConfig (alias: {self.account_alias}): exchange cannot be empty.")
        if not self.account_type.strip():
            raise ValueError(f"AccountConfig (alias: {self.account_alias}): account_type cannot be empty.")
        if not self.api_key_env_var.strip():
            raise ValueError(f"AccountConfig (alias: {self.account_alias}): api_key_env_var cannot be empty.")
        if not self.api_secret_env_var.strip(): 
            raise ValueError(f"AccountConfig (alias: {self.account_alias}): api_secret_env_var cannot be empty.")

@dataclass
class GlobalLiveSettings:
    """
    Global settings for live trading operations.
    """
    run_live_trading: bool
    max_concurrent_strategies: int
    default_position_sizing_pct_capital: float
    global_risk_limit_pct_capital: float
    session_cycle_interval_seconds: int = 60

    def __post_init__(self):
        if self.max_concurrent_strategies <= 0:
            raise ValueError("GlobalLiveSettings: max_concurrent_strategies must be positive.")
        if not (0 < self.default_position_sizing_pct_capital <= 1):
            raise ValueError("GlobalLiveSettings: default_position_sizing_pct_capital must be between 0 (exclusive) and 1 (inclusive).")
        if not (0 < self.global_risk_limit_pct_capital <= 1):
            raise ValueError("GlobalLiveSettings: global_risk_limit_pct_capital must be between 0 (exclusive) and 1 (inclusive).")
        if self.session_cycle_interval_seconds <= 0:
            raise ValueError("GlobalLiveSettings: session_cycle_interval_seconds must be positive.")

@dataclass
class OverrideRiskSettings:
    """
    Optional risk setting overrides for a specific strategy deployment.
    """
    position_sizing_pct_capital: Optional[float] = None
    max_loss_per_trade_pct: Optional[float] = None

    def __post_init__(self):
        if self.position_sizing_pct_capital is not None and not (0 < self.position_sizing_pct_capital <= 1):
            raise ValueError("OverrideRiskSettings: position_sizing_pct_capital must be between 0 (exclusive) and 1 (inclusive) if provided.")
        if self.max_loss_per_trade_pct is not None and not (0 < self.max_loss_per_trade_pct <= 1):
            raise ValueError("OverrideRiskSettings: max_loss_per_trade_pct must be between 0 (exclusive) and 1 (inclusive) if provided.")

@dataclass
class StrategyDeployment:
    """
    Configuration for deploying a specific optimized strategy to live trading.
    """
    active: bool
    strategy_id: str
    results_config_path: str
    account_alias_to_use: str
    override_risk_settings: Optional[OverrideRiskSettings] = None

    def __post_init__(self):
        if self.active:
            if not self.strategy_id.strip():
                raise ValueError("StrategyDeployment: strategy_id cannot be empty for an active deployment.")
            if not self.results_config_path.strip():
                raise ValueError(f"StrategyDeployment (ID: {self.strategy_id}): results_config_path cannot be empty for an active deployment.")
            if not self.account_alias_to_use.strip():
                raise ValueError(f"StrategyDeployment (ID: {self.strategy_id}): account_alias_to_use cannot be empty for an active deployment.")

@dataclass
class LiveLoggingConfig:
    """
    Logging configuration specifically for live trading sessions.
    """
    level: str = "INFO"
    log_to_file: bool = True
    log_filename_live: str = "live_trading_run.log"
    log_levels_by_module: Optional[Dict[str, str]] = None

@dataclass
class LiveFetchConfig:
    """
    Configuration for fetching data during live trading.
    """
    crypto_pairs: List[str]
    intervals: List[str]
    limit_init_history: int
    limit_per_fetch: int
    max_retries: int
    retry_backoff: float

    def __post_init__(self):
        if self.limit_init_history < 0:
            raise ValueError("LiveFetchConfig: limit_init_history cannot be negative.")
        if self.limit_per_fetch <= 0:
            raise ValueError("LiveFetchConfig: limit_per_fetch must be positive.")
        if self.max_retries < 0:
            raise ValueError("LiveFetchConfig: max_retries cannot be negative.")
        if self.retry_backoff <= 0:
            raise ValueError("LiveFetchConfig: retry_backoff must be positive.")

@dataclass
class LiveConfig:
    """
    Aggregated configuration for live trading operations.
    """
    global_live_settings: GlobalLiveSettings
    strategy_deployments: List[StrategyDeployment]
    live_fetch: LiveFetchConfig
    live_logging: LiveLoggingConfig

@dataclass
class ApiKeys:
    """
    Container for API credentials, mapped by account alias.
    """
    credentials: Dict[str, Tuple[Optional[str], Optional[str]]] = field(default_factory=dict)

@dataclass
class AppConfig:
    """
    Root configuration class for the entire application.
    """
    global_config: GlobalConfig
    data_config: DataConfig
    strategies_config: StrategiesConfig
    live_config: LiveConfig
    accounts_config: List[AccountConfig]
    api_keys: ApiKeys
    project_root: str = field(init=False) # type: ignore

    def __post_init__(self):
        if self.accounts_config:
            aliases = [acc.account_alias for acc in self.accounts_config]
            if len(aliases) != len(set(aliases)):
                raise ValueError("AppConfig: account_alias in accounts_config must be unique.")
