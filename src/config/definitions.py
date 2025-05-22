import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union, Tuple # Ensure Tuple is imported

@dataclass
class PathsConfig:
    """
    Configuration for various data and logging paths used by the application.
    """
    data_historical_raw: str
    """str: Path to the directory for raw historical 1-minute kline data, typically versioned by fetch run."""
    data_historical_processed_cleaned: str
    """str: Path to the directory for cleaned 1-minute historical data (e.g., Parquet format)."""
    data_historical_processed_enriched: str
    """str: Path to the directory for enriched historical data (1-minute base + aggregated K-lines + base indicators)."""
    logs_backtest_optimization: str
    """str: Path to the base directory for logs generated during backtest optimization runs (WFO)."""
    logs_live: str
    """str: Path to the base directory for logs generated during live trading sessions."""
    results: str
    """str: Path to the base directory for storing generated reports and selected live configurations."""
    data_live_raw: str
    """str: Path to the directory for raw 1-minute kline data fetched during live trading."""
    data_live_processed: str
    """str: Path to the directory for data processed specifically for live trading strategies (e.g., indicators calculated on the fly)."""
    live_state: str
    """str: Path to the directory for storing persistent state files for live trading managers."""

@dataclass
class LoggingConfig:
    """
    Configuration for global application logging.
    """
    level: str
    """str: Global logging level (e.g., "INFO", "DEBUG")."""
    format: str
    """str: Logging format string."""
    log_to_file: bool
    """bool: Whether to log to a file in addition to the console."""
    log_filename_global: str
    """str: Filename for the global log file if log_to_file is True."""
    log_filename_live: Optional[str] = None
    """Optional[str]: Specific filename for live trading logs, if different from global."""
    log_levels_by_module: Optional[Dict[str, str]] = None
    """Optional[Dict[str, str]]: Allows specifying different log levels for specific modules (e.g., {"src.live.manager": "DEBUG"})."""

@dataclass
class SimulationDefaults:
    """
    Default settings for backtesting simulations.
    """
    initial_capital: float
    """float: The initial capital amount for simulations."""
    margin_leverage: int
    """int: The margin leverage to be used in simulations."""
    transaction_fee_pct: float
    """float: Transaction fee percentage (e.g., 0.001 for 0.1%)."""
    slippage_pct: float
    """float: Slippage percentage to apply to trades (e.g., 0.0005 for 0.05%)."""

@dataclass
class WfoSettings:
    """
    Settings for Walk-Forward Optimization (WFO).
    """
    n_splits: int
    """int: Number of splits or segments for the In-Sample period in WFO."""
    oos_percent: int
    """int: Percentage of the total data to be used as the Out-of-Sample (OOS) period."""
    metric_to_optimize: str
    """str: The primary metric to optimize during the In-Sample phase of WFO (legacy, see OptunaSettings for multi-objective)."""
    optimization_direction: str
    """str: Direction of optimization for the metric_to_optimize ("maximize" or "minimize") (legacy)."""

@dataclass
class OptunaSettings:
    """
    Configuration for Optuna hyperparameter optimization.
    """
    n_trials: int
    """int: Number of optimization trials to run."""
    sampler_name: str
    """str: Name of the Optuna sampler to use (e.g., "TPESampler", "NSGAIISampler")."""
    sampler_params: Optional[Dict[str, Any]] = None
    """Optional[Dict[str, Any]]: Dictionary of parameters to pass to the sampler's constructor (e.g., {"n_startup_trials": 5})."""
    pruner_name: str
    """str: Name of the Optuna pruner to use (e.g., "MedianPruner", "HyperbandPruner")."""
    pruner_params: Optional[Dict[str, Any]] = None
    """Optional[Dict[str, Any]]: Dictionary of parameters to pass to the pruner's constructor."""
    storage: Optional[str] = None
    """Optional[str]: Optuna storage URL (e.g., "sqlite:///optuna_studies.db"). If None, in-memory storage is used."""
    n_jobs: int = 1
    """int: Number of parallel jobs for Optuna optimization (-1 to use all available CPUs)."""
    objectives_names: List[str] = field(default_factory=lambda: ["Total Net PnL USDC", "Sharpe Ratio"])
    """List[str]: Names of the objectives for multi-objective optimization."""
    objectives_directions: List[str] = field(default_factory=lambda: ["maximize", "maximize"])
    """List[str]: Directions for each objective ("maximize" or "minimize")."""
    pareto_selection_strategy: Optional[str] = "PNL_MAX"
    """Optional[str]: Strategy to select the best trial from the Pareto front for OOS validation (e.g., "PNL_MAX", "SCORE_COMPOSITE")."""
    pareto_selection_weights: Optional[Dict[str, float]] = None
    """Optional[Dict[str, float]]: Weights for objectives if pareto_selection_strategy is "SCORE_COMPOSITE" (e.g., {"Total Net PnL USDC": 0.7, "Sharpe Ratio": 0.3})."""
    pareto_selection_pnl_threshold: Optional[float] = None
    """Optional[float]: Minimum PnL threshold to consider a trial from the Pareto front for OOS validation."""
    n_best_for_oos: int = 10
    """int: Number of best In-Sample trials (from Pareto front) to validate on Out-of-Sample data."""

    def __post_init__(self):
        """
        Validates Optuna settings after initialization.
        """
        if len(self.objectives_names) != len(self.objectives_directions):
            raise ValueError("OptunaSettings: objectives_names and objectives_directions must have the same length.")
        for direction in self.objectives_directions:
            if direction not in ["maximize", "minimize"]:
                raise ValueError(f"OptunaSettings: Invalid objective direction '{direction}'. Must be 'maximize' or 'minimize'.")
        if self.pareto_selection_strategy == "SCORE_COMPOSITE":
            if not self.pareto_selection_weights:
                raise ValueError("OptunaSettings: pareto_selection_weights must be provided if pareto_selection_strategy is 'SCORE_COMPOSITE'.")
            if not all(name in self.objectives_names for name in self.pareto_selection_weights.keys()):
                raise ValueError("OptunaSettings: All keys in pareto_selection_weights must correspond to an objective name in objectives_names.")
        if self.n_best_for_oos <= 0:
            raise ValueError("OptunaSettings: n_best_for_oos must be a positive integer.")
        if self.n_jobs < -1 or self.n_jobs == 0: # Optuna n_jobs: -1 means all CPUs, 1 means sequential, >1 for parallel. 0 is invalid.
             raise ValueError(f"OptunaSettings: n_jobs ({self.n_jobs}) must be -1 or >= 1.")


@dataclass
class GlobalConfig:
    """
    Global configuration for the application, aggregating other specific configurations.
    """
    project_name: str
    """str: Name of the trading bot project."""
    paths: PathsConfig
    """PathsConfig: Dataclass instance holding all relevant paths."""
    logging: LoggingConfig
    """LoggingConfig: Dataclass instance for global logging setup."""
    simulation_defaults: SimulationDefaults
    """SimulationDefaults: Dataclass instance for default backtesting simulation parameters."""
    wfo_settings: WfoSettings
    """WfoSettings: Dataclass instance for Walk-Forward Optimization settings."""
    optuna_settings: OptunaSettings
    """OptunaSettings: Dataclass instance for Optuna optimization settings."""

@dataclass
class SourceDetails:
    """
    Details about the data source.
    """
    exchange: str
    """str: Name of the exchange (e.g., "binance")."""
    asset_type: str
    """str: Type of asset (e.g., "spot", "margin", "futures")."""

@dataclass
class AssetsAndTimeframes:
    """
    Configuration for assets (pairs) and timeframes to be processed or used.
    """
    pairs: List[str]
    """List[str]: List of trading pairs/symbols (e.g., ["BTCUSDT", "ETHUSDT"])."""
    timeframes: List[str]
    """List[str]: List of K-line timeframes to be used or generated (e.g., ["1m", "5m", "1h"])."""

@dataclass
class HistoricalPeriod:
    """
    Defines the period for fetching historical data.
    """
    start_date: str
    """str: Start date for historical data fetching (format: "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD")."""
    end_date: Optional[str] = None
    """Optional[str]: End date for historical data fetching. If None, fetches up to the current time."""

    def __post_init__(self):
        """
        Validates the historical period.
        """
        # Basic validation example; more complex date parsing/comparison could be added
        # This requires parsing the dates, which can be complex due to formats.
        # For simplicity, this basic check assumes string comparison works for YYYY-MM-DD.
        # A more robust solution would use datetime objects.
        try:
            # Attempt to parse to ensure they are valid date(time) strings, but don't store datetime objects here
            # to keep the config simple (strings from JSON). loader.py can do full validation if needed.
            if self.end_date:
                # Example: basic check, a more robust date comparison would be better
                if self.start_date > self.end_date: # Lexicographical comparison
                    raise ValueError(f"HistoricalPeriod: start_date ({self.start_date}) cannot be after end_date ({self.end_date}).")
        except TypeError: # Handles if dates are not strings
             raise ValueError("HistoricalPeriod: start_date and end_date must be strings.")


@dataclass
class FetchingOptions:
    """
    Options for data fetching processes.
    """
    max_workers: int
    """int: Maximum number of worker threads for parallel data fetching."""
    batch_size: int
    """int: Number of klines to fetch per API request (batch size)."""

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
    """SourceDetails: Details of the data source."""
    assets_and_timeframes: AssetsAndTimeframes
    """AssetsAndTimeframes: Specification of assets and timeframes."""
    historical_period: HistoricalPeriod
    """HistoricalPeriod: Period for historical data fetching."""
    fetching_options: FetchingOptions
    """FetchingOptions: Options for the data fetching process."""

@dataclass
class ParamDetail:
    """
    Describes an individual parameter within a strategy's parameter space,
    used for Optuna optimization.
    """
    type: str
    """str: Type of the parameter (e.g., "int", "float", "categorical")."""
    low: Optional[Union[float, int]] = None
    """Optional[Union[float, int]]: Lower bound for "int" or "float" type parameters."""
    high: Optional[Union[float, int]] = None
    """Optional[Union[float, int]]: Upper bound for "int" or "float" type parameters."""
    step: Optional[Union[float, int]] = None
    """Optional[Union[float, int]]: Step size for "int" or "float" type parameters."""
    choices: Optional[List[Any]] = None
    """Optional[List[Any]]: List of possible values for "categorical" type parameters."""

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
                if self.type == "int" and not isinstance(self.step, int):
                    raise ValueError("ParamDetail: 'step' must be an integer for 'int' type parameters.")


@dataclass
class StrategyParams:
    """
    Configuration for a specific trading strategy, including its parameter space for optimization.
    """
    active_for_optimization: bool
    """bool: Whether this strategy is active for optimization runs."""
    script_reference: str
    """str: Path to the strategy's Python script (e.g., "src/strategies/my_strategy.py")."""
    class_name: str
    """str: Name of the strategy class within the script."""
    params_space: Dict[str, ParamDetail]
    """Dict[str, ParamDetail]: Dictionary defining the parameter space for Optuna, where keys are parameter names."""

@dataclass
class StrategiesConfig:
    """
    Container for configurations of multiple trading strategies.
    """
    strategies: Dict[str, StrategyParams]
    """Dict[str, StrategyParams]: Dictionary where keys are strategy names and values are their configurations."""

@dataclass
class AccountConfig:
    """
    Configuration for a single API account.
    """
    account_alias: str
    """str: User-defined unique alias for the account (e.g., "binance_margin_main", "binance_futures_test")."""
    exchange: str
    """str: Name of the exchange (e.g., "binance")."""
    account_type: str
    """str: Type of account on the exchange (e.g., "MARGIN", "ISOLATED_MARGIN", "FUTURES", "SPOT")."""
    is_testnet: bool
    """bool: True if this is a testnet account, False otherwise."""
    api_key_env_var: str
    """str: Name of the environment variable holding the API key for this account (e.g., "BINANCE_API_KEY_MAIN")."""
    api_secret_env_var: str
    """str: Name of the environment variable holding the API secret for this account (e.g., "BINANCE_SECRET_KEY_MAIN")."""

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
    """bool: Master switch to enable or disable live trading."""
    # account_type: str # Removed, as this should be per AccountConfig
    # """str: Default account type for live operations if not specified per deployment (e.g., "MARGIN")."""
    max_concurrent_strategies: int
    """int: Maximum number of strategy instances that can run concurrently in live mode."""
    default_position_sizing_pct_capital: float
    """float: Default percentage of available capital to use for position sizing if not overridden by strategy."""
    global_risk_limit_pct_capital: float
    """float: Global risk limit as a percentage of total capital across all live strategies."""
    # is_testnet: bool = False # Removed, as this is per AccountConfig
    # """bool: Global flag for using testnet, can be overridden by AccountConfig."""
    session_cycle_interval_seconds: int = 60
    """int: Interval in seconds for the main orchestrator loop to re-evaluate and manage trading sessions."""

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
    """Optional[float]: Override for position sizing as a percentage of capital for this specific deployment."""
    max_loss_per_trade_pct: Optional[float] = None # This field was not in the original user's file but is good to have.
    """Optional[float]: Maximum allowable loss per trade as a percentage of capital for this deployment."""

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
    """bool: Whether this specific deployment is active."""
    strategy_id: str
    """str: Unique identifier for the strategy deployment (often links to the optimization run, e.g., "ema_macd_BTCUSDT_5m_context_20250101_120000")."""
    results_config_path: str
    """str: Path to the live_config.json file (generated by reporting) containing the optimized parameters for this deployment."""
    account_alias_to_use: str
    """str: The account_alias from AccountConfig to use for this deployment."""
    override_risk_settings: Optional[OverrideRiskSettings] = None
    """Optional[OverrideRiskSettings]: Specific risk setting overrides for this deployment."""

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
    """str: Logging level for live trading (e.g., "INFO", "DEBUG")."""
    log_to_file: bool = True
    """bool: Whether to log live trading events to a file."""
    log_filename_live: str = "live_trading_run.log"
    """str: Filename for live trading logs."""
    log_levels_by_module: Optional[Dict[str, str]] = None
    """Optional[Dict[str, str]]: Allows specifying different log levels for specific modules during live trading."""

@dataclass
class LiveFetchConfig:
    """
    Configuration for fetching data during live trading.
    """
    crypto_pairs: List[str]
    """List[str]: List of crypto pairs for which to fetch live data."""
    intervals: List[str]
    """List[str]: Time intervals for live data fetching/streams (e.g., ["1m", "5m"])."""
    limit_init_history: int
    """int: Number of klines to fetch for initial history if live data file is empty or new."""
    limit_per_fetch: int
    """int: Number of klines to fetch per periodic REST API call (if used for updates)."""
    max_retries: int
    """int: Maximum number of retries for API calls during live data fetching."""
    retry_backoff: float
    """float: Backoff factor for retries (e.g., 1.5 for exponential backoff)."""

    def __post_init__(self):
        if not self.crypto_pairs:
            # Allow empty if no live fetching is intended for certain runs,
            # but could be a warning if global live trading is on.
            pass
        if not self.intervals:
            # Allow empty for same reasons as crypto_pairs.
            pass
        if self.limit_init_history < 0: # 0 might be valid if no history desired
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
    """GlobalLiveSettings: Global settings applicable to all live trading."""
    strategy_deployments: List[StrategyDeployment]
    """List[StrategyDeployment]: List of specific strategy deployments for live trading."""
    live_fetch: LiveFetchConfig
    """LiveFetchConfig: Configuration for live data fetching."""
    live_logging: LiveLoggingConfig
    """LiveLoggingConfig: Logging configuration specific to live trading."""

@dataclass
class ApiKeys:
    """
    Container for API credentials, mapped by account alias.
    Keys are account aliases, values are tuples of (api_key, api_secret).
    This structure is populated by the loader from environment variables
    specified in AccountConfig.
    """
    credentials: Dict[str, Tuple[Optional[str], Optional[str]]] = field(default_factory=dict)
    """Dict[str, Tuple[Optional[str], Optional[str]]]: Stores API key and secret for each account_alias."""

@dataclass
class AppConfig:
    """
    Root configuration class for the entire application.
    """
    global_config: GlobalConfig
    """GlobalConfig: Global application settings."""
    data_config: DataConfig
    """DataConfig: Data source and fetching configurations."""
    strategies_config: StrategiesConfig
    """StrategiesConfig: Configurations for all defined trading strategies."""
    live_config: LiveConfig
    """LiveConfig: Configurations for live trading operations."""
    accounts_config: List[AccountConfig]
    """List[AccountConfig]: List of API account configurations."""
    api_keys: ApiKeys
    """ApiKeys: Container for API keys, populated based on accounts_config and environment variables."""
    project_root: str = field(init=False)
    """str: Absolute path to the project root directory (set by the loader)."""

    def __post_init__(self):
        # accounts_config can be empty if no live trading is intended.
        # Validation of uniqueness of account_alias is good.
        if self.accounts_config: # Only check if list is not empty
            aliases = [acc.account_alias for acc in self.accounts_config]
            if len(aliases) != len(set(aliases)):
                raise ValueError("AppConfig: account_alias in accounts_config must be unique.")
