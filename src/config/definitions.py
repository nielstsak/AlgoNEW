# src/config/definitions.py
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union, Tuple # Ensure Tuple is imported
import pandas as pd # Added for pd.to_datetime

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
    These are used by the new BacktestSimulator.
    """
    initial_capital: float
    margin_leverage: int 
    # transaction_fee_pct: float # Replaced by trading_fee_bps
    # slippage_pct: float # Replaced by slippage_config dictionary
    trading_fee_bps: float # Trading fee in basis points (e.g., 7 for 0.07%)
    
    # Slippage configuration moved to a nested structure for more detail
    slippage_method: str = "percentage" # e.g., "percentage", "volume_based"
    slippage_percentage_max_bps: Optional[float] = None # Max slippage in BPS for percentage method
    slippage_volume_factor: Optional[float] = None # For volume-based slippage
    slippage_volatility_factor: Optional[float] = None # For volatility-based slippage
    slippage_min_bps: Optional[float] = 0.0 # Minimum slippage to apply in BPS
    slippage_max_bps: Optional[float] = 100.0 # Cap for slippage in BPS

    is_futures_trading: bool = False # Added to distinguish spot/futures
    backtest_verbosity: int = 1 # Logging verbosity for backtests
    risk_free_rate: float = 0.0 # For Sharpe ratio, etc.
    run_name_prefix: Optional[str] = "opt" # Prefix for WFO run IDs

    # The new BacktestSimulator uses a slippage_config dictionary.
    # We can construct it from these fields or expect a direct dict.
    # For simplicity, keeping individual fields here and simulator can adapt or expect a dict.
    # Let's make it a direct dict for clarity with the new simulator:
    slippage_config: Dict[str, Any] = field(default_factory=lambda: {
        "method": "percentage",
        "percentage_max_bps": 5.0, # Example default
        "volume_factor": 0.1,      # Example default
        "volatility_factor": 0.1,  # Example default
        "min_slippage_bps": 0.0,   # Example default
        "max_slippage_bps": 50.0   # Example default
    })


    def __post_init__(self):
        if not isinstance(self.margin_leverage, int) or self.margin_leverage < 1:
            if isinstance(self.margin_leverage, float) and self.margin_leverage.is_integer() and self.margin_leverage >= 1:
                self.margin_leverage = int(self.margin_leverage)
            else:
                raise ValueError("SimulationDefaults: margin_leverage must be an integer >= 1.")
        if self.trading_fee_bps < 0:
            raise ValueError("SimulationDefaults: trading_fee_bps cannot be negative.")


@dataclass
class WfoSettings: # Corrected case: WfoSettings
    """
    Settings for Walk-Forward Optimization (WFO).
    Updated for the new WFOGenerator logic.
    """
    n_splits: int
    oos_period_days: int # Duration of the Out-of-Sample period in days
    min_is_period_days: int # Minimum duration of an In-Sample period in days
    fold_type: str # e.g., "expanding" (fixed OOS end, IS start moves back)
    metric_to_optimize: str
    optimization_direction: str # "maximize" or "minimize"
    top_n_trials_for_oos_validation: int = 10 # Number of best IS trials to validate OOS
    top_n_trials_to_report_oos: int = 3 # Number of best OOS trials to save detailed logs for

    def __post_init__(self):
        if self.n_splits <= 0:
            raise ValueError("WfoSettings: n_splits must be positive.")
        if self.oos_period_days <= 0:
            raise ValueError("WfoSettings: oos_period_days must be positive.")
        if self.min_is_period_days <= 0:
            raise ValueError("WfoSettings: min_is_period_days must be positive.")
        if self.fold_type not in ["expanding"]: # Add other types if implemented
            raise ValueError(f"WfoSettings: fold_type '{self.fold_type}' is not supported.")
        if self.optimization_direction not in ["maximize", "minimize"]:
            raise ValueError("WfoSettings: optimization_direction must be 'maximize' or 'minimize'.")
        if self.top_n_trials_for_oos_validation <= 0:
            raise ValueError("WfoSettings: top_n_trials_for_oos_validation must be positive.")
        if self.top_n_trials_to_report_oos <= 0:
            raise ValueError("WfoSettings: top_n_trials_to_report_oos must be positive.")
        if self.top_n_trials_to_report_oos > self.top_n_trials_for_oos_validation:
            raise ValueError("WfoSettings: top_n_trials_to_report_oos cannot be greater than top_n_trials_for_oos_validation.")


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
    objectives_names: List[str] = field(default_factory=lambda: ["Total Net PnL USDC", "Sharpe Ratio"]) # Example, should match PerformanceCalculator metrics
    objectives_directions: List[str] = field(default_factory=lambda: ["maximize", "maximize"])
    pareto_selection_strategy: Optional[str] = "PNL_MAX" # Example
    pareto_selection_weights: Optional[Dict[str, float]] = None
    pareto_selection_pnl_threshold: Optional[float] = None
    # n_best_for_oos: int = 10 # Moved to WfoSettings as top_n_trials_for_oos_validation
    
    default_profile_to_activate: Optional[str] = None
    sampler_pruner_profiles: Optional[Dict[str, Dict[str, Any]]] = field(default_factory=dict)


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
        # if self.n_best_for_oos <= 0: # Moved
        #     raise ValueError("OptunaSettings: n_best_for_oos must be a positive integer.")
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
    wfo_settings: WfoSettings # Corrected case
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
            # Basic check, more robust date parsing would be better if formats vary
            pd_start_date = pd.to_datetime(self.start_date, errors='raise')
            if self.end_date:
                pd_end_date = pd.to_datetime(self.end_date, errors='raise')
                if pd_start_date > pd_end_date: 
                    raise ValueError(f"HistoricalPeriod: start_date ({self.start_date}) cannot be after end_date ({self.end_date}).")
        except Exception as e: 
             raise ValueError(f"HistoricalPeriod: start_date '{self.start_date}' and/or end_date '{self.end_date}' are not valid date strings. Error: {e}")


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
    type: str # "int", "float", "categorical"
    default: Optional[Any] = None # Added default value
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    step: Optional[Union[float, int]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False # For float parameters, suggest in log scale

    def __post_init__(self):
        if self.type not in ["int", "float", "categorical"]:
            raise ValueError(f"ParamDetail: Invalid type '{self.type}'. Must be 'int', 'float', or 'categorical'.")
        if self.type == "categorical":
            if self.choices is None or not isinstance(self.choices, list) or not self.choices:
                raise ValueError("ParamDetail: 'choices' must be a non-empty list for categorical parameters.")
            if self.default is not None and self.default not in self.choices:
                raise ValueError(f"ParamDetail: default value '{self.default}' not in choices {self.choices} for categorical param.")
        elif self.type in ["int", "float"]: # Common checks for int and float
            if self.low is None or self.high is None:
                raise ValueError(f"ParamDetail: 'low' and 'high' must be provided for {self.type} parameters.")
            if not isinstance(self.low, (int, float)) or not isinstance(self.high, (int, float)):
                raise ValueError(f"ParamDetail: 'low' and 'high' must be numeric for {self.type} parameters.")
            if self.low > self.high:
                raise ValueError(f"ParamDetail: 'low' ({self.low}) cannot be greater than 'high' ({self.high}).")
            if self.default is not None:
                if not (self.low <= self.default <= self.high):
                    raise ValueError(f"ParamDetail: default value '{self.default}' out of range [{self.low}, {self.high}].")
            if self.step is not None:
                if not isinstance(self.step, (int, float)):
                    raise ValueError(f"ParamDetail: 'step' must be numeric if provided for {self.type} parameters.")
                if self.step <= 0:
                    raise ValueError(f"ParamDetail: 'step' ({self.step}) must be positive if provided.")
            if self.type == "int": # Specific checks for int
                if self.default is not None and not isinstance(self.default, int):
                     if isinstance(self.default, float) and self.default.is_integer(): self.default = int(self.default)
                     else: raise ValueError(f"ParamDetail: default value '{self.default}' must be an integer for type 'int'.")
                if not isinstance(self.low, int):
                     if isinstance(self.low, float) and self.low.is_integer(): self.low = int(self.low)
                     else: raise ValueError(f"ParamDetail: 'low' ({self.low}) must be an integer for type 'int'.")
                if not isinstance(self.high, int):
                     if isinstance(self.high, float) and self.high.is_integer(): self.high = int(self.high)
                     else: raise ValueError(f"ParamDetail: 'high' ({self.high}) must be an integer for type 'int'.")
                if self.step is not None and not isinstance(self.step, int):
                     if isinstance(self.step, float) and self.step.is_integer(): self.step = int(self.step)
                     else: raise ValueError(f"ParamDetail: 'step' ({self.step}) must be an integer for 'int' type parameters.")
        if self.type == "float" and self.log_scale and self.low <= 0:
            raise ValueError("ParamDetail: 'low' must be > 0 for float parameters with log_scale=True.")


@dataclass
class StrategyParams: # Renamed from StrategyParams to avoid confusion with strategy instance params
    """
    Configuration for a specific trading strategy, including its parameter space for optimization
    and default parameters.
    """
    active_for_optimization: bool
    script_reference: str # e.g., "src.strategies.ma_crossover_strategy"
    class_name: str # e.g., "MACrossoverStrategy"
    default_params: Dict[str, Any] = field(default_factory=dict) # Default parameters for the strategy
    params_space: Dict[str, ParamDetail] = field(default_factory=dict) # Optuna optimization space

@dataclass
class StrategiesConfig:
    """
    Container for configurations of multiple trading strategies.
    The keys are strategy names (e.g., "MACross1").
    """
    strategies: Dict[str, StrategyParams]


@dataclass
class AccountConfig:
    """
    Configuration for a single API account.
    """
    account_alias: str
    exchange: str
    account_type: str # e.g., "SPOT", "MARGIN", "FUTURES_USDT"
    is_testnet: bool
    api_key_env_var: Optional[str] = None # Made optional if key/secret are directly in api_keys
    api_secret_env_var: Optional[str] = None # Made optional

    def __post_init__(self):
        if not self.account_alias.strip():
            raise ValueError("AccountConfig: account_alias cannot be empty.")
        if not self.exchange.strip():
            raise ValueError(f"AccountConfig (alias: {self.account_alias}): exchange cannot be empty.")
        if not self.account_type.strip():
            raise ValueError(f"AccountConfig (alias: {self.account_alias}): account_type cannot be empty.")
        # api_key_env_var and api_secret_env_var can be None if keys are directly provided elsewhere

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
    max_loss_per_trade_pct: Optional[float] = None # Example, can add more

    def __post_init__(self):
        if self.position_sizing_pct_capital is not None and not (0 < self.position_sizing_pct_capital <= 1):
            raise ValueError("OverrideRiskSettings: position_sizing_pct_capital must be between 0 (exclusive) and 1 (inclusive) if provided.")
        if self.max_loss_per_trade_pct is not None and not (0 < self.max_loss_per_trade_pct <= 1): # Example validation
            raise ValueError("OverrideRiskSettings: max_loss_per_trade_pct must be between 0 (exclusive) and 1 (inclusive) if provided.")

@dataclass
class StrategyDeployment:
    """
    Configuration for deploying a specific optimized strategy to live trading.
    """
    active: bool
    strategy_id: str # User-defined ID for this deployment instance
    results_config_path: str # Path to the JSON file from WFO (e.g. wfo_strategy_pair_summary.json)
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
    log_filename_live: str = "live_trading_run.log" # Default filename
    log_levels_by_module: Optional[Dict[str, str]] = None

@dataclass
class LiveFetchConfig:
    """
    Configuration for fetching data during live trading.
    """
    crypto_pairs: List[str] # Pairs needed for ALL deployed strategies
    intervals: List[str]    # Intervals needed
    limit_init_history: int # Number of candles for initial history load
    limit_per_fetch: int    # Number of candles per subsequent fetch
    max_retries: int = 3
    retry_backoff: float = 5.0 # Seconds

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
    strategy_deployments: List[StrategyDeployment] # List of strategies to deploy
    live_fetch: LiveFetchConfig
    live_logging: LiveLoggingConfig


@dataclass
class ApiKeys:
    """
    Container for API credentials, mapped by account alias.
    Values are (api_key, api_secret) tuples.
    """
    credentials: Dict[str, Tuple[Optional[str], Optional[str]]] = field(default_factory=dict)


@dataclass
class ExchangeSettings: # Added for completeness, as used in ResultsAnalyzer and ObjectiveEvaluator
    """
    Settings related to a specific exchange.
    """
    exchange_name: str = "binance" # Example
    exchange_info_file_path: str = "config/binance_exchange_info.json" # Path to cached exchange info
    # trading_fee_bps: float = 7.0 # Moved to SimulationDefaults as it's simulation-specific
    # slippage_config: Dict[str, Any] = field(default_factory=lambda: {...}) # Moved to SimulationDefaults

@dataclass
class AppConfig:
    """
    Root configuration class for the entire application.
    """
    global_config: GlobalConfig
    data_config: DataConfig
    strategies_config: StrategiesConfig
    live_config: LiveConfig
    accounts_config: List[AccountConfig] # List of configured accounts
    api_keys: ApiKeys # API keys mapped by account_alias
    exchange_settings: ExchangeSettings # Added
    project_root: Optional[str] = None # Can be set dynamically at runtime

    def __post_init__(self):
        if self.accounts_config:
            aliases = [acc.account_alias for acc in self.accounts_config]
            if len(aliases) != len(set(aliases)):
                raise ValueError("AppConfig: account_alias in accounts_config must be unique.")
        
        # Example: Set project_root dynamically if not provided
        if self.project_root is None:
            try:
                # Attempt to set project_root based on the location of this file or a known structure
                # This is a placeholder, actual logic might be more complex
                import os
                # Assuming this file is within the project structure, e.g., src/config/definitions.py
                # Then project_root would be two levels up from src/config
                # This is highly dependent on your project structure and where this AppConfig is loaded.
                # For now, let's keep it simple or require it to be set by the loader.
                # self.project_root = str(Path(__file__).resolve().parents[2])
                pass # Requires more context to set robustly
            except Exception:
                pass # Silently ignore if cannot determine, or log a warning

