import json
import logging
import os
import sys
import dataclasses
from dataclasses import fields, is_dataclass
from typing import Dict, Optional, Any, List, Union, Type, get_origin, get_args, Tuple
import typing # Explicit import
from pathlib import Path
from dotenv import load_dotenv

# Attempt to import definitions and logging_setup
try:
    from .definitions import (
        PathsConfig, LoggingConfig, SimulationDefaults, WfoSettings, OptunaSettings,
        GlobalConfig, SourceDetails, AssetsAndTimeframes, HistoricalPeriod,
        FetchingOptions, DataConfig, ParamDetail, StrategyParamsConfig, StrategiesConfig, # MODIFIED HERE
        AccountConfig, GlobalLiveSettings, OverrideRiskSettings, StrategyDeployment,
        LiveLoggingConfig, LiveFetchConfig, LiveConfig, ApiKeys, AppConfig, ExchangeSettings
    )
except ImportError:
    # This fallback is extensive and defined in the user's context.
    # For brevity, I'm not reproducing it here, but it would be included in a real execution.
    logging.error("CRITICAL: Failed to import dataclass definitions from .definitions. Ensure it's in the same directory or PYTHONPATH.")
    # Define minimal fallbacks or raise an error if definitions are absolutely critical for the loader's core logic
    raise

try:
    from src.utils.logging_setup import setup_logging
except ImportError:
    # Fallback setup_logging if the primary import fails
    def setup_logging(*args, **kwargs): # type: ignore
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - FALLBACK_SETUP - %(levelname)s - %(message)s')
        logging.getLogger(__name__).warning("Logging setup skipped due to import error in loader.py. Using basic fallback.")
        pass

logger = logging.getLogger(__name__)

def _load_json(file_path: str) -> Dict[str, Any]:
    """Loads a JSON file into a dictionary."""
    if not os.path.exists(file_path):
        logger.error(f"Configuration file not found: {file_path}")
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file: {file_path} - {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}")
        raise

def _is_optional_union(field_type: Any) -> bool:
    """Checks if a type hint is an Optional Union (Union[T, None])."""
    return get_origin(field_type) is Union and type(None) in get_args(field_type)

def _get_non_none_type_from_optional_union(field_type: Any) -> Any:
    """Extracts the non-None type from an Optional Union."""
    if _is_optional_union(field_type):
        args = get_args(field_type)
        return next((arg for arg in args if arg is not type(None)), type(None)) # Return type(None) if only NoneType found, though unusual for Optional[T]
    return field_type

def _create_dataclass_from_dict(dataclass_type: Type[Any], data: Dict[str, Any], config_file_path: str = "Unknown") -> Any:
    """
    Recursively creates a dataclass instance from a dictionary.
    Handles nested dataclasses, lists of dataclasses, and optional fields.
    """
    if not is_dataclass(dataclass_type):
        logger.debug(f"Type {dataclass_type} is not a dataclass. Returning raw data: {data}")
        return data

    field_info_map = {f.name: f for f in fields(dataclass_type)}
    init_kwargs: Dict[str, Any] = {}

    # Populate with defaults first, so they can be overridden by JSON data
    for field_name, field_obj in field_info_map.items():
        if field_obj.default is not dataclasses.MISSING:
            init_kwargs[field_name] = field_obj.default
        elif field_obj.default_factory is not dataclasses.MISSING:
            init_kwargs[field_name] = field_obj.default_factory()

    # Override with values from data
    for name_from_json, value_from_json in data.items():
        if name_from_json not in field_info_map:
            logger.debug(f"In {config_file_path} for {dataclass_type.__name__}: Key '{name_from_json}' in JSON data is not a field. Skipping.")
            continue

        field_obj = field_info_map[name_from_json]
        actual_field_type_for_conversion = _get_non_none_type_from_optional_union(field_obj.type)
        origin_type = get_origin(actual_field_type_for_conversion)
        type_args = get_args(actual_field_type_for_conversion)

        if value_from_json is None and _is_optional_union(field_obj.type):
            init_kwargs[name_from_json] = None
        elif is_dataclass(actual_field_type_for_conversion) and isinstance(value_from_json, dict):
            init_kwargs[name_from_json] = _create_dataclass_from_dict(actual_field_type_for_conversion, value_from_json, config_file_path)
        elif origin_type is list and type_args and is_dataclass(type_args[0]) and isinstance(value_from_json, list):
            list_item_dc_type = type_args[0]
            init_kwargs[name_from_json] = [_create_dataclass_from_dict(list_item_dc_type, item, config_file_path) for item in value_from_json if isinstance(item, dict)]
        elif origin_type is dict and type_args and len(type_args) == 2 and is_dataclass(type_args[1]) and isinstance(value_from_json, dict):
            dict_value_dc_type = type_args[1]
            processed_dict = {}
            for k, v_item in value_from_json.items():
                if isinstance(v_item, dict):
                    processed_dict[k] = _create_dataclass_from_dict(dict_value_dc_type, v_item, config_file_path)
                else: # Handle cases where dict value might not be another dataclass, e.g. Dict[str, Any] or Dict[str, float]
                    processed_dict[k] = v_item
            init_kwargs[name_from_json] = processed_dict
        elif origin_type is dict and value_from_json is None and _is_optional_union(field_obj.type): # For Optional[Dict[...]]
            init_kwargs[name_from_json] = None
        elif origin_type is dict and isinstance(value_from_json, dict): # For Dict[str, Any] or Dict[str, primitive]
             init_kwargs[name_from_json] = value_from_json
        else:
            try:
                # Basic type conversion if needed (int, float, bool, str)
                if actual_field_type_for_conversion == int and not isinstance(value_from_json, bool): # bool is a subclass of int
                    init_kwargs[name_from_json] = int(value_from_json)
                elif actual_field_type_for_conversion == float:
                    init_kwargs[name_from_json] = float(value_from_json)
                elif actual_field_type_for_conversion == bool:
                    init_kwargs[name_from_json] = bool(value_from_json)
                elif actual_field_type_for_conversion == str:
                    init_kwargs[name_from_json] = str(value_from_json)
                else: # If it's already the correct type or a complex non-dataclass type handled by typing
                    init_kwargs[name_from_json] = value_from_json
            except (ValueError, TypeError) as e_conv:
                logger.warning(f"In {config_file_path} for {dataclass_type.__name__}: Could not convert value '{value_from_json}' "
                               f"for field '{name_from_json}' to {actual_field_type_for_conversion}. Using original value. Error: {e_conv}")
                init_kwargs[name_from_json] = value_from_json
    try:
        return dataclass_type(**init_kwargs)
    except Exception as e_init:
        logger.error(f"Error instantiating {dataclass_type.__name__} from {config_file_path} with keys {list(init_kwargs.keys())}: {e_init}", exc_info=True)
        raise

def _validate_app_config(app_config: AppConfig, project_root_path: Path):
    """Validates the fully loaded AppConfig instance."""
    logger.info("Starting AppConfig validation...")
    errors: List[str] = []

    # OptunaSettings Validation
    if hasattr(app_config.global_config, 'optuna_settings'):
        opt_settings = app_config.global_config.optuna_settings
        supported_samplers = ["TPESampler", "NSGAIISampler", "CmaEsSampler", "RandomSampler"] # Add more as supported
        supported_pruners = ["MedianPruner", "HyperbandPruner", "NopPruner", "PatientPruner", "SuccessiveHalvingPruner"] # Add more
        if not opt_settings.sampler_name or opt_settings.sampler_name not in supported_samplers:
            errors.append(f"OptunaSettings: Invalid sampler_name '{opt_settings.sampler_name}'. Supported: {supported_samplers}")
        if not opt_settings.pruner_name or opt_settings.pruner_name not in supported_pruners:
            errors.append(f"OptunaSettings: Invalid pruner_name '{opt_settings.pruner_name}'. Supported: {supported_pruners}")
        if opt_settings.sampler_params is not None and not isinstance(opt_settings.sampler_params, dict):
            errors.append("OptunaSettings: sampler_params must be a dictionary if provided.")
        if opt_settings.pruner_params is not None and not isinstance(opt_settings.pruner_params, dict):
            errors.append("OptunaSettings: pruner_params must be a dictionary if provided.")
        # __post_init__ in OptunaSettings handles objectives_names/directions and pareto_selection_weights

    # PathsConfig Validation (example: check if a critical input path exists)
    if hasattr(app_config.global_config, 'paths'):
        paths = app_config.global_config.paths
        # Example: if data_historical_processed_enriched is essential for input
        # enriched_path = Path(paths.data_historical_processed_enriched)
        # if not enriched_path.exists():
        #     errors.append(f"PathsConfig: data_historical_processed_enriched path does not exist: {enriched_path}")
        pass # Output paths are created by the loader, input paths checked by specific modules.

    # SimulationDefaults Validation
    if hasattr(app_config.global_config, 'simulation_defaults'):
        sim_defs = app_config.global_config.simulation_defaults
        if not sim_defs.initial_capital > 0:
            errors.append(f"SimulationDefaults: initial_capital ({sim_defs.initial_capital}) must be > 0.")
        # Removed transaction_fee_pct and slippage_pct as they are part of nested slippage_config
        # if not (0 <= sim_defs.transaction_fee_pct < 1):
        #     errors.append(f"SimulationDefaults: transaction_fee_pct ({sim_defs.transaction_fee_pct}) must be [0, 1).")
        # if not (0 <= sim_defs.slippage_pct < 1):
        #     errors.append(f"SimulationDefaults: slippage_pct ({sim_defs.slippage_pct}) must be [0, 1).")
        if not sim_defs.margin_leverage >= 1:
            errors.append(f"SimulationDefaults: margin_leverage ({sim_defs.margin_leverage}) must be >= 1.")

    # WfoSettings Validation
    if hasattr(app_config.global_config, 'wfo_settings'):
        wfo = app_config.global_config.wfo_settings
        if not wfo.n_splits >= 1:
            errors.append(f"WfoSettings: n_splits ({wfo.n_splits}) must be >= 1.")
        # Removed oos_percent as it's not in the current WfoSettings definition
        # if not (0 < wfo.oos_percent < 100):
        #     errors.append(f"WfoSettings: oos_percent ({wfo.oos_percent}) must be between 0 and 100 (exclusive).")

    # StrategiesConfig and ParamDetail Validation
    if hasattr(app_config.strategies_config, 'strategies'):
        for strat_name, strat_params_config in app_config.strategies_config.strategies.items(): # Changed variable name
            if not strat_params_config.script_reference:
                errors.append(f"Strategy '{strat_name}': script_reference cannot be empty.")
            if not strat_params_config.class_name:
                errors.append(f"Strategy '{strat_name}': class_name cannot be empty.")
            for param_name, p_detail in strat_params_config.params_space.items():
                # ParamDetail's __post_init__ handles internal consistency.
                if p_detail.step is not None and p_detail.step <= 0 and p_detail.type in ["int", "float"]:
                     errors.append(f"Strategy '{strat_name}', Param '{param_name}': step ({p_detail.step}) must be positive if provided.")

    # AccountConfig Validation
    if hasattr(app_config, 'accounts_config'):
        valid_exchanges = ["binance"] # Add more as supported
        valid_account_types = ["MARGIN", "ISOLATED_MARGIN", "SPOT", "FUTURES"] # Add more
        for acc_idx, acc_config in enumerate(app_config.accounts_config):
            if not acc_config.exchange or acc_config.exchange.lower() not in valid_exchanges:
                errors.append(f"AccountConfig[{acc_idx}] (alias: {acc_config.account_alias}): Invalid exchange '{acc_config.exchange}'. Supported: {valid_exchanges}")
            if not acc_config.account_type or acc_config.account_type.upper() not in valid_account_types:
                errors.append(f"AccountConfig[{acc_idx}] (alias: {acc_config.account_alias}): Invalid account_type '{acc_config.account_type}'. Supported: {valid_account_types}")

            # API key env var checks (only if env var name is provided)
            if acc_config.api_key_env_var:
                if not os.getenv(acc_config.api_key_env_var):
                    errors.append(f"AccountConfig[{acc_idx}] (alias: {acc_config.account_alias}): Environment variable '{acc_config.api_key_env_var}' for API key is not set or is empty.")
            # else: # If api_key_env_var is None or empty, we assume direct key provision or no key needed for this account type
            #    logger.debug(f"AccountConfig[{acc_idx}] (alias: {acc_config.account_alias}): api_key_env_var not specified. Assuming direct key provision or no key needed.")

            if acc_config.api_secret_env_var:
                if not os.getenv(acc_config.api_secret_env_var):
                     errors.append(f"AccountConfig[{acc_idx}] (alias: {acc_config.account_alias}): Environment variable '{acc_config.api_secret_env_var}' for API secret is not set or is empty.")
            # else:
            #    logger.debug(f"AccountConfig[{acc_idx}] (alias: {acc_config.account_alias}): api_secret_env_var not specified.")


    # StrategyDeployment Validation
    if hasattr(app_config.live_config, 'strategy_deployments'):
        account_aliases_defined = [acc.account_alias for acc in app_config.accounts_config]
        for dep_idx, deployment in enumerate(app_config.live_config.strategy_deployments):
            if deployment.active:
                if not deployment.strategy_id:
                    errors.append(f"StrategyDeployment[{dep_idx}]: strategy_id cannot be empty for an active deployment.")
                results_path = project_root_path / deployment.results_config_path
                if not results_path.is_file():
                    errors.append(f"StrategyDeployment[{dep_idx}] (ID: {deployment.strategy_id}): results_config_path '{results_path}' does not point to a valid file.")
                if not deployment.account_alias_to_use:
                     errors.append(f"StrategyDeployment[{dep_idx}] (ID: {deployment.strategy_id}): account_alias_to_use cannot be empty.")
                elif deployment.account_alias_to_use not in account_aliases_defined:
                    errors.append(f"StrategyDeployment[{dep_idx}] (ID: {deployment.strategy_id}): account_alias_to_use '{deployment.account_alias_to_use}' does not match any defined account_alias in accounts_config.")

    if errors:
        error_summary = "\n - ".join(["Configuration validation failed:"] + errors)
        logger.error(error_summary)
        raise ValueError(error_summary)

    logger.info("AppConfig validation successful.")


def load_all_configs(project_root: Optional[str] = None) -> AppConfig:
    """
    Loads all JSON configuration files, environment variables for API keys,
    populates dataclass instances, and validates the final AppConfig.
    """
    if project_root is None:
        try:
            # Assumes this script (loader.py) is in src/config/
            project_root_path = Path(__file__).resolve().parent.parent.parent
        except NameError: # Fallback if __file__ is not defined (e.g. interactive)
            project_root_path = Path('.').resolve()
            logger.info(f"__file__ not defined, project root detected as current directory: {project_root_path}")
    else:
        project_root_path = Path(project_root).resolve()

    logger.info(f"Project root determined as: {project_root_path}")

    # Load .env file from project root
    env_path = project_root_path / '.env'
    if env_path.is_file():
        loaded_env = load_dotenv(dotenv_path=env_path, verbose=True)
        logger.info(f"Loaded .env file from {env_path}" if loaded_env else f".env file at {env_path} found but python-dotenv could not load it.")
    else:
        logger.warning(f".env file not found at {env_path}. API keys must be set directly in the environment if not specified in config_accounts.json with direct values (not recommended).")

    config_dir = project_root_path / 'config'
    global_config_path = str(config_dir / 'config_global.json')
    data_config_path = str(config_dir / 'config_data.json')
    strategies_config_path = str(config_dir / 'config_strategies.json')
    live_config_path = str(config_dir / 'config_live.json')
    accounts_config_path = str(config_dir / 'config_accounts.json') # New accounts config file

    logger.info(f"Loading global_config from: {global_config_path}")
    global_config_dict = _load_json(global_config_path)
    global_cfg = _create_dataclass_from_dict(GlobalConfig, global_config_dict, global_config_path)

    logger.info(f"Loading data_config from: {data_config_path}")
    data_config_dict = _load_json(data_config_path)
    data_cfg = _create_dataclass_from_dict(DataConfig, data_config_dict, data_config_path)

    logger.info(f"Loading strategies_config from: {strategies_config_path}")
    strategies_config_dict_raw = _load_json(strategies_config_path) # This is Dict[str, Dict]
    # Wrap it for _create_dataclass_from_dict expects a dict that maps to StrategyParamsConfig fields
    strategies_cfg = _create_dataclass_from_dict(StrategiesConfig, {"strategies": strategies_config_dict_raw}, strategies_config_path)

    logger.info(f"Loading live_config from: {live_config_path}")
    live_config_dict = _load_json(live_config_path)
    live_cfg = _create_dataclass_from_dict(LiveConfig, live_config_dict, live_config_path)

    logger.info(f"Loading accounts_config from: {accounts_config_path}")
    accounts_config_list_raw = _load_json(accounts_config_path) # This is List[Dict]
    if not isinstance(accounts_config_list_raw, list):
        raise ValueError(f"{accounts_config_path} should contain a JSON list of account configurations.")
    accounts_cfg_list: List[AccountConfig] = [
        _create_dataclass_from_dict(AccountConfig, acc_dict, accounts_config_path)
        for acc_dict in accounts_config_list_raw
    ]

    # Populate ApiKeys based on loaded AccountConfig and environment variables
    api_key_credentials: Dict[str, Tuple[Optional[str], Optional[str]]] = {} # MODIFIED HERE: Tuple values can be Optional
    for acc_conf in accounts_cfg_list:
        key_env_var = acc_conf.api_key_env_var
        secret_env_var = acc_conf.api_secret_env_var
        key: Optional[str] = None
        secret: Optional[str] = None

        if key_env_var:
            key = os.getenv(key_env_var)
            if key:
                 logger.info(f"Successfully loaded API key for account alias '{acc_conf.account_alias}' from env var '{key_env_var}'.")
            else:
                logger.warning(f"Environment variable '{key_env_var}' for API key of account '{acc_conf.account_alias}' not found or empty.")
        else:
            logger.debug(f"api_key_env_var not specified for account '{acc_conf.account_alias}'.")

        if secret_env_var:
            secret = os.getenv(secret_env_var)
            if secret:
                logger.info(f"Successfully loaded API secret for account alias '{acc_conf.account_alias}' from env var '{secret_env_var}'.")
            else:
                logger.warning(f"Environment variable '{secret_env_var}' for API secret of account '{acc_conf.account_alias}' not found or empty.")
        else:
            logger.debug(f"api_secret_env_var not specified for account '{acc_conf.account_alias}'.")

        # Store even if one or both are None, validation will handle if they are required later for an operation
        api_key_credentials[acc_conf.account_alias] = (key, secret)

    api_keys_instance = ApiKeys(credentials=api_key_credentials)


    # Resolve paths in PathsConfig to be absolute
    if hasattr(global_cfg, 'paths') and global_cfg.paths:
        logger.info("Resolving paths in PathsConfig to absolute paths...")
        for path_field_obj in fields(global_cfg.paths):
            field_name = path_field_obj.name
            relative_path_val = getattr(global_cfg.paths, field_name)
            if isinstance(relative_path_val, str):
                absolute_path = (project_root_path / relative_path_val).resolve()
                setattr(global_cfg.paths, field_name, str(absolute_path))
                logger.debug(f"Resolved path '{field_name}': {relative_path_val} -> {absolute_path}")
                # Create directories if they are typical output dirs
                if any(d_name in field_name.lower() for d_name in ["log", "data", "results", "state"]):
                    try:
                        absolute_path.mkdir(parents=True, exist_ok=True)
                    except OSError as e_mkdir:
                        logger.error(f"Failed to create directory {absolute_path}: {e_mkdir}")
            else:
                logger.warning(f"Path value for field '{field_name}' in PathsConfig is not a string ('{relative_path_val}'). Skipping resolution.")
    else:
        logger.warning("GlobalConfig.paths is missing or not a PathsConfig object. Path resolution skipped.")


    # Setup application-wide logging using the loaded configuration
    log_conf_to_use: Union[LoggingConfig, LiveLoggingConfig, None] = None
    log_dir_final: str = str(project_root_path / 'logs' / 'default_loader_logs')
    log_filename_final: str = 'app_run_default.log'
    root_log_level_final = logging.INFO

    if hasattr(global_cfg, 'logging') and global_cfg.logging:
        log_conf_to_use = global_cfg.logging
        if hasattr(global_cfg.paths, 'logs_backtest_optimization') and global_cfg.paths.logs_backtest_optimization:
             log_dir_final = global_cfg.paths.logs_backtest_optimization
        log_filename_final = getattr(log_conf_to_use, 'log_filename_global', log_filename_final)
        root_log_level_str = getattr(log_conf_to_use, 'level', 'INFO').upper()
        root_log_level_final = getattr(logging, root_log_level_str, logging.INFO)

    if log_conf_to_use:
        try:
            Path(log_dir_final).mkdir(parents=True, exist_ok=True)
            setup_logging(log_config=log_conf_to_use, log_dir=log_dir_final, log_filename=log_filename_final, root_level=root_log_level_final)
            logger.info(f"Application logging configured using {'LiveLoggingConfig' if isinstance(log_conf_to_use, LiveLoggingConfig) else 'LoggingConfig'}. Log dir: {log_dir_final}, File: {log_filename_final}")
        except Exception as e_log_setup:
            logging.basicConfig(level=logging.WARNING, format='%(asctime)s - LOADER_LOG_SETUP_FAIL - %(levelname)s - %(message)s')
            logger.error(f"Failed during application-wide logging setup in loader: {e_log_setup}. Basic logging active.", exc_info=True)
    else:
        logging.basicConfig(level=root_log_level_final, format='%(asctime)s - LOADER_BASIC_CFG - %(levelname)s - %(message)s')
        logger.warning("LoggingConfig or LiveLoggingConfig section missing or not applicable. Using basicConfig for application logging.")

    # Create ExchangeSettings instance (assuming default values or loaded from a specific file if needed)
    # For now, using default values as per definitions.py
    # If exchange_settings.json existed, it would be loaded similarly to other configs.
    exchange_settings_instance = ExchangeSettings()
    # If you add config_exchange.json:
    # exchange_config_path = str(config_dir / 'config_exchange.json')
    # if Path(exchange_config_path).exists():
    #     exchange_config_dict = _load_json(exchange_config_path)
    #     exchange_settings_instance = _create_dataclass_from_dict(ExchangeSettings, exchange_config_dict, exchange_config_path)
    # else:
    #     logger.info("config_exchange.json not found, using default ExchangeSettings.")


    app_config = AppConfig(
        global_config=global_cfg,
        data_config=data_cfg,
        strategies_config=strategies_cfg,
        live_config=live_cfg,
        accounts_config=accounts_cfg_list,
        api_keys=api_keys_instance,
        exchange_settings=exchange_settings_instance # Pass the instance
    )
    app_config.project_root = str(project_root_path)

    try:
        _validate_app_config(app_config, project_root_path)
    except ValueError as val_err:
        logger.critical(f"AppConfig validation failed: {val_err}", exc_info=True)
        raise

    logger.info(f"All configurations loaded and validated successfully for project: {app_config.global_config.project_name}")
    return app_config

def load_strategy_config_by_name(strategy_name: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Charge la configuration pour une stratégie spécifique par son nom depuis config_strategies.json.
    Retourne un dictionnaire des paramètres de la stratégie.
    """
    if project_root is None:
        project_root_path = Path(__file__).resolve().parent.parent.parent
    else:
        project_root_path = Path(project_root).resolve()

    strategies_config_path = project_root_path / 'config' / 'config_strategies.json'
    
    if not strategies_config_path.exists():
        logger.error(f"Fichier config_strategies.json non trouvé à {strategies_config_path}")
        raise FileNotFoundError(f"Fichier config_strategies.json non trouvé à {strategies_config_path}")

    all_strategies_dict = _load_json(str(strategies_config_path))
    
    strategy_config_data = all_strategies_dict.get(strategy_name)
    
    if strategy_config_data is None:
        logger.error(f"Configuration pour la stratégie '{strategy_name}' non trouvée dans {strategies_config_path}")
        raise ValueError(f"Configuration pour la stratégie '{strategy_name}' non trouvée.")
        
    if not isinstance(strategy_config_data, dict):
        logger.error(f"La configuration pour la stratégie '{strategy_name}' n'est pas un dictionnaire valide.")
        raise TypeError(f"La configuration pour la stratégie '{strategy_name}' doit être un dictionnaire.")

    # Pas besoin de convertir en dataclass ici, car la fonction est censée retourner un dict
    # pour une utilisation plus flexible par l'appelant (par exemple, BaseStrategy).
    # Si une instance de StrategyParamsConfig était nécessaire, on pourrait faire:
    # return _create_dataclass_from_dict(StrategyParamsConfig, strategy_config_data, str(strategies_config_path))
    return strategy_config_data

def load_exchange_config(exchange_info_file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Charge les informations de l'exchange (par exemple, filtres de symboles, précisions)
    depuis un fichier JSON.
    """
    if exchange_info_file_path is None:
        # Fallback to a default path if not provided
        try:
            default_root = Path(__file__).resolve().parent.parent.parent
        except NameError:
            default_root = Path(".").resolve()
        exchange_info_file_path = default_root / "config" / "binance_exchange_info.json"
        logger.info(f"exchange_info_file_path non fourni, utilisation du chemin par défaut: {exchange_info_file_path}")

    path_obj = Path(exchange_info_file_path)
    if not path_obj.exists() or not path_obj.is_file():
        logger.error(f"Fichier d'informations de l'exchange non trouvé: {path_obj}")
        # Retourner un dict vide ou lever une exception selon la criticité
        return {"symbols": []} # Exemple de structure de base attendue par get_pair_config_for_symbol
    
    try:
        with open(path_obj, 'r', encoding='utf-8') as f:
            exchange_info = json.load(f)
        if not isinstance(exchange_info, dict):
            logger.error(f"Le contenu de {path_obj} n'est pas un dictionnaire JSON valide.")
            return {"symbols": []}
        logger.info(f"Informations de l'exchange chargées depuis {path_obj}")
        return exchange_info
    except json.JSONDecodeError as e:
        logger.error(f"Erreur de décodage JSON depuis {path_obj}: {e}")
        return {"symbols": []}
    except Exception as e_load:
        logger.error(f"Erreur inattendue lors du chargement de {path_obj}: {e_load}")
        return {"symbols": []}

