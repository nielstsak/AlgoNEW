# src/utils/run_utils.py
import dataclasses
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

from src.config.definitions import AppConfig, OptunaConfig, StrategyConfig, WFOConfig
from src.utils.file_utils import ensure_directory_exists, get_project_root
from src.utils.time_utils import get_kst_time_now_str_filename_friendly

logger = logging.getLogger(__name__)

T = TypeVar("T")


def save_orchestration_config(app_config: AppConfig, run_id: str, output_dir: Path) -> Path:
    """
    Saves the relevant parts of the AppConfig for orchestration reproducibility.
    Excludes non-serializable fields like loggers.
    Converts Path objects to strings.
    """
    ensure_directory_exists(output_dir)
    config_path = output_dir / f"orchestration_config_{run_id}.json"

    def to_serializable_dict(obj: Any) -> Any:
        """
        Recursively converts an object to a serializable dictionary.
        Excludes logger fields and converts Path objects to strings.
        """
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            result = {}
            for f in dataclasses.fields(obj):
                value = getattr(obj, f.name)
                # Exclude logger instances by field name 'logger'
                if f.name == "logger" and isinstance(value, logging.Logger):
                    continue
                result[f.name] = to_serializable_dict(value)
            return result
        elif isinstance(obj, list):
            return [to_serializable_dict(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: to_serializable_dict(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        # Add handling for other non-serializable types if necessary
        return obj

    try:
        # Use the custom conversion function
        config_dict = to_serializable_dict(app_config)

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        logger.info(f"Orchestration config saved to {config_path}")
        return config_path
    except TypeError as e:
        logger.error(f"Failed to serialize app_config: {e}")
        logger.error(
            "Consider adding more types to 'to_serializable_dict' or checking AppConfig fields."
        )
        raise


def load_orchestration_config(config_path: Path, expected_type: Type[T] = AppConfig) -> T:
    """
    Loads the orchestration config from a JSON file.
    This is a simplified loader; complex types might need custom rehydration.
    """
    if not config_path.exists():
        logger.error(f"Orchestration config file not found: {config_path}")
        raise FileNotFoundError(f"Orchestration config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Basic rehydration. For more complex cases, a dedicated deserialization
    # function per dataclass or a library like dacite would be better.
    # This assumes that the structure in JSON matches the dataclass fields.
    # Path objects are stored as strings, so they need to be converted back.

    def rehydrate_paths(data: Any, cls: Type) -> Any:
        if dataclasses.is_dataclass(cls) and not isinstance(cls, type): # Check if it's a dataclass type
            field_types = {f.name: f.type for f in dataclasses.fields(cls)}
            hydrated_data = {}
            for k, v in data.items():
                if k in field_types:
                    field_type = field_types[k]
                    # Handle Optional types by getting the underlying type
                    if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                        # Assuming Optional[Path] is Union[Path, NoneType]
                        args = [arg for arg in field_type.__args__ if arg is not type(None)]
                        if args and args[0] is Path:
                            field_type = Path
                        # Potentially handle other Union types or nested dataclasses
                        elif args and dataclasses.is_dataclass(args[0]):
                             hydrated_data[k] = rehydrate_paths(v, args[0])
                             continue


                    if field_type is Path and isinstance(v, str):
                        hydrated_data[k] = Path(v)
                    elif dataclasses.is_dataclass(field_type) and isinstance(v, dict):
                        hydrated_data[k] = rehydrate_paths(v, field_type)
                    elif isinstance(v, list) and hasattr(field_type, "__args__") and field_type.__args__:
                        # Handle lists of dataclasses or Paths if necessary
                        # For simplicity, assuming list of basic types or already processed types
                        element_type = field_type.__args__[0]
                        if element_type is Path:
                            hydrated_data[k] = [Path(item) if isinstance(item, str) else item for item in v]
                        elif dataclasses.is_dataclass(element_type): # List of dataclasses
                            hydrated_data[k] = [rehydrate_paths(item, element_type) for item in v if isinstance(item, dict)]
                        else:
                            hydrated_data[k] = v # Keep as is
                    else:
                        hydrated_data[k] = v
                else: # field not in dataclass, might be from an older config
                    hydrated_data[k] = v
            try:
                # Filter out keys not in the dataclass definition to prevent TypeError on construction
                cls_fields = {f.name for f in dataclasses.fields(cls)}
                filtered_hydrated_data = {k: v for k, v in hydrated_data.items() if k in cls_fields}
                return cls(**filtered_hydrated_data)
            except TypeError as e:
                logger.error(f"TypeError during instantiation of {cls.__name__}: {e}")
                logger.error(f"Data provided: {filtered_hydrated_data}")
                raise
        elif isinstance(data, list): # Handle list of dataclasses at the top level if expected_type is List[Dataclass]
             # This part is tricky if expected_type is, e.g., List[SomeDataclass]
             # For now, assume expected_type is a single dataclass
            pass
        return data


    # Rehydrate known nested dataclasses and Path objects
    # This is a simplified rehydration. For a robust solution, you might need
    # a library like `dacite` or more sophisticated custom logic.
    if "global_config" in config_dict and "project_root" in config_dict["global_config"]:
        config_dict["global_config"]["project_root"] = Path(
            config_dict["global_config"]["project_root"]
        )
    if "data_config" in config_dict and "base_data_path" in config_dict["data_config"]:
        config_dict["data_config"]["base_data_path"] = Path(
            config_dict["data_config"]["base_data_path"]
        )
    # Add more specific rehydrations for other Path or complex objects if needed

    try:
        # Attempt to reconstruct the AppConfig object
        # This is a simplified reconstruction. For complex nested dataclasses,
        # you might need a more robust deserialization mechanism (e.g., using dacite).
        
        # Rehydrate paths for top-level AppConfig and nested structures
        rehydrated_config_dict = rehydrate_paths(config_dict, expected_type)
        
        # If rehydrate_paths returns the instantiated object, use it directly
        if isinstance(rehydrated_config_dict, expected_type):
            loaded_config = rehydrated_config_dict
        else: # Fallback if rehydrate_paths only modified the dict (older version)
            # Filter keys to match expected_type fields to avoid TypeErrors
            expected_fields = {f.name for f in dataclasses.fields(expected_type)}
            filtered_dict = {k: v for k, v in rehydrated_config_dict.items() if k in expected_fields}
            loaded_config = expected_type(**filtered_dict)

        logger.info(f"Orchestration config loaded from {config_path}")
        return loaded_config
    except TypeError as e:
        logger.error(
            f"Failed to instantiate {expected_type.__name__} from config_dict: {e}"
        )
        logger.error(f"Config dict was: {config_dict}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading orchestration config: {e}")
        raise


def setup_run_output_directory(
    base_dir_name: str, run_name_prefix: str, config: AppConfig, resume_run_id: str = None
) -> Path:
    """
    Sets up the output directory for a run.
    If resume_run_id is provided, it attempts to use that directory.
    Otherwise, creates a new directory with a timestamp.
    """
    project_root = get_project_root()
    base_output_path = project_root / base_dir_name

    if resume_run_id:
        run_id = resume_run_id
        output_dir = base_output_path / run_id
        if not output_dir.is_dir():
            logger.warning(
                f"Resume run ID '{resume_run_id}' provided, but directory {output_dir} not found."
            )
            logger.warning("A new run will be started instead.")
            resume_run_id = None  # Fallback to new run
        else:
            logger.info(f"Resuming run using output directory: {output_dir}")
            # Optionally, clean up specific files or subdirectories if needed for a resume
            # For example, remove old status files or temporary data
            # For now, we assume the directory is in a usable state for resuming
            return output_dir.resolve()

    # Create a new run directory if not resuming or fallback
    timestamp_str = get_kst_time_now_str_filename_friendly()
    run_id = f"{run_name_prefix}_{timestamp_str}"
    output_dir = base_output_path / run_id

    ensure_directory_exists(output_dir)
    logger.info(f"Run output directory set up at: {output_dir}")

    # Update config with the resolved output directory if it's part of AppConfig
    # (Assuming AppConfig might have a field for it, e.g., config.run_output_dir)
    # Example: if hasattr(config, 'run_output_dir'):
    #              config.run_output_dir = output_dir.resolve()
    # This depends on AppConfig's definition.

    return output_dir.resolve()


def backup_previous_run_results(output_dir: Path) -> None:
    """
    Backs up existing files in the output directory to a '_backup' subdirectory.
    This is useful if a run ID is reused or for safety before overwriting.
    """
    if not output_dir.exists() or not any(output_dir.iterdir()):
        logger.info(f"No existing results in {output_dir} to back up.")
        return

    backup_subdir_name = f"_backup_{get_kst_time_now_str_filename_friendly()}"
    backup_path = output_dir / backup_subdir_name
    ensure_directory_exists(backup_path)

    logger.info(f"Backing up existing results from {output_dir} to {backup_path}...")
    moved_count = 0
    for item in output_dir.iterdir():
        if item.name == backup_path.name or item.name.startswith("_backup_"):
            continue  # Don't move previous backup folders or the current one

        try:
            destination = backup_path / item.name
            if item.is_dir():
                shutil.move(str(item), str(destination))
            else:
                shutil.move(str(item), str(destination))
            moved_count += 1
        except Exception as e:
            logger.error(f"Failed to move {item.name} to backup: {e}")

    if moved_count > 0:
        logger.info(f"Successfully backed up {moved_count} items to {backup_path}.")
    else:
        logger.info(f"No items were moved to backup from {output_dir}.")


# Example usage or placeholder for other utility functions
if __name__ == "__main__":
    # This section is for demonstration or testing of utility functions
    # Create a dummy AppConfig for testing save/load
    from src.config.definitions import (
        GlobalConfig,
        DataConfig,
        StrategyConfig,
        WFOConfig,
        OptunaConfig,
        BacktestConfig,
        ReportingConfig,
        LiveTradingConfig,
        AccountConfig,
        ExchangeAPIConfig,
        PerformanceConfig,
        RiskManagementConfig,
        NotificationConfig,
        OptimizationParams,
        ScheduleConfig
    )
    
    # Create dummy instances for nested configs
    dummy_global_config = GlobalConfig(
        project_root=Path("/tmp/project"),
        log_level="INFO",
        base_currency="USDT",
        is_simulation=True
    )
    dummy_data_config = DataConfig(
        exchange_name="binance",
        data_source_type="local", # or "api"
        base_data_path=Path("/tmp/data"),
        fetch_start_date="2020-01-01T00:00:00Z",
        klines_cache_enabled=True,
        klines_cache_dir=Path("/tmp/cache/klines"),
        klines_cache_retention_days=30
    )
    dummy_strategy_config = StrategyConfig(
        name="MA_Crossover",
        strategy_id="MA_Cross_01",
        parameters={"short_window": 10, "long_window": 30},
        param_space_path="config/param_spaces/ma_crossover_space.json",
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1h", "4h"],
        default_symbol="BTC/USDT",
        default_timeframe="1h"
    )
    dummy_wfo_config = WFOConfig(
        enabled=True,
        n_splits=3,
        gap_duration_days=7,
        test_duration_days=30,
        min_is_period_days=90,
        max_is_period_days=180,
        oos_validation_metric="total_pnl"
    )
    dummy_optuna_config = OptunaConfig(
        n_trials=50,
        timeout_seconds=3600,
        study_name_prefix="optuna_study",
        direction="maximize",
        sampler_type="TPE", # or "Random", "CMAES"
        pruner_type="MedianPruner", # or "NopPruner", "Hyperband"
        storage_url="sqlite:///optuna_studies.db"
    )
    dummy_backtest_config = BacktestConfig(
        start_date="2021-01-01T00:00:00Z",
        end_date="2022-01-01T00:00:00Z",
        initial_capital=10000.0,
        commission_fee_pct=0.001,
        slippage_pct=0.0005,
        max_active_trades=5,
        cache_enabled=True,
        cache_dir=Path("/tmp/cache/backtests")
    )

    dummy_reporting_config = ReportingConfig(
        output_dir=Path("/tmp/reports"),
        generate_charts=True,
        performance_metrics=["sharpe_ratio", "max_drawdown"],
        save_trades_csv=True,
        report_format="html" # or "pdf", "json"
    )
    
    dummy_app_config = AppConfig(
        run_id="test_run_123",
        global_config=dummy_global_config,
        data_config=dummy_data_config,
        strategy_configs=[dummy_strategy_config], # List of StrategyConfig
        wfo_config=dummy_wfo_config,
        optuna_config=dummy_optuna_config,
        backtest_config=dummy_backtest_config,
        reporting_config=dummy_reporting_config,
        # logger=logging.getLogger("dummy_logger") # Logger will be excluded
    )

    # Test saving
    test_output_dir = Path("./test_run_output")
    ensure_directory_exists(test_output_dir)
    
    print(f"Attempting to save dummy config to {test_output_dir}...")
    saved_path = save_orchestration_config(dummy_app_config, "dummy_run", test_output_dir)
    print(f"Dummy config saved to: {saved_path}")

    # Test loading
    if saved_path and saved_path.exists():
        print(f"Attempting to load dummy config from {saved_path}...")
        try:
            loaded_app_config = load_orchestration_config(saved_path, AppConfig)
            print("Dummy config loaded successfully.")
            # Verify some fields
            assert loaded_app_config.run_id == "test_run_123"
            assert loaded_app_config.global_config.project_root == Path("/tmp/project")
            assert loaded_app_config.data_config.base_data_path == Path("/tmp/data")
            assert loaded_app_config.strategy_configs[0].name == "MA_Crossover"
            assert loaded_app_config.reporting_config.output_dir == Path("/tmp/reports")
            print("Key fields verified successfully after loading.")
            if hasattr(loaded_app_config, 'logger') and loaded_app_config.logger is not None:
                 print(f"Logger found after loading: {loaded_app_config.logger}") # Should not happen if excluded
            else:
                 print("Logger attribute is not present or is None after loading, as expected.")

        except Exception as e:
            print(f"Error during loading or verification: {e}")
    else:
        print("Saved path not found, skipping load test.")

    # Clean up test directory
    # shutil.rmtree(test_output_dir)
    # print(f"Cleaned up test directory: {test_output_dir}")
