import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Any, List, Union, TYPE_CHECKING

# Attempt to import python-json-logger
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
    logging.getLogger(__name__).debug("python-json-logger library found.")
except ImportError:
    JSON_LOGGER_AVAILABLE = False
    logging.getLogger(__name__).debug("python-json-logger library NOT found. JSON logging will be disabled.")

if TYPE_CHECKING:
    # Assuming LoggingConfig and LiveLoggingConfig are defined in definitions.py
    # and loader.py makes them available.
    # For a standalone script, you might need to adjust this import
    # or define placeholder classes if definitions.py is not directly accessible.
    try:
        from src.config.definitions import LoggingConfig, LiveLoggingConfig
    except ImportError:
        # Define dummy classes for type hinting if actual definitions are not available
        class LoggingConfig:
            level: str
            format: str
            log_to_file: bool
            log_filename_global: str
            log_levels_by_module: Optional[Dict[str, str]] = None
            # Add other fields if they exist in your actual LoggingConfig

        class LiveLoggingConfig(LoggingConfig):
            log_filename_live: Optional[str] = None
            # Add other fields if they exist in your actual LiveLoggingConfig


logger = logging.getLogger(__name__)

# Global list to keep track of handlers managed by this setup function
_handlers_managed_by_setup_logging: List[logging.Handler] = []

# Constants for RotatingFileHandler, can be moved to config if needed
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5

class ContextFilter(logging.Filter):
    """
    A logging filter to inject contextual data into log records.
    """
    def __init__(self, context_data_to_add: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context_data_to_add = context_data_to_add or {}

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self.context_data_to_add.items():
            setattr(record, key, value)
        return True

def setup_logging(
    log_config: Union['LoggingConfig', 'LiveLoggingConfig'],
    log_dir: str,
    log_filename: str,
    root_level: int = logging.INFO,
    context_data: Optional[Dict[str, Any]] = None,
    use_json_format: bool = False
) -> None:
    """
    Sets up application-wide logging with support for JSON format,
    file rotation, per-module levels, and contextual data.

    Args:
        log_config: The logging configuration object.
        log_dir: Directory where log files will be saved.
        log_filename: Base name for the log file.
        root_level: Logging level for the root logger.
        context_data: Optional dictionary of contextual data to add to all logs.
        use_json_format: Flag to enable JSON formatted logs.
    """
    global _handlers_managed_by_setup_logging
    logger.debug(f"Setting up logging. JSON format: {use_json_format}, Log dir: {log_dir}, File: {log_filename}")

    try:
        # --- 1. Clean up previously managed handlers ---
        root_logger = logging.getLogger()
        if _handlers_managed_by_setup_logging:
            logger.debug(f"Removing {len(_handlers_managed_by_setup_logging)} previously managed handlers.")
            for handler_instance in _handlers_managed_by_setup_logging:
                if handler_instance in root_logger.handlers:
                    root_logger.removeHandler(handler_instance)
            _handlers_managed_by_setup_logging.clear()
        else:
            # If the list is empty, it's possible handlers were added outside this function's control.
            # For a truly clean setup, one might consider removing ALL handlers from root_logger.handlers
            # but this can be disruptive if other parts of the system expect their handlers to persist.
            # For now, we only manage handlers added by this function.
            pass


        # --- 2. Configure Root Logger ---
        root_logger.setLevel(root_level)

        # --- 3. Create Formatter ---
        formatter: logging.Formatter
        log_level_str = getattr(log_config, 'level', 'INFO').upper() # General level for handlers
        handler_log_level = getattr(logging, log_level_str, logging.INFO)

        # Prepare format string for text logs, incorporating context keys if present
        base_format_str = getattr(log_config, 'format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        if context_data:
            # Example: Automatically create context string part
            # This is a simple way; for complex needs, modify formatter's format record.
            # For JsonFormatter, fields are added directly.
            # For text, if you want context in the main message string:
            # context_str_parts = [f"%({key})s" for key in context_data.keys()]
            # context_format_part = " - " + " - ".join(context_str_parts) if context_str_parts else ""
            # This would require context_data keys to be simple for % formatting.
            # A better way for text is to ensure ContextFilter sets attributes, and format string uses them.
            # E.g., if context_data={"run_id": "123"}, format string could be "%(asctime)s - %(run_id)s - %(message)s"
            # The default format string from the prompt is:
            # %(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s
            # If context_data keys are 'run_id', 'strategy_name', a new format could be:
            # '%(asctime)s - RID:%(run_id)s - SNAME:%(strategy_name)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s'
            # For this implementation, we assume the provided log_config.format might already include placeholders
            # for context data that the ContextFilter will make available on the LogRecord.
            pass


        if use_json_format and JSON_LOGGER_AVAILABLE:
            # Example JSON format string, customize as needed
            # This includes standard fields. JsonFormatter automatically adds fields from 'extra' or context filter.
            json_format_str = '%(asctime)s %(levelname)s %(name)s %(module)s %(lineno)d %(message)s'
            # Add context keys to the format if you want them explicitly structured by the formatter
            # Or rely on the filter to add them and JsonFormatter to pick them up.
            # custom_attrs_format = " ".join([f"%({key})s" for key in (context_data or {}).keys()])
            # if custom_attrs_format:
            #    json_format_str += " " + custom_attrs_format
            formatter = jsonlogger.JsonFormatter(json_format_str)
            logger.info("Using JSON log format.")
        else:
            if use_json_format and not JSON_LOGGER_AVAILABLE:
                logger.warning("JSON logging requested but python-json-logger is not available. Falling back to text format.")
            formatter = logging.Formatter(base_format_str)
            logger.info(f"Using text log format: {base_format_str}")

        # --- 4. Create Context Filter (if context_data is provided) ---
        context_filter: Optional[ContextFilter] = None
        if context_data:
            context_filter = ContextFilter(context_data)
            logger.debug(f"ContextFilter created with data: {context_data.keys()}")

        # --- 5. Console Handler ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(handler_log_level)
        console_handler.setFormatter(formatter)
        if context_filter:
            console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)
        _handlers_managed_by_setup_logging.append(console_handler)
        logger.debug(f"Console handler added with level {log_level_str}.")

        # --- 6. File Handler (if log_to_file is True) ---
        if getattr(log_config, 'log_to_file', False):
            try:
                log_dir_path = Path(log_dir)
                log_dir_path.mkdir(parents=True, exist_ok=True)
                log_file_path = log_dir_path / log_filename

                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path,
                    maxBytes=LOG_FILE_MAX_BYTES,
                    backupCount=LOG_FILE_BACKUP_COUNT,
                    encoding='utf-8'
                )
                file_handler.setLevel(handler_log_level)
                file_handler.setFormatter(formatter)
                if context_filter:
                    file_handler.addFilter(context_filter)
                root_logger.addHandler(file_handler)
                _handlers_managed_by_setup_logging.append(file_handler)
                logger.info(f"File handler added. Logging to: {log_file_path} (Level: {log_level_str}, Rotation: {LOG_FILE_MAX_BYTES / (1024*1024):.1f}MB, {LOG_FILE_BACKUP_COUNT} backups)")
            except OSError as e_os:
                logger.error(f"Failed to create log directory or file handler for {log_filename} in {log_dir}: {e_os}")
                logger.info("Logging to console only due to file handler OS error.")
            except Exception as e_file:
                logger.error(f"Unexpected error setting up file logging: {e_file}", exc_info=True)
                logger.info("Logging to console only due to unexpected file handler error.")
        else:
            logger.info(f"File logging is disabled. Logging to console only at level {log_level_str}.")

        # --- 7. Apply Per-Module Log Levels ---
        log_levels_by_module = getattr(log_config, 'log_levels_by_module', None)
        if isinstance(log_levels_by_module, dict):
            for module_name, level_str in log_levels_by_module.items():
                module_logger = logging.getLogger(module_name)
                level_int = getattr(logging, level_str.upper(), None)
                if level_int is not None:
                    module_logger.setLevel(level_int)
                    # Ensure propagation if you want these module logs to also go to root handlers
                    # module_logger.propagate = True (usually true by default)
                    logger.debug(f"Set log level for module '{module_name}' to {level_str.upper()} ({level_int}).")
                else:
                    logger.warning(f"Invalid log level string '{level_str}' for module '{module_name}'. Ignoring.")
        
        logger.info(f"Logging setup complete. Root level: {logging.getLevelName(root_level)}. Total managed handlers: {len(_handlers_managed_by_setup_logging)}.")

    except AttributeError as e_attr:
        # Fallback if log_config object doesn't have expected attributes
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - FALLBACK_LOGGING_ATTR_ERR - %(message)s')
        logger.error(f"Logging setup failed due to AttributeError (likely invalid log_config object): {e_attr}. Falling back to basic console logging.", exc_info=True)
    except Exception as e_critical:
        # General fallback for any other unexpected errors during setup
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - FALLBACK_LOGGING_CRITICAL_ERR - %(message)s')
        logger.critical(f"Critical error in logging setup: {e_critical}. Falling back to basic console logging.", exc_info=True)