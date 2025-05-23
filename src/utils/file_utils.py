import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def calculate_file_sha256(file_path: Path) -> Optional[str]:
    """
    Calculates the SHA256 hash of a file.

    Args:
        file_path (Path): The path to the file.

    Returns:
        Optional[str]: The hexadecimal SHA256 hash string if successful, None otherwise.
    """
    if not file_path.is_file():
        logger.error(f"File not found or is not a file: {file_path}")
        return None

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        logger.error(f"Error reading file {file_path} for SHA256 calculation: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during SHA256 calculation for {file_path}: {e}", exc_info=True)
        return None

def ensure_dir_exists(dir_path: Path) -> bool:
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        dir_path (Path): The path to the directory.

    Returns:
        bool: True if the directory exists or was created successfully, False otherwise.
    """
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        return False

# Add other file utility functions here if needed (e.g., for reading/writing JSON, CSV, Parquet)
# For example:
# import json
# import pandas as pd

# def read_json_file(file_path: Path) -> Optional[Any]:
#     if not file_path.is_file():
#         logger.error(f"JSON file not found: {file_path}")
#         return None
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except json.JSONDecodeError:
#         logger.error(f"Error decoding JSON from {file_path}")
#         return None
#     except Exception as e:
#         logger.error(f"Error reading JSON file {file_path}: {e}")
#         return None

# def write_json_file(data: Any, file_path: Path, indent: int = 4) -> bool:
#     try:
#         ensure_dir_exists(file_path.parent)
#         with open(file_path, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=indent, default=str) # default=str for datetime etc.
#         return True
#     except Exception as e:
#         logger.error(f"Error writing JSON to {file_path}: {e}")
#         return False
