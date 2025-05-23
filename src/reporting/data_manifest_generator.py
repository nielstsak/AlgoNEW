
    
    
    
import json
import logging
import hashlib # Already imported in file_utils, but good practice for direct use if needed
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from datetime import timezone # Import specifically timezone

# Attempt to import from local utils, assuming file_utils.py is in the same parent directory (src/utils)
try:
    from src.utils.file_utils import calculate_file_sha256, ensure_dir_exists
except ImportError:
    logger_init = logging.getLogger(__name__ + "_init")
    logger_init.warning(
        "Could not import 'calculate_file_sha256' or 'ensure_dir_exists' from 'src.utils.file_utils'. "
        "SHA256 hash calculation and directory creation might fail or use fallback."
    )
    # Fallback basic implementations if import fails, for completeness,
    # though ideally the project structure ensures utils are available.
    def calculate_file_sha256(file_path: Path) -> Optional[str]:
        logger_init.error("Fallback calculate_file_sha256 called. src.utils.file_utils not imported.")
        if not file_path.is_file(): return None
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception: return None

    def ensure_dir_exists(dir_path: Path) -> bool:
        logger_init.error("Fallback ensure_dir_exists called. src.utils.file_utils not imported.")
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception: return False


logger = logging.getLogger(__name__)

def get_relative_path(absolute_path: Path, project_root: Optional[Path] = None) -> str:
    """
    Tries to make a path relative to the project root.
    If project_root is None or path is not under root, returns the absolute path as string.
    """
    if project_root:
        try:
            return str(absolute_path.relative_to(project_root))
        except ValueError: # Path is not a subpath of project_root
            return str(absolute_path)
    return str(absolute_path)


def generate_data_manifest(
    data_file_path: Path,
    output_dir: Path,
    project_root_for_relative_path: Optional[Path] = None, # For making file_path_relative_to_project
    git_commit_hash: Optional[str] = None,
    source_script_path: Optional[Path] = None
) -> bool:
    """
    Generates a data_manifest.json file documenting key characteristics of an input data file.

    Args:
        data_file_path (Path): Absolute path to the input data file (e.g., _enriched.parquet).
        output_dir (Path): Directory where the data_manifest.json will be saved.
        project_root_for_relative_path (Optional[Path]): Optional project root to make paths relative.
        git_commit_hash (Optional[str]): Optional Git commit hash of the source code.
        source_script_path (Optional[Path]): Optional path to the script that generated the data file.

    Returns:
        bool: True if the manifest generation was successful, False otherwise.
    """
    logger.info(f"Generating data manifest for: {data_file_path}")
    if not data_file_path.is_file():
        logger.error(f"Data file not found or is not a file: {data_file_path}")
        return False

    manifest_content: Dict[str, Any] = {}

    try:
        # --- 1. Basic File Information ---
        manifest_content["file_name"] = data_file_path.name
        manifest_content["file_path_absolute"] = str(data_file_path.resolve())
        if project_root_for_relative_path:
            manifest_content["file_path_relative_to_project"] = get_relative_path(
                data_file_path.resolve(), project_root_for_relative_path
            )
        else:
             manifest_content["file_path_relative_to_project"] = str(data_file_path.resolve())


        file_stat = data_file_path.stat()
        manifest_content["file_size_bytes"] = file_stat.st_size
        manifest_content["last_modified_timestamp_utc"] = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc).isoformat()

        # --- 2. File Hash ---
        logger.debug(f"Calculating SHA256 hash for {data_file_path.name}...")
        calculated_hash = calculate_file_sha256(data_file_path)
        if calculated_hash:
            manifest_content["file_sha256_hash"] = calculated_hash
        else:
            logger.warning(f"Could not calculate SHA256 hash for {data_file_path.name}.")
            manifest_content["file_sha256_hash"] = None

        # --- 3. Data Metadata (specific to Parquet for now) ---
        if data_file_path.suffix.lower() == ".parquet":
            logger.debug(f"Reading Parquet metadata for {data_file_path.name}...")
            try:
                # Read only 'timestamp' column for date range, and schema for columns and row count
                parquet_file = pd.io.parquet.ParquetFile(data_file_path)

                if 'timestamp' in parquet_file.schema.names:
                    # Efficiently get min/max from 'timestamp' column statistics if available
                    # This requires pyarrow and that statistics are written to the Parquet file.
                    # For a more robust way to get min/max without reading all data:
                    # timestamp_chunks = pd.read_parquet(data_file_path, columns=['timestamp'], engine='pyarrow')
                    # manifest_content["data_start_date_utc"] = timestamp_chunks['timestamp'].min().isoformat()
                    # manifest_content["data_end_date_utc"] = timestamp_chunks['timestamp'].max().isoformat()
                    # The above can still be memory intensive for huge files.
                    # A truly efficient way might involve reading metadata from Parquet file footer if stats are there.
                    # For simplicity, if the file isn't excessively large, reading the column is acceptable.
                    # Let's try reading just the timestamp column.
                    df_sample = pd.read_parquet(data_file_path, columns=['timestamp'])
                    if not df_sample.empty and pd.api.types.is_datetime64_any_dtype(df_sample['timestamp']):
                        manifest_content["data_start_date_utc"] = df_sample['timestamp'].min().isoformat()
                        manifest_content["data_end_date_utc"] = df_sample['timestamp'].max().isoformat()
                    else:
                        logger.warning(f"'timestamp' column in {data_file_path.name} is empty or not datetime after loading.")
                else:
                    logger.warning(f"'timestamp' column not found in Parquet schema of {data_file_path.name}.")

                manifest_content["number_of_rows"] = parquet_file.metadata.num_rows
                manifest_content["columns"] = parquet_file.schema.names

            except Exception as e_parquet:
                logger.error(f"Error reading Parquet metadata from {data_file_path.name}: {e_parquet}", exc_info=True)
                manifest_content["data_metadata_error"] = str(e_parquet)
        elif data_file_path.suffix.lower() == ".csv": # Basic CSV metadata
            logger.debug(f"Reading CSV metadata for {data_file_path.name}...")
            try:
                # For CSV, getting min/max timestamp without loading all data is harder.
                # We might just load 'timestamp' or a sample.
                # For row count, we can iterate or load without data.
                df_sample_csv = pd.read_csv(data_file_path, usecols=['timestamp'], parse_dates=['timestamp'], infer_datetime_format=True)
                if not df_sample_csv.empty and pd.api.types.is_datetime64_any_dtype(df_sample_csv['timestamp']):
                    manifest_content["data_start_date_utc"] = df_sample_csv['timestamp'].min().tz_localize('UTC').isoformat() # Assume UTC if not specified
                    manifest_content["data_end_date_utc"] = df_sample_csv['timestamp'].max().tz_localize('UTC').isoformat()
                
                # Count rows (can be slow for very large CSVs)
                # manifest_content["number_of_rows"] = sum(1 for row in open(data_file_path, 'r', encoding='utf-8')) -1 # -1 for header
                # A more pandas-centric way that's still somewhat efficient for just columns:
                df_cols_only = pd.read_csv(data_file_path, nrows=0) # Reads only header
                manifest_content["columns"] = list(df_cols_only.columns)
                # Getting exact row count without loading all data is tricky for CSV.
                # For now, we'll omit it or accept a full read if necessary.
                # manifest_content["number_of_rows"] = len(pd.read_csv(data_file_path)) # This loads all data
                logger.warning("Full row count for CSV is not efficiently implemented yet in manifest.")

            except Exception as e_csv:
                logger.error(f"Error reading CSV metadata from {data_file_path.name}: {e_csv}", exc_info=True)
                manifest_content["data_metadata_error"] = str(e_csv)
        else:
            logger.warning(f"Metadata extraction not implemented for file type: {data_file_path.suffix}")

        # --- 4. Provenance Information ---
        if git_commit_hash:
            manifest_content["source_code_git_commit"] = git_commit_hash
        if source_script_path:
            manifest_content["generating_script_path"] = str(source_script_path.resolve())
            if project_root_for_relative_path:
                 manifest_content["generating_script_path_relative"] = get_relative_path(
                     source_script_path.resolve(), project_root_for_relative_path
                 )


        # --- 5. Save the Manifest ---
        if not ensure_dir_exists(output_dir):
            logger.error(f"Output directory {output_dir} could not be created. Cannot save manifest.")
            return False

        manifest_file_path = output_dir / "data_manifest.json"
        with open(manifest_file_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_content, f, indent=4, default=str) # default=str for datetime etc.
        
        logger.info(f"Data manifest successfully generated and saved to: {manifest_file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate data manifest for {data_file_path}: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a dummy project structure and data file for testing
    test_project_root = Path("./temp_test_project_manifest").resolve()
    test_project_root.mkdir(parents=True, exist_ok=True)
    
    dummy_data_dir = test_project_root / "data" / "historical" / "processed" / "enriched"
    dummy_data_dir.mkdir(parents=True, exist_ok=True)
    
    dummy_log_dir = test_project_root / "logs" / "backtest_optimization" / "test_run_123"
    dummy_log_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy Parquet file
    dummy_parquet_file_path = dummy_data_dir / "TESTPAIR_enriched.parquet"
    timestamps = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00'], utc=True)
    dummy_df = pd.DataFrame({
        'timestamp': timestamps,
        'open': [10, 11, 12],
        'high': [15, 16, 17],
        'low': [9, 10, 11],
        'close': [11, 12, 13],
        'volume': [100, 110, 120]
    })
    try:
        dummy_df.to_parquet(dummy_parquet_file_path, index=False)
        logger.info(f"Dummy Parquet file created at: {dummy_parquet_file_path}")

        success = generate_data_manifest(
            data_file_path=dummy_parquet_file_path,
            output_dir=dummy_log_dir, # Save manifest in the run's log directory
            project_root_for_relative_path=test_project_root,
            git_commit_hash="testcommithash123",
            source_script_path=Path(__file__).resolve() # Example source script
        )
        if success:
            logger.info("Manifest generation example successful.")
            with open(dummy_log_dir / "data_manifest.json", 'r') as f_manifest:
                print("\nGenerated Manifest Content:")
                print(f.read())
        else:
            logger.error("Manifest generation example failed.")

    except Exception as e_test:
        logger.error(f"Error in test execution: {e_test}", exc_info=True)
    finally:
        # Clean up dummy files/dirs
        # import shutil
        # if test_project_root.exists():
        #     shutil.rmtree(test_project_root)
        #     logger.info(f"Cleaned up temp test directory: {test_project_root}")
        pass
