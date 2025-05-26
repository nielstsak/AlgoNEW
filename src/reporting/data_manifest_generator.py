# src/reporting/data_manifest_generator.py
"""
Ce module est responsable de la génération d'un fichier `data_manifest.json`
qui documente les caractéristiques clés d'un fichier de données d'entrée
(ex: un fichier Parquet enrichi ou un fichier CSV brut).
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List, Union # Union ajouté
from datetime import datetime, timezone

import pandas as pd
# pyarrow est nécessaire pour lire les métadonnées Parquet sans charger tout le fichier
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "La bibliothèque pyarrow n'est pas installée. "
        "L'extraction détaillée des métadonnées des fichiers Parquet (comme num_rows) sera limitée."
    )

# Utilisation des fonctions de file_utils si elles ont été définies et importées correctement.
# Sinon, des fallbacks simples peuvent être utilisés ici ou les fonctions peuvent être copiées.
try:
    from src.utils.file_utils import ensure_dir_exists, calculate_file_sha256, save_json as save_json_util
except ImportError:
    logging.getLogger(__name__).warning(
        "data_manifest_generator: src.utils.file_utils non trouvé ou incomplet. "
        "Utilisation de fallbacks pour ensure_dir_exists, calculate_file_sha256, save_json."
    )
    # Fallbacks (identiques à ceux dans file_utils.py pour la cohérence)
    def ensure_dir_exists(dir_path: Path) -> bool: # type: ignore
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception: return False

    def calculate_file_sha256(file_path: Path) -> Optional[str]: # type: ignore
        if not file_path.is_file(): return None
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception: return None

    def save_json_util(file_path: Path, data: Any, indent: int = 4, default_serializer=str) -> bool: # type: ignore
        try:
            if not ensure_dir_exists(file_path.parent): return False
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=default_serializer)
            return True
        except Exception: return False


logger = logging.getLogger(__name__)

def get_relative_path(absolute_path: Path, project_root: Path) -> Optional[str]:
    """
    Convertit un chemin absolu en chemin relatif par rapport à une racine de projet donnée.

    Args:
        absolute_path (Path): Le chemin absolu à convertir.
        project_root (Path): Le chemin racine du projet.

    Returns:
        Optional[str]: Le chemin relatif sous forme de chaîne, ou None si
                       `absolute_path` n'est pas sous `project_root`.
    """
    try:
        relative = absolute_path.relative_to(project_root)
        return str(relative.as_posix()) # Utiliser as_posix pour des slashs cohérents
    except ValueError:
        # Se produit si absolute_path n'est pas un sous-chemin de project_root
        logger.warning(f"get_relative_path: Le chemin {absolute_path} n'est pas un sous-chemin de {project_root}.")
        return None
    except Exception as e: # pylint: disable=broad-except
        logger.error(f"get_relative_path: Erreur inattendue lors du calcul du chemin relatif pour {absolute_path} : {e}")
        return None


def generate_data_manifest(
    data_file_path: Path,
    output_dir: Path,
    project_root_for_relative_path: Optional[Path] = None,
    git_commit_hash: Optional[str] = None,
    source_script_path: Optional[Path] = None
) -> bool:
    """
    Génère un fichier `data_manifest.json` pour un fichier de données d'entrée,
    documentant ses caractéristiques clés et sa provenance.

    Args:
        data_file_path (Path): Chemin absolu vers le fichier de données d'entrée
                               (ex: .parquet, .csv).
        output_dir (Path): Répertoire où `data_manifest.json` sera sauvegardé.
                           Le manifeste sera nommé `data_file_path.name + "_manifest.json"`.
        project_root_for_relative_path (Optional[Path]): Racine du projet pour rendre
            les chemins relatifs dans le manifeste. Si None, les chemins seront absolus.
        git_commit_hash (Optional[str]): Hash du commit Git associé à la génération
                                         de ce fichier de données (optionnel).
        source_script_path (Optional[Path]): Chemin vers le script ayant généré
                                             le fichier de données (optionnel).

    Returns:
        bool: True si la génération du manifeste est réussie et le fichier sauvegardé,
              False sinon.
    """
    log_prefix = f"[DataManifestGen][{data_file_path.name}]"
    logger.info(f"{log_prefix} Démarrage de la génération du manifeste de données.")

    if not data_file_path.is_file():
        logger.error(f"{log_prefix} Le fichier de données spécifié n'existe pas ou n'est pas un fichier : {data_file_path}")
        return False

    manifest_content: Dict[str, Any] = {
        "manifest_generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_file_name": data_file_path.name,
        "data_file_path_absolute": str(data_file_path.resolve().as_posix()),
        "data_file_path_relative": None, # Sera peuplé si project_root est fourni
        "data_file_size_bytes": None,
        "data_file_last_modified_utc": None,
        "data_file_sha256_hash": None,
        "data_format_specific_metadata": {},
        "provenance": {}
    }

    # Informations de Base du Fichier
    try:
        manifest_content["data_file_size_bytes"] = data_file_path.stat().st_size
        last_mod_timestamp = data_file_path.stat().st_mtime
        manifest_content["data_file_last_modified_utc"] = datetime.fromtimestamp(last_mod_timestamp, tz=timezone.utc).isoformat()
    except OSError as e_stat:
        logger.warning(f"{log_prefix} Impossible de récupérer les informations stat du fichier {data_file_path}: {e_stat}")
        # Continuer avec les infos disponibles

    if project_root_for_relative_path:
        manifest_content["data_file_path_relative"] = get_relative_path(data_file_path.resolve(), project_root_for_relative_path)

    # Hash du Fichier
    manifest_content["data_file_sha256_hash"] = calculate_file_sha256(data_file_path)
    if not manifest_content["data_file_sha256_hash"]:
        logger.warning(f"{log_prefix} Échec du calcul du hash SHA256 pour {data_file_path}.")

    # Métadonnées Spécifiques au Format
    file_extension = data_file_path.suffix.lower()
    data_metadata: Dict[str, Any] = {"file_type_detected": file_extension}

    if file_extension == ".parquet":
        if PYARROW_AVAILABLE:
            try:
                parquet_file = pq.ParquetFile(data_file_path)
                data_metadata["num_rows"] = parquet_file.metadata.num_rows
                data_metadata["num_row_groups"] = parquet_file.metadata.num_row_groups
                data_metadata["column_names"] = parquet_file.schema.names
                # Essayer d'extraire la plage de dates si une colonne 'timestamp' existe
                if 'timestamp' in parquet_file.schema.names:
                    # Lire seulement la colonne timestamp pour min/max (plus efficace que lire tout le df)
                    # Ceci nécessite toujours de lire une partie des données.
                    # Pour les très gros fichiers, on pourrait lire les stats des row groups si disponibles.
                    try:
                        ts_series = pd.read_parquet(data_file_path, columns=['timestamp'])['timestamp']
                        ts_series_dt = pd.to_datetime(ts_series, errors='coerce', utc=True).dropna()
                        if not ts_series_dt.empty:
                            data_metadata["data_start_date_utc"] = ts_series_dt.min().isoformat()
                            data_metadata["data_end_date_utc"] = ts_series_dt.max().isoformat()
                        else:
                             logger.warning(f"{log_prefix} Colonne 'timestamp' trouvée dans Parquet, mais vide ou toutes les valeurs sont invalides après conversion.")
                    except Exception as e_ts_pq: # pylint: disable=broad-except
                        logger.warning(f"{log_prefix} Erreur lors de la lecture de la colonne 'timestamp' du fichier Parquet pour les dates min/max : {e_ts_pq}")
            except Exception as e_pq: # pylint: disable=broad-except
                logger.error(f"{log_prefix} Erreur lors de la lecture des métadonnées du fichier Parquet {data_file_path}: {e_pq}", exc_info=True)
        else:
            logger.warning(f"{log_prefix} pyarrow non disponible. Les métadonnées Parquet détaillées ne peuvent pas être extraites. "
                           "Tentative de lecture des colonnes avec pandas.")
            try:
                df_sample = pd.read_parquet(data_file_path) # Charge tout le fichier, moins efficace
                data_metadata["num_rows"] = len(df_sample)
                data_metadata["column_names"] = df_sample.columns.tolist()
                if 'timestamp' in df_sample.columns:
                    ts_series_dt = pd.to_datetime(df_sample['timestamp'], errors='coerce', utc=True).dropna()
                    if not ts_series_dt.empty:
                        data_metadata["data_start_date_utc"] = ts_series_dt.min().isoformat()
                        data_metadata["data_end_date_utc"] = ts_series_dt.max().isoformat()
            except Exception as e_pd_pq: # pylint: disable=broad-except
                logger.error(f"{log_prefix} Erreur lors de la lecture du fichier Parquet avec pandas : {e_pd_pq}", exc_info=True)

    elif file_extension == ".csv":
        try:
            # Lire seulement l'en-tête pour les noms de colonnes (plus rapide)
            df_header = pd.read_csv(data_file_path, nrows=0)
            data_metadata["column_names"] = df_header.columns.tolist()
            
            # Pour num_rows et dates, il faut lire plus.
            # On peut lire seulement la colonne timestamp pour les dates si elle existe.
            if 'timestamp' in data_metadata["column_names"]:
                try:
                    # Lire seulement la colonne timestamp pour éviter de charger tout en mémoire
                    ts_series_csv = pd.read_csv(data_file_path, usecols=['timestamp'], squeeze=True) # type: ignore # squeeze est déprécié mais fonctionne
                    ts_series_dt_csv = pd.to_datetime(ts_series_csv, errors='coerce', utc=True).dropna()
                    if not ts_series_dt_csv.empty:
                        data_metadata["data_start_date_utc"] = ts_series_dt_csv.min().isoformat()
                        data_metadata["data_end_date_utc"] = ts_series_dt_csv.max().isoformat()
                        # Estimer num_rows à partir de la longueur de cette série (si complète)
                        # C'est une approximation si 'timestamp' a des NaNs qui seraient enlevés.
                        data_metadata["num_rows_estimated_from_timestamp_col"] = len(ts_series_csv)
                    else:
                        logger.warning(f"{log_prefix} Colonne 'timestamp' trouvée dans CSV, mais vide ou toutes les valeurs sont invalides après conversion.")
                except pd.errors.EmptyDataError:
                    logger.warning(f"{log_prefix} Fichier CSV {data_file_path} est vide (EmptyDataError).")
                    data_metadata["num_rows"] = 0
                except ValueError as e_val_csv_ts: # ex: 'timestamp' n'est pas une colonne valide
                    logger.warning(f"{log_prefix} Erreur lors de la lecture de la colonne 'timestamp' du CSV {data_file_path}: {e_val_csv_ts}")
                except Exception as e_csv_ts: # pylint: disable=broad-except
                    logger.warning(f"{log_prefix} Erreur inattendue lors de la lecture de la colonne 'timestamp' du CSV {data_file_path}: {e_csv_ts}")

            # Pour un compte de lignes plus précis pour les CSV (mais plus coûteux) :
            # if "num_rows" not in data_metadata:
            #     try:
            #         with open(data_file_path, 'r', encoding='utf-8') as f_count:
            #             data_metadata["num_rows"] = sum(1 for row in f_count) -1 # Moins l'en-tête
            #     except Exception: pass
        except pd.errors.EmptyDataError:
            logger.warning(f"{log_prefix} Fichier CSV {data_file_path} est vide (EmptyDataError).")
            data_metadata["num_rows"] = 0
            data_metadata["column_names"] = []
        except Exception as e_csv: # pylint: disable=broad-except
            logger.error(f"{log_prefix} Erreur lors de la lecture des métadonnées du fichier CSV {data_file_path}: {e_csv}", exc_info=True)
    else:
        logger.warning(f"{log_prefix} Type de fichier non supporté pour l'extraction de métadonnées détaillées : {file_extension}")

    manifest_content["data_format_specific_metadata"] = data_metadata

    # Informations de Provenance
    if git_commit_hash:
        manifest_content["provenance"]["source_code_git_commit"] = git_commit_hash
    if source_script_path:
        manifest_content["provenance"]["generating_script_path_absolute"] = str(source_script_path.resolve().as_posix())
        if project_root_for_relative_path:
            manifest_content["provenance"]["generating_script_path_relative"] = get_relative_path(source_script_path.resolve(), project_root_for_relative_path)

    # Sauvegarde du Manifeste
    if not ensure_dir_exists(output_dir):
        logger.error(f"{log_prefix} Échec de la création du répertoire de sortie {output_dir} pour le manifeste.")
        return False

    manifest_file_name = data_file_path.name + "_manifest.json"
    manifest_file_full_path = output_dir / manifest_file_name

    if save_json_util(manifest_file_full_path, manifest_content):
        logger.info(f"{log_prefix} Manifeste de données généré et sauvegardé avec succès : {manifest_file_full_path}")
        return True
    else:
        logger.error(f"{log_prefix} Échec de la sauvegarde du fichier manifeste : {manifest_file_full_path}")
        return False


if __name__ == '__main__':
    # Configuration du logging pour les tests directs de ce module
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Test de data_manifest_generator ---")
    
    # Créer un répertoire de test et des fichiers factices
    test_output_root = Path("./temp_manifest_generator_tests").resolve()
    ensure_dir_exists(test_output_root)
    
    test_data_dir = test_output_root / "sample_data"
    ensure_dir_exists(test_data_dir)
    
    test_manifest_output_dir = test_output_root / "manifests"
    ensure_dir_exists(test_manifest_output_dir)

    # Créer un fichier Parquet factice
    dummy_parquet_file = test_data_dir / "sample_data_enriched.parquet"
    timestamps_pq = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00'], utc=True)
    data_pq = {'timestamp': timestamps_pq, 'open': [100, 101, 102], 'close': [101, 102, 103], 'feature1': [0.5, 0.6, 0.7]}
    df_pq = pd.DataFrame(data_pq)
    try:
        df_pq.to_parquet(dummy_parquet_file, index=False)
        logger.info(f"Fichier Parquet factice créé : {dummy_parquet_file}")
    except Exception as e_pq_create:
        logger.error(f"Erreur création Parquet factice: {e_pq_create}. PYARROW_AVAILABLE={PYARROW_AVAILABLE}")
        # Si pyarrow n'est pas là, pd.to_parquet peut échouer ou utiliser un autre moteur.

    # Créer un fichier CSV factice
    dummy_csv_file = test_data_dir / "sample_data_raw.csv"
    timestamps_csv = ['2023-02-01T10:00:00Z', '2023-02-01T10:01:00Z', '2023-02-01T10:02:00Z']
    data_csv = {'timestamp': timestamps_csv, 'value': [10, 20, 15], 'category': ['A', 'B', 'A']}
    df_csv = pd.DataFrame(data_csv)
    df_csv.to_csv(dummy_csv_file, index=False)
    logger.info(f"Fichier CSV factice créé : {dummy_csv_file}")

    # Générer les manifestes
    logger.info("\n--- Génération du manifeste pour le fichier Parquet ---")
    success_pq = generate_data_manifest(
        data_file_path=dummy_parquet_file,
        output_dir=test_manifest_output_dir,
        project_root_for_relative_path=test_output_root,
        git_commit_hash="testcommit123",
        source_script_path=Path(__file__).resolve() # Utiliser ce script comme source factice
    )
    logger.info(f"Résultat de la génération du manifeste Parquet : {success_pq}")
    assert success_pq

    logger.info("\n--- Génération du manifeste pour le fichier CSV ---")
    success_csv = generate_data_manifest(
        data_file_path=dummy_csv_file,
        output_dir=test_manifest_output_dir,
        project_root_for_relative_path=test_output_root
    )
    logger.info(f"Résultat de la génération du manifeste CSV : {success_csv}")
    assert success_csv

    # Vérifier le contenu d'un manifeste (optionnel)
    expected_manifest_parquet = test_manifest_output_dir / (dummy_parquet_file.name + "_manifest.json")
    if expected_manifest_parquet.exists():
        with open(expected_manifest_parquet, 'r', encoding='utf-8') as f_check:
            manifest_data_check = json.load(f_check)
            logger.debug(f"Contenu du manifeste Parquet généré : \n{json.dumps(manifest_data_check, indent=2)}")
            assert manifest_data_check["data_file_name"] == "sample_data_enriched.parquet"
            if PYARROW_AVAILABLE: # num_rows est plus fiable avec pyarrow
                 assert manifest_data_check["data_format_specific_metadata"].get("num_rows") == 3
            assert "timestamp" in manifest_data_check["data_format_specific_metadata"].get("column_names", [])
            assert manifest_data_check["provenance"].get("source_code_git_commit") == "testcommit123"

    logger.info("--- Tests de data_manifest_generator terminés ---")
    # Nettoyage optionnel
    # import shutil
    # if test_output_root.exists():
    #     shutil.rmtree(test_output_root)
    #     logger.info(f"Répertoire de test nettoyé : {test_output_root}")
