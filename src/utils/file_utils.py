# src/utils/file_utils.py
"""
Ce module fournit des fonctions utilitaires pour les opérations sur les fichiers,
telles que le calcul de hash, la création de répertoires, et la lecture/écriture
de fichiers JSON.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Any, Callable # Callable ajouté pour default_serializer

logger = logging.getLogger(__name__)

def calculate_file_sha256(file_path: Path) -> Optional[str]:
    """
    Calcule le hash SHA256 d'un fichier.

    Args:
        file_path (Path): Le chemin vers le fichier.

    Returns:
        Optional[str]: Le hash SHA256 hexadécimal en chaîne de caractères si réussi,
                       None sinon.
    """
    log_prefix = f"[CalcSHA256][{file_path.name}]"
    if not isinstance(file_path, Path):
        logger.error(f"{log_prefix} file_path doit être un objet Path. Reçu : {type(file_path)}")
        return None
    if not file_path.is_file():
        logger.error(f"{log_prefix} Fichier non trouvé ou n'est pas un fichier : {file_path}")
        return None

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Lire et mettre à jour le hash par blocs de 4K pour gérer les gros fichiers
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        hex_digest = sha256_hash.hexdigest()
        logger.debug(f"{log_prefix} Hash SHA256 calculé avec succès : {hex_digest}")
        return hex_digest
    except IOError as e_io:
        logger.error(f"{log_prefix} Erreur d'IO lors de la lecture du fichier {file_path} pour le calcul SHA256 : {e_io}", exc_info=True)
        return None
    except Exception as e_unexpected: # pylint: disable=broad-except
        logger.error(f"{log_prefix} Une erreur inattendue s'est produite lors du calcul SHA256 pour {file_path} : {e_unexpected}", exc_info=True)
        return None

def ensure_dir_exists(dir_path: Path) -> bool:
    """
    S'assure qu'un répertoire existe, en le créant (y compris les répertoires parents)
    si nécessaire.

    Args:
        dir_path (Path): Le chemin vers le répertoire.

    Returns:
        bool: True si le répertoire existe ou a été créé avec succès, False sinon.
    """
    log_prefix = f"[EnsureDir][{dir_path}]"
    if not isinstance(dir_path, Path):
        logger.error(f"{log_prefix} dir_path doit être un objet Path. Reçu : {type(dir_path)}")
        return False
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"{log_prefix} Répertoire assuré/créé avec succès.")
        return True
    except OSError as e_os:
        logger.error(f"{log_prefix} Échec de la création du répertoire {dir_path} : {e_os}", exc_info=True)
        return False
    except Exception as e_unexpected: # pylint: disable=broad-except
        logger.error(f"{log_prefix} Erreur inattendue lors de la création du répertoire {dir_path} : {e_unexpected}", exc_info=True)
        return False

def save_json(
    file_path: Path,
    data: Any,
    indent: int = 4,
    default_serializer: Optional[Callable[[Any], Any]] = str
) -> bool:
    """
    Écrit des données Python (typiquement dict ou list) dans un fichier JSON.

    Args:
        file_path (Path): Le chemin complet du fichier JSON à sauvegarder.
        data (Any): Les données à écrire dans le fichier.
        indent (int): Le niveau d'indentation pour le formatage JSON. Par défaut 4.
        default_serializer (Optional[Callable[[Any], Any]]): Une fonction optionnelle
            à passer à `json.dump` pour sérialiser les types non standards.
            Par défaut `str` pour convertir les objets non sérialisables en leur
            représentation chaîne. Mettre à `None` pour utiliser le comportement
            par défaut de `json.dump` (qui lèvera une TypeError pour les types non sérialisables).

    Returns:
        bool: True si la sauvegarde a réussi, False sinon.
    """
    log_prefix = f"[SaveJSON][{file_path.name}]"
    if not isinstance(file_path, Path):
        logger.error(f"{log_prefix} file_path doit être un objet Path. Reçu : {type(file_path)}")
        return False

    try:
        # S'assurer que le répertoire parent existe
        if not ensure_dir_exists(file_path.parent):
            logger.error(f"{log_prefix} Échec de la création du répertoire parent {file_path.parent} pour le fichier JSON.")
            return False

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=default_serializer, ensure_ascii=False)
        logger.info(f"{log_prefix} Données JSON sauvegardées avec succès dans : {file_path}")
        return True
    except TypeError as e_type:
        logger.error(f"{log_prefix} Erreur de type lors de la sérialisation JSON vers {file_path} (un objet n'est pas sérialisable) : {e_type}", exc_info=True)
        return False
    except IOError as e_io:
        logger.error(f"{log_prefix} Erreur d'IO lors de l'écriture JSON vers {file_path} : {e_io}", exc_info=True)
        return False
    except Exception as e_unexpected: # pylint: disable=broad-except
        logger.error(f"{log_prefix} Erreur inattendue lors de l'écriture JSON vers {file_path} : {e_unexpected}", exc_info=True)
        return False

def load_json(file_path: Path) -> Optional[Any]:
    """
    Lit un fichier JSON et le retourne en tant qu'objet Python (dict ou list).

    Args:
        file_path (Path): Le chemin complet du fichier JSON à lire.

    Returns:
        Optional[Any]: Les données chargées depuis le fichier JSON, ou None en cas d'erreur.
    """
    log_prefix = f"[LoadJSON][{file_path.name}]"
    if not isinstance(file_path, Path):
        logger.error(f"{log_prefix} file_path doit être un objet Path. Reçu : {type(file_path)}")
        return None

    if not file_path.is_file():
        logger.error(f"{log_prefix} Fichier JSON non trouvé : {file_path}")
        return None # FileNotFoundError sera gérée par l'appelant si nécessaire, ici on logue et retourne None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"{log_prefix} Données JSON chargées avec succès depuis : {file_path}")
        return data
    except json.JSONDecodeError as e_decode:
        logger.error(f"{log_prefix} Erreur de décodage JSON depuis {file_path} : {e_decode}", exc_info=True)
        return None
    except IOError as e_io:
        logger.error(f"{log_prefix} Erreur d'IO lors de la lecture JSON depuis {file_path} : {e_io}", exc_info=True)
        return None
    except Exception as e_unexpected: # pylint: disable=broad-except
        logger.error(f"{log_prefix} Erreur inattendue lors de la lecture JSON depuis {file_path} : {e_unexpected}", exc_info=True)
        return None

if __name__ == '__main__':
    # Configuration du logging pour les tests directs de ce module
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Test de file_utils ---")
    temp_test_dir = Path("./temp_file_utils_tests").resolve()
    ensure_dir_exists(temp_test_dir) # Créer le répertoire de test

    # Test ensure_dir_exists
    test_subdir = temp_test_dir / "subdir1" / "subdir2"
    logger.info(f"Test ensure_dir_exists pour : {test_subdir}")
    success_ensure = ensure_dir_exists(test_subdir)
    logger.info(f"Résultat ensure_dir_exists : {success_ensure} (Existe: {test_subdir.exists()})")
    assert success_ensure and test_subdir.exists()

    # Test save_json et load_json
    dummy_data = {
        "nom": "Test Algo",
        "version": 1.0,
        "parametres": {"alpha": 0.5, "beta": [1, 2, 3]},
        "active": True,
        "date_creation": datetime.now() # Test avec datetime
    }
    json_test_file = temp_test_dir / "test_data.json"
    
    logger.info(f"Test save_json vers : {json_test_file}")
    success_save = save_json(json_test_file, dummy_data)
    logger.info(f"Résultat save_json : {success_save}")
    assert success_save and json_test_file.exists()

    if success_save:
        logger.info(f"Test load_json depuis : {json_test_file}")
        loaded_data = load_json(json_test_file)
        if loaded_data:
            logger.info(f"Données chargées : {loaded_data}")
            assert loaded_data["nom"] == dummy_data["nom"]
            assert loaded_data["parametres"]["beta"] == dummy_data["parametres"]["beta"]
            # La date sera une chaîne après chargement si default=str a été utilisé
            assert isinstance(loaded_data["date_creation"], str)
        else:
            logger.error("Échec du chargement des données JSON pour le test.")
            assert False, "load_json a échoué"

    # Test calculate_file_sha256
    if json_test_file.exists():
        logger.info(f"Test calculate_file_sha256 pour : {json_test_file}")
        file_hash = calculate_file_sha256(json_test_file)
        logger.info(f"Hash SHA256 calculé : {file_hash}")
        assert file_hash is not None and len(file_hash) == 64

    # Test avec un fichier inexistant pour SHA256
    non_existent_file = temp_test_dir / "fichier_inexistant.txt"
    logger.info(f"Test calculate_file_sha256 pour fichier inexistant : {non_existent_file}")
    hash_non_existent = calculate_file_sha256(non_existent_file)
    logger.info(f"Résultat pour fichier inexistant : {hash_non_existent}")
    assert hash_non_existent is None
    
    # Test load_json avec un fichier inexistant
    logger.info(f"Test load_json pour fichier inexistant : {non_existent_file}")
    data_non_existent = load_json(non_existent_file)
    logger.info(f"Résultat pour fichier inexistant : {data_non_existent}")
    assert data_non_existent is None

    # Test load_json avec un fichier JSON mal formaté
    malformed_json_file = temp_test_dir / "malformed.json"
    with open(malformed_json_file, "w", encoding="utf-8") as f_malformed:
        f_malformed.write("{'nom': 'test', 'valeur': 123,}") # Virgule en trop, apostrophes
    logger.info(f"Test load_json pour fichier mal formaté : {malformed_json_file}")
    data_malformed = load_json(malformed_json_file)
    logger.info(f"Résultat pour fichier mal formaté : {data_malformed}")
    assert data_malformed is None


    logger.info("--- Tests de file_utils terminés ---")
    # Nettoyage optionnel
    # import shutil
    # if temp_test_dir.exists():
    #     shutil.rmtree(temp_test_dir)
    #     logger.info(f"Répertoire de test nettoyé : {temp_test_dir}")
