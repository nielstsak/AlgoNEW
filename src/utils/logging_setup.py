# src/utils/logging_setup.py
"""
Ce module est responsable de la configuration du système de logging de l'application.
Il utilise un objet LoggingConfig pour définir les paramètres des loggers,
handlers, et formateurs.
"""
import logging
import logging.handlers
import sys
import os # Importé pour une éventuelle utilisation, bien que Path soit préféré
from pathlib import Path
from typing import Dict, Optional, Any, List, Union, TYPE_CHECKING

# Tentative d'importation de python-json-logger pour le formatage JSON optionnel
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
    # Ne pas logger ici au niveau du module, car le logging n'est pas encore configuré.
    # Un logger spécifique à ce module sera créé plus bas.
except ImportError:
    JSON_LOGGER_AVAILABLE = False
    # De même, pas de log ici. L'information sera loguée par la fonction setup_logging.

if TYPE_CHECKING:
    # Importation pour le type hinting seulement, pour éviter les dépendances circulaires
    # au moment de l'exécution si definitions.py importe des éléments de utils.
    # Le loader.py s'assure que LoggingConfig est disponible au moment de l'appel.
    from src.config.definitions import LoggingConfig, LiveLoggingConfig

# Logger spécifique pour ce module. Il sera configuré par la fonction setup_logging
# ou utilisera la configuration par défaut de Python si setup_logging n'est pas appelée avant.
module_logger = logging.getLogger(__name__)

# Constantes pour la rotation des fichiers de log
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5

# Variable globale pour suivre les handlers gérés par cette fonction setup_logging.
# Cela aide à éviter d'ajouter des handlers en double si setup_logging est appelée plusieurs fois.
_handlers_managed_by_this_setup: List[logging.Handler] = []

class ContextFilter(logging.Filter):
    """
    Un filtre de logging pour injecter des données contextuelles supplémentaires
    dans chaque enregistrement de log (LogRecord).
    """
    def __init__(self, context_data: Optional[Dict[str, Any]] = None):
        """
        Initialise le filtre avec les données contextuelles à ajouter.

        Args:
            context_data (Optional[Dict[str, Any]]): Un dictionnaire de données
                contextuelles. Les clés de ce dictionnaire deviendront des attributs
                des objets LogRecord.
        """
        super().__init__()
        self.context_data = context_data if context_data is not None else {}

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Modifie l'enregistrement de log pour y ajouter les données contextuelles.

        Args:
            record (logging.LogRecord): L'enregistrement de log à traiter.

        Returns:
            bool: Toujours True, car ce filtre ne supprime pas d'enregistrements.
        """
        for key, value in self.context_data.items():
            setattr(record, key, value)
        return True

def setup_logging(
    log_config: Union['LoggingConfig', 'LiveLoggingConfig'],
    log_dir: Path,
    log_filename: Optional[str] = None, # Rendu optionnel, car LiveLoggingConfig peut avoir son propre nom
    root_level: int = logging.INFO,
    context_data: Optional[Dict[str, Any]] = None,
    use_json_format: bool = False
) -> None:
    """
    Configure le système de logging de l'application.

    Cette fonction configure le logger racine avec des handlers pour la console
    et, optionnellement, pour un fichier rotatif. Elle permet également de définir
    des niveaux de log spécifiques par module et d'utiliser un format JSON pour les logs.

    Args:
        log_config (Union[LoggingConfig, LiveLoggingConfig]): L'objet de configuration
            contenant les paramètres de logging (niveau, format, nom de fichier, etc.).
        log_dir (Path): Le répertoire où les fichiers de log seront sauvegardés.
        log_filename (Optional[str]): Le nom de base pour le fichier de log.
            Si log_config est une LiveLoggingConfig et a log_filename_live, ce dernier sera utilisé.
            Sinon, log_config.log_filename_global sera utilisé. Ce paramètre peut surcharger.
        root_level (int): Le niveau de logging pour le logger racine. Par défaut logging.INFO.
        context_data (Optional[Dict[str, Any]]): Un dictionnaire optionnel de données
            contextuelles à injecter dans chaque enregistrement de log.
        use_json_format (bool): Si True et si python-json-logger est disponible,
            les logs seront formatés en JSON. Sinon, un format textuel sera utilisé.
    """
    global _handlers_managed_by_this_setup
    # Utiliser le logger de ce module pour les messages de configuration du logging
    # Il est important que ce logger soit configuré après le root logger pour hériter de ses handlers.
    # Pour les messages initiaux de cette fonction, ils pourraient aller à une config par défaut si appelée avant.

    # Message initial avant la reconfiguration potentielle du logger de ce module
    initial_setup_message = (
        f"Début de la configuration du logging. Log dir: {log_dir}, "
        f"Fichier base: {log_filename or 'déduit de log_config'}, "
        f"Format JSON demandé: {use_json_format}, "
        f"JSON logger disponible: {JSON_LOGGER_AVAILABLE}."
    )
    # print(initial_setup_message) # Imprimer directement car le logger n'est pas encore configuré

    # Nettoyer les handlers précédemment gérés par cette fonction
    root_logger = logging.getLogger()
    if _handlers_managed_by_this_setup:
        # print(f"Nettoyage de {len(_handlers_managed_by_this_setup)} handlers précédemment gérés.")
        for handler in _handlers_managed_by_this_setup:
            if handler in root_logger.handlers: # Vérifier si le handler est toujours attaché
                root_logger.removeHandler(handler)
                handler.close() # Important pour fermer les fichiers
        _handlers_managed_by_this_setup.clear()

    root_logger.setLevel(root_level)

    # Déterminer le nom de fichier de log final
    final_log_filename: str
    if log_filename: # Si un nom de fichier est explicitement passé, il a la priorité
        final_log_filename = log_filename
    elif hasattr(log_config, 'log_filename_live') and getattr(log_config, 'log_filename_live'):
        final_log_filename = getattr(log_config, 'log_filename_live')
    elif hasattr(log_config, 'log_filename_global') and getattr(log_config, 'log_filename_global'):
        final_log_filename = getattr(log_config, 'log_filename_global')
    else:
        final_log_filename = "app_default.log" # Un fallback si aucun nom n'est trouvé
        # print(f"WARN: Nom de fichier de log non trouvé dans log_config ou non fourni. Utilisation de '{final_log_filename}'.")


    # Créer le formateur
    actual_use_json = use_json_format and JSON_LOGGER_AVAILABLE
    log_format_str = log_config.format if hasattr(log_config, 'format') and log_config.format else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter: logging.Formatter
    if actual_use_json:
        # Pour python-json-logger, le format string est une liste de champs standard.
        # Les champs supplémentaires (du filtre de contexte) sont ajoutés automatiquement.
        # Un format commun pourrait être :
        json_format = "%(asctime)s %(levelname)s %(name)s %(module)s %(lineno)d %(message)s"
        # Si context_data est { "run_id": "xyz" }, "run_id" sera ajouté au JSON.
        formatter = jsonlogger.JsonFormatter(json_format)
    else:
        formatter = logging.Formatter(log_format_str)
        if use_json_format and not JSON_LOGGER_AVAILABLE:
            pass # Un message sera loggué plus tard par module_logger

    # Créer le filtre de contexte si des données contextuelles sont fournies
    current_context_filter: Optional[ContextFilter] = None
    if context_data:
        current_context_filter = ContextFilter(context_data)

    # Configurer le StreamHandler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler_level_str = log_config.level.upper() if hasattr(log_config, 'level') else "INFO"
    console_handler.setLevel(getattr(logging, console_handler_level_str, logging.INFO))
    if current_context_filter:
        console_handler.addFilter(current_context_filter)
    root_logger.addHandler(console_handler)
    _handlers_managed_by_this_setup.append(console_handler)

    # Configurer le RotatingFileHandler si log_to_file est True
    if hasattr(log_config, 'log_to_file') and log_config.log_to_file:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / final_log_filename

            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=LOG_FILE_MAX_BYTES,
                backupCount=LOG_FILE_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler_level_str = log_config.level.upper() if hasattr(log_config, 'level') else "INFO"
            file_handler.setLevel(getattr(logging, file_handler_level_str, logging.INFO))
            if current_context_filter:
                file_handler.addFilter(current_context_filter)
            root_logger.addHandler(file_handler)
            _handlers_managed_by_this_setup.append(file_handler)
            # print(f"INFO: Logging vers fichier activé : {log_file_path}")
        except OSError as e:
            # print(f"ERROR: Échec de la création du répertoire de log ou du file handler : {e}")
            # Le logger de ce module n'est pas encore prêt à être utilisé pour ce message.
            # Il sera utilisé pour les messages suivants.
            pass # Le message sera loggué par le logger de ce module plus bas.
        except Exception as e_unexp:
            # print(f"ERROR: Erreur inattendue lors de la configuration du file logging : {e_unexp}")
            pass


    # Appliquer les niveaux de log par module
    if hasattr(log_config, 'log_levels_by_module') and log_config.log_levels_by_module:
        for module_name_key, level_val_str in log_config.log_levels_by_module.items():
            module_specific_logger = logging.getLogger(module_name_key)
            level_to_set = getattr(logging, level_val_str.upper(), None)
            if level_to_set is not None:
                module_specific_logger.setLevel(level_to_set)
            else:
                # print(f"WARN: Niveau de log invalide '{level_val_str}' pour le module '{module_name_key}'.")
                pass # Sera loggué par module_logger

    # Maintenant que le logging est (re)configuré, on peut utiliser module_logger
    module_logger.info(initial_setup_message) # Loguer le message initial avec la nouvelle config
    if use_json_format and not JSON_LOGGER_AVAILABLE:
        module_logger.warning("Format JSON demandé mais python-json-logger n'est pas disponible. Utilisation du format textuel.")
    
    if hasattr(log_config, 'log_to_file') and log_config.log_to_file:
        log_file_path_check = log_dir / final_log_filename
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == log_file_path_check.resolve() for h in _handlers_managed_by_this_setup):
             module_logger.error(f"Échec de la configuration du file handler pour {log_file_path_check}. Vérifiez les permissions ou les erreurs précédentes.")
        else:
             module_logger.info(f"Logging vers fichier configuré : {log_file_path_check}")
    else:
        module_logger.info("Logging vers fichier désactivé dans la configuration.")

    if hasattr(log_config, 'log_levels_by_module') and log_config.log_levels_by_module:
        for module_name_key, level_val_str in log_config.log_levels_by_module.items():
            level_to_set = getattr(logging, level_val_str.upper(), None)
            if level_to_set is None:
                 module_logger.warning(f"Niveau de log invalide '{level_val_str}' spécifié pour le module '{module_name_key}'. Ignoré.")
            else:
                 module_logger.debug(f"Niveau de log pour le module '{module_name_key}' configuré à {level_val_str.upper()}.")
    
    module_logger.info(f"Configuration du logging terminée. Niveau racine: {logging.getLevelName(root_logger.level)}. "
                       f"Nombre total de handlers gérés: {len(_handlers_managed_by_this_setup)}.")

