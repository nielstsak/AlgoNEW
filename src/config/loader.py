# src/config/loader.py
"""
Ce module est responsable du chargement de toutes les configurations de l'application,
de l'initialisation du système de logging, de l'instanciation des services principaux,
et de la fourniture de l'instance AppConfig globale.
"""
import json
import logging
import os
import sys
import dataclasses # Pour fields, is_dataclass, replace
from typing import Dict, Optional, Any, List, Union, Type, Tuple, cast, Callable, get_origin, get_args
from pathlib import Path
from dotenv import load_dotenv
from decimal import Decimal # Utilisé dans les dataclasses
import math # Utilisé dans les dataclasses

# Tentative d'importation des définitions de dataclasses et des interfaces/services
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.config.definitions import (
        PathsConfig, LoggingConfig, SimulationDefaults, WfoSettings, OptunaSettings,
        GlobalConfig, SourceDetails, AssetsAndTimeframes, HistoricalPeriod,
        FetchingOptions, DataConfig, ParamDetail, StrategyParamsConfig, StrategiesConfig,
        AccountConfig, GlobalLiveSettings, OverrideRiskSettings, StrategyDeployment,
        LiveLoggingConfig, LiveFetchConfig, LiveConfig, ApiKeys, AppConfig, ExchangeSettings,
        SamplerPrunerProfile, FoldType
    )
    from src.utils.logging_setup import setup_logging
    from src.core.interfaces import (
        IDataValidator, ICacheManager, IEventDispatcher, IStrategyLoader, IErrorHandler
    )
    from src.core.data_validator import DataValidator
    from src.core.cache_manager import CacheManager
    from src.strategies.strategy_factory import StrategyFactory
    from src.backtesting.optimization.objective_function_evaluator import (
        SimpleStrategyLoader, SimpleErrorHandler
    )
    # Placeholder pour IEventDispatcher si pas d'implémentation concrète à instancier ici
    # from src.core.event_dispatcher import SimpleEventDispatcher # Exemple
else:
    # Imports réels
    from src.config.definitions import (
        PathsConfig, LoggingConfig, SimulationDefaults, WfoSettings, OptunaSettings,
        GlobalConfig, SourceDetails, AssetsAndTimeframes, HistoricalPeriod,
        FetchingOptions, DataConfig, ParamDetail, StrategyParamsConfig, StrategiesConfig,
        AccountConfig, GlobalLiveSettings, OverrideRiskSettings, StrategyDeployment,
        LiveLoggingConfig, LiveFetchConfig, LiveConfig, ApiKeys, AppConfig, ExchangeSettings,
        SamplerPrunerProfile, FoldType
    )
    from src.utils.logging_setup import setup_logging
    from src.core.interfaces import (
        IDataValidator, ICacheManager, IEventDispatcher, IStrategyLoader, IErrorHandler
    )
    from src.core.data_validator import DataValidator
    from src.core.cache_manager import CacheManager
    from src.strategies.strategy_factory import StrategyFactory
    from src.backtesting.optimization.objective_function_evaluator import (
        SimpleStrategyLoader, SimpleErrorHandler # Ce sont des implémentations de IStrategyLoader/IErrorHandler
    )
    # Implémentation d'EventDispatcher (si elle existe)
    # from src.core.event_dispatcher import SimpleEventDispatcher # Exemple


logger = logging.getLogger(__name__)

# --- Classe Placeholder pour IEventDispatcher si aucune implémentation concrète n'est fournie ---
class PlaceholderEventDispatcher(IEventDispatcher):
    def register_listener(self, event_type: str, listener: Callable[[Dict[str, Any]], None]) -> None:
        logger.debug(f"[PlaceholderEventDispatcher] Listener enregistré (conceptuellement) pour {event_type}")
    def dispatch(self, event_type: str, event_data: Dict[str, Any]) -> None:
        logger.info(f"[PlaceholderEventDispatcher] Événement distribué: Type='{event_type}', Données='{str(event_data)[:100]}...'")


def _load_json(file_path: Path) -> Dict[str, Any]:
    """Charge un fichier JSON et retourne son contenu sous forme de dictionnaire."""
    if not file_path.is_file():
        logger.error(f"Fichier de configuration non trouvé : {file_path}")
        raise FileNotFoundError(f"Fichier de configuration non trouvé : {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Erreur de décodage JSON du fichier : {file_path} - {e}")
        raise
    except Exception as e_gen:
        logger.error(f"Erreur inattendue lors de la lecture de {file_path}: {e_gen}")
        raise

def _is_optional_union(field_type: Any) -> bool:
    """Vérifie si un type hint est un Optional Union (Union[T, None])."""
    return get_origin(field_type) is Union and type(None) in get_args(field_type)

def _get_non_none_type_from_optional_union(field_type: Any) -> Any:
    """Extrait le type non-None d'un Optional Union."""
    if _is_optional_union(field_type):
        args = get_args(field_type)
        return next((arg for arg in args if arg is not type(None)), type(None))
    return field_type

def _create_dataclass_from_dict(
    dataclass_type: Type[Any],
    data: Dict[str, Any],
    config_file_path: str = "Inconnu"
) -> Any:
    """Crée récursivement une instance de dataclass à partir d'un dictionnaire."""
    if not dataclasses.is_dataclass(dataclass_type):
        # Gérer le cas où le type attendu est un Enum (comme FoldType)
        if isinstance(dataclass_type, type) and issubclass(dataclass_type, enum.Enum):
            try:
                return dataclass_type(data) # data est la valeur de l'enum (ex: "expanding")
            except ValueError as e_enum:
                logger.error(f"Dans {config_file_path} pour {dataclass_type.__name__}: Valeur d'enum invalide '{data}'. Erreur: {e_enum}")
                raise # Ou retourner une valeur par défaut / None si plus approprié
        logger.debug(f"Type {dataclass_type} n'est pas une dataclass ou Enum géré. Retour de la donnée brute : {data}")
        return data

    field_info_map = {f.name: f for f in dataclasses.fields(dataclass_type)}
    init_kwargs: Dict[str, Any] = {}

    # Pré-remplir avec les valeurs par défaut
    for field_name, field_obj in field_info_map.items():
        if field_obj.default is not dataclasses.MISSING:
            init_kwargs[field_name] = field_obj.default
        elif field_obj.default_factory is not dataclasses.MISSING:
            init_kwargs[field_name] = field_obj.default_factory()
        # Laisser les champs sans défaut comme non initialisés pour l'instant

    # Remplir/écraser avec les données du JSON
    for name_from_json, value_from_json in data.items():
        if name_from_json not in field_info_map:
            logger.debug(f"Dans {config_file_path} pour {dataclass_type.__name__}: Clé JSON '{name_from_json}' non trouvée comme champ. Ignorée.")
            continue

        field_obj = field_info_map[name_from_json]
        actual_field_type_for_conversion = _get_non_none_type_from_optional_union(field_obj.type)
        origin_type = get_origin(actual_field_type_for_conversion)
        type_args = get_args(actual_field_type_for_conversion)

        if value_from_json is None and _is_optional_union(field_obj.type):
            init_kwargs[name_from_json] = None
        elif dataclasses.is_dataclass(actual_field_type_for_conversion) and isinstance(value_from_json, dict):
            init_kwargs[name_from_json] = _create_dataclass_from_dict(actual_field_type_for_conversion, value_from_json, config_file_path)
        elif origin_type is list and type_args and dataclasses.is_dataclass(type_args[0]) and isinstance(value_from_json, list):
            list_item_dc_type = type_args[0]
            init_kwargs[name_from_json] = [_create_dataclass_from_dict(list_item_dc_type, item, config_file_path) for item in value_from_json if isinstance(item, dict)]
        elif origin_type is dict and type_args and len(type_args) == 2 and dataclasses.is_dataclass(type_args[1]) and isinstance(value_from_json, dict):
            dict_value_dc_type = type_args[1]
            processed_dict = {}
            for k, v_item in value_from_json.items():
                if isinstance(v_item, dict):
                    processed_dict[k] = _create_dataclass_from_dict(dict_value_dc_type, v_item, config_file_path)
                else:
                    processed_dict[k] = v_item # Laisser tel quel si la valeur n'est pas un dict pour le dataclass
            init_kwargs[name_from_json] = processed_dict
        elif isinstance(actual_field_type_for_conversion, type) and issubclass(actual_field_type_for_conversion, enum.Enum):
             try:
                init_kwargs[name_from_json] = actual_field_type_for_conversion(value_from_json)
             except ValueError as e_enum_val:
                logger.warning(f"Dans {config_file_path} pour {dataclass_type.__name__}: Valeur d'enum invalide '{value_from_json}' pour '{name_from_json}'. Erreur: {e_enum_val}. Utilisation de la valeur originale ou défaut.")
                # Laisser la valeur par défaut si déjà settée, ou la valeur JSON si pas de défaut
                if name_from_json not in init_kwargs: # Si pas de valeur par défaut
                    init_kwargs[name_from_json] = value_from_json
        else:
            try:
                # Tenter une conversion de type si possible
                if actual_field_type_for_conversion == int and not isinstance(value_from_json, bool):
                    init_kwargs[name_from_json] = int(value_from_json)
                elif actual_field_type_for_conversion == float:
                    init_kwargs[name_from_json] = float(value_from_json)
                elif actual_field_type_for_conversion == bool:
                    init_kwargs[name_from_json] = bool(value_from_json)
                elif actual_field_type_for_conversion == str:
                    init_kwargs[name_from_json] = str(value_from_json)
                elif actual_field_type_for_conversion == Path:
                    init_kwargs[name_from_json] = Path(value_from_json)
                else:
                    init_kwargs[name_from_json] = value_from_json # Laisser tel quel
            except (ValueError, TypeError) as e_conv:
                logger.warning(f"Dans {config_file_path} pour {dataclass_type.__name__}: "
                               f"Impossible de convertir la valeur '{value_from_json}' pour le champ '{name_from_json}' vers {actual_field_type_for_conversion}. "
                               f"Utilisation de la valeur originale ou du défaut. Erreur: {e_conv}")
                if name_from_json not in init_kwargs: # Si pas de valeur par défaut et conversion échoue
                    init_kwargs[name_from_json] = value_from_json


    # Vérifier les champs requis qui n'ont pas de défaut et n'ont pas été fournis
    for field_name, field_obj in field_info_map.items():
        if field_obj.default is dataclasses.MISSING and \
           field_obj.default_factory is dataclasses.MISSING and \
           field_name not in init_kwargs:
            # Ce champ est requis mais n'a pas été fourni ni n'a de défaut
            raise ValueError(f"Champ requis '{field_name}' manquant pour {dataclass_type.__name__} dans {config_file_path}")

    try:
        return dataclass_type(**init_kwargs)
    except Exception as e_init:
        logger.error(f"Erreur lors de l'instanciation de {dataclass_type.__name__} depuis {config_file_path} avec kwargs {init_kwargs}: {e_init}", exc_info=True)
        raise ValueError(f"Échec de l'instanciation de {dataclass_type.__name__}: {e_init}") from e_init


def _validate_strategy_configurations(app_config: 'AppConfig', project_root_path: Path) -> List[str]:
    """Valide les configurations des stratégies dans AppConfig.strategies_config."""
    # ... (logique de validation existante, s'assurer qu'elle est à jour si StrategyParamsConfig a changé) ...
    errors: List[str] = []
    log_prefix_val_strat = "[ValidateStrategyConfigs]"
    # ... (le reste de la fonction est supposé être correct et à jour) ...
    return errors

def _validate_app_config(app_config: 'AppConfig', project_root_path: Path):
    """Valide l'instance AppConfig complètement chargée."""
    # ... (logique de validation existante, s'assurer qu'elle est à jour) ...
    # Ajouter la validation pour les nouvelles instances de service
    if not isinstance(app_config.data_validator_instance, IDataValidator): # type: ignore
        raise ValueError("AppConfig.data_validator_instance n'est pas une instance valide de IDataValidator.")
    if not isinstance(app_config.cache_manager_instance, ICacheManager): # type: ignore
        raise ValueError("AppConfig.cache_manager_instance n'est pas une instance valide de ICacheManager.")
    if not isinstance(app_config.strategy_factory_instance, StrategyFactory): # type: ignore
        raise ValueError("AppConfig.strategy_factory_instance n'est pas une instance valide de StrategyFactory.")
    if not isinstance(app_config.strategy_loader_instance, IStrategyLoader): # type: ignore
        raise ValueError("AppConfig.strategy_loader_instance n'est pas une instance valide de IStrategyLoader.")
    if not isinstance(app_config.error_handler_instance, IErrorHandler): # type: ignore
        raise ValueError("AppConfig.error_handler_instance n'est pas une instance valide de IErrorHandler.")
    if not isinstance(app_config.event_dispatcher_instance, IEventDispatcher): # type: ignore
        raise ValueError("AppConfig.event_dispatcher_instance n'est pas une instance valide de IEventDispatcher.")
    # ... (le reste de la fonction est supposé être correct et à jour) ...
    logger.info("Validation de AppConfig (avec instances de service) réussie.")


def load_all_configs(project_root: Optional[str] = None) -> 'AppConfig':
    """
    Charge toutes les configurations JSON, les variables d'environnement pour les clés API,
    instancie les dataclasses, initialise le logging, instancie les services principaux,
    et valide l'instance AppConfig finale.
    """
    # ... (Détermination de project_root_path et chargement .env comme avant) ...
    if project_root is None:
        try:
            project_root_path = Path(__file__).resolve().parent.parent.parent
        except NameError:
            project_root_path = Path('.').resolve()
            print(f"INFO (loader): __file__ non défini, racine du projet déduite comme CWD : {project_root_path}")
    else:
        project_root_path = Path(project_root).resolve()
    print(f"INFO (loader): Racine du projet utilisée : {project_root_path}")
    env_path = project_root_path / '.env'
    if env_path.is_file(): load_dotenv(dotenv_path=env_path, verbose=True)

    config_dir = project_root_path / 'config'
    global_config_path = config_dir / 'config_global.json'
    data_config_path = config_dir / 'config_data.json'
    strategies_config_path = config_dir / 'config_strategies.json'
    live_config_path = config_dir / 'config_live.json'
    accounts_config_path = config_dir / 'config_accounts.json'

    # --- Chargement des configurations JSON en dataclasses ---
    print(f"INFO (loader): Chargement global_config depuis : {global_config_path}")
    global_config_dict = _load_json(global_config_path)
    global_cfg: GlobalConfig = _create_dataclass_from_dict(GlobalConfig, global_config_dict, str(global_config_path))

    print(f"INFO (loader): Chargement data_config depuis : {data_config_path}")
    data_config_dict = _load_json(data_config_path)
    data_cfg: DataConfig = _create_dataclass_from_dict(DataConfig, data_config_dict, str(data_config_path))

    print(f"INFO (loader): Chargement strategies_config depuis : {strategies_config_path}")
    strategies_config_dict_raw = _load_json(strategies_config_path) # C'est un Dict[str, Dict]
    # _create_dataclass_from_dict attend un dict pour le champ 'strategies' de StrategiesConfig
    strategies_cfg: StrategiesConfig = _create_dataclass_from_dict(StrategiesConfig, {"strategies": strategies_config_dict_raw}, str(strategies_config_path))


    print(f"INFO (loader): Chargement live_config depuis : {live_config_path}")
    live_config_dict = _load_json(live_config_path)
    live_cfg: LiveConfig = _create_dataclass_from_dict(LiveConfig, live_config_dict, str(live_config_path))

    print(f"INFO (loader): Chargement accounts_config depuis : {accounts_config_path}")
    accounts_config_list_raw = _load_json(accounts_config_path)
    if not isinstance(accounts_config_list_raw, list):
        raise ValueError(f"{accounts_config_path} doit contenir une liste JSON de configurations de compte.")
    accounts_cfg_list: List[AccountConfig] = [
        _create_dataclass_from_dict(AccountConfig, acc_dict, str(accounts_config_path))
        for acc_dict in accounts_config_list_raw
    ]

    api_key_credentials: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    # ... (logique de chargement des clés API comme avant) ...
    for acc_conf in accounts_cfg_list:
        key_env_var = acc_conf.api_key_env_var
        secret_env_var = acc_conf.api_secret_env_var
        key: Optional[str] = os.getenv(key_env_var) if key_env_var else None
        secret: Optional[str] = os.getenv(secret_env_var) if secret_env_var else None
        api_key_credentials[acc_conf.account_alias] = (key, secret)

    api_keys_instance = ApiKeys(credentials=api_key_credentials)

    # --- Résolution des chemins et initialisation du Logging ---
    # (Comme avant, mais s'assurer que le logger est bien configuré avant les logs des instances de service)
    paths_cfg_mutable = dataclasses.replace(global_cfg.paths) # Créer une copie mutable
    for path_field_obj in dataclasses.fields(paths_cfg_mutable):
        field_name = path_field_obj.name
        relative_path_val = getattr(paths_cfg_mutable, field_name)
        if isinstance(relative_path_val, str):
            absolute_path = (project_root_path / relative_path_val).resolve()
            setattr(paths_cfg_mutable, field_name, str(absolute_path))
            try: # Créer les répertoires si ce sont des logs, results, state, etc.
                if any(keyword in field_name.lower() for keyword in ["log", "result", "state", "data"]):
                    absolute_path.mkdir(parents=True, exist_ok=True)
            except OSError as e_mkdir:
                print(f"ERROR (loader): Échec création répertoire {absolute_path}: {e_mkdir}")
    global_cfg_updated_paths = dataclasses.replace(global_cfg, paths=paths_cfg_mutable)


    # Initialisation du Logging (avant l'instanciation des services qui pourraient logger)
    log_conf_to_use: Union[LoggingConfig, LiveLoggingConfig] = global_cfg_updated_paths.logging
    log_dir_for_setup = Path(global_cfg_updated_paths.paths.logs_backtest_optimization)
    log_filename_for_setup = global_cfg_updated_paths.logging.log_filename_global

    setup_logging(
        log_config=log_conf_to_use,
        log_dir=log_dir_for_setup,
        log_filename=log_filename_for_setup,
        root_level=getattr(logging, global_cfg_updated_paths.logging.level.upper(), logging.INFO)
    )
    logger.info("Système de logging initialisé avec la configuration globale.")

    # --- Instanciation des Services ---
    logger.info("Instanciation des services principaux...")
    data_validator_service = DataValidator() # Utilise ses propres valeurs par défaut si besoin
    
    # Pour CacheManager, on pourrait prendre les paramètres de config_global.json si on les y ajoute
    cache_persist_path = project_root_path / "cache_data" / "global_cache.pkl"
    cache_manager_service = CacheManager(
        max_memory_mb=getattr(global_cfg_updated_paths.optuna_settings, 'cache_max_memory_mb', 1024), # Exemple
        default_ttl_seconds=getattr(global_cfg_updated_paths.optuna_settings, 'cache_default_ttl_seconds', 3600), # Exemple
        persist_path=cache_persist_path,
        auto_load_persist=True
    )
    
    strategy_factory_service = StrategyFactory(
        strategies_package_path="src.strategies", # Chemin standard
        project_root=project_root_path,
        auto_discover=True
    )
    
    # Utiliser les implémentations simples pour l'instant
    strategy_loader_service = SimpleStrategyLoader()
    error_handler_service = SimpleErrorHandler()
    
    # Pour IEventDispatcher, utiliser un placeholder si pas d'implémentation concrète
    event_dispatcher_service = PlaceholderEventDispatcher()
    # Si SimpleEventDispatcher existe:
    # from src.core.event_dispatcher import SimpleEventDispatcher
    # event_dispatcher_service = SimpleEventDispatcher()

    logger.info("Services principaux instanciés.")

    exchange_settings_instance = _create_dataclass_from_dict(ExchangeSettings, global_config_dict.get("exchange_settings", {}))
    # Résoudre le chemin du fichier d'info exchange
    exchange_info_file_path_resolved = project_root_path / exchange_settings_instance.exchange_info_file_path
    if not exchange_info_file_path_resolved.is_file():
         logger.warning(f"Fichier d'info exchange '{exchange_info_file_path_resolved}' non trouvé.")
    # Remplacer le chemin relatif par l'absolu dans l'instance
    exchange_settings_instance_final = dataclasses.replace(exchange_settings_instance, exchange_info_file_path=str(exchange_info_file_path_resolved))


    # --- Création de l'instance AppConfig finale ---
    app_config = AppConfig(
        global_config=global_cfg_updated_paths, # Utiliser celui avec les chemins absolus
        data_config=data_cfg,
        strategies_config=strategies_cfg,
        live_config=live_cfg,
        accounts_config=accounts_cfg_list,
        api_keys=api_keys_instance,
        exchange_settings=exchange_settings_instance_final, # Utiliser celui avec chemin résolu
        project_root=str(project_root_path),
        # Assigner les instances de service
        data_validator_instance=data_validator_service,
        cache_manager_instance=cache_manager_service,
        strategy_factory_instance=strategy_factory_service,
        strategy_loader_instance=strategy_loader_service,
        error_handler_instance=error_handler_service,
        event_dispatcher_instance=event_dispatcher_service
    )

    # --- Validation Finale de AppConfig ---
    _validate_app_config(app_config, project_root_path)

    logger.info(f"Toutes les configurations chargées et AppConfig validée pour: {app_config.global_config.project_name}")
    return app_config

# --- Fonctions de chargement spécifiques (conservées pour compatibilité ou usage direct) ---
def load_strategy_config_by_name(strategy_name: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    """Charge la configuration pour une stratégie spécifique par son nom."""
    # ... (logique existante) ...
    if project_root is None: project_root_path = Path(__file__).resolve().parent.parent.parent
    else: project_root_path = Path(project_root).resolve()
    strategies_config_path = project_root_path / 'config' / 'config_strategies.json'
    all_strategies_dict = _load_json(strategies_config_path)
    strategy_config_data = all_strategies_dict.get(strategy_name)
    if strategy_config_data is None: raise ValueError(f"Config pour strat '{strategy_name}' non trouvée.")
    return cast(Dict[str, Any], strategy_config_data)


def load_exchange_config(exchange_info_file_path_str: Optional[Union[str, Path]] = None, project_root: Optional[str] = None) -> Dict[str, Any]:
    """Charge les informations de l'exchange depuis un fichier JSON."""
    # ... (logique existante, s'assurer qu'elle utilise project_root correctement) ...
    root_path = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent.parent
    
    file_path_to_load: Path
    if exchange_info_file_path_str:
        file_path_to_load = Path(exchange_info_file_path_str)
        if not file_path_to_load.is_absolute():
            file_path_to_load = root_path / file_path_to_load
    else: # Fallback sur le chemin par défaut dans ExchangeSettings, relatif à project_root
        default_rel_path = ExchangeSettings().exchange_info_file_path
        file_path_to_load = root_path / default_rel_path
    
    try:
        return _load_json(file_path_to_load)
    except FileNotFoundError:
        logger.error(f"Fichier d'info exchange non trouvé à : {file_path_to_load}. Retour d'une config vide.")
        return {"symbols": []} # Retourner une structure minimale valide
    except Exception as e:
        logger.error(f"Erreur chargement info exchange {file_path_to_load}: {e}", exc_info=True)
        return {"symbols": []}

