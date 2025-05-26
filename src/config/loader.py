# src/config/loader.py
"""
Ce module est responsable du chargement de toutes les configurations de l'application,
de l'initialisation du système de logging, et de la fourniture de l'instance
AppConfig globale.
"""
import json
import logging
import os
import sys
import dataclasses # Pour fields, is_dataclass
from typing import Dict, Optional, Any, List, Union, Type, Tuple, get_origin, get_args, cast # Ajout de cast
from pathlib import Path
from dotenv import load_dotenv

# Tentative d'importation des définitions de dataclasses
# Utilisation de TYPE_CHECKING pour éviter les imports circulaires au runtime,
# tout en permettant à mypy de vérifier les types.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.config.definitions import (
        PathsConfig, LoggingConfig, SimulationDefaults, WfoSettings, OptunaSettings,
        GlobalConfig, SourceDetails, AssetsAndTimeframes, HistoricalPeriod,
        FetchingOptions, DataConfig, ParamDetail, StrategyParamsConfig, StrategiesConfig,
        AccountConfig, GlobalLiveSettings, OverrideRiskSettings, StrategyDeployment,
        LiveLoggingConfig, LiveFetchConfig, LiveConfig, ApiKeys, AppConfig, ExchangeSettings,
        SamplerPrunerProfile # Ajouté SamplerPrunerProfile
    )
    from src.utils.logging_setup import setup_logging # Importation pour type hinting
else:
    # Si ce n'est pas pour le type checking, on peut essayer un import direct
    # mais cela suppose que definitions.py n'importe pas loader.py
    # (ce qui devrait être le cas).
    from src.config.definitions import (
        PathsConfig, LoggingConfig, SimulationDefaults, WfoSettings, OptunaSettings,
        GlobalConfig, SourceDetails, AssetsAndTimeframes, HistoricalPeriod,
        FetchingOptions, DataConfig, ParamDetail, StrategyParamsConfig, StrategiesConfig,
        AccountConfig, GlobalLiveSettings, OverrideRiskSettings, StrategyDeployment,
        LiveLoggingConfig, LiveFetchConfig, LiveConfig, ApiKeys, AppConfig, ExchangeSettings,
        SamplerPrunerProfile # Ajouté SamplerPrunerProfile
    )
    from src.utils.logging_setup import setup_logging


# Logger pour ce module. Il sera configuré par setup_logging une fois la config chargée.
# Avant cela, les messages iront à la configuration par défaut de Python (généralement WARNING vers stderr).
logger = logging.getLogger(__name__)

def _load_json(file_path: Path) -> Dict[str, Any]:
    """
    Charge un fichier JSON et retourne son contenu sous forme de dictionnaire.

    Args:
        file_path (Path): Le chemin vers le fichier JSON.

    Returns:
        Dict[str, Any]: Le contenu du fichier JSON.

    Raises:
        FileNotFoundError: Si le fichier n'est pas trouvé.
        json.JSONDecodeError: Si le fichier JSON est mal formaté.
    """
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
    """
    Crée récursivement une instance de dataclass à partir d'un dictionnaire.
    Gère les dataclasses imbriquées, les listes de dataclasses, et les champs optionnels.

    Args:
        dataclass_type (Type[Any]): Le type de la dataclass à instancier.
        data (Dict[str, Any]): Le dictionnaire de données.
        config_file_path (str): Chemin du fichier de configuration source (pour logging).

    Returns:
        Any: Une instance de la dataclass peuplée.
    """
    if not dataclasses.is_dataclass(dataclass_type):
        # Si ce n'est pas une dataclass, cela pourrait être un type simple ou un Dict/List générique.
        # Laissons Python gérer la conversion de type plus tard si nécessaire, ou retourner la donnée brute.
        logger.debug(f"Type {dataclass_type} n'est pas une dataclass. Retour de la donnée brute : {data}")
        return data

    field_info_map = {f.name: f for f in dataclasses.fields(dataclass_type)}
    init_kwargs: Dict[str, Any] = {}

    # Appliquer les valeurs par défaut en premier
    for field_name, field_obj in field_info_map.items():
        if field_obj.default is not dataclasses.MISSING:
            init_kwargs[field_name] = field_obj.default
        elif field_obj.default_factory is not dataclasses.MISSING:
            init_kwargs[field_name] = field_obj.default_factory()
        # Si ni default ni default_factory, et que le champ n'est pas dans data,
        # une erreur sera levée lors de l'instanciation de la dataclass si le champ est requis.

    # Peupler avec les données du dictionnaire JSON
    for name_from_json, value_from_json in data.items():
        if name_from_json not in field_info_map:
            logger.debug(f"Dans {config_file_path} pour {dataclass_type.__name__}: Clé JSON '{name_from_json}' non trouvée comme champ. Ignorée.")
            continue

        field_obj = field_info_map[name_from_json]
        # Déterminer le type réel du champ, en gérant Optional[T]
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
            # Gère Dict[K, DataclassType]
            dict_value_dc_type = type_args[1]
            processed_dict = {}
            for k, v_item in value_from_json.items():
                if isinstance(v_item, dict):
                    processed_dict[k] = _create_dataclass_from_dict(dict_value_dc_type, v_item, config_file_path)
                else: # Si la valeur n'est pas un dict, on la prend telle quelle (peut être un type simple)
                    processed_dict[k] = v_item
            init_kwargs[name_from_json] = processed_dict
        elif origin_type is dict and isinstance(value_from_json, dict): # Gère Dict[K, V] où V n'est pas une dataclass
            init_kwargs[name_from_json] = value_from_json
        elif origin_type is tuple and type_args and isinstance(value_from_json, list) and len(value_from_json) == len(type_args):
             # Gère Tuple[type1, type2, ...]
            try:
                init_kwargs[name_from_json] = tuple(
                    arg_type(val) if not dataclasses.is_dataclass(arg_type) else _create_dataclass_from_dict(arg_type, cast(dict, val), config_file_path)
                    for arg_type, val in zip(type_args, value_from_json)
                )
            except (ValueError, TypeError) as e_tuple_conv:
                logger.warning(f"Dans {config_file_path} pour {dataclass_type.__name__}: "
                               f"Impossible de convertir la valeur '{value_from_json}' pour le champ tuple '{name_from_json}'. "
                               f"Types attendus: {type_args}. Erreur: {e_tuple_conv}")
                init_kwargs[name_from_json] = value_from_json # Fallback
        else: # Types de base ou autres types gérés par Python directement
            try:
                # Conversion de type explicite si nécessaire, sinon assigner directement
                if actual_field_type_for_conversion == int and not isinstance(value_from_json, bool):
                    init_kwargs[name_from_json] = int(value_from_json)
                elif actual_field_type_for_conversion == float:
                    init_kwargs[name_from_json] = float(value_from_json)
                elif actual_field_type_for_conversion == bool:
                    init_kwargs[name_from_json] = bool(value_from_json)
                elif actual_field_type_for_conversion == str:
                    init_kwargs[name_from_json] = str(value_from_json)
                else: # Si le type est déjà correct ou un type complexe non-dataclass
                    init_kwargs[name_from_json] = value_from_json
            except (ValueError, TypeError) as e_conv:
                logger.warning(f"Dans {config_file_path} pour {dataclass_type.__name__}: "
                               f"Impossible de convertir la valeur '{value_from_json}' pour le champ '{name_from_json}' vers {actual_field_type_for_conversion}. "
                               f"Utilisation de la valeur originale. Erreur: {e_conv}")
                init_kwargs[name_from_json] = value_from_json
    try:
        return dataclass_type(**init_kwargs)
    except Exception as e_init:
        logger.error(f"Erreur lors de l'instanciation de {dataclass_type.__name__} depuis {config_file_path} avec les clés {list(init_kwargs.keys())} et valeurs partielles: { {k: (type(v), v[:50] if isinstance(v,str) else v) for k,v in init_kwargs.items()} }: {e_init}", exc_info=True)
        raise ValueError(f"Échec de l'instanciation de {dataclass_type.__name__}: {e_init}") from e_init

def _validate_app_config(app_config: 'AppConfig', project_root_path: Path):
    """
    Valide l'instance AppConfig complètement chargée.
    Lève une ValueError si la validation échoue.
    """
    logger.info("Validation de AppConfig...")
    errors: List[str] = []

    # Les validations __post_init__ dans chaque dataclass gèrent déjà beaucoup de choses.
    # Ici, on peut ajouter des validations inter-configurations ou des vérifications
    # qui dépendent de l'état complet de AppConfig.

    # Exemple : Vérifier que les alias de compte dans StrategyDeployment existent dans accounts_config
    if app_config.live_config and app_config.live_config.strategy_deployments and app_config.accounts_config:
        defined_account_aliases = {acc.account_alias for acc in app_config.accounts_config}
        for i, deployment in enumerate(app_config.live_config.strategy_deployments):
            if deployment.active and deployment.account_alias_to_use not in defined_account_aliases:
                errors.append(
                    f"StrategyDeployment[{i}] (ID: {deployment.strategy_id}): "
                    f"account_alias_to_use '{deployment.account_alias_to_use}' "
                    f"n'est pas défini dans accounts_config."
                )
            # Vérifier que results_config_path est un fichier existant
            if deployment.active:
                results_file = project_root_path / deployment.results_config_path
                if not results_file.is_file():
                    errors.append(
                        f"StrategyDeployment[{i}] (ID: {deployment.strategy_id}): "
                        f"results_config_path '{results_file}' n'existe pas ou n'est pas un fichier."
                    )

    # Vérifier que les clés API sont présentes si les variables d'environnement sont spécifiées
    if app_config.accounts_config and app_config.api_keys:
        for acc_cfg in app_config.accounts_config:
            creds = app_config.api_keys.credentials.get(acc_cfg.account_alias)
            if acc_cfg.api_key_env_var and (not creds or not creds[0]):
                errors.append(f"Clé API pour '{acc_cfg.account_alias}' via env var '{acc_cfg.api_key_env_var}' non trouvée ou vide.")
            if acc_cfg.api_secret_env_var and (not creds or not creds[1]):
                errors.append(f"Secret API pour '{acc_cfg.account_alias}' via env var '{acc_cfg.api_secret_env_var}' non trouvé ou vide.")

    # Vérifier l'existence du fichier d'info de l'exchange
    if app_config.exchange_settings:
        exchange_info_path = project_root_path / app_config.exchange_settings.exchange_info_file_path
        if not exchange_info_path.is_file():
            errors.append(f"Fichier d'information de l'exchange '{exchange_info_path}' non trouvé.")

    if errors:
        error_summary = "\n - ".join(["La validation de AppConfig a échoué :"] + errors)
        logger.error(error_summary)
        raise ValueError(error_summary)

    logger.info("Validation de AppConfig réussie.")

def load_all_configs(project_root: Optional[str] = None) -> 'AppConfig':
    """
    Charge toutes les configurations JSON, les variables d'environnement pour les clés API,
    instancie les dataclasses, initialise le logging, et valide l'instance AppConfig finale.

    Args:
        project_root (Optional[str]): Le chemin vers la racine du projet.
            Si None, il est déterminé automatiquement.

    Returns:
        AppConfig: L'instance AppConfig pleinement peuplée et validée.
    """
    if project_root is None:
        try:
            # Supposer que ce script (loader.py) est dans src/config/
            project_root_path = Path(__file__).resolve().parent.parent.parent
        except NameError: # Fallback si __file__ n'est pas défini (ex: interactif)
            project_root_path = Path('.').resolve()
            # Utiliser print ici car le logger n'est pas encore configuré
            print(f"INFO: __file__ non défini, racine du projet détectée comme répertoire courant : {project_root_path}")
    else:
        project_root_path = Path(project_root).resolve()

    # Message initial avant que le logger soit configuré
    print(f"INFO: Racine du projet déterminée comme : {project_root_path}")

    # Charger le fichier .env depuis la racine du projet
    env_path = project_root_path / '.env'
    if env_path.is_file():
        loaded_env = load_dotenv(dotenv_path=env_path, verbose=True)
        print(f"INFO: Fichier .env chargé depuis {env_path}" if loaded_env else f"WARN: Fichier .env à {env_path} trouvé mais python-dotenv n'a pas pu le charger.")
    else:
        print(f"WARN: Fichier .env non trouvé à {env_path}. Les clés API doivent être définies directement dans l'environnement.")

    config_dir = project_root_path / 'config'
    global_config_path = config_dir / 'config_global.json'
    data_config_path = config_dir / 'config_data.json'
    strategies_config_path = config_dir / 'config_strategies.json'
    live_config_path = config_dir / 'config_live.json'
    accounts_config_path = config_dir / 'config_accounts.json'

    # Charger les fichiers JSON et créer les instances de dataclass
    # GlobalConfig
    print(f"INFO: Chargement de global_config depuis : {global_config_path}")
    global_config_dict = _load_json(global_config_path)
    global_cfg: GlobalConfig = _create_dataclass_from_dict(GlobalConfig, global_config_dict, str(global_config_path))

    # DataConfig
    print(f"INFO: Chargement de data_config depuis : {data_config_path}")
    data_config_dict = _load_json(data_config_path)
    data_cfg: DataConfig = _create_dataclass_from_dict(DataConfig, data_config_dict, str(data_config_path))

    # StrategiesConfig
    print(f"INFO: Chargement de strategies_config depuis : {strategies_config_path}")
    strategies_config_dict_raw = _load_json(strategies_config_path)
    strategies_cfg: StrategiesConfig = _create_dataclass_from_dict(StrategiesConfig, {"strategies": strategies_config_dict_raw}, str(strategies_config_path))

    # LiveConfig
    print(f"INFO: Chargement de live_config depuis : {live_config_path}")
    live_config_dict = _load_json(live_config_path)
    live_cfg: LiveConfig = _create_dataclass_from_dict(LiveConfig, live_config_dict, str(live_config_path))

    # AccountConfig (liste)
    print(f"INFO: Chargement de accounts_config depuis : {accounts_config_path}")
    accounts_config_list_raw = _load_json(accounts_config_path)
    if not isinstance(accounts_config_list_raw, list):
        raise ValueError(f"{accounts_config_path} doit contenir une liste JSON de configurations de compte.")
    accounts_cfg_list: List[AccountConfig] = [
        _create_dataclass_from_dict(AccountConfig, acc_dict, str(accounts_config_path))
        for acc_dict in accounts_config_list_raw
    ]

    # Peupler ApiKeys
    api_key_credentials: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for acc_conf in accounts_cfg_list:
        key_env_var = acc_conf.api_key_env_var
        secret_env_var = acc_conf.api_secret_env_var
        key: Optional[str] = os.getenv(key_env_var) if key_env_var else None
        secret: Optional[str] = os.getenv(secret_env_var) if secret_env_var else None
        api_key_credentials[acc_conf.account_alias] = (key, secret)
        if key_env_var and not key: print(f"WARN: Variable d'environnement '{key_env_var}' pour la clé API du compte '{acc_conf.account_alias}' non trouvée ou vide.")
        if secret_env_var and not secret: print(f"WARN: Variable d'environnement '{secret_env_var}' pour le secret API du compte '{acc_conf.account_alias}' non trouvée ou vide.")
    api_keys_instance = ApiKeys(credentials=api_key_credentials)

    # Résoudre les chemins dans PathsConfig et créer les répertoires
    if global_cfg.paths:
        print("INFO: Résolution des chemins dans PathsConfig en chemins absolus...")
        for path_field_obj in dataclasses.fields(global_cfg.paths):
            field_name = path_field_obj.name
            relative_path_val = getattr(global_cfg.paths, field_name)
            if isinstance(relative_path_val, str):
                absolute_path = (project_root_path / relative_path_val).resolve()
                setattr(global_cfg.paths, field_name, str(absolute_path))
                try:
                    absolute_path.mkdir(parents=True, exist_ok=True)
                except OSError as e_mkdir:
                    print(f"ERROR: Échec de la création du répertoire {absolute_path}: {e_mkdir}")
            else:
                print(f"WARN: La valeur du chemin pour '{field_name}' n'est pas une chaîne. Résolution ignorée.")

    # Initialiser le logging
    # Le logger de ce module (loader.py) commencera à utiliser cette configuration.
    # Les messages "print" précédents étaient nécessaires car le logging n'était pas encore prêt.
    log_conf_to_use: Union[LoggingConfig, LiveLoggingConfig] = global_cfg.logging
    log_dir_for_setup = Path(global_cfg.paths.logs_backtest_optimization) # Ou un autre chemin de log par défaut
    log_filename_for_setup = global_cfg.logging.log_filename_global

    # Vérifier si on est dans un contexte live pour utiliser LiveLoggingConfig
    # Pour l'instant, on utilise LoggingConfig global par défaut.
    # Si un script spécifique (ex: run_live.py) a besoin d'une config de log différente,
    # il peut appeler setup_logging à nouveau avec LiveLoggingConfig.
    setup_logging(
        log_config=log_conf_to_use,
        log_dir=log_dir_for_setup,
        log_filename=log_filename_for_setup,
        root_level=getattr(logging, global_cfg.logging.level.upper(), logging.INFO)
    )
    logger.info("Système de logging initialisé avec la configuration globale.") # Ce message utilisera le logger configuré.

    # Charger ExchangeSettings (par défaut pour l'instant)
    exchange_settings_instance = ExchangeSettings()
    exchange_info_path = project_root_path / exchange_settings_instance.exchange_info_file_path
    if not exchange_info_path.is_file():
        logger.warning(f"Fichier d'info de l'exchange non trouvé à {exchange_info_path}. Certaines fonctionnalités pourraient être limitées.")
        # Ne pas lever d'erreur ici, car le fichier est optionnel pour certaines opérations.
        # La validation dans _validate_app_config le signalera si c'est critique.

    # Assembler AppConfig
    app_config = AppConfig(
        global_config=global_cfg,
        data_config=data_cfg,
        strategies_config=strategies_cfg,
        live_config=live_cfg,
        accounts_config=accounts_cfg_list,
        api_keys=api_keys_instance,
        exchange_settings=exchange_settings_instance,
        project_root=str(project_root_path)
    )

    # Valider AppConfig
    _validate_app_config(app_config, project_root_path)

    logger.info(f"Toutes les configurations ont été chargées et validées avec succès pour le projet : {app_config.global_config.project_name}")
    return app_config

def load_strategy_config_by_name(strategy_name: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Charge la configuration pour une stratégie spécifique par son nom depuis config_strategies.json.
    Retourne un dictionnaire des paramètres de la stratégie.

    Args:
        strategy_name (str): Le nom de la stratégie à charger.
        project_root (Optional[str]): Le chemin vers la racine du projet.

    Returns:
        Dict[str, Any]: La configuration de la stratégie.
    """
    if project_root is None:
        project_root_path = Path(__file__).resolve().parent.parent.parent
    else:
        project_root_path = Path(project_root).resolve()

    strategies_config_path = project_root_path / 'config' / 'config_strategies.json'
    
    all_strategies_dict = _load_json(strategies_config_path)
    strategy_config_data = all_strategies_dict.get(strategy_name)
    
    if strategy_config_data is None:
        logger.error(f"Configuration pour la stratégie '{strategy_name}' non trouvée dans {strategies_config_path}")
        raise ValueError(f"Configuration pour la stratégie '{strategy_name}' non trouvée.")
        
    if not isinstance(strategy_config_data, dict):
        logger.error(f"La configuration pour la stratégie '{strategy_name}' n'est pas un dictionnaire valide.")
        raise TypeError(f"La configuration pour la stratégie '{strategy_name}' doit être un dictionnaire.")
    
    logger.debug(f"Configuration chargée pour la stratégie '{strategy_name}'.")
    return strategy_config_data

def load_exchange_config(exchange_info_file_path: Optional[Union[str, Path]] = None, project_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Charge les informations de l'exchange (ex: filtres de symboles, précisions)
    depuis un fichier JSON.

    Args:
        exchange_info_file_path (Optional[Union[str, Path]]): Chemin vers le fichier d'info.
            Si None, utilise le chemin par défaut de ExchangeSettings.
        project_root (Optional[str]): Racine du projet pour résoudre les chemins relatifs.

    Returns:
        Dict[str, Any]: Les informations de l'exchange.
    """
    root_path: Path
    if project_root is None:
        root_path = Path(__file__).resolve().parent.parent.parent
    else:
        root_path = Path(project_root).resolve()

    file_path_to_load: Path
    if exchange_info_file_path:
        file_path_to_load = Path(exchange_info_file_path)
        if not file_path_to_load.is_absolute():
            file_path_to_load = root_path / file_path_to_load
    else:
        # Utiliser le chemin par défaut de ExchangeSettings (qui est relatif à 'config')
        default_path_from_def = ExchangeSettings().exchange_info_file_path
        file_path_to_load = root_path / default_path_from_def
    
    logger.debug(f"Tentative de chargement des informations de l'exchange depuis : {file_path_to_load}")
    
    try:
        exchange_data = _load_json(file_path_to_load)
        if not isinstance(exchange_data, dict) or "symbols" not in exchange_data:
            logger.error(f"Le contenu de {file_path_to_load} n'est pas un dictionnaire JSON valide ou manque la clé 'symbols'.")
            return {"symbols": []} # Retourner une structure de base vide
        logger.info(f"Informations de l'exchange chargées avec succès depuis {file_path_to_load}.")
        return exchange_data
    except FileNotFoundError:
        logger.error(f"Fichier d'information de l'exchange non trouvé à : {file_path_to_load}. Retour d'une config vide.")
        return {"symbols": []}
    except Exception as e:
        logger.error(f"Erreur lors du chargement ou du parsing du fichier d'info de l'exchange {file_path_to_load}: {e}", exc_info=True)
        return {"symbols": []}

