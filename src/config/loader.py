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
from typing import Dict, Optional, Any, List, Union, Type, Tuple, get_origin, get_args, cast 
from pathlib import Path
from dotenv import load_dotenv

# Tentative d'importation des définitions de dataclasses
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.config.definitions import (
        PathsConfig, LoggingConfig, SimulationDefaults, WfoSettings, OptunaSettings,
        GlobalConfig, SourceDetails, AssetsAndTimeframes, HistoricalPeriod,
        FetchingOptions, DataConfig, ParamDetail, StrategyParamsConfig, StrategiesConfig,
        AccountConfig, GlobalLiveSettings, OverrideRiskSettings, StrategyDeployment,
        LiveLoggingConfig, LiveFetchConfig, LiveConfig, ApiKeys, AppConfig, ExchangeSettings,
        SamplerPrunerProfile
    )
    from src.utils.logging_setup import setup_logging 
    # from src.data.data_utils import get_kline_prefix_effective # Pas directement utilisé ici mais par les validateurs
else:
    from src.config.definitions import (
        PathsConfig, LoggingConfig, SimulationDefaults, WfoSettings, OptunaSettings,
        GlobalConfig, SourceDetails, AssetsAndTimeframes, HistoricalPeriod,
        FetchingOptions, DataConfig, ParamDetail, StrategyParamsConfig, StrategiesConfig,
        AccountConfig, GlobalLiveSettings, OverrideRiskSettings, StrategyDeployment,
        LiveLoggingConfig, LiveFetchConfig, LiveConfig, ApiKeys, AppConfig, ExchangeSettings,
        SamplerPrunerProfile
    )
    from src.utils.logging_setup import setup_logging
    # from src.data.data_utils import get_kline_prefix_effective


logger = logging.getLogger(__name__)

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
        logger.debug(f"Type {dataclass_type} n'est pas une dataclass. Retour de la donnée brute : {data}")
        return data

    field_info_map = {f.name: f for f in dataclasses.fields(dataclass_type)}
    init_kwargs: Dict[str, Any] = {}

    for field_name, field_obj in field_info_map.items():
        if field_obj.default is not dataclasses.MISSING:
            init_kwargs[field_name] = field_obj.default
        elif field_obj.default_factory is not dataclasses.MISSING:
            init_kwargs[field_name] = field_obj.default_factory()

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
                    processed_dict[k] = v_item
            init_kwargs[name_from_json] = processed_dict
        elif origin_type is dict and isinstance(value_from_json, dict): 
            init_kwargs[name_from_json] = value_from_json
        elif origin_type is tuple and type_args and isinstance(value_from_json, list) and len(value_from_json) == len(type_args):
            try:
                init_kwargs[name_from_json] = tuple(
                    arg_type(val) if not dataclasses.is_dataclass(arg_type) else _create_dataclass_from_dict(arg_type, cast(dict, val), config_file_path)
                    for arg_type, val in zip(type_args, value_from_json)
                )
            except (ValueError, TypeError) as e_tuple_conv:
                logger.warning(f"Dans {config_file_path} pour {dataclass_type.__name__}: "
                               f"Impossible de convertir la valeur '{value_from_json}' pour le champ tuple '{name_from_json}'. "
                               f"Types attendus: {type_args}. Erreur: {e_tuple_conv}")
                init_kwargs[name_from_json] = value_from_json 
        else: 
            try:
                if actual_field_type_for_conversion == int and not isinstance(value_from_json, bool):
                    init_kwargs[name_from_json] = int(value_from_json)
                elif actual_field_type_for_conversion == float:
                    init_kwargs[name_from_json] = float(value_from_json)
                elif actual_field_type_for_conversion == bool:
                    init_kwargs[name_from_json] = bool(value_from_json)
                elif actual_field_type_for_conversion == str:
                    init_kwargs[name_from_json] = str(value_from_json)
                else: 
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

def _validate_strategy_configurations(app_config: 'AppConfig', project_root_path: Path) -> List[str]:
    """
    Valide les configurations des stratégies dans AppConfig.strategies_config.
    Vérifie l'existence des scripts, la cohérence des fréquences d'indicateurs,
    et la validité des ParamDetail.

    Args:
        app_config (AppConfig): L'instance AppConfig à valider.
        project_root_path (Path): Le chemin racine du projet.

    Returns:
        List[str]: Une liste de messages d'erreur. Vide si aucune erreur.
    """
    errors: List[str] = []
    log_prefix_val_strat = "[ValidateStrategyConfigs]"
    logger.info(f"{log_prefix_val_strat} Démarrage de la validation des configurations de stratégies...")

    if not app_config.strategies_config or not app_config.strategies_config.strategies:
        logger.info(f"{log_prefix_val_strat} Aucune stratégie configurée. Validation sautée.")
        return errors

    available_timeframes_from_config = set(app_config.data_config.assets_and_timeframes.timeframes)
    if not available_timeframes_from_config:
        # Ne pas ajouter d'erreur ici si c'est vide, car cela pourrait être valide pour certaines configurations
        # (par exemple, si aucune stratégie n'utilise de fréquences spécifiques).
        # La validation par paramètre ci-dessous lèvera une erreur si une fréquence est utilisée mais non disponible.
        logger.warning(f"{log_prefix_val_strat} data_config.assets_and_timeframes.timeframes est vide. "
                       "La validation des fréquences d'indicateurs pourrait être limitée.")

    for strat_name, strat_cfg in app_config.strategies_config.strategies.items():
        strat_log_prefix = f"{log_prefix_val_strat}[{strat_name}]"
        
        # 1. Vérifier script_reference
        if not strat_cfg.script_reference or not isinstance(strat_cfg.script_reference, str):
            errors.append(f"{strat_log_prefix} 'script_reference' manquant ou invalide.")
        else:
            script_file_path = project_root_path / strat_cfg.script_reference
            if not script_file_path.is_file():
                # Si la stratégie n'est pas active pour l'optimisation, logguer un avertissement au lieu d'une erreur bloquante.
                if strat_cfg.active_for_optimization:
                    errors.append(f"{strat_log_prefix} Fichier de script '{script_file_path}' non trouvé pour 'script_reference' (stratégie active).")
                else:
                    logger.warning(f"{strat_log_prefix} Fichier de script '{script_file_path}' non trouvé pour 'script_reference' (stratégie inactive).")


        # 2. Vérifier class_name
        if not strat_cfg.class_name or not isinstance(strat_cfg.class_name, str):
            errors.append(f"{strat_log_prefix} 'class_name' manquant ou invalide.")

        # 3. Valider params_space et default_params pour les fréquences d'indicateurs
        frequency_param_names = [
            key for key in (strat_cfg.default_params or {}) 
            if "frequence" in key.lower() or "frequency" in key.lower()
        ]
        if strat_cfg.params_space:
            frequency_param_names.extend([
                key for key in strat_cfg.params_space 
                if "frequence" in key.lower() or "frequency" in key.lower()
            ])
        frequency_param_names = sorted(list(set(frequency_param_names))) 

        if not available_timeframes_from_config and frequency_param_names:
            # Si des paramètres de fréquence existent mais qu'aucun timeframe n'est défini globalement, c'est une erreur.
             errors.append(f"{strat_log_prefix} Des paramètres de fréquence d'indicateur sont définis, mais "
                           "data_config.assets_and_timeframes.timeframes est vide.")

        if available_timeframes_from_config: 
            if strat_cfg.default_params:
                for param_name, param_value in strat_cfg.default_params.items():
                    if param_name in frequency_param_names:
                        if not isinstance(param_value, str) or param_value not in available_timeframes_from_config:
                            errors.append(f"{strat_log_prefix} default_params: Fréquence '{param_value}' pour '{param_name}' "
                                          f"n'est pas dans les timeframes disponibles: {available_timeframes_from_config}.")
            
            if strat_cfg.params_space:
                for param_name, param_detail in strat_cfg.params_space.items():
                    if param_name in frequency_param_names and param_detail.type == "categorical":
                        if not param_detail.choices:
                            errors.append(f"{strat_log_prefix} params_space: '{param_name}' est de type catégoriel mais 'choices' est vide ou manquant.")
                            continue
                        for choice_freq in param_detail.choices:
                            if not isinstance(choice_freq, str) or choice_freq not in available_timeframes_from_config:
                                errors.append(f"{strat_log_prefix} params_space: Fréquence '{choice_freq}' dans les choix de '{param_name}' "
                                              f"n'est pas dans les timeframes disponibles: {available_timeframes_from_config}.")
        
        if strat_cfg.default_params and strat_cfg.params_space:
            for param_name, default_value in strat_cfg.default_params.items():
                if param_name in strat_cfg.params_space:
                    p_detail = strat_cfg.params_space[param_name]
                    if p_detail.type == "categorical":
                        if p_detail.choices and default_value not in p_detail.choices:
                            errors.append(f"{strat_log_prefix} default_params: Valeur par défaut '{default_value}' pour '{param_name}' "
                                          f"n'est pas dans les 'choices' de params_space: {p_detail.choices}.")
                    elif p_detail.type in ["int", "float"]:
                        is_valid_range = True
                        if p_detail.low is not None and default_value < p_detail.low:
                            errors.append(f"{strat_log_prefix} default_params: Valeur par défaut '{default_value}' pour '{param_name}' "
                                          f"est inférieure à 'low' ({p_detail.low}) de params_space.")
                            is_valid_range = False
                        if p_detail.high is not None and default_value > p_detail.high:
                            errors.append(f"{strat_log_prefix} default_params: Valeur par défaut '{default_value}' pour '{param_name}' "
                                          f"est supérieure à 'high' ({p_detail.high}) de params_space.")
                            is_valid_range = False
                        
                        if is_valid_range and p_detail.step is not None and p_detail.low is not None:
                            # Vérifier si default_value est un multiple de step par rapport à low
                            # (default_value - low) % step == 0
                            # Gérer les problèmes de flottants avec une petite tolérance
                            if isinstance(default_value, (int, float)) and isinstance(p_detail.low, (int, float)) and isinstance(p_detail.step, (int, float)) and p_detail.step > 1e-9:
                                remainder = (Decimal(str(default_value)) - Decimal(str(p_detail.low))) % Decimal(str(p_detail.step))
                                if not (remainder.is_zero() or remainder.is_nan() or abs(remainder) < Decimal('1e-9') or abs(remainder - Decimal(str(p_detail.step))) < Decimal('1e-9')):
                                     errors.append(f"{strat_log_prefix} default_params: Valeur par défaut '{default_value}' pour '{param_name}' "
                                                   f"ne respecte pas le 'step' ({p_detail.step}) par rapport à 'low' ({p_detail.low}). Reste: {remainder}")


    if errors:
        logger.error(f"{log_prefix_val_strat} Erreurs de validation trouvées dans config_strategies.json.")
    else:
        logger.info(f"{log_prefix_val_strat} Validation des configurations de stratégies terminée avec succès.")
    return errors


def _validate_app_config(app_config: 'AppConfig', project_root_path: Path):
    """Valide l'instance AppConfig complètement chargée."""
    logger.info("Validation de AppConfig...")
    errors: List[str] = []

    if app_config.live_config and app_config.live_config.strategy_deployments and app_config.accounts_config:
        defined_account_aliases = {acc.account_alias for acc in app_config.accounts_config}
        for i, deployment in enumerate(app_config.live_config.strategy_deployments):
            if deployment.active and deployment.account_alias_to_use not in defined_account_aliases:
                errors.append(
                    f"StrategyDeployment[{i}] (ID: {deployment.strategy_id}): "
                    f"account_alias_to_use '{deployment.account_alias_to_use}' "
                    f"n'est pas défini dans accounts_config."
                )
            if deployment.active:
                # Le chemin est relatif à project_root
                results_file = project_root_path / deployment.results_config_path
                if not results_file.is_file():
                    errors.append(
                        f"StrategyDeployment[{i}] (ID: {deployment.strategy_id}): "
                        f"results_config_path '{results_file}' (résolu depuis '{deployment.results_config_path}') "
                        "n'existe pas ou n'est pas un fichier."
                    )

    if app_config.accounts_config and app_config.api_keys:
        for acc_cfg in app_config.accounts_config:
            creds = app_config.api_keys.credentials.get(acc_cfg.account_alias)
            if acc_cfg.api_key_env_var and (not creds or not creds[0]):
                errors.append(f"Clé API pour '{acc_cfg.account_alias}' via env var '{acc_cfg.api_key_env_var}' non trouvée ou vide.")
            if acc_cfg.api_secret_env_var and (not creds or not creds[1]):
                errors.append(f"Secret API pour '{acc_cfg.account_alias}' via env var '{acc_cfg.api_secret_env_var}' non trouvé ou vide.")

    if app_config.exchange_settings:
        # Le chemin est relatif à project_root
        exchange_info_path = project_root_path / app_config.exchange_settings.exchange_info_file_path
        if not exchange_info_path.is_file():
            errors.append(f"Fichier d'information de l'exchange '{exchange_info_path}' (résolu depuis '{app_config.exchange_settings.exchange_info_file_path}') non trouvé.")
    
    # Appel de la validation spécifique des configurations de stratégies
    strategy_config_errors = _validate_strategy_configurations(app_config, project_root_path)
    errors.extend(strategy_config_errors)

    if errors:
        error_summary = "\n - ".join(["La validation de AppConfig a échoué :"] + errors)
        logger.error(error_summary)
        raise ValueError(error_summary)

    logger.info("Validation de AppConfig réussie.")

def load_all_configs(project_root: Optional[str] = None) -> 'AppConfig':
    """Charge toutes les configurations JSON, les variables d'environnement pour les clés API,
    instancie les dataclasses, initialise le logging, et valide l'instance AppConfig finale."""
    if project_root is None:
        try:
            project_root_path = Path(__file__).resolve().parent.parent.parent
        except NameError: 
            project_root_path = Path('.').resolve()
            print(f"INFO: __file__ non défini, racine du projet détectée comme répertoire courant : {project_root_path}")
    else:
        project_root_path = Path(project_root).resolve()

    print(f"INFO: Racine du projet déterminée comme : {project_root_path}")

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

    print(f"INFO: Chargement de global_config depuis : {global_config_path}")
    global_config_dict = _load_json(global_config_path)
    global_cfg: GlobalConfig = _create_dataclass_from_dict(GlobalConfig, global_config_dict, str(global_config_path))

    print(f"INFO: Chargement de data_config depuis : {data_config_path}")
    data_config_dict = _load_json(data_config_path)
    data_cfg: DataConfig = _create_dataclass_from_dict(DataConfig, data_config_dict, str(data_config_path))

    print(f"INFO: Chargement de strategies_config depuis : {strategies_config_path}")
    strategies_config_dict_raw = _load_json(strategies_config_path)
    strategies_cfg: StrategiesConfig = _create_dataclass_from_dict(StrategiesConfig, {"strategies": strategies_config_dict_raw}, str(strategies_config_path))

    print(f"INFO: Chargement de live_config depuis : {live_config_path}")
    live_config_dict = _load_json(live_config_path)
    live_cfg: LiveConfig = _create_dataclass_from_dict(LiveConfig, live_config_dict, str(live_config_path))

    print(f"INFO: Chargement de accounts_config depuis : {accounts_config_path}")
    accounts_config_list_raw = _load_json(accounts_config_path)
    if not isinstance(accounts_config_list_raw, list):
        raise ValueError(f"{accounts_config_path} doit contenir une liste JSON de configurations de compte.")
    accounts_cfg_list: List[AccountConfig] = [
        _create_dataclass_from_dict(AccountConfig, acc_dict, str(accounts_config_path))
        for acc_dict in accounts_config_list_raw
    ]

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

    if global_cfg.paths:
        print("INFO: Résolution des chemins dans PathsConfig en chemins absolus...")
        for path_field_obj in dataclasses.fields(global_cfg.paths):
            field_name = path_field_obj.name
            relative_path_val = getattr(global_cfg.paths, field_name)
            if isinstance(relative_path_val, str):
                absolute_path = (project_root_path / relative_path_val).resolve()
                setattr(global_cfg.paths, field_name, str(absolute_path))
                try:
                    if "log" in field_name.lower() or "result" in field_name.lower() or "state" in field_name.lower():
                        absolute_path.mkdir(parents=True, exist_ok=True)
                except OSError as e_mkdir:
                    print(f"ERROR: Échec de la création du répertoire {absolute_path}: {e_mkdir}")
            else:
                print(f"WARN: La valeur du chemin pour '{field_name}' n'est pas une chaîne. Résolution ignorée.")

    log_conf_to_use: Union[LoggingConfig, LiveLoggingConfig] = global_cfg.logging
    log_dir_for_setup = Path(global_cfg.paths.logs_backtest_optimization) 
    log_filename_for_setup = global_cfg.logging.log_filename_global

    setup_logging(
        log_config=log_conf_to_use,
        log_dir=log_dir_for_setup,
        log_filename=log_filename_for_setup,
        root_level=getattr(logging, global_cfg.logging.level.upper(), logging.INFO)
    )
    logger.info("Système de logging initialisé avec la configuration globale.") 

    exchange_settings_instance = ExchangeSettings()
    # Le chemin dans ExchangeSettings est relatif à 'config' par défaut, donc on le joint à project_root_path
    exchange_info_path_resolved = project_root_path / exchange_settings_instance.exchange_info_file_path
    if not exchange_info_path_resolved.is_file():
        logger.warning(f"Fichier d'info de l'exchange non trouvé à {exchange_info_path_resolved} "
                       f"(configuré comme '{exchange_settings_instance.exchange_info_file_path}' "
                       "relativement à la racine du projet). Certaines fonctionnalités pourraient être limitées.")


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

    _validate_app_config(app_config, project_root_path)

    logger.info(f"Toutes les configurations ont été chargées et validées avec succès pour le projet : {app_config.global_config.project_name}")
    return app_config

def load_strategy_config_by_name(strategy_name: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    """Charge la configuration pour une stratégie spécifique par son nom depuis config_strategies.json."""
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
    """Charge les informations de l'exchange depuis un fichier JSON."""
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
        # exchange_info_file_path dans ExchangeSettings est relatif à la racine du projet.
        default_path_from_def = ExchangeSettings().exchange_info_file_path
        file_path_to_load = root_path / default_path_from_def
    
    logger.debug(f"Tentative de chargement des informations de l'exchange depuis : {file_path_to_load}")
    
    try:
        exchange_data = _load_json(file_path_to_load)
        if not isinstance(exchange_data, dict) or "symbols" not in exchange_data:
            logger.error(f"Le contenu de {file_path_to_load} n'est pas un dictionnaire JSON valide ou manque la clé 'symbols'.")
            return {"symbols": []} 
        logger.info(f"Informations de l'exchange chargées avec succès depuis {file_path_to_load}.")
        return exchange_data
    except FileNotFoundError:
        logger.error(f"Fichier d'information de l'exchange non trouvé à : {file_path_to_load}. Retour d'une config vide.")
        return {"symbols": []}
    except Exception as e:
        logger.error(f"Erreur lors du chargement ou du parsing du fichier d'info de l'exchange {file_path_to_load}: {e}", exc_info=True)
        return {"symbols": []}
