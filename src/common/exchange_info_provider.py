# src/common/exchange_info_provider.py
"""
Ce module fournit les informations de l'exchange (filtres de symboles, précisions)
pour une paire de trading donnée, avec gestion de cache et rafraîchissement API.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any, TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import AccountConfig # Pour le typage

# Imports depuis l'application
try:
    from src.live.execution import OrderExecutionClient # Pour interagir avec l'API de l'exchange
    from src.utils.exchange_utils import get_pair_config_for_symbol # Pour extraire les infos d'une paire
    from src.utils.file_utils import ensure_dir_exists # Pour créer le répertoire config si besoin
except ImportError as e:
    # Ce log est un fallback, le logging principal est configuré ailleurs
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"ExchangeInfoProvider: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH."
    )
    # Rendre la fonction principale inutilisable si les imports échouent
    def get_symbol_exchange_info(*args, **kwargs) -> Optional[Dict[str, Any]]:
        logging.getLogger(__name__).critical("Imports critiques échoués, get_symbol_exchange_info est inopérable.")
        return None
    raise # Renvoyer l'erreur pour arrêter si c'est un contexte critique

logger = logging.getLogger(__name__)

# Variable globale pour un cache en mémoire simple (optionnel, le cache fichier est principal)
_in_memory_exchange_info_cache: Optional[Dict[str, Any]] = None
_cache_file_path_checked: Optional[Path] = None

def _load_info_from_cache(cache_file_path: Path) -> Optional[Dict[str, Any]]:
    """Charge les informations de l'exchange depuis le fichier cache JSON."""
    global _in_memory_exchange_info_cache, _cache_file_path_checked
    log_prefix = "[ExchangeInfoCache]"

    if _in_memory_exchange_info_cache and _cache_file_path_checked == cache_file_path:
        logger.debug(f"{log_prefix} Utilisation du cache en mémoire pour {cache_file_path}.")
        return _in_memory_exchange_info_cache

    if cache_file_path.is_file():
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and "symbols" in data:
                logger.info(f"{log_prefix} Informations de l'exchange chargées avec succès depuis le cache : {cache_file_path}")
                _in_memory_exchange_info_cache = data
                _cache_file_path_checked = cache_file_path
                return data
            else:
                logger.warning(f"{log_prefix} Le fichier cache {cache_file_path} a un format invalide (doit être un dict avec une clé 'symbols').")
                _in_memory_exchange_info_cache = None # Invalider le cache mémoire
                _cache_file_path_checked = None
                return None
        except json.JSONDecodeError:
            logger.error(f"{log_prefix} Erreur de décodage JSON du fichier cache {cache_file_path}.")
            _in_memory_exchange_info_cache = None
            _cache_file_path_checked = None
            return None
        except Exception as e:
            logger.error(f"{log_prefix} Erreur inattendue lors de la lecture du fichier cache {cache_file_path}: {e}", exc_info=True)
            _in_memory_exchange_info_cache = None
            _cache_file_path_checked = None
            return None
    else:
        logger.info(f"{log_prefix} Fichier cache d'informations de l'exchange non trouvé : {cache_file_path}")
        return None

def _save_info_to_cache(data: Dict[str, Any], cache_file_path: Path) -> bool:
    """Sauvegarde les informations de l'exchange dans le fichier cache JSON."""
    global _in_memory_exchange_info_cache, _cache_file_path_checked
    log_prefix = "[ExchangeInfoCache]"
    try:
        # S'assurer que le répertoire parent existe
        ensure_dir_exists(cache_file_path.parent)
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2) # indent=2 pour une meilleure lisibilité
        logger.info(f"{log_prefix} Informations de l'exchange sauvegardées dans le cache : {cache_file_path}")
        _in_memory_exchange_info_cache = data # Mettre à jour le cache mémoire
        _cache_file_path_checked = cache_file_path
        return True
    except Exception as e:
        logger.error(f"{log_prefix} Échec de la sauvegarde des informations de l'exchange dans le cache {cache_file_path}: {e}", exc_info=True)
        return False

def _fetch_exchange_info_from_api(app_config: 'AppConfig') -> Optional[Dict[str, Any]]:
    """Récupère les informations complètes de l'exchange via l'API."""
    log_prefix = "[ExchangeInfoAPI]"
    logger.info(f"{log_prefix} Tentative de rafraîchissement des informations de l'exchange depuis l'API.")

    # Sélectionner un compte pour l'appel API (ex: premier compte Binance non-testnet)
    selected_account: Optional['AccountConfig'] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    if app_config.accounts_config:
        for acc_cfg in app_config.accounts_config:
            if acc_cfg.exchange.lower() == "binance" and not acc_cfg.is_testnet:
                creds = app_config.api_keys.credentials.get(acc_cfg.account_alias)
                if creds and creds[0] and creds[1]:
                    selected_account = acc_cfg
                    api_key, api_secret = creds
                    break
        # Fallback si aucun compte live non-testnet n'est trouvé, essayer un compte testnet
        if not selected_account:
            for acc_cfg in app_config.accounts_config:
                if acc_cfg.exchange.lower() == "binance": # Testnet ou non
                    creds = app_config.api_keys.credentials.get(acc_cfg.account_alias)
                    if creds and creds[0] and creds[1]:
                        selected_account = acc_cfg
                        api_key, api_secret = creds
                        logger.warning(f"{log_prefix} Utilisation du compte (potentiellement testnet) '{acc_cfg.account_alias}' car aucun compte live principal Binance n'a été trouvé avec des clés complètes.")
                        break
    
    if not selected_account or not api_key or not api_secret:
        logger.error(f"{log_prefix} Aucun compte Binance approprié avec des clés API complètes n'a été trouvé dans la configuration pour rafraîchir les informations de l'exchange.")
        return None

    logger.info(f"{log_prefix} Utilisation du compte '{selected_account.account_alias}' (Testnet: {selected_account.is_testnet}) pour l'appel API.")

    try:
        exec_client = OrderExecutionClient(
            api_key=api_key,
            api_secret=api_secret,
            account_type=selected_account.account_type, # Le type de compte peut influencer l'endpoint, mais get_exchange_info est général
            is_testnet=selected_account.is_testnet
        )
        # Accéder directement au client SDK Binance sous-jacent pour get_exchange_info()
        # car OrderExecutionClient pourrait ne pas l'exposer directement.
        if hasattr(exec_client, 'client') and hasattr(exec_client.client, 'get_exchange_info'):
            exchange_info_data = exec_client.client.get_exchange_info() # type: ignore
            logger.info(f"{log_prefix} Informations de l'exchange récupérées avec succès depuis l'API.")
            return cast(Dict[str, Any], exchange_info_data)
        else:
            logger.error(f"{log_prefix} L'instance OrderExecutionClient ou son client SDK sous-jacent ne supporte pas get_exchange_info().")
            return None
    except Exception as e:
        logger.error(f"{log_prefix} Erreur lors de la récupération des informations de l'exchange depuis l'API : {e}", exc_info=True)
        return None

def get_symbol_exchange_info(
    pair_symbol: str,
    app_config: 'AppConfig',
    force_refresh: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Fournit les informations de l'exchange (filtres, précisions) pour une paire de trading donnée.
    Gère un cache local de ces informations et peut les rafraîchir depuis l'API.

    Args:
        pair_symbol (str): Le symbole de la paire (ex: BTCUSDT).
        app_config (AppConfig): L'instance de configuration globale.
        force_refresh (bool): Si True, force le rafraîchissement des données depuis l'API,
                              même si un cache local existe. Par défaut False.

    Returns:
        Optional[Dict[str, Any]]: Un dictionnaire contenant la configuration de la paire
                                  si trouvée, sinon None.
    """
    log_prefix = f"[ExchangeInfoProvider][{pair_symbol.upper()}]"
    logger.info(f"{log_prefix} Demande d'informations pour la paire. Force refresh: {force_refresh}.")

    if not app_config.project_root:
        logger.error(f"{log_prefix} project_root n'est pas défini dans AppConfig.")
        return None # Ou lever une exception

    cache_file_rel_path = app_config.exchange_settings.exchange_info_file_path
    cache_file_abs_path = Path(app_config.project_root) / cache_file_rel_path

    all_exchange_data: Optional[Dict[str, Any]] = None

    if not force_refresh:
        logger.debug(f"{log_prefix} Tentative de chargement depuis le cache : {cache_file_abs_path}")
        all_exchange_data = _load_info_from_cache(cache_file_abs_path)

    if force_refresh or all_exchange_data is None:
        if force_refresh:
            logger.info(f"{log_prefix} Rafraîchissement forcé demandé.")
        else:
            logger.info(f"{log_prefix} Cache non trouvé ou invalide, tentative de rafraîchissement depuis l'API.")

        fetched_data = _fetch_exchange_info_from_api(app_config)
        if fetched_data:
            if _save_info_to_cache(fetched_data, cache_file_abs_path):
                all_exchange_data = fetched_data
            else:
                logger.error(f"{log_prefix} Échec de la sauvegarde des données API dans le cache. Utilisation des données API non sauvegardées pour cette session si possible.")
                all_exchange_data = fetched_data # Utiliser quand même les données fraîchement récupérées
        elif all_exchange_data: # Si le fetch API a échoué mais qu'on avait un cache (même si force_refresh était true)
            logger.warning(f"{log_prefix} Échec du rafraîchissement API. Retour aux données du cache précédemment chargées (si disponibles).")
            # all_exchange_data contient déjà les données du cache s'il a été chargé avant la tentative de refresh.
        else: # Fetch API a échoué et pas de cache initial
            logger.error(f"{log_prefix} Échec du rafraîchissement API et aucun cache disponible. Impossible de fournir les informations de l'exchange.")
            return None
    
    if all_exchange_data:
        # Utiliser la fonction utilitaire pour extraire les informations de la paire spécifique
        pair_specific_info = get_pair_config_for_symbol(pair_symbol.upper(), all_exchange_data)
        if pair_specific_info:
            logger.info(f"{log_prefix} Informations trouvées pour la paire {pair_symbol.upper()}.")
            return pair_specific_info
        else:
            logger.warning(f"{log_prefix} Paire {pair_symbol.upper()} non trouvée dans les informations de l'exchange (cache ou API).")
            return None
    else:
        # Ce cas ne devrait pas être atteint si la logique ci-dessus est correcte,
        # mais par sécurité :
        logger.error(f"{log_prefix} Aucune donnée d'exchange (ni cache, ni API) n'a pu être obtenue.")
        return None

if __name__ == '__main__':
    # Section pour des tests directs du module (nécessite une AppConfig factice)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Exécution de exchange_info_provider.py directement pour tests.")

    # Créer une AppConfig factice pour le test
    from src.config.definitions import (
        GlobalConfig, PathsConfig, LoggingConfig, SimulationDefaults, WfoSettings, OptunaSettings,
        DataConfig, SourceDetails, AssetsAndTimeframes, HistoricalPeriod, FetchingOptions,
        StrategiesConfig, LiveConfig, AccountConfig, ApiKeys, ExchangeSettings, LiveLoggingConfig,
        GlobalLiveSettings, LiveFetchConfig, StrategyDeployment
    )

    # Créer un répertoire de test et un fichier cache factice
    test_project_root_info = Path("./temp_exchange_info_test_project").resolve()
    test_project_root_info.mkdir(parents=True, exist_ok=True)
    (test_project_root_info / "config").mkdir(parents=True, exist_ok=True)
    
    dummy_cache_file = test_project_root_info / "config" / "dummy_exchange_info.json"
    
    # Données factices pour le cache (incluant certaines des paires demandées)
    dummy_cache_content = {
        "timezone": "UTC",
        "serverTime": 1672531200000,
        "rateLimits": [],
        "exchangeFilters": [],
        "symbols": [
            {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"}]},
            {"symbol": "ETHUSDC", "baseAsset": "ETH", "quoteAsset": "USDC", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"}]},
            {"symbol": "DOGEUSDC", "baseAsset": "DOGE", "quoteAsset": "USDC", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.000001"}]},
            # Ajouter d'autres paires demandées si nécessaire pour le test du cache
            {"symbol": "AIXBTUSDC", "baseAsset": "AIXBT", "quoteAsset": "USDC", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.0001"}]},
            {"symbol": "XRPUSDC", "baseAsset": "XRP", "quoteAsset": "USDC", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.0001"}]},
            {"symbol": "PEPEUSDC", "baseAsset": "PEPE", "quoteAsset": "USDC", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.00000001"}]},
        ]
    }
    with open(dummy_cache_file, 'w') as f_dummy:
        json.dump(dummy_cache_content, f_dummy)
    logger.info(f"Fichier cache factice créé : {dummy_cache_file}")

    mock_exchange_settings = ExchangeSettings(
        exchange_name="binance",
        exchange_info_file_path=str(Path("config") / "dummy_exchange_info.json") # Relatif au project_root
    )
    
    # Simuler une config de compte pour l'appel API (nécessite des clés API valides dans .env pour un vrai test API)
    # Pour ce test, on se concentre sur le chargement du cache.
    mock_account = AccountConfig(account_alias="test_binance_live", exchange="binance", account_type="MARGIN", is_testnet=False, api_key_env_var="TEST_BINANCE_API_KEY", api_secret_env_var="TEST_BINANCE_API_SECRET")
    # S'assurer que les variables d'env sont settées si on veut tester le refresh API
    # os.environ["TEST_BINANCE_API_KEY"] = "your_key"
    # os.environ["TEST_BINANCE_API_SECRET"] = "your_secret"

    mock_api_keys = ApiKeys(credentials={"test_binance_live": (None,None)}) # Pas de vraies clés pour ce test de cache
    
    mock_app_config_instance = AppConfig(
        global_config=GlobalConfig(project_name="TestInfoProvider", paths=PathsConfig("","","","","","","","",""), logging=LoggingConfig("INFO","",False,""), simulation_defaults=SimulationDefaults(1000,1,0), wfo_settings=WfoSettings(1,1,1,"",""), optuna_settings=OptunaSettings(1)),
        data_config=DataConfig(SourceDetails("",""), AssetsAndTimeframes([],[]), HistoricalPeriod("2020-01-01"), FetchingOptions(1,100)),
        strategies_config=StrategiesConfig(strategies={}),
        live_config=LiveConfig(GlobalLiveSettings(False,1,0.01,0.1),[],LiveFetchConfig([],[],1,1),LiveLoggingConfig()),
        accounts_config=[mock_account],
        api_keys=mock_api_keys,
        exchange_settings=mock_exchange_settings,
        project_root=str(test_project_root_info)
    )

    # Test 1: Charger depuis le cache
    logger.info("\n--- Test 1: Chargement depuis le cache ---")
    info_btc = get_symbol_exchange_info("BTCUSDT", mock_app_config_instance)
    if info_btc:
        logger.info(f"Info BTCUSDT depuis cache: {info_btc.get('filters')[0] if info_btc.get('filters') else 'Pas de filtres'}") # type: ignore
    else:
        logger.error("Échec du chargement de BTCUSDT depuis le cache.")

    info_doge = get_symbol_exchange_info("DOGEUSDC", mock_app_config_instance)
    if info_doge:
        logger.info(f"Info DOGEUSDC depuis cache: {info_doge.get('filters')[0] if info_doge.get('filters') else 'Pas de filtres'}") # type: ignore
    else:
        logger.error("Échec du chargement de DOGEUSDC depuis le cache.")

    info_new = get_symbol_exchange_info("ADABUSD", mock_app_config_instance) # N'est pas dans le cache factice
    if info_new:
        logger.info(f"Info ADABUSD (non cachée): {info_new}")
    else:
        logger.info("Info ADABUSD non trouvée dans le cache (attendu).")

    # Test 2: Forcer le rafraîchissement (nécessiterait des clés API valides et une connexion internet)
    # Décommenter pour tester le rafraîchissement API si les clés sont configurées dans .env
    # logger.info("\n--- Test 2: Forcer le rafraîchissement (peut nécessiter des clés API valides) ---")
    # Note: Pour que ce test fonctionne, TEST_BINANCE_API_KEY et TEST_BINANCE_API_SECRET doivent être définis
    # dans l'environnement et être des clés valides (même pour un compte testnet si is_testnet=True).
    # if os.getenv("TEST_BINANCE_API_KEY") and os.getenv("TEST_BINANCE_API_SECRET"):
    #     mock_api_keys_real = ApiKeys(credentials={"test_binance_live": (os.getenv("TEST_BINANCE_API_KEY"), os.getenv("TEST_BINANCE_API_SECRET"))})
    #     mock_app_config_instance_for_api = dataclasses.replace(mock_app_config_instance, api_keys=mock_api_keys_real)
    #     logger.info("Tentative de rafraîchissement forcé pour BTCUSDT...")
    #     info_btc_refreshed = get_symbol_exchange_info("BTCUSDT", mock_app_config_instance_for_api, force_refresh=True)
    #     if info_btc_refreshed:
    #         logger.info(f"Info BTCUSDT rafraîchie: {info_btc_refreshed.get('filters')[0] if info_btc_refreshed.get('filters') else 'Pas de filtres'}")
    #         # Vérifier si le fichier cache a été mis à jour (timestamp de modification)
    #         if dummy_cache_file.exists():
    #             logger.info(f"Timestamp du fichier cache après refresh: {datetime.fromtimestamp(dummy_cache_file.stat().st_mtime)}")
    #     else:
    #         logger.error("Échec du rafraîchissement forcé de BTCUSDT (vérifiez les clés API et la connexion).")
    # else:
    #     logger.warning("Variables d'environnement TEST_BINANCE_API_KEY/SECRET non définies. Saut du test de rafraîchissement API.")


    # Nettoyage
    # import shutil
    # if test_project_root_info.exists():
    #     shutil.rmtree(test_project_root_info)
    #     logger.info(f"Répertoire de test nettoyé : {test_project_root_info}")
    pass # Laisser les fichiers pour inspection
