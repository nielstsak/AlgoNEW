# src/live/execution.py
"""
Ce module définit la classe OrderExecutionClient, responsable de toutes les
interactions avec l'API de l'exchange (Binance). Cela inclut la récupération
d'informations de compte, d'informations sur les symboles, et le
placement/gestion des ordres sur marge.
"""
import logging
import os
import sys # Pour le logger initial si les imports échouent
import time
import json # Pour le logging de paramètres complexes
from typing import Dict, Optional, Any, List, Union, Callable, cast # cast ajouté

import requests # Pour requests.exceptions.Timeout et autres

# Gestion des imports Binance avec fallback
try:
    import binance # Pour binance.__version__
    # Renommé pour clarté et éviter confusion avec OrderExecutionClient
    from binance.client import Client as BinanceSdkClient
    from binance.exceptions import BinanceAPIException, BinanceRequestException, BinanceOrderException
    BINANCE_IMPORTS_OK = True
    BINANCE_VERSION = getattr(binance, '__version__', 'unknown')
    # Le logger n'est pas encore configuré au niveau du module, donc pas de log ici.
except ImportError as e_import_binance:
    BINANCE_IMPORTS_OK = False
    BINANCE_VERSION = 'not_installed'
    # Logger un message critique si la bibliothèque est manquante.
    # Utiliser un logger basique car le logging de l'application n'est peut-être pas encore prêt.
    _initial_logger = logging.getLogger(__name__ + "_bootstrap")
    _initial_logger.addHandler(logging.StreamHandler(sys.stderr))
    _initial_logger.setLevel(logging.CRITICAL)
    _initial_logger.critical(
        f"ÉCHEC CRITIQUE (execution.py): L'import de la bibliothèque python-binance a échoué : {e_import_binance}. "
        "L'application ne pourra pas interagir avec l'API Binance. "
        "Veuillez vous assurer que 'python-binance' (version >= 1.0.19 recommandée) est installée.",
        exc_info=True
    )
    # Définition de classes factices pour permettre au reste du module de se charger
    # (pour l'analyse statique ou des tests limités), mais les opérations réelles échoueront.
    class BinanceAPIException(Exception): pass # type: ignore
    class BinanceRequestException(Exception): pass # type: ignore
    class BinanceOrderException(Exception): pass # type: ignore
    class BinanceSdkClient: # type: ignore
        """Client SDK Binance factice pour fallback."""
        KLINE_INTERVAL_1MINUTE = "1m" # Constante de classe
        ORDER_TYPE_LIMIT = 'LIMIT'
        ORDER_TYPE_MARKET = 'MARKET'
        # ... autres constantes si nécessaires pour le typage ...

        def __init__(self, api_key=None, api_secret=None, tld='com', testnet=False, requests_params=None):
            _initial_logger.critical("Utilisation du client BinanceSdkClient FACTICE en raison d'un échec d'importation. Les appels API NE fonctionneront PAS.")
        def ping(self): raise NotImplementedError("BinanceSdkClient factice")
        def get_server_time(self): return {'serverTime': int(time.time() * 1000)}
        def get_exchange_info(self): raise NotImplementedError("BinanceSdkClient factice") # Ajouté
        def get_symbol_info(self, symbol): raise NotImplementedError("BinanceSdkClient factice")
        def get_isolated_margin_account(self, **params): raise NotImplementedError("BinanceSdkClient factice")
        def get_margin_account(self, **params): raise NotImplementedError("BinanceSdkClient factice")
        def get_open_margin_orders(self, **params): raise NotImplementedError("BinanceSdkClient factice")
        def get_all_oco_orders(self, **params): raise NotImplementedError("BinanceSdkClient factice")
        def get_margin_order(self, **params): raise NotImplementedError("BinanceSdkClient factice")
        # def get_all_margin_orders(self, **params): raise NotImplementedError("BinanceSdkClient factice") # Moins utilisé, peut être omis
        def create_margin_order(self, **params): raise NotImplementedError("BinanceSdkClient factice")
        def create_margin_oco_order(self, **params): raise NotImplementedError("BinanceSdkClient factice")
        def repay_margin_loan(self, **params): raise NotImplementedError("BinanceSdkClient factice")
        def cancel_margin_order(self, **params): raise NotImplementedError("BinanceSdkClient factice")
        def cancel_margin_oco_order(self, **params): raise NotImplementedError("BinanceSdkClient factice")
        def close_connection(self): pass # Méthode de fermeture factice

logger = logging.getLogger(__name__) # Logger standard pour le reste du module

ACCOUNT_TYPE_MAP: Dict[str, str] = {
    "SPOT": "SPOT",
    "MARGIN": "MARGIN", # Cross Margin
    "ISOLATED_MARGIN": "ISOLATED_MARGIN",
    "FUTURES_USDT": "FUTURES_USDT", # Pour USD-M Futures
    "FUTURES_COIN": "FUTURES_COIN"  # Pour COIN-M Futures (non entièrement géré par ce client pour les ordres)
}
# Actif de cotation principal (peut être rendu configurable si d'autres sont utilisés)
USDC_ASSET = "USDC" # Ou BUSD, USDT, etc.
DEFAULT_API_TIMEOUT_SECONDS = 15
MAX_API_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 2.0


class OrderExecutionClient:
    """
    Client pour interagir avec l'API de l'exchange (Binance), gérant les appels API,
    la récupération d'informations, et le placement/gestion des ordres sur marge.
    """
    def __init__(self,
                 api_key: Optional[str],
                 api_secret: Optional[str],
                 account_type: str = "MARGIN", # Ex: "SPOT", "MARGIN", "ISOLATED_MARGIN", "FUTURES_USDT"
                 is_testnet: bool = False):
        """
        Initialise le client d'exécution d'ordres.

        Args:
            api_key (Optional[str]): Clé API Binance.
            api_secret (Optional[str]): Secret API Binance.
            account_type (str): Type de compte à utiliser. Valeurs valides dans ACCOUNT_TYPE_MAP.
                                Par défaut "MARGIN" (cross margin).
            is_testnet (bool): Si True, utilise l'environnement de testnet de Binance.
                               Par défaut False.
        
        Raises:
            ImportError: Si la bibliothèque python-binance n'est pas installée.
            ValueError: Si les clés API sont manquantes ou si account_type est invalide.
            ConnectionError: Si l'initialisation du client SDK Binance échoue.
        """
        self.log_prefix = f"[OrderExecClient][{account_type.upper()}{'-TESTNET' if is_testnet else ''}]"
        logger.info(f"{self.log_prefix} Initialisation...")

        if not BINANCE_IMPORTS_OK:
            # Le message critique a déjà été loggué au niveau du module.
            raise ImportError("OrderExecutionClient ne peut pas fonctionner car la bibliothèque python-binance n'a pas pu être importée.")

        if not api_key or not api_secret:
            msg = "Clé API et Secret API Binance sont requis."
            logger.error(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        
        self.api_key: str = api_key
        self.api_secret: str = api_secret

        self.raw_account_type: str = account_type.upper()
        self.mapped_account_type: Optional[str] = ACCOUNT_TYPE_MAP.get(self.raw_account_type)
        
        if not self.mapped_account_type:
            logger.warning(f"{self.log_prefix} account_type '{account_type}' non supporté ou invalide. "
                           f"Options valides : {list(ACCOUNT_TYPE_MAP.keys())}. Utilisation de 'MARGIN' par défaut.")
            self.mapped_account_type = "MARGIN" # Cross Margin
            self.raw_account_type = "MARGIN"

        self.is_testnet: bool = is_testnet
        self.is_isolated_margin_trading: bool = (self.raw_account_type == "ISOLATED_MARGIN")
        
        try:
            requests_params = {'timeout': DEFAULT_API_TIMEOUT_SECONDS}
            self.client: BinanceSdkClient = BinanceSdkClient(
                self.api_key, self.api_secret, testnet=self.is_testnet, requests_params=requests_params
            )
            logger.info(f"{self.log_prefix} Client SDK Binance initialisé (Testnet: {self.is_testnet}). Version SDK: {BINANCE_VERSION}")
        except Exception as e_init_sdk:
            logger.critical(f"{self.log_prefix} Échec de l'initialisation du client SDK Binance : {e_init_sdk}", exc_info=True)
            raise ConnectionError(f"Initialisation du client SDK Binance échouée : {e_init_sdk}") from e_init_sdk

        self._symbol_info_cache: Dict[str, Dict[str, Any]] = {}
        self._exchange_info_cache: Optional[Dict[str, Any]] = None # Cache pour get_exchange_info_raw
        
        logger.info(f"{self.log_prefix} OrderExecutionClient initialisé avec succès.")

    def _make_api_call(self,
                       api_method: Callable[..., Any],
                       log_context_override: Optional[str] = None,
                       **kwargs: Any) -> Optional[Any]:
        """
        Wrapper robuste pour les appels à l'API Binance, gérant les re-essais et les erreurs.

        Args:
            api_method (Callable[..., Any]): La méthode du client SDK Binance à appeler.
            log_context_override (Optional[str]): Contexte de log spécifique pour cet appel.
                                                  Si None, le nom de api_method est utilisé.
            **kwargs: Arguments à passer à api_method.

        Returns:
            Optional[Any]: La réponse brute de l'API en cas de succès, ou un dictionnaire
                           standardisé `{"status": "ERROR", ...}` en cas d'échec après
                           re-essais, ou None si une erreur imprévue majeure survient.
        """
        num_retries = MAX_API_RETRIES
        current_retry_delay = INITIAL_RETRY_DELAY_SECONDS
        
        effective_log_context = log_context_override if log_context_override else \
                                (api_method.__name__ if hasattr(api_method, '__name__') else 'unknown_api_method')

        # Tronquer les kwargs pour le logging si nécessaire (ex: données de payload volumineuses)
        # Pour l'instant, on logue les kwargs tels quels, en supposant qu'ils ne sont pas trop gros.
        # Si des secrets sont passés directement (ce qui ne devrait pas être le cas ici), ils seraient loggués.
        # Les clés API sont dans self.client, pas dans kwargs pour les méthodes du SDK.
        
        for attempt in range(num_retries):
            try:
                # Utiliser json.dumps avec default=str pour les kwargs peut aider si des objets non sérialisables sont passés,
                # mais les méthodes du SDK Binance attendent généralement des types simples.
                logger.debug(f"{self.log_prefix}[{effective_log_context}] Appel API (tentative {attempt + 1}/{num_retries}). "
                             f"Params: {json.dumps(kwargs, default=str, indent=None, ensure_ascii=False)[:250]}...") # Tronquer pour lisibilité
                
                response = api_method(**kwargs)
                
                # Loguer un snippet de la réponse pour le débogage
                response_str_snippet = str(response)
                if len(response_str_snippet) > 300:
                    response_str_snippet = response_str_snippet[:297] + "..."
                logger.debug(f"{self.log_prefix}[{effective_log_context}] Réponse API reçue : {response_str_snippet}")
                return response # Succès
            
            except BinanceAPIException as e_api:
                logger.error(f"{self.log_prefix}[{effective_log_context}] Exception API Binance (Tentative {attempt + 1}/{num_retries}): "
                             f"HTTP={e_api.status_code}, Code={e_api.code}, Msg='{e_api.message}'")
                if e_api.code == -1021: # Erreur de timestamp -> synchro horloge système nécessaire
                    logger.warning(f"{self.log_prefix}[{effective_log_context}] Erreur de timestamp (-1021). "
                                   "Vérifiez la synchronisation de l'horloge système. Pas de re-essai pour cette erreur.")
                    return {"status": "ERROR", "code": e_api.code, "message": e_api.message, "is_timestamp_error": True}
                
                is_rate_limit = e_api.status_code in [429, 418] or e_api.code == -1003 # Rate limit ou IP ban
                if attempt < num_retries - 1: # Si ce n'est pas la dernière tentative
                    delay_multiplier = 2.0 if is_rate_limit else 1.5 # Backoff plus agressif pour rate limit
                    actual_sleep_time = current_retry_delay * delay_multiplier
                    log_level = logging.WARNING if is_rate_limit else logging.INFO
                    logger.log(log_level, f"{self.log_prefix}[{effective_log_context}] Re-essai dans {actual_sleep_time:.2f}s...")
                    time.sleep(actual_sleep_time)
                    current_retry_delay = actual_sleep_time # Augmenter le délai pour le prochain re-essai potentiel
                    continue # Passer à la prochaine tentative
                else: # Dernière tentative échouée
                    logger.error(f"{self.log_prefix}[{effective_log_context}] Nombre maximum de re-essais atteint après exception API Binance.")
                    return {"status": "ERROR", "code": e_api.code, "message": f"Max retries: {e_api.message}"}

            except BinanceRequestException as e_req: # Erreurs dans les paramètres de la requête, etc. Généralement non récupérables par re-essai.
                logger.error(f"{self.log_prefix}[{effective_log_context}] Exception de Requête Binance : {e_req}. Pas de re-essai.", exc_info=True)
                return {"status": "ERROR", "message": f"Request Exception: {str(e_req)}"}
            
            except requests.exceptions.Timeout as e_timeout:
                logger.error(f"{self.log_prefix}[{effective_log_context}] Timeout de la requête (Tentative {attempt + 1}/{num_retries}): {e_timeout}")
                if attempt < num_retries - 1:
                    time.sleep(current_retry_delay)
                    current_retry_delay *= 2 # Backoff exponentiel
                    continue
                logger.error(f"{self.log_prefix}[{effective_log_context}] Nombre maximum de re-essais atteint après Timeout.")
                return {"status": "ERROR", "message": "Request Timeout"}

            except requests.exceptions.ConnectionError as e_conn:
                logger.error(f"{self.log_prefix}[{effective_log_context}] Erreur de Connexion (Tentative {attempt + 1}/{num_retries}): {e_conn}")
                if attempt < num_retries - 1:
                    time.sleep(current_retry_delay)
                    current_retry_delay *= 2
                    continue
                logger.error(f"{self.log_prefix}[{effective_log_context}] Nombre maximum de re-essais atteint après Erreur de Connexion.")
                return {"status": "ERROR", "message": "Connection Error"}

            except Exception as e_general: # pylint: disable=broad-except
                logger.error(f"{self.log_prefix}[{effective_log_context}] Erreur inattendue durant l'appel API (Tentative {attempt + 1}/{num_retries}): {e_general}", exc_info=True)
                if attempt < num_retries - 1:
                    time.sleep(current_retry_delay)
                    current_retry_delay *= 1.5
                    continue
                logger.error(f"{self.log_prefix}[{effective_log_context}] Nombre maximum de re-essais atteint après erreur inattendue.")
                return {"status": "ERROR", "message": f"Unexpected error: {str(e_general)}"}
        
        # Si la boucle se termine sans succès (ne devrait pas arriver si les erreurs retournent un dict)
        logger.critical(f"{self.log_prefix}[{effective_log_context}] La boucle _make_api_call s'est terminée sans succès ni retour d'erreur explicite. C'est inattendu.")
        return None


    def test_connection(self) -> bool:
        """
        Teste la connexion à l'API en effectuant un ping et en récupérant l'heure du serveur.

        Returns:
            bool: True si la connexion est réussie, False sinon.
        """
        log_ctx = "test_connection"
        try:
            # client.ping() retourne None en cas de succès, ou lève une exception.
            # _make_api_call gère les exceptions et retourne un dict d'erreur si échec.
            ping_response = self._make_api_call(self.client.ping, log_context_override=f"{log_ctx}_ping")
            if isinstance(ping_response, dict) and ping_response.get("status") == "ERROR":
                logger.error(f"{self.log_prefix}[{log_ctx}] Ping API échoué : {ping_response.get('message')}")
                return False
            # Si ping_response n'est pas un dict d'erreur, c'est un succès (None).
            
            server_time_response = self._make_api_call(self.client.get_server_time, log_context_override=f"{log_ctx}_getServerTime")
            if server_time_response and isinstance(server_time_response, dict) and server_time_response.get('serverTime'):
                logger.info(f"{self.log_prefix}[{log_ctx}] Connexion API réussie. Heure serveur : {server_time_response['serverTime']}")
                return True
            else:
                err_msg = server_time_response.get('message') if isinstance(server_time_response, dict) else str(server_time_response)
                logger.error(f"{self.log_prefix}[{log_ctx}] Échec de la récupération de l'heure du serveur. Réponse : {err_msg}")
                return False
        except Exception as e: # Attraper toute autre exception durant le processus
            logger.error(f"{self.log_prefix}[{log_ctx}] Exception durant le test de connexion : {e}", exc_info=True)
            return False

    def get_exchange_info_raw(self) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations complètes de l'exchange (exchangeInfo).
        Utilisé par ExchangeInfoProvider pour mettre à jour son cache.

        Returns:
            Optional[Dict[str, Any]]: Les informations brutes de l'exchange, ou None en cas d'erreur.
        """
        log_ctx = "get_exchange_info_raw"
        logger.debug(f"{self.log_prefix}[{log_ctx}] Récupération des informations complètes de l'exchange.")
        
        # Utiliser le client SDK Binance directement car cette méthode est standard
        if hasattr(self.client, 'get_exchange_info'):
            response = self._make_api_call(self.client.get_exchange_info, log_context_override=log_ctx)
            if response and isinstance(response, dict) and "symbols" in response and not response.get("status") == "ERROR":
                logger.info(f"{self.log_prefix}[{log_ctx}] Informations de l'exchange récupérées avec succès ({len(response.get('symbols',[]))} symboles).")
                return response
            else:
                err_msg = response.get('message') if isinstance(response, dict) else str(response)
                logger.error(f"{self.log_prefix}[{log_ctx}] Échec de la récupération des informations de l'exchange. Réponse : {err_msg}")
                return None
        else:
            logger.error(f"{self.log_prefix}[{log_ctx}] La méthode get_exchange_info n'est pas disponible sur le client SDK.")
            return None


    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Récupère et met en cache les informations pour un symbole spécifique de l'exchange.

        Args:
            symbol (str): Le symbole de la paire de trading (ex: BTCUSDT).

        Returns:
            Optional[Dict[str, Any]]: Un dictionnaire contenant les informations du symbole
                                      si trouvé et récupéré avec succès, sinon None.
        """
        symbol_upper = symbol.upper()
        log_ctx = f"get_symbol_info_{symbol_upper}"

        if symbol_upper not in self._symbol_info_cache:
            logger.debug(f"{self.log_prefix}[{log_ctx}] Cache miss. Récupération depuis l'API...")
            # L'appel à get_symbol_info du SDK Binance est direct.
            info = self._make_api_call(self.client.get_symbol_info, symbol=symbol_upper, log_context_override=log_ctx)
            
            if info and isinstance(info, dict) and not (info.get("status") == "ERROR"):
                self._symbol_info_cache[symbol_upper] = info
                logger.info(f"{self.log_prefix}[{log_ctx}] Informations du symbole mises en cache.")
            else:
                err_msg = info.get('message') if isinstance(info, dict) else str(info)
                logger.error(f"{self.log_prefix}[{log_ctx}] Échec de la récupération des informations du symbole. Réponse : {err_msg}")
                return None # Ne pas mettre en cache une réponse d'erreur
        else:
            logger.debug(f"{self.log_prefix}[{log_ctx}] Cache hit.")
        return self._symbol_info_cache.get(symbol_upper)


    def get_margin_asset_balance(self, asset: str, symbol_pair_for_isolated: Optional[str] = None) -> Optional[float]:
        """
        Récupère le solde 'free' (disponible) d'un actif spécifique dans le compte sur marge.

        Args:
            asset (str): Le symbole de l'actif (ex: "USDT", "BTC").
            symbol_pair_for_isolated (Optional[str]): Requis si le compte est de type
                ISOLATED_MARGIN, spécifie la paire de marge isolée (ex: "BTCUSDT").
                Ignoré pour les comptes MARGIN (cross).

        Returns:
            Optional[float]: Le solde disponible de l'actif, ou None en cas d'erreur.
                             Retourne 0.0 si l'actif n'est pas trouvé avec un solde.
        """
        asset_upper = asset.upper()
        log_ctx = f"get_margin_balance_{asset_upper}"
        
        account_details_response: Optional[Dict[str, Any]] = None
        try:
            if self.is_isolated_margin_trading:
                if not symbol_pair_for_isolated:
                    logger.error(f"{self.log_prefix}[{log_ctx}] symbol_pair_for_isolated est requis pour le type de compte ISOLATED_MARGIN.")
                    return None
                log_ctx += f"_iso_{symbol_pair_for_isolated.upper()}"
                # Le SDK python-binance pour get_isolated_margin_account avec un seul symbole dans `symbols`
                # retourne un dictionnaire contenant les détails de cette paire.
                account_details_response = self._make_api_call(self.client.get_isolated_margin_account, symbols=symbol_pair_for_isolated.upper(), log_context_override=log_ctx)
            
            elif self.raw_account_type == "MARGIN": # Cross Margin
                account_details_response = self._make_api_call(self.client.get_margin_account, log_context_override=log_ctx)
            else:
                logger.error(f"{self.log_prefix}[{log_ctx}] Type de compte non supporté pour get_margin_asset_balance : {self.raw_account_type}")
                return None

            if not account_details_response or (isinstance(account_details_response, dict) and account_details_response.get("status") == "ERROR"):
                err_msg = account_details_response.get('message') if isinstance(account_details_response, dict) else str(account_details_response)
                logger.warning(f"{self.log_prefix}[{log_ctx}] Échec de la récupération des détails du compte. Réponse : {err_msg}")
                return None

            asset_data_found: Optional[Dict[str, Any]] = None
            if self.is_isolated_margin_trading and isinstance(account_details_response, dict) and 'assets' in account_details_response:
                # La réponse pour un symbole isolé est une liste contenant un dict pour la paire.
                isolated_pair_assets_list = account_details_response.get('assets', [])
                if isolated_pair_assets_list:
                    pair_specific_data = isolated_pair_assets_list[0] # Prendre le premier (et unique) élément
                    if pair_specific_data.get('baseAsset', {}).get('asset', '').upper() == asset_upper:
                        asset_data_found = pair_specific_data.get('baseAsset')
                    elif pair_specific_data.get('quoteAsset', {}).get('asset', '').upper() == asset_upper:
                        asset_data_found = pair_specific_data.get('quoteAsset')
            elif self.raw_account_type == "MARGIN" and isinstance(account_details_response, dict) and 'userAssets' in account_details_response: # Cross
                asset_data_found = next((a for a in account_details_response.get('userAssets', []) if a.get('asset', '').upper() == asset_upper), None)

            if asset_data_found and 'free' in asset_data_found:
                balance = float(asset_data_found['free'])
                logger.info(f"{self.log_prefix}[{log_ctx}] Solde disponible pour {asset_upper} : {balance}")
                return balance
            else:
                logger.warning(f"{self.log_prefix}[{log_ctx}] Actif {asset_upper} non trouvé ou solde 'free' manquant dans les détails du compte.")
                return 0.0 # Actif non présent ou solde nul

        except Exception as e: # pylint: disable=broad-except
            logger.error(f"{self.log_prefix}[{log_ctx}] Erreur lors de la récupération du solde de l'actif {asset_upper} sur marge : {e}", exc_info=True)
            return None

    def get_margin_usdc_balance(self, symbol_pair_for_isolated: Optional[str] = None) -> Optional[float]:
        """
        Raccourci pour récupérer le solde USDC disponible dans le compte sur marge.

        Args:
            symbol_pair_for_isolated (Optional[str]): Requis si compte ISOLATED_MARGIN.

        Returns:
            Optional[float]: Solde USDC disponible, ou None en cas d'erreur.
        """
        return self.get_margin_asset_balance(USDC_ASSET, symbol_pair_for_isolated=symbol_pair_for_isolated)

    def get_active_margin_loans(self, asset: Optional[str] = None, isolated_symbol_pair: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les prêts sur marge actifs, optionnellement filtrés par actif.

        Args:
            asset (Optional[str]): Symbole de l'actif à filtrer (ex: "BTC"). Si None, retourne les prêts pour tous les actifs.
            isolated_symbol_pair (Optional[str]): Requis si compte ISOLATED_MARGIN.

        Returns:
            List[Dict[str, Any]]: Liste des prêts actifs. Chaque dict contient 'asset', 'borrowed', etc.
                                  Retourne une liste vide en cas d'erreur ou si aucun prêt.
        """
        asset_filter_upper = asset.upper() if asset else None
        log_ctx = f"get_active_loans_{asset_filter_upper or 'ALL'}"
        
        account_details_response: Optional[Dict[str, Any]] = None
        active_loans_list: List[Dict[str, Any]] = []
        try:
            if self.is_isolated_margin_trading:
                if not isolated_symbol_pair:
                    logger.error(f"{self.log_prefix}[{log_ctx}] symbol_pair_for_isolated est requis pour les prêts ISOLATED_MARGIN.")
                    return []
                log_ctx += f"_iso_{isolated_symbol_pair.upper()}"
                account_details_response = self._make_api_call(self.client.get_isolated_margin_account, symbols=isolated_symbol_pair.upper(), log_context_override=log_ctx)

                if account_details_response and isinstance(account_details_response, dict) and 'assets' in account_details_response:
                    isolated_pair_assets_list = account_details_response.get('assets', [])
                    if isolated_pair_assets_list:
                        pair_data = isolated_pair_assets_list[0]
                        for asset_key_in_pair in ['baseAsset', 'quoteAsset']: # Vérifier base et quote asset de la paire isolée
                            asset_info = pair_data.get(asset_key_in_pair, {})
                            if float(asset_info.get('borrowed', 0.0)) > 1e-9: # Seuil pour considérer un prêt comme actif
                                if not asset_filter_upper or asset_info.get('asset', '').upper() == asset_filter_upper:
                                    active_loans_list.append(asset_info)
            elif self.raw_account_type == "MARGIN": # Cross Margin
                account_details_response = self._make_api_call(self.client.get_margin_account, log_context_override=log_ctx)
                if account_details_response and isinstance(account_details_response, dict) and 'userAssets' in account_details_response:
                    for user_asset_info in account_details_response.get('userAssets', []):
                        if float(user_asset_info.get('borrowed', 0.0)) > 1e-9:
                            if not asset_filter_upper or user_asset_info.get('asset', '').upper() == asset_filter_upper:
                                active_loans_list.append(user_asset_info)
            else:
                logger.error(f"{self.log_prefix}[{log_ctx}] Type de compte non supporté : {self.raw_account_type}")
                return []
            
            if isinstance(account_details_response, dict) and account_details_response.get("status") == "ERROR":
                 logger.warning(f"{self.log_prefix}[{log_ctx}] Échec de la récupération des détails du compte pour les prêts. Réponse: {account_details_response.get('message')}")
                 return []

            logger.info(f"{self.log_prefix}[{log_ctx}] Trouvé {len(active_loans_list)} prêt(s) actif(s)" + (f" pour l'actif {asset_filter_upper}." if asset_filter_upper else "."))
            return active_loans_list
        except Exception as e: # pylint: disable=broad-except
            logger.error(f"{self.log_prefix}[{log_ctx}] Erreur lors de la récupération des prêts sur marge actifs : {e}", exc_info=True)
            return []

    def _get_is_isolated_param_for_api(self, is_operation_isolated: bool) -> Dict[str, str]:
        """Prépare le paramètre 'isIsolated' pour les appels API."""
        return {"isIsolated": "TRUE"} if is_operation_isolated else {"isIsolated": "FALSE"}

    def get_all_open_margin_orders(self, symbol: str, is_isolated_override: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Récupère tous les ordres sur marge ouverts pour un symbole donné.

        Args:
            symbol (str): Symbole de la paire (ex: BTCUSDT).
            is_isolated_override (Optional[bool]): Permet de surcharger le mode de marge
                (isolé/cross) pour cet appel spécifique. Si None, utilise la configuration
                du client (`self.is_isolated_margin_trading`).

        Returns:
            List[Dict[str, Any]]: Liste des ordres ouverts, ou liste vide en cas d'erreur.
        """
        symbol_upper = symbol.upper()
        log_ctx = f"get_open_margin_orders_{symbol_upper}"
        
        params_api: Dict[str, Any] = {"symbol": symbol_upper}
        
        actual_is_isolated_mode = self.is_isolated_margin_trading if is_isolated_override is None else is_isolated_override
        params_api.update(self._get_is_isolated_param_for_api(actual_is_isolated_mode))
        if actual_is_isolated_mode: log_ctx += "_iso"

        open_orders_response = self._make_api_call(self.client.get_open_margin_orders, **params_api, log_context_override=log_ctx)
        
        if isinstance(open_orders_response, list):
            return open_orders_response
        elif isinstance(open_orders_response, dict) and open_orders_response.get("status") == "ERROR":
            logger.error(f"{self.log_prefix}[{log_ctx}] Erreur API lors de la récupération des ordres ouverts : {open_orders_response.get('message')}")
        else:
            logger.warning(f"{self.log_prefix}[{log_ctx}] Réponse inattendue lors de la récupération des ordres ouverts : {open_orders_response}")
        return []

    def get_open_margin_oco_orders(self, symbol: str, is_isolated_override: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Récupère tous les ordres OCO (One-Cancels-the-Other) ouverts sur marge pour un symbole.

        Args:
            symbol (str): Symbole de la paire.
            is_isolated_override (Optional[bool]): Surcharge pour le mode de marge.

        Returns:
            List[Dict[str, Any]]: Liste des ordres OCO actifs, ou liste vide.
        """
        symbol_upper = symbol.upper()
        log_ctx = f"get_open_oco_orders_{symbol_upper}"
        params_api: Dict[str, Any] = {}

        actual_is_isolated_mode = self.is_isolated_margin_trading if is_isolated_override is None else is_isolated_override

        if actual_is_isolated_mode:
            params_api["symbol"] = symbol_upper # Requis pour les OCO isolés
            params_api["isIsolated"] = "TRUE"
            log_ctx += "_iso"
        # Pour cross margin, get_all_oco_orders ne prend pas de `symbol` ni `isIsolated` (il retourne tout).
        # Le filtrage par symbole se fait après.

        all_open_ocos_raw = self._make_api_call(self.client.get_all_oco_orders, **params_api, log_context_override=log_ctx)
        
        active_ocos_for_symbol: List[Dict[str, Any]] = []
        if isinstance(all_open_ocos_raw, list):
            for oco_order_list in all_open_ocos_raw:
                # Pour cross, filtrer par symbole. Pour isolé, le symbole devrait déjà correspondre.
                symbol_matches = actual_is_isolated_mode or (oco_order_list.get('symbol', '').upper() == symbol_upper)
                # Un OCO est actif si son listOrderStatus est "EXECUTING"
                if symbol_matches and oco_order_list.get('listOrderStatus') == "EXECUTING":
                    active_ocos_for_symbol.append(oco_order_list)
        elif isinstance(all_open_ocos_raw, dict) and all_open_ocos_raw.get("status") == "ERROR":
             logger.error(f"{self.log_prefix}[{log_ctx}] Erreur API lors de la récupération des ordres OCO : {all_open_ocos_raw.get('message')}")

        logger.info(f"{self.log_prefix}[{log_ctx}] Trouvé {len(active_ocos_for_symbol)} ordre(s) OCO actif(s) pour {symbol_upper}.")
        return active_ocos_for_symbol

    def get_margin_order_status(self, symbol: str, order_id: Optional[Union[int, str]] = None,
                                orig_client_order_id: Optional[str] = None, is_isolated_override: Optional[bool] = None
                               ) -> Optional[Dict[str, Any]]:
        """
        Récupère le statut d'un ordre sur marge spécifique.

        Args:
            symbol (str): Symbole de la paire.
            order_id (Optional[Union[int, str]]): ID de l'ordre de l'exchange.
            orig_client_order_id (Optional[str]): ID client original de l'ordre.
            is_isolated_override (Optional[bool]): Surcharge pour le mode de marge.

        Returns:
            Optional[Dict[str, Any]]: Dictionnaire du statut de l'ordre, ou un dictionnaire
                                      d'erreur standardisé, ou None si erreur majeure.
        """
        log_ctx = f"get_order_status_{symbol.upper()}_id_{order_id or orig_client_order_id}"
        if not order_id and not orig_client_order_id:
            logger.error(f"{self.log_prefix}[{log_ctx}] order_id ou orig_client_order_id est requis.")
            return {"status": "ERROR", "message": "order_id ou orig_client_order_id requis."}

        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id: params_api["orderId"] = int(str(order_id)) # L'API Binance attend souvent un int pour orderId
        if orig_client_order_id: params_api["origClientOrderId"] = orig_client_order_id
        
        actual_is_isolated_mode = self.is_isolated_margin_trading if is_isolated_override is None else is_isolated_override
        params_api.update(self._get_is_isolated_param_for_api(actual_is_isolated_mode))
        if actual_is_isolated_mode: log_ctx += "_iso"

        order_status_response = self._make_api_call(self.client.get_margin_order, **params_api, log_context_override=log_ctx)

        if isinstance(order_status_response, dict):
            # Vérifier si c'est une réponse d'erreur de _make_api_call
            if order_status_response.get("status") == "ERROR":
                # Gérer spécifiquement l'erreur "Order does not exist" (-2013)
                if order_status_response.get("code") == -2013:
                    logger.warning(f"{self.log_prefix}[{log_ctx}] Ordre non trouvé sur l'exchange (Code -2013).")
                    # Retourner un statut clair pour "non trouvé"
                    return {"api_status": "NOT_FOUND", "original_response": order_status_response}
                logger.error(f"{self.log_prefix}[{log_ctx}] Erreur API lors de la récupération du statut de l'ordre : {order_status_response.get('message')}")
                return {"api_status": "API_ERROR_FETCH", "original_response": order_status_response}
            # Si ce n'est pas un dict d'erreur de _make_api_call, c'est la réponse de l'API
            return order_status_response
        
        logger.error(f"{self.log_prefix}[{log_ctx}] Réponse inattendue ou None de _make_api_call pour get_margin_order.")
        return {"api_status": "UNKNOWN_ERROR_FETCH", "original_response": None}


    def place_margin_order(self, **params: Any) -> Dict[str, Any]:
        """
        Place un ordre sur marge (achat ou vente).

        Args:
            **params: Paramètres de l'ordre (symbol, side, type, quantity, price, etc.).
                      `isIsolated` sera ajouté automatiquement basé sur la configuration du client.

        Returns:
            Dict[str, Any]: Un dictionnaire standardisé :
                            `{"status": "SUCCESS", "data": response_api}` ou
                            `{"status": "API_ERROR", "message": ..., "code": ..., "params_sent": ...}`.
        """
        symbol_op = str(params.get("symbol", "SYM_INCONNU"))
        log_ctx = f"place_margin_order_{symbol_op}_{params.get('side','SIDE')}"
        
        # Préparer les paramètres finaux pour l'API
        final_api_params = params.copy()
        final_api_params.update(self._get_is_isolated_param_for_api(self.is_isolated_margin_trading and symbol_op == self.symbol if hasattr(self, 'symbol') else self.is_isolated_margin_trading))
        # Note: La condition `symbol_op == self.symbol` est une heuristique si `self.symbol` est la paire principale isolée.
        # Idéalement, l'appelant spécifie `isIsolated` si nécessaire pour une paire non principale.

        logger.info(f"{self.log_prefix}[{log_ctx}] Tentative de placement d'ordre sur marge. Params API (partiel) : { {k:v for k,v in final_api_params.items() if k != 'newClientOrderId'} }")
        api_response = self._make_api_call(self.client.create_margin_order, **final_api_params, log_context_override=log_ctx)

        if api_response and isinstance(api_response, dict) and \
           (api_response.get("orderId") or api_response.get("clientOrderId")) and \
           not (api_response.get("status") == "ERROR" or api_response.get("code") is not None and api_response.get("code") != 0): # Succès si orderId/clientOrderId et pas un code d'erreur connu
            logger.info(f"{self.log_prefix}[{log_ctx}] Placement d'ordre sur marge réussi : OrderID={api_response.get('orderId')}, ClientOrderID={api_response.get('clientOrderId')}")
            return {"status": "SUCCESS", "data": api_response}
        else:
            err_msg = api_response.get("message", "Erreur inconnue") if isinstance(api_response, dict) else str(api_response)
            err_code = api_response.get("code") if isinstance(api_response, dict) else None
            logger.error(f"{self.log_prefix}[{log_ctx}] Échec du placement d'ordre sur marge. Réponse API : {api_response}. Paramètres envoyés : {final_api_params}")
            return {"status": "API_ERROR", "message": err_msg, "code": err_code, "params_sent": final_api_params}

    def place_margin_oco_order(self, **params: Any) -> Dict[str, Any]:
        """Place un ordre OCO (One-Cancels-the-Other) sur marge."""
        symbol_op = str(params.get("symbol", "SYM_INCONNU"))
        log_ctx = f"place_margin_oco_{symbol_op}_{params.get('side','SIDE')}"
        final_api_params = params.copy()
        final_api_params.update(self._get_is_isolated_param_for_api(self.is_isolated_margin_trading and symbol_op == self.symbol if hasattr(self, 'symbol') else self.is_isolated_margin_trading))

        logger.info(f"{self.log_prefix}[{log_ctx}] Tentative de placement d'ordre OCO. Params API (partiel) : { {k:v for k,v in final_api_params.items() if 'clientOrderId' not in k.lower()} }")
        api_response = self._make_api_call(self.client.create_margin_oco_order, **final_api_params, log_context_override=log_ctx)

        if api_response and isinstance(api_response, dict) and api_response.get("orderListId") and not (api_response.get("status") == "ERROR" or api_response.get("code")):
            logger.info(f"{self.log_prefix}[{log_ctx}] Placement d'ordre OCO sur marge réussi : OrderListID={api_response.get('orderListId')}")
            return {"status": "SUCCESS", "data": api_response}
        else:
            err_msg = api_response.get("message", "Erreur inconnue") if isinstance(api_response, dict) else str(api_response)
            err_code = api_response.get("code") if isinstance(api_response, dict) else None
            logger.error(f"{self.log_prefix}[{log_ctx}] Échec du placement d'ordre OCO sur marge. Réponse API : {api_response}. Params envoyés : {final_api_params}")
            return {"status": "API_ERROR", "message": err_msg, "code": err_code, "params_sent": final_api_params}

    def repay_margin_loan(self, asset: str, amount: str, isolated_symbol_pair: Optional[str] = None) -> Dict[str, Any]:
        """Rembourse un prêt sur marge."""
        asset_upper = asset.upper()
        log_ctx = f"repay_margin_loan_{asset_upper}_amt_{amount}"
        
        repay_params: Dict[str,Any] = {"asset": asset_upper, "amount": str(amount)} # L'API attend amount comme string
        if self.is_isolated_margin_trading:
            if not isolated_symbol_pair:
                msg = "isolated_symbol_pair est requis pour le remboursement de prêt sur marge ISOLATED_MARGIN."
                logger.error(f"{self.log_prefix}[{log_ctx}] {msg}")
                return {"status": "ERROR", "message": msg}
            repay_params["isIsolated"] = "TRUE"
            repay_params["symbol"] = isolated_symbol_pair.upper()
            log_ctx += f"_iso_{isolated_symbol_pair.upper()}"
        elif self.raw_account_type == "MARGIN": # Cross
            repay_params["isIsolated"] = "FALSE" # Certaines API de remboursement peuvent nécessiter isIsolated=FALSE pour cross

        logger.info(f"{self.log_prefix}[{log_ctx}] Tentative de remboursement de prêt.")
        api_response = self._make_api_call(self.client.repay_margin_loan, **repay_params, log_context_override=log_ctx)

        if api_response and isinstance(api_response, dict) and api_response.get("tranId") and not (api_response.get("status") == "ERROR" or api_response.get("code")):
            logger.info(f"{self.log_prefix}[{log_ctx}] Remboursement de prêt réussi. TranID : {api_response.get('tranId')}")
            return {"status": "SUCCESS", "data": api_response}
        else:
            err_msg = api_response.get("message", "Erreur inconnue") if isinstance(api_response, dict) else str(api_response)
            err_code = api_response.get("code") if isinstance(api_response, dict) else None
            logger.error(f"{self.log_prefix}[{log_ctx}] Échec du remboursement de prêt. Réponse API : {api_response}. Params envoyés : {repay_params}")
            return {"status": "API_ERROR", "message": err_msg, "code": err_code, "params_sent": repay_params}

    def cancel_margin_order(self, symbol: str, order_id: Optional[Union[int, str]] = None,
                            orig_client_order_id: Optional[str] = None, is_isolated_override: Optional[bool] = None
                           ) -> Dict[str, Any]:
        """Annule un ordre sur marge."""
        log_ctx = f"cancel_margin_order_{symbol.upper()}_id_{order_id or orig_client_order_id}"
        if not order_id and not orig_client_order_id:
            return {"status": "ERROR", "message": "order_id ou orig_client_order_id est requis pour l'annulation."}

        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id: params_api["orderId"] = int(str(order_id))
        if orig_client_order_id: params_api["origClientOrderId"] = orig_client_order_id
        
        actual_is_isolated_mode = self.is_isolated_margin_trading if is_isolated_override is None else is_isolated_override
        params_api.update(self._get_is_isolated_param_for_api(actual_is_isolated_mode))
        if actual_is_isolated_mode: log_ctx+="_iso"

        logger.info(f"{self.log_prefix}[{log_ctx}] Tentative d'annulation d'ordre.")
        api_response = self._make_api_call(self.client.cancel_margin_order, **params_api, log_context_override=log_ctx)
        
        if api_response and isinstance(api_response, dict) and \
           (api_response.get("orderId") or api_response.get("clientOrderId")) and \
           not (api_response.get("status") == "ERROR" or api_response.get("code")):
            logger.info(f"{self.log_prefix}[{log_ctx}] Annulation d'ordre réussie : {api_response.get('orderId') or api_response.get('clientOrderId')}")
            return {"status": "SUCCESS", "data": api_response}
        else:
            err_code = api_response.get("code") if isinstance(api_response, dict) else None
            err_msg = api_response.get("message", "Erreur inconnue") if isinstance(api_response, dict) else str(api_response)
            if err_code == -2011: # Ordre déjà rempli, annulé ou inexistant
                logger.warning(f"{self.log_prefix}[{log_ctx}] Tentative d'annulation d'un ordre déjà traité ou inexistant (Code -2011). Message : {err_msg}")
                return {"status": "ORDER_NOT_FOUND_OR_ALREADY_PROCESSED", "code": err_code, "message": err_msg, "params_sent": params_api}
            logger.error(f"{self.log_prefix}[{log_ctx}] Échec de l'annulation d'ordre. Réponse API : {api_response}. Params envoyés : {params_api}")
            return {"status": "API_ERROR", "code": err_code, "message": err_msg, "params_sent": params_api}

    def cancel_margin_oco_order(self, symbol: str, order_list_id: Optional[int] = None,
                                list_client_order_id: Optional[str] = None, is_isolated_override: Optional[bool] = None
                               ) -> Dict[str, Any]:
        """Annule un ordre OCO sur marge."""
        log_ctx = f"cancel_margin_oco_{symbol.upper()}_id_{order_list_id or list_client_order_id}"
        if not order_list_id and not list_client_order_id:
            return {"status": "ERROR", "message": "order_list_id ou list_client_order_id est requis pour l'annulation OCO."}

        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_list_id: params_api["orderListId"] = order_list_id # Doit être int
        if list_client_order_id: params_api["listClientOrderId"] = list_client_order_id
        
        actual_is_isolated_mode = self.is_isolated_margin_trading if is_isolated_override is None else is_isolated_override
        params_api.update(self._get_is_isolated_param_for_api(actual_is_isolated_mode))
        if actual_is_isolated_mode: log_ctx+="_iso"

        logger.info(f"{self.log_prefix}[{log_ctx}] Tentative d'annulation d'ordre OCO.")
        api_response = self._make_api_call(self.client.cancel_margin_oco_order, **params_api, log_context_override=log_ctx)

        if api_response and isinstance(api_response, dict) and api_response.get("orderListId") and not (api_response.get("status") == "ERROR" or api_response.get("code")):
            logger.info(f"{self.log_prefix}[{log_ctx}] Annulation d'ordre OCO réussie : OrderListID={api_response.get('orderListId')}")
            return {"status": "SUCCESS", "data": api_response}
        else:
            err_code = api_response.get("code") if isinstance(api_response, dict) else None
            err_msg = api_response.get("message", "Erreur inconnue") if isinstance(api_response, dict) else str(api_response)
            if err_code == -2011: # Liste d'ordres OCO déjà traitée ou inexistante
                 logger.warning(f"{self.log_prefix}[{log_ctx}] Tentative d'annulation d'une liste OCO déjà traitée ou inexistante (Code -2011). Message : {err_msg}")
                 return {"status": "ORDER_LIST_NOT_FOUND_OR_ALREADY_PROCESSED", "code": err_code, "message": err_msg, "params_sent": params_api}
            logger.error(f"{self.log_prefix}[{log_ctx}] Échec de l'annulation d'ordre OCO. Réponse API : {api_response}. Params envoyés : {params_api}")
            return {"status": "API_ERROR", "code": err_code, "message": err_msg, "params_sent": params_api}

    def close(self) -> None:
        """
        Ferme la session du client SDK Binance sous-jacent, si la version le supporte.
        """
        log_ctx = "close_client_session"
        logger.info(f"{self.log_prefix}[{log_ctx}] Fermeture de la session OrderExecutionClient...")
        if hasattr(self.client, 'close_connection') and callable(self.client.close_connection):
            try:
                self.client.close_connection()
                logger.info(f"{self.log_prefix}[{log_ctx}] Session du client SDK Binance fermée avec succès.")
            except Exception as e_close: # pylint: disable=broad-except
                logger.error(f"{self.log_prefix}[{log_ctx}] Erreur lors de la fermeture de la session du client SDK Binance : {e_close}", exc_info=True)
        else:
            logger.debug(f"{self.log_prefix}[{log_ctx}] Le client SDK Binance (version: {BINANCE_VERSION}) "
                         "ne possède pas de méthode 'close_connection'. Aucune fermeture explicite de session n'est nécessaire.")

