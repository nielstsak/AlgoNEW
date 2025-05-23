import logging
import os
import sys # Pour le logger initial si les imports échouent
import time
import json # Pour le logging de paramètres complexes
from typing import Dict, Optional, Any, List, Union, Callable

import requests # Pour requests.exceptions.Timeout

# Gestion des imports Binance avec fallback
try:
    import binance # Pour binance.__version__
    # Renommé pour clarté et éviter confusion avec OrderExecutionClient
    from binance.client import Client as BinanceSdkClient
    from binance.exceptions import BinanceAPIException, BinanceRequestException, BinanceOrderException
    BINANCE_IMPORTS_OK = True
    BINANCE_VERSION = getattr(binance, '__version__', 'unknown')
    logging.getLogger(__name__).info(f"Successfully imported python-binance version: {BINANCE_VERSION}")
except ImportError as e_import:
    BINANCE_IMPORTS_OK = False
    BINANCE_VERSION = 'not_installed'
    logging.getLogger(__name__).critical(
        f"CRITICAL FAILURE: python-binance library import failed: {e_import}. "
        "The application will not be able to interact with the Binance API. "
        "Please ensure 'python-binance' (version >= 1.0.19 recommended) is installed.",
        exc_info=True
    )
    # Définition de classes factices pour permettre au reste du module de se charger (pour analyse statique/tests limités)
    # mais les opérations réelles échoueront.
    class BinanceAPIException(Exception): pass
    class BinanceRequestException(Exception): pass
    class BinanceOrderException(Exception): pass
    class BinanceSdkClient: # type: ignore
        def __init__(self, api_key=None, api_secret=None, tld='com', testnet=False, requests_params=None):
            logging.critical("Dummy BinanceSdkClient used due to import failure. API calls will NOT work.")
        def ping(self): raise NotImplementedError("Dummy BinanceSdkClient")
        def get_server_time(self): return {'serverTime': int(time.time() * 1000)}
        def get_symbol_info(self, symbol): raise NotImplementedError("Dummy BinanceSdkClient")
        def get_isolated_margin_account(self, **params): raise NotImplementedError("Dummy BinanceSdkClient")
        def get_margin_account(self, **params): raise NotImplementedError("Dummy BinanceSdkClient")
        def get_open_margin_orders(self, **params): raise NotImplementedError("Dummy BinanceSdkClient")
        def get_all_oco_orders(self, **params): raise NotImplementedError("Dummy BinanceSdkClient") # Note: params for this can be tricky
        def get_margin_order(self, **params): raise NotImplementedError("Dummy BinanceSdkClient")
        def get_all_margin_orders(self, **params): raise NotImplementedError("Dummy BinanceSdkClient")
        def create_margin_order(self, **params): raise NotImplementedError("Dummy BinanceSdkClient")
        def create_margin_oco_order(self, **params): raise NotImplementedError("Dummy BinanceSdkClient")
        def repay_margin_loan(self, **params): raise NotImplementedError("Dummy BinanceSdkClient")
        def cancel_margin_order(self, **params): raise NotImplementedError("Dummy BinanceSdkClient")
        def cancel_margin_oco_order(self, **params): raise NotImplementedError("Dummy BinanceSdkClient")
        def close(self): pass # Dummy close

logger = logging.getLogger(__name__)

ACCOUNT_TYPE_MAP = {
    "SPOT": "SPOT", # Bien que ce client soit orienté marge, SPOT est un type valide.
    "MARGIN": "MARGIN", # Cross Margin
    "ISOLATED_MARGIN": "ISOLATED_MARGIN",
    "FUTURES": "FUTURES_USD_M" # Exemple, non entièrement supporté par ce client pour les ordres.
}
USDC_ASSET = "USDC" # Ou BUSD, USDT selon la paire de cotation principale
DEFAULT_API_TIMEOUT_SECONDS = 15 # Augmenté légèrement
MAX_API_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 2.0 # Float pour time.sleep

class OrderExecutionClient:
    def __init__(self,
                 api_key: Optional[str],
                 api_secret: Optional[str],
                 account_type: str = "MARGIN",
                 is_testnet: bool = False):
        
        self.log_prefix = f"[ExecClient][{account_type.upper()}{'-TESTNET' if is_testnet else ''}]"
        logger.info(f"{self.log_prefix} Initializing...")

        if not BINANCE_IMPORTS_OK:
            # Log critique déjà émis lors de l'échec de l'import
            raise ImportError(
                "OrderExecutionClient cannot function because python-binance library failed to import."
            )

        self.api_key = api_key
        self.api_secret = api_secret

        if not self.api_key or not self.api_secret:
            logger.error(f"{self.log_prefix} Binance API key or secret not provided.")
            raise ValueError("Binance API key and secret are required.")
        
        self.raw_account_type = account_type.upper()
        self.mapped_account_type = ACCOUNT_TYPE_MAP.get(self.raw_account_type)
        
        if not self.mapped_account_type:
            logger.warning(f"{self.log_prefix} Unsupported account_type '{account_type}'. Defaulting to MARGIN.")
            self.mapped_account_type = "MARGIN" # Cross Margin
            self.raw_account_type = "MARGIN"

        self.is_testnet = is_testnet
        self.is_isolated_margin_trading = (self.raw_account_type == "ISOLATED_MARGIN")
        
        try:
            requests_params = {'timeout': DEFAULT_API_TIMEOUT_SECONDS}
            # Utiliser BinanceSdkClient (le SDK Binance)
            self.client: BinanceSdkClient = BinanceSdkClient(
                self.api_key, self.api_secret, testnet=self.is_testnet, requests_params=requests_params
            )
            logger.info(f"{self.log_prefix} Binance SDK Client initialized (Testnet: {self.is_testnet}).")
        except Exception as e_init:
            logger.critical(f"{self.log_prefix} Failed to initialize Binance SDK Client: {e_init}", exc_info=True)
            raise ConnectionError(f"Binance SDK Client initialization failed: {e_init}") from e_init

        self._symbol_info_cache: Dict[str, Dict[str, Any]] = {}
        logger.info(f"{self.log_prefix} OrderExecutionClient initialized successfully.")

    def _prepare_margin_params(self, params: Dict[str, Any], symbol_for_isolated: Optional[str] = None) -> Dict[str, Any]:
        """
        Adds 'isIsolated' parameter if operating on an ISOLATED_MARGIN account type
        and the symbol matches, or if it's a general MARGIN (cross) account.
        """
        prepared_params = params.copy()
        if self.raw_account_type == "ISOLATED_MARGIN":
            if symbol_for_isolated and prepared_params.get('symbol', '').upper() == symbol_for_isolated.upper():
                prepared_params['isIsolated'] = "TRUE"
            elif not symbol_for_isolated and 'symbol' in prepared_params : # If symbol_for_isolated not given, but symbol is in params
                 prepared_params['isIsolated'] = "TRUE" # Assume it's for this isolated pair
            # If symbol_for_isolated is given but doesn't match, or if no symbol at all, it's an issue.
            # The calling function should ensure symbol_for_isolated is correct for isolated margin operations.
        elif self.raw_account_type == "MARGIN": # Cross Margin
            # isIsolated is typically not sent or can be "FALSE" for cross margin.
            # Some endpoints might require it to be "FALSE" if the parameter is supported.
            # For safety, if an endpoint *could* take isIsolated, and we are cross, set to FALSE.
            # However, many cross margin endpoints don't use this param.
            # The python-binance SDK handles this for most cases.
            # We will explicitly add it if the method is known to support it for cross.
            # For now, we assume the SDK handles it or the specific methods will add it if needed.
            pass
        return prepared_params

    def _make_api_call(self, api_method: Callable[..., Any], *args: Any, **kwargs: Any) -> Optional[Any]:
        num_retries = kwargs.pop('num_retries', MAX_API_RETRIES)
        current_retry_delay = kwargs.pop('initial_delay', INITIAL_RETRY_DELAY_SECONDS) # Use this for current delay
        log_context = kwargs.pop('log_context', api_method.__name__ if hasattr(api_method, '__name__') else 'unknown_api_method')

        # Remove internal params before passing to SDK method
        kwargs.pop('initial_delay', None)


        for attempt in range(num_retries):
            try:
                logger.debug(f"{self.log_prefix}[{log_context}] API call attempt {attempt + 1}/{num_retries}. Args: {args}, Kwargs: {json.dumps(kwargs, default=str)[:200]}...")
                response = api_method(*args, **kwargs)
                logger.debug(f"{self.log_prefix}[{log_context}] API response received: {str(response)[:300]}...") # Log snippet
                return response
            except BinanceAPIException as e:
                logger.error(f"{self.log_prefix}[{log_context}] Binance API Exception (Attempt {attempt + 1}/{num_retries}): Code={e.code}, Msg='{e.message}'")
                if e.code == -1021: # Timestamp error
                    logger.warning(f"{self.log_prefix}[{log_context}] Timestamp error (-1021). Check system clock sync. No retry for this.")
                    return {"status": "ERROR", "code": e.code, "message": e.message, "is_timestamp_error": True} # Special return
                if e.status_code in [429, 418] or e.code == -1003: # Rate limit or IP ban
                    if attempt < num_retries - 1:
                        logger.warning(f"{self.log_prefix}[{log_context}] Rate limit (HTTP {e.status_code}, Code {e.code}). Retrying in {current_retry_delay:.2f}s...")
                        time.sleep(current_retry_delay)
                        current_retry_delay *= 2 # Exponential backoff
                        continue
                # For other BinanceAPIErrors, decide if retry is useful. Some are permanent.
                # e.g. -2010 (Insufficient balance), -1121 (Invalid symbol) should not be retried indefinitely.
                # For now, we retry all non-timestamp BinanceAPIErrors up to num_retries.
                if attempt < num_retries - 1:
                    time.sleep(current_retry_delay)
                    current_retry_delay *= 1.5 # Gentler backoff for other errors
                    continue
                logger.error(f"{self.log_prefix}[{log_context}] Max retries reached after Binance API error.")
                return {"status": "ERROR", "code": e.code, "message": e.message} # Return error dict
            except BinanceRequestException as e: # Errors in request parameters, etc. Usually not recoverable by retry.
                logger.error(f"{self.log_prefix}[{log_context}] Binance Request Exception: {e}. No retry.")
                return {"status": "ERROR", "message": f"Request Exception: {e}"}
            except requests.exceptions.Timeout as e_timeout:
                logger.error(f"{self.log_prefix}[{log_context}] Request Timeout (Attempt {attempt + 1}/{num_retries}): {e_timeout}")
                if attempt < num_retries - 1:
                    time.sleep(current_retry_delay)
                    current_retry_delay *= 2
                    continue
                logger.error(f"{self.log_prefix}[{log_context}] Max retries reached after Timeout.")
                return {"status": "ERROR", "message": "Request Timeout"}
            except requests.exceptions.ConnectionError as e_conn:
                logger.error(f"{self.log_prefix}[{log_context}] Connection Error (Attempt {attempt + 1}/{num_retries}): {e_conn}")
                if attempt < num_retries - 1:
                    time.sleep(current_retry_delay)
                    current_retry_delay *= 2
                    continue
                logger.error(f"{self.log_prefix}[{log_context}] Max retries reached after Connection Error.")
                return {"status": "ERROR", "message": "Connection Error"}
            except Exception as e_general:
                logger.error(f"{self.log_prefix}[{log_context}] Unexpected error during API call (Attempt {attempt + 1}/{num_retries}): {e_general}", exc_info=True)
                if attempt < num_retries - 1:
                    time.sleep(current_retry_delay)
                    current_retry_delay *= 1.5
                    continue
                logger.error(f"{self.log_prefix}[{log_context}] Max retries reached after unexpected error.")
                return {"status": "ERROR", "message": f"Unexpected error: {e_general}"}
        return None # Should be unreachable if errors return dicts, but as a fallback

    def test_connection(self) -> bool:
        """Tests API connection by pinging and getting server time."""
        log_ctx = "test_connection"
        try:
            ping_response = self._make_api_call(self.client.ping, log_context=f"{log_ctx}_ping")
            # ping() returns None on success or raises exception.
            # We need to check if it raised an exception implicitly via _make_api_call returning error dict.
            if isinstance(ping_response, dict) and ping_response.get("status") == "ERROR":
                logger.error(f"{self.log_prefix}[{log_ctx}] Ping failed: {ping_response.get('message')}")
                return False
            
            server_time_response = self._make_api_call(self.client.get_server_time, log_context=f"{log_ctx}_getServerTime")
            if server_time_response and isinstance(server_time_response, dict) and server_time_response.get('serverTime'):
                logger.info(f"{self.log_prefix}[{log_ctx}] API connection successful. Server time: {server_time_response['serverTime']}")
                return True
            else:
                logger.error(f"{self.log_prefix}[{log_ctx}] Failed to get server time. Response: {server_time_response}")
                return False
        except Exception as e: # Catch any other exception during the process
            logger.error(f"{self.log_prefix}[{log_ctx}] Exception during connection test: {e}", exc_info=True)
            return False

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves and caches symbol information from the exchange."""
        symbol_upper = symbol.upper()
        log_ctx = f"get_symbol_info_{symbol_upper}"
        if symbol_upper not in self._symbol_info_cache:
            logger.debug(f"{self.log_prefix}[{log_ctx}] Cache miss. Fetching from API...")
            info = self._make_api_call(self.client.get_symbol_info, symbol_upper, log_context=log_ctx)
            if info and isinstance(info, dict) and not (info.get("status") == "ERROR"): # Check it's not an error dict from _make_api_call
                self._symbol_info_cache[symbol_upper] = info
                logger.info(f"{self.log_prefix}[{log_ctx}] Symbol info cached.")
            else:
                logger.error(f"{self.log_prefix}[{log_ctx}] Failed to fetch symbol info. Response: {info}")
                return None
        else:
            logger.debug(f"{self.log_prefix}[{log_ctx}] Cache hit.")
        return self._symbol_info_cache.get(symbol_upper)

    def get_margin_asset_balance(self, asset: str, symbol_pair_for_isolated: Optional[str] = None) -> Optional[float]:
        """Gets the 'free' balance of a specific asset in the margin account."""
        asset_upper = asset.upper()
        log_ctx = f"get_margin_balance_{asset_upper}"
        
        account_details: Optional[Dict[str, Any]] = None
        try:
            if self.is_isolated_margin_trading:
                if not symbol_pair_for_isolated:
                    logger.error(f"{self.log_prefix}[{log_ctx}] symbol_pair_for_isolated is required for ISOLATED_MARGIN account type.")
                    return None
                log_ctx += f"_iso_{symbol_pair_for_isolated.upper()}"
                # The 'symbols' parameter for get_isolated_margin_account can take a single symbol string
                account_details_response = self._make_api_call(self.client.get_isolated_margin_account, symbols=symbol_pair_for_isolated.upper(), log_context=log_ctx)
                if account_details_response and isinstance(account_details_response, dict) and 'assets' in account_details_response:
                    account_details = account_details_response # The response itself is the dict for one symbol pair
                elif isinstance(account_details_response, list) and account_details_response: # If API changes to list for single symbol
                     account_details = account_details_response[0]

            elif self.raw_account_type == "MARGIN": # Cross Margin
                account_details = self._make_api_call(self.client.get_margin_account, log_context=log_ctx)
            else:
                logger.error(f"{self.log_prefix}[{log_ctx}] Unsupported account type for get_margin_asset_balance: {self.raw_account_type}")
                return None

            if not account_details or (isinstance(account_details, dict) and account_details.get("status") == "ERROR"):
                logger.warning(f"{self.log_prefix}[{log_ctx}] Failed to fetch account details. Response: {account_details}")
                return None

            asset_info: Optional[Dict[str, Any]] = None
            if self.is_isolated_margin_trading and isinstance(account_details, dict) and 'assets' in account_details:
                # For isolated, account_details is for a single pair, structure is different.
                # The response for a single symbol is a dict like:
                # {"assets": [{"baseAsset": {...}, "quoteAsset": {...}, "symbol": "BTCUSDT", ...}]}
                # If the API returns a list even for one symbol:
                # actual_pair_data = next((p_data for p_data in account_details.get('assets',[]) if p_data.get('symbol') == symbol_pair_for_isolated.upper()), None)
                # For now, assuming if symbols=ONE_PAIR, response is dict for that pair directly.
                # The python-binance SDK for get_isolated_margin_account(symbols='BTCUSDT') returns a dict.
                # If symbols='BTCUSDT,ETHUSDT', it returns a dict with an 'assets' key which is a list.
                # Let's assume the SDK normalizes this if a single symbol string is passed to `symbols`.
                # The structure from SDK for single symbol in `symbols` is:
                # { 'assets': [ { 'baseAsset': ..., 'quoteAsset': ..., 'symbol': 'THE_SYMBOL_PAIR', ... } ] }
                # So we need to iterate `assets` list.
                pair_assets_list = account_details.get('assets', [])
                if pair_assets_list:
                    pair_data = pair_assets_list[0] # Assuming single symbol query returns list with one item
                    if pair_data.get('baseAsset', {}).get('asset', '').upper() == asset_upper:
                        asset_info = pair_data.get('baseAsset')
                    elif pair_data.get('quoteAsset', {}).get('asset', '').upper() == asset_upper:
                        asset_info = pair_data.get('quoteAsset')
            elif self.raw_account_type == "MARGIN" and isinstance(account_details, dict) and 'userAssets' in account_details: # Cross
                asset_info = next((a for a in account_details.get('userAssets', []) if a.get('asset', '').upper() == asset_upper), None)

            if asset_info and 'free' in asset_info:
                balance = float(asset_info['free'])
                logger.info(f"{self.log_prefix}[{log_ctx}] Free balance for {asset_upper}: {balance}")
                return balance
            else:
                logger.warning(f"{self.log_prefix}[{log_ctx}] Asset {asset_upper} not found or 'free' balance missing in account details.")
                return 0.0 # Asset might not be in margin wallet or no balance

        except Exception as e:
            logger.error(f"{self.log_prefix}[{log_ctx}] Error getting margin asset balance for {asset_upper}: {e}", exc_info=True)
            return None

    def get_margin_usdc_balance(self, symbol_pair_for_isolated: Optional[str] = None) -> Optional[float]:
        """Convenience method to get the USDC balance in the margin account."""
        return self.get_margin_asset_balance(USDC_ASSET, symbol_pair_for_isolated=symbol_pair_for_isolated)

    def get_active_margin_loans(self, asset: Optional[str] = None, isolated_symbol_pair: Optional[str] = None) -> List[Dict[str, Any]]:
        """Gets active margin loans, optionally filtered by asset."""
        asset_filter = asset.upper() if asset else None
        log_ctx = f"get_active_loans_{asset_filter or 'ALL'}"
        
        account_details: Optional[Dict[str, Any]] = None
        active_loans_found: List[Dict[str, Any]] = []
        try:
            if self.is_isolated_margin_trading:
                if not isolated_symbol_pair:
                    logger.error(f"{self.log_prefix}[{log_ctx}] symbol_pair_for_isolated is required for ISOLATED_MARGIN loans.")
                    return []
                log_ctx += f"_iso_{isolated_symbol_pair.upper()}"
                account_details_response = self._make_api_call(self.client.get_isolated_margin_account, symbols=isolated_symbol_pair.upper(), log_context=log_ctx)
                if account_details_response and isinstance(account_details_response, dict) and 'assets' in account_details_response:
                     account_details = account_details_response
                elif isinstance(account_details_response, list) and account_details_response:
                     account_details = account_details_response[0]


                if account_details and 'assets' in account_details:
                    pair_assets_list = account_details.get('assets', [])
                    if pair_assets_list:
                        pair_data = pair_assets_list[0]
                        for asset_key in ['baseAsset', 'quoteAsset']:
                            asset_info = pair_data.get(asset_key, {})
                            if float(asset_info.get('borrowed', 0.0)) > 0:
                                if not asset_filter or asset_info.get('asset', '').upper() == asset_filter:
                                    active_loans_found.append(asset_info)
            elif self.raw_account_type == "MARGIN": # Cross Margin
                account_details = self._make_api_call(self.client.get_margin_account, log_context=log_ctx)
                if account_details and isinstance(account_details, dict) and 'userAssets' in account_details:
                    for user_asset in account_details.get('userAssets', []):
                        if float(user_asset.get('borrowed', 0.0)) > 0:
                            if not asset_filter or user_asset.get('asset', '').upper() == asset_filter:
                                active_loans_found.append(user_asset)
            else:
                logger.error(f"{self.log_prefix}[{log_ctx}] Unsupported account type: {self.raw_account_type}")
                return []
            
            logger.info(f"{self.log_prefix}[{log_ctx}] Found {len(active_loans_found)} active loan(s)" + (f" for asset {asset_filter}." if asset_filter else "."))
            return active_loans_found
        except Exception as e:
            logger.error(f"{self.log_prefix}[{log_ctx}] Error getting active margin loans: {e}", exc_info=True)
            return []

    # --- Order Management Methods ---
    def _get_is_isolated_param_for_call(self, symbol_for_operation: str) -> Dict[str, str]:
        """ Determines the 'isIsolated' parameter based on account type and symbol. """
        if self.is_isolated_margin_trading:
            # For isolated margin, the symbol in the operation must match the isolated pair.
            # This check should ideally be done by the caller or ensured by design.
            # Here, we assume if it's isolated, the operation is for ITS isolated pair.
            return {"isIsolated": "TRUE"}
        elif self.raw_account_type == "MARGIN": # Cross Margin
            return {"isIsolated": "FALSE"} # Explicitly FALSE for cross, as some endpoints require it.
        return {} # Default for SPOT or if logic is unclear (should not happen)

    def get_all_open_margin_orders(self, symbol: str, is_isolated_override: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Gets all open margin orders for a symbol."""
        log_ctx = f"get_open_margin_orders_{symbol.upper()}"
        
        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        
        # Determine isIsolated status
        # If is_isolated_override is provided, it takes precedence.
        # Otherwise, use the client's configured is_isolated_margin_trading.
        final_is_isolated = self.is_isolated_margin_trading
        if is_isolated_override is not None:
            final_is_isolated = is_isolated_override
        
        if final_is_isolated:
            params_api["isIsolated"] = "TRUE"
            log_ctx += "_iso"
        elif self.raw_account_type == "MARGIN": # Explicitly for cross margin if endpoint supports it
             params_api["isIsolated"] = "FALSE"


        open_orders = self._make_api_call(self.client.get_open_margin_orders, **params_api, log_context=log_ctx)
        if isinstance(open_orders, list):
            return open_orders
        elif isinstance(open_orders, dict) and open_orders.get("status") == "ERROR":
            logger.error(f"{self.log_prefix}[{log_ctx}] API error fetching open orders: {open_orders.get('message')}")
        return []

    def get_open_margin_oco_orders(self, symbol: str, is_isolated_override: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Gets all open OCO margin orders for a symbol."""
        log_ctx = f"get_open_margin_oco_{symbol.upper()}"
        params_api: Dict[str, Any] = {} # get_all_oco_orders might not take symbol for cross

        final_is_isolated = self.is_isolated_margin_trading
        if is_isolated_override is not None:
            final_is_isolated = is_isolated_override

        if final_is_isolated:
            params_api["symbol"] = symbol.upper() # Symbol is required for isolated OCO list
            params_api["isIsolated"] = "TRUE"
            log_ctx += "_iso"
        # For cross margin, get_all_oco_orders does not take a symbol. We fetch all and filter.
        
        all_open_ocos_raw = self._make_api_call(self.client.get_all_oco_orders, **params_api, log_context=log_ctx)
        
        active_ocos: List[Dict[str, Any]] = []
        if isinstance(all_open_ocos_raw, list):
            for oco in all_open_ocos_raw:
                # Filter by symbol if it was a cross margin call (all OCOs returned)
                symbol_match = final_is_isolated or (oco.get('symbol', '').upper() == symbol.upper())
                if symbol_match and oco.get('listOrderStatus') in ["EXECUTING"]: # "ALL_DONE_PARTIALLY_FILLED" implies one leg filled, other cancelled
                    active_ocos.append(oco)
        elif isinstance(all_open_ocos_raw, dict) and all_open_ocos_raw.get("status") == "ERROR":
             logger.error(f"{self.log_prefix}[{log_ctx}] API error fetching OCO orders: {all_open_ocos_raw.get('message')}")

        logger.info(f"{self.log_prefix}[{log_ctx}] Found {len(active_ocos)} active OCO(s) for {symbol}.")
        return active_ocos

    def get_margin_order_status(self, symbol: str, order_id: Optional[Union[int, str]] = None,
                                orig_client_order_id: Optional[str] = None, is_isolated_override: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        log_ctx = f"get_margin_order_status_{symbol.upper()}_id_{order_id or orig_client_order_id}"
        if not order_id and not orig_client_order_id:
            logger.error(f"{self.log_prefix}[{log_ctx}] order_id or orig_client_order_id is required.")
            return {"status": "ERROR", "message": "Order ID or Client Order ID required."}

        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id: params_api["orderId"] = int(order_id) # Binance API often expects int for orderId
        if orig_client_order_id: params_api["origClientOrderId"] = orig_client_order_id
        
        final_is_isolated = self.is_isolated_margin_trading
        if is_isolated_override is not None:
            final_is_isolated = is_isolated_override
        
        if final_is_isolated:
            params_api["isIsolated"] = "TRUE"
            log_ctx += "_iso"
        elif self.raw_account_type == "MARGIN":
             params_api["isIsolated"] = "FALSE"


        order_status = self._make_api_call(self.client.get_margin_order, **params_api, log_context=log_ctx)

        if isinstance(order_status, dict):
            if order_status.get("status") == "ERROR" and order_status.get("code") == -2013: # Order does not exist
                logger.warning(f"{self.log_prefix}[{log_ctx}] Order not found on exchange (Code -2013).")
                return {"status_order": "NOT_FOUND", "api_response": order_status} # Standardized somewhat
            elif order_status.get("status") == "ERROR":
                logger.error(f"{self.log_prefix}[{log_ctx}] API error getting order status: {order_status.get('message')}")
                return {"status_order": "API_ERROR_FETCH", "api_response": order_status}
            return order_status # Successful fetch
        
        # Fallback if _make_api_call returned None or unexpected type
        logger.error(f"{self.log_prefix}[{log_ctx}] Unexpected response type or None from _make_api_call for get_margin_order.")
        return {"status_order": "UNKNOWN_ERROR_FETCH", "api_response": None}


    def place_margin_order(self, **params: Any) -> Dict[str, Any]:
        log_ctx = f"place_margin_order_{params.get('symbol','SYM')}_{params.get('side','SIDE')}"
        final_params = params.copy()
        final_params.update(self._get_is_isolated_param_for_call(str(params.get("symbol"))))
        
        logger.info(f"{self.log_prefix}[{log_ctx}] Attempting: {final_params}")
        response = self._make_api_call(self.client.create_margin_order, **final_params, log_context=log_ctx)

        if response and isinstance(response, dict) and (response.get("orderId") or response.get("clientOrderId")) and not response.get("code"): # Success usually has orderId
            logger.info(f"{self.log_prefix}[{log_ctx}] Margin order placement successful: {response.get('orderId')}")
            return {"status": "SUCCESS", "data": response}
        else: # Error or unexpected response
            err_msg = response.get("message") if isinstance(response, dict) else str(response)
            logger.error(f"{self.log_prefix}[{log_ctx}] Margin order placement failed. API Response: {response}. Params: {final_params}")
            return {"status": "API_ERROR", "message": err_msg, "code": response.get("code") if isinstance(response, dict) else None, "params_sent": final_params}

    def place_margin_oco_order(self, **params: Any) -> Dict[str, Any]:
        log_ctx = f"place_margin_oco_{params.get('symbol','SYM')}_{params.get('side','SIDE')}"
        final_params = params.copy()
        final_params.update(self._get_is_isolated_param_for_call(str(params.get("symbol"))))

        logger.info(f"{self.log_prefix}[{log_ctx}] Attempting OCO: {final_params}")
        response = self._make_api_call(self.client.create_margin_oco_order, **final_params, log_context=log_ctx)

        if response and isinstance(response, dict) and response.get("orderListId") and not response.get("code"):
            logger.info(f"{self.log_prefix}[{log_ctx}] Margin OCO placement successful: {response.get('orderListId')}")
            return {"status": "SUCCESS", "data": response}
        else:
            err_msg = response.get("message") if isinstance(response, dict) else str(response)
            logger.error(f"{self.log_prefix}[{log_ctx}] Margin OCO placement failed. API Response: {response}. Params: {final_params}")
            return {"status": "API_ERROR", "message": err_msg, "code": response.get("code") if isinstance(response, dict) else None, "params_sent": final_params}

    def repay_margin_loan(self, asset: str, amount: str, isolated_symbol_pair: Optional[str] = None) -> Dict[str, Any]:
        log_ctx = f"repay_margin_loan_{asset.upper()}_{amount}"
        repay_params: Dict[str,Any] = {"asset": asset.upper(), "amount": str(amount)}
        if self.is_isolated_margin_trading:
            if not isolated_symbol_pair:
                return {"status": "ERROR", "message": "isolated_symbol_pair required for ISOLATED_MARGIN loan repayment."}
            repay_params["isIsolated"] = "TRUE"
            repay_params["symbol"] = isolated_symbol_pair.upper()
            log_ctx += f"_iso_{isolated_symbol_pair.upper()}"
        elif self.raw_account_type == "MARGIN":
            repay_params["isIsolated"] = "FALSE"


        logger.info(f"{self.log_prefix}[{log_ctx}] Attempting loan repayment.")
        response = self._make_api_call(self.client.repay_margin_loan, **repay_params, log_context=log_ctx)

        if response and isinstance(response, dict) and response.get("tranId") and not response.get("code"): # Successful repayment has tranId
            logger.info(f"{self.log_prefix}[{log_ctx}] Loan repayment successful. TranID: {response.get('tranId')}")
            return {"status": "SUCCESS", "data": response}
        else:
            err_msg = response.get("message") if isinstance(response, dict) else str(response)
            logger.error(f"{self.log_prefix}[{log_ctx}] Loan repayment failed. API Response: {response}. Params: {repay_params}")
            return {"status": "API_ERROR", "message": err_msg, "code": response.get("code") if isinstance(response, dict) else None, "params_sent": repay_params}

    def cancel_margin_order(self, symbol: str, order_id: Optional[Union[int, str]] = None,
                            orig_client_order_id: Optional[str] = None, is_isolated_override: Optional[bool] = None) -> Dict[str, Any]:
        log_ctx = f"cancel_margin_order_{symbol.upper()}_id_{order_id or orig_client_order_id}"
        if not order_id and not orig_client_order_id:
            return {"status": "ERROR", "message": "order_id or orig_client_order_id required."}

        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id: params_api["orderId"] = int(order_id)
        if orig_client_order_id: params_api["origClientOrderId"] = orig_client_order_id
        
        final_is_isolated = self.is_isolated_margin_trading
        if is_isolated_override is not None: final_is_isolated = is_isolated_override
        
        if final_is_isolated: params_api["isIsolated"] = "TRUE"; log_ctx+="_iso"
        elif self.raw_account_type == "MARGIN": params_api["isIsolated"] = "FALSE"

        logger.info(f"{self.log_prefix}[{log_ctx}] Attempting cancel.")
        response = self._make_api_call(self.client.cancel_margin_order, **params_api, log_context=log_ctx)
        
        if response and isinstance(response, dict) and (response.get("orderId") or response.get("clientOrderId")) and not response.get("code"):
            return {"status": "SUCCESS", "data": response}
        else:
            err_code = response.get("code") if isinstance(response, dict) else None
            err_msg = response.get("message") if isinstance(response, dict) else str(response)
            if err_code == -2011: # Order already filled or cancelled
                return {"status": "ORDER_NOT_FOUND_OR_ALREADY_PROCESSED", "code": err_code, "message": err_msg, "params_sent": params_api}
            return {"status": "API_ERROR", "code": err_code, "message": err_msg, "params_sent": params_api}

    def cancel_margin_oco_order(self, symbol: str, order_list_id: Optional[int] = None,
                                list_client_order_id: Optional[str] = None, is_isolated_override: Optional[bool] = None) -> Dict[str, Any]:
        log_ctx = f"cancel_margin_oco_{symbol.upper()}_id_{order_list_id or list_client_order_id}"
        if not order_list_id and not list_client_order_id:
            return {"status": "ERROR", "message": "order_list_id or list_client_order_id required."}

        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_list_id: params_api["orderListId"] = order_list_id
        if list_client_order_id: params_api["listClientOrderId"] = list_client_order_id
        
        final_is_isolated = self.is_isolated_margin_trading
        if is_isolated_override is not None: final_is_isolated = is_isolated_override

        if final_is_isolated: params_api["isIsolated"] = "TRUE"; log_ctx+="_iso"
        elif self.raw_account_type == "MARGIN": params_api["isIsolated"] = "FALSE"

        logger.info(f"{self.log_prefix}[{log_ctx}] Attempting OCO cancel.")
        response = self._make_api_call(self.client.cancel_margin_oco_order, **params_api, log_context=log_ctx)

        if response and isinstance(response, dict) and response.get("orderListId") and not response.get("code"):
            return {"status": "SUCCESS", "data": response}
        else:
            err_code = response.get("code") if isinstance(response, dict) else None
            err_msg = response.get("message") if isinstance(response, dict) else str(response)
            if err_code == -2011: # Order list does not exist (e.g. already cancelled/filled)
                 return {"status": "ORDER_LIST_NOT_FOUND_OR_ALREADY_PROCESSED", "code": err_code, "message": err_msg, "params_sent": params_api}
            return {"status": "API_ERROR", "code": err_code, "message": err_msg, "params_sent": params_api}

    def close(self):
        """Closes the underlying Binance SDK client session if applicable."""
        log_ctx = "close_client_session"
        logger.info(f"{self.log_prefix}[{log_ctx}] Closing OrderExecutionClient session...")
        if hasattr(self.client, 'close_connection'): # python-binance >= 1.0.16
            try:
                self.client.close_connection() # type: ignore
                logger.info(f"{self.log_prefix}[{log_ctx}] Binance SDK client session closed.")
            except Exception as e:
                logger.error(f"{self.log_prefix}[{log_ctx}] Error closing Binance SDK client session: {e}", exc_info=True)
        else:
            logger.debug(f"{self.log_prefix}[{log_ctx}] Binance SDK client does not have a 'close_connection' method (version: {BINANCE_VERSION}). No explicit close needed.")
