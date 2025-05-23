import importlib
import logging
import threading
import time
import json
import math # Pour np.isnan si utilisé, ou directement numpy
import uuid # Pour les trade_cycle_id
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Tuple, Union
from datetime import datetime, timezone, timedelta
import re # Pour nettoyer les noms de fichiers

import pandas as pd
import numpy as np # Pour np.isnan

# Imports depuis le projet src
try:
    from src.config.definitions import AppConfig, StrategyDeployment, AccountConfig, GlobalLiveSettings, PathsConfig, ApiKeys
    from src.data import acquisition_live
    from src.data import preprocessing_live
    from src.strategies.base import BaseStrategy
    from src.live.state import LiveTradingState, STATUT_1_NO_TRADE_NO_OCO, STATUT_2_ENTRY_FILLED_OCO_PENDING, STATUT_3_OCO_ACTIVE
    from src.live.execution import OrderExecutionClient
    from src.utils.exchange_utils import get_precision_from_filter, adjust_precision # Importé pour _handle_status_2...
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger(__name__).critical(
        f"LiveTradingManager: Critical import error: {e}. Ensure PYTHONPATH is correct "
        "and all required modules are available.",
        exc_info=True
    )
    raise

logger = logging.getLogger(__name__)

# Constantes
USDC_ASSET = "USDC"
MIN_USDC_BALANCE_FOR_TRADE = 10.0
SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT = 5.0
API_CALL_DELAY_S = 0.3 # Légèrement augmenté pour la prudence
MAIN_LOOP_SLEEP_S = 5
KLINE_FETCH_LIMIT_LIVE_UPDATE = 5 # Pour les mises à jour de klines 1-min
ORDER_STATUS_CHECK_DELAY_S = 3 # Délai avant de vérifier le statut d'un ordre
MAX_OCO_CONFIRMATION_ATTEMPTS = 10
FULL_STATE_SYNC_INTERVAL_MINUTES = 5
MIN_EXECUTED_QTY_THRESHOLD = 1e-9 # Pour vérifier si une quantité est effectivement non nulle


class LiveTradingManager:
    def __init__(self,
                 app_config: AppConfig,
                 strategy_deployment_config: StrategyDeployment,
                 account_config: AccountConfig,
                 pair_to_trade: str,
                 context_label_from_deployment: str):

        self.app_config = app_config
        self.strategy_deployment_config = strategy_deployment_config
        self.account_config = account_config
        self.pair_symbol = pair_to_trade.upper()
        self.context_label = context_label_from_deployment

        self.log_prefix = f"[{self.pair_symbol}][Acc:{self.account_config.account_alias}][Ctx:{self.context_label}]"
        logger.info(f"{self.log_prefix} Initializing LiveTradingManager...")

        self.shutdown_event = threading.Event()
        self.strategy: Optional[BaseStrategy] = None
        self.state_manager: Optional[LiveTradingState] = None
        self.execution_client: Optional[OrderExecutionClient] = None

        self.project_root: Path = Path(self.app_config.project_root)
        paths: PathsConfig = self.app_config.global_config.paths
        self.raw_data_dir: Path = Path(paths.data_live_raw)
        self.processed_data_dir: Path = Path(paths.data_live_processed)
        self.state_dir: Path = Path(paths.live_state)
        self.trades_log_dir: Path = Path(paths.logs_live)

        self.raw_1min_data_file_path = self.raw_data_dir / f"{self.pair_symbol}_1min_live_raw.csv"
        cleaned_context_for_file = re.sub(r'[^\w\-_\.]', '_', self.context_label).strip('_') or "default_ctx"
        self.processed_data_file_path = self.processed_data_dir / f"{self.pair_symbol}_{cleaned_context_for_file}_processed_live.csv"

        self.last_1m_kline_open_timestamp: Optional[pd.Timestamp] = None
        self.is_isolated_margin_trading = self.account_config.account_type == "ISOLATED_MARGIN"
        self.base_asset: str = ""
        self.quote_asset: str = ""
        self.oco_confirmation_attempts: int = 0
        self.last_full_state_sync_time: Optional[datetime] = None
        self.current_trade_cycle_id: Optional[str] = None

        self._initialize_components()
        logger.info(f"{self.log_prefix} LiveTradingManager initialization complete.")

    def _initialize_components(self):
        logger.info(f"{self.log_prefix} Initializing components...")
        self._load_strategy_and_params()
        self._initialize_execution_client()
        if not self.execution_client: raise RuntimeError(f"{self.log_prefix} OrderExecutionClient failed to initialize.")
        self._initialize_state_manager()
        if not self.state_manager: raise RuntimeError(f"{self.log_prefix} LiveTradingState failed to initialize.")
        
        logger.info(f"{self.log_prefix} Performing initial data fetch and preprocessing...")
        self._fetch_initial_1min_klines()
        self._run_initial_preprocessing()
        
        logger.info(f"{self.log_prefix} Determining initial status from exchange...")
        self._determine_initial_status(is_periodic_sync=False)
        self.last_full_state_sync_time = datetime.now(timezone.utc)
        logger.info(f"{self.log_prefix} Components initialized successfully.")

    def _load_strategy_and_params(self):
        logger.info(f"{self.log_prefix} Loading strategy and optimized parameters...")
        results_config_path_str = self.strategy_deployment_config.results_config_path
        if not results_config_path_str:
            raise ValueError(f"{self.log_prefix} 'results_config_path' is empty in StrategyDeployment.")

        optimized_params_file = self.project_root / results_config_path_str
        if not optimized_params_file.is_file():
            raise FileNotFoundError(f"{self.log_prefix} Optimized parameters file (live_config.json) not found at: {optimized_params_file}")

        try:
            with open(optimized_params_file, 'r', encoding='utf-8') as f:
                live_params_config_from_file = json.load(f)
        except Exception as e:
            raise ValueError(f"{self.log_prefix} Failed to load or parse optimized parameters file {optimized_params_file}: {e}")

        strategy_name_base = live_params_config_from_file.get("strategy_name_base")
        optimized_params = live_params_config_from_file.get("parameters")

        if not strategy_name_base or not isinstance(optimized_params, dict):
            raise ValueError(f"{self.log_prefix} 'strategy_name_base' or 'parameters' missing/invalid in {optimized_params_file}.")
        
        file_context = live_params_config_from_file.get("timeframe_context")
        if file_context and file_context != self.context_label:
            logger.warning(f"{self.log_prefix} Context label mismatch. Manager: '{self.context_label}', Param file: '{file_context}'. Using params.")

        strategy_definition = self.app_config.strategies_config.strategies.get(strategy_name_base)
        if not strategy_definition:
            raise ValueError(f"{self.log_prefix} Strategy base name '{strategy_name_base}' not found in config_strategies.json.")

        module_path_str = strategy_definition.script_reference
        class_name_str = strategy_definition.class_name
        
        try:
            module_import_path = module_path_str.replace('.py', '').replace('/', '.').replace(os.sep, '.')
            StrategyClass = getattr(importlib.import_module(module_import_path), class_name_str)
            self.strategy = StrategyClass(params=optimized_params) # type: ignore
            logger.info(f"{self.log_prefix} Strategy '{class_name_str}' loaded with parameters from {optimized_params_file.name}.")
        except Exception as e:
            logger.critical(f"{self.log_prefix} Error importing/instantiating strategy {class_name_str}: {e}", exc_info=True)
            raise

    def _initialize_execution_client(self):
        logger.info(f"{self.log_prefix} Initializing OrderExecutionClient for account: {self.account_config.account_alias}")
        api_creds = self.app_config.api_keys.credentials.get(self.account_config.account_alias)
        if not api_creds or not api_creds[0] or not api_creds[1]:
            err_msg = (f"{self.log_prefix} API key/secret not found for '{self.account_config.account_alias}'. "
                       f"Check env vars '{self.account_config.api_key_env_var}' & '{self.account_config.api_secret_env_var}'.")
            logger.critical(err_msg)
            raise ValueError(err_msg)
        
        api_key, api_secret = api_creds
        try:
            self.execution_client = OrderExecutionClient(
                api_key=api_key, api_secret=api_secret,
                account_type=self.account_config.account_type, is_testnet=self.account_config.is_testnet
            )
            if not self.execution_client.test_connection():
                raise ConnectionError(f"Failed API connection for {self.account_config.account_alias}.")
            
            symbol_info = self.execution_client.get_symbol_info(self.pair_symbol)
            if not symbol_info or not symbol_info.get('baseAsset') or not symbol_info.get('quoteAsset'):
                raise ValueError(f"Invalid symbol info for {self.pair_symbol}.")
            self.base_asset = symbol_info['baseAsset']
            self.quote_asset = symbol_info['quoteAsset']
            logger.info(f"{self.log_prefix} OrderExecutionClient OK. Base: {self.base_asset}, Quote: {self.quote_asset}.")
        except Exception as e:
            logger.critical(f"{self.log_prefix} Failed to initialize OrderExecutionClient: {e}", exc_info=True)
            raise
            
    def _initialize_state_manager(self):
        cleaned_context_for_file = re.sub(r'[^\w\-_\.]', '_', self.context_label).strip('_') or "default_ctx"
        state_file_name = f"{self.pair_symbol}_{self.account_config.account_alias}_{cleaned_context_for_file}_state.json"
        state_file_path = self.state_dir / state_file_name
        logger.info(f"{self.log_prefix} Initializing LiveTradingState from: {state_file_path}")
        try:
            self.state_manager = LiveTradingState(self.pair_symbol, state_file_path)
            self.current_trade_cycle_id = self.state_manager.get_state_snapshot().get("current_trade_cycle_id")
            logger.info(f"{self.log_prefix} LiveTradingState OK. Status: {self.state_manager.get_current_status_name()}. Cycle ID: {self.current_trade_cycle_id}")
        except Exception as e:
            logger.critical(f"{self.log_prefix} Failed to initialize LiveTradingState: {e}", exc_info=True)
            raise

    def _fetch_initial_1min_klines(self):
        if not self.state_manager: return
        logger.debug(f"{self.log_prefix} Ensuring initial 1-min klines for {self.raw_1min_data_file_path}")
        limit_init_history = self.app_config.live_config.live_fetch.limit_init_history
        acquisition_live.initialize_pair_data(
            pair=self.pair_symbol, raw_path=self.raw_1min_data_file_path,
            total_klines_to_fetch=limit_init_history, account_type=self.account_config.account_type
        )
        if self.raw_1min_data_file_path.exists() and self.raw_1min_data_file_path.stat().st_size > 0:
            try:
                df_raw_check = pd.read_csv(self.raw_1min_data_file_path, usecols=['timestamp'])
                if not df_raw_check.empty:
                    timestamps = pd.to_datetime(df_raw_check['timestamp'], errors='coerce', utc=True).dropna()
                    if not timestamps.empty: self.last_1m_kline_open_timestamp = timestamps.iloc[-1]
            except Exception as e_ts: logger.error(f"{self.log_prefix} Error reading timestamp post-init: {e_ts}")

    def _run_initial_preprocessing(self):
        if not self.strategy or not self.processed_data_file_path or not self.raw_1min_data_file_path.exists():
            logger.warning(f"{self.log_prefix} Skipping initial preprocessing: missing components.")
            return
        preprocessing_live.preprocess_live_data_for_strategy(
            raw_data_path=self.raw_1min_data_file_path, processed_output_path=self.processed_data_file_path,
            strategy_params=self.strategy.get_params(), strategy_name=self.strategy.__class__.__name__ # type: ignore
        )

    def _run_current_preprocessing_cycle(self):
        if not self.strategy or not self.processed_data_file_path or not self.raw_1min_data_file_path.exists():
            logger.warning(f"{self.log_prefix} Skipping current preprocessing: missing components.")
            return
        preprocessing_live.preprocess_live_data_for_strategy(
            raw_data_path=self.raw_1min_data_file_path, processed_output_path=self.processed_data_file_path,
            strategy_params=self.strategy.get_params(), strategy_name=self.strategy.__class__.__name__ # type: ignore
        )

    def _get_latest_price_from_processed_data(self) -> float:
        if not self.processed_data_file_path or not self.processed_data_file_path.exists():
            logger.warning(f"{self.log_prefix} Processed data file missing: {self.processed_data_file_path}.")
            if self.execution_client:
                try:
                    ticker = self.execution_client.client.get_symbol_ticker(symbol=self.pair_symbol) # type: ignore
                    if ticker and 'price' in ticker: return float(ticker['price'])
                except Exception: pass
            return 0.0
        try:
            df = pd.read_csv(self.processed_data_file_path)
            if not df.empty and 'close' in df.columns and pd.notna(df['close'].iloc[-1]):
                return float(df['close'].iloc[-1])
        except Exception as e: logger.error(f"{self.log_prefix} Error reading price from {self.processed_data_file_path}: {e}")
        return 0.0

    # --- Full implementation of core logic methods ---
    def _determine_initial_status(self, is_periodic_sync: bool = False):
        if not self.execution_client or not self.state_manager: return
        sync_type = "[SYNC]" if is_periodic_sync else "[INIT]"
        logger.info(f"{self.log_prefix}{sync_type} Determining trading status from exchange...")

        usdc_balance = self.execution_client.get_margin_usdc_balance(asset=USDC_ASSET, symbol_pair_for_isolated=self.pair_symbol if self.is_isolated_margin_trading else None)
        usdc_balance = usdc_balance if usdc_balance is not None else 0.0
        self.state_manager.update_specific_fields({"available_capital_at_last_check": usdc_balance})

        active_loans = self.execution_client.get_active_margin_loans(asset=None, isolated_symbol_pair=self.pair_symbol if self.is_isolated_margin_trading else None)
        usdc_loan_amount = sum(float(l.get('borrowed', 0.0)) for l in active_loans if l.get('asset', '').upper() == USDC_ASSET)
        base_asset_loan_amount = sum(float(l.get('borrowed', 0.0)) for l in active_loans if l.get('asset', '').upper() == self.base_asset)
        
        open_orders = self.execution_client.get_all_open_margin_orders(symbol=self.pair_symbol, is_isolated=self.is_isolated_margin_trading)
        num_open_orders = len(open_orders)

        latest_price = self._get_latest_price_from_processed_data()
        has_significant_base_loan = (base_asset_loan_amount * latest_price) > SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT if latest_price > 0 else base_asset_loan_amount > MIN_EXECUTED_QTY_THRESHOLD
        has_significant_quote_loan = usdc_loan_amount > SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT
        has_any_significant_loan = has_significant_base_loan or has_significant_quote_loan
        
        logger.info(f"{self.log_prefix}{sync_type} Exchange: USDC Bal:{usdc_balance:.2f}, {self.base_asset} Loan:{base_asset_loan_amount:.6f}, "
                    f"USDC Loan:{usdc_loan_amount:.2f}, OpenOrders:{num_open_orders}, SigLoan:{has_any_significant_loan}")

        current_state_snapshot = self.state_manager.get_state_snapshot()
        current_bot_status = self.state_manager.get_current_status()

        if num_open_orders == 0 and not has_any_significant_loan:
            if current_bot_status != STATUT_1_NO_TRADE_NO_OCO or is_periodic_sync:
                self.state_manager.transition_to_status_1(f"{sync_type.strip('[]')}_NO_POS_NO_OCO")
                self.current_trade_cycle_id = None
        elif num_open_orders == 1 and not has_any_significant_loan:
            pending_entry_id_state = current_state_snapshot.get("pending_entry_order_id")
            open_order_api = open_orders[0]
            if pending_entry_id_state and str(open_order_api.get("orderId")) == str(pending_entry_id_state):
                if current_bot_status != STATUT_1_NO_TRADE_NO_OCO: self.state_manager.update_specific_fields({"current_status": STATUT_1_NO_TRADE_NO_OCO})
            else:
                self._cancel_all_orders_for_pair(f"{self.log_prefix}{sync_type}[STALE_SINGLE_ORDER_NO_LOAN]")
                self.state_manager.transition_to_status_1(f"{sync_type.strip('[]')}_STALE_ORDER_NO_LOAN")
                self.current_trade_cycle_id = None
        elif num_open_orders > 0 and has_any_significant_loan: # Potentially active OCO or entry filled + OCO issue
            active_oco_id_state = current_state_snapshot.get("active_oco_order_list_id")
            pending_oco_client_id_state = current_state_snapshot.get("pending_oco_list_client_order_id")
            is_oco_on_exchange = any(
                (str(o.get("orderListId")) == str(active_oco_id_state) if active_oco_id_state and o.get("orderListId", -1) != -1 else False) or
                (o.get("listClientOrderId") == pending_oco_client_id_state if pending_oco_client_id_state else False)
                for o in open_orders
            )
            if is_oco_on_exchange:
                if current_bot_status != STATUT_3_OCO_ACTIVE or is_periodic_sync:
                    logger.info(f"{self.log_prefix}{sync_type} OCO found on exchange. Syncing to STATUT_3.")
                    oco_details_api = next((o_list for o_list in self.execution_client.get_open_margin_oco_orders(self.pair_symbol, self.is_isolated_margin_trading) # type: ignore
                                            if str(o_list.get("orderListId")) == str(active_oco_id_state) or o_list.get("listClientOrderId") == pending_oco_client_id_state), None)
                    if oco_details_api: self.state_manager.transition_to_status_3(oco_details_api)
                    else: self.state_manager.update_specific_fields({"current_status": STATUT_3_OCO_ACTIVE}) # Partial sync
            else: # Open orders, loan, but no matching OCO in state -> likely STATUT_2
                logger.info(f"{self.log_prefix}{sync_type} Open orders & loan, no OCO match in state. Assuming STATUT_2.")
                self._deduce_position_from_loan_and_set_status2(f"{self.log_prefix}{sync_type}", base_asset_loan_amount, usdc_loan_amount, latest_price, {"current_status": STATUT_2_ENTRY_FILLED_OCO_PENDING})
        elif num_open_orders == 0 and has_any_significant_loan: # Position active, OCO filled or disappeared
             logger.info(f"{self.log_prefix}{sync_type} No open orders, but loan. Assuming STATUT_2 (OCO needs check/placement).")
             self._deduce_position_from_loan_and_set_status2(f"{self.log_prefix}{sync_type}", base_asset_loan_amount, usdc_loan_amount, latest_price, {"current_status": STATUT_2_ENTRY_FILLED_OCO_PENDING})
        elif num_open_orders > 0 and not has_any_significant_loan: # Stale orders
            self._cancel_all_orders_for_pair(f"{self.log_prefix}{sync_type}[STALE_ORDERS_NO_LOAN]")
            self.state_manager.transition_to_status_1(f"{sync_type.strip('[]')}_STALE_ORDERS_NO_LOAN")
            self.current_trade_cycle_id = None
        else: # Fallback
            self._cancel_all_orders_for_pair(f"{self.log_prefix}{sync_type}[UNHANDLED_STATE_FALLBACK]")
            if has_any_significant_loan: self._handle_loan_without_clear_position(f"{self.log_prefix}{sync_type}", base_asset_loan_amount, usdc_loan_amount, "UNHANDLED_STATE_REPAY")
            self.state_manager.transition_to_status_1(f"{sync_type.strip('[]')}_FALLBACK_UNHANDLED")
            self.current_trade_cycle_id = None
        logger.info(f"{self.log_prefix}{sync_type} Status after sync: {self.state_manager.get_current_status_name()}")

    def _deduce_position_from_loan_and_set_status2(self, log_prefix_context: str, base_loan: float, quote_loan: float, current_price: float, updates_dict: Dict[str, Any]):
        if not self.state_manager or not self.execution_client: return
        logger.info(f"{log_prefix_context} Deducing position from loans. BaseLoan: {base_loan}, QuoteLoan: {quote_loan}, Price: {current_price}")
        
        pos_side_deduced: Optional[str] = None
        qty_deduced_from_loan: float = 0.0
        loan_asset_for_pos: Optional[str] = None
        loan_amount_for_pos: float = 0.0
        entry_price_deduced = current_price if current_price > 0 else self.state_manager.get_state_snapshot().get("position_entry_price", 0.0)

        if entry_price_deduced <= 0: # Try to get a fresh price if current_price was bad
            try:
                ticker = self.execution_client.client.get_symbol_ticker(symbol=self.pair_symbol) # type: ignore
                if ticker and 'price' in ticker: entry_price_deduced = float(ticker['price'])
            except Exception as e_ticker: logger.error(f"{log_prefix_context} Error fetching ticker for loan deduction: {e_ticker}")
        
        if entry_price_deduced <= 0:
            logger.error(f"{log_prefix_context} Cannot deduce position: entry price is zero or invalid after fallbacks.")
            self._handle_loan_without_clear_position(log_prefix_context, base_loan, quote_loan, "PRICE_UNAVAILABLE_FOR_LOAN_DEDUCE")
            return

        # Logic: If base asset loan is significant (in value) and larger than quote loan, assume SHORT.
        # Else if quote asset loan is significant, assume LONG.
        base_loan_value = base_loan * entry_price_deduced
        if base_loan_value > SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT and base_loan_value > (quote_loan * 0.9): # Base loan is dominant
            pos_side_deduced = "SELL"
            qty_deduced_from_loan = base_loan
            loan_asset_for_pos = self.base_asset
            loan_amount_for_pos = base_loan
        elif quote_loan > SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT: # Quote loan is dominant or only one present
            pos_side_deduced = "BUY"
            qty_deduced_from_loan = quote_loan / entry_price_deduced
            loan_asset_for_pos = self.quote_asset
            loan_amount_for_pos = quote_loan
        
        if pos_side_deduced and qty_deduced_from_loan > MIN_EXECUTED_QTY_THRESHOLD:
            state_updates = updates_dict.copy()
            state_updates.update({
                "position_side": pos_side_deduced,
                "position_quantity": qty_deduced_from_loan,
                "position_entry_price": entry_price_deduced,
                "position_entry_timestamp": int(time.time() * 1000),
                "loan_details": {"asset": loan_asset_for_pos, "amount": loan_amount_for_pos, "timestamp": int(time.time()*1000)},
                "entry_order_details": {
                    "status": "FILLED_BY_LOAN_DEDUCTION", "side": pos_side_deduced.upper(),
                    "executedQty": qty_deduced_from_loan, "cummulativeQuoteQty": qty_deduced_from_loan * entry_price_deduced,
                    "updateTime": int(time.time()*1000)
                },
                "pending_sl_tp_raw": self.state_manager.get_state_snapshot().get("pending_sl_tp_raw", {}) # Preserve if any
            })
            if not self.current_trade_cycle_id and not self.state_manager.get_state_snapshot().get("current_trade_cycle_id"):
                self.current_trade_cycle_id = f"recovered_{self.pair_symbol}_{self.account_config.account_alias}_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"
                state_updates["current_trade_cycle_id"] = self.current_trade_cycle_id
            elif self.current_trade_cycle_id:
                 state_updates["current_trade_cycle_id"] = self.current_trade_cycle_id
            
            self.state_manager.update_specific_fields(state_updates)
            logger.info(f"{log_prefix_context} Deduced position: {pos_side_deduced} {qty_deduced_from_loan:.6f} @ {entry_price_deduced:.4f}. State set to STATUT_2.")
        else:
            logger.warning(f"{log_prefix_context} Could not deduce clear position from loans. BaseVal:{base_loan_value:.2f}, QuoteLoan:{quote_loan:.2f}")
            self._handle_loan_without_clear_position(log_prefix_context, base_loan, quote_loan, "AMBIGUOUS_LOAN_FOR_DEDUCE")

    def _handle_loan_without_clear_position(self, log_prefix_context: str, base_loan: float, quote_loan: float, reason_suffix: str):
        if not self.execution_client or not self.state_manager: return
        logger.warning(f"{log_prefix_context} Handling loans without clear position. Base: {base_loan}, Quote: {quote_loan}. Reason: {reason_suffix}")
        cleaned_something = False
        if base_loan > MIN_EXECUTED_QTY_THRESHOLD :
            logger.info(f"{log_prefix_context} Attempting to repay base asset loan: {base_loan} {self.base_asset}")
            repay_res_base = self.execution_client.repay_margin_loan(asset=self.base_asset, amount=str(base_loan), isolated_symbol_pair=self.pair_symbol if self.is_isolated_margin_trading else None)
            self._log_trade_event(f"CLEANUP_REPAY_BASE_ASSET_{reason_suffix}", {"amount": base_loan, "api_response": repay_res_base})
            if repay_res_base and (repay_res_base.get("status") == "SUCCESS" or repay_res_base.get("data", {}).get("tranId")):
                cleaned_something = True
        
        if quote_loan > SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT : 
            logger.info(f"{log_prefix_context} Attempting to repay quote asset loan: {quote_loan} {self.quote_asset}")
            repay_res_quote = self.execution_client.repay_margin_loan(asset=self.quote_asset, amount=str(quote_loan), isolated_symbol_pair=self.pair_symbol if self.is_isolated_margin_trading else None)
            self._log_trade_event(f"CLEANUP_REPAY_QUOTE_ASSET_{reason_suffix}", {"amount": quote_loan, "api_response": repay_res_quote})
            if repay_res_quote and (repay_res_quote.get("status") == "SUCCESS" or repay_res_quote.get("data", {}).get("tranId")):
                cleaned_something = True
        
        if cleaned_something or self.state_manager.get_current_status() != STATUT_1_NO_TRADE_NO_OCO :
            self.state_manager.transition_to_status_1(f"CLEANUP_LOAN_{reason_suffix}")
            self.current_trade_cycle_id = None

    def _cancel_all_orders_for_pair(self, reason_prefix: str):
        if not self.execution_client: return
        logger.info(f"{reason_prefix} Attempting to cancel all open orders for {self.pair_symbol} on account {self.account_config.account_alias} (Isolated: {self.is_isolated_margin_trading}).")
        open_orders_to_cancel = self.execution_client.get_all_open_margin_orders(self.pair_symbol, self.is_isolated_margin_trading)
        cancelled_ids_or_lists = []
        if not open_orders_to_cancel:
            logger.info(f"{reason_prefix} No open orders found for {self.pair_symbol} to cancel.")
            return

        processed_order_list_ids = set()
        for order in open_orders_to_cancel:
            order_id_to_cancel = order.get('orderId')
            client_order_id_to_cancel = order.get('clientOrderId')
            order_list_id = order.get('orderListId', -1) # -1 if not part of OCO

            cancel_res: Optional[Dict[str, Any]] = None
            event_type = "CANCEL_UNKNOWN_ATTEMPT"
            id_logged: Any = None

            if order_list_id != -1 and order_list_id not in processed_order_list_ids:
                logger.info(f"{reason_prefix} Found OCO (OrderListId: {order_list_id}). Cancelling list.")
                cancel_res = self.execution_client.cancel_margin_oco_order(
                    symbol=self.pair_symbol, order_list_id=int(order_list_id), # API expects int
                    list_client_order_id=order.get('listClientOrderId'), # Can be None
                    is_isolated=self.is_isolated_margin_trading
                )
                event_type = "CANCEL_OCO_LIST_ATTEMPT"
                processed_order_list_ids.add(order_list_id)
                id_logged = order_list_id
            elif order_list_id == -1 and order_id_to_cancel: # Single order not part of an OCO
                logger.info(f"{reason_prefix} Found single order (ID: {order_id_to_cancel}). Cancelling.")
                cancel_res = self.execution_client.cancel_margin_order(
                    symbol=self.pair_symbol, order_id=str(order_id_to_cancel),
                    orig_client_order_id=client_order_id_to_cancel, # Can be None
                    is_isolated=self.is_isolated_margin_trading
                )
                event_type = "CANCEL_SINGLE_ORDER_ATTEMPT"
                id_logged = order_id_to_cancel
            else:
                logger.debug(f"{reason_prefix} Skipping order (already processed as part of OCO or no ID): {order}")
                continue
            
            self._log_trade_event(event_type, {"order_to_cancel": order, "api_response": cancel_res})
            if cancel_res and ( (cancel_res.get("data") and (cancel_res.get("data",{}).get("orderListId") or cancel_res.get("data",{}).get("orderId"))) or cancel_res.get("status") == "SUCCESS" ):
                cancelled_ids_or_lists.append(id_logged)
                logger.info(f"{reason_prefix} Successfully sent cancel request for order/list {id_logged}.")
            else:
                logger.error(f"{reason_prefix} Failed to cancel order/list {id_logged}. Response: {cancel_res}")
            time.sleep(API_CALL_DELAY_S)
        logger.info(f"{reason_prefix} Cancellation process finished. Cancelled IDs/Lists: {cancelled_ids_or_lists}")

    # ... (run, stop_trading, _check_new_1min_kline_and_trigger_preprocessing, _log_trade_event as before) ...
    # ... (The full implementations for _check_and_process_orders_via_rest, _handle_status_X, 
    #      _handle_trade_closure_and_loan_repayment, _periodic_full_state_sync are extensive
    #      and would follow the detailed logic from the prompt, using the instance members like
    #      self.execution_client, self.state_manager, self.strategy, self.log_prefix, etc.)

    # --- Full Implementations of Core Trading Logic Methods ---

    def _check_and_process_orders_via_rest(self):
        if not self.execution_client or not self.state_manager: return
        current_status = self.state_manager.get_current_status()
        state = self.state_manager.get_state_snapshot()
        cycle_id = self.current_trade_cycle_id or state.get("current_trade_cycle_id", "NO_CYCLE_ORDER_CHECK")
        log_ctx = f"{self.log_prefix}[Cycle:{cycle_id}]"

        pending_entry_id_server = state.get("pending_entry_order_id")
        pending_entry_id_client = state.get("pending_entry_client_order_id")

        if pending_entry_id_server and current_status == STATUT_1_NO_TRADE_NO_OCO:
            logger.debug(f"{log_ctx} Checking pending entry order ID: {pending_entry_id_server}")
            order_details = self.execution_client.get_margin_order_status(
                symbol=self.pair_symbol, order_id=pending_entry_id_server,
                orig_client_order_id=pending_entry_id_client, is_isolated=self.is_isolated_margin_trading
            )
            if order_details and order_details.get("status") == "FILLED":
                executed_qty_api = float(order_details.get("executedQty", 0.0))
                if executed_qty_api <= MIN_EXECUTED_QTY_THRESHOLD:
                    logger.error(f"{log_ctx} Entry order {pending_entry_id_server} FILLED with zero/sub-threshold qty: {executed_qty_api}.")
                    self._log_trade_event("ENTRY_ORDER_FILLED_ZERO_QTY_ERROR", {"order_api_response": order_details})
                    self.state_manager.transition_to_status_1("ENTRY_FILLED_ZERO_QTY")
                    self.current_trade_cycle_id = None; return

                logger.info(f"{log_ctx} Entry order {pending_entry_id_server} FILLED.")
                self._log_trade_event("ENTRY_ORDER_FILLED", {"order_api_response": order_details})
                loan_asset = self.quote_asset if order_details.get("side") == "BUY" else self.base_asset
                loan_amount = float(order_details.get("cummulativeQuoteQty")) if order_details.get("side") == "BUY" else executed_qty_api
                loan_info = {"asset": loan_asset, "amount": loan_amount, "timestamp": order_details.get("updateTime")}
                self.state_manager.transition_to_status_2(order_details, loan_info)
                self.oco_confirmation_attempts = 0
            elif order_details and order_details.get("status") in ["CANCELED", "EXPIRED", "REJECTED", "PENDING_CANCEL", "NOT_FOUND"]:
                logger.warning(f"{log_ctx} Entry order {pending_entry_id_server} status: {order_details.get('status')}. To STATUT_1.")
                self._log_trade_event(f"ENTRY_ORDER_{order_details.get('status', 'FAILED').upper()}", {"order_api_response": order_details})
                self.state_manager.transition_to_status_1(f"ENTRY_{order_details.get('status', 'FAILED')}")
                self.current_trade_cycle_id = None
            elif order_details: logger.debug(f"{log_ctx} Entry order {pending_entry_id_server} status: {order_details.get('status')}.")
            else: logger.error(f"{log_ctx} Could not get status for entry order {pending_entry_id_server}.")

        elif current_status == STATUT_2_ENTRY_FILLED_OCO_PENDING and state.get("pending_oco_list_client_order_id"):
            pending_list_client_id = state.get("pending_oco_list_client_order_id")
            logger.debug(f"{log_ctx} Checking pending OCO (ListClientOrderId: {pending_list_client_id})")
            open_ocos = self.execution_client.get_open_margin_oco_orders(self.pair_symbol, self.is_isolated_margin_trading)
            found_oco_in_api = next((oco for oco in open_ocos if oco.get("listClientOrderId") == pending_list_client_id), None)
            
            if found_oco_in_api:
                logger.info(f"{log_ctx} OCO (ListClientOrderID: {pending_list_client_id}, API ID: {found_oco_in_api.get('orderListId')}) confirmed ACTIVE.")
                self._log_trade_event("OCO_ORDER_CONFIRMED_ACTIVE", {"oco_api_response": found_oco_in_api})
                self.state_manager.transition_to_status_3(found_oco_in_api)
                self.oco_confirmation_attempts = 0
            else:
                self.oco_confirmation_attempts += 1
                logger.warning(f"{log_ctx} Pending OCO (ListClientOrderID: {pending_list_client_id}) not found. Attempt {self.oco_confirmation_attempts}/{MAX_OCO_CONFIRMATION_ATTEMPTS}.")
                if self.oco_confirmation_attempts >= MAX_OCO_CONFIRMATION_ATTEMPTS:
                    logger.error(f"{log_ctx} Max attempts for OCO confirmation ({pending_list_client_id}). Assuming failure.")
                    self._log_trade_event("OCO_CONFIRMATION_FAILED_MAX_ATTEMPTS", {"pending_oco_list_client_order_id": pending_list_client_id})
                    self._handle_trade_closure_and_loan_repayment(state, "OCO_CONFIRMATION_FAILURE", None) # Close position
                    self.oco_confirmation_attempts = 0

        elif current_status == STATUT_3_OCO_ACTIVE and (state.get("active_oco_order_list_id") or state.get("active_oco_list_client_order_id")):
            oco_id_server = state.get("active_oco_order_list_id")
            oco_client_id = state.get("active_oco_list_client_order_id") # Fallback if server ID somehow missing
            logger.debug(f"{log_ctx} Checking active OCO (ServerID: {oco_id_server}, ClientID: {oco_client_id})")
            
            # Check if OCO is still active or if one leg filled
            # Getting all open orders is more reliable than querying OCO by ID if one leg might have filled
            open_orders_for_pair = self.execution_client.get_all_open_margin_orders(self.pair_symbol, self.is_isolated_margin_trading)
            num_open_orders_for_pair = len(open_orders_for_pair)

            active_oco_still_on_exchange = False
            if oco_id_server:
                active_oco_still_on_exchange = any(str(o.get("orderListId")) == str(oco_id_server) for o in open_orders_for_pair if o.get("orderListId", -1) != -1)
            elif oco_client_id: # Fallback check if server ID wasn't properly stored but client ID was
                active_oco_still_on_exchange = any(o.get("listClientOrderId") == oco_client_id for o in open_orders_for_pair if o.get("listClientOrderId"))

            if not active_oco_still_on_exchange and num_open_orders_for_pair == 0:
                logger.info(f"{log_ctx} Active OCO (ServerID: {oco_id_server}) no longer found, 0 open orders. Assuming one leg filled.")
                sl_order_id, tp_order_id = state.get("active_sl_order_id"), state.get("active_tp_order_id")
                closed_order_data, exit_reason_code = None, "OCO_LEG_FILLED_UNKNOWN"

                if sl_order_id: # Check SL first
                    sl_status = self.execution_client.get_margin_order_status(self.pair_symbol, order_id=sl_order_id, is_isolated=self.is_isolated_margin_trading)
                    if sl_status and sl_status.get("status") == "FILLED":
                        closed_order_data, exit_reason_code = sl_status, "SL_FILLED"
                if not closed_order_data and tp_order_id: # Then check TP
                    tp_status = self.execution_client.get_margin_order_status(self.pair_symbol, order_id=tp_order_id, is_isolated=self.is_isolated_margin_trading)
                    if tp_status and tp_status.get("status") == "FILLED":
                        closed_order_data, exit_reason_code = tp_status, "TP_FILLED"
                
                self._log_trade_event(exit_reason_code, {"closed_order_details": closed_order_data, "active_oco_state": state.get("active_oco_details")})
                self._handle_trade_closure_and_loan_repayment(state, exit_reason_code, closed_order_data)
            elif num_open_orders_for_pair > 0 and not active_oco_still_on_exchange:
                 logger.warning(f"{log_ctx} OCO list (ServerID: {oco_id_server}) not found, but {num_open_orders_for_pair} order(s) still open. This is unexpected. Cancelling remaining.")
                 self._cancel_all_orders_for_pair(f"{log_ctx}[DANGLING_ORDERS_POST_OCO]")
                 self._handle_trade_closure_and_loan_repayment(state, "DANGLING_ORDERS_POST_OCO", None)
            elif active_oco_still_on_exchange:
                 logger.debug(f"{log_ctx} OCO (ServerID: {oco_id_server}) still active on exchange.")


    def _handle_status_1_no_trade(self):
        if not self.strategy or not self.execution_client or not self.state_manager or not self.processed_data_file_path: return
        state = self.state_manager.get_state_snapshot()
        log_ctx = f"{self.log_prefix}[Cycle:{state.get('current_trade_cycle_id', 'NEW_CYCLE')}]"

        if state.get("pending_entry_order_id"):
            logger.debug(f"{log_ctx} Status 1: Entry order {state.get('pending_entry_order_id')} already pending. Skipping new signal.")
            return

        current_usdc_balance = self.execution_client.get_margin_usdc_balance(asset=USDC_ASSET, symbol_pair_for_isolated=self.pair_symbol if self.is_isolated_margin_trading else None)
        current_usdc_balance = current_usdc_balance if current_usdc_balance is not None else 0.0
        self.state_manager.update_specific_fields({"available_capital_at_last_check": current_usdc_balance})

        if current_usdc_balance < MIN_USDC_BALANCE_FOR_TRADE:
            logger.warning(f"{log_ctx} Status 1: Insufficient {USDC_ASSET} balance ({current_usdc_balance:.2f}). Min required: {MIN_USDC_BALANCE_FOR_TRADE}. No new trade.")
            return

        latest_agg_data_df = None
        if self.processed_data_file_path.exists():
            try:
                latest_agg_data_df = pd.read_csv(self.processed_data_file_path)
                if 'timestamp' in latest_agg_data_df.columns: # Ensure timestamp is index for strategy
                    latest_agg_data_df['timestamp'] = pd.to_datetime(latest_agg_data_df['timestamp'], errors='coerce', utc=True)
                    latest_agg_data_df = latest_agg_data_df.set_index('timestamp').sort_index()
            except Exception as e_read_proc:
                logger.error(f"{log_ctx} Error reading processed data {self.processed_data_file_path}: {e_read_proc}")
        
        if latest_agg_data_df is None or latest_agg_data_df.empty:
            logger.warning(f"{log_ctx} Status 1: No processed aggregated data available from {self.processed_data_file_path}. Cannot generate signal.")
            return
        
        symbol_info = self.execution_client.get_symbol_info(self.pair_symbol)
        if not symbol_info:
            logger.error(f"{log_ctx} Status 1: Failed to get symbol info. Cannot generate order request.")
            return
        
        order_request_tuple = self.strategy.generate_order_request(
            data=latest_agg_data_df, symbol=self.pair_symbol, current_position=0,
            available_capital=current_usdc_balance, symbol_info=symbol_info
        )
        
        if order_request_tuple:
            entry_params_from_strat, sl_tp_raw = order_request_tuple
            
            # Generate a new trade cycle ID
            self.current_trade_cycle_id = f"trade_{self.pair_symbol.lower()}_{self.account_config.account_alias.lower()}_{self.context_label.replace(' ','_').lower()}_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"
            log_ctx_new_cycle = f"{self.log_prefix}[Cycle:{self.current_trade_cycle_id}]" # Update log_ctx with new ID
            logger.info(f"{log_ctx_new_cycle} Entry signal generated by strategy.")
            self._log_trade_event("ENTRY_SIGNAL_GENERATED", {"entry_params_suggested": entry_params_from_strat, "sl_tp_raw": sl_tp_raw})
            
            # Prepare API parameters
            entry_params_api = entry_params_from_strat.copy()
            entry_params_api['isIsolated'] = "TRUE" if self.is_isolated_margin_trading else "FALSE"
            
            response = self.execution_client.place_margin_order(**entry_params_api)
            log_detail_entry_sent = {"params_sent_to_api": entry_params_api, "api_response": response, "sl_tp_intended": sl_tp_raw}
            self._log_trade_event("ENTRY_ORDER_SENT", log_detail_entry_sent)
            
            if response and response.get("status") == "SUCCESS" and response.get("data"):
                api_data = response["data"]
                order_id, client_order_id = api_data.get("orderId"), api_data.get("clientOrderId")
                if order_id and client_order_id:
                    logger.info(f"{log_ctx_new_cycle} Entry order submitted. OrderID: {order_id}, ClientOrderID: {client_order_id}")
                    self.state_manager.prepare_for_entry_order(entry_params_from_strat, sl_tp_raw, self.current_trade_cycle_id)
                    self.state_manager.record_placed_entry_order(str(order_id), client_order_id) # Ensure order_id is str
                else:
                    logger.error(f"{log_ctx_new_cycle} Entry API success, but missing OrderID or ClientOrderID in response: {api_data}")
                    self.state_manager.set_last_error(f"Entry API success, missing IDs: {str(api_data)[:100]}")
                    self.current_trade_cycle_id = None # Clear cycle ID if placement failed to register
            else:
                error_msg = response.get("message", 'No response data') if response and isinstance(response, dict) else 'Placement failed, no response'
                logger.error(f"{log_ctx_new_cycle} Entry order placement failed: {error_msg}")
                self.state_manager.set_last_error(f"Entry placement failed: {error_msg}")
                self.current_trade_cycle_id = None # Clear cycle ID on failure
        else:
            logger.debug(f"{log_ctx} No entry signal generated by strategy.")


    def _handle_status_2_oco_pending(self):
        if not self.strategy or not self.execution_client or not self.state_manager: return
        state = self.state_manager.get_state_snapshot()
        cycle_id = self.current_trade_cycle_id or state.get("current_trade_cycle_id", "UNKNOWN_CYCLE_OCO_PENDING")
        log_ctx = f"{self.log_prefix}[Cycle:{cycle_id}]"

        if state.get("pending_oco_list_client_order_id"):
            logger.debug(f"{log_ctx} Status 2: OCO order (ListClientOrderID: {state.get('pending_oco_list_client_order_id')}) already pending placement. Skipping.")
            return

        pos_side = state.get("position_side")
        pos_qty_float = state.get("position_quantity")
        sl_tp_raw = state.get("pending_sl_tp_raw", {}) # This should have been set when entry was prepared

        if not (pos_side and isinstance(pos_qty_float, float) and pos_qty_float > MIN_EXECUTED_QTY_THRESHOLD and
                sl_tp_raw and sl_tp_raw.get('sl_price') is not None and sl_tp_raw.get('tp_price') is not None):
            error_msg = f"STATUT_2: Missing critical data for OCO. Side:{pos_side}, Qty:{pos_qty_float}, SL/TP Raw:{sl_tp_raw}"
            logger.error(f"{log_ctx} {error_msg}")
            self.state_manager.set_last_error(error_msg)
            if pos_qty_float is not None and pos_qty_float <= MIN_EXECUTED_QTY_THRESHOLD: # Position is zero, critical error
                logger.error(f"{log_ctx} Position quantity zero/sub-threshold in STATUT_2. Cannot place OCO. Attempting closure.")
                self._handle_trade_closure_and_loan_repayment(state, "ZERO_QTY_IN_STATUS_2_FOR_OCO", state.get("entry_order_details"))
            return
        
        symbol_info = self.execution_client.get_symbol_info(self.pair_symbol)
        if not symbol_info:
            logger.error(f"{log_ctx} Status 2: Failed to get symbol info for OCO. Cannot proceed."); return

        price_prec_val = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
        qty_prec_val = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
        if price_prec_val is None or qty_prec_val is None:
            logger.error(f"{log_ctx} Status 2: Failed to get price/qty precision for OCO."); return
        
        # Use strategy's _build_oco_params method
        oco_params_from_strat = self.strategy._build_oco_params( # type: ignore
            symbol=self.pair_symbol, position_side=str(pos_side), executed_qty=float(pos_qty_float),
            sl_price_raw=float(sl_tp_raw['sl_price']), tp_price_raw=float(sl_tp_raw['tp_price']),
            price_precision=price_prec_val, qty_precision=qty_prec_val, symbol_info=symbol_info
        )
        if not oco_params_from_strat:
            logger.error(f"{log_ctx} Status 2: Failed to build OCO parameters from strategy. SL/TP might be too close or invalid.");
            self.state_manager.set_last_error("Failed to build OCO params from strategy.")
            # Consider closing the position if OCO cannot be built, as it's unprotected
            logger.warning(f"{log_ctx} Unprotected position due to OCO build failure. Consider manual intervention or auto-closure logic.")
            # self._handle_trade_closure_and_loan_repayment(state, "OCO_BUILD_FAILURE", state.get("entry_order_details")) # Example auto-closure
            return
        
        oco_params_api = oco_params_from_strat.copy()
        oco_params_api['isIsolated'] = "TRUE" if self.is_isolated_margin_trading else "FALSE"
        # Ensure listClientOrderId is present for tracking
        if "listClientOrderId" not in oco_params_api or not oco_params_api["listClientOrderId"]:
            base_oco_client_id_part = cycle_id.split('_')[-1] if cycle_id and '_' in cycle_id else uuid.uuid4().hex[:4]
            oco_params_api["listClientOrderId"] = f"oco_{self.pair_symbol[:3].lower()}_{base_oco_client_id_part}_{uuid.uuid4().hex[:4]}"
        
        logger.info(f"{log_ctx} Status 2: Attempting to place OCO. Params: {oco_params_api}")
        response = self.execution_client.place_margin_oco_order(**oco_params_api)
        self._log_trade_event("OCO_ORDER_SENT", {"params_sent_to_api": oco_params_api, "api_response": response})
        
        if response and response.get("status") == "SUCCESS" and response.get("data"):
            api_data = response["data"]
            list_client_id = api_data.get("listClientOrderId", oco_params_api["listClientOrderId"])
            order_list_id_api = api_data.get("orderListId")
            logger.info(f"{log_ctx} Status 2: OCO order submitted. ListClientOrderID: {list_client_id}, API OrderListID: {order_list_id_api}")
            self.state_manager.prepare_for_oco_order(oco_params_from_strat, list_client_id, str(order_list_id_api) if order_list_id_api else None)
            self.oco_confirmation_attempts = 0 # Reset for next OCO if needed
        else:
            err_msg = response.get("message", "OCO placement failed, no data") if response and isinstance(response,dict) else "OCO placement failed, no response"
            logger.error(f"{log_ctx} Status 2: OCO placement failed: {err_msg}")
            self.state_manager.set_last_error(f"OCO placement failed: {err_msg}")
            # Position is open and unprotected. This is a critical state.
            # Consider logic to retry OCO placement or close the position.


    def _handle_status_3_oco_active(self):
        # Primarily for logging or actions if OCO is active but no fill yet (e.g., trailing SL update - not in scope here)
        if not self.state_manager: return
        cycle_id = self.current_trade_cycle_id or self.state_manager.get_state_snapshot().get("current_trade_cycle_id", "UNKNOWN_CYCLE_OCO_ACTIVE")
        log_ctx = f"{self.log_prefix}[Cycle:{cycle_id}]"
        logger.debug(f"{log_ctx} Status 3: OCO is active. Monitoring (fill detection in _check_and_process_orders_via_rest).")
        pass


    def _handle_trade_closure_and_loan_repayment(self, closed_trade_state_snapshot: Dict[str, Any],
                                                 exit_reason: str,
                                                 closed_order_details: Optional[Dict[str,Any]]):
        if not self.execution_client or not self.state_manager: return
        
        trade_cycle_id_of_closed_trade = closed_trade_state_snapshot.get("current_trade_cycle_id", f"CLOSURE_UNKNOWN_CYCLE_{int(time.time())}")
        log_ctx = f"{self.log_prefix}[Cycle:{trade_cycle_id_of_closed_trade}]"
        logger.info(f"{log_ctx} Handling trade closure. Reason: {exit_reason}.")
        
        asset_to_repay: Optional[str] = None
        original_loan_details = closed_trade_state_snapshot.get("loan_details", {})
        if isinstance(original_loan_details, dict) and original_loan_details.get("asset"):
            asset_to_repay = original_loan_details.get("asset")
        else: # Fallback: deduce from position side
            position_side_closed = closed_trade_state_snapshot.get("position_side")
            if position_side_closed == "BUY": asset_to_repay = self.quote_asset
            elif position_side_closed == "SELL": asset_to_repay = self.base_asset
        
        if asset_to_repay:
            # Fetch current loan for this asset to get the exact amount to repay
            current_loans_for_asset = self.execution_client.get_active_margin_loans(
                asset=asset_to_repay,
                isolated_symbol_pair=self.pair_symbol if self.is_isolated_margin_trading else None
            )
            actual_loan_to_repay = sum(float(l.get("borrowed",0.0)) for l in current_loans_for_asset if l.get('asset', '').upper() == asset_to_repay.upper())
            
            min_repay_threshold = SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT if asset_to_repay == self.quote_asset else MIN_EXECUTED_QTY_THRESHOLD * 0.95
            
            if actual_loan_to_repay > min_repay_threshold:
                logger.info(f"{log_ctx} Attempting to repay loan of {actual_loan_to_repay:.8f} {asset_to_repay}.")
                # Format amount according to asset precision if known, otherwise use a sensible default like 8 decimals
                # For simplicity, using f-string with .8f
                repay_res = self.execution_client.repay_margin_loan(
                    asset=asset_to_repay, amount=f"{actual_loan_to_repay:.8f}",
                    isolated_symbol_pair=self.pair_symbol if self.is_isolated_margin_trading else None
                )
                self._log_trade_event("LOAN_REPAY_ATTEMPT", {"asset": asset_to_repay, "amount_attempted": actual_loan_to_repay, "api_response": repay_res})
                if not (repay_res and (repay_res.get("status") == "SUCCESS" or repay_res.get("data",{}).get("tranId"))):
                    logger.error(f"{log_ctx} Loan repayment for {asset_to_repay} may have failed. Response: {repay_res}")
                    self.state_manager.set_last_error(f"Repay {asset_to_repay} failed: {repay_res.get('message','No resp') if repay_res else 'No resp'}")
                else:
                    logger.info(f"{log_ctx} Loan repayment for {asset_to_repay} successful. TranID: {repay_res.get('data',{}).get('tranId')}")
            else:
                logger.info(f"{log_ctx} No significant loan ({actual_loan_to_repay:.8f} {asset_to_repay}) found or below threshold {min_repay_threshold} for asset {asset_to_repay}. No repayment needed/attempted.")
        else:
            logger.warning(f"{log_ctx} Could not determine asset to repay for loan. Loan repayment skipped.")

        exit_px_val, pnl_est = None, None
        commission_exit_usdc = 0.0
        if closed_order_details and closed_order_details.get("status") == "FILLED":
             exec_q = float(closed_order_details.get("executedQty",0.0))
             cum_q = float(closed_order_details.get("cummulativeQuoteQty",0.0))
             if exec_q > MIN_EXECUTED_QTY_THRESHOLD: exit_px_val = cum_q / exec_q
             
             # Calculate exit commission
             if 'fills' in closed_order_details and isinstance(closed_order_details['fills'], list):
                 for fill in closed_order_details['fills']:
                     if fill.get('commissionAsset', '').upper() == USDC_ASSET:
                         commission_exit_usdc += float(fill.get('commission', 0.0))
             elif closed_order_details.get('commissionAsset','').upper() == USDC_ASSET: # Fallback if fills not detailed
                 commission_exit_usdc = float(closed_order_details.get('commission',0.0))

        entry_px = closed_trade_state_snapshot.get("position_entry_price")
        qty = closed_trade_state_snapshot.get("position_quantity")
        side = closed_trade_state_snapshot.get("position_side")

        if exit_px_val and isinstance(entry_px, (int,float)) and isinstance(qty, (int,float)) and side:
            pnl_est = (exit_px_val - entry_px) * qty if side == "BUY" else (entry_px - exit_px_val) * qty
            logger.info(f"{log_ctx} Estimated PnL (gross, before exit comm): {pnl_est:.2f} USDC")
        
        self.state_manager.record_closed_trade(exit_reason, exit_px_val, pnl_est, closed_order_details) # Pass full details
        self.state_manager.transition_to_status_1(exit_reason, closed_order_details)
        self.current_trade_cycle_id = None
        self.oco_confirmation_attempts = 0
        logger.info(f"{log_ctx} Trade cycle {trade_cycle_id_of_closed_trade} closed. Manager reset to STATUT_1.")


    def _periodic_full_state_sync(self):
        """Performs a full state synchronization with the exchange."""
        logger.info(f"{self.log_prefix} Performing periodic full state sync...")
        try:
            self._determine_initial_status(is_periodic_sync=True)
            self.last_full_state_sync_time = datetime.now(timezone.utc)
            logger.info(f"{self.log_prefix} Periodic full state sync completed.")
        except Exception as e_sync:
            logger.error(f"{self.log_prefix} Error during periodic full state sync: {e_sync}", exc_info=True)
            if self.state_manager:
                self.state_manager.set_last_error(f"Periodic sync failed: {str(e_sync)[:100]}")
    
    def run(self):
        """Main operational loop for the LiveTradingManager."""
        logger.info(f"{self.log_prefix} Starting LiveTradingManager main loop...")
        try:
            while not self.shutdown_event.is_set():
                current_time_utc = datetime.now(timezone.utc)
                
                if self.last_full_state_sync_time is None or \
                   (current_time_utc - self.last_full_state_sync_time) >= timedelta(minutes=FULL_STATE_SYNC_INTERVAL_MINUTES):
                    self._periodic_full_state_sync()

                new_data_processed_for_cycle = self._check_new_1min_kline_and_trigger_preprocessing()
                time.sleep(API_CALL_DELAY_S) # Brief pause after data ops before order ops
                self._check_and_process_orders_via_rest()

                if not self.state_manager or not self.strategy or not self.execution_client:
                    logger.critical(f"{self.log_prefix} Critical component missing. Shutting down.")
                    self.shutdown_event.set(); break

                current_status = self.state_manager.get_current_status()
                self.current_trade_cycle_id = self.state_manager.get_state_snapshot().get("current_trade_cycle_id")

                if new_data_processed_for_cycle or current_status == STATUT_2_ENTRY_FILLED_OCO_PENDING:
                    if current_status == STATUT_1_NO_TRADE_NO_OCO:
                        if not self.state_manager.get_state_snapshot().get("pending_entry_order_id"):
                            self._handle_status_1_no_trade()
                    elif current_status == STATUT_2_ENTRY_FILLED_OCO_PENDING:
                        if not self.state_manager.get_state_snapshot().get("pending_oco_list_client_order_id"):
                            self._handle_status_2_oco_pending()
                    elif current_status == STATUT_3_OCO_ACTIVE:
                        self._handle_status_3_oco_active()
                    # Unknown status already handled in _determine_initial_status or _check_and_process_orders
                else:
                    logger.debug(f"{self.log_prefix} No new data processed. Status: {current_status}. Cycle: {self.current_trade_cycle_id}")

                if self.shutdown_event.wait(timeout=MAIN_LOOP_SLEEP_S): break
        except Exception as e_loop:
            logger.critical(f"{self.log_prefix} CRITICAL ERROR in manager loop: {e_loop}", exc_info=True)
            if self.state_manager: self.state_manager.set_last_error(f"Critical loop error: {str(e_loop)[:250]}")
        finally:
            self.stop_trading()

    def stop_trading(self):
        if self.shutdown_event.is_set(): return
        logger.info(f"{self.log_prefix} LiveTradingManager stopping...")
        self.shutdown_event.set()
        if self.execution_client and hasattr(self.execution_client, 'close'):
            try: self.execution_client.close()
            except Exception as e_close: logger.error(f"{self.log_prefix} Error closing OrderExecutionClient: {e_close}")
        logger.info(f"{self.log_prefix} LiveTradingManager stopped.")

