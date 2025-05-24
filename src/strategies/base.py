# src/strategies/base.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
import uuid
from typing import Dict, Any, Tuple, Optional

from src.config.loader import load_strategy_config_by_name
from src.utils.exchange_utils import get_pair_config_for_symbol, adjust_price_to_tick_size, adjust_quantity_to_step_size

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Attributes:
        strategy_name (str): Name of the strategy.
        params (dict): Strategy-specific parameters.
        symbol (str): Trading symbol (e.g., "BTCUSDT").
        pair_config (dict): Configuration for the trading pair from exchange info.
        is_futures (bool): True if trading futures, False for spot.
        leverage (int): Leverage to use (relevant for backtesting context).
        initial_equity (float): Initial equity for backtesting context.
        account_type (str): Account type, e.g., "SPOT", "MARGIN", "ISOLATED_MARGIN".
                            Used for OOS logging.
    """
    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        """
        Initializes the BaseStrategy.

        Args:
            strategy_name (str): The name of the strategy.
            symbol (str): The trading symbol.
            params (dict): A dictionary of strategy-specific parameters.
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.params = params
        
        # These will be set by set_backtest_context or a similar method in live trading
        self.pair_config: Optional[Dict[str, Any]] = None
        self.is_futures: bool = False
        self.leverage: int = 1
        self.initial_equity: float = 0.0
        self.price_precision: Optional[int] = None
        self.quantity_precision: Optional[int] = None

        # Load strategy-specific config to get account_type
        # This assumes a structure in config_strategies.json like:
        # "STRATEGY_NAME": { "account_type": "MARGIN", ... }
        strategy_global_config = load_strategy_config_by_name(self.strategy_name)
        self.account_type = strategy_global_config.get("account_type", "SPOT") # Default to SPOT

        self._validate_params()

    def set_backtest_context(self, 
                             pair_config: Dict[str, Any], 
                             is_futures: bool, 
                             leverage: int, 
                             initial_equity: float):
        """
        Sets context information typically available during backtesting.

        Args:
            pair_config (dict): Configuration for the trading pair.
            is_futures (bool): Whether the simulation is for futures.
            leverage (int): Leverage used in the simulation.
            initial_equity (float): Initial equity for the simulation.
        """
        self.pair_config = pair_config
        self.is_futures = is_futures
        self.leverage = leverage
        self.initial_equity = initial_equity
        if self.pair_config:
            self.price_precision = self.pair_config.get('pricePrecision')
            self.quantity_precision = self.pair_config.get('quantityPrecision')
        else:
            logger.warning(f"Pair config not provided to strategy {self.strategy_name} for {self.symbol}. Precision adjustments might fail.")


    @abstractmethod
    def _validate_params(self):
        """
        Validates the strategy-specific parameters.
        Should be implemented by subclasses.
        Raises ValueError if params are invalid.
        """
        pass

    @abstractmethod
    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates technical indicators required by the strategy.

        Args:
            data_feed (pd.DataFrame): DataFrame with OHLCV data.
                                      It should contain 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            pd.DataFrame: DataFrame with original data and calculated indicators.
        """
        pass

    @abstractmethod
    def _generate_signals(self, 
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int, # 1 for long, -1 for short, 0 for none
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Generates trading signals based on indicators and current market state.
        This is the core logic of the strategy.

        Args:
            data_with_indicators (pd.DataFrame): DataFrame with OHLCV data and calculated indicators.
                                                 The last row represents the current candle/time step.
            current_position_open (bool): True if a position is currently open.
            current_position_direction (int): 1 if long, -1 if short, 0 if no position.
            current_entry_price (float): The entry price of the current open position.

        Returns:
            Tuple containing:
            - signal (int): -1 for sell/short, 0 for hold, 1 for buy/long, 2 for exit current position.
            - order_type (str, optional): 'MARKET' or 'LIMIT'. Defaults to 'MARKET' if None.
            - limit_price (float, optional): The limit price if order_type is 'LIMIT'.
            - sl_price (float, optional): Stop loss price.
            - tp_price (float, optional): Take profit price.
            - position_size_pct (float, optional): Percentage of max possible size to trade (0.0 to 1.0).
                                                   Defaults to 1.0 (100%).
        """
        pass

    def get_signal(self, 
                   data_feed: pd.DataFrame, 
                   current_position_open: bool, 
                   current_position_direction: int, 
                   current_entry_price: float,
                   current_equity: float # Added for dynamic position sizing or risk management
                  ) -> Dict[str, Any]:
        """
        Main method to get trading signals and order parameters.

        Args:
            data_feed (pd.DataFrame): DataFrame with OHLCV data.
            current_position_open (bool): True if a position is currently open.
            current_position_direction (int): 1 if long, -1 if short.
            current_entry_price (float): Entry price of the current position.
            current_equity (float): Current equity of the backtest/live account.

        Returns:
            dict: A dictionary containing the signal and order parameters:
                {
                    "signal": int,
                    "order_type": str | None,
                    "limit_price": float | None,
                    "sl_price": float | None,
                    "tp_price": float | None,
                    "position_size_pct": float | None,
                    "entry_order_params": dict | None, # Parameters for the entry order itself
                    "oco_params": dict | None          # Parameters for OCO (SL/TP) if applicable
                }
        """
        if data_feed.empty:
            return self._get_default_signal_response("Data feed is empty.")

        try:
            data_with_indicators = self._calculate_indicators(data_feed.copy()) # Use a copy
        except Exception as e:
            logger.error(f"Error calculating indicators for {self.strategy_name} on {self.symbol}: {e}", exc_info=True)
            return self._get_default_signal_response(f"Indicator calculation error: {e}")

        if data_with_indicators.empty:
            return self._get_default_signal_response("Indicators calculation resulted in empty data.")

        signal, order_type, limit_price, sl_price, tp_price, position_size_pct = \
            self._generate_signals(data_with_indicators, 
                                   current_position_open, 
                                   current_position_direction, 
                                   current_entry_price)

        # Default to MARKET if not specified, 100% size if not specified
        order_type = order_type if order_type else "MARKET"
        position_size_pct = position_size_pct if position_size_pct is not None else 1.0

        # Prepare order parameters for detailed logging (used by simulator)
        entry_order_params = None
        oco_params = None
        
        # Only build entry/oco params if there's an actual entry signal (1 or -1)
        # and not currently in a position (or if strategy allows multiple positions - current logic assumes one)
        if signal in [1, -1] and not current_position_open :
            # Estimate quantity - simulator will calculate final quantity
            # For param generation, we might not have the exact quantity yet,
            # but some strategies might pre-calculate it.
            # For now, we'll pass None or a placeholder; simulator uses position_size_pct.
            # The `_build_..._params` methods should ideally take a calculated quantity.
            # Let's assume for now the strategy might determine a theoretical quantity
            # or the simulator will handle it. For the JSON log, we need a theoretical quantity.
            # This is a slight chicken-and-egg. The simulator calculates max size, then applies pct.
            # For now, let's put a placeholder or make it optional in build methods.
            
            theoretical_quantity_for_params = 0.001 # Placeholder, actual quantity determined by simulator
            # A better approach: strategy's _generate_signals could also return a 'theoretical_quantity_base'
            # if it has its own sizing logic beyond just a percentage.

            entry_side = "BUY" if signal == 1 else "SELL"
            
            entry_order_params = self._build_entry_params_formatted(
                side=entry_side,
                quantity=theoretical_quantity_for_params, # This is theoretical for logging
                order_type=order_type,
                price=limit_price,
                # client_order_id_suffix=f"entry_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}" # Example suffix
            )

            if sl_price is not None and tp_price is not None:
                oco_params = self._build_oco_params(
                    entry_side=entry_side, # Side of the main entry order
                    quantity=theoretical_quantity_for_params, # Theoretical
                    entry_price=limit_price if order_type == "LIMIT" else data_with_indicators['close'].iloc[-1], # Estimated entry
                    sl_price=sl_price,
                    tp_price=tp_price,
                    # client_order_id_suffix=f"oco_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                )
        
        # Adjust prices to tick size if pair_config is available
        if self.pair_config:
            if limit_price is not None:
                limit_price = adjust_price_to_tick_size(limit_price, self.pair_config['filters'], self.price_precision)
            if sl_price is not None:
                sl_price = adjust_price_to_tick_size(sl_price, self.pair_config['filters'], self.price_precision)
            if tp_price is not None:
                tp_price = adjust_price_to_tick_size(tp_price, self.pair_config['filters'], self.price_precision)


        return {
            "signal": signal,
            "order_type": order_type,
            "limit_price": limit_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "position_size_pct": position_size_pct,
            "entry_order_params": entry_order_params,
            "oco_params": oco_params
        }

    def _get_default_signal_response(self, reason: str) -> Dict[str, Any]:
        """Returns a default (hold) signal response dictionary."""
        logger.warning(f"Returning default signal for {self.strategy_name} on {self.symbol}: {reason}")
        return {
            "signal": 0, "order_type": None, "limit_price": None, 
            "sl_price": None, "tp_price": None, "position_size_pct": None,
            "entry_order_params": None, "oco_params": None
        }

    def _build_entry_params_formatted(self, 
                                      side: str, 
                                      quantity: float, 
                                      order_type: str, 
                                      price: Optional[float] = None,
                                      client_order_id_suffix: Optional[str] = None,
                                      time_in_force: Optional[str] = None) -> Dict[str, Any]:
        """
        Constructs parameters for a single entry order, aligning with OOS log structure.
        This is for the 'workingOrder' part of an OTOCO or a standalone entry.
        """
        # Ensure quantity and price are formatted correctly if precisions are known
        if self.quantity_precision is not None:
            quantity = round(quantity, self.quantity_precision)
        if price is not None and self.price_precision is not None:
            price = round(price, self.price_precision)

        base_client_order_id = f"sim_{self.strategy_name[:5]}_{str(uuid.uuid4())[:8]}"
        if client_order_id_suffix:
             entry_client_order_id = f"{base_client_order_id}_{client_order_id_suffix}"
        else:
            entry_client_order_id = f"{base_client_order_id}_entry"


        params = {
            "symbol": self.symbol,
            "side": side.upper(), # "BUY" or "SELL"
            "type": order_type.upper(), # "LIMIT", "MARKET", "STOP_LOSS_LIMIT" etc.
            "quantity": str(quantity),
            "newClientOrderId": entry_client_order_id,
            # "workingType_entry": order_type.upper(), # This seems redundant if "type" is there. For OOS log matching:
            # "workingSide_entry": side.upper(),
            # "workingClientOrderId_entry_simulated": entry_client_order_id,
        }
        if order_type.upper() in ["LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
            if price is None:
                raise ValueError(f"Price must be specified for order type {order_type}")
            params["price"] = str(price)
        
        if time_in_force:
            params["timeInForce"] = time_in_force
        
        # For direct use in OOS log structure under "working..."
        # This method is primarily for the single order. The OOS log structure combines it.
        # The simulator's _record_oos_detailed_trade will pick these fields.
        # Let's return the direct params for an order, simulator will map.
        # The keys requested in OOS log for "working..." are:
        # "workingType_entry", "workingSide_entry", "workingClientOrderId_entry_simulated",
        # "workingPrice_entry_theoretical", "workingQuantity_requested"
        # So, the current `params` dict is mostly aligned.
        # Simulator will map "type" to "workingType_entry", "side" to "workingSide_entry", etc.
        
        return params

    def _build_oco_params(self,
                          entry_side: str, # "BUY" or "SELL" (side of the main entry order)
                          quantity: float,
                          entry_price: float, # Theoretical entry price for SL/TP calculation basis
                          sl_price: float,
                          tp_price: float,
                          sl_order_type: str = "STOP_MARKET", # Or "STOP" for Binance, "STOP_LOSS" for OOS log
                          tp_order_type: str = "LIMIT_MAKER", # Or "LIMIT" for Binance, "TAKE_PROFIT_LIMIT" for OOS log
                          client_order_id_suffix: Optional[str] = None
                         ) -> Dict[str, Any]:
        """
        Constructs parameters for an OCO (One-Cancels-Other) order group,
        specifically for SL (Stop Loss) and TP (Take Profit) orders linked to an entry.
        Aligns with OOS log structure.

        Args:
            entry_side (str): "BUY" (for long entry) or "SELL" (for short entry). SL/TP will be the opposite.
            quantity (float): Quantity for SL and TP orders (usually same as entry).
            entry_price (float): The theoretical price at which the main entry order is placed/filled.
            sl_price (float): The stop price for the stop loss order.
            tp_price (float): The limit price for the take profit order.
            sl_order_type (str): Type for SL order, e.g., "STOP_MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT".
            tp_order_type (str): Type for TP order, e.g., "LIMIT_MAKER", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT".
            client_order_id_suffix (str, optional): Suffix for generating client order IDs.

        Returns:
            Dict[str, Any]: Parameters for the OCO order group.
        """
        if self.quantity_precision is not None:
            quantity = round(quantity, self.quantity_precision)
        if self.price_precision is not None:
            sl_price = round(sl_price, self.price_precision)
            tp_price = round(tp_price, self.price_precision)

        exit_side = "SELL" if entry_side == "BUY" else "BUY"

        base_client_order_id = f"sim_{self.strategy_name[:5]}_{str(uuid.uuid4())[:8]}"
        if client_order_id_suffix:
            list_client_order_id = f"{base_client_order_id}_{client_order_id_suffix}_grp"
            sl_client_order_id = f"{base_client_order_id}_{client_order_id_suffix}_sl"
            tp_client_order_id = f"{base_client_order_id}_{client_order_id_suffix}_tp"
        else:
            list_client_order_id = f"{base_client_order_id}_grp"
            sl_client_order_id = f"{base_client_order_id}_sl"
            tp_client_order_id = f"{base_client_order_id}_tp"

        # Parameters for the Stop Loss leg
        sl_leg_params = {
            "symbol": self.symbol,
            "side": exit_side,
            "type": sl_order_type.upper(), # e.g. STOP_MARKET or STOP_LOSS / STOP_LOSS_LIMIT for OOS log
            "quantity": str(quantity),
            "stopPrice": str(sl_price),
            "newClientOrderId": sl_client_order_id
        }
        # If SL is STOP_LOSS_LIMIT, it might need a limit price (trigger price != execution price)
        # For simplicity, current example implies STOP_MARKET or simple STOP_LOSS.
        # If sl_order_type is "STOP_LOSS_LIMIT", a "price" field (limit price) would be needed.
        # The OOS log example has "limitPrice_theoretical" under "sl_details".
        # Let's assume for now STOP_LOSS (market execution after stop trigger) or STOP_LOSS_LIMIT
        # where limit price for SL is also sl_price (worst case fill at stop).
        # A more advanced strategy could set a limit offset for STOP_LOSS_LIMIT.
        if "LIMIT" in sl_order_type.upper():
             sl_leg_params["price"] = str(sl_price) # Example: if it's a stop-limit, limit price might be same as stop or slightly offset

        # Parameters for the Take Profit leg
        tp_leg_params = {
            "symbol": self.symbol,
            "side": exit_side,
            "type": tp_order_type.upper(), # e.g. LIMIT_MAKER or TAKE_PROFIT / TAKE_PROFIT_LIMIT for OOS log
            "quantity": str(quantity),
            "price": str(tp_price),
            "newClientOrderId": tp_client_order_id
        }
        # If LIMIT_MAKER, timeInForce might be GTC or specific to maker.
        # OOS log uses "LIMIT_MAKER" for TP type.

        # OCO structure for OOS log (inspired by Binance OTOCO structure but for logging)
        # The simulator will map these to its OOS log structure.
        # This method returns the components.
        oco_log_structure = {
            "listClientOrderId": list_client_order_id, # For the group
            # "orders": [entry_order_params, sl_leg_params, tp_leg_params], # If logging full OTOCO structure
            # For the OOS log format, we need sl_details and tp_details separately.
            # The `current_oco_params` in simulator expects a dict with `sl_params` and `tp_params` keys.
            "sl_details": { # Corresponds to "sl_details" in OOS log
                "type": sl_order_type.upper(), # "STOP_LOSS" or "STOP_LOSS_LIMIT"
                "clientOrderId_simulated": sl_client_order_id,
                "stopPrice_theoretical": str(sl_price),
                "limitPrice_theoretical": str(sl_leg_params.get("price")) if "LIMIT" in sl_order_type.upper() else None
            },
            "tp_details": { # Corresponds to "tp_details" in OOS log
                "type": tp_order_type.upper(), # "LIMIT_MAKER" or "TAKE_PROFIT"
                "clientOrderId_simulated": tp_client_order_id,
                "price_theoretical": str(tp_price)
            },
            # The following are for actual API call if needed, not directly for OOS log structure in simulator
            # but good to have if strategy needs to generate full API params.
            # "stopLimitTimeInForce": "GTC", # Example for SL leg if it's STOP_LOSS_LIMIT
            # "limitMakerTimeInForce": "GTC", # Example for TP leg if it's LIMIT_MAKER
        }
        # The simulator expects `current_oco_params` to have `sl_params` and `tp_params` which are the direct order params.
        # So, let's adjust what this returns to be more direct for the simulator's current expectation.
        
        return {
            "listClientOrderId_simulated": list_client_order_id, # Group ID
            "sl_params": sl_leg_params, # Actual SL order params
            "tp_params": tp_leg_params  # Actual TP order params
        }

    def get_params_for_optuna(self) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters that are relevant for Optuna optimization.
        This typically includes parameters that define indicator periods, thresholds, etc.
        Should be overridden by subclasses if they have optimizable parameters.
        """
        return self.params # By default, return all params. Subclasses can filter.

    def update_params(self, new_params: Dict[str, Any]):
        """
        Updates the strategy's parameters.
        Used during optimization.
        """
        self.params.update(new_params)
        self._validate_params() # Re-validate after update
        # Potentially re-initialize or clear cached indicators if params change
        logger.info(f"Strategy {self.strategy_name} parameters updated to: {self.params}")

