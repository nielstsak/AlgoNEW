# src/backtesting/simulator.py
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timezone
import logging

from src.utils.slippage import SlippageSimulator
from src.utils.fees import FeeSimulator
from src.config.loader import load_exchange_config, load_strategy_config_by_name
from src.utils.exchange_utils import get_pair_config_for_symbol
from src.strategies.base import BaseStrategy # For type hinting

logger = logging.getLogger(__name__)

class BacktestSimulator:
    """
    Simulates trading strategies on historical data.

    Handles order execution, position management, PnL calculation,
    and performance metric tracking.
    """
    def __init__(self,
                 df_ohlcv: pd.DataFrame,
                 strategy_instance: BaseStrategy,
                 initial_equity: float,
                 leverage: int,
                 symbol: str,
                 trading_fee_bps: float,
                 slippage_config: dict,
                 is_futures: bool,
                 run_id: str, # For logging context if needed
                 is_oos_simulation: bool = False, # Flag for detailed OOS logging
                 verbosity: int = 1): # 0: silent, 1: basic, 2: detailed
        """
        Initializes the BacktestSimulator.

        Args:
            df_ohlcv (pd.DataFrame): DataFrame with OHLCV data, indexed by timestamp.
                                     Required columns: 'open', 'high', 'low', 'close', 'volume'.
            strategy_instance (BaseStrategy): An instance of the trading strategy.
            initial_equity (float): Starting capital for the backtest.
            leverage (int): Leverage factor to be used.
            symbol (str): Trading symbol (e.g., "BTCUSDT").
            trading_fee_bps (float): Trading fee in basis points (e.g., 7 for 0.07%).
            slippage_config (dict): Configuration for slippage simulation.
            is_futures (bool): True if simulating futures trading, False for spot.
            run_id (str): Unique identifier for the current run.
            is_oos_simulation (bool): If True, enables detailed logging for OOS trades.
            verbosity (int): Controls the level of logging output.
        """
        self.df_ohlcv = df_ohlcv.copy()
        self.strategy = strategy_instance
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.leverage = leverage
        self.symbol = symbol
        self.is_futures = is_futures
        self.run_id = run_id # Store run_id
        self.is_oos_simulation = is_oos_simulation # Store OOS flag
        self.verbosity = verbosity

        self.exchange_config = load_exchange_config() # Assuming default exchange for now
        self.pair_config = get_pair_config_for_symbol(self.symbol, self.exchange_config)
        if not self.pair_config:
            raise ValueError(f"Pair configuration not found for symbol {self.symbol}")

        self.base_asset = self.pair_config['baseAsset']
        self.quote_asset = self.pair_config['quoteAsset']
        self.price_precision = self.pair_config['pricePrecision']
        self.quantity_precision = self.pair_config['quantityPrecision']
        self.min_notional = self.pair_config.get('minNotionalFilter', {}).get('minNotional', 0.0)
        self.min_quantity = self.pair_config.get('lotSizeFilter', {}).get('minQty', 0.0)


        self.slippage_simulator = SlippageSimulator(
            method=slippage_config.get("method", "percentage"),
            percentage_max_bps=slippage_config.get("percentage_max_bps"),
            volume_factor=slippage_config.get("volume_factor"),
            volatility_factor=slippage_config.get("volatility_factor"),
            min_slippage_bps=slippage_config.get("min_slippage_bps", 0),
            max_slippage_bps=slippage_config.get("max_slippage_bps", 100)
        )
        self.fee_simulator = FeeSimulator(fee_bps=trading_fee_bps)

        self.position_open = False
        self.position_size_base = 0.0  # Size in base asset
        self.position_size_quote = 0.0 # Size in quote asset (cost for long, proceeds for short)
        self.entry_price = 0.0
        self.entry_time = None
        self.exit_price = 0.0
        self.exit_time = None
        self.current_sl_price = None
        self.current_tp_price = None
        self.trade_direction = 0  # 1 for long, -1 for short

        self.trades = []
        self.equity_curve = []
        self.daily_equity = {} # For daily equity tracking

        # For margin trading simulation
        self.active_loan = None # Stores details of the current loan
        # Example: {"asset": "USDT", "amount": 10000.0, "entry_price_for_loan_calc": 50000} for shorts
        # Example: {"asset": "USDT", "amount": 10000.0} for longs (loan is in quote)
        self.initial_margin_used = 0.0 # Quote asset value locked as margin

        # For detailed OOS trade logging
        self.oos_detailed_trades_log = []
        self.current_trade_cycle_id = None
        self.current_entry_order_params = None # To store intent for OOS log
        self.current_oco_params = None # To store intent for OOS log

        # Initialize strategy with backtesting context
        self.strategy.set_backtest_context(
            pair_config=self.pair_config,
            is_futures=self.is_futures,
            leverage=self.leverage,
            initial_equity=self.initial_equity # Strategy might need this for position sizing logic
        )


    def _log(self, message: str, level: int = 1):
        """Helper for conditional logging based on verbosity."""
        if self.verbosity >= level:
            logger.info(message)

    def _calculate_max_position_size_base(self, entry_price_estimate: float) -> float:
        """
        Calculates the maximum position size in base asset based on current equity and leverage.
        Considers available equity after accounting for potential margin already used if a position is open (though typically new trades are not opened if one is active).
        """
        if entry_price_estimate <= 0:
            return 0.0

        # Effective equity available for a new trade's margin
        # If a position is already open, this logic might need refinement,
        # but typically strategies manage one position at a time.
        available_margin_capital = self.current_equity - self.initial_margin_used

        if available_margin_capital <= 0:
            return 0.0

        max_total_value_quote = available_margin_capital * self.leverage
        max_size_base = max_total_value_quote / entry_price_estimate
        
        # Apply exchange quantity precision
        max_size_base = np.floor(max_size_base * (10**self.quantity_precision)) / (10**self.quantity_precision)
        return max_size_base

    def _apply_exchange_filters(self, quantity_base: float, price: float) -> float:
        """Applies exchange quantity and notional filters."""
        # Apply quantity precision
        quantity_base = np.floor(quantity_base * (10**self.quantity_precision)) / (10**self.quantity_precision)

        # Check min quantity
        if quantity_base < self.min_quantity:
            if self.verbosity >=2:
                self._log(f"Quantity {quantity_base} less than min_quantity {self.min_quantity}. Setting to 0.", level=2)
            return 0.0

        # Check min notional
        notional_value = quantity_base * price
        if notional_value < self.min_notional:
            if self.verbosity >=2:
                self._log(f"Notional {notional_value} less than min_notional {self.min_notional}. Setting to 0.", level=2)
            return 0.0
        return quantity_base


    def _simulate_order_execution(self,
                                  timestamp: pd.Timestamp,
                                  current_row: pd.Series,
                                  order_type: str, # 'MARKET', 'LIMIT_ENTRY', 'SL', 'TP', 'MARKET_EXIT'
                                  trade_direction: int, # 1 for long, -1 for short
                                  limit_price: float = None,
                                  stop_price: float = None, # For SL orders
                                  target_quantity_base: float = None, # Desired quantity for entry
                                  is_entry: bool = False):
        """
        Simulates the execution of an order, applying slippage and fees.
        Updates position and equity.

        Returns:
            Tuple (executed_price, executed_quantity_base, fee_paid_quote) or (None, None, None) if not executed.
        """
        open_p, high_p, low_p, close_p = current_row['open'], current_row['high'], current_row['low'], current_row['close']
        executed_price = None
        executed_quantity_base = target_quantity_base if is_entry else self.position_size_base
        
        # Determine execution price based on order type and candle
        if order_type == 'MARKET' or order_type == 'MARKET_EXIT':
            # Assume market orders fill at the open of the next candle (current_row is that candle)
            # For more realistic simulation, could use a fraction of candle range or VWAP if available
            base_exec_price = open_p
        elif order_type == 'LIMIT_ENTRY':
            if trade_direction == 1: # Buy limit
                if low_p <= limit_price: # Limit price touched or crossed
                    base_exec_price = min(open_p, limit_price) # Fill at open or limit, whichever is better for us (higher for buy)
                else:
                    return None, None, None # Limit not reached
            else: # Sell limit (short entry)
                if high_p >= limit_price:
                    base_exec_price = max(open_p, limit_price) # Fill at open or limit, whichever is better for us (lower for sell)
                else:
                    return None, None, None # Limit not reached
        elif order_type == 'SL': # Stop Loss
            if trade_direction == 1: # SL for a long position (sell stop)
                if low_p <= stop_price:
                    base_exec_price = min(open_p, stop_price) # Worst case: stop_price, or open if gapped through
                else:
                    return None, None, None
            else: # SL for a short position (buy stop)
                if high_p >= stop_price:
                    base_exec_price = max(open_p, stop_price) # Worst case: stop_price, or open if gapped through
                else:
                    return None, None, None
        elif order_type == 'TP': # Take Profit
            if trade_direction == 1: # TP for a long position (sell limit)
                if high_p >= limit_price:
                    base_exec_price = max(open_p, limit_price)
                else:
                    return None, None, None
            else: # TP for a short position (buy limit)
                if low_p <= limit_price:
                    base_exec_price = min(open_p, limit_price)
                else:
                    return None, None, None
        else:
            logger.warning(f"Unknown order type: {order_type} at {timestamp}")
            return None, None, None

        # Apply slippage
        executed_price = self.slippage_simulator.simulate_slippage(
            price=base_exec_price,
            volume_at_price=current_row.get('volume', 0), # Use candle volume
            volatility=(high_p - low_p) / low_p if low_p > 0 else 0,
            direction=trade_direction if is_entry else -trade_direction # Slippage is against us
        )
        # Apply price precision
        executed_price = round(executed_price, self.price_precision)

        # For entries, adjust quantity based on filters AFTER slippage affects price
        if is_entry:
            executed_quantity_base = self._apply_exchange_filters(target_quantity_base, executed_price)
            if executed_quantity_base == 0.0:
                self._log(f"Order at {timestamp} for {target_quantity_base} {self.base_asset} resulted in 0 quantity after filters.", level=2)
                return None, None, None # Order effectively cancelled due to filters

        # Calculate fees
        trade_value_quote = executed_quantity_base * executed_price
        fee_paid_quote = self.fee_simulator.calculate_fee(trade_value_quote)

        # --- PnL and Equity Update ---
        if is_entry:
            # Store entry details
            self.position_open = True
            self.trade_direction = trade_direction
            self.entry_price = executed_price
            self.entry_time = timestamp
            self.position_size_base = executed_quantity_base
            self.position_size_quote = trade_value_quote # Cost for long, proceeds for short

            # Simulate Loan
            cost_of_asset_for_entry = executed_price * executed_quantity_base
            margin_locked_quote = cost_of_asset_for_entry / self.leverage
            self.initial_margin_used = margin_locked_quote

            if trade_direction == 1: # Long
                self.active_loan = {
                    "asset": self.quote_asset,
                    "amount": cost_of_asset_for_entry,
                    "timestamp_utc": timestamp.tz_convert(timezone.utc).isoformat()
                }
                self._log(f"{timestamp}: LONG ENTRY @ {executed_price:.{self.price_precision}f}, Qty: {executed_quantity_base:.{self.quantity_precision}f}, Loan: {self.active_loan['amount']:.2f} {self.quote_asset}", level=1)
            else: # Short
                self.active_loan = {
                    "asset": self.base_asset,
                    "amount": executed_quantity_base,
                    "entry_price_for_loan_calc": executed_price, # Store entry price for PnL calc
                    "timestamp_utc": timestamp.tz_convert(timezone.utc).isoformat()
                }
                self._log(f"{timestamp}: SHORT ENTRY @ {executed_price:.{self.price_precision}f}, Qty: {executed_quantity_base:.{self.quantity_precision}f}, Loan: {self.active_loan['amount']:.{self.quantity_precision}f} {self.base_asset}", level=1)

            # Equity impact at entry: only the fee
            self.current_equity -= fee_paid_quote
            self.equity_curve.append({'timestamp': timestamp, 'equity': self.current_equity, 'type': 'entry_fee'})

            # For OOS logging
            if self.is_oos_simulation:
                self.current_trade_cycle_id = str(uuid.uuid4())


        # Not an entry, so it's an exit
        else:
            if not self.position_open: # Should not happen if logic is correct
                logger.error(f"Attempted to exit position at {timestamp} but no position was open.")
                return None, None, None

            self.exit_price = executed_price
            self.exit_time = timestamp
            
            pnl_final_impact_on_equity = 0.0
            exit_reason_for_log = order_type # SL, TP, MARKET_EXIT

            if self.active_loan:
                if self.trade_direction == 1: # Closing a Long
                    proceeds_from_sale_quote = executed_price * self.position_size_base
                    repayment_amount_quote = self.active_loan['amount']
                    net_cash_flow_from_trade_op = proceeds_from_sale_quote - repayment_amount_quote
                    pnl_final_impact_on_equity = net_cash_flow_from_trade_op - fee_paid_quote
                    self._log(f"{timestamp}: LONG EXIT @ {executed_price:.{self.price_precision}f}, Qty: {self.position_size_base:.{self.quantity_precision}f}. Loan Repaid: {repayment_amount_quote:.2f} {self.quote_asset}. PnL (post-loan, pre-exit-fee): {net_cash_flow_from_trade_op:.2f}", level=1)

                else: # Closing a Short
                    cost_to_cover_short_quote = executed_price * self.position_size_base
                    # Loan was in base_asset, repayment is buying it back.
                    # Initial proceeds were from selling self.active_loan['amount'] of base_asset at self.active_loan['entry_price_for_loan_calc']
                    initial_proceeds_quote = self.active_loan['entry_price_for_loan_calc'] * self.active_loan['amount']
                    net_cash_flow_from_trade_op = initial_proceeds_quote - cost_to_cover_short_quote
                    pnl_final_impact_on_equity = net_cash_flow_from_trade_op - fee_paid_quote
                    self._log(f"{timestamp}: SHORT EXIT @ {executed_price:.{self.price_precision}f}, Qty: {self.position_size_base:.{self.quantity_precision}f}. Base Asset Loan Repaid by buying. PnL (post-loan, pre-exit-fee): {net_cash_flow_from_trade_op:.2f}", level=1)
            else: # Should not happen if loan tracking is correct for margin trades
                logger.error(f"Exiting position at {timestamp} but no active loan found. This is unexpected for margin trading simulation.")
                # Fallback to simple PnL if no loan (e.g. if leverage was 1 and loan simulation was skipped)
                pnl_raw = (executed_price - self.entry_price) * self.position_size_base * self.trade_direction
                pnl_final_impact_on_equity = pnl_raw - fee_paid_quote


            # Update equity with the PnL from this trade (after loan repayment & exit fee)
            self.current_equity += pnl_final_impact_on_equity
            self.equity_curve.append({'timestamp': timestamp, 'equity': self.current_equity, 'type': 'trade_pnl'})
            
            # Store trade details
            entry_fee_obj = next((item for item in reversed(self.equity_curve) if item['type'] == 'entry_fee' and item['timestamp'] == self.entry_time), None)
            entry_fee_value = self.trades[-1]['entry_fee'] if self.trades and self.trades[-1]['exit_time'] is None else 0 # Approximate if needed, best to get it from entry
            # This needs to be more robust if multiple trades are possible before one closes.
            # For now, assume one trade at a time.
            # The actual entry fee was already deducted. We need to record it.
            # Let's assume the fee_paid_quote at entry was stored somewhere or re-calculated.
            # For simplicity, we'll re-calculate entry fee for the record if not easily available.
            # Better: store entry_fee when trade is opened.
            # Let's add self.current_trade_entry_fee

            trade_info = {
                'entry_time': self.entry_time,
                'exit_time': self.exit_time,
                'entry_price': self.entry_price,
                'exit_price': executed_price,
                'size_base': self.position_size_base,
                'size_quote': self.position_size_quote, # Entry value
                'pnl_net_quote': pnl_final_impact_on_equity, # This is the net PnL impacting equity
                'entry_fee_quote': self.current_trade_entry_fee,
                'exit_fee_quote': fee_paid_quote,
                'total_fees_quote': self.current_trade_entry_fee + fee_paid_quote,
                'direction': 'long' if self.trade_direction == 1 else 'short',
                'leverage': self.leverage,
                'initial_margin_used_quote': self.initial_margin_used,
                'exit_reason': exit_reason_for_log
            }
            self.trades.append(trade_info)

            # OOS Detailed Log
            if self.is_oos_simulation and self.current_trade_cycle_id:
                self._record_oos_detailed_trade(trade_info)

            # Reset position state
            self._reset_position_state()

        return executed_price, executed_quantity_base, fee_paid_quote

    def _record_oos_detailed_trade(self, trade_info: dict):
        """Helper to construct and append the detailed OOS trade log entry."""
        if not self.current_entry_order_params: # Should be set at entry signal
            logger.warning(f"Missing current_entry_order_params for OOS detailed log for trade ending {trade_info['exit_time']}")
            entry_params_fallback = {}
        else:
            entry_params_fallback = self.current_entry_order_params

        if not self.current_oco_params: # Should be set at entry signal if OCO
             #logger.warning(f"Missing current_oco_params for OOS detailed log for trade ending {trade_info['exit_time']}")
             oco_params_fallback = {} # It's okay if not an OCO order
        else:
            oco_params_fallback = self.current_oco_params
        
        is_short = trade_info['direction'] == 'short'
        entry_side_effect = "AUTO_BORROW_REPAY" if is_short else "MARGIN_BUY"
        
        # Determine order_intent_type
        intent_type = ""
        if oco_params_fallback and oco_params_fallback.get('sl_details') and oco_params_fallback.get('tp_details'):
            intent_type = f"OTOCO_ENTRY_{'SHORT' if is_short else 'LONG'}"
        elif entry_params_fallback.get('workingType_entry'):
            intent_type = f"{entry_params_fallback['workingType_entry'].upper()}_ENTRY_{'SHORT' if is_short else 'LONG'}"
        else: # Fallback
            intent_type = f"UNKNOWN_ENTRY_{'SHORT' if is_short else 'LONG'}"


        log_entry = {
            "trade_cycle_id": self.current_trade_cycle_id,
            "entry_timestamp_simulated_utc": trade_info['entry_time'].tz_convert(timezone.utc).isoformat(),
            "order_intent_type": intent_type,
            "symbol": self.symbol,
            "isIsolated": str(self.strategy.account_type == "ISOLATED_MARGIN").upper(), # Assuming strategy has account_type

            "sideEffectType_entry": entry_side_effect,
            "listClientOrderId_simulated": entry_params_fallback.get("listClientOrderId", f"sim_otoco_{self.current_trade_cycle_id[:8]}"),
            
            "workingType_entry": entry_params_fallback.get("type", "MARKET"), # from _build_entry_params_formatted
            "workingSide_entry": "SELL" if is_short else "BUY",
            "workingClientOrderId_entry_simulated": entry_params_fallback.get("newClientOrderId", f"sim_entry_{self.current_trade_cycle_id[:8]}"),
            "workingPrice_entry_theoretical": str(entry_params_fallback.get("price")) if entry_params_fallback.get("price") is not None else None,
            "workingPrice_entry_executed_slipped": str(trade_info['entry_price']),
            "workingQuantity_requested": str(entry_params_fallback.get("quantity")),
            "workingQuantity_executed": str(trade_info['size_base']),
            
            "pendingSide_exit": "BUY" if is_short else "SELL",
            "pendingQuantity_exit": str(trade_info['size_base']),
            
            "sl_details": {
                "type": oco_params_fallback.get("sl_params", {}).get("type"),
                "clientOrderId_simulated": oco_params_fallback.get("sl_params", {}).get("newClientOrderId", f"sim_sl_{self.current_trade_cycle_id[:8]}" if oco_params_fallback.get("sl_params") else None),
                "stopPrice_theoretical": str(oco_params_fallback.get("sl_params", {}).get("stopPrice")) if oco_params_fallback.get("sl_params", {}).get("stopPrice") is not None else None,
                "limitPrice_theoretical": str(oco_params_fallback.get("sl_params", {}).get("price")) if oco_params_fallback.get("sl_params", {}).get("type", "").endswith("LIMIT") and oco_params_fallback.get("sl_params", {}).get("price") is not None else None,
            },
            "tp_details": {
                "type": oco_params_fallback.get("tp_params", {}).get("type"),
                "clientOrderId_simulated": oco_params_fallback.get("tp_params", {}).get("newClientOrderId", f"sim_tp_{self.current_trade_cycle_id[:8]}" if oco_params_fallback.get("tp_params") else None),
                "price_theoretical": str(oco_params_fallback.get("tp_params", {}).get("price")) if oco_params_fallback.get("tp_params", {}).get("price") is not None else None,
            },
            "exit_timestamp_simulated_utc": trade_info['exit_time'].tz_convert(timezone.utc).isoformat() if trade_info['exit_time'] else None,
            "exit_reason": trade_info['exit_reason'],
            "exit_price_executed_slipped": str(trade_info['exit_price']) if trade_info['exit_price'] else None,
            "pnl_net_usd": trade_info['pnl_net_quote'] # Assuming quote is USD or similar
        }
        self.oos_detailed_trades_log.append(log_entry)
        # Reset for next OOS trade cycle
        self.current_trade_cycle_id = None
        self.current_entry_order_params = None
        self.current_oco_params = None

    def _reset_position_state(self):
        """Resets all position-related state variables."""
        self.position_open = False
        self.position_size_base = 0.0
        self.position_size_quote = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.exit_price = 0.0 # Technically not needed to reset here, but good for clarity
        self.exit_time = None # ditto
        self.trade_direction = 0
        self.current_sl_price = None
        self.current_tp_price = None
        
        self.active_loan = None
        self.initial_margin_used = 0.0
        self.current_trade_entry_fee = 0.0 # Reset stored entry fee for the closed trade


    def run_simulation(self) -> tuple[list, pd.DataFrame, dict, list]:
        """
        Runs the backtest simulation loop.

        Returns:
            A tuple containing:
            - trades (list): List of dictionaries, each representing a trade.
            - equity_curve (pd.DataFrame): DataFrame of equity over time.
            - daily_equity (dict): Dictionary of daily equity values.
            - oos_detailed_trades_log (list): List of detailed OOS trade logs (if applicable).
        """
        if self.df_ohlcv.empty:
            logger.warning("OHLCV data is empty. Skipping simulation.")
            return [], pd.DataFrame(self.equity_curve), self.daily_equity, self.oos_detailed_trades_log

        self.equity_curve.append({'timestamp': self.df_ohlcv.index[0], 'equity': self.initial_equity, 'type': 'initial'})
        self.current_trade_entry_fee = 0.0 # Fee for the current open trade

        for i in range(len(self.df_ohlcv)):
            current_timestamp = self.df_ohlcv.index[i]
            current_row = self.df_ohlcv.iloc[i]
            
            # Prepare data for strategy (current candle + lookback)
            # Ensure enough data for lookback. Strategy should handle this if get_signal is robust.
            # For simplicity, we pass a growing window or rely on strategy to manage its needed history.
            # A more robust way is to pass df_ohlcv up to current_timestamp.
            # self.strategy.update_data(self.df_ohlcv.iloc[:i+1]) # If strategy needs to see all past data
            
            # Update daily equity
            if current_timestamp.date() not in self.daily_equity or current_timestamp == self.df_ohlcv[self.df_ohlcv.index.date == current_timestamp.date()].index[-1]:
                 self.daily_equity[current_timestamp.date()] = self.current_equity


            # --- Check for SL/TP Hits ---
            if self.position_open:
                exit_simulated = False
                exit_reason = None
                # Check TP first, then SL. Exchange behavior can vary.
                # TP check
                if self.current_tp_price:
                    order_t = 'TP'
                    limit_p = self.current_tp_price
                    stop_p = None
                    # trade_direction for TP/SL exit is the direction of the original trade
                    exec_p, exec_q, fee_q = self._simulate_order_execution(current_timestamp, current_row, order_t, self.trade_direction, limit_price=limit_p)
                    if exec_p is not None:
                        self._log(f"{current_timestamp}: TP HIT for {'LONG' if self.trade_direction==1 else 'SHORT'} at {exec_p:.{self.price_precision}f}", level=1)
                        exit_simulated = True
                        exit_reason = "TP_FILLED"
                        # _simulate_order_execution handles trade logging and position reset

                # SL check (only if TP not hit in same candle)
                if not exit_simulated and self.current_sl_price:
                    order_t = 'SL'
                    limit_p = None # For market stop
                    stop_p = self.current_sl_price
                    # If SL is STOP_LIMIT, strategy needs to provide limit offset. For now, assume market SL.
                    exec_p, exec_q, fee_q = self._simulate_order_execution(current_timestamp, current_row, order_t, self.trade_direction, stop_price=stop_p)
                    if exec_p is not None:
                        self._log(f"{current_timestamp}: SL HIT for {'LONG' if self.trade_direction==1 else 'SHORT'} at {exec_p:.{self.price_precision}f}", level=1)
                        exit_simulated = True
                        exit_reason = "SL_FILLED"
                        # _simulate_order_execution handles trade logging and position reset
                
                if exit_simulated: # Position was closed by SL/TP
                    continue # Move to next candle


            # --- Get Signal from Strategy ---
            # Pass historical data up to the point just before the current candle's open
            # The strategy will use current_row['open'] or current_row['close'] of df_slice.iloc[-1] as current price
            df_slice_for_signal = self.df_ohlcv.iloc[:i+1] # Includes current (incomplete) candle for signal generation

            signal_result = self.strategy.get_signal(
                data_feed=df_slice_for_signal, # Strategy uses the latest row for current price context
                current_position_open=self.position_open,
                current_position_direction=self.trade_direction,
                current_entry_price=self.entry_price,
                current_equity=self.current_equity # Pass current equity for dynamic position sizing
            )
            
            # Unpack signal_result (assuming it's a tuple or dict)
            # Expected: signal, order_type_strat, limit_price_strat, sl_price_strat, tp_price_strat, position_size_pct_strat, entry_order_params, oco_params
            # The new strategy interface might return a more structured object.
            # For now, let's assume the strategy returns a dictionary or specific tuple.
            # This part needs to align with BaseStrategy.get_signal's return signature.

            signal = signal_result.get("signal") # -1 (sell/short), 0 (hold), 1 (buy/long), 2 (exit current)
            order_type_strat = signal_result.get("order_type", "MARKET") # MARKET, LIMIT
            limit_price_strat = signal_result.get("limit_price")
            sl_price_strat = signal_result.get("sl_price")
            tp_price_strat = signal_result.get("tp_price")
            position_size_pct_strat = signal_result.get("position_size_pct", 1.0) # Percentage of max possible size
            
            # For detailed OOS logging - strategy must return these
            entry_order_params_strat = signal_result.get("entry_order_params")
            oco_params_strat = signal_result.get("oco_params")


            # --- Execute Based on Signal ---
            if self.position_open:
                if signal == 2 or \
                   (self.trade_direction == 1 and signal == -1) or \
                   (self.trade_direction == -1 and signal == 1): # Exit signal or reversal
                    
                    self._log(f"{current_timestamp}: EXIT SIGNAL received for {'LONG' if self.trade_direction==1 else 'SHORT'}", level=1)
                    # Market exit based on signal
                    exec_p, exec_q, fee_q = self._simulate_order_execution(current_timestamp, current_row, 'MARKET_EXIT', self.trade_direction, is_entry=False)
                    if exec_p is not None:
                        self._log(f"Exited at {exec_p:.{self.price_precision}f}", level=2)
                        # _simulate_order_execution handles trade logging and position reset
                        # If a reversal is signaled, the entry part will be handled in the next iteration or logic block
                    # else: # Market exit failed (should be rare unless illiquid)
                        # self._log(f"Market exit FAILED at {current_timestamp}", level=1)

            # Try to open a new position if no position is open and signal is entry
            if not self.position_open and (signal == 1 or signal == -1):
                target_direction = signal # 1 for long, -1 for short
                
                # Estimate entry price for position sizing (can use current open or close)
                entry_price_estimate = current_row['open'] # Or current_row['close'] or limit_price_strat
                if order_type_strat == 'LIMIT' and limit_price_strat is not None:
                    entry_price_estimate = limit_price_strat
                
                max_size_base = self._calculate_max_position_size_base(entry_price_estimate)
                if max_size_base <= 0:
                    self._log(f"{current_timestamp}: Cannot open position. Max size base is {max_size_base} or less.", level=2)
                    continue

                target_quantity_base = max_size_base * position_size_pct_strat
                target_quantity_base = self._apply_exchange_filters(target_quantity_base, entry_price_estimate)

                if target_quantity_base > 0 :
                    # Store theoretical params for OOS log BEFORE execution attempt
                    if self.is_oos_simulation:
                        self.current_entry_order_params = entry_order_params_strat or \
                            strategy_instance._build_entry_params_formatted( # Call directly if not returned
                                side="BUY" if target_direction == 1 else "SELL",
                                quantity=target_quantity_base, # Theoretical before final filter in exec
                                order_type=order_type_strat,
                                price=limit_price_strat,
                                # TODO: Add client order IDs if strategy generates them
                            )
                        self.current_oco_params = oco_params_strat or \
                            strategy_instance._build_oco_params( # Call directly if not returned
                                side="BUY" if target_direction == 1 else "SELL", # Entry side
                                quantity=target_quantity_base,
                                entry_price=entry_price_estimate, # Theoretical entry
                                sl_price=sl_price_strat,
                                tp_price=tp_price_strat,
                                # TODO: Add client order IDs
                            ) if sl_price_strat and tp_price_strat else None


                    sim_order_type = 'LIMIT_ENTRY' if order_type_strat == 'LIMIT' else 'MARKET'
                    exec_p, exec_q, fee_q = self._simulate_order_execution(
                        timestamp=current_timestamp,
                        current_row=current_row,
                        order_type=sim_order_type,
                        trade_direction=target_direction,
                        limit_price=limit_price_strat if order_type_strat == 'LIMIT' else None,
                        target_quantity_base=target_quantity_base,
                        is_entry=True
                    )

                    if exec_p is not None: # Entry was successful
                        self.current_sl_price = sl_price_strat
                        self.current_tp_price = tp_price_strat
                        self.current_trade_entry_fee = fee_q # Store entry fee for this trade
                        # self._log(f"Position opened: {self.trade_direction} at {self.entry_price}, SL: {self.current_sl_price}, TP: {self.current_tp_price}", level=2)
                    # else: # Entry failed (e.g. limit not hit, or filters)
                        # self._log(f"Entry FAILED at {current_timestamp} for direction {target_direction}", level=2)
                        # if self.is_oos_simulation: # Clear potentially stored params if entry failed
                        #     self.current_trade_cycle_id = None
                        #     self.current_entry_order_params = None
                        #     self.current_oco_params = None
            
            # Update equity curve at each step (can be made less frequent, e.g., end of day)
            # self.equity_curve.append({'timestamp': current_timestamp, 'equity': self.current_equity, 'type': 'eod_step'})


        # Final equity update for the last day
        if self.df_ohlcv.index[-1].date() not in self.daily_equity:
            self.daily_equity[self.df_ohlcv.index[-1].date()] = self.current_equity
        self.equity_curve.append({'timestamp': self.df_ohlcv.index[-1], 'equity': self.current_equity, 'type': 'final'})

        return self.trades, pd.DataFrame(self.equity_curve), self.daily_equity, self.oos_detailed_trades_log

