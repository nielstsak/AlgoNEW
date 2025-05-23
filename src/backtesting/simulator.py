import json
import logging
import math
import os # Maintenu pour une éventuelle utilisation, bien que pathlib soit préféré
import time # Non utilisé directement, mais peut être utile pour des logs de performance futurs
import uuid # Non utilisé directement, mais pourrait l'être pour des ID de trade uniques si besoin
import datetime # Pour le typage de self.entry_timestamp et les logs
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from src.strategies.base import BaseStrategy
    from src.utils.exchange_utils import adjust_precision, get_precision_from_filter, get_filter_value
    from src.backtesting.performance import calculate_performance_metrics
except ImportError as e:
    # Fallback logger if main logging isn't set up yet
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger(__name__).critical(
        f"BacktestSimulator: Critical import error: {e}. Ensure PYTHONPATH is correct "
        "and all required modules (BaseStrategy, exchange_utils, performance) are available.",
        exc_info=True
    )
    # Define dummy/placeholder functions or classes if absolutely necessary for the rest of the module to load
    # This is generally not a good practice for runtime, but can help with static analysis or partial loading.
    class BaseStrategy: # type: ignore
        def __init__(self, params: dict): self.params = params if params is not None else {}
        def generate_signals(self, data: pd.DataFrame): raise NotImplementedError
        def get_signals(self) -> pd.DataFrame: return pd.DataFrame()
        def get_params(self) -> Dict: return getattr(self, 'params', {})
        def generate_order_request(self, data: pd.DataFrame, symbol: str, current_position: int, available_capital: float, symbol_info: dict) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]: raise NotImplementedError
        def _calculate_quantity(self, *args, **kwargs): return None # Placeholder

    def adjust_precision(value, precision, method=None): return value # type: ignore
    def get_precision_from_filter(info, ftype, key): return 8 # type: ignore
    def get_filter_value(info, ftype, key): return None # type: ignore
    def calculate_performance_metrics(*args, **kwargs) -> Dict[str, Any]: # type: ignore
        return {"error": "Performance calculation module missing.", "Sharpe Ratio": np.nan, "Total Net PnL USDC": 0.0}
    # It's better to let the ImportError propagate if these are truly critical.

logger = logging.getLogger(__name__)

DEFAULT_EARLY_STOP_EQUITY_THRESHOLD = 0.0 # Default to no early stop unless specified

class BacktestSimulator:
    def __init__(self,
                 historical_data_with_indicators: pd.DataFrame,
                 strategy_instance: BaseStrategy,
                 simulation_settings: Dict[str, Any],
                 output_dir: Optional[Union[str, Path]]):

        self.symbol = simulation_settings.get('symbol', 'UNKNOWN_SYMBOL') # Get symbol early for logging
        self.sim_log_prefix = f"[{self.__class__.__name__}][{self.symbol}]"
        logger.info(f"{self.sim_log_prefix} Initializing simulator...")

        if historical_data_with_indicators is None or historical_data_with_indicators.empty:
            raise ValueError(f"{self.sim_log_prefix} Historical data (with indicators) cannot be None or empty.")
        if not isinstance(strategy_instance, BaseStrategy):
            raise TypeError(f"{self.sim_log_prefix} strategy_instance must be an instance inheriting from BaseStrategy.")
        if simulation_settings is None:
            raise ValueError(f"{self.sim_log_prefix} simulation_settings cannot be None.")

        required_settings = ['initial_capital', 'transaction_fee_pct', 'slippage_pct', 'margin_leverage', 'symbol', 'symbol_info']
        missing_settings = [k for k in required_settings if k not in simulation_settings]
        if missing_settings:
            raise ValueError(f"{self.sim_log_prefix} Missing required simulation settings: {missing_settings}")
        if not isinstance(simulation_settings['initial_capital'], (int,float)) or simulation_settings['initial_capital'] <= 0:
            raise ValueError(f"{self.sim_log_prefix} initial_capital must be a positive number.")
        if not isinstance(simulation_settings['symbol_info'], dict):
            raise ValueError(f"{self.sim_log_prefix} symbol_info in simulation_settings must be a dictionary.")


        self.strategy = strategy_instance
        self.settings = simulation_settings
        self.strategy_name = strategy_instance.__class__.__name__ # type: ignore

        self.data_input = historical_data_with_indicators.copy()
        if not isinstance(self.data_input.index, pd.DatetimeIndex):
            if 'timestamp' in self.data_input.columns:
                self.data_input['timestamp'] = pd.to_datetime(self.data_input['timestamp'], errors='coerce', utc=True)
                self.data_input.dropna(subset=['timestamp'], inplace=True)
                self.data_input = self.data_input.set_index('timestamp')
                if self.data_input.index.isnull().any():
                    raise ValueError(f"{self.sim_log_prefix} Failed to convert 'timestamp' column to a valid DatetimeIndex or contains NaNs.")
            else:
                raise TypeError(f"{self.sim_log_prefix} Input historical_data_with_indicators must have a DatetimeIndex or a 'timestamp' column.")

        if self.data_input.index.tz is None:
            self.data_input.index = self.data_input.index.tz_localize('UTC')
        elif self.data_input.index.tz.utcoffset(self.data_input.index[0]) != datetime.timezone.utc.utcoffset(self.data_input.index[0]): # type: ignore
            self.data_input.index = self.data_input.index.tz_convert('UTC')

        if not self.data_input.index.is_monotonic_increasing:
            self.data_input.sort_index(inplace=True)
        if not self.data_input.index.is_unique:
            initial_len = len(self.data_input)
            self.data_input = self.data_input[~self.data_input.index.duplicated(keep='first')]
            logger.debug(f"{self.sim_log_prefix} Removed {initial_len - len(self.data_input)} duplicate index entries.")

        required_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlcv_cols = [c for c in required_ohlcv_cols if c not in self.data_input.columns]
        if missing_ohlcv_cols:
            raise ValueError(f"{self.sim_log_prefix} Input data missing OHLCV columns: {missing_ohlcv_cols}")

        nan_ohlcv_before = self.data_input[required_ohlcv_cols].isnull().sum().sum()
        if nan_ohlcv_before > 0:
            logger.debug(f"{self.sim_log_prefix} Found {nan_ohlcv_before} NaNs in OHLCV. Applying ffill & bfill.")
            self.data_input[required_ohlcv_cols] = self.data_input[required_ohlcv_cols].ffill().bfill()
            if self.data_input[required_ohlcv_cols].isnull().sum().sum() > 0:
                logger.debug(f"{self.sim_log_prefix} NaNs still present after ffill/bfill. Dropping rows with any OHLCV NaN.")
                self.data_input.dropna(subset=required_ohlcv_cols, inplace=True)
        
        if self.data_input.empty:
            raise ValueError(f"{self.sim_log_prefix} Input data became empty after NaN handling for OHLCV.")
        
        self.output_dir: Optional[Path] = None
        if output_dir:
            self.output_dir = Path(output_dir)
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"{self.sim_log_prefix} Output directory set to: {self.output_dir}")
            except OSError as e:
                logger.warning(f"{self.sim_log_prefix} Could not create output directory {self.output_dir}: {e}. No detailed artifacts will be saved.")
                self.output_dir = None

        self.initial_capital = float(self.settings['initial_capital'])
        self.equity = self.initial_capital
        self.equity_curve = pd.Series(dtype=float, index=self.data_input.index) # Initialize with correct index
        self.position: float = 0.0 # Explicitly float for quantity
        self.entry_price: float = 0.0
        self.entry_timestamp: Optional[pd.Timestamp] = None
        self.current_sl_price: Optional[float] = None
        self.current_tp_price: Optional[float] = None
        self.order_intent: Optional[Dict[str, Any]] = None # For logging the intended OTOCO order
        self.trades_log_full: List[Dict[str, Any]] = []
        self.trades_log_simple: List[Dict[str, Any]] = []
        self.order_id_counter = 0 # Simple counter for simulated order IDs

        self.early_stop_equity_threshold = float(self.settings.get('early_stop_equity_threshold', DEFAULT_EARLY_STOP_EQUITY_THRESHOLD))
        self.early_stopped_reason: Optional[str] = None
        
        logger.info(f"{self.sim_log_prefix} Simulator initialized. Capital: {self.initial_capital:.2f}, "
                    f"Early stop threshold: {self.early_stop_equity_threshold:.2f} (0 means no stop). "
                    f"Data range: {self.data_input.index.min()} to {self.data_input.index.max()} ({len(self.data_input)} rows).")


    def _calculate_quantity(self, entry_price: float) -> Optional[float]:
        """
        Determines the quantity to trade by calling the strategy's _calculate_quantity method.
        """
        if self.equity <= 0 or entry_price <= 0:
            logger.debug(f"{self.sim_log_prefix} _calculate_quantity: Equity ({self.equity:.2f}) or entry price ({entry_price:.4f}) invalid.")
            return None

        symbol_info_data = self.settings.get('symbol_info', {}) # Passed in simulation_settings
        qty_precision = get_precision_from_filter(symbol_info_data, 'LOT_SIZE', 'stepSize')
        # qty_precision = qty_precision if qty_precision is not None else 8 # Default precision if not found

        # The strategy instance has access to its own parameters (including capital_allocation_pct, margin_leverage)
        # through self.strategy.params
        qty = self.strategy._calculate_quantity(
            entry_price=entry_price,
            available_capital=self.equity, # Current equity is the available capital for the next trade
            qty_precision=qty_precision,    # Can be None, strategy's method should handle it
            symbol_info=symbol_info_data,
            symbol=self.symbol
        )
        logger.debug(f"{self.sim_log_prefix} _calculate_quantity: EntryPx={entry_price:.4f}, AvailCapital={self.equity:.2f}, "
                     f"QtyPrec={qty_precision} -> Calculated Qty={qty}")
        return qty


    def _apply_slippage(self, price: float, side: str) -> float:
        """Applies slippage to the given price based on the trade side."""
        slippage = float(self.settings.get('slippage_pct', 0.0))
        if side.upper() == 'BUY':
            slipped_price = price * (1 + slippage)
        elif side.upper() == 'SELL':
            slipped_price = price * (1 - slippage)
        else:
            slipped_price = price # No slippage if side is unclear
        return max(0.0, slipped_price) # Price cannot be negative


    def _calculate_commission(self, position_value: float) -> float:
        """Calculates commission based on position value and fee percentage."""
        fee = float(self.settings.get('transaction_fee_pct', 0.0))
        return abs(position_value) * fee

    def _generate_otoco_intent(self, timestamp: pd.Timestamp, entry_price: float, quantity: float,
                               side: str, sl_price: Optional[float], tp_price: Optional[float]) -> Dict[str, Any]:
        """Generates a dictionary representing the OTOCO order intent for logging."""
        return {
            "intent_timestamp_utc": timestamp.isoformat(),
            "entry_side": side.upper(),
            "entry_price_theoretical": entry_price,
            "quantity": quantity,
            "intended_sl": sl_price,
            "intended_tp": tp_price,
        }

    def run_simulation(self) -> Dict[str, Any]:
        """Runs the backtesting simulation loop."""
        logger.info(f"{self.sim_log_prefix} Starting simulation for strategy '{self.strategy_name}'.")
        
        try:
            self.strategy.generate_signals(self.data_input) # data_input already contains *_strat columns
            signals_df = self.strategy.get_signals()
            if signals_df.empty:
                logger.warning(f"{self.sim_log_prefix} No signals generated by the strategy. Simulation will run without trades.")
                # Initialize with empty signal columns if not present
                for col in ['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']:
                    if col not in self.data_input.columns:
                        self.data_input[col] = False if 'exit' in col or 'entry' in col else np.nan
                self.data_for_loop = self.data_input.copy()
            else:
                logger.debug(f"{self.sim_log_prefix} Signals generated. Entry_long: {signals_df['entry_long'].sum()}, Entry_short: {signals_df['entry_short'].sum()}.")
                # Ensure signals_df aligns with data_input index
                if not self.data_input.index.equals(signals_df.index):
                    logger.debug(f"{self.sim_log_prefix} Aligning signal index with data_input index.")
                    signals_df = signals_df.reindex(self.data_input.index)

                self.data_for_loop = pd.merge(self.data_input, signals_df, left_index=True, right_index=True, how='left')

        except Exception as e_signal_gen:
            logger.error(f"{self.sim_log_prefix} Error during strategy.generate_signals: {e_signal_gen}", exc_info=True)
            return self._finalize_simulation(success=False, error_message=f"generate_signals failed: {e_signal_gen}")

        # Ensure signal columns are correctly typed and NaNs handled
        signal_bool_cols = ['entry_long', 'exit_long', 'entry_short', 'exit_short']
        signal_float_cols = ['sl', 'tp']
        for col in signal_bool_cols:
            if col not in self.data_for_loop.columns: self.data_for_loop[col] = False
            self.data_for_loop[col] = self.data_for_loop[col].fillna(False).astype(bool)
        for col in signal_float_cols:
            if col not in self.data_for_loop.columns: self.data_for_loop[col] = np.nan
            self.data_for_loop[col] = pd.to_numeric(self.data_for_loop[col], errors='coerce')

        if self.data_for_loop.empty:
            logger.error(f"{self.sim_log_prefix} Data for loop is empty after merging signals.")
            return self._finalize_simulation(success=False, error_message="Empty data after signal join")

        self.equity_curve.iloc[0] = self.initial_capital
        last_equity_update_idx = 0
        
        logger.info(f"{self.sim_log_prefix} Starting simulation loop over {len(self.data_for_loop)} klines.")

        for i in range(len(self.data_for_loop)): # Iterate over all rows including the first
            current_timestamp = self.data_for_loop.index[i]
            current_row = self.data_for_loop.iloc[i]

            # Propagate equity if it's not the first bar and no trade happened
            if i > 0:
                self.equity_curve.iloc[i] = self.equity_curve.iloc[i-1]
            
            current_bar_open = current_row['open']
            current_bar_high = current_row['high']
            current_bar_low = current_row['low']
            # current_bar_close = current_row['close'] # For MOC exit if needed

            if pd.isna(current_bar_high) or pd.isna(current_bar_low) or pd.isna(current_bar_open):
                logger.debug(f"{self.sim_log_prefix} Skipping bar {current_timestamp} due to NaN in OHLC.")
                if i > 0 : self.equity_curve.iloc[i] = self.equity_curve.iloc[i-1]
                else: self.equity_curve.iloc[i] = self.initial_capital # Should not happen if data cleaned
                continue
            
            current_equity_before_event = self.equity # Equity at the start of this bar's processing

            # --- 1. Check for Exits (SL/TP/Strategy Signal) ---
            if self.position != 0:
                exit_reason: Optional[str] = None
                simulated_exit_price: Optional[float] = None # Price after slippage

                # Check SL first
                if self.current_sl_price is not None and pd.notna(self.current_sl_price):
                    if (self.position > 0 and current_bar_low <= self.current_sl_price) or \
                       (self.position < 0 and current_bar_high >= self.current_sl_price):
                        exit_reason = 'SL_HIT'
                        # Assume SL executes at sl_price or worse (open if bar gaps through SL)
                        # For simplicity, assume execution at sl_price for now
                        simulated_exit_price = self._apply_slippage(self.current_sl_price, 'SELL' if self.position > 0 else 'BUY')
                        logger.debug(f"{self.sim_log_prefix} {current_timestamp}: SL hit for {self.position_side_log()} at {self.current_sl_price:.4f} (slipped: {simulated_exit_price:.4f}). Low: {current_bar_low}, High: {current_bar_high}")
                
                # Check TP if not SL hit
                if exit_reason is None and self.current_tp_price is not None and pd.notna(self.current_tp_price):
                    if (self.position > 0 and current_bar_high >= self.current_tp_price) or \
                       (self.position < 0 and current_bar_low <= self.current_tp_price):
                        exit_reason = 'TP_HIT'
                        simulated_exit_price = self._apply_slippage(self.current_tp_price, 'SELL' if self.position > 0 else 'BUY')
                        logger.debug(f"{self.sim_log_prefix} {current_timestamp}: TP hit for {self.position_side_log()} at {self.current_tp_price:.4f} (slipped: {simulated_exit_price:.4f}). Low: {current_bar_low}, High: {current_bar_high}")

                # Check Strategy Exit Signal (if still no SL/TP hit)
                # Assuming strategy exit signals are acted upon at the open of the *next* bar (i.e., current bar's open if signal was on prev close)
                if exit_reason is None:
                    if (self.position > 0 and current_row.get('exit_long', False)) or \
                       (self.position < 0 and current_row.get('exit_short', False)):
                        exit_reason = 'EXIT_SIGNAL'
                        # Assume exit at current bar's open price for strategy-based exits
                        simulated_exit_price = self._apply_slippage(current_bar_open, 'SELL' if self.position > 0 else 'BUY')
                        logger.debug(f"{self.sim_log_prefix} {current_timestamp}: Strategy exit signal for {self.position_side_log()} at open {current_bar_open:.4f} (slipped: {simulated_exit_price:.4f}).")
                
                if exit_reason and simulated_exit_price is not None:
                    position_value_at_exit = abs(self.position) * simulated_exit_price
                    pnl_gross = (simulated_exit_price - self.entry_price) * self.position
                    commission_exit = self._calculate_commission(position_value_at_exit)
                    pnl_net = pnl_gross - commission_exit

                    self.equity += pnl_net
                    self.equity_curve.iloc[i] = self.equity
                    last_equity_update_idx = i
                    
                    logger.info(f"{self.sim_log_prefix} {current_timestamp}: EXIT {exit_reason} {self.position_side_log()} "
                                f"Qty: {abs(self.position):.4f} at {simulated_exit_price:.4f} (Entry: {self.entry_price:.4f}). "
                                f"PnL Net: {pnl_net:.2f}. Equity: {self.equity:.2f}")

                    pnl_percent = (pnl_net / current_equity_before_event) * 100 if current_equity_before_event > 1e-9 else 0.0
                    
                    trade_closure_details = {
                        "entry_timestamp_iso": self.entry_timestamp.isoformat() if self.entry_timestamp else None,
                        "exit_timestamp_iso": current_timestamp.isoformat(),
                        "entry_price": self.entry_price, "exit_price": simulated_exit_price,
                        "pnl_usd_net": pnl_net, "pnl_usd_gross": pnl_gross, "commission_exit_usd": commission_exit,
                        "pnl_percent_net": pnl_percent, "exit_reason": exit_reason,
                        "trade_outcome": 'WIN' if pnl_net > 0 else ('LOSS' if pnl_net < 0 else 'BREAKEVEN')
                    }
                    full_trade_record = {"order_intent": self.order_intent, "closure_details": trade_closure_details, "final_equity": self.equity}
                    self.trades_log_full.append(full_trade_record)
                    
                    simple_trade_record = {
                        "entry_timestamp": self.entry_timestamp, "exit_timestamp": current_timestamp, "symbol": self.symbol,
                        "side": "LONG" if self.position > 0 else "SHORT", "entry_price": self.entry_price, "exit_price": simulated_exit_price,
                        "quantity": abs(self.position),
                        "initial_sl_price": self.order_intent.get('intended_sl') if self.order_intent else None,
                        "initial_tp_price": self.order_intent.get('intended_tp') if self.order_intent else None,
                        "exit_reason": exit_reason, "pnl_gross_usd": pnl_gross, "commission_usd": commission_exit, # commission_entry is logged at entry
                        "pnl_net_usd": pnl_net, "pnl_net_pct": pnl_percent, "cumulative_equity_usd": self.equity
                    }
                    self.trades_log_simple.append(simple_trade_record)

                    self.position = 0.0; self.entry_price = 0.0; self.entry_timestamp = None
                    self.current_sl_price = None; self.current_tp_price = None; self.order_intent = None
                    
                    if self.early_stop_equity_threshold > 0 and self.equity < self.early_stop_equity_threshold:
                        self.early_stopped_reason = "EQUITY_BELOW_THRESHOLD"
                        logger.warning(f"{self.sim_log_prefix} {current_timestamp}: EARLY STOP - Equity ({self.equity:.2f}) "
                                       f"< Threshold ({self.early_stop_equity_threshold:.2f}).")
                        self.equity_curve.iloc[i:] = self.equity # Propagate last equity to end
                        break # Exit simulation loop


            # --- 2. Check for Entries (if flat) ---
            if self.position == 0 and self.early_stopped_reason is None: # No new entries if already stopped
                entry_side: Optional[str] = None
                theoretical_entry_price: Optional[float] = None
                # Signals are based on previous bar's close, so entry is on current bar's open
                
                sl_price_signal = current_row.get('sl')
                tp_price_signal = current_row.get('tp')
                sl_price_signal = float(sl_price_signal) if pd.notna(sl_price_signal) else None
                tp_price_signal = float(tp_price_signal) if pd.notna(tp_price_signal) else None

                if current_row.get('entry_long', False):
                    entry_side = "LONG"
                    theoretical_entry_price = current_bar_open
                elif current_row.get('entry_short', False):
                    entry_side = "SHORT"
                    theoretical_entry_price = current_bar_open
                
                if entry_side and theoretical_entry_price is not None:
                    logger.debug(f"{self.sim_log_prefix} {current_timestamp}: Entry signal {entry_side} at open {theoretical_entry_price:.4f}. SL_sig: {sl_price_signal}, TP_sig: {tp_price_signal}")
                    
                    valid_sl_tp = True
                    if sl_price_signal is None or tp_price_signal is None:
                        valid_sl_tp = False
                        logger.debug(f"{self.sim_log_prefix} Signal SL/TP is None. No entry.")
                    elif entry_side == "LONG" and (sl_price_signal >= theoretical_entry_price or tp_price_signal <= theoretical_entry_price):
                        valid_sl_tp = False
                        logger.debug(f"{self.sim_log_prefix} Invalid SL/TP for LONG: SL {sl_price_signal} >= Entry {theoretical_entry_price} or TP {tp_price_signal} <= Entry. No entry.")
                    elif entry_side == "SHORT" and (sl_price_signal <= theoretical_entry_price or tp_price_signal >= theoretical_entry_price):
                        valid_sl_tp = False
                        logger.debug(f"{self.sim_log_prefix} Invalid SL/TP for SHORT: SL {sl_price_signal} <= Entry {theoretical_entry_price} or TP {tp_price_signal} >= Entry. No entry.")

                    if valid_sl_tp:
                        quantity = self._calculate_quantity(theoretical_entry_price)
                        if quantity is not None and quantity > 1e-9: # Check against very small float
                            self.current_sl_price = sl_price_signal
                            self.current_tp_price = tp_price_signal
                            self.order_intent = self._generate_otoco_intent(current_timestamp, theoretical_entry_price, quantity, entry_side, sl_price_signal, tp_price_signal)
                            
                            simulated_entry_price_after_slippage = self._apply_slippage(theoretical_entry_price, 'BUY' if entry_side == 'LONG' else 'SELL')
                            position_value = quantity * simulated_entry_price_after_slippage
                            commission_entry = self._calculate_commission(position_value)
                            
                            leverage = float(self.settings.get('margin_leverage', 1.0))
                            if leverage <= 0: leverage = 1.0 # Should be caught by config validation
                            required_margin = position_value / leverage

                            if current_equity_before_event - commission_entry >= required_margin : # Check against equity before this bar's events
                                self.equity = current_equity_before_event - commission_entry # Deduct commission
                                self.equity_curve.iloc[i] = self.equity
                                last_equity_update_idx = i

                                self.position = quantity if entry_side == 'LONG' else -quantity
                                self.entry_price = simulated_entry_price_after_slippage
                                self.entry_timestamp = current_timestamp
                                
                                logger.info(f"{self.sim_log_prefix} {current_timestamp}: ENTRY {entry_side} "
                                            f"Qty: {quantity:.4f} at {self.entry_price:.4f} (Théo: {theoretical_entry_price:.4f}). "
                                            f"SL: {self.current_sl_price:.4f}, TP: {self.current_tp_price:.4f}. Comm: {commission_entry:.2f}. Equity: {self.equity:.2f}")
                                
                                # Add entry commission to the simple trade log's "commission_usd" field for this trade later
                                # For now, simple log just stores exit commission. Full log will have both.

                                if self.early_stop_equity_threshold > 0 and self.equity < self.early_stop_equity_threshold:
                                    self.early_stopped_reason = "EQUITY_BELOW_THRESHOLD_POST_ENTRY"
                                    logger.warning(f"{self.sim_log_prefix} {current_timestamp}: EARLY STOP - Equity ({self.equity:.2f}) "
                                                   f"< Threshold ({self.early_stop_equity_threshold:.2f}) immediately after entry. Position will be closed at EOD.")
                                    self.equity_curve.iloc[i:] = self.equity # Propagate last equity
                                    break # Exit simulation loop
                            else:
                                logger.warning(f"{self.sim_log_prefix} {current_timestamp}: Insufficient margin for {entry_side} entry. "
                                               f"Required: ~{required_margin:.2f} + Comm: {commission_entry:.2f}. Available Equity: {current_equity_before_event:.2f}. No entry.")
                                self.order_intent = None; self.current_sl_price = None; self.current_tp_price = None
                        else:
                            logger.debug(f"{self.sim_log_prefix} {current_timestamp}: Calculated quantity is None or zero ({quantity}). No entry for {entry_side} signal.")
                            self.current_sl_price = None; self.current_tp_price = None; self.order_intent = None # Clear intent
            
            # Check equity at the end of each bar's processing
            if self.equity <= 0 and not self.early_stopped_reason:
                self.early_stopped_reason = "EQUITY_ZERO_OR_NEGATIVE"
                logger.warning(f"{self.sim_log_prefix} {current_timestamp}: EARLY STOP - Equity is zero or negative ({self.equity:.2f}).")
                self.equity_curve.iloc[i:] = self.equity
                break # Exit simulation loop
            
            # If no trade event modified equity on this bar, ensure equity curve is propagated correctly
            if last_equity_update_idx < i : # if equity_curve[i] was not updated by an event
                 self.equity_curve.iloc[i] = self.equity_curve.iloc[i-1] if i > 0 else self.initial_capital


        logger.info(f"{self.sim_log_prefix} Simulation loop finished." +
                    (f" Reason for early stop: {self.early_stopped_reason}" if self.early_stopped_reason else ""))
        return self._finalize_simulation()

    def _finalize_simulation(self, success: bool = True, error_message: Optional[str] = None) -> Dict[str, Any]:
        """Finalizes the simulation, closes open positions, calculates metrics, and saves artifacts."""
        log_final_prefix = f"{self.sim_log_prefix}[Finalize]"
        logger.info(f"{log_final_prefix} Finalizing simulation. Initial success flag: {success}" +
                    (f", Error msg: {error_message}" if error_message else "") +
                    (f", Early stopped: {self.early_stopped_reason}" if self.early_stopped_reason else ""))

        if self.position != 0 and not self.data_for_loop.empty:
            last_valid_idx = self.data_for_loop.index.get_indexer([self.data_for_loop.index[-1]])[0]
            last_timestamp = self.data_for_loop.index[last_valid_idx]
            # Use last available close price to close position
            last_close_price = self.data_for_loop['close'].iloc[last_valid_idx]
            
            if pd.notna(last_close_price):
                logger.info(f"{log_final_prefix} Closing open position ({self.position_side_log()} Qty: {abs(self.position):.4f}) "
                            f"at end of data (Timestamp: {last_timestamp}, Price: {last_close_price:.4f}).")
                
                exit_reason_final = self.early_stopped_reason if self.early_stopped_reason else 'END_OF_DATA'
                simulated_exit_price = self._apply_slippage(last_close_price, 'SELL' if self.position > 0 else 'BUY')
                position_value_at_exit = abs(self.position) * simulated_exit_price
                pnl_gross = (simulated_exit_price - self.entry_price) * self.position
                commission_exit = self._calculate_commission(position_value_at_exit)
                pnl_net = pnl_gross - commission_exit
                
                equity_before_final_exit = self.equity
                self.equity += pnl_net
                
                # Ensure equity curve is updated at the very last timestamp
                self.equity_curve.loc[last_timestamp] = self.equity
                
                logger.info(f"{log_final_prefix} FINAL EXIT ({exit_reason_final}): PnL Net: {pnl_net:.2f}, Final Equity: {self.equity:.2f}")
                
                pnl_percent = (pnl_net / equity_before_final_exit) * 100 if equity_before_final_exit > 1e-9 else 0.0
                trade_closure_details = {
                    "entry_timestamp_iso": self.entry_timestamp.isoformat() if self.entry_timestamp else None,
                    "exit_timestamp_iso": last_timestamp.isoformat(), "entry_price": self.entry_price,
                    "exit_price": simulated_exit_price, "pnl_usd_net": pnl_net, "pnl_usd_gross": pnl_gross,
                    "commission_exit_usd": commission_exit, "pnl_percent_net": pnl_percent,
                    "exit_reason": exit_reason_final, "trade_outcome": 'WIN' if pnl_net > 0 else ('LOSS' if pnl_net < 0 else 'BREAKEVEN')
                }
                full_trade_record = {"order_intent": self.order_intent, "closure_details": trade_closure_details, "final_equity": self.equity}
                self.trades_log_full.append(full_trade_record)
                simple_trade_record = {
                    "entry_timestamp": self.entry_timestamp, "exit_timestamp": last_timestamp, "symbol": self.symbol,
                    "side": "LONG" if self.position > 0 else "SHORT", "entry_price": self.entry_price, "exit_price": simulated_exit_price,
                    "quantity": abs(self.position),
                    "initial_sl_price": self.order_intent.get('intended_sl') if self.order_intent else None,
                    "initial_tp_price": self.order_intent.get('intended_tp') if self.order_intent else None,
                    "exit_reason": exit_reason_final, "pnl_gross_usd": pnl_gross, "commission_usd": commission_exit,
                    "pnl_net_usd": pnl_net, "pnl_net_pct": pnl_percent, "cumulative_equity_usd": self.equity
                }
                self.trades_log_simple.append(simple_trade_record)
            else:
                logger.warning(f"{log_final_prefix} Cannot close open position at EOD: last close price is NaN.")
            self.position = 0.0 # Position is now closed

        # Fill any remaining NaNs in equity curve (e.g., if simulation stopped early)
        if not self.equity_curve.empty:
            self.equity_curve = self.equity_curve.ffill()
            if self.equity_curve.iloc[0] is np.nan : self.equity_curve.iloc[0] = self.initial_capital
            self.equity_curve = self.equity_curve.fillna(method='ffill').fillna(self.initial_capital) # Final fallback
        else: # Create a minimal equity curve if it's somehow still empty
            idx = self.data_for_loop.index if hasattr(self, 'data_for_loop') and not self.data_for_loop.empty else pd.to_datetime(['now'], utc=True)
            self.equity_curve = pd.Series(self.initial_capital, index=idx)

        trades_df = pd.DataFrame(self.trades_log_simple)
        if not trades_df.empty:
            for col in ['entry_timestamp', 'exit_timestamp']:
                if col in trades_df.columns:
                    trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce', utc=True)
            numeric_cols_trades = ['entry_price', 'exit_price', 'quantity', 'initial_sl_price', 'initial_tp_price',
                                   'pnl_gross_usd', 'commission_usd', 'pnl_net_usd', 'pnl_net_pct', 'cumulative_equity_usd']
            for col in numeric_cols_trades:
                if col in trades_df.columns:
                    trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')
        
        equity_series_final = self.equity_curve.rename('equity_usd')
        if equity_series_final.index.name != 'timestamp': equity_series_final.index.name = 'timestamp'


        metrics: Dict[str, Any] = {}
        if self.early_stopped_reason:
            logger.warning(f"{log_final_prefix} Simulation was stopped early: {self.early_stopped_reason}. Metrics will reflect this.")
            metrics = {
                "Final Equity USDC": self.equity, "Total Net PnL USDC": self.equity - self.initial_capital,
                "Total Trades": len(trades_df), "Sharpe Ratio": np.nan, "Win Rate Pct": np.nan, "Profit Factor": np.nan,
                "Status": f"EARLY_STOPPED_{self.early_stopped_reason.upper()}"
            }
            try: # Attempt to calculate Max Drawdown even if early stopped
                 temp_metrics_mdd = calculate_performance_metrics(trades_df, equity_series_final, self.initial_capital)
                 metrics["Max Drawdown Pct"] = temp_metrics_mdd.get("Max Drawdown Pct", np.nan)
                 metrics["Max Drawdown USDC"] = temp_metrics_mdd.get("Max Drawdown USDC", np.nan)
            except: pass
            # Ensure main objectives are severely penalized for Optuna if early stopped
            if "Total Net PnL USDC" not in metrics or metrics["Total Net PnL USDC"] > -abs(self.initial_capital * 0.5) : # If PNL is not already terrible
                 metrics["Total Net PnL USDC"] = min(metrics.get("Total Net PnL USDC", 0), -abs(self.initial_capital * 5)) # Severe penalty
            if "Win Rate Pct" not in metrics: metrics["Win Rate Pct"] = 0.0
            if "Sharpe Ratio" not in metrics: metrics["Sharpe Ratio"] = -99.0 # Very bad Sharpe


        elif not trades_df.empty and not equity_series_final.empty and equity_series_final.notna().any():
            try:
                metrics = calculate_performance_metrics(trades_df.dropna(subset=['pnl_net_usd']), equity_series_final.dropna(), self.initial_capital)
            except Exception as e_metrics:
                logger.error(f"{log_final_prefix} Error calculating performance metrics: {e_metrics}", exc_info=True)
                metrics = {"error": f"Metric calculation failed: {e_metrics}", "Sharpe Ratio": np.nan, "Total Net PnL USDC": self.equity - self.initial_capital, "Win Rate Pct": np.nan, "Status": "METRICS_ERROR"}
        else:
            logger.info(f"{log_final_prefix} No trades executed or equity curve unusable. Using default metrics.")
            metrics = {
                "Final Equity USDC": self.equity, "Total Net PnL USDC": self.equity - self.initial_capital,
                "Total Trades": 0, "Sharpe Ratio": np.nan, "Max Drawdown Pct": 0.0, "Max Drawdown USDC": 0.0,
                "Win Rate Pct": np.nan, "Profit Factor": np.nan, "Status": "NO_TRADES"
            }
        
        if "Start Date" not in metrics and not equity_series_final.empty and equity_series_final.index.min() is not pd.NaT:
            metrics["Start Date"] = equity_series_final.index.min().isoformat() # type: ignore
        if "End Date" not in metrics and not equity_series_final.empty and equity_series_final.index.max() is not pd.NaT:
            metrics["End Date"] = equity_series_final.index.max().isoformat() # type: ignore
        if "Status" not in metrics and success: metrics["Status"] = "Completed"
        elif "Status" not in metrics and not success: metrics["Status"] = f"FAILED_SIMULATION: {error_message or 'Unknown reason'}"


        logger.info(f"{log_final_prefix} Final performance metrics: PnL Total={metrics.get('Total Net PnL USDC', 'N/A')}, "
                    f"Sharpe={metrics.get('Sharpe Ratio', 'N/A')}, WinRate={metrics.get('Win Rate Pct', 'N/A')}%, Status={metrics.get('Status', 'N/A')}")

        strat_params_serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.strategy.get_params().items()}
        sim_settings_serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.settings.items()}
        if 'symbol_info' in sim_settings_serializable and isinstance(sim_settings_serializable['symbol_info'], dict):
            sim_settings_serializable['symbol_info'] = {
                "symbol": sim_settings_serializable['symbol_info'].get('symbol'),
                "status": sim_settings_serializable['symbol_info'].get('status'),
                "baseAsset": sim_settings_serializable['symbol_info'].get('baseAsset'),
                "quoteAsset": sim_settings_serializable['symbol_info'].get('quoteAsset'),
                "filters_summary": f"{len(sim_settings_serializable['symbol_info'].get('filters',[]))} filters present"
            }


        summary_data_dict = {
            "simulation_settings": sim_settings_serializable,
            "strategy_params": strat_params_serializable,
            "metrics": metrics
        }
        if not success and error_message: summary_data_dict["error_message"] = error_message
        if self.early_stopped_reason: summary_data_dict["early_stopped_reason"] = self.early_stopped_reason

        # Serialize summary_data_dict to handle non-native JSON types before saving
        try:
            # Custom default function for json.dumps
            def json_serializer(obj: Any) -> Any:
                if isinstance(obj, (np.integer, np.int64)): return int(obj)
                if isinstance(obj, (np.floating, np.float64)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (datetime.datetime, datetime.date, pd.Timestamp)): return obj.isoformat()
                if isinstance(obj, Path): return str(obj)
                if pd.isna(obj): return None # Convert pandas NaT/NaN to None
                if isinstance(obj, tuple) and all(isinstance(x, (type(None), int, float, str, bool, list, dict)) for x in obj) :
                     return list(obj) # Convert simple tuples to lists
                try: # Default attempt to serialize
                    return str(obj) # Fallback for other types
                except TypeError:
                    return f"Unserializable_Type_{type(obj).__name__}"


            # Need to dump and reload to apply the custom serializer properly if metrics contain NaNs for allow_nan=False
            temp_json_str = json.dumps(summary_data_dict, default=json_serializer)
            summary_data_for_json = json.loads(temp_json_str)
            # Replace remaining float NaNs/Infs with None for stricter JSON compatibility if needed
            if isinstance(summary_data_for_json.get("metrics"), dict):
                summary_data_for_json["metrics"] = {
                    k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                    for k, v in summary_data_for_json["metrics"].items()
                }

        except Exception as e_serial:
            logger.error(f"{log_final_prefix} Error serializing summary_data for JSON: {e_serial}")
            summary_data_for_json = {"error": "Failed to serialize summary_data", **summary_data_dict}


        if self.output_dir:
            logger.info(f"{log_final_prefix} Saving simulation artifacts to {self.output_dir}")
            try:
                parquet_engine = 'pyarrow' if 'pyarrow' in sys.modules else 'fastparquet'
                
                trade_json_path = self.output_dir / f"trade_log_full_{self.symbol}.json"
                with open(trade_json_path, 'w', encoding='utf-8') as f_json_full:
                    json.dump(self.trades_log_full, f_json_full, indent=2, default=json_serializer) # Use serializer
                logger.debug(f"{log_final_prefix} Full trade log saved: {trade_json_path}")

                trades_path = self.output_dir / f"trades_{self.symbol}.parquet"
                if not trades_df.empty:
                    trades_df.to_parquet(trades_path, index=False, engine=parquet_engine)
                    logger.debug(f"{log_final_prefix} Simple trades log saved as Parquet: {trades_path}")
                
                equity_path = self.output_dir / f"equity_curve_{self.symbol}.parquet"
                equity_series_final.to_frame().to_parquet(equity_path, index=True, engine=parquet_engine)
                logger.debug(f"{log_final_prefix} Equity curve saved as Parquet: {equity_path}")

                summary_json_path = self.output_dir / f"summary_{self.symbol}.json"
                with open(summary_json_path, 'w', encoding='utf-8') as f_summary:
                    json.dump(summary_data_for_json, f_summary, indent=4) # Use the pre-serialized version
                logger.info(f"{log_final_prefix} Simulation summary saved: {summary_json_path}")
            except Exception as e_save_artifacts:
                logger.error(f"{log_final_prefix} Error saving simulation artifacts: {e_save_artifacts}", exc_info=True)
                # summary_data_for_json["save_error"] = f"Failed to save artifacts: {e_save_artifacts}" # Already a dict

        return {
            "trades": trades_df if not trades_df.empty else pd.DataFrame(),
            "equity_curve": equity_series_final,
            "metrics": metrics,
            "summary_data": summary_data_for_json # Return the JSON-ready version
        }

    def position_side_log(self) -> str:
        """Returns a string representation of the current position side."""
        if self.position > 0: return "LONG"
        if self.position < 0: return "SHORT"
        return "FLAT"
