# src/backtesting/simulator.py
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timezone
import logging
from typing import Dict, Any, Optional, List, Tuple # Ajout de Tuple

from src.utils.slippage import SlippageSimulator
from src.utils.fees import FeeSimulator
# Retrait de load_exchange_config et get_pair_config_for_symbol car pair_config est maintenant passé
from src.strategies.base import BaseStrategy # Pour type hinting

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
                 pair_config: Dict[str, Any], # <<< MODIFIÉ: Accepte pair_config directement
                 trading_fee_bps: float,
                 slippage_config: dict,
                 is_futures: bool,
                 run_id: str, 
                 is_oos_simulation: bool = False, 
                 verbosity: int = 1):
        """
        Initializes the BacktestSimulator.

        Args:
            df_ohlcv (pd.DataFrame): DataFrame with OHLCV data, indexed by timestamp.
            strategy_instance (BaseStrategy): An instance of the trading strategy.
            initial_equity (float): Starting capital for the backtest.
            leverage (int): Leverage factor to be used.
            symbol (str): Trading symbol (e.g., "BTCUSDT").
            pair_config (Dict[str, Any]): Configuration pour la paire de trading (infos de l'exchange).
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
        self.run_id = run_id 
        self.is_oos_simulation = is_oos_simulation 
        self.verbosity = verbosity

        self.pair_config = pair_config # <<< UTILISÉ DIRECTEMENT
        if not self.pair_config:
            # Cette erreur ne devrait plus se produire si pair_config est correctement passé.
            raise ValueError(f"Pair configuration not found for symbol {self.symbol}")

        self.base_asset = self.pair_config.get('baseAsset', '')
        self.quote_asset = self.pair_config.get('quoteAsset', '')
        
        # Utiliser get_precision_from_filter de BaseStrategy ou utils pour la robustesse
        # Pour l'instant, on suppose que pair_config contient directement les précisions
        # ou qu'elles sont extraites correctement par BaseStrategy.set_backtest_context
        # et que la stratégie les utilise ou les expose.
        # Pour BacktestSimulator, il est préférable d'utiliser les valeurs de pair_config.
        from src.utils.exchange_utils import get_precision_from_filter # Import local pour cette logique
        
        self.price_precision = get_precision_from_filter(self.pair_config, 'PRICE_FILTER', 'tickSize')
        self.quantity_precision = get_precision_from_filter(self.pair_config, 'LOT_SIZE', 'stepSize')

        if self.price_precision is None:
            logger.warning(f"Price precision not found in pair_config for {self.symbol}. Defaulting to 8.")
            self.price_precision = 8 
        if self.quantity_precision is None:
            logger.warning(f"Quantity precision not found in pair_config for {self.symbol}. Defaulting to 8.")
            self.quantity_precision = 8

        min_notional_filter_details = next((f for f in self.pair_config.get('filters', []) if f.get('filterType') == 'MIN_NOTIONAL'), None)
        self.min_notional = float(min_notional_filter_details.get('minNotional', 0.0)) if min_notional_filter_details else 0.0
        
        lot_size_filter_details = next((f for f in self.pair_config.get('filters', []) if f.get('filterType') == 'LOT_SIZE'), None)
        self.min_quantity = float(lot_size_filter_details.get('minQty', 0.0)) if lot_size_filter_details else 0.0


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
        self.position_size_base = 0.0
        self.position_size_quote = 0.0
        self.entry_price = 0.0
        self.entry_time: Optional[pd.Timestamp] = None # Type hint pd.Timestamp
        self.exit_price = 0.0
        self.exit_time: Optional[pd.Timestamp] = None # Type hint pd.Timestamp
        self.current_sl_price: Optional[float] = None # Type hint float
        self.current_tp_price: Optional[float] = None # Type hint float
        self.trade_direction = 0

        self.trades: List[Dict[str, Any]] = [] # Type hint
        self.equity_curve: List[Dict[str, Any]] = [] # Type hint
        self.daily_equity: Dict[Any, float] = {} # Type hint

        self.active_loan: Optional[Dict[str, Any]] = None
        self.initial_margin_used = 0.0

        self.oos_detailed_trades_log: List[Dict[str, Any]] = [] # Type hint
        self.current_trade_cycle_id: Optional[str] = None
        self.current_entry_order_params: Optional[Dict[str, Any]] = None
        self.current_oco_params: Optional[Dict[str, Any]] = None
        self.current_trade_entry_fee: float = 0.0 # Ajouté pour stocker les frais d'entrée du trade courant

        self.strategy.set_backtest_context(
            pair_config=self.pair_config, # Passe le pair_config reçu
            is_futures=self.is_futures,
            leverage=self.leverage,
            initial_equity=self.initial_equity
        )

    def _log(self, message: str, level: int = 1):
        if self.verbosity >= level:
            logger.info(message)

    def _calculate_max_position_size_base(self, entry_price_estimate: float) -> float:
        if entry_price_estimate <= 0:
            return 0.0
        available_margin_capital = self.current_equity - self.initial_margin_used
        if available_margin_capital <= 0:
            return 0.0
        max_total_value_quote = available_margin_capital * self.leverage
        max_size_base = max_total_value_quote / entry_price_estimate
        
        # Utiliser self.quantity_precision qui est maintenant un int
        if self.quantity_precision is not None:
             max_size_base = np.floor(max_size_base * (10**self.quantity_precision)) / (10**self.quantity_precision)
        else: # Fallback si la précision n'a pas pu être déterminée
            max_size_base = round(max_size_base, 8) 
        return max_size_base

    def _apply_exchange_filters(self, quantity_base: float, price: float) -> float:
        if self.quantity_precision is not None:
            quantity_base = np.floor(quantity_base * (10**self.quantity_precision)) / (10**self.quantity_precision)
        else:
            quantity_base = round(quantity_base, 8)


        if quantity_base < self.min_quantity:
            if self.verbosity >=2:
                self._log(f"Quantity {quantity_base} less than min_quantity {self.min_quantity}. Setting to 0.", level=2)
            return 0.0

        notional_value = quantity_base * price
        if self.min_notional > 0 and notional_value < self.min_notional: # Vérifier si min_notional est applicable
            if self.verbosity >=2:
                self._log(f"Notional {notional_value} less than min_notional {self.min_notional}. Setting to 0.", level=2)
            return 0.0
        return quantity_base


    def _simulate_order_execution(self,
                                  timestamp: pd.Timestamp,
                                  current_row: pd.Series,
                                  order_type: str, 
                                  trade_direction: int, 
                                  limit_price: Optional[float] = None, # Type hint float
                                  stop_price: Optional[float] = None, # Type hint float
                                  target_quantity_base: Optional[float] = None, # Type hint float
                                  is_entry: bool = False
                                 ) -> Tuple[Optional[float], Optional[float], Optional[float]]: # Type hint pour le retour
        
        open_p, high_p, low_p, close_p = current_row['open'], current_row['high'], current_row['low'], current_row['close']
        base_exec_price: Optional[float] = None # Initialiser à None
        
        executed_quantity_base = target_quantity_base if is_entry and target_quantity_base is not None else self.position_size_base
        if executed_quantity_base is None : executed_quantity_base = 0.0 # S'assurer que ce n'est pas None

        if order_type == 'MARKET' or order_type == 'MARKET_EXIT':
            base_exec_price = open_p
        elif order_type == 'LIMIT_ENTRY' and limit_price is not None:
            if trade_direction == 1: 
                if low_p <= limit_price: 
                    base_exec_price = min(open_p, limit_price) 
            else: 
                if high_p >= limit_price:
                    base_exec_price = max(open_p, limit_price) 
        elif order_type == 'SL' and stop_price is not None: 
            if trade_direction == 1: 
                if low_p <= stop_price:
                    base_exec_price = min(open_p, stop_price) 
            else: 
                if high_p >= stop_price:
                    base_exec_price = max(open_p, stop_price) 
        elif order_type == 'TP' and limit_price is not None: 
            if trade_direction == 1: 
                if high_p >= limit_price:
                    base_exec_price = max(open_p, limit_price)
            else: 
                if low_p <= limit_price:
                    base_exec_price = min(open_p, limit_price)
        
        if base_exec_price is None: # L'ordre n'a pas été exécuté (ex: limite non atteinte)
            return None, None, None

        executed_price = self.slippage_simulator.simulate_slippage(
            price=base_exec_price,
            volume_at_price=current_row.get('volume', 0), 
            volatility=(high_p - low_p) / low_p if low_p > 0 else 0,
            direction=trade_direction if is_entry else -trade_direction 
        )
        if self.price_precision is not None:
            executed_price = round(executed_price, self.price_precision)
        else:
            executed_price = round(executed_price, 8)


        if is_entry:
            if target_quantity_base is None: # Devrait être fourni pour une entrée
                 logger.error(f"target_quantity_base est None pour une entrée à {timestamp}")
                 return None, None, None
            executed_quantity_base = self._apply_exchange_filters(target_quantity_base, executed_price)
            if executed_quantity_base == 0.0:
                self._log(f"Order at {timestamp} for {target_quantity_base} {self.base_asset} resulted in 0 quantity after filters.", level=2)
                return None, None, None 
        
        if executed_quantity_base == 0.0 and not is_entry : # Si on essaie de fermer une position de taille nulle
             return None, None, None


        trade_value_quote = executed_quantity_base * executed_price
        fee_paid_quote = self.fee_simulator.calculate_fee(trade_value_quote)

        if is_entry:
            self.position_open = True
            self.trade_direction = trade_direction
            self.entry_price = executed_price
            self.entry_time = timestamp
            self.position_size_base = executed_quantity_base
            self.position_size_quote = trade_value_quote 

            cost_of_asset_for_entry = executed_price * executed_quantity_base
            margin_locked_quote = cost_of_asset_for_entry / self.leverage
            self.initial_margin_used = margin_locked_quote
            self.current_trade_entry_fee = fee_paid_quote # Stocker les frais d'entrée

            if trade_direction == 1: 
                self.active_loan = {
                    "asset": self.quote_asset,
                    "amount": cost_of_asset_for_entry, # Emprunt en actif de cotation
                    "timestamp_utc": timestamp.tz_convert(timezone.utc).isoformat()
                }
                self._log(f"{timestamp}: LONG ENTRY @ {executed_price:.{self.price_precision or 8}f}, Qty: {executed_quantity_base:.{self.quantity_precision or 8}f}, Loan: {self.active_loan['amount']:.2f} {self.quote_asset}", level=1)
            else: 
                self.active_loan = {
                    "asset": self.base_asset, # Emprunt en actif de base pour shorter
                    "amount": executed_quantity_base,
                    "entry_price_for_loan_calc": executed_price, 
                    "timestamp_utc": timestamp.tz_convert(timezone.utc).isoformat()
                }
                self._log(f"{timestamp}: SHORT ENTRY @ {executed_price:.{self.price_precision or 8}f}, Qty: {executed_quantity_base:.{self.quantity_precision or 8}f}, Loan: {self.active_loan['amount']:.{self.quantity_precision or 8}f} {self.base_asset}", level=1)

            self.current_equity -= fee_paid_quote
            self.equity_curve.append({'timestamp': timestamp, 'equity': self.current_equity, 'type': 'entry_fee'})

            if self.is_oos_simulation:
                self.current_trade_cycle_id = str(uuid.uuid4())
        else: # C'est une sortie
            if not self.position_open: 
                logger.error(f"Attempted to exit position at {timestamp} but no position was open.")
                return None, None, None

            self.exit_price = executed_price
            self.exit_time = timestamp
            
            pnl_final_impact_on_equity = 0.0
            exit_reason_for_log = order_type

            if self.active_loan:
                if self.trade_direction == 1: 
                    proceeds_from_sale_quote = executed_price * self.position_size_base
                    repayment_amount_quote = self.active_loan['amount']
                    net_cash_flow_from_trade_op = proceeds_from_sale_quote - repayment_amount_quote
                    pnl_final_impact_on_equity = net_cash_flow_from_trade_op - fee_paid_quote
                    self._log(f"{timestamp}: LONG EXIT @ {executed_price:.{self.price_precision or 8}f}, Qty: {self.position_size_base:.{self.quantity_precision or 8}f}. Loan Repaid: {repayment_amount_quote:.2f} {self.quote_asset}. PnL (post-loan, pre-exit-fee): {net_cash_flow_from_trade_op:.2f}", level=1)
                else: 
                    cost_to_cover_short_quote = executed_price * self.position_size_base
                    initial_proceeds_quote = self.active_loan['entry_price_for_loan_calc'] * self.active_loan['amount']
                    net_cash_flow_from_trade_op = initial_proceeds_quote - cost_to_cover_short_quote
                    pnl_final_impact_on_equity = net_cash_flow_from_trade_op - fee_paid_quote
                    self._log(f"{timestamp}: SHORT EXIT @ {executed_price:.{self.price_precision or 8}f}, Qty: {self.position_size_base:.{self.quantity_precision or 8}f}. Base Asset Loan Repaid. PnL (post-loan, pre-exit-fee): {net_cash_flow_from_trade_op:.2f}", level=1)
            else: 
                logger.error(f"Exiting position at {timestamp} but no active loan found.")
                pnl_raw = (executed_price - self.entry_price) * self.position_size_base * self.trade_direction
                pnl_final_impact_on_equity = pnl_raw - fee_paid_quote

            self.current_equity += pnl_final_impact_on_equity
            self.equity_curve.append({'timestamp': timestamp, 'equity': self.current_equity, 'type': 'trade_pnl'})
            
            trade_info = {
                'entry_time': self.entry_time,
                'exit_time': self.exit_time,
                'entry_price': self.entry_price,
                'exit_price': executed_price,
                'size_base': self.position_size_base,
                'size_quote': self.position_size_quote, 
                'pnl_net_quote': pnl_final_impact_on_equity, 
                'entry_fee_quote': self.current_trade_entry_fee,
                'exit_fee_quote': fee_paid_quote,
                'total_fees_quote': self.current_trade_entry_fee + fee_paid_quote,
                'direction': 'long' if self.trade_direction == 1 else 'short',
                'leverage': self.leverage,
                'initial_margin_used_quote': self.initial_margin_used,
                'exit_reason': exit_reason_for_log
            }
            self.trades.append(trade_info)

            if self.is_oos_simulation and self.current_trade_cycle_id:
                self._record_oos_detailed_trade(trade_info)

            self._reset_position_state()

        return executed_price, executed_quantity_base, fee_paid_quote

    def _record_oos_detailed_trade(self, trade_info: dict):
        if not self.current_entry_order_params: 
            entry_params_fallback = {}
        else:
            entry_params_fallback = self.current_entry_order_params

        if not self.current_oco_params: 
             oco_params_fallback = {} 
        else:
            oco_params_fallback = self.current_oco_params
        
        is_short = trade_info['direction'] == 'short'
        entry_side_effect = "AUTO_BORROW_REPAY" if is_short else "MARGIN_BUY" # Simplifié
        
        intent_type = ""
        if oco_params_fallback and oco_params_fallback.get('sl_details') and oco_params_fallback.get('tp_details'):
            intent_type = f"OTOCO_ENTRY_{'SHORT' if is_short else 'LONG'}"
        elif entry_params_fallback.get("type"): # Utiliser 'type' de entry_params_fallback
            intent_type = f"{entry_params_fallback['type'].upper()}_ENTRY_{'SHORT' if is_short else 'LONG'}"
        else: 
            intent_type = f"UNKNOWN_ENTRY_{'SHORT' if is_short else 'LONG'}"

        log_entry = {
            "trade_cycle_id": self.current_trade_cycle_id,
            "entry_timestamp_simulated_utc": trade_info['entry_time'].tz_convert(timezone.utc).isoformat() if trade_info['entry_time'] else None,
            "order_intent_type": intent_type,
            "symbol": self.symbol,
            "isIsolated": str(self.strategy.account_type == "ISOLATED_MARGIN").upper() if hasattr(self.strategy, 'account_type') else "UNKNOWN",

            "sideEffectType_entry": entry_side_effect,
            "listClientOrderId_simulated": oco_params_fallback.get("listClientOrderId_simulated", f"sim_otoco_{self.current_trade_cycle_id[:8]}" if oco_params_fallback else None),
            
            "workingType_entry": entry_params_fallback.get("type", "MARKET"), 
            "workingSide_entry": "SELL" if is_short else "BUY",
            "workingClientOrderId_entry_simulated": entry_params_fallback.get("newClientOrderId", f"sim_entry_{self.current_trade_cycle_id[:8]}"),
            "workingPrice_entry_theoretical": str(entry_params_fallback.get("price")) if entry_params_fallback.get("price") is not None else None,
            "workingPrice_entry_executed_slipped": str(trade_info['entry_price']),
            "workingQuantity_requested": str(entry_params_fallback.get("quantity")),
            "workingQuantity_executed": str(trade_info['size_base']),
            
            "pendingSide_exit": "BUY" if is_short else "SELL",
            "pendingQuantity_exit": str(trade_info['size_base']),
            
            "sl_details": oco_params_fallback.get("sl_details", {}),
            "tp_details": oco_params_fallback.get("tp_details", {}),

            "exit_timestamp_simulated_utc": trade_info['exit_time'].tz_convert(timezone.utc).isoformat() if trade_info['exit_time'] else None,
            "exit_reason": trade_info['exit_reason'],
            "exit_price_executed_slipped": str(trade_info['exit_price']) if trade_info['exit_price'] else None,
            "pnl_net_usd": trade_info['pnl_net_quote'] 
        }
        self.oos_detailed_trades_log.append(log_entry)
        self.current_trade_cycle_id = None
        self.current_entry_order_params = None
        self.current_oco_params = None

    def _reset_position_state(self):
        self.position_open = False
        self.position_size_base = 0.0
        self.position_size_quote = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.exit_price = 0.0 
        self.exit_time = None 
        self.trade_direction = 0
        self.current_sl_price = None
        self.current_tp_price = None
        self.active_loan = None
        self.initial_margin_used = 0.0
        self.current_trade_entry_fee = 0.0 


    def run_simulation(self) -> tuple[list, pd.DataFrame, dict, list]:
        if self.df_ohlcv.empty:
            logger.warning("OHLCV data is empty. Skipping simulation.")
            return [], pd.DataFrame(self.equity_curve), self.daily_equity, self.oos_detailed_trades_log

        self.equity_curve.append({'timestamp': self.df_ohlcv.index[0], 'equity': self.initial_equity, 'type': 'initial'})
        self.current_trade_entry_fee = 0.0 

        for i in range(len(self.df_ohlcv)):
            current_timestamp = self.df_ohlcv.index[i]
            current_row = self.df_ohlcv.iloc[i]
            
            if current_timestamp.date() not in self.daily_equity or \
               (isinstance(self.df_ohlcv.index, pd.DatetimeIndex) and \
                current_timestamp == self.df_ohlcv[self.df_ohlcv.index.date == current_timestamp.date()].index[-1]):
                 self.daily_equity[current_timestamp.date()] = self.current_equity


            if self.position_open:
                exit_simulated = False
                if self.current_tp_price:
                    exec_p, _, _ = self._simulate_order_execution(current_timestamp, current_row, 'TP', self.trade_direction, limit_price=self.current_tp_price)
                    if exec_p is not None:
                        self._log(f"{current_timestamp}: TP HIT for {'LONG' if self.trade_direction==1 else 'SHORT'} at {exec_p:.{self.price_precision or 8}f}", level=1)
                        exit_simulated = True
                if not exit_simulated and self.current_sl_price:
                    exec_p, _, _ = self._simulate_order_execution(current_timestamp, current_row, 'SL', self.trade_direction, stop_price=self.current_sl_price)
                    if exec_p is not None:
                        self._log(f"{current_timestamp}: SL HIT for {'LONG' if self.trade_direction==1 else 'SHORT'} at {exec_p:.{self.price_precision or 8}f}", level=1)
                        exit_simulated = True
                if exit_simulated: continue 

            df_slice_for_signal = self.df_ohlcv.iloc[:i+1] 
            signal_result = self.strategy.get_signal(
                data_feed=df_slice_for_signal, 
                current_position_open=self.position_open,
                current_position_direction=self.trade_direction,
                current_entry_price=self.entry_price,
                current_equity=self.current_equity 
            )
            
            signal = signal_result.get("signal") 
            order_type_strat = signal_result.get("order_type", "MARKET") 
            limit_price_strat = signal_result.get("limit_price")
            sl_price_strat = signal_result.get("sl_price")
            tp_price_strat = signal_result.get("tp_price")
            position_size_pct_strat = signal_result.get("position_size_pct", 1.0)
            
            entry_order_params_strat = signal_result.get("entry_order_params_theoretical_for_oos_log")
            oco_params_strat = signal_result.get("oco_params_theoretical_for_oos_log")

            if self.position_open:
                if signal == 2 or \
                   (self.trade_direction == 1 and signal == -1) or \
                   (self.trade_direction == -1 and signal == 1): 
                    self._log(f"{current_timestamp}: EXIT SIGNAL received for {'LONG' if self.trade_direction==1 else 'SHORT'}", level=1)
                    exec_p, _, _ = self._simulate_order_execution(current_timestamp, current_row, 'MARKET_EXIT', self.trade_direction, is_entry=False)
                    if exec_p is not None: self._log(f"Exited at {exec_p:.{self.price_precision or 8}f}", level=2)
            
            if not self.position_open and (signal == 1 or signal == -1):
                target_direction = signal 
                entry_price_estimate = current_row['open'] 
                if order_type_strat == 'LIMIT' and limit_price_strat is not None:
                    entry_price_estimate = limit_price_strat
                
                max_size_base = self._calculate_max_position_size_base(entry_price_estimate)
                if max_size_base <= 0:
                    self._log(f"{current_timestamp}: Cannot open position. Max size base is {max_size_base} or less.", level=2)
                    continue

                target_quantity_base = max_size_base * (position_size_pct_strat if position_size_pct_strat is not None else 1.0)
                # Note: _apply_exchange_filters est appelé DANS _simulate_order_execution pour les entrées

                if target_quantity_base > 0 :
                    if self.is_oos_simulation: # Stocker les params théoriques AVANT exécution
                        self.current_entry_order_params = entry_order_params_strat
                        self.current_oco_params = oco_params_strat

                    sim_order_type = 'LIMIT_ENTRY' if order_type_strat == 'LIMIT' else 'MARKET'
                    exec_p, exec_q, fee_q = self._simulate_order_execution(
                        timestamp=current_timestamp, current_row=current_row,
                        order_type=sim_order_type, trade_direction=target_direction,
                        limit_price=limit_price_strat if order_type_strat == 'LIMIT' else None,
                        target_quantity_base=target_quantity_base, is_entry=True
                    )
                    if exec_p is not None and exec_q is not None and fee_q is not None: 
                        self.current_sl_price = sl_price_strat
                        self.current_tp_price = tp_price_strat
                        # self.current_trade_entry_fee est déjà mis à jour dans _simulate_order_execution
                    elif self.is_oos_simulation: # Si l'entrée a échoué, effacer les params OOS
                        self.current_trade_cycle_id = None
                        self.current_entry_order_params = None
                        self.current_oco_params = None

        if not self.df_ohlcv.empty and self.df_ohlcv.index[-1].date() not in self.daily_equity:
            self.daily_equity[self.df_ohlcv.index[-1].date()] = self.current_equity
        if not self.df_ohlcv.empty:
            self.equity_curve.append({'timestamp': self.df_ohlcv.index[-1], 'equity': self.current_equity, 'type': 'final'})

        return self.trades, pd.DataFrame(self.equity_curve), self.daily_equity, self.oos_detailed_trades_log

