# src/backtesting/core_simulator.py
"""
Ce module définit BacktestRunner, le moteur principal de simulation de backtest.
Il prend en entrée des données OHLCV avec tous les indicateurs nécessaires déjà
calculés et une instance de stratégie configurée, puis simule l'exécution des trades.
"""
import pandas as pd
import numpy as np
import uuid # Pour les identifiants de cycle de trade uniques pour le log OOS
from datetime import timezone # Pour s'assurer que les timestamps sont UTC
import logging
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.base import BaseStrategy
    from src.config.definitions import SlippageConfig # Pour le typage de slippage_config_dict

# Utilisation des simulateurs de slippage et de frais des modules existants.
try:
    from src.utils.simulation_elements.slippage_simulator import SlippageSimulator
    from src.utils.simulation_elements.fees_simulator import FeeSimulator
    from src.utils.exchange_utils import get_precision_from_filter, get_filter_value, adjust_precision, adjust_quantity_to_step_size
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"BacktestRunner: Erreur d'importation critique pour Slippage/FeeSimulator ou exchange_utils: {e}. "
        "Assurez-vous que ces modules existent dans src.utils.* et sont corrects."
    )
    class SlippageSimulator: # type: ignore
        def __init__(self, **kwargs): pass
        def simulate_slippage(self, price: float, direction: int, **kwargs) -> float: return price
    class FeeSimulator: # type: ignore
        def __init__(self, **kwargs): pass
        def calculate_fee(self, trade_value_quote: float) -> float: return 0.0
    def get_precision_from_filter(*args) -> Optional[int]: return 8 # type: ignore
    def get_filter_value(*args) -> Optional[float]: return 0.0 # type: ignore
    def adjust_precision(value: Optional[Union[float, str]], precision: Optional[int], rounding_mode_str: str = "ROUND_HALF_UP", tick_size: Optional[Union[float, str]] = None) -> Optional[float]: # Correction des types et ordre
        if value is None: return None
        return round(float(str(value)), precision or 8) 
    def adjust_quantity_to_step_size(quantity: float, symbol_info: Dict[str, Any], rounding_mode_str: str = "ROUND_FLOOR") -> float: return round(quantity, 8) 

    raise 

logger = logging.getLogger(__name__)

class BacktestRunner:
    """
    Simule des stratégies de trading sur des données historiques.
    Gère l'exécution des ordres, la gestion de position, le calcul du PnL,
    et le suivi des métriques de performance.
    """
    def __init__(self,
                 df_ohlcv_with_indicators: pd.DataFrame,
                 strategy_instance: 'BaseStrategy',
                 initial_equity: float,
                 leverage: int,
                 symbol: str,
                 pair_config: Dict[str, Any], 
                 trading_fee_bps: float,
                 slippage_config_dict: Dict[str, Any], 
                 is_futures: bool, 
                 run_id: str, 
                 is_oos_simulation: bool = False, 
                 verbosity: int = 1):
        
        self.df_ohlcv = df_ohlcv_with_indicators.copy()
        self.strategy = strategy_instance
        self.initial_equity = float(initial_equity)
        self.current_equity = float(initial_equity)
        self.leverage = int(leverage)
        self.symbol = symbol.upper()
        self.pair_config = pair_config
        self.is_futures = is_futures
        self.run_id = run_id 
        self.is_oos_simulation = is_oos_simulation
        self.verbosity = verbosity

        self.log_prefix = f"[BacktestRunner][{self.symbol}][Run:{self.run_id}]"
        self._log(f"Initialisation. Equity: {self.initial_equity:.2f}, Levier: {self.leverage}x, "
                  f"Futures: {self.is_futures}, OOS: {self.is_oos_simulation}, Verbosity: {self.verbosity}", level=1)
        self._log(f"Données OHLCV+Indicateurs reçues shape: {self.df_ohlcv.shape}", level=2)

        if not isinstance(self.df_ohlcv.index, pd.DatetimeIndex):
            msg = "L'index de df_ohlcv_with_indicators doit être un DatetimeIndex."
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        if self.df_ohlcv.index.tz is None:
            logger.warning(f"{self.log_prefix} L'index de df_ohlcv n'a pas de timezone. Conversion en UTC.")
            self.df_ohlcv.index = self.df_ohlcv.index.tz_localize('UTC') # type: ignore
        elif self.df_ohlcv.index.tz.utcoffset(self.df_ohlcv.index[0] if not self.df_ohlcv.empty else None) != timezone.utc.utcoffset(None): # type: ignore
             logger.warning(f"{self.log_prefix} L'index de df_ohlcv n'est pas en UTC (actuel: {self.df_ohlcv.index.tz}). Conversion en UTC.")
             self.df_ohlcv.index = self.df_ohlcv.index.tz_convert('UTC') # type: ignore


        self.slippage_simulator = SlippageSimulator(**slippage_config_dict)
        self.fee_simulator = FeeSimulator(fee_bps=trading_fee_bps)
        self._log(f"Slippage config: {slippage_config_dict}, Fee BPS: {trading_fee_bps}", level=2)

        self.base_asset: str = self.pair_config.get('baseAsset', '')
        self.quote_asset: str = self.pair_config.get('quoteAsset', '')
        self.price_precision: Optional[int] = get_precision_from_filter(self.pair_config, 'PRICE_FILTER', 'tickSize')
        self.quantity_precision: Optional[int] = get_precision_from_filter(self.pair_config, 'LOT_SIZE', 'stepSize')
        self.min_notional_filter: float = get_filter_value(self.pair_config, 'MIN_NOTIONAL', 'minNotional') or 0.0
        self.min_quantity_filter: float = get_filter_value(self.pair_config, 'LOT_SIZE', 'minQty') or 0.0
        
        self.price_tick_size: Optional[float] = get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize')
        self.quantity_step_size: Optional[float] = get_filter_value(self.pair_config, 'LOT_SIZE', 'stepSize')


        if self.price_precision is None: logger.warning(f"{self.log_prefix} Précision de prix (tickSize) non trouvée, défaut à 8.")
        if self.quantity_precision is None: logger.warning(f"{self.log_prefix} Précision de quantité (stepSize) non trouvée, défaut à 8.")
        self._log(f"Pair info: Base='{self.base_asset}', Quote='{self.quote_asset}', "
                  f"PricePrec={self.price_precision}, PriceTick={self.price_tick_size}, "
                  f"QtyPrec={self.quantity_precision}, QtyStep={self.quantity_step_size}, "
                  f"MinNotional={self.min_notional_filter}, MinQty={self.min_quantity_filter}", level=2)

        self._reset_position_state() 
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        if not self.df_ohlcv.empty:
             self.equity_curve.append({'timestamp': self.df_ohlcv.index.min(), 'equity': self.initial_equity, 'type': 'initial'})
        else: 
             self.equity_curve.append({'timestamp': pd.Timestamp.now(tz='UTC'), 'equity': self.initial_equity, 'type': 'initial'})

        self.daily_equity: Dict[Any, float] = {}
        self.oos_detailed_trades_log: List[Dict[str, Any]] = []
        
        self._log(f"BacktestRunner initialisé pour la stratégie '{self.strategy.strategy_name}'.", level=1)

    def _log(self, message: str, level: int = 1, is_error: bool = False) -> None:
        """Logue un message si le niveau de verbosité est suffisant."""
        if self.verbosity >= level:
            if is_error:
                logger.error(f"{self.log_prefix} {message}")
            else:
                logger.info(f"{self.log_prefix} {message}")

    def _reset_position_state(self) -> None:
        """Réinitialise l'état de la position après une clôture ou à l'initialisation."""
        self.position_open = False
        self.position_size_base: float = 0.0
        self.position_size_quote: float = 0.0
        self.entry_price: float = 0.0
        self.entry_time: Optional[pd.Timestamp] = None
        self.current_sl_price: Optional[float] = None
        self.current_tp_price: Optional[float] = None
        self.trade_direction: int = 0
        self.current_trade_entry_fee: float = 0.0
        
        self.current_trade_cycle_id: Optional[str] = None
        self.current_entry_order_params_theoretical: Optional[Dict[str, Any]] = None
        self.current_oco_params_theoretical: Optional[Dict[str, Any]] = None
        self._log("État de la position réinitialisé.", level=2)

    def _calculate_max_position_size_base(self, entry_price_estimate: float, capital_for_trade: float) -> float:
        """
        Calcule la taille maximale de position en actif de base.
        """
        if entry_price_estimate <= 1e-9:
            self._log(f"Prix d'entrée estimé ({entry_price_estimate}) trop bas. Max size = 0.", level=2)
            return 0.0
        
        max_notional_value_quote = capital_for_trade * self.leverage
        max_size_base_raw = max_notional_value_quote / entry_price_estimate
        
        self._log(f"CalcMaxPosSize: CapitalForTrade={capital_for_trade:.2f}, Lev={self.leverage}, "
                  f"PrixEst={entry_price_estimate:.{self.price_precision or 8}f}, MaxNotional={max_notional_value_quote:.2f}, "
                  f"MaxSizeBaseRaw={max_size_base_raw:.8f}", level=2)
        return max_size_base_raw

    def _apply_exchange_filters(self, quantity_base: float, price: float) -> float:
        """
        Ajuste la quantité et valide par rapport aux filtres.
        """
        qty_adjusted_to_step = adjust_quantity_to_step_size(
            quantity_base,
            self.pair_config, 
            rounding_mode_str="ROUND_FLOOR" 
        )

        if qty_adjusted_to_step <= 1e-9: 
             self._log(f"Quantité {quantity_base:.8f} est devenue {qty_adjusted_to_step:.{self.quantity_precision or 8}f} "
                       f"après ajustement au stepSize. Considérée comme nulle.", level=2)
             return 0.0

        if self.min_quantity_filter > 0 and qty_adjusted_to_step < self.min_quantity_filter:
            self._log(f"Filtre Échec: Quantité ajustée {qty_adjusted_to_step:.{self.quantity_precision or 8}f} < minQty requis {self.min_quantity_filter:.{self.quantity_precision or 8}f}.", level=2)
            return 0.0

        notional_value = qty_adjusted_to_step * price
        if self.min_notional_filter > 0 and notional_value < self.min_notional_filter:
            self._log(f"Filtre Échec: Valeur notionnelle {notional_value:.2f} (Qty: {qty_adjusted_to_step:.{self.quantity_precision or 8}f} @ Prix: {price:.{self.price_precision or 8}f}) "
                      f"< minNotional requis {self.min_notional_filter:.2f}.", level=2)
            return 0.0
        
        self._log(f"Filtres appliqués: Qty brute: {quantity_base:.8f} -> Qty finale: {qty_adjusted_to_step:.{self.quantity_precision or 8}f}. Notionnel: {notional_value:.2f}", level=2)
        return qty_adjusted_to_step

    def _simulate_order_execution(self,
                                  timestamp: pd.Timestamp,
                                  current_bar_ohlc: pd.Series,
                                  order_type_sim: str, 
                                  signal_direction: int, 
                                  limit_price_target: Optional[float] = None,
                                  stop_price_trigger: Optional[float] = None,
                                  target_quantity_base_for_entry: Optional[float] = None
                                 ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Simule l'exécution d'un ordre pour la barre actuelle.
        """
        exec_log_prefix = f"{self.log_prefix}[ExecSim][{timestamp.strftime('%Y-%m-%d %H:%M')}]"
        
        open_p = float(current_bar_ohlc['open'])
        high_p = float(current_bar_ohlc['high'])
        low_p = float(current_bar_ohlc['low'])
        
        base_execution_price: Optional[float] = None
        is_entry_order = "ENTRY" in order_type_sim

        if order_type_sim == 'MARKET_ENTRY' or order_type_sim == 'MARKET_EXIT':
            base_execution_price = open_p 
        elif order_type_sim == 'LIMIT_ENTRY' and limit_price_target is not None:
            if signal_direction == 1: 
                if low_p <= limit_price_target: 
                    base_execution_price = min(open_p, limit_price_target) 
            else: 
                if high_p >= limit_price_target: 
                    base_execution_price = max(open_p, limit_price_target) 
        elif order_type_sim == 'SL_EXIT' and stop_price_trigger is not None:
            if signal_direction == 1: 
                if low_p <= stop_price_trigger: 
                    base_execution_price = min(open_p, stop_price_trigger) 
            else: 
                if high_p >= stop_price_trigger: 
                    base_execution_price = max(open_p, stop_price_trigger) 
        elif order_type_sim == 'TP_EXIT' and limit_price_target is not None:
            if signal_direction == 1: 
                if high_p >= limit_price_target: 
                    base_execution_price = max(open_p, limit_price_target) 
            else: 
                if low_p <= limit_price_target: 
                    base_execution_price = min(open_p, limit_price_target) 
        
        if base_execution_price is None:
            self._log(f"{exec_log_prefix}[{order_type_sim}] Non exécutable. Lmt:{limit_price_target}, Stp:{stop_price_trigger}, Dir:{signal_direction}, OHLC:[{open_p:.4f},{high_p:.4f},{low_p:.4f}]", level=2)
            return None, None, None

        slippage_impact_direction = 0
        if (is_entry_order and signal_direction == 1) or (not is_entry_order and signal_direction == -1): 
            slippage_impact_direction = 1
        elif (is_entry_order and signal_direction == -1) or (not is_entry_order and signal_direction == 1): 
            slippage_impact_direction = -1
        
        slipped_price = base_execution_price
        if slippage_impact_direction != 0:
            slipped_price = self.slippage_simulator.simulate_slippage(
                price=base_execution_price,
                direction=slippage_impact_direction,
                order_book_depth_at_price=current_bar_ohlc.get('volume', 0), 
                market_volatility_pct=(high_p - low_p) / low_p if low_p > 1e-9 else 0.0 
            )
        
        final_executed_price = adjust_precision(
            value=slipped_price, 
            precision=self.price_precision, 
            tick_size=self.price_tick_size, 
            rounding_mode_str="ROUND_HALF_UP" 
        )

        if final_executed_price is None:
            logger.error(f"{exec_log_prefix}[{order_type_sim}] Erreur d'ajustement de précision pour prix slippé {slipped_price}.")
            return None, None, None

        executed_quantity_base: float
        if is_entry_order:
            if target_quantity_base_for_entry is None or target_quantity_base_for_entry <= 1e-9:
                self._log(f"{exec_log_prefix}[{order_type_sim}] Quantité cible d'entrée invalide ({target_quantity_base_for_entry}).", level=1, is_error=True)
                return None, None, None
            executed_quantity_base = self._apply_exchange_filters(target_quantity_base_for_entry, final_executed_price)
            if executed_quantity_base <= 1e-9:
                self._log(f"{exec_log_prefix}[{order_type_sim}] Quantité d'entrée {target_quantity_base_for_entry:.8f} nulle après filtres.", level=1)
                return None, None, None
        else: 
            if not self.position_open or self.position_size_base <= 1e-9:
                self._log(f"{exec_log_prefix}[{order_type_sim}] Tentative de sortie mais pas de position valide.", level=2)
                return None, None, None
            executed_quantity_base = self.position_size_base 

        trade_value_quote = executed_quantity_base * final_executed_price
        fee_paid_quote = self.fee_simulator.calculate_fee(trade_value_quote)

        self._log(f"{exec_log_prefix}[{order_type_sim}] Exécution: PxBase={base_execution_price:.{self.price_precision or 8}f}, "
                  f"PxSlip={slipped_price:.{self.price_precision or 8}f}, PxFinal={final_executed_price:.{self.price_precision or 8}f}, "
                  f"QtyBase={executed_quantity_base:.{self.quantity_precision or 8}f}, FraisQt={fee_paid_quote:.4f}", level=2)

        if is_entry_order:
            self.position_open = True
            self.trade_direction = signal_direction
            self.entry_price = final_executed_price
            self.entry_time = timestamp
            self.position_size_base = executed_quantity_base
            self.position_size_quote = trade_value_quote
            self.current_trade_entry_fee = fee_paid_quote
            self.current_equity -= fee_paid_quote
            self.equity_curve.append({'timestamp': timestamp, 'equity': self.current_equity, 'type': 'entry_fee'})
            self._log(f"ENTRÉE {'LONG' if self.trade_direction == 1 else 'SHORT'} @ {self.entry_price:.{self.price_precision or 8}f}, "
                      f"Qty: {self.position_size_base:.{self.quantity_precision or 8}f}. Frais: {fee_paid_quote:.4f}. Equity: {self.current_equity:.2f}", level=1)
            if self.is_oos_simulation:
                self.current_trade_cycle_id = str(uuid.uuid4())
        else: 
            exit_price = final_executed_price
            pnl_gross_quote = (exit_price - self.entry_price) * self.position_size_base \
                if self.trade_direction == 1 else (self.entry_price - exit_price) * self.position_size_base
            pnl_net_quote = pnl_gross_quote - self.current_trade_entry_fee - fee_paid_quote
            self.current_equity += pnl_net_quote
            self.equity_curve.append({'timestamp': timestamp, 'equity': self.current_equity, 'type': 'trade_pnl'})
            
            trade_details = {
                'entry_time': self.entry_time, 'exit_time': timestamp,
                'entry_price': self.entry_price, 'exit_price': exit_price,
                'size_base': self.position_size_base, 'size_quote_at_entry': self.position_size_quote,
                'pnl_gross_quote': pnl_gross_quote, 'pnl_net_quote': pnl_net_quote,
                'entry_fee_quote': self.current_trade_entry_fee, 'exit_fee_quote': fee_paid_quote,
                'total_fees_quote': self.current_trade_entry_fee + fee_paid_quote,
                'direction': 'long' if self.trade_direction == 1 else 'short',
                'leverage': self.leverage, 'exit_reason': order_type_sim
            }
            self.trades.append(trade_details)
            self._log(f"SORTIE {'LONG' if self.trade_direction == 1 else 'SHORT'} @ {exit_price:.{self.price_precision or 8}f} (Raison: {order_type_sim}). "
                      f"PnL Net: {pnl_net_quote:.2f}. Frais Sortie: {fee_paid_quote:.4f}. Equity: {self.current_equity:.2f}", level=1)
            if self.is_oos_simulation:
                self._record_oos_detailed_trade(trade_details)
            self._reset_position_state()

        return final_executed_price, executed_quantity_base, fee_paid_quote

    def _record_oos_detailed_trade(self, trade_info: Dict[str, Any]) -> None:
        """Enregistre un log détaillé du cycle de trade pour l'analyse OOS."""
        if not self.current_trade_cycle_id:
            self.current_trade_cycle_id = str(uuid.uuid4())
            self._log(f"ID de cycle OOS manquant, nouveau généré : {self.current_trade_cycle_id}", level=2)

        entry_params_theo = self.current_entry_order_params_theoretical or {}
        oco_params_theo = self.current_oco_params_theoretical or {}
        
        intent_type = "UNKNOWN_INTENT"
        if entry_params_theo.get("type"):
            intent_type = f"{entry_params_theo['type'].upper()}_ENTRY_{trade_info['direction'].upper()}"
            if oco_params_theo: intent_type = f"OTOCO_VIA_{intent_type}"

        log_entry = {
            "trade_cycle_id": self.current_trade_cycle_id, "run_id": self.run_id,
            "strategy_name": self.strategy.strategy_name, "pair_symbol": self.symbol,
            "is_oos_simulation": True,
            "entry_timestamp_simulated_utc": trade_info['entry_time'].isoformat() if pd.notna(trade_info.get('entry_time')) else None,
            "order_intent_type_simulated": intent_type,
            "entry_order_theoretical_params": entry_params_theo,
            "oco_order_theoretical_params": oco_params_theo,
            "entry_execution_price_slipped": trade_info['entry_price'],
            "entry_executed_quantity_base": trade_info['size_base'],
            "entry_fee_quote": trade_info['entry_fee_quote'],
            "exit_timestamp_simulated_utc": trade_info['exit_time'].isoformat() if pd.notna(trade_info.get('exit_time')) else None,
            "exit_reason_simulated": trade_info['exit_reason'],
            "exit_execution_price_slipped": trade_info['exit_price'],
            "exit_fee_quote": trade_info['exit_fee_quote'],
            "pnl_gross_quote_simulated": trade_info['pnl_gross_quote'],
            "pnl_net_quote_simulated": trade_info['pnl_net_quote'],
            "total_fees_quote_simulated": trade_info['total_fees_quote']
        }
        self.oos_detailed_trades_log.append(log_entry)
        self._log(f"Log OOS détaillé enregistré pour cycle ID {self.current_trade_cycle_id}", level=2)

    def run_simulation(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[Any, float], List[Dict[str, Any]]]:
        """Exécute la simulation de backtest barre par barre."""
        self._log(f"Démarrage de la simulation pour {len(self.df_ohlcv)} barres.", level=1)
        if self.df_ohlcv.empty:
            logger.warning(f"{self.log_prefix} Données OHLCV vides. Simulation sautée.")
            return self.trades, pd.DataFrame(self.equity_curve), self.daily_equity, self.oos_detailed_trades_log

        last_recorded_daily_equity_date: Optional[pd.Timestamp] = None

        for i in range(len(self.df_ohlcv)):
            current_timestamp: pd.Timestamp = self.df_ohlcv.index[i] # type: ignore
            current_row_data: pd.Series = self.df_ohlcv.iloc[i]

            current_date_normalized = current_timestamp.normalize()
            if last_recorded_daily_equity_date is None or current_date_normalized > last_recorded_daily_equity_date:
                self.daily_equity[current_date_normalized] = self.current_equity
                last_recorded_daily_equity_date = current_date_normalized
            elif i == len(self.df_ohlcv) - 1: 
                 self.daily_equity[current_date_normalized] = self.current_equity

            exit_executed_this_bar = False
            if self.position_open:
                if self.current_tp_price is not None:
                    exec_p_tp, _, _ = self._simulate_order_execution(
                        current_timestamp, current_row_data, 'TP_EXIT', self.trade_direction, limit_price_target=self.current_tp_price)
                    if exec_p_tp is not None: exit_executed_this_bar = True; self._log(f"TP ATTEINT @ {exec_p_tp:.{self.price_precision or 8}f}", level=1)
                
                if not exit_executed_this_bar and self.current_sl_price is not None:
                    exec_p_sl, _, _ = self._simulate_order_execution(
                        current_timestamp, current_row_data, 'SL_EXIT', self.trade_direction, stop_price_trigger=self.current_sl_price)
                    if exec_p_sl is not None: exit_executed_this_bar = True; self._log(f"SL ATTEINT @ {exec_p_sl:.{self.price_precision or 8}f}", level=1)
                
                if exit_executed_this_bar: continue

            df_feed_for_strategy = self.df_ohlcv.iloc[:i+1]
            
            signal_decision = self.strategy.get_signal(
                df_feed_for_strategy, self.position_open, self.trade_direction, self.entry_price, self.current_equity)

            sig_type = signal_decision.get("signal", 0)
            sig_order_type = signal_decision.get("order_type", "MARKET")
            sig_limit_px = signal_decision.get("limit_price")
            sig_sl_px = signal_decision.get("sl_price")
            sig_tp_px = signal_decision.get("tp_price")
            sig_pos_size_pct = signal_decision.get("position_size_pct", 1.0)

            if self.is_oos_simulation and sig_type in [1, -1] and not self.position_open:
                self.current_entry_order_params_theoretical = signal_decision.get("entry_order_params_theoretical_for_oos_log")
                self.current_oco_params_theoretical = signal_decision.get("oco_params_theoretical_for_oos_log")

            if self.position_open:
                if sig_type == 2 or (self.trade_direction == 1 and sig_type == -1) or (self.trade_direction == -1 and sig_type == 1):
                    self._log(f"SIGNAL DE SORTIE STRATÉGIE reçu.", level=1)
                    self._simulate_order_execution(current_timestamp, current_row_data, 'MARKET_EXIT', self.trade_direction)
            elif not self.position_open and sig_type in [1, -1]:
                self._log(f"SIGNAL D'ENTRÉE {'LONG' if sig_type == 1 else 'SHORT'} reçu.", level=1)
                
                entry_px_est = sig_limit_px if sig_order_type == 'LIMIT' and sig_limit_px is not None else float(current_row_data['open'])
                if pd.isna(entry_px_est) or entry_px_est <= 1e-9:
                    self._log(f"Prix d'estimation pour entrée NaN ou invalide ({entry_px_est}). Pas d'entrée.", level=1, is_error=True)
                    continue

                capital_to_risk_for_trade = self.current_equity * (sig_pos_size_pct if sig_pos_size_pct is not None else 1.0)
                max_qty_base = self._calculate_max_position_size_base(entry_px_est, capital_to_risk_for_trade)
                
                if max_qty_base <= 1e-9:
                    self._log(f"Quantité max possible {max_qty_base:.8f} trop petite. Pas d'entrée.", level=1)
                    continue
                
                target_qty_for_entry = max_qty_base 

                sim_entry_type = 'LIMIT_ENTRY' if sig_order_type == 'LIMIT' else 'MARKET_ENTRY'
                exec_px_entry, exec_qty_entry, _ = self._simulate_order_execution(
                    current_timestamp, current_row_data, sim_entry_type, sig_type,
                    limit_price_target=sig_limit_px if sig_order_type == 'LIMIT' else None,
                    target_quantity_base_for_entry=target_qty_for_entry
                )
                if exec_px_entry is not None and exec_qty_entry is not None and exec_qty_entry > 1e-9:
                    self.current_sl_price = sig_sl_px
                    self.current_tp_price = sig_tp_px
                    self._log(f"SL/TP mis à jour après entrée: SL={self.current_sl_price}, TP={self.current_tp_price}", level=2)
                else:
                    self._log(f"Ordre d'entrée non exécuté ou quantité nulle. Pas de position.", level=1)
                    if self.is_oos_simulation: self._reset_position_state() 

        if not self.df_ohlcv.empty:
            final_date_norm = self.df_ohlcv.index[-1].normalize() # type: ignore
            if final_date_norm not in self.daily_equity: self.daily_equity[final_date_norm] = self.current_equity
            self.equity_curve.append({'timestamp': self.df_ohlcv.index[-1], 'equity': self.current_equity, 'type': 'final'})

        self._log(f"Simulation terminée. Trades: {len(self.trades)}. Equity finale: {self.current_equity:.2f}", level=1)
        
        equity_curve_df_final = pd.DataFrame(self.equity_curve)
        if not equity_curve_df_final.empty and 'timestamp' in equity_curve_df_final.columns:
            equity_curve_df_final['timestamp'] = pd.to_datetime(equity_curve_df_final['timestamp'], errors='coerce', utc=True)

        return self.trades, equity_curve_df_final, self.daily_equity, self.oos_detailed_trades_log

