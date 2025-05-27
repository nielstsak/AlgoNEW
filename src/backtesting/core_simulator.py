# src/backtesting/core_simulator.py
"""
Ce module définit BacktestRunner, le moteur principal de simulation de backtest.
Il prend en entrée des données OHLCV avec tous les indicateurs nécessaires déjà
calculés et une instance de stratégie configurée, puis simule l'exécution des trades.
"""
import logging
import uuid # Pour les identifiants de cycle de trade uniques pour le log OOS
from datetime import timezone # Pour s'assurer que les timestamps sont UTC
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING, Union 

import pandas as pd # Assurer l'import
import numpy as np

if TYPE_CHECKING:
    from src.strategies.base import BaseStrategy
    from decimal import Decimal # Pour les types dans les placeholders

# Utilisation des simulateurs de slippage et de frais des modules existants.
try:
    from src.utils.simulation_elements.slippage_simulator import SlippageSimulator
    from src.utils.simulation_elements.fees_simulator import FeeSimulator
    from src.utils.exchange_utils import get_precision_from_filter, get_filter_value, adjust_precision, adjust_quantity_to_step_size
except ImportError as e:
    logging.basicConfig(level=logging.ERROR) # Fallback logging
    logging.getLogger(__name__).critical(
        f"BacktestRunner: Erreur d'importation critique pour Slippage/FeeSimulator ou exchange_utils: {e}. "
        "Assurez-vous que ces modules existent dans src.utils.* et sont corrects."
    )
    # Définition de placeholders pour permettre au reste du module de se charger en cas d'erreur d'import
    class SlippageSimulator: # type: ignore
        def __init__(self, **kwargs): pass
        def simulate_slippage(self, price: float, direction: int, **kwargs) -> float: return price
    class FeeSimulator: # type: ignore
        def __init__(self, **kwargs): pass
        def calculate_fee(self, trade_value_quote: float) -> float: return 0.0
    
    def get_precision_from_filter(symbol_info: Dict[str, Any], filter_type: str, key: str) -> Optional[int]: return 8
    def get_filter_value(symbol_info: Dict[str, Any], filter_type: str, key: str) -> Optional[float]: return 0.0
    def adjust_precision(value: Union[float, str, 'Decimal'], precision: Optional[int], tick_size: Optional[Union[float, str, 'Decimal']] = None, rounding_mode_str: str = "ROUND_HALF_UP") -> Optional[float]:
        if value is None: return None
        try: return round(float(str(value)), precision or 8)
        except: return None
    def adjust_quantity_to_step_size(quantity: Union[float, str, 'Decimal'], symbol_info: Dict[str, Any], qty_precision: Optional[int] = None, rounding_mode_str: str = "ROUND_FLOOR") -> Optional[float]:
        try: return round(float(str(quantity)), qty_precision or 8)
        except: return None
    
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
        # Sourcery: Remove unnecessary casts (initial_equity, leverage)
        self.initial_equity = initial_equity 
        self.current_equity = initial_equity
        self.leverage = leverage
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
        elif str(self.df_ohlcv.index.tz).upper() != 'UTC': 
             logger.warning(f"{self.log_prefix} L'index de df_ohlcv n'est pas en UTC (actuel: {self.df_ohlcv.index.tz}). Conversion en UTC.")
             self.df_ohlcv.index = self.df_ohlcv.index.tz_convert('UTC') # type: ignore

        self.slippage_simulator = SlippageSimulator(**slippage_config_dict)
        self.fee_simulator = FeeSimulator(fee_bps=trading_fee_bps)
        self._log(f"Slippage config: {slippage_config_dict}, Fee BPS: {trading_fee_bps}", level=2)

        self.base_asset: str = self.pair_config.get('baseAsset', '')
        self.quote_asset: str = self.pair_config.get('quoteAsset', '')
        self.price_precision: Optional[int] = get_precision_from_filter(self.pair_config, 'PRICE_FILTER', 'tickSize')
        self.quantity_precision: Optional[int] = get_precision_from_filter(self.pair_config, 'LOT_SIZE', 'stepSize')
        
        self.min_notional_filter: float = get_filter_value(self.pair_config, 'MIN_NOTIONAL', 'minNotional') or \
                                         get_filter_value(self.pair_config, 'NOTIONAL', 'minNotional') or 0.0
        self.min_quantity_filter: float = get_filter_value(self.pair_config, 'LOT_SIZE', 'minQty') or 0.0
        
        self.price_tick_size: Optional[float] = get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize')
        self.quantity_step_size: Optional[float] = get_filter_value(self.pair_config, 'LOT_SIZE', 'stepSize')

        if self.price_precision is None: logger.warning(f"{self.log_prefix} Précision de prix (dérivée de tickSize) non trouvée. Utilisation d'une valeur par défaut ou arrondi standard.")
        if self.quantity_precision is None: logger.warning(f"{self.log_prefix} Précision de quantité (dérivée de stepSize) non trouvée. Utilisation d'une valeur par défaut ou arrondi standard.")
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
        if self.verbosity >= level:
            if is_error:
                logger.error(f"{self.log_prefix} {message}")
            else: 
                if level <= 1: logger.info(f"{self.log_prefix} {message}")
                else: logger.debug(f"{self.log_prefix} {message}")

    def _reset_position_state(self) -> None:
        self.position_open: bool = False
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
        if entry_price_estimate <= 1e-9: 
            self._log(f"Prix d'entrée estimé ({entry_price_estimate}) trop bas ou nul. Taille max de position = 0.", level=2, is_error=True)
            return 0.0
        
        max_notional_value_quote = capital_for_trade * self.leverage
        max_size_base_raw = max_notional_value_quote / entry_price_estimate
        
        self._log(f"CalcMaxPosSize: CapitalPourTrade={capital_for_trade:.2f}, Levier={self.leverage}, "
                  f"PrixEstimé={entry_price_estimate:.{self.price_precision or 8}f}, MaxNotionnel={max_notional_value_quote:.2f}, "
                  f"TailleMaxBaseBrute={max_size_base_raw:.8f}", level=2)
        return max_size_base_raw

    def _apply_exchange_filters(self, quantity_base: float, price: float) -> float:
        qty_adjusted_to_step_val = adjust_quantity_to_step_size(
            quantity_base,
            self.pair_config, 
            qty_precision=self.quantity_precision,
            rounding_mode_str="ROUND_FLOOR" 
        )
        
        if qty_adjusted_to_step_val is None:
            self._log(f"Erreur lors de l'ajustement de la quantité {quantity_base:.8f} au stepSize. Considérée comme nulle.", level=2, is_error=True)
            return 0.0
            
        qty_adjusted_to_step = qty_adjusted_to_step_val

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
        
        self._log(f"Filtres appliqués avec succès: Qty brute: {quantity_base:.8f} -> Qty finale: {qty_adjusted_to_step:.{self.quantity_precision or 8}f}. Notionnel: {notional_value:.2f}", level=2)
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
        exec_log_prefix = f"{self.log_prefix}[ExecSim][{timestamp.strftime('%Y-%m-%d %H:%M')}]"
        
        open_p = current_bar_ohlc['open'] 
        high_p = current_bar_ohlc['high']
        low_p = current_bar_ohlc['low']
        
        base_execution_price: Optional[float] = None 
        is_entry_order = "ENTRY" in order_type_sim.upper() 

        # Sourcery: Hoist nested repeated code outside conditional statements
        # Sourcery: Merge else clause's nested if statement into elif
        # Sourcery: Swap positions of nested conditionals
        if order_type_sim in ('MARKET_ENTRY', 'MARKET_EXIT'):
            base_execution_price = open_p
        elif limit_price_target is not None and order_type_sim == 'LIMIT_ENTRY':
            if signal_direction == 1:  # Achat LIMIT
                if low_p <= limit_price_target:
                    base_execution_price = min(open_p, limit_price_target)
            elif high_p >= limit_price_target:  # Vente LIMIT (signal_direction == -1)
                base_execution_price = max(open_p, limit_price_target)
        elif limit_price_target is not None and order_type_sim == 'TP_EXIT':
            if self.trade_direction == 1 and high_p >= limit_price_target: # TP sur LONG
                base_execution_price = max(open_p, limit_price_target)
            elif self.trade_direction == -1 and low_p <= limit_price_target: # TP sur SHORT
                base_execution_price = min(open_p, limit_price_target)
        elif stop_price_trigger is not None and order_type_sim == 'SL_EXIT':
            if self.trade_direction == 1 and low_p <= stop_price_trigger: # SL sur LONG
                base_execution_price = min(open_p, stop_price_trigger)
            elif self.trade_direction == -1 and high_p >= stop_price_trigger: # SL sur SHORT
                base_execution_price = max(open_p, stop_price_trigger)
        
        if base_execution_price is None: 
            self._log(f"{exec_log_prefix}[{order_type_sim}] Non exécutable. LmtCible:{limit_price_target}, StpDéclench:{stop_price_trigger}, "
                      f"DirSignalEntrée:{signal_direction if is_entry_order else 'N/A'}, DirPosActuelle:{self.trade_direction if not is_entry_order else 'N/A'}, "
                      f"OHLC Barre:[O:{open_p:.4f},H:{high_p:.4f},L:{low_p:.4f}]", level=2)
            return None, None, None

        slippage_impact_direction = 0
        if is_entry_order: 
            slippage_impact_direction = 1 if signal_direction == 1 else -1 
        else: 
            slippage_impact_direction = -1 if self.trade_direction == 1 else 1
        
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
            logger.error(f"{exec_log_prefix}[{order_type_sim}] Erreur d'ajustement de précision pour prix slippé {slipped_price}. Prix de base: {base_execution_price}")
            return None, None, None

        executed_quantity_base: float
        if is_entry_order:
            if target_quantity_base_for_entry is None or target_quantity_base_for_entry <= 1e-9:
                self._log(f"{exec_log_prefix}[{order_type_sim}] Quantité cible d'entrée invalide ({target_quantity_base_for_entry}). Pas d'exécution.", level=1, is_error=True)
                return None, None, None
            executed_quantity_base = self._apply_exchange_filters(target_quantity_base_for_entry, final_executed_price)
            if executed_quantity_base <= 1e-9: 
                self._log(f"{exec_log_prefix}[{order_type_sim}] Quantité d'entrée {target_quantity_base_for_entry:.8f} est devenue nulle ou invalide après application des filtres. Pas d'exécution.", level=1)
                return None, None, None
        else: 
            if not self.position_open or self.position_size_base <= 1e-9:
                self._log(f"{exec_log_prefix}[{order_type_sim}] Tentative de sortie mais pas de position ouverte valide (Taille: {self.position_size_base}). Pas d'exécution.", level=2)
                return None, None, None
            executed_quantity_base = self.position_size_base 

        trade_value_quote = executed_quantity_base * final_executed_price
        fee_paid_quote = self.fee_simulator.calculate_fee(trade_value_quote)

        self._log(f"{exec_log_prefix}[{order_type_sim}] Exécution simulée: PxBase={base_execution_price:.{self.price_precision or 8}f}, "
                  f"PxSlippé={slipped_price:.{self.price_precision or 8}f}, PxFinalExé={final_executed_price:.{self.price_precision or 8}f}, "
                  f"QtyBaseExé={executed_quantity_base:.{self.quantity_precision or 8}f}, ValeurNotionnelleQt={trade_value_quote:.2f}, FraisPayésQt={fee_paid_quote:.4f}", level=2)

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
                      f"Qty: {self.position_size_base:.{self.quantity_precision or 8}f}. Frais: {fee_paid_quote:.4f}. Equity après frais: {self.current_equity:.2f}", level=1)
            
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
                      f"PnL Net: {pnl_net_quote:.2f}. Frais Sortie: {fee_paid_quote:.4f}. Equity après trade: {self.current_equity:.2f}", level=1)
            
            if self.is_oos_simulation:
                self._record_oos_detailed_trade(trade_details) 
            
            self._reset_position_state() 

        return final_executed_price, executed_quantity_base, fee_paid_quote

    def _record_oos_detailed_trade(self, trade_info: Dict[str, Any]) -> None:
        if not self.current_trade_cycle_id: 
            self.current_trade_cycle_id = str(uuid.uuid4())
            self._log(f"ID de cycle OOS manquant pour enregistrement détaillé, nouveau généré : {self.current_trade_cycle_id}", level=2)

        entry_params_theo = self.current_entry_order_params_theoretical or {}
        oco_params_theo = self.current_oco_params_theoretical or {}
        
        intent_type = "UNKNOWN_INTENT"
        if entry_params_theo.get("type"): 
            intent_type = f"{str(entry_params_theo['type']).upper()}_ENTRY_{trade_info['direction'].upper()}"
            if oco_params_theo: 
                intent_type = f"OTOCO_VIA_{intent_type}" 

        log_entry_oos = {
            "trade_cycle_id": self.current_trade_cycle_id,
            "run_id": self.run_id, 
            "strategy_name": self.strategy.strategy_name,
            "pair_symbol": self.symbol,
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
        self.oos_detailed_trades_log.append(log_entry_oos)
        self._log(f"Log OOS détaillé enregistré pour cycle ID {self.current_trade_cycle_id}", level=2)


    def run_simulation(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[Any, float], List[Dict[str, Any]]]:
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
                        current_timestamp, current_row_data, 'TP_EXIT', 
                        self.trade_direction, 
                        limit_price_target=self.current_tp_price
                    )
                    if exec_p_tp is not None: 
                        exit_executed_this_bar = True
                        self._log(f"TP ATTEINT & EXÉCUTÉ @ {exec_p_tp:.{self.price_precision or 8}f}", level=1)
                
                if not exit_executed_this_bar and self.current_sl_price is not None:
                    exec_p_sl, _, _ = self._simulate_order_execution(
                        current_timestamp, current_row_data, 'SL_EXIT', 
                        self.trade_direction, 
                        stop_price_trigger=self.current_sl_price
                    )
                    if exec_p_sl is not None: 
                        exit_executed_this_bar = True
                        self._log(f"SL ATTEINT & EXÉCUTÉ @ {exec_p_sl:.{self.price_precision or 8}f}", level=1)
                
                if exit_executed_this_bar: 
                    continue 

            df_feed_for_strategy = self.df_ohlcv.iloc[:i+1]
            
            signal_decision = self.strategy.get_signal(
                df_feed_for_strategy, 
                self.position_open, 
                self.trade_direction, 
                self.entry_price, 
                self.current_equity
            )

            sig_type = signal_decision.get("signal", 0)
            sig_order_type = signal_decision.get("order_type", "MARKET")
            sig_limit_px = signal_decision.get("limit_price")
            sig_sl_px = signal_decision.get("sl_price")
            sig_tp_px = signal_decision.get("tp_price")
            sig_pos_size_pct = signal_decision.get("position_size_pct", 1.0) 

            # Sourcery: Use set for in comparison
            if self.is_oos_simulation and sig_type in {1, -1} and not self.position_open:
                self.current_entry_order_params_theoretical = signal_decision.get("entry_order_params_theoretical_for_oos_log")
                self.current_oco_params_theoretical = signal_decision.get("oco_params_theoretical_for_oos_log")

            if self.position_open: 
                if sig_type == 2 or \
                   (self.trade_direction == 1 and sig_type == -1) or \
                   (self.trade_direction == -1 and sig_type == 1):
                    
                    self._log(f"SIGNAL DE SORTIE STRATÉGIE reçu (Type: {sig_type}). Tentative de sortie au marché.", level=1)
                    self._simulate_order_execution(
                        current_timestamp, current_row_data, 
                        'MARKET_EXIT', 
                        self.trade_direction 
                    )
            # Sourcery: Use set for in comparison
            elif sig_type in {1, -1}: # not self.position_open is implied
                self._log(f"SIGNAL D'ENTRÉE {'LONG' if sig_type == 1 else 'SHORT'} reçu de la stratégie.", level=1)
                
                # Sourcery: Swap if/else branches of if expression to remove negation
                entry_px_est_for_qty_calc = (
                    sig_limit_px
                    if sig_order_type == 'LIMIT' and sig_limit_px is not None
                    else float(current_row_data['open'])
                )
                if pd.isna(entry_px_est_for_qty_calc) or entry_px_est_for_qty_calc <= 1e-9:
                    self._log(f"Prix d'estimation pour l'entrée ({entry_px_est_for_qty_calc}) est NaN ou invalide. Pas d'entrée.", level=1, is_error=True)
                    continue 
                
                capital_to_risk_for_trade = self.current_equity * (sig_pos_size_pct if sig_pos_size_pct is not None else 1.0)
                
                target_qty_base_for_entry_raw = self._calculate_max_position_size_base(entry_px_est_for_qty_calc, capital_to_risk_for_trade)
                
                if target_qty_base_for_entry_raw <= 1e-9:
                    self._log(f"Quantité maximale de base calculée ({target_qty_base_for_entry_raw:.8f}) trop petite. Pas d'entrée.", level=1)
                    continue
                
                simulated_entry_order_type = 'LIMIT_ENTRY' if sig_order_type == 'LIMIT' else 'MARKET_ENTRY'
                exec_px_entry, exec_qty_entry, _ = self._simulate_order_execution(
                    current_timestamp, current_row_data, 
                    simulated_entry_order_type, 
                    sig_type, 
                    limit_price_target=sig_limit_px if sig_order_type == 'LIMIT' else None,
                    target_quantity_base_for_entry=target_qty_base_for_entry_raw 
                )
                
                if exec_px_entry is not None and exec_qty_entry is not None and exec_qty_entry > 1e-9:
                    self.current_sl_price = sig_sl_px
                    self.current_tp_price = sig_tp_px
                    self._log(f"SL/TP mis à jour après entrée: SL={self.current_sl_price}, TP={self.current_tp_price}", level=2)
                else: 
                    self._log(f"Ordre d'entrée non exécuté ou quantité nulle. Pas de nouvelle position.", level=1)
                    if self.is_oos_simulation and self.current_trade_cycle_id:
                        self._reset_position_state() 

        # Sourcery: Remove redundant conditional & f-string
        final_date_norm_sim_end = self.df_ohlcv.index[-1].normalize() if not self.df_ohlcv.empty else pd.Timestamp.now(tz='UTC').normalize() # type: ignore
        if final_date_norm_sim_end not in self.daily_equity or self.daily_equity[final_date_norm_sim_end] != self.current_equity :
                self.daily_equity[final_date_norm_sim_end] = self.current_equity
        
        if not self.df_ohlcv.empty:
            self.equity_curve.append({'timestamp': self.df_ohlcv.index[-1], 'equity': self.current_equity, 'type': 'final'})

        self._log(f"Simulation terminée. Nombre total de trades: {len(self.trades)}. Equity finale: {self.current_equity:.2f}", level=1)
        
        equity_curve_df_final = pd.DataFrame(self.equity_curve)
        if not equity_curve_df_final.empty and 'timestamp' in equity_curve_df_final.columns:
            equity_curve_df_final['timestamp'] = pd.to_datetime(equity_curve_df_final['timestamp'], errors='coerce', utc=True)
            equity_curve_df_final.dropna(subset=['timestamp'], inplace=True)

        return self.trades, equity_curve_df_final, self.daily_equity, self.oos_detailed_trades_log
