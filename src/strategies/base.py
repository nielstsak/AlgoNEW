# src/strategies/base.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
import uuid # Pour les client_order_id uniques
from typing import Dict, Any, Tuple, Optional, List 

# Assurez-vous que ces imports sont corrects par rapport à votre structure
from src.config.loader import load_strategy_config_by_name 
from src.utils.exchange_utils import (
    get_pair_config_for_symbol,
    adjust_precision, # <<< CORRIGÉ : Doit être 'adjust_precision'
    adjust_quantity_to_step_size,
    get_precision_from_filter, 
    get_filter_value 
)


logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    REQUIRED_PARAMS: List[str] = [] 

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        self.strategy_name = strategy_name
        self.symbol = symbol.upper()
        self.params = params 
        self.log_prefix = f"[{self.strategy_name}][{self.symbol}]" 

        self.pair_config: Optional[Dict[str, Any]] = None
        self.is_futures: bool = False
        self.leverage: int = 1
        self.initial_equity: float = 0.0
        self.price_precision: Optional[int] = None
        self.quantity_precision: Optional[int] = None
        
        self._validate_params() 

    def get_param(self, param_name: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a parameter value for the strategy.
        """
        return self.params.get(param_name, default)

    def set_backtest_context(self,
                             pair_config: Dict[str, Any],
                             is_futures: bool,
                             leverage: int,
                             initial_equity: float):
        self.pair_config = pair_config
        self.is_futures = is_futures
        self.leverage = leverage
        self.initial_equity = initial_equity
        if self.pair_config:
            self.price_precision = get_precision_from_filter(self.pair_config, 'PRICE_FILTER', 'tickSize')
            self.quantity_precision = get_precision_from_filter(self.pair_config, 'LOT_SIZE', 'stepSize')
            if self.price_precision is None:
                 logger.warning(f"{self.log_prefix} Price precision could not be determined from pair_config for PRICE_FILTER.")
            if self.quantity_precision is None:
                 logger.warning(f"{self.log_prefix} Quantity precision could not be determined from pair_config for LOT_SIZE.")
        else:
            logger.warning(f"{self.log_prefix} Pair config not provided. Precision adjustments might fail.")

    @abstractmethod
    def _validate_params(self):
        """
        Validates the strategy-specific parameters stored in self.params.
        """
        pass

    @abstractmethod
    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures necessary indicators are present in the data_feed.
        """
        pass

    @abstractmethod
    def _generate_signals(self,
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int,
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Generates trading signals based on indicators and current market state.
        """
        pass

    @abstractmethod
    def generate_order_request(self,
                               data: pd.DataFrame, 
                               current_position: int, 
                               available_capital: float, 
                               symbol_info: dict 
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Generates order parameters for live trading or detailed simulation.
        """
        pass

    def get_signal(self,
                   data_feed: pd.DataFrame,
                   current_position_open: bool,
                   current_position_direction: int,
                   current_entry_price: float,
                   current_equity: float 
                  ) -> Dict[str, Any]:
        """
        Main method for backtester to get trading signals and order parameters.
        """
        if data_feed.empty:
            return self._get_default_signal_response("Data feed is empty.")

        try:
            data_with_indicators = self._calculate_indicators(data_feed.copy())
        except Exception as e:
            logger.error(f"{self.log_prefix} Error in _calculate_indicators: {e}", exc_info=True)
            return self._get_default_signal_response(f"Indicator check/prep error: {e}")

        if data_with_indicators.empty:
            return self._get_default_signal_response("Indicators calculation resulted in empty data.")

        signal, order_type_strat, limit_price_strat, sl_price_strat, tp_price_strat, position_size_pct_strat = \
            self._generate_signals(data_with_indicators,
                                   current_position_open,
                                   current_position_direction,
                                   current_entry_price)

        order_type = order_type_strat if order_type_strat else self.get_param("order_type_preference", "MARKET")
        position_size_pct = position_size_pct_strat if position_size_pct_strat is not None else self.get_param('capital_allocation_pct', 1.0)

        entry_order_params_for_log = None
        oco_params_for_log = None

        if signal in [1, -1] and not current_position_open:
            theoretical_entry_price_for_sizing = limit_price_strat if order_type == 'LIMIT' and limit_price_strat is not None else data_with_indicators['open'].iloc[-1]
            
            theoretical_quantity_base = self._calculate_quantity(
                entry_price=theoretical_entry_price_for_sizing,
                available_capital=current_equity, 
                qty_precision=self.quantity_precision,
                symbol_info=self.pair_config, # type: ignore
                symbol=self.symbol,
                position_size_pct=position_size_pct
            )
            if theoretical_quantity_base is None: theoretical_quantity_base = 0.0

            entry_order_params_for_log = self._build_entry_params_formatted(
                side="BUY" if signal == 1 else "SELL",
                quantity_str=f"{theoretical_quantity_base:.{self.quantity_precision or 8}f}",
                order_type=order_type, # type: ignore
                entry_price_str=f"{limit_price_strat:.{self.price_precision or 8}f}" if limit_price_strat else None
            )
            if sl_price_strat is not None and tp_price_strat is not None:
                oco_params_for_log = self._build_oco_params_formatted(
                    entry_side="BUY" if signal == 1 else "SELL",
                    quantity_str=f"{theoretical_quantity_base:.{self.quantity_precision or 8}f}",
                    sl_price_str=f"{sl_price_strat:.{self.price_precision or 8}f}",
                    tp_price_str=f"{tp_price_strat:.{self.price_precision or 8}f}"
                )
        
        final_sl_price = None
        if sl_price_strat is not None and self.pair_config and self.price_precision is not None:
            final_sl_price = adjust_precision(sl_price_strat, self.price_precision, tick_size=get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize'))
        elif sl_price_strat is not None:
            final_sl_price = sl_price_strat

        final_tp_price = None
        if tp_price_strat is not None and self.pair_config and self.price_precision is not None:
            final_tp_price = adjust_precision(tp_price_strat, self.price_precision, tick_size=get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize'))
        elif tp_price_strat is not None:
            final_tp_price = tp_price_strat

        final_limit_price = None
        if limit_price_strat is not None and self.pair_config and self.price_precision is not None:
            final_limit_price = adjust_precision(limit_price_strat, self.price_precision, tick_size=get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize'))
        elif limit_price_strat is not None:
            final_limit_price = limit_price_strat

        return {
            "signal": signal,
            "order_type": order_type,
            "limit_price": final_limit_price,
            "sl_price": final_sl_price,
            "tp_price": final_tp_price,
            "position_size_pct": position_size_pct,
            "entry_order_params_theoretical_for_oos_log": entry_order_params_for_log,
            "oco_params_theoretical_for_oos_log": oco_params_for_log
        }

    def _get_default_signal_response(self, reason: str) -> Dict[str, Any]:
        logger.warning(f"{self.log_prefix} Returning default signal (HOLD): {reason}")
        return {
            "signal": 0, "order_type": "MARKET", "limit_price": None,
            "sl_price": None, "tp_price": None, "position_size_pct": 1.0,
            "entry_order_params_theoretical_for_oos_log": None,
            "oco_params_theoretical_for_oos_log": None
        }

    def _calculate_quantity(self,
                            entry_price: float,
                            available_capital: float, 
                            qty_precision: Optional[int],
                            symbol_info: Dict[str, Any], 
                            symbol: str, 
                            position_size_pct: Optional[float] = None
                           ) -> Optional[float]:
        if entry_price <= 0:
            logger.error(f"{self.log_prefix} Prix d'entrée invalide ({entry_price}) pour le calcul de la quantité.")
            return None
        if available_capital <= 0:
            logger.warning(f"{self.log_prefix} Capital disponible nul ou négatif ({available_capital}).")
            return 0.0

        capital_to_use_pct = position_size_pct if position_size_pct is not None else self.get_param('capital_allocation_pct', 1.0)
        if not (capital_to_use_pct is not None and 0 < capital_to_use_pct <= 1.0): # type: ignore
            logger.warning(f"{self.log_prefix} capital_allocation_pct ({capital_to_use_pct}) invalide. Utilisation de 1.0.")
            capital_to_use_pct = 1.0
        
        capital_for_trade = available_capital * capital_to_use_pct # type: ignore
        
        total_position_value_quote = capital_for_trade * self.leverage
        
        quantity_base_raw = total_position_value_quote / entry_price

        if qty_precision is None:
            logger.warning(f"{self.log_prefix} qty_precision non disponible pour {symbol}. La quantité ne sera pas ajustée à la précision de l'exchange.")
            return round(quantity_base_raw, 8) if quantity_base_raw > 1e-8 else 0.0 

        adjusted_quantity = adjust_quantity_to_step_size(quantity_base_raw, symbol_info, qty_precision)

        min_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
        min_notional_filter = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')

        if min_qty_filter is not None and adjusted_quantity < min_qty_filter:
            logger.warning(f"{self.log_prefix} Quantité ajustée {adjusted_quantity} < minQty {min_qty_filter}. Ordre non possible.")
            return 0.0
        
        if min_notional_filter is not None and (adjusted_quantity * entry_price) < min_notional_filter:
            logger.warning(f"{self.log_prefix} Notionnel de l'ordre ({adjusted_quantity * entry_price}) < minNotional {min_notional_filter}. Ordre non possible.")
            return 0.0
            
        return adjusted_quantity


    def _build_entry_params_formatted(self,
                                      side: str,
                                      quantity_str: str,
                                      order_type: str,
                                      entry_price_str: Optional[str] = None,
                                      time_in_force: Optional[str] = None,
                                      new_client_order_id: Optional[str] = None
                                     ) -> Dict[str, Any]:
        if new_client_order_id is None:
            client_id_suffix = str(uuid.uuid4().hex)[:8] 
            new_client_order_id = f"sim_{self.strategy_name[:4]}_{self.symbol[:3]}_{int(pd.Timestamp.now(tz='UTC').timestamp()*1000)}_{client_id_suffix}"
            new_client_order_id = new_client_order_id[:36] 

        params: Dict[str, Any] = {
            "symbol": self.symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity_str,
            "newClientOrderId": new_client_order_id
        }
        if order_type.upper() in ["LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
            if entry_price_str is None:
                raise ValueError(f"Le prix (entry_price_str) doit être spécifié pour le type d'ordre {order_type}")
            params["price"] = entry_price_str
        
        if time_in_force: 
            params["timeInForce"] = time_in_force
        
        return params

    def _build_oco_params_formatted(self,
                                    entry_side: str, 
                                    quantity_str: str,
                                    sl_price_str: str,
                                    tp_price_str: str,
                                    stop_limit_price_str: Optional[str] = None, 
                                    stop_limit_time_in_force: Optional[str] = None,
                                    list_client_order_id: Optional[str] = None
                                   ) -> Dict[str, Any]:
        exit_side = "SELL" if entry_side == "BUY" else "BUY"
        
        if list_client_order_id is None:
            client_id_suffix = str(uuid.uuid4().hex)[:7]
            list_client_order_id = f"simOCO_{self.symbol[:3]}_{int(pd.Timestamp.now(tz='UTC').timestamp()*1000)}_{client_id_suffix}"
            list_client_order_id = list_client_order_id[:32]

        oco_params: Dict[str, Any] = {
            "symbol": self.symbol,
            "side": exit_side, 
            "quantity": quantity_str,
            "price": tp_price_str,  
            "stopPrice": sl_price_str, 
            "listClientOrderId": list_client_order_id,
        }
        if stop_limit_price_str:
            oco_params["stopLimitPrice"] = stop_limit_price_str
        
        if stop_limit_time_in_force: 
            oco_params["stopLimitTimeInForce"] = stop_limit_time_in_force
            
        return oco_params
