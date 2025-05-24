import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.utils.exchange_utils import (adjust_precision,
                                      get_precision_from_filter,
                                      get_filter_value) # Ajouté

logger = logging.getLogger(__name__)

class PsarReversalOtocoStrategy(BaseStrategy):
    REQUIRED_PARAMS = [
        'psar_step', 'psar_max_step',
        'atr_period_sl_tp', 'sl_atr_mult', 'tp_atr_mult',
        'indicateur_frequence_psar', 'atr_base_frequency_sl_tp',
        'capital_allocation_pct', 'order_type_preference' # Ajoutés
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        super().__init__(strategy_name, symbol, params)
        self.log_prefix = f"[{self.strategy_name}][{self.symbol}]"
        
        self.psarl_col_strat = "PSARl_strat"
        self.psars_col_strat = "PSARs_strat"
        self.atr_col_strat = "ATR_strat"

        logger.info(f"{self.log_prefix} Stratégie initialisée. Paramètres: {self.params}")

    def _validate_params(self):
        missing_params = [p for p in self.REQUIRED_PARAMS if self.get_param(p) is None]
        if missing_params:
            raise ValueError(f"{self.log_prefix} Missing required parameters: {', '.join(missing_params)}")
        
        if not (0 < self.get_param('capital_allocation_pct') <= 1):
            raise ValueError(f"{self.log_prefix} capital_allocation_pct must be > 0 and <= 1.")
        logger.debug(f"{self.log_prefix} Parameters validated successfully.")

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        expected_strat_cols = [self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat]
        for col_name in expected_strat_cols:
            if col_name not in df.columns:
                logger.error(f"{self.log_prefix} Colonne indicateur attendue '{col_name}' manquante.")
                df[col_name] = np.nan
        
        required_ohlc = ['open', 'high', 'low', 'close', 'volume']
        for col in required_ohlc:
            if col not in df.columns:
                logger.error(f"{self.log_prefix} Colonne OHLCV de base '{col}' manquante.")
                df[col] = np.nan
        return df

    def _generate_signals(self,
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int,
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        
        if len(data_with_indicators) < 2:
            return 0, self.get_param("order_type_preference", "MARKET"), None, None, None, self.get_param('capital_allocation_pct', 1.0)

        latest_row = data_with_indicators.iloc[-1]
        previous_row = data_with_indicators.iloc[-2]

        close_curr = latest_row.get('close')
        psarl_curr = latest_row.get(self.psarl_col_strat)
        psars_curr = latest_row.get(self.psars_col_strat)
        atr_curr = latest_row.get(self.atr_col_strat)
        
        psarl_prev = previous_row.get(self.psarl_col_strat)
        psars_prev = previous_row.get(self.psars_col_strat)

        signal = 0
        sl_price = None
        tp_price = None

        # Un PSAR est actif à la fois (soit PSARl, soit PSARs a une valeur, l'autre est NaN)
        # Entrée Long: si PSAR était au-dessus (PSARs_prev notna) et croise en dessous (PSARl_curr notna)
        # Entrée Short: si PSAR était en dessous (PSARl_prev notna) et croise au-dessus (PSARs_curr notna)
        entry_long_triggered = pd.notna(psars_prev) and pd.isna(psarl_prev) and \
                               pd.notna(psarl_curr) and pd.isna(psars_curr)
        entry_short_triggered = pd.notna(psarl_prev) and pd.isna(psars_prev) and \
                                pd.notna(psars_curr) and pd.isna(psarl_curr)

        if pd.isna(close_curr) or pd.isna(atr_curr) or atr_curr <= 0: # type: ignore
            logger.debug(f"{self.log_prefix} Close ou ATR invalide à {data_with_indicators.index[-1]}. Signal Hold.")
            return 0, self.get_param("order_type_preference", "MARKET"), None, None, None, self.get_param('capital_allocation_pct', 1.0)

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        if not current_position_open:
            if entry_long_triggered:
                signal = 1 # Buy
                sl_price = close_curr - sl_atr_mult * atr_curr # type: ignore
                tp_price = close_curr + tp_atr_mult * atr_curr # type: ignore
            elif entry_short_triggered:
                signal = -1 # Sell/Short
                sl_price = close_curr + sl_atr_mult * atr_curr # type: ignore
                tp_price = close_curr - tp_atr_mult * atr_curr # type: ignore
        else: # Position is open
            if current_position_direction == 1 and entry_short_triggered: # Exit long on PSAR short signal
                signal = 2
            elif current_position_direction == -1 and entry_long_triggered: # Exit short on PSAR long signal
                signal = 2
        
        order_type = self.get_param('order_type_preference', "MARKET")
        limit_price = None
        if signal != 0 and order_type == "LIMIT":
            limit_price = close_curr

        position_size_pct = self.get_param('capital_allocation_pct', 1.0)
        
        return signal, order_type, limit_price, sl_price, tp_price, position_size_pct

    def generate_order_request(self,
                               data: pd.DataFrame,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        
        data_with_indicators = self._calculate_indicators(data)
        if data_with_indicators.empty or len(data_with_indicators) < 2:
            logger.warning(f"{self.log_prefix} [Live] Pas assez de données pour generate_order_request.")
            return None

        if current_position != 0:
             logger.debug(f"{self.log_prefix} [Live] Position déjà ouverte. generate_order_request ne génère pas de nouvel ordre d'entrée.")
             return None

        signal, order_type, limit_price, sl_price_raw, tp_price_raw, pos_size_pct = \
            self._generate_signals(data_with_indicators, False, 0, 0.0)

        if signal not in [1, -1]:
            logger.debug(f"{self.log_prefix} [Live] Aucun signal d'entrée généré.")
            return None

        latest_row = data_with_indicators.iloc[-1]
        entry_price_theoretical = latest_row.get('close') # Utiliser close pour calcul de quantité
        if order_type == "LIMIT" and limit_price is not None:
            entry_price_theoretical = limit_price
        
        if pd.isna(entry_price_theoretical):
            logger.error(f"{self.log_prefix} [Live] Prix d'entrée théorique est NaN.")
            return None

        quantity_base = self._calculate_quantity(
            entry_price=entry_price_theoretical,
            available_capital=available_capital,
            qty_precision=get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize'),
            symbol_info=symbol_info,
            symbol=self.symbol,
            position_size_pct=pos_size_pct
        )

        if quantity_base is None or quantity_base <= 0:
            logger.warning(f"{self.log_prefix} [Live] Quantité calculée est None ou <= 0 ({quantity_base}).")
            return None

        price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
        if price_precision is None: return None
        
        entry_price_for_order_str: Optional[str] = None
        if order_type == "LIMIT" and limit_price is not None:
            adjusted_limit_price = adjust_precision(limit_price, price_precision)
            if adjusted_limit_price is None: return None
            entry_price_for_order_str = f"{adjusted_limit_price:.{price_precision}f}"
        
        entry_order_params = self._build_entry_params_formatted(
            side="BUY" if signal == 1 else "SELL",
            quantity_str=f"{quantity_base:.{get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize') or 8}f}",
            order_type=order_type, # type: ignore
            entry_price_str=entry_price_for_order_str
        )
        if not entry_order_params: return None

        sl_tp_prices_for_live = {}
        if sl_price_raw is not None: sl_tp_prices_for_live['sl_price'] = sl_price_raw
        if tp_price_raw is not None: sl_tp_prices_for_live['tp_price'] = tp_price_raw
        
        logger.info(f"{self.log_prefix} [Live] Requête d'ordre générée: {entry_order_params}, SL/TP bruts: {sl_tp_prices_for_live}")
        return entry_order_params, sl_tp_prices_for_live
