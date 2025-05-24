import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.utils.exchange_utils import (adjust_precision,
                                      get_precision_from_filter)

logger = logging.getLogger(__name__)

class PsarReversalOtocoStrategy(BaseStrategy):
    REQUIRED_PARAMS = [
        'psar_step', 'psar_max_step',
        'atr_period_sl_tp', 'sl_atr_mult', 'tp_atr_mult',
        'indicateur_frequence_psar', 'atr_base_frequency_sl_tp'
    ]

    def __init__(self, params: dict):
        super().__init__(params)
        self.log_prefix = f"[{self.__class__.__name__}]"
        
        self.psarl_col_strat = "PSARl_strat"
        self.psars_col_strat = "PSARs_strat"
        self.atr_col_strat = "ATR_strat"

        self._signals: Optional[pd.DataFrame] = None
        logger.info(f"{self.log_prefix} Stratégie initialisée. Paramètres: {self.params}")

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        expected_strat_cols = [self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat]
        for col_name in expected_strat_cols:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur attendue '{col_name}' manquante. Ajoutée avec NaN.")
                df[col_name] = np.nan
        
        required_ohlc = ['open', 'high', 'low', 'close']
        for col in required_ohlc:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne OHLC de base '{col}' manquante. Ajoutée avec NaN.")
                df[col] = np.nan
        return df

    def generate_signals(self, data: pd.DataFrame) -> None:
        df_with_indicators = self._calculate_indicators(data)
        required_cols_for_signal = [self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat, 'close']
        if df_with_indicators.empty or \
           any(col not in df_with_indicators.columns for col in required_cols_for_signal) or \
           df_with_indicators[required_cols_for_signal].isnull().all().all() or \
           len(df_with_indicators) < 2:
            logger.warning(f"{self.log_prefix} Données/colonnes insuffisantes pour générer les signaux. Signaux vides générés.")
            self._signals = pd.DataFrame(index=data.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            self._signals[self._signals.select_dtypes(include=['object', 'bool']).columns] = False
            return

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        close_curr = df_with_indicators['close']
        psarl_curr = df_with_indicators[self.psarl_col_strat]
        psars_curr = df_with_indicators[self.psars_col_strat]
        atr_curr = df_with_indicators[self.atr_col_strat]

        entry_long_trigger = psarl_curr.notna() & psars_curr.shift(1).notna()
        entry_short_trigger = psars_curr.notna() & psarl_curr.shift(1).notna()
        
        valid_data_for_signal = atr_curr.notna() & close_curr.notna()

        signals_df = pd.DataFrame(index=df_with_indicators.index)
        signals_df['entry_long'] = entry_long_trigger & valid_data_for_signal
        signals_df['entry_short'] = entry_short_trigger & valid_data_for_signal
        signals_df['exit_long'] = entry_short_trigger & valid_data_for_signal
        signals_df['exit_short'] = entry_long_trigger & valid_data_for_signal

        signals_df['sl'] = np.nan
        signals_df['tp'] = np.nan

        signals_df.loc[signals_df['entry_long'], 'sl'] = close_curr - sl_atr_mult * atr_curr
        signals_df.loc[signals_df['entry_long'], 'tp'] = close_curr + tp_atr_mult * atr_curr
        signals_df.loc[signals_df['entry_short'], 'sl'] = close_curr + sl_atr_mult * atr_curr
        signals_df.loc[signals_df['entry_short'], 'tp'] = close_curr - tp_atr_mult * atr_curr

        self._signals = signals_df[['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']].reindex(data.index)

    def generate_order_request(self,
                               data: pd.DataFrame,
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        if current_position != 0:
            return None
        if data.empty or len(data) < 2:
            logger.warning(f"{self.log_prefix} Données d'entrée vides ou insuffisantes pour generate_order_request.")
            return None

        df_verified_indicators = self._calculate_indicators(data.copy())
        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2]

        required_cols = ['close', 'open', self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat]
        if latest_data[required_cols].isnull().any() or \
           previous_data[[self.psarl_col_strat, self.psars_col_strat]].isnull().any():
            return None

        psarl_curr = latest_data[self.psarl_col_strat]
        psars_curr = latest_data[self.psars_col_strat]
        psarl_prev = previous_data[self.psarl_col_strat]
        psars_prev = previous_data[self.psars_col_strat]
        entry_price_theoretical = latest_data['open']
        atr_value = latest_data[self.atr_col_strat]

        if pd.isna(atr_value) or atr_value <= 1e-9:
            return None

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        side: Optional[str] = None
        sl_price_raw: Optional[float] = None
        tp_price_raw: Optional[float] = None

        if pd.notna(psars_prev) and pd.notna(psarl_curr): # Était baissier (PSARs), devient haussier (PSARl)
            side = 'BUY'
            sl_price_raw = entry_price_theoretical - sl_atr_mult * atr_value
            tp_price_raw = entry_price_theoretical + tp_atr_mult * atr_value
        elif pd.notna(psarl_prev) and pd.notna(psars_curr): # Était haussier (PSARl), devient baissier (PSARs)
            side = 'SELL'
            sl_price_raw = entry_price_theoretical + sl_atr_mult * atr_value
            tp_price_raw = entry_price_theoretical - tp_atr_mult * atr_value

        if side and sl_price_raw is not None and tp_price_raw is not None:
            if (side == 'BUY' and (sl_price_raw >= entry_price_theoretical or tp_price_raw <= entry_price_theoretical)) or \
               (side == 'SELL' and (sl_price_raw <= entry_price_theoretical or tp_price_raw >= entry_price_theoretical)):
                return None
                
            price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            if price_precision is None or qty_precision is None: return None

            quantity = self._calculate_quantity(
                entry_price=entry_price_theoretical, available_capital=available_capital,
                qty_precision=qty_precision, symbol_info=symbol_info, symbol=symbol
            )
            if quantity is None or quantity <= 1e-9: return None

            entry_price_for_order_request = adjust_precision(entry_price_theoretical, price_precision, round)
            if entry_price_for_order_request is None: return None
                
            entry_price_str = f"{entry_price_for_order_request:.{price_precision}f}"
            quantity_str = f"{quantity:.{qty_precision}f}"
            
            order_type_pref = self.get_param('order_type_preference', "LIMIT")
            entry_order_params = self._build_entry_params_formatted(
                symbol=symbol, side=side, quantity_str=quantity_str,
                entry_price_str=entry_price_str if order_type_pref == "LIMIT" else None,
                order_type=order_type_pref
            )
            if not entry_order_params: return None
            sl_tp_raw_prices = {'sl_price': sl_price_raw, 'tp_price': tp_price_raw}
            return entry_order_params, sl_tp_raw_prices
        return None
