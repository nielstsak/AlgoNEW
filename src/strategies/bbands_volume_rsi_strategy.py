import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.utils.exchange_utils import (adjust_precision,
                                      get_filter_value,
                                      get_precision_from_filter)

logger = logging.getLogger(__name__)

class BbandsVolumeRsiStrategy(BaseStrategy):
    REQUIRED_PARAMS = [
        'bbands_period', 'bbands_std_dev', 'indicateur_frequence_bbands',
        'volume_ma_period', 'indicateur_frequence_volume',
        'rsi_period', 'indicateur_frequence_rsi',
        'rsi_buy_breakout_threshold', 'rsi_sell_breakout_threshold',
        'atr_period_sl_tp', 'atr_base_frequency_sl_tp', 'sl_atr_mult', 'tp_atr_mult'
    ]

    def __init__(self, params: dict):
        super().__init__(params)
        self.log_prefix = f"[{self.__class__.__name__}]"
        
        self.bb_upper_col_strat = "BB_UPPER_strat"
        self.bb_middle_col_strat = "BB_MIDDLE_strat"
        self.bb_lower_col_strat = "BB_LOWER_strat"
        self.bb_bandwidth_col_strat = "BB_BANDWIDTH_strat"
        
        vol_freq_param = self.get_param('indicateur_frequence_volume')
        if vol_freq_param and str(vol_freq_param).lower() != "1min":
            self.volume_kline_col_strat = f"Kline_{vol_freq_param}_volume"
        else:
            self.volume_kline_col_strat = "volume" 

        self.volume_ma_col_strat = "Volume_MA_strat"
        self.rsi_col_strat = "RSI_strat"
        self.atr_col_strat = "ATR_strat"

        self._signals: Optional[pd.DataFrame] = None
        logger.info(f"{self.log_prefix} Stratégie initialisée. Paramètres: {self.params}")
        logger.info(f"{self.log_prefix} Colonne volume source attendue pour comparaison: {self.volume_kline_col_strat}")


    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        expected_strat_cols = [
            self.bb_upper_col_strat, self.bb_middle_col_strat, self.bb_lower_col_strat, 
            self.bb_bandwidth_col_strat, self.volume_kline_col_strat, 
            self.volume_ma_col_strat, self.rsi_col_strat, self.atr_col_strat
        ]
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
        required_cols_for_signal = [
            self.bb_upper_col_strat, self.bb_lower_col_strat,
            self.volume_kline_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat, 'close', 'open'
        ]
        if df_with_indicators.empty or \
           any(col not in df_with_indicators.columns for col in required_cols_for_signal) or \
           df_with_indicators[required_cols_for_signal].isnull().all().all() or \
           len(df_with_indicators) < 2 :
            logger.warning(f"{self.log_prefix} Données insuffisantes ou colonnes essentielles NaN pour générer les signaux.")
            self._signals = pd.DataFrame(index=data.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            self._signals[self._signals.select_dtypes(include=['object', 'bool']).columns] = False
            return

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))
        rsi_buy_thresh = float(self.get_param('rsi_buy_breakout_threshold'))
        rsi_sell_thresh = float(self.get_param('rsi_sell_breakout_threshold'))

        close_curr = df_with_indicators['close'] 
        close_prev = close_curr.shift(1)
        bb_upper_curr = df_with_indicators[self.bb_upper_col_strat]
        bb_lower_curr = df_with_indicators[self.bb_lower_col_strat]
        bb_upper_prev = bb_upper_curr.shift(1)
        bb_lower_prev = bb_lower_curr.shift(1)
        volume_kline_curr = df_with_indicators[self.volume_kline_col_strat]
        volume_ma_curr = df_with_indicators[self.volume_ma_col_strat]
        rsi_curr = df_with_indicators[self.rsi_col_strat]
        atr_curr = df_with_indicators[self.atr_col_strat]

        long_bb_breakout_curr = close_curr > bb_upper_curr
        long_not_breakout_prev = close_prev <= bb_upper_prev # Condition sur la barre précédente
        long_volume_confirm_curr = volume_kline_curr > volume_ma_curr
        long_rsi_confirm_curr = rsi_curr > rsi_buy_thresh
        entry_long_trigger = long_bb_breakout_curr & long_not_breakout_prev & \
                             long_volume_confirm_curr & long_rsi_confirm_curr

        short_bb_breakout_curr = close_curr < bb_lower_curr
        short_not_breakout_prev = close_prev >= bb_lower_prev # Condition sur la barre précédente
        short_volume_confirm_curr = volume_kline_curr > volume_ma_curr 
        short_rsi_confirm_curr = rsi_curr < rsi_sell_thresh
        entry_short_trigger = short_bb_breakout_curr & short_not_breakout_prev & \
                              short_volume_confirm_curr & short_rsi_confirm_curr
        
        signals_df = pd.DataFrame(index=df_with_indicators.index)
        signals_df['entry_long'] = entry_long_trigger
        signals_df['entry_short'] = entry_short_trigger
        signals_df['exit_long'] = False 
        signals_df['exit_short'] = False 
        
        entry_price_series_ref = close_curr 
        valid_data_for_sltp = atr_curr.notna() & entry_price_series_ref.notna()

        signals_df['sl'] = np.nan
        signals_df['tp'] = np.nan
        signals_df.loc[entry_long_trigger & valid_data_for_sltp, 'sl'] = entry_price_series_ref - sl_atr_mult * atr_curr
        signals_df.loc[entry_long_trigger & valid_data_for_sltp, 'tp'] = entry_price_series_ref + tp_atr_mult * atr_curr
        signals_df.loc[entry_short_trigger & valid_data_for_sltp, 'sl'] = entry_price_series_ref + sl_atr_mult * atr_curr
        signals_df.loc[entry_short_trigger & valid_data_for_sltp, 'tp'] = entry_price_series_ref - tp_atr_mult * atr_curr
        
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
        if len(df_verified_indicators) < 2:
             logger.warning(f"{self.log_prefix} Pas assez de lignes après _calculate_indicators pour évaluer les conditions.")
             return None
             
        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2]

        required_cols_check = [
            'close', 'open', self.bb_upper_col_strat, self.bb_lower_col_strat,
            self.volume_kline_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat
        ]
        if latest_data[required_cols_check].isnull().any():
            nan_cols_latest = latest_data[required_cols_check].index[latest_data[required_cols_check].isnull()].tolist()
            logger.warning(f"{self.log_prefix} Indicateurs essentiels NaN sur dernière donnée: {nan_cols_latest}. Pas de requête.")
            return None
        
        required_cols_prev = [col for col in required_cols_check if col != self.atr_col_strat and col in previous_data.index] # Check if col exists
        # Ensure columns exist before trying to access them for isnull check
        valid_prev_cols_to_check = [col for col in required_cols_prev if col in previous_data.index]
        if previous_data[valid_prev_cols_to_check].isnull().any():
            nan_cols_previous = previous_data[valid_prev_cols_to_check].index[previous_data[valid_prev_cols_to_check].isnull()].tolist()
            logger.warning(f"{self.log_prefix} Indicateurs essentiels NaN sur avant-dernière donnée: {nan_cols_previous}. Pas de requête.")
            return None

        close_curr = latest_data['close']
        bb_upper_curr = latest_data[self.bb_upper_col_strat]
        bb_lower_curr = latest_data[self.bb_lower_col_strat]
        volume_kline_curr = latest_data[self.volume_kline_col_strat]
        volume_ma_curr = latest_data[self.volume_ma_col_strat]
        rsi_curr = latest_data[self.rsi_col_strat]
        atr_value = latest_data[self.atr_col_strat]
        
        close_prev = previous_data['close']
        # Handle potential NaN for prev BB values if data is short
        bb_upper_prev = previous_data.get(self.bb_upper_col_strat, np.nan)
        bb_lower_prev = previous_data.get(self.bb_lower_col_strat, np.nan)
        volume_kline_prev = previous_data.get(self.volume_kline_col_strat, np.nan)
        volume_ma_prev = previous_data.get(self.volume_ma_col_strat, np.nan)
        rsi_prev = previous_data.get(self.rsi_col_strat, np.nan)
        
        rsi_buy_thresh = float(self.get_param('rsi_buy_breakout_threshold'))
        rsi_sell_thresh = float(self.get_param('rsi_sell_breakout_threshold'))

        current_long_bb_breakout = close_curr > bb_upper_curr
        current_long_volume_conf = volume_kline_curr > volume_ma_curr
        current_long_rsi_conf = rsi_curr > rsi_buy_thresh
        all_current_long_conditions_met = current_long_bb_breakout and current_long_volume_conf and current_long_rsi_conf

        previous_long_bb_breakout = close_prev > bb_upper_prev if pd.notna(bb_upper_prev) and pd.notna(close_prev) else False
        previous_long_volume_conf = volume_kline_prev > volume_ma_prev if pd.notna(volume_kline_prev) and pd.notna(volume_ma_prev) else False
        previous_long_rsi_conf = rsi_prev > rsi_buy_thresh if pd.notna(rsi_prev) else False
        all_previous_long_conditions_met = previous_long_bb_breakout and previous_long_volume_conf and previous_long_rsi_conf

        current_short_bb_breakout = close_curr < bb_lower_curr
        current_short_volume_conf = volume_kline_curr > volume_ma_curr
        current_short_rsi_conf = rsi_curr < rsi_sell_thresh
        all_current_short_conditions_met = current_short_bb_breakout and current_short_volume_conf and current_short_rsi_conf

        previous_short_bb_breakout = close_prev < bb_lower_prev if pd.notna(bb_lower_prev) and pd.notna(close_prev) else False
        previous_short_volume_conf = volume_kline_prev > volume_ma_prev if pd.notna(volume_kline_prev) and pd.notna(volume_ma_prev) else False
        previous_short_rsi_conf = rsi_prev < rsi_sell_thresh if pd.notna(rsi_prev) else False
        all_previous_short_conditions_met = previous_short_bb_breakout and previous_short_volume_conf and previous_short_rsi_conf

        side: Optional[str] = None
        sl_price_raw: Optional[float] = None
        tp_price_raw: Optional[float] = None
        entry_price_theoretical = latest_data['open']
        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        if pd.isna(atr_value) or atr_value <= 1e-9:
            logger.warning(f"{self.log_prefix} Valeur ATR invalide ({atr_value}). Pas de requête d'ordre.")
            return None

        if all_current_long_conditions_met and not all_previous_long_conditions_met:
            side = 'BUY'
            sl_price_raw = entry_price_theoretical - sl_atr_mult * atr_value
            tp_price_raw = entry_price_theoretical + tp_atr_mult * atr_value
        elif all_current_short_conditions_met and not all_previous_short_conditions_met:
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
            
            tick_size_str = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')
            tick_size = float(tick_size_str) if tick_size_str is not None and isinstance(tick_size_str, (float, int, str)) and str(tick_size_str).replace('.','',1).isdigit() else 0.00000001

            if side == 'BUY':
                if sl_price_raw >= entry_price_for_order_request : sl_price_raw = entry_price_for_order_request - tick_size
                if tp_price_raw <= entry_price_for_order_request : tp_price_raw = entry_price_for_order_request + tick_size
            elif side == 'SELL':
                if sl_price_raw <= entry_price_for_order_request : sl_price_raw = entry_price_for_order_request + tick_size
                if tp_price_raw >= entry_price_for_order_request : tp_price_raw = entry_price_for_order_request - tick_size

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
