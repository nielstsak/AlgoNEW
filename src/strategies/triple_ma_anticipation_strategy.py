import logging
import math 
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.strategies.base import BaseStrategy
from src.utils.exchange_utils import (adjust_precision,
                                      get_precision_from_filter)
from src.data import data_utils # Pour get_kline_prefix_effective

logger = logging.getLogger(__name__)

class TripleMAAnticipationStrategy(BaseStrategy):
    REQUIRED_PARAMS = [
        'ma_short_period', 'ma_medium_period', 'ma_long_period',
        'indicateur_frequence_mms', 
        'indicateur_frequence_mmm', 
        'indicateur_frequence_mml', 
        'atr_period_sl_tp', 'atr_base_frequency_sl_tp',
        'sl_atr_mult', 'tp_atr_mult',
        'allow_shorting', 'order_type_preference',
        'anticipate_crossovers', 
    ]

    def __init__(self, params: dict):
        super().__init__(params)
        self.log_prefix = f"[{self.__class__.__name__}]"
        
        self.ma_short_col_strat = "MA_SHORT_strat"
        self.ma_medium_col_strat = "MA_MEDIUM_strat"
        self.ma_long_col_strat = "MA_LONG_strat"
        self.atr_col_strat = "ATR_strat"

        self.allow_shorting = bool(self.get_param('allow_shorting', False))
        self.order_type_preference = self.get_param('order_type_preference', "MARKET")

        self.anticipate_crossovers = bool(self.get_param('anticipate_crossovers', False))
        if self.anticipate_crossovers:
            self.slope_ma_short_col_strat = "SLOPE_MA_SHORT_strat"
            self.slope_ma_medium_col_strat = "SLOPE_MA_MEDIUM_strat"
            self.anticipation_slope_period = int(self.get_param('anticipation_slope_period', 3))
            if self.anticipation_slope_period < 2:
                logger.warning(f"{self.log_prefix} anticipation_slope_period ({self.anticipation_slope_period}) < 2. Forcé à 2.")
                self.anticipation_slope_period = 2
            self.anticipation_convergence_threshold_pct = float(self.get_param('anticipation_convergence_threshold_pct', 0.005))
            if 'anticipation_slope_period' not in self.params or self.get_param('anticipation_slope_period') is None:
                 raise ValueError("Missing 'anticipation_slope_period' required when 'anticipate_crossovers' is true.")
            if 'anticipation_convergence_threshold_pct' not in self.params or self.get_param('anticipation_convergence_threshold_pct') is None:
                 raise ValueError("Missing 'anticipation_convergence_threshold_pct' required when 'anticipate_crossovers' is true.")
        else:
            self.slope_ma_short_col_strat = None # Explicitly set to None
            self.slope_ma_medium_col_strat = None # Explicitly set to None

        self._signals: Optional[pd.DataFrame] = None
        logger.info(f"{self.log_prefix} Stratégie initialisée. Anticipation: {self.anticipate_crossovers}. Paramètres: {self.params}")

    def _calculate_slope(self, series: pd.Series, window: int) -> pd.Series:
        if not isinstance(series, pd.Series) or series.empty or series.isnull().all() or len(series) < window or window < 2:
            return pd.Series([np.nan] * len(series), index=series.index, name=f"{series.name}_slope{window}" if series.name else f"slope{window}")
        
        def get_slope_value(y_values_window):
            y_clean = y_values_window.dropna()
            if len(y_clean) < 2: 
                return np.nan
            x_clean = np.arange(len(y_clean))
            try:
                slope = np.polyfit(x_clean, y_clean, 1)[0]
                return slope
            except (np.linalg.LinAlgError, TypeError, ValueError):
                return np.nan

        slopes = series.rolling(window=window, min_periods=window).apply(get_slope_value, raw=False)
        return slopes.rename(f"{series.name}_slope{window}" if series.name else f"slope{window}")


    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        expected_base_strat_cols = [
            self.ma_short_col_strat, self.ma_medium_col_strat,
            self.ma_long_col_strat, self.atr_col_strat
        ]
        for col_name in expected_base_strat_cols:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur de base attendue '{col_name}' manquante. Ajoutée avec NaN.")
                df[col_name] = np.nan

        if self.anticipate_crossovers:
            # Ensure slope columns are initialized if they are supposed to be used
            if self.slope_ma_short_col_strat and self.slope_ma_short_col_strat not in df.columns:
                logger.info(f"{self.log_prefix} Colonne '{self.slope_ma_short_col_strat}' non fournie. Tentative de calcul à partir de '{self.ma_short_col_strat}'.")
                if self.ma_short_col_strat in df and df[self.ma_short_col_strat].notna().any():
                    df[self.slope_ma_short_col_strat] = self._calculate_slope(df[self.ma_short_col_strat], self.anticipation_slope_period)
                else:
                    df[self.slope_ma_short_col_strat] = np.nan # Ensure column exists
            
            if self.slope_ma_medium_col_strat and self.slope_ma_medium_col_strat not in df.columns:
                logger.info(f"{self.log_prefix} Colonne '{self.slope_ma_medium_col_strat}' non fournie. Tentative de calcul à partir de '{self.ma_medium_col_strat}'.")
                if self.ma_medium_col_strat in df and df[self.ma_medium_col_strat].notna().any():
                     df[self.slope_ma_medium_col_strat] = self._calculate_slope(df[self.ma_medium_col_strat], self.anticipation_slope_period)
                else:
                    df[self.slope_ma_medium_col_strat] = np.nan # Ensure column exists
        
        required_ohlc = ['open', 'high', 'low', 'close']
        for col in required_ohlc:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne OHLC de base '{col}' manquante. Ajoutée avec NaN.")
                df[col] = np.nan
        return df

    def generate_signals(self, data: pd.DataFrame) -> None:
        df_with_indicators = self._calculate_indicators(data)

        required_cols_for_signal = [
            self.ma_short_col_strat, self.ma_medium_col_strat, self.ma_long_col_strat,
            self.atr_col_strat, 'close'
        ]
        # Add slope columns to required if anticipation is on and columns are defined
        if self.anticipate_crossovers:
            if self.slope_ma_short_col_strat: required_cols_for_signal.append(self.slope_ma_short_col_strat)
            if self.slope_ma_medium_col_strat: required_cols_for_signal.append(self.slope_ma_medium_col_strat)


        if df_with_indicators.empty or \
           any(col not in df_with_indicators.columns for col in required_cols_for_signal) or \
           df_with_indicators[[self.ma_short_col_strat, self.ma_medium_col_strat, self.atr_col_strat, 'close']].isnull().all().all() or \
           len(df_with_indicators) < 2:
            logger.warning(f"{self.log_prefix} Données/colonnes insuffisantes pour générer les signaux. Signaux vides générés.")
            self._signals = pd.DataFrame(index=data.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            self._signals[self._signals.select_dtypes(include=['object', 'bool']).columns] = False
            return

        df = df_with_indicators
        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        ma_short = df[self.ma_short_col_strat]
        ma_medium = df[self.ma_medium_col_strat]
        ma_long = df[self.ma_long_col_strat]
        atr_val = df[self.atr_col_strat]
        close_price = df['close']

        actual_long_entry_cross = (ma_short > ma_medium) & (ma_short.shift(1) <= ma_medium.shift(1))
        actual_long_exit_cross = (ma_short < ma_medium) & (ma_short.shift(1) >= ma_medium.shift(1))
        actual_short_entry_cross = pd.Series(False, index=df.index)
        actual_short_exit_cross = pd.Series(False, index=df.index)
        if self.allow_shorting:
            actual_short_entry_cross = (ma_short < ma_medium) & (ma_short.shift(1) >= ma_medium.shift(1))
            actual_short_exit_cross = (ma_short > ma_medium) & (ma_short.shift(1) <= ma_medium.shift(1))

        anticipated_long_entry = pd.Series(False, index=df.index)
        anticipated_long_exit = pd.Series(False, index=df.index)
        anticipated_short_entry = pd.Series(False, index=df.index)
        anticipated_short_exit = pd.Series(False, index=df.index)

        if self.anticipate_crossovers and self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat and \
           self.slope_ma_short_col_strat in df and self.slope_ma_medium_col_strat in df and \
           df[self.slope_ma_short_col_strat].notna().any() and df[self.slope_ma_medium_col_strat].notna().any():
            
            slope_short = df[self.slope_ma_short_col_strat]
            slope_medium = df[self.slope_ma_medium_col_strat]
            convergence_distance = ma_medium * self.anticipation_convergence_threshold_pct
            ma_diff_abs = abs(ma_short - ma_medium)

            is_converging_up = slope_short > slope_medium
            is_below_and_closing_for_long = (ma_short < ma_medium) & (ma_diff_abs < convergence_distance)
            main_trend_bullish = ma_medium > ma_long
            anticipated_long_entry = is_converging_up & is_below_and_closing_for_long & main_trend_bullish

            is_converging_down_for_exit_long = slope_short < slope_medium
            is_above_and_closing_for_long_exit = (ma_short > ma_medium) & (ma_diff_abs < convergence_distance)
            anticipated_long_exit = is_converging_down_for_exit_long & is_above_and_closing_for_long_exit

            if self.allow_shorting:
                is_converging_down_for_entry_short = slope_short < slope_medium
                is_above_and_closing_for_short_entry = (ma_short > ma_medium) & (ma_diff_abs < convergence_distance)
                main_trend_bearish = ma_medium < ma_long
                anticipated_short_entry = is_converging_down_for_entry_short & is_above_and_closing_for_short_entry & main_trend_bearish
                
                is_converging_up_for_exit_short = slope_short > slope_medium
                is_below_and_closing_for_short_exit = (ma_short < ma_medium) & (ma_diff_abs < convergence_distance)
                anticipated_short_exit = is_converging_up_for_exit_short & is_below_and_closing_for_short_exit
        
        signals_df = pd.DataFrame(index=df.index)
        signals_df['entry_long'] = actual_long_entry_cross | anticipated_long_entry
        signals_df['exit_long'] = actual_long_exit_cross | anticipated_long_exit
        signals_df['entry_short'] = actual_short_entry_cross | anticipated_short_entry
        signals_df['exit_short'] = actual_short_exit_cross | anticipated_short_exit
        
        signals_df['sl'] = np.nan
        signals_df['tp'] = np.nan
        valid_sltp_data = atr_val.notna() & close_price.notna()

        signals_df.loc[signals_df['entry_long'] & valid_sltp_data, 'sl'] = close_price - (atr_val * sl_atr_mult)
        signals_df.loc[signals_df['entry_long'] & valid_sltp_data, 'tp'] = close_price + (atr_val * tp_atr_mult)
        if self.allow_shorting:
            signals_df.loc[signals_df['entry_short'] & valid_sltp_data, 'sl'] = close_price + (atr_val * sl_atr_mult)
            signals_df.loc[signals_df['entry_short'] & valid_sltp_data, 'tp'] = close_price - (atr_val * tp_atr_mult)

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
        latest_signal_info = self.get_signals()
        if latest_signal_info is None or latest_signal_info.empty:
            logger.warning(f"{self.log_prefix} Aucun signal disponible via get_signals(). Pas de requête d'ordre.")
            return None
        
        latest_signals = latest_signal_info.iloc[-1]
        latest_data_row = df_verified_indicators.iloc[-1]

        entry_price_theoretical = latest_data_row['open']
        atr_value_for_sltp = latest_data_row.get(self.atr_col_strat)

        if pd.isna(entry_price_theoretical) or pd.isna(atr_value_for_sltp) or atr_value_for_sltp <= 1e-9:
            return None

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))
        
        side: Optional[str] = None
        sl_price_raw: Optional[float] = None
        tp_price_raw: Optional[float] = None

        if latest_signals.get('entry_long', False):
            side = 'BUY'
            sl_price_raw = entry_price_theoretical - (atr_value_for_sltp * sl_atr_mult)
            tp_price_raw = entry_price_theoretical + (atr_value_for_sltp * tp_atr_mult)
        elif self.allow_shorting and latest_signals.get('entry_short', False):
            side = 'SELL'
            sl_price_raw = entry_price_theoretical + (atr_value_for_sltp * sl_atr_mult)
            tp_price_raw = entry_price_theoretical - (atr_value_for_sltp * tp_atr_mult)

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
            
            entry_order_params = self._build_entry_params_formatted(
                symbol=symbol, side=side, quantity_str=quantity_str,
                entry_price_str=entry_price_str if self.order_type_preference == "LIMIT" else None,
                order_type=self.order_type_preference
            )
            if not entry_order_params: return None
            sl_tp_raw_prices = {'sl_price': sl_price_raw, 'tp_price': tp_price_raw}
            return entry_order_params, sl_tp_raw_prices
        return None
