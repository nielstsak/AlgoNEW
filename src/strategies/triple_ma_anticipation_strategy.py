import logging
import math 
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.strategies.base import BaseStrategy
from src.utils.exchange_utils import (adjust_precision,
                                      get_precision_from_filter,
                                      get_filter_value) # Ajouté
# from src.data import data_utils # Importé localement si besoin

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
        'capital_allocation_pct' # Ajouté
        # 'anticipation_slope_period', # Conditionnellement requis
        # 'anticipation_convergence_threshold_pct' # Conditionnellement requis
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        super().__init__(strategy_name, symbol, params)
        self.log_prefix = f"[{self.strategy_name}][{self.symbol}]"
        
        self.ma_short_col_strat = "MA_SHORT_strat"
        self.ma_medium_col_strat = "MA_MEDIUM_strat"
        self.ma_long_col_strat = "MA_LONG_strat"
        self.atr_col_strat = "ATR_strat"

        self.allow_shorting = bool(self.get_param('allow_shorting', False))
        # order_type_preference est déjà dans REQUIRED_PARAMS et géré par get_param
        self.anticipate_crossovers = bool(self.get_param('anticipate_crossovers', False))

        if self.anticipate_crossovers:
            self.slope_ma_short_col_strat = "SLOPE_MA_SHORT_strat"
            self.slope_ma_medium_col_strat = "SLOPE_MA_MEDIUM_strat"
            # La validation de ces paramètres se fait dans _validate_params
            self.anticipation_slope_period = int(self.get_param('anticipation_slope_period', 3)) # Default si non fourni
            self.anticipation_convergence_threshold_pct = float(self.get_param('anticipation_convergence_threshold_pct', 0.005)) # Default
        else:
            self.slope_ma_short_col_strat = None
            self.slope_ma_medium_col_strat = None
            self.anticipation_slope_period = 0 
            self.anticipation_convergence_threshold_pct = 0.0

        logger.info(f"{self.log_prefix} Stratégie initialisée. Anticipation: {self.anticipate_crossovers}. Params: {self.params}")

    def _validate_params(self):
        # Construire la liste des paramètres requis dynamiquement
        current_required_params = self.REQUIRED_PARAMS[:] # Copie
        if self.anticipate_crossovers:
            if 'anticipation_slope_period' not in current_required_params:
                 current_required_params.append('anticipation_slope_period')
            if 'anticipation_convergence_threshold_pct' not in current_required_params:
                 current_required_params.append('anticipation_convergence_threshold_pct')
        
        missing_params = [p for p in current_required_params if self.get_param(p) is None]
        if missing_params:
            raise ValueError(f"{self.log_prefix} Missing required parameters: {', '.join(missing_params)}")
        
        if self.anticipate_crossovers:
            slope_period = self.get_param('anticipation_slope_period')
            if not isinstance(slope_period, int) or slope_period < 2:
                 raise ValueError(f"{self.log_prefix} anticipation_slope_period ({slope_period}) must be an integer >= 2 when anticipate_crossovers is true.")
        
        if not (0 < self.get_param('capital_allocation_pct') <= 1):
            raise ValueError(f"{self.log_prefix} capital_allocation_pct must be > 0 and <= 1.")
        logger.debug(f"{self.log_prefix} Parameters validated successfully.")

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        expected_base_strat_cols = [
            self.ma_short_col_strat, self.ma_medium_col_strat,
            self.ma_long_col_strat, self.atr_col_strat
        ]
        for col_name in expected_base_strat_cols:
            if col_name not in df.columns:
                logger.error(f"{self.log_prefix} Colonne indicateur de base attendue '{col_name}' manquante.")
                df[col_name] = np.nan

        if self.anticipate_crossovers:
            if self.slope_ma_short_col_strat and self.slope_ma_short_col_strat not in df.columns:
                logger.error(f"{self.log_prefix} Colonne pente '{self.slope_ma_short_col_strat}' manquante pour anticipation.")
                df[self.slope_ma_short_col_strat] = np.nan
            if self.slope_ma_medium_col_strat and self.slope_ma_medium_col_strat not in df.columns:
                logger.error(f"{self.log_prefix} Colonne pente '{self.slope_ma_medium_col_strat}' manquante pour anticipation.")
                df[self.slope_ma_medium_col_strat] = np.nan
        
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

        df = data_with_indicators
        latest_row = df.iloc[-1]
        previous_row = df.iloc[-2]

        ma_short = latest_row.get(self.ma_short_col_strat)
        ma_medium = latest_row.get(self.ma_medium_col_strat)
        ma_long = latest_row.get(self.ma_long_col_strat)
        atr_val = latest_row.get(self.atr_col_strat)
        close_price = latest_row.get('close')

        ma_short_prev = previous_row.get(self.ma_short_col_strat)
        ma_medium_prev = previous_row.get(self.ma_medium_col_strat)

        signal = 0
        sl_price = None
        tp_price = None
        
        essential_values = [ma_short, ma_medium, ma_long, atr_val, close_price, ma_short_prev, ma_medium_prev]
        slope_short_val = np.nan
        slope_medium_val = np.nan

        if self.anticipate_crossovers:
            slope_short_val = latest_row.get(self.slope_ma_short_col_strat)
            slope_medium_val = latest_row.get(self.slope_ma_medium_col_strat)
            essential_values.extend([slope_short_val, slope_medium_val])

        if any(pd.isna(val) for val in essential_values):
            logger.debug(f"{self.log_prefix} Valeurs NaN dans les indicateurs/prix à {df.index[-1]}. Signal Hold.")
            return 0, self.get_param("order_type_preference", "MARKET"), None, None, None, self.get_param('capital_allocation_pct', 1.0)

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        actual_long_entry_cross = (ma_short > ma_medium) and (ma_short_prev <= ma_medium_prev)
        actual_long_exit_cross = (ma_short < ma_medium) and (ma_short_prev >= ma_medium_prev)
        actual_short_entry_cross = False
        actual_short_exit_cross = False
        if self.allow_shorting:
            actual_short_entry_cross = (ma_short < ma_medium) and (ma_short_prev >= ma_medium_prev)
            actual_short_exit_cross = (ma_short > ma_medium) and (ma_short_prev <= ma_medium_prev)

        anticipated_long_entry = False
        anticipated_long_exit = False
        anticipated_short_entry = False
        anticipated_short_exit = False

        if self.anticipate_crossovers and pd.notna(slope_short_val) and pd.notna(slope_medium_val):
            convergence_distance = ma_medium * self.anticipation_convergence_threshold_pct # type: ignore
            ma_diff_abs = abs(ma_short - ma_medium) # type: ignore

            is_converging_up = slope_short_val > slope_medium_val
            is_below_and_closing_for_long = (ma_short < ma_medium) and (ma_diff_abs < convergence_distance)
            main_trend_bullish = ma_medium > ma_long
            anticipated_long_entry = is_converging_up and is_below_and_closing_for_long and main_trend_bullish

            is_converging_down_for_exit_long = slope_short_val < slope_medium_val
            is_above_and_closing_for_long_exit = (ma_short > ma_medium) and (ma_diff_abs < convergence_distance)
            anticipated_long_exit = is_converging_down_for_exit_long and is_above_and_closing_for_long_exit

            if self.allow_shorting:
                is_converging_down_for_entry_short = slope_short_val < slope_medium_val
                is_above_and_closing_for_short_entry = (ma_short > ma_medium) and (ma_diff_abs < convergence_distance)
                main_trend_bearish = ma_medium < ma_long
                anticipated_short_entry = is_converging_down_for_entry_short and is_above_and_closing_for_short_entry and main_trend_bearish
                
                is_converging_up_for_exit_short = slope_short_val > slope_medium_val
                is_below_and_closing_for_short_exit = (ma_short < ma_medium) and (ma_diff_abs < convergence_distance)
                anticipated_short_exit = is_converging_up_for_exit_short and is_below_and_closing_for_short_exit
        
        final_entry_long = actual_long_entry_cross or anticipated_long_entry
        final_exit_long = actual_long_exit_cross or anticipated_long_exit
        final_entry_short = actual_short_entry_cross or anticipated_short_entry
        final_exit_short = actual_short_exit_cross or anticipated_short_exit

        if not current_position_open:
            if final_entry_long:
                signal = 1
                if pd.notna(atr_val) and atr_val > 0:
                    sl_price = close_price - (atr_val * sl_atr_mult)
                    tp_price = close_price + (atr_val * tp_atr_mult)
            elif final_entry_short: 
                signal = -1
                if pd.notna(atr_val) and atr_val > 0:
                    sl_price = close_price + (atr_val * sl_atr_mult)
                    tp_price = close_price - (atr_val * tp_atr_mult)
        else: 
            if current_position_direction == 1 and final_exit_long:
                signal = 2 
            elif current_position_direction == -1 and final_exit_short:
                signal = 2
        
        order_type = self.get_param("order_type_preference", "MARKET")
        limit_price = None
        if signal != 0 and order_type == "LIMIT" and pd.notna(close_price):
            limit_price = close_price 

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
        entry_price_theoretical = latest_row.get('close')
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
