import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy # Assurez-vous que BaseStrategy a get_param et _build_entry_params_formatted
from src.utils.exchange_utils import (adjust_precision,
                                      get_precision_from_filter,
                                      get_filter_value, # Ajouté pour _calculate_quantity
                                      validate_order_parameters) # Ajouté pour validation optionnelle
# Il est préférable que _calculate_quantity et _build_entry_params_formatted soient dans BaseStrategy

logger = logging.getLogger(__name__)

class MaCrossoverStrategy(BaseStrategy):
    REQUIRED_PARAMS = [
        'fast_ma_period', 'slow_ma_period', 'ma_type',
        'atr_period_sl_tp', 'sl_atr_multiplier', 'tp_atr_multiplier',
        'indicateur_frequence_ma_rapide', 'indicateur_frequence_ma_lente',
        'atr_base_frequency_sl_tp',
        'capital_allocation_pct', # Ajouté pour _calculate_quantity
        'order_type_preference' # Ajouté pour _build_entry_params_formatted
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        super().__init__(strategy_name, symbol, params) # strategy_name et symbol sont maintenant passés
        self.log_prefix = f"[{self.strategy_name}][{self.symbol}]"

        self.fast_ma_col_strat = "MA_FAST_strat"
        self.slow_ma_col_strat = "MA_SLOW_strat"
        self.atr_col_strat = "ATR_strat"
        logger.info(f"{self.log_prefix} Stratégie initialisée. Paramètres: {self.params}")

    def _validate_params(self):
        missing_params = [p for p in self.REQUIRED_PARAMS if self.get_param(p) is None]
        if missing_params:
            raise ValueError(f"{self.log_prefix} Missing required parameters: {', '.join(missing_params)}")
        
        if self.get_param('fast_ma_period') >= self.get_param('slow_ma_period'):
            logger.warning(f"{self.log_prefix} fast_ma_period ({self.get_param('fast_ma_period')}) "
                           f"should ideally be less than slow_ma_period ({self.get_param('slow_ma_period')}).")
        
        if not (0 < self.get_param('capital_allocation_pct') <= 1):
            raise ValueError(f"{self.log_prefix} capital_allocation_pct must be > 0 and <= 1.")
        logger.debug(f"{self.log_prefix} Parameters validated successfully.")

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Cette méthode est appelée par BaseStrategy.get_signal et par generate_order_request.
        # Elle doit s'assurer que les colonnes _strat sont présentes.
        # En mode backtest, ObjectiveEvaluator les prépare.
        # En mode live, LiveTradingManager doit s'assurer que le DataFrame passé à generate_order_request
        # contient ces colonnes (via preprocessing_live.py).
        df = data.copy()
        expected_strat_cols = [self.fast_ma_col_strat, self.slow_ma_col_strat, self.atr_col_strat]
        for col_name in expected_strat_cols:
            if col_name not in df.columns:
                logger.error(f"{self.log_prefix} Colonne indicateur attendue '{col_name}' manquante dans les données d'entrée.")
                # En mode live, cela pourrait être critique. En backtest, c'est un bug dans la préparation.
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
            logger.debug(f"{self.log_prefix} Pas assez de données ({len(data_with_indicators)}) pour générer des signaux.")
            return 0, self.get_param("order_type_preference", "MARKET"), None, None, None, self.get_param('capital_allocation_pct', 1.0)

        latest_row = data_with_indicators.iloc[-1]
        previous_row = data_with_indicators.iloc[-2]

        ma_fast_curr = latest_row.get(self.fast_ma_col_strat)
        ma_slow_curr = latest_row.get(self.slow_ma_col_strat)
        ma_fast_prev = previous_row.get(self.fast_ma_col_strat)
        ma_slow_prev = previous_row.get(self.slow_ma_col_strat)
        
        close_curr = latest_row.get('close')
        atr_curr = latest_row.get(self.atr_col_strat)

        signal = 0
        sl_price = None
        tp_price = None
        
        essential_values = [ma_fast_curr, ma_slow_curr, ma_fast_prev, ma_slow_prev, atr_curr, close_curr]
        if any(pd.isna(val) for val in essential_values):
            logger.debug(f"{self.log_prefix} Valeurs NaN dans les indicateurs/prix à {data_with_indicators.index[-1]}. Signal Hold.")
            return 0, self.get_param("order_type_preference", "MARKET"), None, None, None, self.get_param('capital_allocation_pct', 1.0)

        sl_mult = float(self.get_param('sl_atr_multiplier'))
        tp_mult = float(self.get_param('tp_atr_multiplier'))

        bullish_crossover = (ma_fast_curr > ma_slow_curr) and (ma_fast_prev <= ma_slow_prev)
        bearish_crossover = (ma_fast_curr < ma_slow_curr) and (ma_fast_prev >= ma_slow_prev)

        if not current_position_open:
            if bullish_crossover:
                signal = 1
                if pd.notna(atr_curr) and atr_curr > 0:
                    sl_price = close_curr - sl_mult * atr_curr
                    tp_price = close_curr + tp_mult * atr_curr
            elif bearish_crossover:
                signal = -1
                if pd.notna(atr_curr) and atr_curr > 0:
                    sl_price = close_curr + sl_mult * atr_curr
                    tp_price = close_curr - tp_mult * atr_curr
        else:
            if current_position_direction == 1 and bearish_crossover:
                signal = 2 # Exit long
            elif current_position_direction == -1 and bullish_crossover:
                signal = 2 # Exit short
        
        order_type = self.get_param('order_type_preference', "MARKET")
        limit_price = None
        if signal != 0 and order_type == "LIMIT" and pd.notna(close_curr):
            limit_price = close_curr 

        position_size_pct = self.get_param('capital_allocation_pct', 1.0)

        return signal, order_type, limit_price, sl_price, tp_price, position_size_pct

    def generate_order_request(self,
                               data: pd.DataFrame, # Doit contenir les indicateurs _strat
                               current_position: int, # 0: no pos, 1: long, -1: short
                               available_capital: float, # En actif de cotation (ex: USDC)
                               symbol_info: dict # Infos de l'exchange pour la paire
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        
        # 1. S'assurer que les indicateurs sont présents (appel _calculate_indicators)
        data_with_indicators = self._calculate_indicators(data)
        if data_with_indicators.empty or len(data_with_indicators) < 2:
            logger.warning(f"{self.log_prefix} [Live] Pas assez de données pour generate_order_request.")
            return None

        # 2. Utiliser la logique de _generate_signals pour obtenir le signal de base et SL/TP
        # Convertir current_position en bool et direction pour _generate_signals
        is_pos_open = current_position != 0
        pos_direction = current_position # 1 pour long, -1 pour short, 0 pour none
        
        # current_entry_price n'est pas directement pertinent pour décider d'une *nouvelle* entrée,
        # mais _generate_signals le prend. Pour une nouvelle entrée, on peut passer 0.0
        # Si on est déjà en position, cette méthode ne devrait pas générer un nouvel ordre d'ENTRÉE.
        if is_pos_open:
             logger.debug(f"{self.log_prefix} [Live] Position déjà ouverte. generate_order_request ne génère pas de nouvel ordre d'entrée.")
             return None # Ne génère pas de NOUVEL ordre d'entrée si déjà en position.

        # Pour une nouvelle entrée, current_entry_price n'est pas encore défini.
        # _generate_signals est conçu pour le backtest où on a cette info.
        # On adapte : on ne s'attend qu'à des signaux d'entrée (1 ou -1) ici.
        
        signal, order_type, limit_price, sl_price_raw, tp_price_raw, pos_size_pct = \
            self._generate_signals(data_with_indicators, False, 0, 0.0)

        if signal not in [1, -1]: # Pas de signal d'entrée
            logger.debug(f"{self.log_prefix} [Live] Aucun signal d'entrée généré.")
            return None

        # 3. Déterminer le prix d'entrée théorique
        latest_row = data_with_indicators.iloc[-1]
        entry_price_theoretical = latest_row.get('close') # Utiliser le close pour le calcul de quantité
        if order_type == "LIMIT" and limit_price is not None:
            entry_price_theoretical = limit_price # Si ordre limite, utiliser ce prix pour calcul de quantité
        
        if pd.isna(entry_price_theoretical):
            logger.error(f"{self.log_prefix} [Live] Prix d'entrée théorique (close ou limit) est NaN.")
            return None

        # 4. Calculer la quantité
        # _calculate_quantity est une méthode de BaseStrategy
        quantity_base = self._calculate_quantity(
            entry_price=entry_price_theoretical,
            available_capital=available_capital,
            # qty_precision est dans symbol_info ou self.pair_config
            # symbol_info est le pair_config
            qty_precision=get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize'),
            symbol_info=symbol_info,
            symbol=self.symbol,
            position_size_pct=pos_size_pct
        )

        if quantity_base is None or quantity_base <= 0:
            logger.warning(f"{self.log_prefix} [Live] Quantité calculée est None ou <= 0 ({quantity_base}).")
            return None

        # 5. Formater les paramètres de l'ordre
        price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
        if price_precision is None:
            logger.error(f"{self.log_prefix} [Live] Impossible d'obtenir price_precision.")
            return None
        
        entry_price_for_order_str: Optional[str] = None
        if order_type == "LIMIT" and limit_price is not None:
            adjusted_limit_price = adjust_precision(limit_price, price_precision)
            if adjusted_limit_price is None: return None
            entry_price_for_order_str = f"{adjusted_limit_price:.{price_precision}f}"
        
        # _build_entry_params_formatted est une méthode de BaseStrategy
        entry_order_params = self._build_entry_params_formatted(
            side="BUY" if signal == 1 else "SELL",
            quantity_str=f"{quantity_base:.{get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize') or 8}f}", # Utiliser la précision de quantité
            order_type=order_type, # type: ignore
            entry_price_str=entry_price_for_order_str
        )
        if not entry_order_params:
            logger.error(f"{self.log_prefix} [Live] Échec de la construction des paramètres d'ordre.")
            return None

        sl_tp_prices_for_live = {}
        if sl_price_raw is not None: sl_tp_prices_for_live['sl_price'] = sl_price_raw
        if tp_price_raw is not None: sl_tp_prices_for_live['tp_price'] = tp_price_raw
        
        logger.info(f"{self.log_prefix} [Live] Requête d'ordre générée: {entry_order_params}, SL/TP bruts: {sl_tp_prices_for_live}")
        return entry_order_params, sl_tp_prices_for_live
