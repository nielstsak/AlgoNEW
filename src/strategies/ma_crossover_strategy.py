# src/strategies/ma_crossover_strategy.py
"""
Stratégie de trading basée sur le croisement de deux moyennes mobiles (MA),
avec gestion du Stop-Loss (SL) et Take-Profit (TP) basée sur l'Average True Range (ATR).
"""
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd # Assurer l'import pour Pylance

from src.strategies.base import BaseStrategy
from src.data.data_utils import get_kline_prefix_effective 
from src.utils.exchange_utils import adjust_precision, get_filter_value


logger = logging.getLogger(__name__)

class MaCrossoverStrategy(BaseStrategy):
    """
    Stratégie de croisement de moyennes mobiles.
    """

    REQUIRED_PARAMS: List[str] = [
        'fast_ma_period', 'slow_ma_period', 'ma_type',
        'atr_period_sl_tp', 'sl_atr_multiplier', 'tp_atr_multiplier',
        'indicateur_frequence_ma_rapide', 'indicateur_frequence_ma_lente',
        'atr_base_frequency_sl_tp',
        'capital_allocation_pct',
        'order_type_preference',
        'margin_leverage' 
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        super().__init__(strategy_name, symbol, params)
        self.fast_ma_col_strat: str = "MA_FAST_strat"
        self.slow_ma_col_strat: str = "MA_SLOW_strat"
        self.atr_col_strat: str = "ATR_strat"

    def _validate_params(self) -> None:
        """Valide les paramètres spécifiques à MaCrossoverStrategy."""
        # Sourcery: Use named expression (walrus operator)
        if missing_params := [
            p for p in self.REQUIRED_PARAMS if self.get_param(p) is None
        ]:
            raise ValueError(f"{self.log_prefix} Paramètres requis manquants : {', '.join(missing_params)}")

        fast_ma_p = self.get_param('fast_ma_period')
        slow_ma_p = self.get_param('slow_ma_period')
        if not (isinstance(fast_ma_p, int) and fast_ma_p > 0):
            raise ValueError(f"{self.log_prefix} 'fast_ma_period' ({fast_ma_p}) doit être un entier positif.")
        if not (isinstance(slow_ma_p, int) and slow_ma_p > 0):
            raise ValueError(f"{self.log_prefix} 'slow_ma_period' ({slow_ma_p}) doit être un entier positif.")
        if fast_ma_p >= slow_ma_p:
            raise ValueError(f"{self.log_prefix} 'fast_ma_period' ({fast_ma_p}) doit être inférieur à 'slow_ma_period' ({slow_ma_p}).")

        ma_type_val = self.get_param('ma_type')
        supported_ma_types = ['sma', 'ema', 'wma', 'hma', 'tema', 'dema', 'rma', 'vwma', 'zlma'] 
        if not isinstance(ma_type_val, str) or ma_type_val.lower() not in supported_ma_types:
            raise ValueError(f"{self.log_prefix} 'ma_type' ({ma_type_val}) invalide. Supportés : {supported_ma_types}")

        atr_p = self.get_param('atr_period_sl_tp')
        if not (isinstance(atr_p, int) and atr_p > 0):
            raise ValueError(f"{self.log_prefix} 'atr_period_sl_tp' ({atr_p}) doit être un entier positif.")

        sl_mult = self.get_param('sl_atr_multiplier')
        tp_mult = self.get_param('tp_atr_multiplier')
        if not (isinstance(sl_mult, (int, float)) and sl_mult > 0):
            raise ValueError(f"{self.log_prefix} 'sl_atr_multiplier' ({sl_mult}) doit être un nombre positif.")
        if not (isinstance(tp_mult, (int, float)) and tp_mult > 0):
            raise ValueError(f"{self.log_prefix} 'tp_atr_multiplier' ({tp_mult}) doit être un nombre positif.")

        cap_alloc = self.get_param('capital_allocation_pct')
        if not (isinstance(cap_alloc, (int, float)) and 0 < cap_alloc <= 1.0):
            raise ValueError(f"{self.log_prefix} 'capital_allocation_pct' ({cap_alloc}) doit être entre 0 (exclusif) et 1 (inclusif).")
        
        margin_lev = self.get_param('margin_leverage')
        if not (isinstance(margin_lev, (int, float)) and margin_lev >= 1.0):
            raise ValueError(f"{self.log_prefix} 'margin_leverage' ({margin_lev}) doit être >= 1.0.")

        order_type_pref_val = self.get_param('order_type_preference')
        if order_type_pref_val not in ["MARKET", "LIMIT"]:
            raise ValueError(f"{self.log_prefix} 'order_type_preference' ({order_type_pref_val}) doit être 'MARKET' ou 'LIMIT'.")
        
        for freq_param_name in ['indicateur_frequence_ma_rapide', 'indicateur_frequence_ma_lente', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' ("
                                 f"{freq_val}) doit être une chaîne de caractères non vide.")
        
        logger.debug(f"{self.log_prefix} Validation des paramètres terminée avec succès.")

    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        freq_ma_rapide_param = str(self.params['indicateur_frequence_ma_rapide'])
        kline_prefix_ma_rapide = get_kline_prefix_effective(freq_ma_rapide_param)
        source_col_close_ma_rapide = f"{kline_prefix_ma_rapide}_close" if kline_prefix_ma_rapide else "close"

        freq_ma_lente_param = str(self.params['indicateur_frequence_ma_lente'])
        kline_prefix_ma_lente = get_kline_prefix_effective(freq_ma_lente_param)
        source_col_close_ma_lente = f"{kline_prefix_ma_lente}_close" if kline_prefix_ma_lente else "close"

        freq_atr_param = str(self.params['atr_base_frequency_sl_tp'])
        kline_prefix_atr = get_kline_prefix_effective(freq_atr_param)
        source_col_high_atr = f"{kline_prefix_atr}_high" if kline_prefix_atr else "high"
        source_col_low_atr = f"{kline_prefix_atr}_low" if kline_prefix_atr else "low"
        source_col_close_atr = f"{kline_prefix_atr}_close" if kline_prefix_atr else "close"

        configs = [
            {
                'indicator_name': str(self.params['ma_type']).lower(),
                'params': {'length': int(self.params['fast_ma_period'])},
                'inputs': {'close': source_col_close_ma_rapide},
                'outputs': self.fast_ma_col_strat
            },
            {
                'indicator_name': str(self.params['ma_type']).lower(),
                'params': {'length': int(self.params['slow_ma_period'])},
                'inputs': {'close': source_col_close_ma_lente},
                'outputs': self.slow_ma_col_strat
            },
            {
                'indicator_name': 'atr',
                'params': {'length': int(self.params['atr_period_sl_tp'])},
                'inputs': { 
                    'high': source_col_high_atr,
                    'low': source_col_low_atr,
                    'close': source_col_close_atr
                },
                'outputs': self.atr_col_strat
            }
        ]
        logger.debug(f"{self.log_prefix} Configurations d'indicateurs requises (standardisées) : {configs}")
        return configs

    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        df = data_feed.copy()
        expected_strat_cols = [self.fast_ma_col_strat, self.slow_ma_col_strat, self.atr_col_strat]
        
        base_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        # Sourcery: Use named expression (walrus operator)
        if missing_ohlcv := [
            col for col in base_ohlcv if col not in df.columns
        ]:
            msg = f"{self.log_prefix} Colonnes OHLCV de base manquantes dans data_feed: {missing_ohlcv}. Ces colonnes sont essentielles."
            logger.critical(msg)
            raise ValueError(msg)

        # Sourcery: Use named expression (walrus operator)
        if missing_strat_cols_initially := [
            col_name
            for col_name in expected_strat_cols
            if col_name not in df.columns
        ]:
            for col_name in missing_strat_cols_initially:
                logger.warning(f"{self.log_prefix} Colonne indicateur attendue '{col_name}' manquante dans data_feed. "
                               "Elle sera ajoutée avec NaN (problème en amont attendu: IndicatorCalculator).")
                df[col_name] = np.nan
            logger.debug(f"{self.log_prefix} Après vérification des colonnes dans _calculate_indicators, "
                          f"colonnes manquantes initialement ajoutées (avec NaN): {missing_strat_cols_initially}. "
                          f"Colonnes actuelles dans le DataFrame de travail: {df.columns.tolist()}")

        cols_to_ffill_present = [col for col in expected_strat_cols if col in df.columns]
        if cols_to_ffill_present:
            if needs_ffill_check := df[cols_to_ffill_present].isnull().values.any():
                logger.debug(f"{self.log_prefix} Application de ffill aux colonnes _strat: {cols_to_ffill_present}")
            df[cols_to_ffill_present] = df[cols_to_ffill_present].ffill()
        else:
            logger.debug(f"{self.log_prefix} Aucune colonne _strat à ffill.")

        return df

    def _generate_signals(self,
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int,
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        signal_type: int = 0 
        limit_price: Optional[float] = None
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        if len(data_with_indicators) < 2: 
            logger.debug(f"{self.log_prefix} Pas assez de données ({len(data_with_indicators)}) pour générer des signaux.")
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        latest_row = data_with_indicators.iloc[-1]
        previous_row = data_with_indicators.iloc[-2]

        ma_fast_curr = latest_row.get(self.fast_ma_col_strat)
        ma_slow_curr = latest_row.get(self.slow_ma_col_strat)
        ma_fast_prev = previous_row.get(self.fast_ma_col_strat)
        ma_slow_prev = previous_row.get(self.slow_ma_col_strat)
        close_price_curr = latest_row.get('close') 
        atr_value_curr = latest_row.get(self.atr_col_strat)

        essential_values = [ma_fast_curr, ma_slow_curr, ma_fast_prev, ma_slow_prev, close_price_curr, atr_value_curr]
        if any(pd.isna(val) for val in essential_values):
            nan_details = { 
                "ma_fast_c": ma_fast_curr, "ma_slow_c": ma_slow_curr, 
                "ma_fast_p": ma_fast_prev, "ma_slow_p": ma_slow_prev, 
                "close_c": close_price_curr, "atr_c": atr_value_curr 
            }
            logger.debug(f"{self.log_prefix} Valeurs d'indicateur/prix manquantes (NaN) à {latest_row.name}. Détails: {nan_details}. Signal HOLD.")
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        bullish_crossover = (ma_fast_curr > ma_slow_curr) and (ma_fast_prev <= ma_slow_prev) # type: ignore
        bearish_crossover = (ma_fast_curr < ma_slow_curr) and (ma_fast_prev >= ma_slow_prev) # type: ignore

        sl_multiplier = float(self.get_param('sl_atr_multiplier'))
        tp_multiplier = float(self.get_param('tp_atr_multiplier'))

        # Sourcery: Swap if/else branches, Merge else clause's nested if statement into elif, Remove redundant pass statement
        if current_position_open:
            if current_position_direction == 1 and bearish_crossover:
                signal_type = 2 
                logger.info(f"{self.log_prefix} Signal EXIT LONG (croisement baissier) @ {close_price_curr:.4f}")
        elif bullish_crossover:
            signal_type = 1
            if atr_value_curr > 0: # type: ignore
                sl_price = close_price_curr - (atr_value_curr * sl_multiplier) # type: ignore
                tp_price = close_price_curr + (atr_value_curr * tp_multiplier) # type: ignore
            # Sourcery: Replace if-expression with `or`
            logger.info(f"{self.log_prefix} Signal BUY @ {close_price_curr:.4f}. SL={sl_price}, TP={tp_price or 'N/A'}")
        elif bearish_crossover: 
            logger.debug(f"{self.log_prefix} Croisement baissier détecté, mais pas de position ouverte. Shorting non explicitement géré pour entrée. Signal HOLD.")
        
        order_type_preference = str(self.get_param("order_type_preference", "MARKET"))
        if signal_type != 0 and order_type_preference == "LIMIT":
            limit_price = float(close_price_curr) # type: ignore

        position_size_pct = float(self.get_param('capital_allocation_pct', 1.0))

        return signal_type, order_type_preference, limit_price, sl_price, tp_price, position_size_pct

    def generate_order_request(self,
                               data: pd.DataFrame,
                               current_position: int, 
                               available_capital: float, 
                               symbol_info: Dict[str, Any] 
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        if current_position != 0:
            logger.debug(f"{self.log_prefix} [Live] Position déjà ouverte ({current_position}). Pas de nouvelle requête d'ordre d'entrée.")
            return None

        data_with_indicators = self._calculate_indicators(data.copy())
        if data_with_indicators.empty or len(data_with_indicators) < 2:
            logger.warning(f"{self.log_prefix} [Live] Données insuffisantes après _calculate_indicators pour generate_order_request. Shape: {data_with_indicators.shape}")
            return None

        signal, order_type, limit_price_sugg, sl_price_raw, tp_price_raw, pos_size_pct = \
            self._generate_signals(data_with_indicators, False, 0, 0.0) 

        if signal != 1: # Cette stratégie se concentre sur les entrées LONG
            logger.debug(f"{self.log_prefix} [Live] Aucun signal d'entrée LONG (1) généré. Signal actuel: {signal}")
            return None
        
        latest_bar = data_with_indicators.iloc[-1]
        entry_price_theoretical: float = limit_price_sugg if order_type == "LIMIT" and limit_price_sugg is not None else float(latest_bar.get('close', 0.0))
        
        if entry_price_theoretical <= 0: 
            entry_price_theoretical = float(latest_bar.get('open', 0.0))
        
        if pd.isna(entry_price_theoretical) or entry_price_theoretical <= 0:
            logger.error(f"{self.log_prefix} [Live] Prix d'entrée théorique pour calcul de quantité invalide ({entry_price_theoretical}).")
            return None

        if self.quantity_precision is None or self.pair_config is None: 
            logger.error(f"{self.log_prefix} [Live] Contexte (précisions, pair_config) non défini via set_backtest_context.")
            return None
            
        quantity_base = self._calculate_quantity(
            entry_price=entry_price_theoretical,
            available_capital=available_capital,
            qty_precision=self.quantity_precision,
            symbol_info=self.pair_config, 
            symbol=self.symbol,
            position_size_pct=pos_size_pct
        )

        if quantity_base is None or quantity_base <= 1e-9:
            logger.warning(f"{self.log_prefix} [Live] Quantité calculée nulle ou invalide ({quantity_base}). Pas d'ordre.")
            return None

        if self.price_precision is None: 
            logger.error(f"{self.log_prefix} [Live] price_precision non défini.")
            return None

        entry_price_for_order_str: Optional[str] = None
        if order_type == "LIMIT" and limit_price_sugg is not None:
            tick_size_price = get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize')
            if (adjusted_limit_price := adjust_precision(
                limit_price_sugg, self.price_precision, tick_size=tick_size_price
            )) is None:
                return None
            entry_price_for_order_str = f"{adjusted_limit_price:.{self.price_precision}f}"
        
        quantity_str_formatted = f"{quantity_base:.{self.quantity_precision}f}"

        entry_order_params = self._build_entry_params_formatted(
            side="BUY", # Toujours BUY pour cette stratégie d'entrée
            quantity_str=quantity_str_formatted,
            order_type=str(order_type), 
            entry_price_str=entry_price_for_order_str
        )
        if not entry_order_params: 
            logger.error(f"{self.log_prefix} [Live] Échec de la construction des paramètres de l'ordre d'entrée.")
            return None

        sl_tp_raw_prices_dict: Dict[str, float] = {}
        if sl_price_raw is not None: sl_tp_raw_prices_dict['sl_price'] = float(sl_price_raw)
        if tp_price_raw is not None: sl_tp_raw_prices_dict['tp_price'] = float(tp_price_raw)
        
        logger.info(f"{self.log_prefix} [Live] Requête d'ordre d'entrée générée : {entry_order_params}. SL/TP bruts : {sl_tp_raw_prices_dict}")
        return entry_order_params, sl_tp_raw_prices_dict
