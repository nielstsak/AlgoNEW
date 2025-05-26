# src/strategies/bbands_volume_rsi_strategy.py
"""
Stratégie de trading utilisant les Bandes de Bollinger (BBands), le volume,
et le Relative Strength Index (RSI) pour identifier les signaux d'entrée.
Les Stop-Loss (SL) et Take-Profit (TP) sont basés sur l'Average True Range (ATR).
"""
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.data.data_utils import get_kline_prefix_effective # Pour déterminer la colonne source du volume

logger = logging.getLogger(__name__)

class BbandsVolumeRsiStrategy(BaseStrategy):
    """
    Stratégie de trading combinant Bandes de Bollinger, Volume et RSI.
    """

    REQUIRED_PARAMS: List[str] = [
        'bbands_period', 'bbands_std_dev', 'indicateur_frequence_bbands',
        'volume_ma_period', 'indicateur_frequence_volume', 
        'rsi_period', 'indicateur_frequence_rsi',
        'rsi_buy_breakout_threshold', 'rsi_sell_breakout_threshold',
        'atr_period_sl_tp', 'atr_base_frequency_sl_tp',
        'sl_atr_mult', 'tp_atr_mult',
        'capital_allocation_pct',
        'order_type_preference',
        'margin_leverage' # Ajouté pour cohérence
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        super().__init__(strategy_name, symbol, params)

        self.bb_upper_col_strat: str = "BB_UPPER_strat"
        self.bb_middle_col_strat: str = "BB_MIDDLE_strat"
        self.bb_lower_col_strat: str = "BB_LOWER_strat"
        self.bb_bandwidth_col_strat: str = "BB_BANDWIDTH_strat"
        
        self.volume_ma_col_strat: str = "Volume_MA_strat"
        self.rsi_col_strat: str = "RSI_strat"
        self.atr_col_strat: str = "ATR_strat"

        vol_freq_param = self.get_param('indicateur_frequence_volume')
        kline_prefix_vol_src = get_kline_prefix_effective(str(vol_freq_param))
        self.volume_kline_col_source: str = f"{kline_prefix_vol_src}_volume" if kline_prefix_vol_src else "volume"
        
        logger.info(f"{self.log_prefix} Colonne source de volume pour la MA de volume (stratégie) : '{self.volume_kline_col_source}'")


    def _validate_params(self) -> None:
        missing_params = [p for p in self.REQUIRED_PARAMS if self.get_param(p) is None]
        if missing_params:
            raise ValueError(f"{self.log_prefix} Paramètres requis manquants : {', '.join(missing_params)}")

        if not (isinstance(self.get_param('bbands_period'), int) and self.get_param('bbands_period') > 0):
            raise ValueError(f"{self.log_prefix} 'bbands_period' doit être un entier positif.")
        if not (isinstance(self.get_param('bbands_std_dev'), (int,float)) and self.get_param('bbands_std_dev') > 0):
            raise ValueError(f"{self.log_prefix} 'bbands_std_dev' doit être un nombre positif.")
        if not (isinstance(self.get_param('volume_ma_period'), int) and self.get_param('volume_ma_period') > 0):
            raise ValueError(f"{self.log_prefix} 'volume_ma_period' doit être un entier positif.")
        if not (isinstance(self.get_param('rsi_period'), int) and self.get_param('rsi_period') > 0):
            raise ValueError(f"{self.log_prefix} 'rsi_period' doit être un entier positif.")
        
        rsi_buy_thresh = self.get_param('rsi_buy_breakout_threshold')
        rsi_sell_thresh = self.get_param('rsi_sell_breakout_threshold')
        if not (isinstance(rsi_buy_thresh, (int,float)) and 50 < rsi_buy_thresh < 100):
            raise ValueError(f"{self.log_prefix} 'rsi_buy_breakout_threshold' ({rsi_buy_thresh}) doit être entre 50 et 100.")
        if not (isinstance(rsi_sell_thresh, (int,float)) and 0 < rsi_sell_thresh < 50):
            raise ValueError(f"{self.log_prefix} 'rsi_sell_breakout_threshold' ({rsi_sell_thresh}) doit être entre 0 et 50.")
        if rsi_sell_thresh >= rsi_buy_thresh:
            raise ValueError(f"{self.log_prefix} 'rsi_sell_breakout_threshold' ({rsi_sell_thresh}) doit être inférieur à 'rsi_buy_breakout_threshold' ({rsi_buy_thresh}).")

        atr_p = self.get_param('atr_period_sl_tp')
        if not (isinstance(atr_p, int) and atr_p > 0):
            raise ValueError(f"{self.log_prefix} 'atr_period_sl_tp' doit être un entier positif.")
        sl_mult = self.get_param('sl_atr_mult')
        tp_mult = self.get_param('tp_atr_mult')
        if not (isinstance(sl_mult, (int, float)) and sl_mult > 0):
            raise ValueError(f"{self.log_prefix} 'sl_atr_mult' doit être un nombre positif.")
        if not (isinstance(tp_mult, (int, float)) and tp_mult > 0):
            raise ValueError(f"{self.log_prefix} 'tp_atr_mult' doit être un nombre positif.")

        cap_alloc = self.get_param('capital_allocation_pct')
        if not (isinstance(cap_alloc, (int, float)) and 0 < cap_alloc <= 1.0):
            raise ValueError(f"{self.log_prefix} 'capital_allocation_pct' doit être entre 0 (exclusif) et 1 (inclusif).")
        
        margin_lev = self.get_param('margin_leverage')
        if not (isinstance(margin_lev, (int, float)) and margin_lev >= 1.0):
            raise ValueError(f"{self.log_prefix} 'margin_leverage' ({margin_lev}) doit être >= 1.0.")

        order_type_pref_val = self.get_param('order_type_preference')
        if order_type_pref_val not in ["MARKET", "LIMIT"]:
            raise ValueError(f"{self.log_prefix} 'order_type_preference' doit être 'MARKET' ou 'LIMIT'.")

        for freq_param_name in ['indicateur_frequence_bbands', 'indicateur_frequence_volume', 'indicateur_frequence_rsi', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' doit être une chaîne non vide.")
        logger.debug(f"{self.log_prefix} Validation des paramètres terminée avec succès.")

    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        bbands_params = {'length': int(self.params['bbands_period']), 'std': float(self.params['bbands_std_dev'])}
        
        freq_bbands = str(self.params['indicateur_frequence_bbands'])
        kline_prefix_bbands = get_kline_prefix_effective(freq_bbands)
        source_col_close_bbands = f"{kline_prefix_bbands}_close" if kline_prefix_bbands else "close"
        
        freq_rsi = str(self.params['indicateur_frequence_rsi'])
        kline_prefix_rsi = get_kline_prefix_effective(freq_rsi)
        source_col_close_rsi = f"{kline_prefix_rsi}_close" if kline_prefix_rsi else "close"

        freq_atr = str(self.params['atr_base_frequency_sl_tp'])
        kline_prefix_atr = get_kline_prefix_effective(freq_atr)
        source_col_high_atr = f"{kline_prefix_atr}_high" if kline_prefix_atr else "high"
        source_col_low_atr = f"{kline_prefix_atr}_low" if kline_prefix_atr else "low"
        source_col_close_atr = f"{kline_prefix_atr}_close" if kline_prefix_atr else "close"

        configs = [
            {
                'indicator_name': 'bbands',
                'params': bbands_params,
                'inputs': {'close': source_col_close_bbands},
                'outputs': { 
                    'lower': self.bb_lower_col_strat,   
                    'middle': self.bb_middle_col_strat, 
                    'upper': self.bb_upper_col_strat,   
                    'bandwidth': self.bb_bandwidth_col_strat 
                }
            },
            {
                'indicator_name': 'sma', 
                'params': {'length': int(self.params['volume_ma_period'])},
                'inputs': {'close': self.volume_kline_col_source}, 
                'outputs': self.volume_ma_col_strat
            },
            {
                'indicator_name': 'rsi',
                'params': {'length': int(self.params['rsi_period'])},
                'inputs': {'close': source_col_close_rsi},
                'outputs': self.rsi_col_strat
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
        """
        Vérifie la présence des colonnes d'indicateurs _strat attendues (fournies par
        IndicatorCalculator) et de la colonne source pour la MA de volume.
        Applique ffill pour propager les valeurs.
        """
        df = data_feed.copy()
        expected_strat_cols = [
            self.bb_upper_col_strat, self.bb_middle_col_strat, self.bb_lower_col_strat,
            self.bb_bandwidth_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat
        ]
        # La colonne source pour la MA de volume doit aussi être présente dans data_feed
        # car IndicatorCalculator l'utilise pour calculer self.volume_ma_col_strat,
        # mais la stratégie elle-même utilise self.volume_kline_col_source pour la comparaison directe.
        all_expected_cols_in_data_feed = expected_strat_cols + [self.volume_kline_col_source]
        
        # Vérifier les colonnes OHLCV de base
        base_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlcv = [col for col in base_ohlcv if col not in df.columns]
        if missing_ohlcv:
            msg = f"{self.log_prefix} Colonnes OHLCV de base manquantes dans data_feed: {missing_ohlcv}."
            logger.critical(msg)
            raise ValueError(msg)

        # Vérifier les colonnes d'indicateurs _strat et la source de volume
        missing_cols = []
        for col_name in all_expected_cols_in_data_feed:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne attendue '{col_name}' manquante dans data_feed. "
                               "Elle sera ajoutée avec NaN, indiquant un problème en amont.")
                df[col_name] = np.nan
                missing_cols.append(col_name)
        
        if missing_cols:
             logger.debug(f"{self.log_prefix} Après vérification _calculate_indicators, "
                          f"colonnes manquantes ajoutées (NaN): {missing_cols}. "
                          f"Colonnes actuelles: {df.columns.tolist()}")

        # Appliquer ffill seulement aux colonnes _strat qui sont censées être des indicateurs
        cols_to_ffill_present = [col for col in expected_strat_cols if col in df.columns]
        if cols_to_ffill_present:
            df[cols_to_ffill_present] = df[cols_to_ffill_present].ffill()
            logger.debug(f"{self.log_prefix} ffill appliqué aux colonnes _strat présentes : {cols_to_ffill_present}")
        
        # La colonne self.volume_kline_col_source ne doit pas être ffillée ici si elle représente un volume brut.
        # Elle est déjà supposée être correcte en sortie de IndicatorCalculator (ou df_source_enriched).

        return df

    def _generate_signals(self,
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int,
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Génère les signaux de trading pour la stratégie BbandsVolumeRsiStrategy.
        """
        signal_type: int = 0
        limit_price: Optional[float] = None
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        if len(data_with_indicators) < 2: 
            logger.debug(f"{self.log_prefix} Pas assez de données ({len(data_with_indicators)}) pour générer des signaux.")
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        latest_row = data_with_indicators.iloc[-1]
        previous_row = data_with_indicators.iloc[-2]

        close_curr = latest_row.get('close')
        bb_upper_curr = latest_row.get(self.bb_upper_col_strat)
        bb_lower_curr = latest_row.get(self.bb_lower_col_strat)
        volume_kline_curr = latest_row.get(self.volume_kline_col_source) # Volume brut de la kline source
        volume_ma_curr = latest_row.get(self.volume_ma_col_strat) # MA du volume
        rsi_curr = latest_row.get(self.rsi_col_strat)
        atr_curr = latest_row.get(self.atr_col_strat)
        
        close_prev = previous_row.get('close')
        bb_upper_prev = previous_row.get(self.bb_upper_col_strat)
        bb_lower_prev = previous_row.get(self.bb_lower_col_strat)

        essential_values = [
            close_curr, bb_upper_curr, bb_lower_curr, volume_kline_curr, volume_ma_curr, rsi_curr, atr_curr,
            close_prev, bb_upper_prev, bb_lower_prev
        ]
        if any(pd.isna(val) for val in essential_values):
            nan_details = {k:v for k,v in locals().items() if k in ['close_curr', 'bb_upper_curr', 'bb_lower_curr', 'volume_kline_curr', 'volume_ma_curr', 'rsi_curr', 'atr_curr', 'close_prev', 'bb_upper_prev', 'bb_lower_prev']}
            logger.debug(f"{self.log_prefix} Valeurs d'indicateur/prix manquantes (NaN) à {latest_row.name}. Détails: {nan_details}. Signal HOLD.")
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))
        rsi_buy_thresh = float(self.get_param('rsi_buy_breakout_threshold'))
        rsi_sell_thresh = float(self.get_param('rsi_sell_breakout_threshold'))
        allow_shorting = self.get_param('allow_shorting', False) 

        long_bb_breakout = close_curr > bb_upper_curr and close_prev <= bb_upper_prev # type: ignore
        long_volume_confirm = volume_kline_curr > volume_ma_curr # type: ignore
        long_rsi_confirm = rsi_curr > rsi_buy_thresh # type: ignore
        entry_long_triggered = long_bb_breakout and long_volume_confirm and long_rsi_confirm

        entry_short_triggered = False
        if allow_shorting:
            short_bb_breakout = close_curr < bb_lower_curr and close_prev >= bb_lower_prev # type: ignore
            short_volume_confirm = volume_kline_curr > volume_ma_curr # type: ignore 
            short_rsi_confirm = rsi_curr < rsi_sell_thresh # type: ignore
            entry_short_triggered = short_bb_breakout and short_volume_confirm and short_rsi_confirm
        
        if not current_position_open:
            if entry_long_triggered:
                signal_type = 1
                if atr_curr > 0: # type: ignore
                    sl_price = close_curr - (sl_atr_mult * atr_curr) # type: ignore
                    tp_price = close_curr + (tp_atr_mult * atr_curr) # type: ignore
                logger.info(f"{self.log_prefix} Signal BUY @ {close_curr:.4f}. SL={sl_price}, TP={tp_price}")
            elif entry_short_triggered: 
                signal_type = -1
                if atr_curr > 0: # type: ignore
                    sl_price = close_curr + (sl_atr_mult * atr_curr) # type: ignore
                    tp_price = close_curr - (tp_atr_mult * atr_curr) # type: ignore
                logger.info(f"{self.log_prefix} Signal SELL @ {close_curr:.4f}. SL={sl_price}, TP={tp_price}")
        else: 
            pass 

        order_type_preference = str(self.get_param("order_type_preference", "MARKET"))
        if signal_type != 0 and order_type_preference == "LIMIT":
            limit_price = float(close_curr)

        position_size_pct = float(self.get_param('capital_allocation_pct', 1.0))

        return signal_type, order_type_preference, limit_price, sl_price, tp_price, position_size_pct

    def generate_order_request(self,
                               data: pd.DataFrame,
                               current_position: int,
                               available_capital: float,
                               symbol_info: Dict[str, Any]
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        if current_position != 0:
            logger.debug(f"{self.log_prefix} [Live] Position déjà ouverte (état: {current_position}). Pas de nouvelle requête d'ordre d'entrée.")
            return None

        data_with_indicators = self._calculate_indicators(data.copy())
        if data_with_indicators.empty or len(data_with_indicators) < 2:
            logger.warning(f"{self.log_prefix} [Live] Données insuffisantes après _calculate_indicators pour generate_order_request.")
            return None

        signal, order_type, limit_price_sugg, sl_price_raw, tp_price_raw, pos_size_pct = \
            self._generate_signals(data_with_indicators, False, 0, 0.0)

        if signal not in [1, -1]:
            logger.debug(f"{self.log_prefix} [Live] Aucun signal d'entrée (1 ou -1) généré. Signal: {signal}")
            return None

        latest_bar = data_with_indicators.iloc[-1]
        entry_price_theoretical: float
        if order_type == "LIMIT" and limit_price_sugg is not None:
            entry_price_theoretical = limit_price_sugg
        else: 
            entry_price_theoretical = float(latest_bar.get('close', 0.0))
            if entry_price_theoretical <= 0: entry_price_theoretical = float(latest_bar.get('open', 0.0))
        
        if pd.isna(entry_price_theoretical) or entry_price_theoretical <= 0:
            logger.error(f"{self.log_prefix} [Live] Prix d'entrée théorique invalide ({entry_price_theoretical}).")
            return None

        if self.quantity_precision is None or self.pair_config is None:
            logger.error(f"{self.log_prefix} [Live] Contexte de backtest (précisions, pair_config) non défini.")
            return None
            
        quantity_base = self._calculate_quantity(
            entry_price=entry_price_theoretical, available_capital=available_capital,
            qty_precision=self.quantity_precision, symbol_info=self.pair_config,
            symbol=self.symbol, position_size_pct=pos_size_pct
        )

        if quantity_base is None or quantity_base <= 1e-9:
            logger.warning(f"{self.log_prefix} [Live] Quantité calculée nulle ou invalide ({quantity_base}).")
            return None

        if self.price_precision is None:
            logger.error(f"{self.log_prefix} [Live] price_precision non défini.")
            return None

        entry_price_for_order_str: Optional[str] = None
        if order_type == "LIMIT" and limit_price_sugg is not None:
            from src.utils.exchange_utils import adjust_precision, get_filter_value 
            adjusted_limit_price = adjust_precision(limit_price_sugg, self.price_precision, tick_size=get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize'))
            if adjusted_limit_price is None: return None
            entry_price_for_order_str = f"{adjusted_limit_price:.{self.price_precision}f}"
        
        quantity_str_formatted = f"{quantity_base:.{self.quantity_precision}f}"
        entry_order_params = self._build_entry_params_formatted(
            side="BUY" if signal == 1 else "SELL",
            quantity_str=quantity_str_formatted,
            order_type=str(order_type),
            entry_price_str=entry_price_for_order_str
        )
        if not entry_order_params: return None

        sl_tp_raw_prices_dict: Dict[str, float] = {}
        if sl_price_raw is not None: sl_tp_raw_prices_dict['sl_price'] = float(sl_price_raw)
        if tp_price_raw is not None: sl_tp_raw_prices_dict['tp_price'] = float(tp_price_raw)
        
        logger.info(f"{self.log_prefix} [Live] Requête d'ordre d'entrée générée : {entry_order_params}. SL/TP bruts : {sl_tp_raw_prices_dict}")
        return entry_order_params, sl_tp_raw_prices_dict
