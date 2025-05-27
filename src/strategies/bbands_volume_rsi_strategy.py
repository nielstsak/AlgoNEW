# src/strategies/bbands_volume_rsi_strategy.py
"""
Stratégie de trading utilisant les Bandes de Bollinger (BBands), le volume,
et le Relative Strength Index (RSI) pour identifier les signaux d'entrée.
Les Stop-Loss (SL) et Take-Profit (TP) sont basés sur l'Average True Range (ATR).
"""
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd # Assurer l'import pour Pylance

from src.strategies.base import BaseStrategy # BaseStrategy a été refactorisé
from src.data.data_utils import get_kline_prefix_effective
from src.utils.exchange_utils import adjust_precision, get_filter_value


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
        'margin_leverage',
        'allow_shorting'
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        super().__init__(strategy_name, symbol, params) # Appelle self.validate_params()

        self.bb_upper_col_strat: str = "BB_UPPER_strat"
        self.bb_middle_col_strat: str = "BB_MIDDLE_strat"
        self.bb_lower_col_strat: str = "BB_LOWER_strat"
        self.bb_bandwidth_col_strat: str = "BB_BANDWIDTH_strat" # Ajouté si pandas_ta le retourne

        self.volume_ma_col_strat: str = "Volume_MA_strat"
        self.rsi_col_strat: str = "RSI_strat"
        self.atr_col_strat: str = "ATR_strat"

        # Déterminer la colonne source pour la MA du volume
        vol_freq_param = self.get_param('indicateur_frequence_volume')
        if not isinstance(vol_freq_param, str):
            self._log(f"'indicateur_frequence_volume' doit être une chaîne. Reçu: {vol_freq_param}. Utilisation de 'volume' par défaut.", level=1, is_error=True)
            self.volume_kline_col_source: str = "volume"
        else:
            kline_prefix_vol_src = get_kline_prefix_effective(vol_freq_param)
            # La colonne source sera soit "volume" (pour 1min) soit "Klines_Xmin_volume"
            self.volume_kline_col_source: str = f"{kline_prefix_vol_src}_volume" if kline_prefix_vol_src else "volume"

        self._log(f"BbandsVolumeRsiStrategy instance créée. Colonne source pour MA Volume: '{self.volume_kline_col_source}'", level=2)


    def validate_params(self) -> None: # Implémentation de la méthode abstraite
        """Valide les paramètres spécifiques à BbandsVolumeRsiStrategy."""
        missing_params = [p for p in self.REQUIRED_PARAMS if self.get_param(p) is None]
        if missing_params:
            raise ValueError(f"{self.log_prefix} Paramètres requis manquants : {', '.join(missing_params)}")

        if not (isinstance(self.get_param('bbands_period'), int) and self.get_param('bbands_period') > 0):
            raise ValueError(f"{self.log_prefix} 'bbands_period' ({self.get_param('bbands_period')}) doit être un entier positif.")
        if not (isinstance(self.get_param('bbands_std_dev'), (int,float)) and self.get_param('bbands_std_dev') > 0):
            raise ValueError(f"{self.log_prefix} 'bbands_std_dev' ({self.get_param('bbands_std_dev')}) doit être un nombre positif.")
        if not (isinstance(self.get_param('volume_ma_period'), int) and self.get_param('volume_ma_period') > 0):
            raise ValueError(f"{self.log_prefix} 'volume_ma_period' ({self.get_param('volume_ma_period')}) doit être un entier positif.")
        if not (isinstance(self.get_param('rsi_period'), int) and self.get_param('rsi_period') > 0):
            raise ValueError(f"{self.log_prefix} 'rsi_period' ({self.get_param('rsi_period')}) doit être un entier positif.")

        rsi_buy_thresh = self.get_param('rsi_buy_breakout_threshold')
        rsi_sell_thresh = self.get_param('rsi_sell_breakout_threshold')
        if not (isinstance(rsi_buy_thresh, (int,float)) and 50 < rsi_buy_thresh < 100):
            raise ValueError(f"{self.log_prefix} 'rsi_buy_breakout_threshold' ({rsi_buy_thresh}) doit être entre 50 (exclusif) et 100 (exclusif).")
        if not (isinstance(rsi_sell_thresh, (int,float)) and 0 < rsi_sell_thresh < 50):
            raise ValueError(f"{self.log_prefix} 'rsi_sell_breakout_threshold' ({rsi_sell_thresh}) doit être entre 0 (exclusif) et 50 (exclusif).")
        if rsi_sell_thresh >= rsi_buy_thresh: # type: ignore
            raise ValueError(f"{self.log_prefix} 'rsi_sell_breakout_threshold' ({rsi_sell_thresh}) doit être strictement inférieur à 'rsi_buy_breakout_threshold' ({rsi_buy_thresh}).")

        atr_p = self.get_param('atr_period_sl_tp')
        if not (isinstance(atr_p, int) and atr_p > 0):
            raise ValueError(f"{self.log_prefix} 'atr_period_sl_tp' ({atr_p}) doit être un entier positif.")
        sl_mult = self.get_param('sl_atr_mult')
        tp_mult = self.get_param('tp_atr_mult')
        if not (isinstance(sl_mult, (int, float)) and sl_mult > 0):
            raise ValueError(f"{self.log_prefix} 'sl_atr_mult' ({sl_mult}) doit être un nombre positif.")
        if not (isinstance(tp_mult, (int, float)) and tp_mult > 0):
            raise ValueError(f"{self.log_prefix} 'tp_atr_mult' ({tp_mult}) doit être un nombre positif.")

        cap_alloc = self.get_param('capital_allocation_pct')
        if not (isinstance(cap_alloc, (int, float)) and 0 < cap_alloc <= 1.0):
            raise ValueError(f"{self.log_prefix} 'capital_allocation_pct' ({cap_alloc}) doit être entre 0 (exclusif) et 1 (inclusif).")

        margin_lev = self.get_param('margin_leverage')
        if not (isinstance(margin_lev, (int, float)) and margin_lev >= 1.0):
            raise ValueError(f"{self.log_prefix} 'margin_leverage' ({margin_lev}) doit être >= 1.0.")

        order_type_pref_val = self.get_param('order_type_preference')
        if order_type_pref_val not in ["MARKET", "LIMIT"]:
            raise ValueError(f"{self.log_prefix} 'order_type_preference' ({order_type_pref_val}) doit être 'MARKET' ou 'LIMIT'.")

        if not isinstance(self.get_param('allow_shorting'), bool):
            raise ValueError(f"{self.log_prefix} 'allow_shorting' ({self.get_param('allow_shorting')}) doit être un booléen (true/false).")

        for freq_param_name in ['indicateur_frequence_bbands', 'indicateur_frequence_volume', 'indicateur_frequence_rsi', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' ({freq_val}) doit être une chaîne de caractères non vide.")
        self._log("Validation des paramètres de BbandsVolumeRsiStrategy terminée avec succès.", level=2)

    def get_required_indicator_configs(self) -> List[Dict[str, Any]]: # Implémentation
        bbands_params = {'length': int(self.params['bbands_period']), 'std': float(self.params['bbands_std_dev'])}

        freq_bbands = str(self.params['indicateur_frequence_bbands'])
        kline_prefix_bbands = get_kline_prefix_effective(freq_bbands)
        source_col_close_bbands = f"{kline_prefix_bbands}_close" if kline_prefix_bbands else "close"

        # La colonne source pour la MA du volume est déjà déterminée dans __init__ (self.volume_kline_col_source)
        # freq_volume = str(self.params['indicateur_frequence_volume'])
        # kline_prefix_volume = get_kline_prefix_effective(freq_volume)
        # source_col_volume_for_ma = f"{kline_prefix_volume}_volume" if kline_prefix_volume else "volume"

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
                'outputs': { # pandas-ta retourne un DataFrame pour bbands
                    'lower': self.bb_lower_col_strat,
                    'middle': self.bb_middle_col_strat,
                    'upper': self.bb_upper_col_strat,
                    'bandwidth': self.bb_bandwidth_col_strat # BBP_length_std, BBB_length_std
                }
            },
            {
                'indicator_name': 'sma', # Ou autre type de MA si paramétrable
                'params': {'length': int(self.params['volume_ma_period'])},
                'inputs': {'close': self.volume_kline_col_source}, # Utiliser la colonne source de volume correcte
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
        self._log(f"Configurations d'indicateurs requises: {configs}", level=3)
        return configs

    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame: # Implémentation
        df = data_feed.copy()
        expected_strat_cols = [
            self.bb_upper_col_strat, self.bb_middle_col_strat, self.bb_lower_col_strat,
            self.bb_bandwidth_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat
        ]
        # La colonne self.volume_kline_col_source est une colonne source, pas une colonne _strat
        # Elle doit être présente dans data_feed avant cet appel.

        base_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if missing_ohlcv := [col for col in base_ohlcv if col not in df.columns]:
            msg = f"Colonnes OHLCV de base manquantes dans data_feed: {missing_ohlcv}."
            self._log(msg, level=1, is_error=True)
            raise ValueError(msg)

        # Vérifier la présence de la colonne source pour la MA du volume
        if self.volume_kline_col_source not in df.columns:
            msg = f"Colonne source pour MA Volume '{self.volume_kline_col_source}' manquante dans data_feed."
            self._log(msg, level=1, is_error=True)
            # Option: ajouter avec NaN ou lever une erreur. Si critique, lever une erreur.
            df[self.volume_kline_col_source] = np.nan # Pour éviter KeyError plus loin, mais cache un problème
            # raise ValueError(msg)


        missing_strat_cols = [col for col in expected_strat_cols if col not in df.columns]
        if missing_strat_cols:
            msg = f"Colonnes indicateur attendues manquantes après IndicatorCalculator: {missing_strat_cols}."
            self._log(msg, level=1, is_error=True)
            for col_name in missing_strat_cols: df[col_name] = np.nan

        cols_to_ffill_present = [col for col in expected_strat_cols if col in df.columns]
        if cols_to_ffill_present:
            if df[cols_to_ffill_present].isnull().values.any():
                self._log(f"Application de ffill aux colonnes indicateur: {cols_to_ffill_present}", level=3)
            df[cols_to_ffill_present] = df[cols_to_ffill_present].ffill()

        self._log(f"Indicateurs prêts pour _generate_signals. Colonnes: {df.columns.tolist()}", level=3)
        return df

    def _generate_signals(self,
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int,
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]: # Implémentation
        signal_type: int = 0
        limit_price: Optional[float] = None
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        if len(data_with_indicators) < 2:
            self._log("Pas assez de données (< 2 barres) pour générer des signaux.", level=2, is_warning=True)
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        latest_row = data_with_indicators.iloc[-1]
        previous_row = data_with_indicators.iloc[-2] # Nécessaire pour détecter le *franchissement* des bandes

        # Accès sécurisé aux colonnes
        close_curr = latest_row.get('close')
        bb_upper_curr = latest_row.get(self.bb_upper_col_strat)
        bb_lower_curr = latest_row.get(self.bb_lower_col_strat)
        # self.volume_kline_col_source est le nom de la colonne de volume source (ex: 'volume' ou 'Klines_5min_volume')
        volume_kline_curr = latest_row.get(self.volume_kline_col_source)
        volume_ma_curr = latest_row.get(self.volume_ma_col_strat)
        rsi_curr = latest_row.get(self.rsi_col_strat)
        atr_curr = latest_row.get(self.atr_col_strat)

        close_prev = previous_row.get('close') # Pour détecter le franchissement
        bb_upper_prev = previous_row.get(self.bb_upper_col_strat)
        bb_lower_prev = previous_row.get(self.bb_lower_col_strat)


        essential_values = [
            close_curr, bb_upper_curr, bb_lower_curr, volume_kline_curr, volume_ma_curr, rsi_curr, atr_curr,
            close_prev, bb_upper_prev, bb_lower_prev
        ]
        if any(pd.isna(val) for val in essential_values):
            nan_details = {k:v for k,v in locals().items() if k in [
                'close_curr', 'bb_upper_curr', 'bb_lower_curr', 'volume_kline_curr', 'volume_ma_curr',
                'rsi_curr', 'atr_curr', 'close_prev', 'bb_upper_prev', 'bb_lower_prev']}
            self._log(f"Valeurs d'indicateur/prix manquantes (NaN) à {latest_row.name}. Détails: {nan_details}. Signal HOLD.", level=2, is_warning=True)
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        # Récupération des paramètres de la stratégie
        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))
        rsi_buy_thresh = float(self.get_param('rsi_buy_breakout_threshold'))
        rsi_sell_thresh = float(self.get_param('rsi_sell_breakout_threshold'))
        allow_shorting = bool(self.get_param('allow_shorting', False))

        # Conditions de signal LONG
        # Franchissement de la bande supérieure de Bollinger
        long_bb_breakout = (close_curr > bb_upper_curr) and (close_prev <= bb_upper_prev)
        # Confirmation par le volume (volume actuel > MA du volume)
        long_volume_confirm = (volume_kline_curr > volume_ma_curr)
        # Confirmation par le RSI (RSI > seuil de surachat ou de breakout haussier)
        long_rsi_confirm = (rsi_curr > rsi_buy_thresh)
        entry_long_triggered = long_bb_breakout and long_volume_confirm and long_rsi_confirm

        # Conditions de signal SHORT (si autorisé)
        entry_short_triggered = False
        if allow_shorting:
            # Franchissement de la bande inférieure de Bollinger
            short_bb_breakout = (close_curr < bb_lower_curr) and (close_prev >= bb_lower_prev)
            # Confirmation par le volume (similaire au long pour cet exemple)
            short_volume_confirm = (volume_kline_curr > volume_ma_curr)
            # Confirmation par le RSI (RSI < seuil de survente ou de breakout baissier)
            short_rsi_confirm = (rsi_curr < rsi_sell_thresh)
            entry_short_triggered = short_bb_breakout and short_volume_confirm and short_rsi_confirm

        if current_position_open:
            # Logique de sortie: pour cette stratégie, on pourrait sortir sur un signal opposé
            # ou laisser SL/TP gérer la sortie. Ici, on sort sur signal opposé si shorting est permis.
            if current_position_direction == 1 and entry_short_triggered: # Si LONG et signal SHORT
                signal_type = 2 # EXIT
                self._log(f"Signal EXIT LONG (signal SHORT opposé) @ {close_curr:.{self.price_precision or 4}f}", level=1)
            elif current_position_direction == -1 and entry_long_triggered: # Si SHORT et signal LONG
                signal_type = 2 # EXIT
                self._log(f"Signal EXIT SHORT (signal LONG opposé) @ {close_curr:.{self.price_precision or 4}f}", level=1)
            else:
                self._log(f"Position ouverte ({'LONG' if current_position_direction == 1 else 'SHORT'}). Pas de signal de sortie actif. Attente SL/TP.", level=3)
        else: # Pas de position ouverte
            if entry_long_triggered:
                signal_type = 1 # BUY
                if atr_curr > 0:
                    sl_price = close_curr - (sl_atr_mult * atr_curr)
                    tp_price = close_curr + (tp_atr_mult * atr_curr)
                self._log(f"Signal BUY @ {close_curr:.{self.price_precision or 4}f}. SL={sl_price}, TP={tp_price or 'N/A'}", level=1)
            elif entry_short_triggered: # Seulement si allow_shorting est True
                signal_type = -1 # SELL (SHORT)
                if atr_curr > 0:
                    sl_price = close_curr + (sl_atr_mult * atr_curr)
                    tp_price = close_curr - (tp_atr_mult * atr_curr)
                self._log(f"Signal SELL (SHORT) @ {close_curr:.{self.price_precision or 4}f}. SL={sl_price}, TP={tp_price or 'N/A'}", level=1)

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
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]: # Implémentation
        self._log(f"[Live] Appel de generate_order_request. Position: {current_position}, Capital: {available_capital:.2f}", level=2)
        if current_position != 0:
            self._log("[Live] Position déjà ouverte. Pas de nouvelle requête d'ordre d'entrée.", level=2)
            return None

        if not self.trading_context:
            self._log("[Live] TradingContext non défini. Configuration de fallback.", level=1, is_warning=True)
            from src.strategies.base import TradingContext # Import local
            temp_context = TradingContext(
                pair_config=symbol_info, is_futures=False,
                leverage=int(self.get_param('margin_leverage', 1)),
                initial_equity=available_capital, account_type="MARGIN"
            )
            val_res = self.set_trading_context(temp_context)
            if not val_res.is_valid:
                self._log(f"[Live] Échec configuration contexte fallback: {val_res.messages}", level=1, is_error=True)
                return None
        try:
            data_with_indicators = self._calculate_indicators(data.copy())
        except Exception as e_calc_live:
            self._log(f"[Live] Erreur _calculate_indicators: {e_calc_live}", level=1, is_error=True)
            return None

        if data_with_indicators.empty or len(data_with_indicators) < 2:
            self._log("[Live] Données insuffisantes post-_calculate_indicators.", level=2, is_warning=True)
            return None
        try:
            signal, order_type, limit_price_sugg, sl_price_raw, tp_price_raw, pos_size_pct = \
                self._generate_signals(data_with_indicators, False, 0, 0.0)
        except Exception as e_sig_live:
            self._log(f"[Live] Erreur _generate_signals: {e_sig_live}", level=1, is_error=True)
            return None

        if signal not in [1, -1]:
            self._log(f"[Live] Aucun signal d'entrée (1 ou -1) généré. Signal: {signal}", level=2)
            return None

        if signal == -1 and not bool(self.get_param('allow_shorting', False)):
            self._log("[Live] Signal Short (-1) mais shorting non autorisé. Pas d'ordre.", level=2)
            return None

        latest_bar = data_with_indicators.iloc[-1]
        entry_price_theoretical = limit_price_sugg if order_type == "LIMIT" and limit_price_sugg is not None \
                                  else float(latest_bar.get('close', 0.0))
        if entry_price_theoretical <= 0: entry_price_theoretical = float(latest_bar.get('open', 0.0))

        if pd.isna(entry_price_theoretical) or entry_price_theoretical <= 0:
            self._log(f"[Live] Prix d'entrée théorique invalide ({entry_price_theoretical}).", level=1, is_error=True)
            return None

        if self.quantity_precision is None or self.pair_config is None:
            self._log("[Live] Contexte de précision ou pair_config non défini.", level=1, is_error=True)
            return None

        quantity_base = self._calculate_quantity(
            entry_price=entry_price_theoretical, available_capital=available_capital,
            qty_precision=self.quantity_precision, symbol_info=self.pair_config,
            symbol=self.symbol, position_size_pct=pos_size_pct
        )
        if quantity_base is None or quantity_base <= 1e-9:
            self._log(f"[Live] Quantité calculée nulle/invalide ({quantity_base}).", level=2, is_warning=True)
            return None

        if self.price_precision is None:
            self._log("[Live] price_precision non défini.", level=1, is_error=True)
            return None

        entry_price_for_order_str: Optional[str] = None
        if order_type == "LIMIT" and limit_price_sugg is not None:
            tick_size_price = get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize')
            adj_limit_px = adjust_precision(limit_price_sugg, self.price_precision, tick_size=tick_size_price)
            if adj_limit_px is None:
                self._log(f"[Live] Échec ajustement prix limite {limit_price_sugg}.", level=1, is_error=True)
                return None
            entry_price_for_order_str = f"{adj_limit_px:.{self.price_precision}f}"

        quantity_str_fmt = f"{quantity_base:.{self.quantity_precision}f}"
        entry_order_params = self._build_entry_params_formatted(
            side="BUY" if signal == 1 else "SELL", quantity_str=quantity_str_fmt,
            order_type=str(order_type), entry_price_str=entry_price_for_order_str
        )
        if not entry_order_params:
            self._log("[Live] Échec construction params ordre d'entrée.", level=1, is_error=True)
            return None

        sl_tp_dict: Dict[str, float] = {}
        if sl_price_raw is not None: sl_tp_dict['sl_price'] = float(sl_price_raw)
        if tp_price_raw is not None: sl_tp_dict['tp_price'] = float(tp_price_raw)

        self._log(f"[Live] Requête d'ordre générée: {entry_order_params}. SL/TP bruts: {sl_tp_dict}", level=1)
        return entry_order_params, sl_tp_dict

