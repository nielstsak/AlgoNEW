# src/strategies/psar_reversal_otoco.py
"""
Stratégie de trading basée sur le renversement du Parabolic SAR (PSAR),
avec gestion du Stop-Loss (SL) et Take-Profit (TP) basée sur l'Average True Range (ATR).
Les ordres sont supposés être de type OTOCO (One-Triggers-One-Cancels-Other)
pour l'entrée avec SL/TP attachés.
"""
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd # Assurer l'import pour Pylance

from src.strategies.base import BaseStrategy # BaseStrategy a été refactorisé
from src.data.data_utils import get_kline_prefix_effective
from src.utils.exchange_utils import adjust_precision, get_filter_value

logger = logging.getLogger(__name__)

class PsarReversalOtocoStrategy(BaseStrategy):
    """
    Stratégie de renversement du Parabolic SAR.
    Génère des signaux d'achat lorsque le PSAR passe de dessus à dessous le prix,
    et des signaux de vente (sortie de long) lorsque le PSAR passe de dessous à dessus le prix.
    Le SL et le TP sont basés sur l'ATR.
    """

    REQUIRED_PARAMS: List[str] = [
        'psar_step',
        'psar_max_step',
        'indicateur_frequence_psar',
        'atr_period_sl_tp',
        'atr_base_frequency_sl_tp',
        'sl_atr_mult',
        'tp_atr_mult',
        'capital_allocation_pct',
        'order_type_preference',
        'margin_leverage'
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        super().__init__(strategy_name, symbol, params)
        self.psarl_col_strat: str = "PSARl_strat"
        self.psars_col_strat: str = "PSARs_strat"
        self.atr_col_strat: str = "ATR_strat"
        self._log("PsarReversalOtocoStrategy instance créée.", level=2)


    def validate_params(self) -> None: # Implémentation de la méthode abstraite
        """Valide les paramètres spécifiques à PsarReversalOtocoStrategy."""
        missing_params = [p for p in self.REQUIRED_PARAMS if self.get_param(p) is None]
        if missing_params:
            raise ValueError(f"{self.log_prefix} Paramètres requis manquants : {', '.join(missing_params)}")

        psar_step = self.get_param('psar_step')
        psar_max_step = self.get_param('psar_max_step')
        if not (isinstance(psar_step, (int, float)) and psar_step > 0):
            raise ValueError(f"{self.log_prefix} 'psar_step' ({psar_step}) doit être un nombre positif.")
        if not (isinstance(psar_max_step, (int, float)) and psar_max_step > 0):
            raise ValueError(f"{self.log_prefix} 'psar_max_step' ({psar_max_step}) doit être un nombre positif.")
        if psar_step >= psar_max_step: # type: ignore
            raise ValueError(f"{self.log_prefix} 'psar_step' ({psar_step}) doit être inférieur à 'psar_max_step' ({psar_max_step}).")

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

        for freq_param_name in ['indicateur_frequence_psar', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' ({freq_val}) doit être une chaîne de caractères non vide.")

        self._log("Validation des paramètres de PsarReversalOtocoStrategy terminée avec succès.", level=2)

    def get_required_indicator_configs(self) -> List[Dict[str, Any]]: # Implémentation
        freq_psar_param = str(self.params['indicateur_frequence_psar'])
        kline_prefix_psar = get_kline_prefix_effective(freq_psar_param)
        source_col_high_psar = f"{kline_prefix_psar}_high" if kline_prefix_psar else "high"
        source_col_low_psar = f"{kline_prefix_psar}_low" if kline_prefix_psar else "low"
        source_col_close_psar = f"{kline_prefix_psar}_close" if kline_prefix_psar else "close"

        freq_atr_param = str(self.params['atr_base_frequency_sl_tp'])
        kline_prefix_atr = get_kline_prefix_effective(freq_atr_param)
        source_col_high_atr = f"{kline_prefix_atr}_high" if kline_prefix_atr else "high"
        source_col_low_atr = f"{kline_prefix_atr}_low" if kline_prefix_atr else "low"
        source_col_close_atr = f"{kline_prefix_atr}_close" if kline_prefix_atr else "close"

        configs = [
            {
                'indicator_name': 'psar',
                'params': {
                    'af': float(self.params['psar_step']),
                    'max_af': float(self.params['psar_max_step'])
                },
                'inputs': {
                    'high': source_col_high_psar,
                    'low': source_col_low_psar,
                    'close': source_col_close_psar # PSAR de pandas-ta utilise aussi 'close'
                },
                'outputs': { # pandas-ta retourne un DataFrame pour psar
                    'long': self.psarl_col_strat,  # Clé conceptuelle, sera mappée
                    'short': self.psars_col_strat, # Clé conceptuelle, sera mappée
                    # 'af': f"PSARaf_{self.psarl_col_strat}", # Optionnel si on veut l'AF
                    # 'reversal': f"PSARr_{self.psarl_col_strat}" # Optionnel si on veut les points de renversement
                }
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
        expected_strat_cols = [self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat]

        base_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if missing_ohlcv := [col for col in base_ohlcv if col not in df.columns]:
            msg = f"Colonnes OHLCV de base manquantes dans data_feed: {missing_ohlcv}."
            self._log(msg, level=1, is_error=True)
            raise ValueError(msg)

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
        previous_row = data_with_indicators.iloc[-2]

        close_curr = latest_row.get('close')
        # Pour PSAR, pandas-ta retourne PSARl (valeur si tendance long) et PSARs (valeur si tendance short)
        # Un seul des deux est non-NaN à un instant T.
        psarl_curr = latest_row.get(self.psarl_col_strat)
        psars_curr = latest_row.get(self.psars_col_strat)
        atr_curr = latest_row.get(self.atr_col_strat)

        psarl_prev = previous_row.get(self.psarl_col_strat)
        psars_prev = previous_row.get(self.psars_col_strat)

        if pd.isna(close_curr) or pd.isna(atr_curr) or \
           (pd.isna(psarl_curr) and pd.isna(psars_curr)) or \
           (pd.isna(psarl_prev) and pd.isna(psars_prev)): # Vérifier si au moins une valeur PSAR existe
            nan_details = {"close": close_curr, "atr": atr_curr,
                           "psarl_c": psarl_curr, "psars_c": psars_curr,
                           "psarl_p": psarl_prev, "psars_p": psars_prev}
            self._log(f"Valeurs d'indicateur/prix manquantes (NaN) à {latest_row.name}. Détails: {nan_details}. Signal HOLD.", level=2, is_warning=True)
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        # Détection de renversement du PSAR
        # Renversement vers le HAUT (signal d'achat) : PSAR était au-dessus (PSARs_prev non-NaN) et passe en dessous (PSARl_curr non-NaN)
        entry_long_triggered = pd.notna(psars_prev) and pd.isna(psarl_prev) and \
                               pd.notna(psarl_curr) and pd.isna(psars_curr)

        # Renversement vers le BAS (signal de vente/sortie de long) : PSAR était en dessous (PSARl_prev non-NaN) et passe au-dessus (PSARs_curr non-NaN)
        exit_long_triggered = pd.notna(psarl_prev) and pd.isna(psars_prev) and \
                              pd.notna(psars_curr) and pd.isna(psarl_curr)
        
        # Cette stratégie de base n'initie pas de positions SHORT.
        # entry_short_triggered = exit_long_triggered (si shorting était permis)
        # exit_short_triggered = entry_long_triggered (si shorting était permis)

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        if not current_position_open:
            if entry_long_triggered:
                signal_type = 1 # BUY
                if atr_curr > 0:
                    sl_price = close_curr - (sl_atr_mult * atr_curr)
                    tp_price = close_curr + (tp_atr_mult * atr_curr)
                self._log(f"Signal BUY (PSAR Long) @ {close_curr:.{self.price_precision or 4}f}. SL={sl_price}, TP={tp_price or 'N/A'}", level=1)
        elif current_position_direction == 1: # Si position LONG ouverte
            if exit_long_triggered:
                signal_type = 2 # EXIT
                self._log(f"Signal EXIT LONG (PSAR a renversé vers le bas) @ {close_curr:.{self.price_precision or 4}f}", level=1)

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

        if not self.trading_context: # Normalement setté par LiveTradingManager
            self._log("[Live] TradingContext non défini. Configuration de fallback.", level=1, is_warning=True)
            from src.strategies.base import TradingContext # Import local pour fallback
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

        if signal != 1: # Cette stratégie initie seulement des LONGs
            self._log(f"[Live] Aucun signal d'entrée LONG (1) généré. Signal: {signal}", level=2)
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
            side="BUY", quantity_str=quantity_str_fmt,
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

