# src/strategies/triple_ma_anticipation_strategy.py
"""
Stratégie de trading basée sur trois moyennes mobiles (courte, moyenne, longue)
avec une logique optionnelle d'anticipation des croisements basée sur les pentes
des moyennes mobiles. Les Stop-Loss (SL) et Take-Profit (TP) sont basés sur
l'Average True Range (ATR).
"""
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd # Assurer l'import pour Pylance
import pandas_ta as ta # Assurer l'import pour Pylance

from src.strategies.base import BaseStrategy # BaseStrategy a été refactorisé
from src.data.data_utils import get_kline_prefix_effective
from src.utils.exchange_utils import adjust_precision, get_filter_value


logger = logging.getLogger(__name__)

class TripleMAAnticipationStrategy(BaseStrategy):
    """
    Stratégie de trading utilisant trois moyennes mobiles (MA) et une logique
    d'anticipation des croisements.
    """

    REQUIRED_PARAMS: List[str] = [
        'ma_short_period', 'ma_medium_period', 'ma_long_period', 'ma_type',
        'indicateur_frequence_mms',
        'indicateur_frequence_mmm',
        'indicateur_frequence_mml',
        'atr_period_sl_tp', 'atr_base_frequency_sl_tp',
        'sl_atr_mult', 'tp_atr_mult',
        'allow_shorting', 'order_type_preference',
        'capital_allocation_pct',
        'margin_leverage',
        'anticipate_crossovers'
        # Les paramètres 'anticipation_slope_period' et 'anticipation_convergence_threshold_pct'
        # sont conditionnellement requis si 'anticipate_crossovers' est True.
        # La validation dans _validate_params gère cette condition.
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        # Initialiser les attributs spécifiques avant d'appeler super().__init__
        # car super().__init__ appelle self.validate_params() qui peut les utiliser.
        self.anticipate_crossovers_enabled: bool = bool(params.get('anticipate_crossovers', False))
        self.slope_ma_short_col_strat: Optional[str] = None
        self.slope_ma_medium_col_strat: Optional[str] = None
        self.anticipation_slope_period_val: Optional[int] = None
        self.anticipation_convergence_threshold_pct_val: Optional[float] = None

        if self.anticipate_crossovers_enabled:
            self.slope_ma_short_col_strat = "SLOPE_MA_SHORT_strat"
            self.slope_ma_medium_col_strat = "SLOPE_MA_MEDIUM_strat"
            # Les valeurs pour slope_period et convergence_threshold sont extraites dans _validate_params

        super().__init__(strategy_name, symbol, params) # Appelle self.validate_params()

        self.ma_short_col_strat: str = "MA_SHORT_strat"
        self.ma_medium_col_strat: str = "MA_MEDIUM_strat"
        self.ma_long_col_strat: str = "MA_LONG_strat"
        self.atr_col_strat: str = "ATR_strat"
        self._log("TripleMAAnticipationStrategy instance créée.", level=2)


    def validate_params(self) -> None: # Implémentation de la méthode abstraite
        """Valide les paramètres spécifiques à TripleMAAnticipationStrategy."""
        current_required_params = self.REQUIRED_PARAMS[:] # Copie pour modification locale
        if self.anticipate_crossovers_enabled:
            # Ajouter les paramètres conditionnels s'ils ne sont pas déjà dans REQUIRED_PARAMS
            # (Bonne pratique: les lister dans REQUIRED_PARAMS et les ignorer si non applicables,
            # ou les ajouter dynamiquement ici si on veut que REQUIRED_PARAMS soit minimaliste)
            if 'anticipation_slope_period' not in current_required_params:
                current_required_params.append('anticipation_slope_period')
            if 'anticipation_convergence_threshold_pct' not in current_required_params:
                current_required_params.append('anticipation_convergence_threshold_pct')

        if missing_params := [p for p in current_required_params if self.get_param(p) is None]:
            raise ValueError(f"{self.log_prefix} Paramètres requis manquants : {', '.join(missing_params)}")

        ma_s = self.get_param('ma_short_period')
        ma_m = self.get_param('ma_medium_period')
        ma_l = self.get_param('ma_long_period')

        if not (isinstance(ma_s, int) and ma_s > 0 and \
                isinstance(ma_m, int) and ma_m > 0 and \
                isinstance(ma_l, int) and ma_l > 0):
            raise ValueError(f"{self.log_prefix} Les périodes MA (short, medium, long) doivent être des entiers positifs. "
                             f"Reçu: S={ma_s}, M={ma_m}, L={ma_l}")
        if not (ma_s < ma_m < ma_l):
            raise ValueError(f"{self.log_prefix} Les périodes MA doivent être en ordre croissant : short ({ma_s}) < medium ({ma_m}) < long ({ma_l}).")

        ma_type_val = self.get_param('ma_type')
        supported_ma_types = ['sma', 'ema', 'wma', 'hma', 'tema', 'dema', 'rma', 'vwma', 'zlma']
        if not isinstance(ma_type_val, str) or ma_type_val.lower() not in supported_ma_types:
            raise ValueError(f"{self.log_prefix} 'ma_type' ({ma_type_val}) invalide. Supportés : {supported_ma_types}")

        if self.anticipate_crossovers_enabled:
            slope_p = self.get_param('anticipation_slope_period')
            conv_thresh = self.get_param('anticipation_convergence_threshold_pct')
            if not (isinstance(slope_p, int) and slope_p >= 2):
                raise ValueError(f"{self.log_prefix} 'anticipation_slope_period' ({slope_p}) doit être un entier >= 2 si l'anticipation est activée.")
            if not (isinstance(conv_thresh, (int, float)) and 0 < conv_thresh < 1):
                raise ValueError(f"{self.log_prefix} 'anticipation_convergence_threshold_pct' ({conv_thresh}) doit être un float > 0 et < 1 si l'anticipation est activée.")
            self.anticipation_slope_period_val = slope_p
            self.anticipation_convergence_threshold_pct_val = float(conv_thresh)
        else:
            # S'assurer que ces attributs sont None si l'anticipation est désactivée
            self.anticipation_slope_period_val = None
            self.anticipation_convergence_threshold_pct_val = None

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

        for freq_param_name in ['indicateur_frequence_mms', 'indicateur_frequence_mmm', 'indicateur_frequence_mml', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' ({freq_val}) doit être une chaîne de caractères non vide.")

        self._log(f"Validation des paramètres terminée. Anticipation: {self.anticipate_crossovers_enabled}", level=2)


    def get_required_indicator_configs(self) -> List[Dict[str, Any]]: # Implémentation
        ma_type_to_use = str(self.params.get('ma_type', 'ema')).lower()

        freq_mms = str(self.params['indicateur_frequence_mms'])
        kline_prefix_mms = get_kline_prefix_effective(freq_mms)
        source_col_close_mms = f"{kline_prefix_mms}_close" if kline_prefix_mms else "close"

        freq_mmm = str(self.params['indicateur_frequence_mmm'])
        kline_prefix_mmm = get_kline_prefix_effective(freq_mmm)
        source_col_close_mmm = f"{kline_prefix_mmm}_close" if kline_prefix_mmm else "close"

        freq_mml = str(self.params['indicateur_frequence_mml'])
        kline_prefix_mml = get_kline_prefix_effective(freq_mml)
        source_col_close_mml = f"{kline_prefix_mml}_close" if kline_prefix_mml else "close"

        freq_atr = str(self.params['atr_base_frequency_sl_tp'])
        kline_prefix_atr = get_kline_prefix_effective(freq_atr)
        source_col_high_atr = f"{kline_prefix_atr}_high" if kline_prefix_atr else "high"
        source_col_low_atr = f"{kline_prefix_atr}_low" if kline_prefix_atr else "low"
        source_col_close_atr = f"{kline_prefix_atr}_close" if kline_prefix_atr else "close"

        configs = [
            {
                'indicator_name': ma_type_to_use,
                'params': {'length': int(self.params['ma_short_period'])},
                'inputs': {'close': source_col_close_mms},
                'outputs': self.ma_short_col_strat
            },
            {
                'indicator_name': ma_type_to_use,
                'params': {'length': int(self.params['ma_medium_period'])},
                'inputs': {'close': source_col_close_mmm},
                'outputs': self.ma_medium_col_strat
            },
            {
                'indicator_name': ma_type_to_use,
                'params': {'length': int(self.params['ma_long_period'])},
                'inputs': {'close': source_col_close_mml},
                'outputs': self.ma_long_col_strat
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
        # Les indicateurs de pente sont calculés dans _calculate_indicators si besoin,
        # pas déclarés ici car ils dépendent des MAs déjà calculées.
        self._log(f"Configurations d'indicateurs de base requises: {configs}", level=3)
        return configs

    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame: # Implémentation
        df = data_feed.copy()

        expected_base_strat_cols = [
            self.ma_short_col_strat, self.ma_medium_col_strat,
            self.ma_long_col_strat, self.atr_col_strat
        ]
        all_final_expected_strat_cols = expected_base_strat_cols[:]
        if self.anticipate_crossovers_enabled:
            if self.slope_ma_short_col_strat: all_final_expected_strat_cols.append(self.slope_ma_short_col_strat)
            if self.slope_ma_medium_col_strat: all_final_expected_strat_cols.append(self.slope_ma_medium_col_strat)

        base_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if missing_ohlcv := [col for col in base_ohlcv if col not in df.columns]:
            msg = f"Colonnes OHLCV de base manquantes dans data_feed: {missing_ohlcv}."
            self._log(msg, level=1, is_error=True)
            raise ValueError(msg)

        missing_initial_strat_cols = [col for col in expected_base_strat_cols if col not in df.columns]
        if missing_initial_strat_cols:
            msg = f"Colonnes indicateur de base attendues manquantes après IndicatorCalculator: {missing_initial_strat_cols}."
            self._log(msg, level=1, is_error=True)
            for col_name in missing_initial_strat_cols: df[col_name] = np.nan
            # raise ValueError(msg) # Ou lever une erreur

        # Calcul des pentes si l'anticipation est activée
        if self.anticipate_crossovers_enabled and \
           self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat and \
           self.anticipation_slope_period_val and self.anticipation_slope_period_val >= 2:
            
            slope_period = self.anticipation_slope_period_val

            for ma_col_name, slope_col_name_target, ma_log_name in [
                (self.ma_short_col_strat, self.slope_ma_short_col_strat, "courte"),
                (self.ma_medium_col_strat, self.slope_ma_medium_col_strat, "moyenne")
            ]:
                if ma_col_name in df.columns and df[ma_col_name].notna().any():
                    # S'assurer que la série est float pour ta.slope
                    ma_series_clean = df[ma_col_name].dropna().astype(float)
                    if len(ma_series_clean) >= slope_period:
                        # pandas_ta.slope retourne une Series avec le même index que l'entrée
                        slope_series = ta.slope(close=ma_series_clean, length=slope_period, append=False) # type: ignore
                        # Réindexer pour correspondre à l'index original du DataFrame df (qui peut avoir des NaNs au début)
                        df[slope_col_name_target] = slope_series.reindex(df.index)
                        self._log(f"Pente MA {ma_log_name} (col: {ma_col_name}) calculée et assignée à '{slope_col_name_target}'.", level=3)
                    else:
                        self._log(f"Pas assez de points non-NaN ({len(ma_series_clean)}) pour MA {ma_log_name} "
                                   f"pour calculer la pente de période {slope_period}. '{slope_col_name_target}' sera NaN.", level=2, is_warning=True)
                        df[slope_col_name_target] = np.nan
                else:
                    self._log(f"Colonne MA {ma_log_name} ('{ma_col_name}') manquante ou entièrement NaN. "
                               f"Impossible de calculer sa pente. '{slope_col_name_target}' sera NaN.", level=2, is_warning=True)
                    df[slope_col_name_target] = np.nan
        
        # Vérifier si toutes les colonnes finales attendues (y compris pentes si activées) sont là
        final_missing_cols_after_calc = [col for col in all_final_expected_strat_cols if col not in df.columns]
        if final_missing_cols_after_calc:
            for col_final_miss in final_missing_cols_after_calc:
                 self._log(f"Colonne _strat finale attendue '{col_final_miss}' toujours manquante après tous les calculs. Ajout avec NaN.", level=2, is_warning=True)
                 df[col_final_miss] = np.nan

        cols_to_ffill_present = [col for col in all_final_expected_strat_cols if col in df.columns]
        if cols_to_ffill_present:
            if df[cols_to_ffill_present].isnull().values.any():
                self._log(f"Application de ffill aux colonnes indicateur finales: {cols_to_ffill_present}", level=3)
            df[cols_to_ffill_present] = df[cols_to_ffill_present].ffill()

        self._log(f"Indicateurs (y compris pentes si activées) prêts pour _generate_signals. Colonnes: {df.columns.tolist()}", level=3)
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

        if len(data_with_indicators) < 2: # Besoin d'au moins 2 barres pour les comparaisons prev/curr
            self._log("Pas assez de données (< 2 barres) pour générer des signaux.", level=2, is_warning=True)
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        latest_row = data_with_indicators.iloc[-1]
        previous_row = data_with_indicators.iloc[-2]

        # Récupération des valeurs des MAs et ATR
        ma_s_curr = latest_row.get(self.ma_short_col_strat)
        ma_m_curr = latest_row.get(self.ma_medium_col_strat)
        ma_l_curr = latest_row.get(self.ma_long_col_strat)
        ma_s_prev = previous_row.get(self.ma_short_col_strat)
        ma_m_prev = previous_row.get(self.ma_medium_col_strat)
        # ma_l_prev n'est pas utilisé dans la logique de croisement direct, mais pourrait l'être pour des conditions de trend
        
        close_price_curr = latest_row.get('close')
        atr_value_curr = latest_row.get(self.atr_col_strat)

        # Vérification des valeurs essentielles
        essential_values_check = [ma_s_curr, ma_m_curr, ma_l_curr, ma_s_prev, ma_m_prev, close_price_curr, atr_value_curr]
        
        slope_s_curr: Optional[float] = None
        slope_m_curr: Optional[float] = None
        if self.anticipate_crossovers_enabled and self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat:
            slope_s_curr = latest_row.get(self.slope_ma_short_col_strat)
            slope_m_curr = latest_row.get(self.slope_ma_medium_col_strat)
            essential_values_check.extend([slope_s_curr, slope_m_curr])
        
        if any(pd.isna(val) for val in essential_values_check):
            nan_details = { # Construire le dict des valeurs pour le log
                "ma_s_c": ma_s_curr, "ma_m_c": ma_m_curr, "ma_l_c": ma_l_curr,
                "ma_s_p": ma_s_prev, "ma_m_p": ma_m_prev,
                "close_c": close_price_curr, "atr_c": atr_value_curr,
                "slope_s_c": slope_s_curr, "slope_m_c": slope_m_curr
            }
            self._log(f"Valeurs d'indicateur/prix manquantes (NaN) à {latest_row.name}. Détails: {nan_details}. Signal HOLD.", level=2, is_warning=True)
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        # Logique de croisement standard
        actual_long_entry_cross = (ma_s_curr > ma_m_curr) and (ma_s_prev <= ma_m_prev)
        actual_long_exit_cross = (ma_s_curr < ma_m_curr) and (ma_s_prev >= ma_m_prev)
        
        allow_shorting_param = bool(self.get_param('allow_shorting', False))
        actual_short_entry_cross = False
        actual_short_exit_cross = False
        if allow_shorting_param:
            actual_short_entry_cross = (ma_s_curr < ma_m_curr) and (ma_s_prev >= ma_m_prev)
            actual_short_exit_cross = (ma_s_curr > ma_m_curr) and (ma_s_prev <= ma_m_prev)

        # Logique d'anticipation (si activée et valeurs de pente valides)
        anticipated_long_entry = False
        anticipated_long_exit = False
        anticipated_short_entry = False
        anticipated_short_exit = False

        if self.anticipate_crossovers_enabled and \
           pd.notna(slope_s_curr) and pd.notna(slope_m_curr) and \
           self.anticipation_convergence_threshold_pct_val is not None:
            
            conv_thresh_pct = self.anticipation_convergence_threshold_pct_val
            # Assurer que ma_m_curr est un float pour la multiplication
            convergence_distance = float(ma_m_curr) * conv_thresh_pct
            ma_diff_abs = abs(float(ma_s_curr) - float(ma_m_curr))

            # Anticipation d'entrée LONG
            is_converging_up_for_long = slope_s_curr > slope_m_curr
            is_below_and_closing_for_long = (ma_s_curr < ma_m_curr) and (ma_diff_abs < convergence_distance)
            main_trend_bullish_for_anticip_long = ma_m_curr > ma_l_curr
            anticipated_long_entry = is_converging_up_for_long and is_below_and_closing_for_long and main_trend_bullish_for_anticip_long

            # Anticipation de sortie de LONG
            is_converging_down_for_exit_long = slope_s_curr < slope_m_curr
            is_above_and_closing_for_long_exit = (ma_s_curr > ma_m_curr) and (ma_diff_abs < convergence_distance)
            anticipated_long_exit = is_converging_down_for_exit_long and is_above_and_closing_for_long_exit

            if allow_shorting_param:
                # Anticipation d'entrée SHORT
                is_converging_down_for_entry_short = slope_s_curr < slope_m_curr
                is_above_and_closing_for_short_entry = (ma_s_curr > ma_m_curr) and (ma_diff_abs < convergence_distance)
                main_trend_bearish_for_anticip_short = ma_m_curr < ma_l_curr
                anticipated_short_entry = is_converging_down_for_entry_short and is_above_and_closing_for_short_entry and main_trend_bearish_for_anticip_short
                
                # Anticipation de sortie de SHORT
                is_converging_up_for_exit_short = slope_s_curr > slope_m_curr
                is_below_and_closing_for_short_exit = (ma_s_curr < ma_m_curr) and (ma_diff_abs < convergence_distance)
                anticipated_short_exit = is_converging_up_for_exit_short and is_below_and_closing_for_short_exit
        
        # Combinaison des signaux réels et anticipés
        final_entry_long = (actual_long_entry_cross or anticipated_long_entry) and (ma_m_curr > ma_l_curr)
        final_exit_long = actual_long_exit_cross or anticipated_long_exit
        
        final_entry_short = False
        final_exit_short = False
        if allow_shorting_param:
            final_entry_short = (actual_short_entry_cross or anticipated_short_entry) and (ma_m_curr < ma_l_curr)
            final_exit_short = actual_short_exit_cross or anticipated_short_exit

        # Calcul SL/TP
        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        if current_position_open:
            if current_position_direction == 1 and final_exit_long:
                signal_type = 2 # EXIT LONG
                self._log(f"Signal EXIT LONG @ {close_price_curr:.{self.price_precision or 4}f}. "
                            f"(Réel: {actual_long_exit_cross}, Anticipé: {anticipated_long_exit})", level=1)
            elif current_position_direction == -1 and final_exit_short:
                signal_type = 2 # EXIT SHORT
                self._log(f"Signal EXIT SHORT @ {close_price_curr:.{self.price_precision or 4}f}. "
                            f"(Réel: {actual_short_exit_cross}, Anticipé: {anticipated_short_exit})", level=1)
            else:
                self._log(f"Position ouverte ({'LONG' if current_position_direction == 1 else 'SHORT'}). Pas de signal de sortie actif. Attente SL/TP.", level=3)
        else: # Pas de position ouverte
            if final_entry_long:
                signal_type = 1 # BUY
                if atr_value_curr > 0:
                    sl_price = close_price_curr - (atr_value_curr * sl_atr_mult)
                    tp_price = close_price_curr + (atr_value_curr * tp_atr_mult)
                self._log(f"Signal BUY @ {close_price_curr:.{self.price_precision or 4}f}. SL={sl_price}, TP={tp_price or 'N/A'}. "
                            f"(Réel: {actual_long_entry_cross}, Anticipé: {anticipated_long_entry})", level=1)
            elif final_entry_short: # Seulement si allow_shorting est True et la condition est remplie
                signal_type = -1 # SELL (SHORT)
                if atr_value_curr > 0:
                    sl_price = close_price_curr + (atr_value_curr * sl_atr_mult)
                    tp_price = close_price_curr - (atr_value_curr * tp_atr_mult)
                self._log(f"Signal SELL (SHORT) @ {close_price_curr:.{self.price_precision or 4}f}. SL={sl_price}, TP={tp_price or 'N/A'}. "
                            f"(Réel: {actual_short_entry_cross}, Anticipé: {anticipated_short_entry})", level=1)
        
        order_type_preference = str(self.get_param("order_type_preference", "MARKET"))
        if signal_type != 0 and order_type_preference == "LIMIT":
            limit_price = float(close_price_curr)

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

