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

from src.strategies.base import BaseStrategy
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
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        self.anticipate_crossovers_enabled: bool = bool(params.get('anticipate_crossovers', False))
        self.slope_ma_short_col_strat: Optional[str] = None
        self.slope_ma_medium_col_strat: Optional[str] = None
        self.anticipation_slope_period_val: Optional[int] = None
        self.anticipation_convergence_threshold_pct_val: Optional[float] = None

        if self.anticipate_crossovers_enabled:
            self.slope_ma_short_col_strat = "SLOPE_MA_SHORT_strat"
            self.slope_ma_medium_col_strat = "SLOPE_MA_MEDIUM_strat"

        super().__init__(strategy_name, symbol, params)

        self.ma_short_col_strat: str = "MA_SHORT_strat"
        self.ma_medium_col_strat: str = "MA_MEDIUM_strat"
        self.ma_long_col_strat: str = "MA_LONG_strat"
        self.atr_col_strat: str = "ATR_strat"


    def _validate_params(self) -> None:
        """Valide les paramètres spécifiques à TripleMAAnticipationStrategy."""
        current_required_params = self.REQUIRED_PARAMS[:] 
        if self.anticipate_crossovers_enabled:
            if 'anticipation_slope_period' not in current_required_params:
                current_required_params.append('anticipation_slope_period')
            if 'anticipation_convergence_threshold_pct' not in current_required_params:
                current_required_params.append('anticipation_convergence_threshold_pct')

        # Sourcery: Use named expression (walrus operator)
        if missing_params := [
            p for p in current_required_params if self.get_param(p) is None
        ]:
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
            self.slope_ma_short_col_strat = None 
            self.slope_ma_medium_col_strat = None
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
        
        logger.debug(f"{self.log_prefix} Validation des paramètres terminée. Anticipation activée: {self.anticipate_crossovers_enabled}")


    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
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
        logger.debug(f"{self.log_prefix} Configurations d'indicateurs requises (MAs et ATR, standardisées) : {configs}")
        return configs

    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
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
        # Sourcery: Use named expression (walrus operator)
        if missing_ohlcv := [col for col in base_ohlcv if col not in df.columns]:
            msg = f"{self.log_prefix} Colonnes OHLCV de base manquantes dans data_feed: {missing_ohlcv}. Essentielles."
            logger.critical(msg)
            raise ValueError(msg)

        # Sourcery: Use named expression (walrus operator)
        if missing_initial_strat_cols := [
            col_name
            for col_name in expected_base_strat_cols
            if col_name not in df.columns
        ]:
            for col_name in missing_initial_strat_cols:
                logger.warning(f"{self.log_prefix} Colonne indicateur de base attendue '{col_name}' manquante. Ajout avec NaN.")
                df[col_name] = np.nan
        
        if self.anticipate_crossovers_enabled and self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat and self.anticipation_slope_period_val:
            slope_period = self.anticipation_slope_period_val 

            for ma_col_name, slope_col_name, ma_name_log in [
                (self.ma_short_col_strat, self.slope_ma_short_col_strat, "courte"),
                (self.ma_medium_col_strat, self.slope_ma_medium_col_strat, "moyenne")
            ]:
                if ma_col_name in df.columns and df[ma_col_name].notna().any():
                    ma_series_clean = df[ma_col_name].dropna()
                    if len(ma_series_clean) >= slope_period:
                        slope_series = ta.slope(ma_series_clean.copy(), length=slope_period, append=False) # type: ignore
                        df[slope_col_name] = slope_series.reindex(df.index) 
                        logger.debug(f"{self.log_prefix} Pente MA {ma_name_log} calculée et assignée à '{slope_col_name}'.")
                    else:
                        logger.warning(f"{self.log_prefix} Pas assez de points de données non-NaN ({len(ma_series_clean)}) pour MA {ma_name_log} "
                                       f"pour calculer la pente de période {slope_period}. '{slope_col_name}' sera NaN.")
                        df[slope_col_name] = np.nan
                else: 
                    logger.warning(f"{self.log_prefix} Colonne MA {ma_name_log} ('{ma_col_name}') manquante ou entièrement NaN. "
                                   f"Impossible de calculer sa pente. '{slope_col_name}' sera NaN.")
                    df[slope_col_name] = np.nan
        
        # Sourcery: Use named expression (walrus operator)
        if final_missing_strat_cols_after_calc := [
            col_final_check
            for col_final_check in all_final_expected_strat_cols
            if col_final_check not in df.columns
        ]:
            for col_final_check in final_missing_strat_cols_after_calc:
                 logger.warning(f"{self.log_prefix} Colonne _strat finale attendue '{col_final_check}' toujours manquante après calculs. Ajout avec NaN.")
                 df[col_final_check] = np.nan

        if missing_initial_strat_cols or final_missing_strat_cols_after_calc: # Check if either list has content
             logger.debug(f"{self.log_prefix} Après vérification/calcul dans _calculate_indicators: "
                          f"Colonnes _strat de base manquantes initialement: {missing_initial_strat_cols or 'Aucune'}, "
                          f"Colonnes _strat (pentes) encore manquantes après calculs: {final_missing_strat_cols_after_calc or 'Aucune'}. "
                          f"Colonnes actuelles dans df: {df.columns.tolist()}")

        cols_to_ffill_present_in_df = [col for col in all_final_expected_strat_cols if col in df.columns]
        # Sourcery: Remove redundant conditional
        if cols_to_ffill_present_in_df:
            df[cols_to_ffill_present_in_df] = df[cols_to_ffill_present_in_df].ffill()
            # Le log de ffill a été commenté/supprimé pour réduire la verbosité
        
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

        ma_s_curr = latest_row.get(self.ma_short_col_strat)
        ma_m_curr = latest_row.get(self.ma_medium_col_strat)
        ma_l_curr = latest_row.get(self.ma_long_col_strat)
        ma_s_prev = previous_row.get(self.ma_short_col_strat)
        ma_m_prev = previous_row.get(self.ma_medium_col_strat)
        
        close_price_curr = latest_row.get('close')
        atr_value_curr = latest_row.get(self.atr_col_strat)

        essential_values_check = [ma_s_curr, ma_m_curr, ma_l_curr, ma_s_prev, ma_m_prev, close_price_curr, atr_value_curr]
        
        slope_s_curr: Optional[float] = None
        slope_m_curr: Optional[float] = None
        if self.anticipate_crossovers_enabled:
            if self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat:
                slope_s_curr = latest_row.get(self.slope_ma_short_col_strat)
                slope_m_curr = latest_row.get(self.slope_ma_medium_col_strat)
                essential_values_check.extend([slope_s_curr, slope_m_curr])
            else: 
                logger.warning(f"{self.log_prefix} Anticipation activée mais noms de colonnes de pente non définis. Désactivation pour ce signal.")

        if any(pd.isna(val) for val in essential_values_check):
            nan_details = {
                "ma_s_c": ma_s_curr, "ma_m_c": ma_m_curr, "ma_l_c": ma_l_curr,
                "ma_s_p": ma_s_prev, "ma_m_p": ma_m_prev,
                "close_c": close_price_curr, "atr_c": atr_value_curr,
                "slope_s_c": slope_s_curr, "slope_m_c": slope_m_curr 
            }
            logger.debug(f"{self.log_prefix} Valeurs d'indicateur/prix manquantes (NaN) à {latest_row.name}. Détails: {nan_details}. Signal HOLD.")
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        actual_long_entry_cross = (ma_s_curr > ma_m_curr) and (ma_s_prev <= ma_m_prev) # type: ignore
        actual_long_exit_cross = (ma_s_curr < ma_m_curr) and (ma_s_prev >= ma_m_prev) # type: ignore
        
        allow_shorting_param = bool(self.get_param('allow_shorting', False))
        actual_short_entry_cross = False
        actual_short_exit_cross = False
        if allow_shorting_param:
            actual_short_entry_cross = (ma_s_curr < ma_m_curr) and (ma_s_prev >= ma_m_prev) # type: ignore
            actual_short_exit_cross = (ma_s_curr > ma_m_curr) and (ma_s_prev <= ma_m_prev) # type: ignore

        anticipated_long_entry = False
        anticipated_long_exit = False
        anticipated_short_entry = False
        anticipated_short_exit = False

        if self.anticipate_crossovers_enabled and pd.notna(slope_s_curr) and pd.notna(slope_m_curr) and \
           self.anticipation_convergence_threshold_pct_val is not None:
            
            conv_thresh_pct = self.anticipation_convergence_threshold_pct_val
            convergence_distance = ma_m_curr * conv_thresh_pct # type: ignore
            ma_diff_abs = abs(ma_s_curr - ma_m_curr) # type: ignore

            is_converging_up_for_long = slope_s_curr > slope_m_curr # type: ignore 
            is_below_and_closing_for_long = (ma_s_curr < ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore 
            main_trend_bullish_for_anticip_long = ma_m_curr > ma_l_curr # type: ignore 
            anticipated_long_entry = is_converging_up_for_long and is_below_and_closing_for_long and main_trend_bullish_for_anticip_long

            is_converging_down_for_exit_long = slope_s_curr < slope_m_curr # type: ignore
            is_above_and_closing_for_long_exit = (ma_s_curr > ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore
            anticipated_long_exit = is_converging_down_for_exit_long and is_above_and_closing_for_long_exit

            if allow_shorting_param:
                is_converging_down_for_entry_short = slope_s_curr < slope_m_curr # type: ignore
                is_above_and_closing_for_short_entry = (ma_s_curr > ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore
                main_trend_bearish_for_anticip_short = ma_m_curr < ma_l_curr # type: ignore
                anticipated_short_entry = is_converging_down_for_entry_short and is_above_and_closing_for_short_entry and main_trend_bearish_for_anticip_short
                
                is_converging_up_for_exit_short = slope_s_curr > slope_m_curr # type: ignore
                is_below_and_closing_for_short_exit = (ma_s_curr < ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore
                anticipated_short_exit = is_converging_up_for_exit_short and is_below_and_closing_for_short_exit
        
        final_entry_long = (actual_long_entry_cross or anticipated_long_entry) and (ma_m_curr > ma_l_curr) # type: ignore 
        final_exit_long = actual_long_exit_cross or anticipated_long_exit
        
        final_entry_short = False
        final_exit_short = False
        if allow_shorting_param:
            final_entry_short = (actual_short_entry_cross or anticipated_short_entry) and (ma_m_curr < ma_l_curr) # type: ignore 
            final_exit_short = actual_short_exit_cross or anticipated_short_exit

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        # Sourcery: Move assignments closer to their usage, Swap if/else branches, Merge else clause's nested if statement into elif
        if current_position_open:
            if current_position_direction == 1 and final_exit_long:
                signal_type = 2
                logger.info(f"{self.log_prefix} Signal EXIT LONG @ {close_price_curr:.4f}. "
                            f"(Croisement réel: {actual_long_exit_cross}, Anticipé: {anticipated_long_exit})")
            elif current_position_direction == -1 and final_exit_short:
                signal_type = 2
                logger.info(f"{self.log_prefix} Signal EXIT SHORT @ {close_price_curr:.4f}. "
                            f"(Croisement réel: {actual_short_exit_cross}, Anticipé: {anticipated_short_exit})")
            else:
                logger.debug(
                    f"{self.log_prefix} Position ouverte ({'LONG' if current_position_direction == 1 else 'SHORT'}). Aucun signal de sortie actif. Attente SL/TP."
                )
        elif final_entry_long:
            signal_type = 1
            if atr_value_curr > 0: # type: ignore
                sl_price = close_price_curr - (atr_value_curr * sl_atr_mult) # type: ignore
                tp_price = close_price_curr + (atr_value_curr * tp_atr_mult) # type: ignore
            # Sourcery: Replace if-expression with `or`
            logger.info(f"{self.log_prefix} Signal BUY @ {close_price_curr:.4f}. SL={sl_price}, TP={tp_price or 'N/A'}. "
                        f"(Croisement réel: {actual_long_entry_cross}, Anticipé: {anticipated_long_entry})")
        elif final_entry_short:
            signal_type = -1
            if atr_value_curr > 0: # type: ignore
                sl_price = close_price_curr + (atr_value_curr * sl_atr_mult) # type: ignore
                tp_price = close_price_curr - (atr_value_curr * tp_atr_mult) # type: ignore
            # Sourcery: Replace if-expression with `or`
            logger.info(f"{self.log_prefix} Signal SELL (SHORT) @ {close_price_curr:.4f}. SL={sl_price}, TP={tp_price or 'N/A'}. "
                        f"(Croisement réel: {actual_short_entry_cross}, Anticipé: {anticipated_short_entry})")
        
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
            logger.debug(f"{self.log_prefix} [Live] Position déjà ouverte (état: {current_position}). Pas de nouvelle requête d'ordre d'entrée.")
            return None

        data_with_indicators = self._calculate_indicators(data.copy())
        if data_with_indicators.empty or len(data_with_indicators) < 2:
            logger.warning(f"{self.log_prefix} [Live] Données insuffisantes après _calculate_indicators pour generate_order_request.")
            return None

        signal, order_type, limit_price_sugg, sl_price_raw, tp_price_raw, pos_size_pct = \
            self._generate_signals(data_with_indicators, False, 0, 0.0) 

        if signal not in [1, -1]: 
            logger.debug(f"{self.log_prefix} [Live] Aucun signal d'entrée (1 ou -1) généré. Signal actuel: {signal}")
            return None
        
        if signal == -1 and not bool(self.get_param('allow_shorting', False)):
            logger.debug(f"{self.log_prefix} [Live] Signal Short (-1) généré, mais 'allow_shorting' est False. Pas d'ordre.")
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
            # Sourcery: Use named expression (walrus operator)
            if (adjusted_limit_price := adjust_precision(
                limit_price_sugg, self.price_precision, tick_size=tick_size_price
            )) is None: 
                return None
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
