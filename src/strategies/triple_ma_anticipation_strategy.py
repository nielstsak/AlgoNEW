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
import pandas as pd
import pandas_ta as ta # Pour le calcul de la pente (slope)

from src.strategies.base import BaseStrategy
from src.data.data_utils import get_kline_prefix_effective # Import pour déterminer les colonnes sources

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
        super().__init__(strategy_name, symbol, params)

        self.ma_short_col_strat: str = "MA_SHORT_strat"
        self.ma_medium_col_strat: str = "MA_MEDIUM_strat"
        self.ma_long_col_strat: str = "MA_LONG_strat"
        self.atr_col_strat: str = "ATR_strat"

        # Ces attributs seront correctement initialisés après l'appel à _validate_params
        self.anticipate_crossovers_enabled: bool = False
        self.slope_ma_short_col_strat: Optional[str] = None
        self.slope_ma_medium_col_strat: Optional[str] = None
        self.anticipation_slope_period_val: Optional[int] = None
        self.anticipation_convergence_threshold_pct_val: Optional[float] = None
        
        # _validate_params est appelé par le __init__ de BaseStrategy,
        # donc les attributs ci-dessus seront mis à jour si anticipate_crossovers est True.

    def _validate_params(self) -> None:
        """Valide les paramètres spécifiques à TripleMAAnticipationStrategy."""
        # Déterminer si l'anticipation est activée pour ajuster les paramètres requis
        self.anticipate_crossovers_enabled = bool(self.get_param('anticipate_crossovers', False))

        current_required = self.REQUIRED_PARAMS[:] # Copie pour modification locale
        if self.anticipate_crossovers_enabled:
            if 'anticipation_slope_period' not in current_required:
                current_required.append('anticipation_slope_period')
            if 'anticipation_convergence_threshold_pct' not in current_required:
                current_required.append('anticipation_convergence_threshold_pct')

        missing_params = [p for p in current_required if self.get_param(p) is None]
        if missing_params:
            raise ValueError(f"{self.log_prefix} Paramètres requis manquants : {', '.join(missing_params)}")

        ma_s = self.get_param('ma_short_period')
        ma_m = self.get_param('ma_medium_period')
        ma_l = self.get_param('ma_long_period')
        if not (isinstance(ma_s, int) and ma_s > 0 and isinstance(ma_m, int) and ma_m > 0 and isinstance(ma_l, int) and ma_l > 0):
            raise ValueError(f"{self.log_prefix} Les périodes MA (short, medium, long) doivent être des entiers positifs.")
        if not (ma_s < ma_m < ma_l):
            raise ValueError(f"{self.log_prefix} Les périodes MA doivent être en ordre croissant : short ({ma_s}) < medium ({ma_m}) < long ({ma_l}).")

        ma_type_val = self.get_param('ma_type')
        supported_ma_types = ['sma', 'ema', 'wma', 'hma', 'tema', 'dema', 'rma', 'vwma', 'zlma']
        if not isinstance(ma_type_val, str) or ma_type_val.lower() not in supported_ma_types:
            raise ValueError(f"{self.log_prefix} 'ma_type' ({ma_type_val}) invalide. Supportés : {supported_ma_types}")

        if self.anticipate_crossovers_enabled:
            self.slope_ma_short_col_strat = "SLOPE_MA_SHORT_strat" # Définir les noms de colonnes ici
            self.slope_ma_medium_col_strat = "SLOPE_MA_MEDIUM_strat"

            slope_p = self.get_param('anticipation_slope_period')
            conv_thresh = self.get_param('anticipation_convergence_threshold_pct')
            if not (isinstance(slope_p, int) and slope_p >= 2): # pandas_ta.slope requiert length >= 2
                raise ValueError(f"{self.log_prefix} 'anticipation_slope_period' ({slope_p}) doit être un entier >= 2 si l'anticipation est activée.")
            if not (isinstance(conv_thresh, (int, float)) and 0 < conv_thresh < 1): # Généralement un petit pourcentage
                raise ValueError(f"{self.log_prefix} 'anticipation_convergence_threshold_pct' ({conv_thresh}) doit être un float entre 0 (exclusif) et 1 (exclusif) si l'anticipation est activée.")
            self.anticipation_slope_period_val = slope_p
            self.anticipation_convergence_threshold_pct_val = float(conv_thresh)
        else: # S'assurer que les attributs liés à l'anticipation sont None si désactivée
            self.slope_ma_short_col_strat = None
            self.slope_ma_medium_col_strat = None
            self.anticipation_slope_period_val = None
            self.anticipation_convergence_threshold_pct_val = None


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

        for freq_param_name in ['indicateur_frequence_mms', 'indicateur_frequence_mmm', 'indicateur_frequence_mml', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' doit être une chaîne non vide.")
        
        logger.debug(f"{self.log_prefix} Validation des paramètres terminée avec succès. Anticipation activée: {self.anticipate_crossovers_enabled}")


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
        """
        Vérifie la présence des colonnes MA et ATR (fournies par IndicatorCalculator),
        calcule les pentes des MAs si l'anticipation est activée, et applique ffill.
        """
        df = data_feed.copy()
        
        expected_base_strat_cols = [
            self.ma_short_col_strat, self.ma_medium_col_strat,
            self.ma_long_col_strat, self.atr_col_strat
        ]
        all_expected_cols = expected_base_strat_cols[:] # Copie pour ajouter les pentes si besoin

        # Si l'anticipation est activée, les colonnes de pente sont aussi attendues (ou seront créées)
        if self.anticipate_crossovers_enabled:
            if self.slope_ma_short_col_strat: all_expected_cols.append(self.slope_ma_short_col_strat)
            if self.slope_ma_medium_col_strat: all_expected_cols.append(self.slope_ma_medium_col_strat)

        # Vérifier les colonnes OHLCV de base
        base_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlcv = [col for col in base_ohlcv if col not in df.columns]
        if missing_ohlcv:
            msg = f"{self.log_prefix} Colonnes OHLCV de base manquantes dans data_feed: {missing_ohlcv}."
            logger.critical(msg)
            raise ValueError(msg)

        # Vérifier les colonnes d'indicateurs de base (MAs, ATR)
        missing_strat_cols = []
        for col_name in expected_base_strat_cols: # Seulement les MAs et ATR ici
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur de base attendue '{col_name}' manquante. "
                               "Elle sera ajoutée avec NaN, indiquant un problème en amont (IndicatorCalculator).")
                df[col_name] = np.nan
                missing_strat_cols.append(col_name)
        
        # Calculer les pentes si l'anticipation est activée et que les MAs sont présentes
        if self.anticipate_crossovers_enabled and self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat and self.anticipation_slope_period_val:
            slope_period = self.anticipation_slope_period_val 

            if self.ma_short_col_strat in df.columns and df[self.ma_short_col_strat].notna().any():
                ma_short_clean = df[self.ma_short_col_strat].dropna()
                if len(ma_short_clean) >= slope_period:
                    slope_short_series = ta.slope(ma_short_clean.copy(), length=slope_period, append=False) # type: ignore
                    df[self.slope_ma_short_col_strat] = slope_short_series.reindex(df.index)
                    logger.debug(f"{self.log_prefix} Pente MA courte calculée.")
                else:
                    logger.warning(f"{self.log_prefix} Pas assez de points de données non-NaN ({len(ma_short_clean)}) pour MA courte pour calculer la pente de période {slope_period}. {self.slope_ma_short_col_strat} sera NaN.")
                    df[self.slope_ma_short_col_strat] = np.nan
            else: # ma_short_col_strat n'est pas dans df ou est entièrement NaN
                logger.warning(f"{self.log_prefix} Colonne {self.ma_short_col_strat} manquante ou entièrement NaN. Impossible de calculer sa pente. {self.slope_ma_short_col_strat} sera NaN.")
                df[self.slope_ma_short_col_strat] = np.nan

            if self.ma_medium_col_strat in df.columns and df[self.ma_medium_col_strat].notna().any():
                ma_medium_clean = df[self.ma_medium_col_strat].dropna()
                if len(ma_medium_clean) >= slope_period:
                    slope_medium_series = ta.slope(ma_medium_clean.copy(), length=slope_period, append=False) # type: ignore
                    df[self.slope_ma_medium_col_strat] = slope_medium_series.reindex(df.index)
                    logger.debug(f"{self.log_prefix} Pente MA moyenne calculée.")
                else:
                    logger.warning(f"{self.log_prefix} Pas assez de points de données non-NaN ({len(ma_medium_clean)}) pour MA moyenne pour calculer la pente de période {slope_period}. {self.slope_ma_medium_col_strat} sera NaN.")
                    df[self.slope_ma_medium_col_strat] = np.nan
            else: # ma_medium_col_strat n'est pas dans df ou est entièrement NaN
                logger.warning(f"{self.log_prefix} Colonne {self.ma_medium_col_strat} manquante ou entièrement NaN. Impossible de calculer sa pente. {self.slope_ma_medium_col_strat} sera NaN.")
                df[self.slope_ma_medium_col_strat] = np.nan
        
        # Vérifier si des colonnes _strat (y compris pentes) sont encore manquantes après calculs
        # et les initialiser à NaN si c'est le cas.
        final_missing_strat_cols = []
        for col_name_final_check in all_expected_cols:
            if col_name_final_check not in df.columns:
                 logger.warning(f"{self.log_prefix} Colonne _strat finale attendue '{col_name_final_check}' toujours manquante. Ajout avec NaN.")
                 df[col_name_final_check] = np.nan
                 final_missing_strat_cols.append(col_name_final_check)

        if missing_strat_cols or final_missing_strat_cols: # Si des colonnes _strat de base ou des pentes ont été ajoutées
             logger.debug(f"{self.log_prefix} Après vérification/calcul _calculate_indicators, "
                          f"colonnes _strat manquantes initialement: {missing_strat_cols}, "
                          f"colonnes _strat encore manquantes après calculs (pentes): {final_missing_strat_cols}. "
                          f"Colonnes actuelles: {df.columns.tolist()}")

        # Appliquer ffill à toutes les colonnes _strat présentes
        cols_to_ffill_present = [col for col in all_expected_cols if col in df.columns]
        if cols_to_ffill_present:
            df[cols_to_ffill_present] = df[cols_to_ffill_present].ffill()
            logger.debug(f"{self.log_prefix} ffill appliqué aux colonnes _strat présentes : {cols_to_ffill_present}")
        
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

        essential_values = [ma_s_curr, ma_m_curr, ma_l_curr, ma_s_prev, ma_m_prev, close_price_curr, atr_value_curr]
        
        slope_s_curr: Optional[float] = None
        slope_m_curr: Optional[float] = None
        if self.anticipate_crossovers_enabled: # Utiliser l'attribut de classe
            if self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat:
                slope_s_curr = latest_row.get(self.slope_ma_short_col_strat)
                slope_m_curr = latest_row.get(self.slope_ma_medium_col_strat)
                essential_values.extend([slope_s_curr, slope_m_curr])
            else: # Si les colonnes de pente ne sont pas définies, l'anticipation ne peut pas fonctionner
                logger.warning(f"{self.log_prefix} Anticipation activée mais noms de colonnes de pente non définis. "
                               "Anticipation désactivée pour ce signal.")
                # Pas besoin de modifier self.anticipate_crossovers_enabled ici, juste ne pas utiliser les pentes.

        if any(pd.isna(val) for val in essential_values):
            nan_details = {k:v for k,v in locals().items() if k in ['ma_s_curr', 'ma_m_curr', 'ma_l_curr', 'slope_s_curr', 'slope_m_curr', 'close_price_curr', 'atr_value_curr']}
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

        if self.anticipate_crossovers_enabled and pd.notna(slope_s_curr) and pd.notna(slope_m_curr) and self.anticipation_convergence_threshold_pct_val is not None:
            conv_thresh_pct = self.anticipation_convergence_threshold_pct_val
            convergence_distance = ma_m_curr * conv_thresh_pct # type: ignore
            ma_diff_abs = abs(ma_s_curr - ma_m_curr) # type: ignore

            is_converging_up = slope_s_curr > slope_m_curr # type: ignore 
            is_below_and_closing_for_long = (ma_s_curr < ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore
            main_trend_bullish_for_anticip = ma_m_curr > ma_l_curr # type: ignore 
            anticipated_long_entry = is_converging_up and is_below_and_closing_for_long and main_trend_bullish_for_anticip

            is_converging_down_for_exit_long = slope_s_curr < slope_m_curr # type: ignore
            is_above_and_closing_for_long_exit = (ma_s_curr > ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore
            anticipated_long_exit = is_converging_down_for_exit_long and is_above_and_closing_for_long_exit

            if allow_shorting_param:
                is_converging_down_for_entry_short = slope_s_curr < slope_m_curr # type: ignore
                is_above_and_closing_for_short_entry = (ma_s_curr > ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore
                main_trend_bearish_for_anticip = ma_m_curr < ma_l_curr # type: ignore
                anticipated_short_entry = is_converging_down_for_entry_short and is_above_and_closing_for_short_entry and main_trend_bearish_for_anticip
                
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

        if not current_position_open:
            if final_entry_long:
                signal_type = 1
                if atr_value_curr > 0: # type: ignore
                    sl_price = close_price_curr - (atr_value_curr * sl_atr_mult) # type: ignore
                    tp_price = close_price_curr + (atr_value_curr * tp_atr_mult) # type: ignore
                logger.info(f"{self.log_prefix} Signal BUY @ {close_price_curr:.4f}. SL={sl_price}, TP={tp_price}. Anticipated: {anticipated_long_entry}")
            elif final_entry_short: 
                signal_type = -1
                if atr_value_curr > 0: # type: ignore
                    sl_price = close_price_curr + (atr_value_curr * sl_atr_mult) # type: ignore
                    tp_price = close_price_curr - (atr_value_curr * tp_atr_mult) # type: ignore
                logger.info(f"{self.log_prefix} Signal SELL @ {close_price_curr:.4f}. SL={sl_price}, TP={tp_price}. Anticipated: {anticipated_short_entry}")
        else: 
            if current_position_direction == 1 and final_exit_long:
                signal_type = 2 
                logger.info(f"{self.log_prefix} Signal EXIT LONG @ {close_price_curr:.4f}. Anticipated: {anticipated_long_exit}")
            elif current_position_direction == -1 and final_exit_short: 
                signal_type = 2 
                logger.info(f"{self.log_prefix} Signal EXIT SHORT @ {close_price_curr:.4f}. Anticipated: {anticipated_short_exit}")

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
        
        if signal == -1 and not bool(self.get_param('allow_shorting', False)):
            logger.debug(f"{self.log_prefix} [Live] Signal Short (-1) généré, mais 'allow_shorting' est False. Pas d'ordre.")
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
