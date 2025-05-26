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
# Les utilitaires d'exchange_utils sont accessibles via les méthodes de BaseStrategy

logger = logging.getLogger(__name__)

class TripleMAAnticipationStrategy(BaseStrategy):
    """
    Stratégie de trading utilisant trois moyennes mobiles (MA) et une logique
    d'anticipation des croisements.

    Signaux d'entrée :
    - Long : Croisement haussier de la MA courte sur la MA moyenne, avec la MA moyenne
      au-dessus de la MA longue (confirmation de tendance).
      Si l'anticipation est activée, peut entrer avant le croisement si les MAs convergent
      et que leurs pentes le suggèrent.
    - Short (si `allow_shorting` est True) : Croisement baissier de la MA courte sous la MA moyenne,
      avec la MA moyenne sous la MA longue. Logique d'anticipation similaire.

    Les sorties sont gérées par SL/TP basés sur l'ATR ou par des croisements inverses.
    """

    # Paramètres requis de base. D'autres peuvent être requis conditionnellement.
    REQUIRED_PARAMS: List[str] = [
        'ma_short_period', 'ma_medium_period', 'ma_long_period', 'ma_type',
        'indicateur_frequence_mms', # MA Mobile Short
        'indicateur_frequence_mmm', # MA Mobile Medium
        'indicateur_frequence_mml', # MA Mobile Long
        'atr_period_sl_tp', 'atr_base_frequency_sl_tp',
        'sl_atr_mult', 'tp_atr_mult',
        'allow_shorting', 'order_type_preference',
        'capital_allocation_pct',
        'anticipate_crossovers' # Toujours requis pour savoir si les params d'anticipation sont nécessaires
        # 'anticipation_slope_period' # Requis si anticipate_crossovers = True
        # 'anticipation_convergence_threshold_pct' # Requis si anticipate_crossovers = True
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        """
        Initialise la stratégie TripleMAAnticipationStrategy.

        Args:
            strategy_name (str): Nom de la stratégie.
            symbol (str): Symbole de la paire de trading.
            params (Dict[str, Any]): Paramètres spécifiques à cette instance.
        """
        super().__init__(strategy_name, symbol, params)

        # Noms des colonnes pour les indicateurs
        self.ma_short_col_strat: str = "MA_SHORT_strat"
        self.ma_medium_col_strat: str = "MA_MEDIUM_strat"
        self.ma_long_col_strat: str = "MA_LONG_strat"
        self.atr_col_strat: str = "ATR_strat"

        self.anticipate_crossovers_enabled: bool = bool(self.get_param('anticipate_crossovers', False))
        self.slope_ma_short_col_strat: Optional[str] = None
        self.slope_ma_medium_col_strat: Optional[str] = None
        self.anticipation_slope_period_val: Optional[int] = None
        self.anticipation_convergence_threshold_pct_val: Optional[float] = None

        if self.anticipate_crossovers_enabled:
            self.slope_ma_short_col_strat = "SLOPE_MA_SHORT_strat"
            self.slope_ma_medium_col_strat = "SLOPE_MA_MEDIUM_strat"
            # Les valeurs seront récupérées et validées dans _validate_params
            self.anticipation_slope_period_val = self.get_param('anticipation_slope_period')
            self.anticipation_convergence_threshold_pct_val = self.get_param('anticipation_convergence_threshold_pct')
        
        # logger.info(f"{self.log_prefix} Stratégie TripleMAAnticipationStrategy initialisée. Anticipation: {self.anticipate_crossovers_enabled}")


    def _validate_params(self) -> None:
        """
        Valide les paramètres spécifiques à TripleMAAnticipationStrategy.
        """
        current_required = self.REQUIRED_PARAMS[:]
        if self.anticipate_crossovers_enabled:
            if 'anticipation_slope_period' not in current_required:
                current_required.append('anticipation_slope_period')
            if 'anticipation_convergence_threshold_pct' not in current_required:
                current_required.append('anticipation_convergence_threshold_pct')

        missing_params = [p for p in current_required if self.get_param(p) is None]
        if missing_params:
            raise ValueError(f"{self.log_prefix} Paramètres requis manquants : {', '.join(missing_params)}")

        # Validation des périodes MA
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

        # Validation des paramètres d'anticipation si activée
        if self.anticipate_crossovers_enabled:
            slope_p = self.get_param('anticipation_slope_period')
            conv_thresh = self.get_param('anticipation_convergence_threshold_pct')
            if not (isinstance(slope_p, int) and slope_p >= 2):
                raise ValueError(f"{self.log_prefix} 'anticipation_slope_period' ({slope_p}) doit être un entier >= 2 si l'anticipation est activée.")
            if not (isinstance(conv_thresh, (int, float)) and 0 < conv_thresh < 1):
                raise ValueError(f"{self.log_prefix} 'anticipation_convergence_threshold_pct' ({conv_thresh}) doit être un float entre 0 et 1 (exclusif) si l'anticipation est activée.")
            # Stocker les valeurs validées
            self.anticipation_slope_period_val = slope_p
            self.anticipation_convergence_threshold_pct_val = float(conv_thresh)


        # Validation des autres paramètres (similaire à MaCrossoverStrategy)
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
        
        order_type_pref_val = self.get_param('order_type_preference')
        if order_type_pref_val not in ["MARKET", "LIMIT"]:
            raise ValueError(f"{self.log_prefix} 'order_type_preference' doit être 'MARKET' ou 'LIMIT'.")

        for freq_param_name in ['indicateur_frequence_mms', 'indicateur_frequence_mmm', 'indicateur_frequence_mml', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' doit être une chaîne non vide.")
        
        logger.debug(f"{self.log_prefix} Validation des paramètres terminée avec succès.")


    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        """
        Déclare les indicateurs requis par TripleMAAnticipationStrategy.
        Les pentes (slopes) seront calculées dans _calculate_indicators.
        """
        ma_type_to_use = str(self.params.get('ma_type', 'ema')).lower()
        configs = [
            {
                'indicator_name': ma_type_to_use,
                'params': {'length': int(self.params['ma_short_period'])},
                'source_kline_frequency_param_name': 'indicateur_frequence_mms',
                'output_column_name': self.ma_short_col_strat
            },
            {
                'indicator_name': ma_type_to_use,
                'params': {'length': int(self.params['ma_medium_period'])},
                'source_kline_frequency_param_name': 'indicateur_frequence_mmm',
                'output_column_name': self.ma_medium_col_strat
            },
            {
                'indicator_name': ma_type_to_use,
                'params': {'length': int(self.params['ma_long_period'])},
                'source_kline_frequency_param_name': 'indicateur_frequence_mml',
                'output_column_name': self.ma_long_col_strat
            },
            {
                'indicator_name': 'atr',
                'params': {'length': int(self.params['atr_period_sl_tp'])},
                'source_kline_frequency_param_name': 'atr_base_frequency_sl_tp',
                'output_column_name': self.atr_col_strat
            }
        ]
        logger.debug(f"{self.log_prefix} Configurations d'indicateurs requises (MAs et ATR) : {configs}")
        return configs

    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie la présence des colonnes MA et ATR, et calcule les pentes des MAs
        si l'anticipation est activée.
        """
        df = data_feed.copy()
        
        # Vérifier les MAs et ATR (devraient être fournies par IndicatorCalculator)
        expected_base_strat_cols = [
            self.ma_short_col_strat, self.ma_medium_col_strat,
            self.ma_long_col_strat, self.atr_col_strat
        ]
        missing_cols_added_nan = False
        for col_name in expected_base_strat_cols:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur de base attendue '{col_name}' manquante. Ajout avec NaN.")
                df[col_name] = np.nan
                missing_cols_added_nan = True
        
        # Calculer les pentes si l'anticipation est activée
        if self.anticipate_crossovers_enabled and self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat and self.anticipation_slope_period_val:
            slope_period = self.anticipation_slope_period_val
            
            if self.ma_short_col_strat in df.columns and df[self.ma_short_col_strat].notna().any():
                # S'assurer qu'il y a assez de points pour la pente après avoir enlevé les NaNs initiaux de la MA
                ma_short_clean = df[self.ma_short_col_strat].dropna()
                if len(ma_short_clean) >= slope_period:
                    slope_short_series = ta.slope(ma_short_clean, length=slope_period, append=False) # type: ignore
                    df[self.slope_ma_short_col_strat] = slope_short_series.reindex(df.index, method='ffill')
                    logger.debug(f"{self.log_prefix} Pente MA courte calculée.")
                else:
                    logger.warning(f"{self.log_prefix} Pas assez de points de données non-NaN ({len(ma_short_clean)}) pour MA courte pour calculer la pente de période {slope_period}. {self.slope_ma_short_col_strat} sera NaN.")
                    df[self.slope_ma_short_col_strat] = np.nan
            else:
                logger.warning(f"{self.log_prefix} Colonne {self.ma_short_col_strat} manquante ou entièrement NaN. Impossible de calculer la pente. {self.slope_ma_short_col_strat} sera NaN.")
                df[self.slope_ma_short_col_strat] = np.nan

            if self.ma_medium_col_strat in df.columns and df[self.ma_medium_col_strat].notna().any():
                ma_medium_clean = df[self.ma_medium_col_strat].dropna()
                if len(ma_medium_clean) >= slope_period:
                    slope_medium_series = ta.slope(ma_medium_clean, length=slope_period, append=False) # type: ignore
                    df[self.slope_ma_medium_col_strat] = slope_medium_series.reindex(df.index, method='ffill')
                    logger.debug(f"{self.log_prefix} Pente MA moyenne calculée.")
                else:
                    logger.warning(f"{self.log_prefix} Pas assez de points de données non-NaN ({len(ma_medium_clean)}) pour MA moyenne pour calculer la pente de période {slope_period}. {self.slope_ma_medium_col_strat} sera NaN.")
                    df[self.slope_ma_medium_col_strat] = np.nan
            else:
                logger.warning(f"{self.log_prefix} Colonne {self.ma_medium_col_strat} manquante ou entièrement NaN. Impossible de calculer la pente. {self.slope_ma_medium_col_strat} sera NaN.")
                df[self.slope_ma_medium_col_strat] = np.nan
            
            if self.slope_ma_short_col_strat in df.columns: missing_cols_added_nan = missing_cols_added_nan or df[self.slope_ma_short_col_strat].isnull().all()
            if self.slope_ma_medium_col_strat in df.columns: missing_cols_added_nan = missing_cols_added_nan or df[self.slope_ma_medium_col_strat].isnull().all()


        # Vérifier les colonnes OHLCV de base
        base_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        for col in base_ohlcv:
            if col not in df.columns:
                logger.error(f"{self.log_prefix} Colonne OHLCV de base '{col}' manquante.")
                df[col] = np.nan
                missing_cols_added_nan = True
        
        if missing_cols_added_nan:
             logger.debug(f"{self.log_prefix} Après vérification/calcul _calculate_indicators, colonnes : {df.columns.tolist()}")

        # Appliquer ffill aux colonnes _strat
        cols_to_ffill = [col for col in df.columns if col.endswith('_strat')]
        if cols_to_ffill:
            df[cols_to_ffill] = df[cols_to_ffill].ffill()
            logger.debug(f"{self.log_prefix} ffill appliqué aux colonnes _strat : {cols_to_ffill}")
        return df

    def _generate_signals(self,
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int,
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Génère les signaux de trading pour la stratégie Triple MA avec anticipation.
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

        # Récupérer les MAs
        ma_s_curr = latest_row.get(self.ma_short_col_strat)
        ma_m_curr = latest_row.get(self.ma_medium_col_strat)
        ma_l_curr = latest_row.get(self.ma_long_col_strat)
        ma_s_prev = previous_row.get(self.ma_short_col_strat)
        ma_m_prev = previous_row.get(self.ma_medium_col_strat)
        
        close_price_curr = latest_row.get('close')
        atr_value_curr = latest_row.get(self.atr_col_strat)

        essential_values = [ma_s_curr, ma_m_curr, ma_l_curr, ma_s_prev, ma_m_prev, close_price_curr, atr_value_curr]
        
        # Pentes (si anticipation activée)
        slope_s_curr: Optional[float] = None
        slope_m_curr: Optional[float] = None
        if self.anticipate_crossovers_enabled and self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat:
            slope_s_curr = latest_row.get(self.slope_ma_short_col_strat)
            slope_m_curr = latest_row.get(self.slope_ma_medium_col_strat)
            essential_values.extend([slope_s_curr, slope_m_curr])

        if any(pd.isna(val) for val in essential_values):
            nan_details = {k:v for k,v in locals().items() if k in ['ma_s_curr', 'ma_m_curr', 'ma_l_curr', 'slope_s_curr', 'slope_m_curr', 'close_price_curr', 'atr_value_curr']}
            logger.debug(f"{self.log_prefix} Valeurs d'indicateur/prix manquantes (NaN) à {latest_row.name}. Détails: {nan_details}. Signal HOLD.")
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        # Conditions de base pour les croisements
        actual_long_entry_cross = (ma_s_curr > ma_m_curr) and (ma_s_prev <= ma_m_prev) # type: ignore
        actual_long_exit_cross = (ma_s_curr < ma_m_curr) and (ma_s_prev >= ma_m_prev) # type: ignore
        
        allow_shorting_param = bool(self.get_param('allow_shorting', False))
        actual_short_entry_cross = False
        actual_short_exit_cross = False
        if allow_shorting_param:
            actual_short_entry_cross = (ma_s_curr < ma_m_curr) and (ma_s_prev >= ma_m_prev) # type: ignore
            actual_short_exit_cross = (ma_s_curr > ma_m_curr) and (ma_s_prev <= ma_m_prev) # type: ignore

        # Conditions d'anticipation
        anticipated_long_entry = False
        anticipated_long_exit = False
        anticipated_short_entry = False
        anticipated_short_exit = False

        if self.anticipate_crossovers_enabled and pd.notna(slope_s_curr) and pd.notna(slope_m_curr) and self.anticipation_convergence_threshold_pct_val is not None:
            conv_thresh_pct = self.anticipation_convergence_threshold_pct_val
            # Distance de convergence (en % de la MA moyenne)
            convergence_distance = ma_m_curr * conv_thresh_pct # type: ignore
            ma_diff_abs = abs(ma_s_curr - ma_m_curr) # type: ignore

            # Anticipation Entrée Long
            is_converging_up = slope_s_curr > slope_m_curr # type: ignore # Pente courte > Pente moyenne
            is_below_and_closing_for_long = (ma_s_curr < ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore
            main_trend_bullish_for_anticip = ma_m_curr > ma_l_curr # type: ignore # Tendance de fond haussière
            anticipated_long_entry = is_converging_up and is_below_and_closing_for_long and main_trend_bullish_for_anticip

            # Anticipation Sortie Long
            is_converging_down_for_exit_long = slope_s_curr < slope_m_curr # type: ignore
            is_above_and_closing_for_long_exit = (ma_s_curr > ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore
            anticipated_long_exit = is_converging_down_for_exit_long and is_above_and_closing_for_long_exit

            if allow_shorting_param:
                # Anticipation Entrée Short
                is_converging_down_for_entry_short = slope_s_curr < slope_m_curr # type: ignore
                is_above_and_closing_for_short_entry = (ma_s_curr > ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore
                main_trend_bearish_for_anticip = ma_m_curr < ma_l_curr # type: ignore
                anticipated_short_entry = is_converging_down_for_entry_short and is_above_and_closing_for_short_entry and main_trend_bearish_for_anticip
                
                # Anticipation Sortie Short
                is_converging_up_for_exit_short = slope_s_curr > slope_m_curr # type: ignore
                is_below_and_closing_for_short_exit = (ma_s_curr < ma_m_curr) and (ma_diff_abs < convergence_distance) # type: ignore
                anticipated_short_exit = is_converging_up_for_exit_short and is_below_and_closing_for_short_exit
        
        # Combinaison des signaux réels et anticipés
        final_entry_long = (actual_long_entry_cross or anticipated_long_entry) and (ma_m_curr > ma_l_curr) # type: ignore # Confirmer tendance de fond
        final_exit_long = actual_long_exit_cross or anticipated_long_exit
        
        final_entry_short = False
        final_exit_short = False
        if allow_shorting_param:
            final_entry_short = (actual_short_entry_cross or anticipated_short_entry) and (ma_m_curr < ma_l_curr) # type: ignore # Confirmer tendance de fond
            final_exit_short = actual_short_exit_cross or anticipated_short_exit

        # Logique de décision
        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        if not current_position_open:
            if final_entry_long:
                signal_type = 1
                if atr_value_curr > 0: # type: ignore
                    sl_price = close_price_curr - (atr_value_curr * sl_atr_mult) # type: ignore
                    tp_price = close_price_curr + (atr_value_curr * tp_atr_mult) # type: ignore
                logger.info(f"{self.log_prefix} Signal BUY @ {close_price_curr:.4f}. SL={sl_price}, TP={tp_price}. Anticipated: {anticipated_long_entry}")
            elif final_entry_short: # Seulement si allow_shorting est vrai
                signal_type = -1
                if atr_value_curr > 0: # type: ignore
                    sl_price = close_price_curr + (atr_value_curr * sl_atr_mult) # type: ignore
                    tp_price = close_price_curr - (atr_value_curr * tp_atr_mult) # type: ignore
                logger.info(f"{self.log_prefix} Signal SELL @ {close_price_curr:.4f}. SL={sl_price}, TP={tp_price}. Anticipated: {anticipated_short_entry}")
        else: # Position ouverte
            if current_position_direction == 1 and final_exit_long:
                signal_type = 2 # Signal de sortie de position Long
                logger.info(f"{self.log_prefix} Signal EXIT LONG @ {close_price_curr:.4f}. Anticipated: {anticipated_long_exit}")
            elif current_position_direction == -1 and final_exit_short: # Seulement si allow_shorting
                signal_type = 2 # Signal de sortie de position Short
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
        """
        Génère une requête d'ordre d'ENTRÉE pour le trading en direct.
        """
        if current_position != 0:
            logger.debug(f"{self.log_prefix} [Live] Position déjà ouverte (état: {current_position}). Pas de nouvelle requête d'ordre d'entrée.")
            return None

        data_with_indicators = self._calculate_indicators(data.copy())
        if data_with_indicators.empty or len(data_with_indicators) < 2:
            logger.warning(f"{self.log_prefix} [Live] Données insuffisantes après _calculate_indicators pour generate_order_request.")
            return None

        signal, order_type, limit_price_sugg, sl_price_raw, tp_price_raw, pos_size_pct = \
            self._generate_signals(data_with_indicators, False, 0, 0.0)

        if signal not in [1, -1]: # Uniquement signaux d'entrée Long ou Short
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
            adjusted_limit_price = adjust_precision(limit_price_sugg, self.price_precision, get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize'))
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

