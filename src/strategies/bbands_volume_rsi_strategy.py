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

    Conditions d'entrée :
    - Long : Cassure de la bande supérieure de Bollinger, confirmée par un volume
      supérieur à sa moyenne mobile et un RSI au-dessus d'un seuil de surachat.
    - Short : Cassure de la bande inférieure de Bollinger, confirmée par un volume
      supérieur à sa moyenne mobile et un RSI en dessous d'un seuil de survente.
    Les sorties sont gérées par SL/TP basés sur l'ATR.
    """

    REQUIRED_PARAMS: List[str] = [
        'bbands_period', 'bbands_std_dev', 'indicateur_frequence_bbands',
        'volume_ma_period', 'indicateur_frequence_volume', # Fréquence pour la source du volume et sa MA
        'rsi_period', 'indicateur_frequence_rsi',
        'rsi_buy_breakout_threshold', 'rsi_sell_breakout_threshold',
        'atr_period_sl_tp', 'atr_base_frequency_sl_tp',
        'sl_atr_mult', 'tp_atr_mult',
        'capital_allocation_pct',
        'order_type_preference'
        # 'allow_shorting' pourrait être ajouté si la stratégie doit le gérer explicitement
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        """
        Initialise la stratégie BbandsVolumeRsiStrategy.

        Args:
            strategy_name (str): Nom de la stratégie.
            symbol (str): Symbole de la paire de trading.
            params (Dict[str, Any]): Paramètres spécifiques à cette instance.
        """
        super().__init__(strategy_name, symbol, params)

        # Noms des colonnes pour les indicateurs calculés
        self.bb_upper_col_strat: str = "BB_UPPER_strat"
        self.bb_middle_col_strat: str = "BB_MIDDLE_strat"
        self.bb_lower_col_strat: str = "BB_LOWER_strat"
        self.bb_bandwidth_col_strat: str = "BB_BANDWIDTH_strat"
        
        self.volume_ma_col_strat: str = "Volume_MA_strat"
        self.rsi_col_strat: str = "RSI_strat"
        self.atr_col_strat: str = "ATR_strat"

        # Déterminer la colonne source pour le volume (brut, avant MA)
        # Cette colonne doit exister dans le df_source_enriched fourni à IndicatorCalculator
        vol_freq_param = self.get_param('indicateur_frequence_volume')
        kline_prefix_vol_src = get_kline_prefix_effective(str(vol_freq_param))
        self.volume_kline_col_strat: str = f"{kline_prefix_vol_src}_volume" if kline_prefix_vol_src else "volume"
        
        logger.info(f"{self.log_prefix} Colonne source de volume pour la stratégie (avant MA) : '{self.volume_kline_col_strat}'")


    def _validate_params(self) -> None:
        """
        Valide les paramètres spécifiques à BbandsVolumeRsiStrategy.
        """
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
        
        order_type_pref_val = self.get_param('order_type_preference')
        if order_type_pref_val not in ["MARKET", "LIMIT"]:
            raise ValueError(f"{self.log_prefix} 'order_type_preference' doit être 'MARKET' ou 'LIMIT'.")

        for freq_param_name in ['indicateur_frequence_bbands', 'indicateur_frequence_volume', 'indicateur_frequence_rsi', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' doit être une chaîne non vide.")
        logger.debug(f"{self.log_prefix} Validation des paramètres terminée avec succès.")

    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        """
        Déclare les indicateurs requis par BbandsVolumeRsiStrategy.
        """
        # pandas-ta bbands retourne un DataFrame. IndicatorCalculator doit mapper les colonnes.
        # Les clés dans 'output_column_map' sont les noms des colonnes retournées par ta.bbands
        # (ex: BBL_length_std, BBM_length_std, etc.)
        bbands_params = {'length': int(self.params['bbands_period']), 'std': float(self.params['bbands_std_dev'])}
        bbands_length = bbands_params['length']
        bbands_std = bbands_params['std']
        
        # Construire les noms de colonnes attendus de pandas-ta.bbands
        # Ex: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0
        # Note: Le format exact peut dépendre de la version de pandas-ta.
        # Il est plus sûr si IndicatorCalculator a une logique pour trouver les colonnes par des sous-chaînes (BBL, BBM, BBU, BBB).
        # Pour l'instant, on suppose que IndicatorCalculator peut gérer un mapping conceptuel.
        bbands_output_map = {
            f"BBL_{bbands_length}_{bbands_std}": self.bb_lower_col_strat,
            f"BBM_{bbands_length}_{bbands_std}": self.bb_middle_col_strat,
            f"BBU_{bbands_length}_{bbands_std}": self.bb_upper_col_strat,
            f"BBB_{bbands_length}_{bbands_std}": self.bb_bandwidth_col_strat
            # Si pandas-ta retourne des noms plus simples comme 'BBL', 'BBM', etc., ajuster ici.
            # Alternativement, IndicatorCalculator peut avoir une logique de mapping plus intelligente.
            # Pour ce prompt, on utilise une clé générique que IndicatorCalculator devra interpréter.
            # 'LOWER': self.bb_lower_col_strat,
            # 'MIDDLE': self.bb_middle_col_strat,
            # 'UPPER': self.bb_upper_col_strat,
            # 'BANDWIDTH': self.bb_bandwidth_col_strat
        }


        configs = [
            {
                'indicator_name': 'bbands',
                'params': bbands_params,
                'source_kline_frequency_param_name': 'indicateur_frequence_bbands',
                'output_column_map': { # Clés conceptuelles pour IndicatorCalculator
                    'lower': self.bb_lower_col_strat,
                    'middle': self.bb_middle_col_strat,
                    'upper': self.bb_upper_col_strat,
                    'bandwidth': self.bb_bandwidth_col_strat
                }
            },
            {
                'indicator_name': 'sma', # Ou un autre type de MA si paramétrable
                'params': {'length': int(self.params['volume_ma_period'])},
                # La source est la colonne de volume (1-min ou agrégée) déterminée dans __init__
                'source_column_name': self.volume_kline_col_strat,
                # Pas besoin de 'source_kline_frequency_param_name' ici car 'source_column_name' est déjà spécifique.
                'output_column_name': self.volume_ma_col_strat
            },
            {
                'indicator_name': 'rsi',
                'params': {'length': int(self.params['rsi_period'])},
                'source_kline_frequency_param_name': 'indicateur_frequence_rsi',
                'output_column_name': self.rsi_col_strat
            },
            {
                'indicator_name': 'atr',
                'params': {'length': int(self.params['atr_period_sl_tp'])},
                'source_kline_frequency_param_name': 'atr_base_frequency_sl_tp',
                'output_column_name': self.atr_col_strat
            }
        ]
        logger.debug(f"{self.log_prefix} Configurations d'indicateurs requises : {configs}")
        return configs

    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie la présence des colonnes d'indicateurs _strat attendues et de la colonne
        de volume source.
        """
        df = data_feed.copy()
        expected_strat_cols = [
            self.bb_upper_col_strat, self.bb_middle_col_strat, self.bb_lower_col_strat,
            self.bb_bandwidth_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat,
            self.volume_kline_col_strat # Colonne source pour le volume (avant MA)
        ]
        
        missing_cols_added_nan = False
        for col_name in expected_strat_cols:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur/source attendue '{col_name}' manquante dans data_feed. Ajout avec NaN.")
                df[col_name] = np.nan
                missing_cols_added_nan = True
        
        base_ohlcv = ['open', 'high', 'low', 'close', 'volume'] # 'volume' ici est le volume 1-min de base
        for col in base_ohlcv:
            if col not in df.columns:
                logger.error(f"{self.log_prefix} Colonne OHLCV de base '{col}' manquante dans data_feed.")
                df[col] = np.nan # Pourrait causer des problèmes si utilisé directement par la logique de signal
                missing_cols_added_nan = True
        
        if missing_cols_added_nan:
             logger.debug(f"{self.log_prefix} Après vérification _calculate_indicators, colonnes : {df.columns.tolist()}")

        cols_to_ffill = [col for col in expected_strat_cols if col in df.columns and col.endswith('_strat')]
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
        Génère les signaux de trading pour la stratégie BbandsVolumeRsiStrategy.
        """
        signal_type: int = 0
        limit_price: Optional[float] = None
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        if len(data_with_indicators) < 2: # Besoin d'au moins 2 barres pour comparer actuel et précédent
            logger.debug(f"{self.log_prefix} Pas assez de données ({len(data_with_indicators)}) pour générer des signaux.")
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        latest_row = data_with_indicators.iloc[-1]
        previous_row = data_with_indicators.iloc[-2]

        # Récupérer les valeurs des indicateurs et du prix
        close_curr = latest_row.get('close')
        bb_upper_curr = latest_row.get(self.bb_upper_col_strat)
        bb_lower_curr = latest_row.get(self.bb_lower_col_strat)
        # self.volume_kline_col_strat est le nom de la colonne volume source (ex: 'volume' ou 'Kline_5min_volume')
        volume_kline_curr = latest_row.get(self.volume_kline_col_strat)
        volume_ma_curr = latest_row.get(self.volume_ma_col_strat)
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

        # Paramètres de la stratégie
        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))
        rsi_buy_thresh = float(self.get_param('rsi_buy_breakout_threshold'))
        rsi_sell_thresh = float(self.get_param('rsi_sell_breakout_threshold'))
        allow_shorting = self.get_param('allow_shorting', False) # Supposer un paramètre pour la vente à découvert

        # Conditions d'entrée Long
        long_bb_breakout = close_curr > bb_upper_curr and close_prev <= bb_upper_prev # type: ignore
        long_volume_confirm = volume_kline_curr > volume_ma_curr # type: ignore
        long_rsi_confirm = rsi_curr > rsi_buy_thresh # type: ignore
        entry_long_triggered = long_bb_breakout and long_volume_confirm and long_rsi_confirm

        # Conditions d'entrée Short
        entry_short_triggered = False
        if allow_shorting:
            short_bb_breakout = close_curr < bb_lower_curr and close_prev >= bb_lower_prev # type: ignore
            short_volume_confirm = volume_kline_curr > volume_ma_curr # type: ignore # Volume élevé pour les deux directions
            short_rsi_confirm = rsi_curr < rsi_sell_thresh # type: ignore
            entry_short_triggered = short_bb_breakout and short_volume_confirm and short_rsi_confirm
        
        # Logique de signal
        if not current_position_open:
            if entry_long_triggered:
                signal_type = 1
                if atr_curr > 0: # type: ignore
                    sl_price = close_curr - (sl_atr_mult * atr_curr) # type: ignore
                    tp_price = close_curr + (tp_atr_mult * atr_curr) # type: ignore
                logger.info(f"{self.log_prefix} Signal BUY @ {close_curr:.4f}. SL={sl_price}, TP={tp_price}")
            elif entry_short_triggered: # Seulement si allow_shorting est vrai
                signal_type = -1
                if atr_curr > 0: # type: ignore
                    sl_price = close_curr + (sl_atr_mult * atr_curr) # type: ignore
                    tp_price = close_curr - (tp_atr_mult * atr_curr) # type: ignore
                logger.info(f"{self.log_prefix} Signal SELL @ {close_curr:.4f}. SL={sl_price}, TP={tp_price}")
        else: # Position ouverte, chercher des signaux de sortie (non définis explicitement par cette stratégie, SL/TP gèrent)
            # Si on voulait ajouter des sorties basées sur des conditions inverses :
            # if current_position_direction == 1 and entry_short_triggered: # Sortir Long si signal Short
            #     signal_type = 2
            # elif current_position_direction == -1 and entry_long_triggered: # Sortir Short si signal Long
            #     signal_type = 2
            pass # Les sorties sont gérées par SL/TP dans le simulateur

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

        if signal not in [1, -1]:
            logger.debug(f"{self.log_prefix} [Live] Aucun signal d'entrée (1 ou -1) généré. Signal: {signal}")
            return None

        latest_bar = data_with_indicators.iloc[-1]
        entry_price_theoretical: float
        if order_type == "LIMIT" and limit_price_sugg is not None:
            entry_price_theoretical = limit_price_sugg
        else: # Pour MARKET, utiliser le close comme estimation
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

