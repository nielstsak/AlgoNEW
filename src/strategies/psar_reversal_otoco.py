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
import pandas as pd

from src.strategies.base import BaseStrategy
# Les utilitaires d'exchange_utils sont accessibles via les méthodes de BaseStrategy
# ou directement si nécessaire pour des logiques très spécifiques non couvertes.

logger = logging.getLogger(__name__)

class PsarReversalOtocoStrategy(BaseStrategy):
    """
    Stratégie de renversement du Parabolic SAR.
    Génère des signaux d'achat lorsque le PSAR passe de dessus à dessous le prix,
    et des signaux de vente lorsque le PSAR passe de dessous à dessus le prix.
    Le SL et le TP sont basés sur l'ATR.
    """

    REQUIRED_PARAMS: List[str] = [
        'psar_step',  # Aussi connu comme 'af' ou 'acceleration factor'
        'psar_max_step', # Aussi connu comme 'max_af'
        'indicateur_frequence_psar',
        'atr_period_sl_tp',
        'atr_base_frequency_sl_tp',
        'sl_atr_mult',
        'tp_atr_mult',
        'capital_allocation_pct',
        'order_type_preference'
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        """
        Initialise la stratégie PsarReversalOtocoStrategy.

        Args:
            strategy_name (str): Nom de la stratégie (clé de configuration).
            symbol (str): Symbole de la paire de trading (ex: BTCUSDT).
            params (Dict[str, Any]): Paramètres spécifiques à cette instance de stratégie.
        """
        super().__init__(strategy_name, symbol, params)
        # self.log_prefix est déjà défini dans BaseStrategy

        # Noms des colonnes pour les indicateurs calculés
        self.psarl_col_strat: str = "PSARl_strat" # PSAR long (support)
        self.psars_col_strat: str = "PSARs_strat" # PSAR short (resistance)
        # Note: pandas-ta.psar retourne un DataFrame avec des colonnes comme PSARl_<step>_<max_step>
        # IndicatorCalculator devra mapper cela correctement.
        self.atr_col_strat: str = "ATR_strat"
        
        # logger.info(f"{self.log_prefix} Stratégie PsarReversalOtocoStrategy initialisée.")

    def _validate_params(self) -> None:
        """
        Valide les paramètres spécifiques à PsarReversalOtocoStrategy.
        Lève une ValueError si un paramètre est invalide.
        """
        missing_params = [p for p in self.REQUIRED_PARAMS if self.get_param(p) is None]
        if missing_params:
            raise ValueError(f"{self.log_prefix} Paramètres requis manquants : {', '.join(missing_params)}")

        psar_step = self.get_param('psar_step')
        psar_max_step = self.get_param('psar_max_step')
        if not (isinstance(psar_step, (int, float)) and psar_step > 0):
            raise ValueError(f"{self.log_prefix} 'psar_step' ({psar_step}) doit être un nombre positif.")
        if not (isinstance(psar_max_step, (int, float)) and psar_max_step > 0):
            raise ValueError(f"{self.log_prefix} 'psar_max_step' ({psar_max_step}) doit être un nombre positif.")
        if psar_step >= psar_max_step:
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

        order_type_pref_val = self.get_param('order_type_preference')
        if order_type_pref_val not in ["MARKET", "LIMIT"]:
            raise ValueError(f"{self.log_prefix} 'order_type_preference' ({order_type_pref_val}) doit être 'MARKET' ou 'LIMIT'.")

        for freq_param_name in ['indicateur_frequence_psar', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' ({freq_val}) doit être une chaîne de caractères non vide.")
        
        logger.debug(f"{self.log_prefix} Validation des paramètres terminée avec succès.")

    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        """
        Déclare les indicateurs requis par PsarReversalOtocoStrategy.
        """
        # pandas-ta psar utilise 'af' pour step et 'max_af' pour max_step.
        # Il retourne un DataFrame. IndicatorCalculator devra mapper les colonnes.
        # Les clés dans 'output_column_map' sont des identifiants conceptuels
        # que IndicatorCalculator utilisera pour trouver les bonnes colonnes dans le
        # DataFrame retourné par ta.psar (ex: la colonne contenant 'psarl' pour 'long_stop').
        configs = [
            {
                'indicator_name': 'psar',
                'params': {
                    'af': self.params['psar_step'],
                    'max_af': self.params['psar_max_step']
                },
                'source_kline_frequency_param_name': 'indicateur_frequence_psar',
                # Pour PSAR, 'inputs' attendues par pandas-ta sont 'high', 'low', 'close'.
                # IndicatorCalculator utilisera la fréquence pour trouver les bonnes colonnes sources.
                'output_column_map': {
                    'PSARl': self.psarl_col_strat, # Clé conceptuelle, mappée à la colonne réelle par IndicatorCalculator
                    'PSARs': self.psars_col_strat
                    # Si IndicatorCalculator a besoin des noms exacts que pandas-ta retourne (ex: PSARl_0.02_0.2),
                    # alors ces noms devraient être construits ici ou la logique de mapping dans
                    # IndicatorCalculator doit être robuste.
                    # Pour l'instant, on suppose que IndicatorCalculator peut gérer ce mapping.
                }
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
        Vérifie la présence des colonnes d'indicateurs _strat attendues.
        """
        df = data_feed.copy()
        expected_strat_cols = [self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat]
        
        missing_cols_added_nan = False
        for col_name in expected_strat_cols:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur attendue '{col_name}' manquante dans data_feed. Ajout avec NaN.")
                df[col_name] = np.nan
                missing_cols_added_nan = True
        
        base_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        for col in base_ohlcv:
            if col not in df.columns:
                logger.error(f"{self.log_prefix} Colonne OHLCV de base '{col}' manquante dans data_feed.")
                df[col] = np.nan
                missing_cols_added_nan = True
        
        if missing_cols_added_nan:
             logger.debug(f"{self.log_prefix} Après vérification _calculate_indicators, colonnes : {df.columns.tolist()}")

        cols_to_ffill = [col for col in expected_strat_cols if col in df.columns]
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
        Génère les signaux de trading pour la stratégie de renversement PSAR.
        """
        signal_type: int = 0
        limit_price: Optional[float] = None
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        if len(data_with_indicators) < 2: # Besoin d'au moins 2 barres pour la logique de renversement
            logger.debug(f"{self.log_prefix} Pas assez de données ({len(data_with_indicators)}) pour générer des signaux.")
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        latest_row = data_with_indicators.iloc[-1]
        previous_row = data_with_indicators.iloc[-2]

        # Récupérer les valeurs des indicateurs et du prix
        close_curr = latest_row.get('close')
        psarl_curr = latest_row.get(self.psarl_col_strat) # PSAR Long (support)
        psars_curr = latest_row.get(self.psars_col_strat) # PSAR Short (résistance)
        atr_curr = latest_row.get(self.atr_col_strat)
        
        psarl_prev = previous_row.get(self.psarl_col_strat)
        psars_prev = previous_row.get(self.psars_col_strat)

        # Vérifier les valeurs essentielles
        # Pour PSAR, une des deux valeurs (psarl, psars) est NaN et l'autre a une valeur.
        # Le signal de renversement se produit lorsque la valeur non-NaN change de colonne.
        if pd.isna(close_curr) or pd.isna(atr_curr) or \
           (pd.isna(psarl_curr) and pd.isna(psars_curr)) or \
           (pd.isna(psarl_prev) and pd.isna(psars_prev)):
            nan_details = {"close": close_curr, "atr": atr_curr, "psarl_c": psarl_curr, "psars_c": psars_curr, "psarl_p": psarl_prev, "psars_p": psars_prev}
            logger.debug(f"{self.log_prefix} Valeurs d'indicateur ou de prix manquantes (NaN) à {latest_row.name}. Détails: {nan_details}. Signal HOLD.")
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        # Logique de renversement PSAR
        # Entrée Long: PSAR était au-dessus (PSARs actif) et croise en dessous (PSARl devient actif)
        entry_long_triggered = pd.notna(psars_prev) and pd.isna(psarl_prev) and \
                               pd.notna(psarl_curr) and pd.isna(psars_curr)
        # Entrée Short: PSAR était en dessous (PSARl actif) et croise au-dessus (PSARs devient actif)
        entry_short_triggered = pd.notna(psarl_prev) and pd.isna(psars_prev) and \
                                pd.notna(psars_curr) and pd.isna(psarl_curr)

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        if not current_position_open:
            if entry_long_triggered:
                signal_type = 1 # Signal d'achat (LONG)
                if atr_curr > 0: # type: ignore
                    sl_price = close_curr - (sl_atr_mult * atr_curr) # type: ignore
                    tp_price = close_curr + (tp_atr_mult * atr_curr) # type: ignore
                logger.info(f"{self.log_prefix} Signal BUY (PSAR Long) @ {close_curr:.4f}. SL={sl_price}, TP={tp_price}")
            elif entry_short_triggered:
                signal_type = -1 # Signal de vente (SHORT)
                if atr_curr > 0: # type: ignore
                    sl_price = close_curr + (sl_atr_mult * atr_curr) # type: ignore
                    tp_price = close_curr - (tp_atr_mult * atr_curr) # type: ignore
                logger.info(f"{self.log_prefix} Signal SELL (PSAR Short) @ {close_curr:.4f}. SL={sl_price}, TP={tp_price}")
        else: # Position ouverte
            if current_position_direction == 1 and entry_short_triggered: # En Long, et signal de renversement Short
                signal_type = 2 # Signal de sortie de position
                logger.info(f"{self.log_prefix} Signal EXIT LONG (PSAR Short triggered) @ {close_curr:.4f}")
            elif current_position_direction == -1 and entry_long_triggered: # En Short, et signal de renversement Long
                signal_type = 2 # Signal de sortie de position
                logger.info(f"{self.log_prefix} Signal EXIT SHORT (PSAR Long triggered) @ {close_curr:.4f}")
        
        order_type_preference = str(self.get_param("order_type_preference", "MARKET"))
        if signal_type != 0 and order_type_preference == "LIMIT":
            limit_price = float(close_curr) # Suggérer le prix de clôture actuel pour l'ordre limite

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
            logger.warning(f"{self.log_prefix} [Live] Données insuffisantes après _calculate_indicators pour generate_order_request. Shape: {data_with_indicators.shape}")
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
            from src.utils.exchange_utils import adjust_precision, get_filter_value # Import local pour éviter dépendance circulaire au niveau module
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

