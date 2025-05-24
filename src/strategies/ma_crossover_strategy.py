import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
# pandas_ta n'est plus nécessaire ici car les indicateurs sont calculés en amont

from src.strategies.base import BaseStrategy # Importation corrigée
from src.utils.exchange_utils import (adjust_precision,
                                      get_precision_from_filter) # get_filter_value n'est pas directement utilisé

logger = logging.getLogger(__name__)

class MaCrossoverStrategy(BaseStrategy):
    """
    Stratégie basée sur le croisement de deux moyennes mobiles.
    Les indicateurs (MME rapide, MME lente, ATR) sont fournis à la stratégie
    via des colonnes suffixées par '_strat' dans le DataFrame d'entrée.
    """
    REQUIRED_PARAMS = [
        'fast_ma_period', 'slow_ma_period', 'ma_type',
        'atr_period_sl_tp', 'sl_atr_multiplier', 'tp_atr_multiplier',
        'indicateur_frequence_ma_rapide', 'indicateur_frequence_ma_lente',
        'atr_base_frequency_sl_tp'
    ]

    def __init__(self, params: dict):
        super().__init__(params)
        # self._validate_strategy_params() # Appelée par super().__init__
        self.log_prefix = f"[{self.__class__.__name__}]"
        # Noms des colonnes d'indicateurs attendues (calculées en amont)
        self.fast_ma_col_strat = "MA_FAST_strat"
        self.slow_ma_col_strat = "MA_SLOW_strat"
        self.atr_col_strat = "ATR_strat" # Pour SL/TP

        self._signals: Optional[pd.DataFrame] = None
        # Le log_prefix est initialisé dans BaseStrategy ou peut être surchargé ici si besoin
        # self.log_prefix = f"[{self.__class__.__name__}]" # Exemple si BaseStrategy ne le fait pas
        logger.info(f"{self.log_prefix} Stratégie initialisée. Paramètres: {self.params}")

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie la présence des colonnes d'indicateurs *_strat requises.
        Ces indicateurs sont calculés en amont par ObjectiveEvaluator.
        """
        df = data.copy()
        # logger.debug(f"{self.log_prefix} Entrée _calculate_indicators. Colonnes reçues: {df.columns.tolist()}")

        # Colonnes d'indicateurs spécifiques à la stratégie attendues
        expected_strat_cols = [self.fast_ma_col_strat, self.slow_ma_col_strat, self.atr_col_strat]
        for col_name in expected_strat_cols:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur attendue '{col_name}' manquante. Ajoutée avec NaN.")
                df[col_name] = np.nan

        # Vérification des colonnes OHLCV de base (au cas où, bien que généralement présentes)
        required_ohlc = ['open', 'high', 'low', 'close'] # 'volume' n'est pas directement utilisé ici
        for col in required_ohlc:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne OHLC de base '{col}' manquante. Ajoutée avec NaN.")
                df[col] = np.nan
        
        # logger.debug(f"{self.log_prefix} Sortie _calculate_indicators. Colonnes présentes: {df.columns.tolist()}")
        return df

    def generate_signals(self, data: pd.DataFrame) -> None:
        """
        Génère les signaux de trading basés sur les indicateurs *_strat.
        """
        # logger.debug(f"{self.log_prefix} Génération des signaux...")
        df_with_indicators = self._calculate_indicators(data)

        required_cols_for_signal = [self.fast_ma_col_strat, self.slow_ma_col_strat, self.atr_col_strat, 'close']
        if df_with_indicators.empty or \
           any(col not in df_with_indicators.columns for col in required_cols_for_signal) or \
           df_with_indicators[required_cols_for_signal].isnull().all().all() or \
           len(df_with_indicators) < 2: # Besoin d'au moins 2 lignes pour .shift(1)
            logger.warning(f"{self.log_prefix} Données/colonnes insuffisantes pour générer les signaux. Signaux vides générés.")
            self._signals = pd.DataFrame(index=data.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan # Assurer que sl et tp sont float
            self._signals[self._signals.select_dtypes(include=['object', 'bool']).columns] = False # Assurer que les booléens sont False
            return

        sl_mult = float(self.get_param('sl_atr_multiplier'))
        tp_mult = float(self.get_param('tp_atr_multiplier'))

        ma_fast_curr = df_with_indicators[self.fast_ma_col_strat]
        ma_slow_curr = df_with_indicators[self.slow_ma_col_strat]
        ma_fast_prev = df_with_indicators[self.fast_ma_col_strat].shift(1)
        ma_slow_prev = df_with_indicators[self.slow_ma_col_strat].shift(1)
        close_curr = df_with_indicators['close'] # Prix de référence pour SL/TP
        atr_curr = df_with_indicators[self.atr_col_strat]

        # Conditions de croisement
        bullish_crossover = (ma_fast_curr > ma_slow_curr) & (ma_fast_prev <= ma_slow_prev)
        bearish_crossover = (ma_fast_curr < ma_slow_curr) & (ma_fast_prev >= ma_slow_prev)

        # S'assurer qu'il y a des valeurs valides pour les MME et l'ATR pour générer un signal
        valid_data_for_signal = ma_fast_curr.notna() & ma_slow_curr.notna() & \
                                ma_fast_prev.notna() & ma_slow_prev.notna() & \
                                atr_curr.notna() & close_curr.notna()

        signals_df = pd.DataFrame(index=df_with_indicators.index)
        signals_df['entry_long'] = bullish_crossover & valid_data_for_signal
        signals_df['entry_short'] = bearish_crossover & valid_data_for_signal
        
        # Les signaux de sortie simples (croisement inverse) peuvent être définis ici si la stratégie les utilise
        # Pour l'instant, on se base sur SL/TP géré par le simulateur.
        signals_df['exit_long'] = bearish_crossover & valid_data_for_signal # Exemple de sortie sur croisement inverse
        signals_df['exit_short'] = bullish_crossover & valid_data_for_signal # Exemple de sortie sur croisement inverse

        # Calculer SL/TP uniquement aux points d'entrée
        signals_df['sl'] = np.nan
        signals_df['tp'] = np.nan

        signals_df.loc[signals_df['entry_long'], 'sl'] = close_curr - sl_mult * atr_curr
        signals_df.loc[signals_df['entry_long'], 'tp'] = close_curr + tp_mult * atr_curr

        signals_df.loc[signals_df['entry_short'], 'sl'] = close_curr + sl_mult * atr_curr
        signals_df.loc[signals_df['entry_short'], 'tp'] = close_curr - tp_mult * atr_curr
        
        self._signals = signals_df[['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']].reindex(data.index)
        # logger.debug(f"{self.log_prefix} Signaux générés. Longs: {self._signals['entry_long'].sum()}, Shorts: {self._signals['entry_short'].sum()}")

    def generate_order_request(self,
                               data: pd.DataFrame,
                               symbol: str,
                               current_position: int, # 0:flat, >0:long, <0:short
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre d'entrée si les conditions sont remplies.
        Utilise les indicateurs *_strat des dernières données.
        """
        # logger.debug(f"{self.log_prefix} Appel de generate_order_request. Position: {current_position}, Capital: {available_capital:.2f}")
        if current_position != 0: # Ne génère que des ordres d'entrée
            # logger.debug(f"{self.log_prefix} Position existante. Pas de nouvelle requête d'ordre.")
            return None

        if data.empty or len(data) < 2: # Besoin d'au moins 2 lignes pour .shift(1)
            logger.warning(f"{self.log_prefix} Données d'entrée vides ou insuffisantes pour generate_order_request.")
            return None

        # Utiliser une copie pour éviter de modifier le DataFrame original passé en argument
        df_verified_indicators = self._calculate_indicators(data.copy())

        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2] # Nécessaire pour détecter le croisement

        # Vérifier la validité des données nécessaires
        required_cols = ['close', 'open', self.fast_ma_col_strat, self.slow_ma_col_strat, self.atr_col_strat]
        if latest_data[required_cols].isnull().any() or \
           previous_data[[self.fast_ma_col_strat, self.slow_ma_col_strat]].isnull().any():
            # logger.debug(f"{self.log_prefix} Données indicateur manquantes (NaN) sur dernière/avant-dernière ligne. Pas de requête.")
            return None

        ma_fast_curr = latest_data[self.fast_ma_col_strat]
        ma_slow_curr = latest_data[self.slow_ma_col_strat]
        ma_fast_prev = previous_data[self.fast_ma_col_strat]
        ma_slow_prev = previous_data[self.slow_ma_col_strat]
        
        # Utiliser le prix d'ouverture de la barre actuelle pour l'entrée (ou 'close' si simulation MOC)
        # BaseStrategy s'attend à un prix théorique. Le simulateur appliquera le slippage.
        # Pour une exécution live, cela pourrait être le prix actuel du marché ou un prix limite.
        entry_price_theoretical = latest_data['open'] # Entrée à l'ouverture de la nouvelle bougie
        atr_value = latest_data[self.atr_col_strat]

        if pd.isna(atr_value) or atr_value <= 1e-9: # ATR doit être valide et positif
            # logger.debug(f"{self.log_prefix} Valeur ATR invalide ({atr_value}). Pas de requête d'ordre.")
            return None

        sl_atr_mult = float(self.get_param('sl_atr_multiplier'))
        tp_atr_mult = float(self.get_param('tp_atr_multiplier'))

        side: Optional[str] = None
        sl_price_raw: Optional[float] = None
        tp_price_raw: Optional[float] = None

        # Logique de croisement pour l'entrée
        if ma_fast_prev <= ma_slow_prev and ma_fast_curr > ma_slow_curr:
            side = 'BUY'
            sl_price_raw = entry_price_theoretical - sl_atr_mult * atr_value
            tp_price_raw = entry_price_theoretical + tp_atr_mult * atr_value
            # logger.info(f"{self.log_prefix} Signal d'ACHAT détecté pour requête d'ordre.")
        elif ma_fast_prev >= ma_slow_prev and ma_fast_curr < ma_slow_curr:
            side = 'SELL'
            sl_price_raw = entry_price_theoretical + sl_atr_mult * atr_value
            tp_price_raw = entry_price_theoretical - tp_atr_mult * atr_value
            # logger.info(f"{self.log_prefix} Signal de VENTE détecté pour requête d'ordre.")

        if side and sl_price_raw is not None and tp_price_raw is not None:
            # Valider SL/TP par rapport au prix d'entrée
            if side == 'BUY' and (sl_price_raw >= entry_price_theoretical or tp_price_raw <= entry_price_theoretical):
                # logger.warning(f"{self.log_prefix} SL/TP invalide pour BUY. SL: {sl_price_raw}, TP: {tp_price_raw}, Entrée: {entry_price_theoretical}")
                return None
            if side == 'SELL' and (sl_price_raw <= entry_price_theoretical or tp_price_raw >= entry_price_theoretical):
                # logger.warning(f"{self.log_prefix} SL/TP invalide pour SELL. SL: {sl_price_raw}, TP: {tp_price_raw}, Entrée: {entry_price_theoretical}")
                return None

            # Obtenir les précisions
            price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')

            if price_precision is None or qty_precision is None:
                logger.error(f"{self.log_prefix} Impossible d'obtenir la précision de prix/quantité pour {symbol}.")
                return None

            # Calcul de la quantité via la méthode de BaseStrategy
            quantity = self._calculate_quantity(
                entry_price=entry_price_theoretical,
                available_capital=available_capital,
                qty_precision=qty_precision, # Peut être None, géré par _calculate_quantity
                symbol_info=symbol_info,
                symbol=symbol
            )

            if quantity is None or quantity <= 1e-9: # Vérifier quantité non nulle
                # logger.debug(f"{self.log_prefix} Quantité calculée invalide ({quantity}). Pas de requête d'ordre.")
                return None

            # Ajuster le prix d'entrée pour la requête d'ordre (ex: pour un ordre LIMIT)
            # Si MARKET, ce prix n'est pas envoyé mais peut être utilisé pour des logs.
            entry_price_for_order_request = adjust_precision(entry_price_theoretical, price_precision, round)
            if entry_price_for_order_request is None:
                logger.error(f"{self.log_prefix} Échec de l'ajustement du prix d'entrée pour la requête.")
                return None

            entry_price_str = f"{entry_price_for_order_request:.{price_precision}f}"
            quantity_str = f"{quantity:.{qty_precision}f}"

            # Construire les paramètres d'ordre via la méthode de BaseStrategy
            order_type_pref = self.get_param('order_type_preference', "LIMIT") # Default to LIMIT if not in params
            entry_order_params = self._build_entry_params_formatted(
                symbol=symbol,
                side=side,
                quantity_str=quantity_str,
                entry_price_str=entry_price_str if order_type_pref == "LIMIT" else None, # Seulement pour LIMIT
                order_type=order_type_pref
            )

            if not entry_order_params:
                # logger.error(f"{self.log_prefix} Échec de la construction des paramètres d'ordre formatés.")
                return None

            sl_tp_raw_prices = {'sl_price': sl_price_raw, 'tp_price': tp_price_raw}
            # logger.info(f"{self.log_prefix} Requête d'ordre générée: {entry_order_params}, SL/TP bruts: {sl_tp_raw_prices}")
            return entry_order_params, sl_tp_raw_prices

        # logger.debug(f"{self.log_prefix} Aucune condition d'entrée remplie pour une nouvelle requête d'ordre.")
        return None