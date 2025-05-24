import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
# pandas_ta n'est plus nécessaire ici car les indicateurs sont calculés en amont

from src.strategies.base import BaseStrategy # Importation corrigée
from src.utils.exchange_utils import (adjust_precision,
                                      get_filter_value, # Utilisé pour tick_size
                                      get_precision_from_filter)

logger = logging.getLogger(__name__)

class BbandsVolumeRsiStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur la cassure de range des Bandes de Bollinger,
    confirmée par le volume et le RSI. L'ATR est utilisé pour SL/TP.
    Les indicateurs (BB_UPPER_strat, RSI_strat, Volume_MA_strat, ATR_strat, etc.)
    sont fournis à la stratégie via le DataFrame d'entrée.
    """
    REQUIRED_PARAMS = [
        'bbands_period', 'bbands_std_dev', 'indicateur_frequence_bbands',
        'volume_ma_period', 'indicateur_frequence_volume',
        'rsi_period', 'indicateur_frequence_rsi',
        'rsi_buy_breakout_threshold', 'rsi_sell_breakout_threshold',
        'atr_period_sl_tp', 'atr_base_frequency_sl_tp', 'sl_atr_mult', 'tp_atr_mult'
    ]

    def __init__(self, params: dict):
        super().__init__(params)
        # self._validate_strategy_params() # Appelée par super().__init__

        # Noms des colonnes d'indicateurs attendues
        self.bb_upper_col_strat = "BB_UPPER_strat"
        self.bb_middle_col_strat = "BB_MIDDLE_strat" # Peut être utilisé pour des exits ou filtres
        self.bb_lower_col_strat = "BB_LOWER_strat"
        self.bb_bandwidth_col_strat = "BB_BANDWIDTH_strat" # Peut être utilisé pour des filtres de volatilité

        # Le nom de la colonne source pour le volume (brut, agrégé)
        # Cette colonne est attendue dans `data` et est utilisée pour calculer Volume_MA_strat en amont
        # OU pour être comparée directement à Volume_MA_strat.
        # `ObjectiveEvaluator` devrait fournir `Volume_strat` (basé sur `indicateur_frequence_volume`)
        # et `Volume_MA_strat` (basé sur `Volume_strat` et `volume_ma_period`).
        # Ici, on s'attend à ce que `volume_kline_col_strat` soit le volume de la K-line à la fréquence de `indicateur_frequence_volume`.
        vol_freq_param = self.get_param('indicateur_frequence_volume')
        if vol_freq_param and str(vol_freq_param).lower() != "1min": # Assurer que vol_freq_param est traité comme str
            # Convention: ObjectiveEvaluator fournit `Kline_{freq}_volume`
            self.volume_kline_col_strat = f"Kline_{vol_freq_param}_volume"
        else:
            self.volume_kline_col_strat = "volume" # Volume 1-min de base

        self.volume_ma_col_strat = "Volume_MA_strat" # MA du volume (calculé sur self.volume_kline_col_strat)
        self.rsi_col_strat = "RSI_strat"
        self.atr_col_strat = "ATR_strat" # Pour SL/TP

        self._signals: Optional[pd.DataFrame] = None
        # self.log_prefix = f"[{self.__class__.__name__}]"
        logger.info(f"{self.log_prefix} Stratégie initialisée. Paramètres: {self.params}")
        logger.info(f"{self.log_prefix} Colonne volume source attendue pour comparaison: {self.volume_kline_col_strat}")


    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie la présence des colonnes d'indicateurs *_strat requises.
        """
        df = data.copy()
        # logger.debug(f"{self.log_prefix} Entrée _calculate_indicators. Colonnes reçues: {df.columns.tolist()}")

        expected_strat_cols = [
            self.bb_upper_col_strat, self.bb_middle_col_strat, self.bb_lower_col_strat,
            self.bb_bandwidth_col_strat,
            self.volume_kline_col_strat, # Volume de la K-line source pour la MA de volume
            self.volume_ma_col_strat,    # MA du volume
            self.rsi_col_strat,
            self.atr_col_strat
        ]
        for col_name in expected_strat_cols:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur attendue '{col_name}' manquante. Ajoutée avec NaN.")
                df[col_name] = np.nan
        
        required_ohlc = ['open', 'high', 'low', 'close'] # volume est géré par volume_kline_col_strat
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
        # logger.debug(f"{self.log_prefix} Génération des signaux (pour backtesting)...")
        df_with_indicators = self._calculate_indicators(data)

        required_cols_for_signal = [
            self.bb_upper_col_strat, self.bb_lower_col_strat,
            self.volume_kline_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat, 'close', 'open'
        ]
        if df_with_indicators.empty or \
           any(col not in df_with_indicators.columns for col in required_cols_for_signal) or \
           df_with_indicators[required_cols_for_signal].isnull().all().all() or \
           len(df_with_indicators) < 2:
            logger.warning(f"{self.log_prefix} Données/colonnes insuffisantes pour générer les signaux. Signaux vides générés.")
            self._signals = pd.DataFrame(index=data.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            self._signals[self._signals.select_dtypes(include=['object', 'bool']).columns] = False
            return

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))
        rsi_buy_thresh = float(self.get_param('rsi_buy_breakout_threshold'))
        rsi_sell_thresh = float(self.get_param('rsi_sell_breakout_threshold'))

        close_curr = df_with_indicators['close']
        close_prev = close_curr.shift(1) # Utilisé pour s'assurer que la condition de breakout n'était pas déjà vraie
        
        bb_upper_curr = df_with_indicators[self.bb_upper_col_strat]
        bb_lower_curr = df_with_indicators[self.bb_lower_col_strat]
        bb_upper_prev = bb_upper_curr.shift(1) # Pour la condition de breakout "juste maintenant"
        bb_lower_prev = bb_lower_curr.shift(1) # Pour la condition de breakout "juste maintenant"

        volume_kline_curr = df_with_indicators[self.volume_kline_col_strat] # Volume de la bougie K-line source
        volume_ma_curr = df_with_indicators[self.volume_ma_col_strat]       # MA du volume K-line source
        rsi_curr = df_with_indicators[self.rsi_col_strat]
        atr_curr = df_with_indicators[self.atr_col_strat]

        # Conditions pour un signal LONG
        # 1. Clôture actuelle > Bande Supérieure de Bollinger actuelle
        long_bb_breakout_curr = close_curr > bb_upper_curr
        # 2. Clôture précédente <= Bande Supérieure de Bollinger précédente (pour capturer la cassure)
        long_not_breakout_prev = close_prev <= bb_upper_prev
        # 3. Volume actuel > Moyenne Mobile du Volume
        long_volume_confirm_curr = volume_kline_curr > volume_ma_curr
        # 4. RSI actuel > Seuil d'achat RSI
        long_rsi_confirm_curr = rsi_curr > rsi_buy_thresh
        
        entry_long_trigger = long_bb_breakout_curr & long_not_breakout_prev & \
                             long_volume_confirm_curr & long_rsi_confirm_curr

        # Conditions pour un signal SHORT
        # 1. Clôture actuelle < Bande Inférieure de Bollinger actuelle
        short_bb_breakout_curr = close_curr < bb_lower_curr
        # 2. Clôture précédente >= Bande Inférieure de Bollinger précédente
        short_not_breakout_prev = close_prev >= bb_lower_prev
        # 3. Volume actuel > Moyenne Mobile du Volume
        short_volume_confirm_curr = volume_kline_curr > volume_ma_curr
        # 4. RSI actuel < Seuil de vente RSI
        short_rsi_confirm_curr = rsi_curr < rsi_sell_thresh

        entry_short_trigger = short_bb_breakout_curr & short_not_breakout_prev & \
                              short_volume_confirm_curr & short_rsi_confirm_curr
        
        signals_df = pd.DataFrame(index=df_with_indicators.index)
        signals_df['entry_long'] = entry_long_trigger
        signals_df['entry_short'] = entry_short_trigger
        signals_df['exit_long'] = False # Pas de signal de sortie explicite autre que SL/TP
        signals_df['exit_short'] = False

        signals_df['sl'] = np.nan
        signals_df['tp'] = np.nan
        
        # Utiliser close_curr comme prix de référence pour SL/TP (le simulateur appliquera le slippage sur l'ouverture de la barre suivante)
        entry_price_series_ref = close_curr 

        valid_data_for_sltp = atr_curr.notna() & entry_price_series_ref.notna()

        signals_df.loc[entry_long_trigger & valid_data_for_sltp, 'sl'] = entry_price_series_ref - sl_atr_mult * atr_curr
        signals_df.loc[entry_long_trigger & valid_data_for_sltp, 'tp'] = entry_price_series_ref + tp_atr_mult * atr_curr

        signals_df.loc[entry_short_trigger & valid_data_for_sltp, 'sl'] = entry_price_series_ref + sl_atr_mult * atr_curr
        signals_df.loc[entry_short_trigger & valid_data_for_sltp, 'tp'] = entry_price_series_ref - tp_atr_mult * atr_curr

        self._signals = signals_df[['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']].reindex(data.index)
        # logger.debug(f"{self.log_prefix} Signaux générés (backtesting). Longs: {self._signals['entry_long'].sum()}, Shorts: {self._signals['entry_short'].sum()}")


    def generate_order_request(self,
                               data: pd.DataFrame,
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        # log_prefix_live = f"{self.log_prefix}[LiveOrder][{symbol}]"
        # logger.info(f"{log_prefix_live} Appel generate_order_request. Pos: {current_position}, Capital: {available_capital:.2f}")

        if current_position != 0:
            # logger.debug(f"{log_prefix_live} Position existante. Pas de nouvelle requête d'ordre.")
            return None
        
        if data.empty or len(data) < 2: # Nécessite au moins 2 points pour previous_data
            logger.warning(f"{self.log_prefix} Données d'entrée vides ou insuffisantes (lignes: {len(data)}) pour generate_order_request.")
            return None

        df_verified_indicators = self._calculate_indicators(data.copy())
        
        if len(df_verified_indicators) < 2:
             logger.warning(f"{self.log_prefix} Pas assez de lignes après _calculate_indicators ({len(df_verified_indicators)}) pour évaluer les conditions.")
             return None
             
        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2]

        required_cols_check = [
            'close', 'open', self.bb_upper_col_strat, self.bb_lower_col_strat,
            self.volume_kline_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat
        ]
        if latest_data[required_cols_check].isnull().any():
            nan_cols_latest = latest_data[required_cols_check].index[latest_data[required_cols_check].isnull()].tolist()
            logger.warning(f"{self.log_prefix} Indicateurs essentiels NaN sur dernière donnée: {nan_cols_latest}. Pas de requête.")
            return None
        
        # Pour previous_data, l'ATR n'est pas critique, mais les autres le sont pour la logique de "juste cassé"
        required_cols_prev = [col for col in required_cols_check if col != self.atr_col_strat]
        if previous_data[required_cols_prev].isnull().any():
            nan_cols_previous = previous_data[required_cols_prev].index[previous_data[required_cols_prev].isnull()].tolist()
            logger.warning(f"{self.log_prefix} Indicateurs essentiels NaN sur avant-dernière donnée: {nan_cols_previous}. Pas de requête.")
            return None

        close_curr = latest_data['close']
        bb_upper_curr = latest_data[self.bb_upper_col_strat]
        bb_lower_curr = latest_data[self.bb_lower_col_strat]
        volume_kline_curr = latest_data[self.volume_kline_col_strat]
        volume_ma_curr = latest_data[self.volume_ma_col_strat]
        rsi_curr = latest_data[self.rsi_col_strat]
        atr_value = latest_data[self.atr_col_strat]
        
        close_prev = previous_data['close']
        bb_upper_prev = previous_data[self.bb_upper_col_strat] # Peut être NaN si période BBands > longueur données
        bb_lower_prev = previous_data[self.bb_lower_col_strat] # Idem
        # Volume et RSI prev sont implicitement utilisés dans la condition "et pas la condition précédente"
        volume_kline_prev = previous_data[self.volume_kline_col_strat]
        volume_ma_prev = previous_data[self.volume_ma_col_strat]
        rsi_prev = previous_data[self.rsi_col_strat]

        rsi_buy_thresh = float(self.get_param('rsi_buy_breakout_threshold'))
        rsi_sell_thresh = float(self.get_param('rsi_sell_breakout_threshold'))

        # Conditions actuelles
        current_long_bb_breakout = close_curr > bb_upper_curr
        current_long_volume_conf = volume_kline_curr > volume_ma_curr
        current_long_rsi_conf = rsi_curr > rsi_buy_thresh
        all_current_long_conditions_met = current_long_bb_breakout and current_long_volume_conf and current_long_rsi_conf

        # Conditions précédentes (pour s'assurer que la condition n'était pas déjà vraie)
        previous_long_bb_breakout = close_prev > bb_upper_prev if pd.notna(bb_upper_prev) else False
        previous_long_volume_conf = volume_kline_prev > volume_ma_prev if pd.notna(volume_ma_prev) else False
        previous_long_rsi_conf = rsi_prev > rsi_buy_thresh if pd.notna(rsi_prev) else False
        all_previous_long_conditions_met = previous_long_bb_breakout and previous_long_volume_conf and previous_long_rsi_conf

        # Conditions actuelles Short
        current_short_bb_breakout = close_curr < bb_lower_curr
        current_short_volume_conf = volume_kline_curr > volume_ma_curr
        current_short_rsi_conf = rsi_curr < rsi_sell_thresh
        all_current_short_conditions_met = current_short_bb_breakout and current_short_volume_conf and current_short_rsi_conf

        # Conditions précédentes Short
        previous_short_bb_breakout = close_prev < bb_lower_prev if pd.notna(bb_lower_prev) else False
        previous_short_volume_conf = volume_kline_prev > volume_ma_prev if pd.notna(volume_ma_prev) else False
        previous_short_rsi_conf = rsi_prev < rsi_sell_thresh if pd.notna(rsi_prev) else False
        all_previous_short_conditions_met = previous_short_bb_breakout and previous_short_volume_conf and previous_short_rsi_conf


        side: Optional[str] = None
        sl_price_raw: Optional[float] = None
        tp_price_raw: Optional[float] = None
        
        entry_price_theoretical = latest_data['open'] # Entrée à l'ouverture de la nouvelle bougie
        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        if pd.isna(atr_value) or atr_value <= 1e-9:
            logger.warning(f"{self.log_prefix} Valeur ATR invalide ({atr_value}). Pas de requête d'ordre.")
            return None

        if all_current_long_conditions_met and not all_previous_long_conditions_met:
            side = 'BUY'
            sl_price_raw = entry_price_theoretical - sl_atr_mult * atr_value
            tp_price_raw = entry_price_theoretical + tp_atr_mult * atr_value
            # logger.info(f"{self.log_prefix} Signal d'ACHAT BBAND détecté pour requête d'ordre.")
        elif all_current_short_conditions_met and not all_previous_short_conditions_met:
            side = 'SELL'
            sl_price_raw = entry_price_theoretical + sl_atr_mult * atr_value
            tp_price_raw = entry_price_theoretical - tp_atr_mult * atr_value
            # logger.info(f"{self.log_prefix} Signal de VENTE BBAND détecté pour requête d'ordre.")

        if side and sl_price_raw is not None and tp_price_raw is not None:
            if side == 'BUY' and (sl_price_raw >= entry_price_theoretical or tp_price_raw <= entry_price_theoretical):
                # logger.warning(f"{self.log_prefix} SL/TP invalide pour BUY. SL: {sl_price_raw}, TP: {tp_price_raw}, Entrée: {entry_price_theoretical}")
                return None
            if side == 'SELL' and (sl_price_raw <= entry_price_theoretical or tp_price_raw >= entry_price_theoretical):
                # logger.warning(f"{self.log_prefix} SL/TP invalide pour SELL. SL: {sl_price_raw}, TP: {tp_price_raw}, Entrée: {entry_price_theoretical}")
                return None

            price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            if price_precision is None or qty_precision is None:
                logger.error(f"{self.log_prefix} Impossible d'obtenir la précision de prix/quantité pour {symbol}.")
                return None

            quantity = self._calculate_quantity(
                entry_price=entry_price_theoretical, available_capital=available_capital,
                qty_precision=qty_precision, symbol_info=symbol_info, symbol=symbol
            )
            if quantity is None or quantity <= 1e-9:
                # logger.warning(f"{self.log_prefix} Quantité calculée invalide ({quantity}). Pas d'ordre.")
                return None

            entry_price_for_order_request = adjust_precision(entry_price_theoretical, price_precision, round)
            if entry_price_for_order_request is None:
                logger.error(f"{self.log_prefix} Échec de l'ajustement du prix d'entrée pour la requête.")
                return None
            
            # S'assurer que SL/TP ne sont pas trop proches ou du mauvais côté après ajustement du prix d'entrée.
            # Ceci est une simplification; une logique plus fine pourrait être nécessaire.
            tick_size_str = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize') # Utilisation correcte de get_filter_value
            tick_size = float(tick_size_str) if tick_size_str is not None else 0.00000001 # Fallback si tick_size non trouvé

            if side == 'BUY':
                if sl_price_raw >= entry_price_for_order_request : sl_price_raw = entry_price_for_order_request - tick_size
                if tp_price_raw <= entry_price_for_order_request : tp_price_raw = entry_price_for_order_request + tick_size
            elif side == 'SELL':
                if sl_price_raw <= entry_price_for_order_request : sl_price_raw = entry_price_for_order_request + tick_size
                if tp_price_raw >= entry_price_for_order_request : tp_price_raw = entry_price_for_order_request - tick_size

            entry_price_str = f"{entry_price_for_order_request:.{price_precision}f}"
            quantity_str = f"{quantity:.{qty_precision}f}"
            
            order_type_pref = self.get_param('order_type_preference', "LIMIT")
            entry_order_params = self._build_entry_params_formatted(
                symbol=symbol, side=side, quantity_str=quantity_str,
                entry_price_str=entry_price_str if order_type_pref == "LIMIT" else None,
                order_type=order_type_pref
            )
            if not entry_order_params:
                # logger.error(f"{self.log_prefix} Échec de la construction des paramètres d'ordre formatés.")
                return None

            sl_tp_raw_prices = {'sl_price': sl_price_raw, 'tp_price': tp_price_raw}
            # logger.info(f"{self.log_prefix} Requête d'ordre générée: {entry_order_params}, SL/TP bruts: {sl_tp_raw_prices}")
            return entry_order_params, sl_tp_raw_prices
        
        # logger.debug(f"{self.log_prefix} Aucune condition d'entrée BBAND remplie pour une nouvelle requête d'ordre.")
        return None