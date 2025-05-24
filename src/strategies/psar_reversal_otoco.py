import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
# pandas_ta n'est plus nécessaire ici car les indicateurs sont calculés en amont

from src.strategies.base import BaseStrategy # Importation corrigée
from src.utils.exchange_utils import (adjust_precision,
                                      get_precision_from_filter) # get_filter_value n'est pas directement utilisé

logger = logging.getLogger(__name__)

class PsarReversalOtocoStrategy(BaseStrategy):
    """
    Stratégie de renversement basée sur le Parabolic SAR (PSAR).
    Les indicateurs (PSAR long, PSAR court, ATR) sont fournis à la stratégie
    via des colonnes suffixées par '_strat' dans le DataFrame d'entrée.
    """
    REQUIRED_PARAMS = [
        'psar_step', 'psar_max_step',
        'atr_period_sl_tp', 'sl_atr_mult', 'tp_atr_mult',
        'indicateur_frequence_psar', 'atr_base_frequency_sl_tp'
    ]

    def __init__(self, params: dict):
        super().__init__(params)
        # self._validate_strategy_params() # Appelée par super().__init__

        # Noms des colonnes d'indicateurs attendues
        self.psarl_col_strat = "PSARl_strat" # PSAR Long (valeur du PSAR quand en tendance haussière)
        self.psars_col_strat = "PSARs_strat" # PSAR Short (valeur du PSAR quand en tendance baissière)
        self.atr_col_strat = "ATR_strat"     # Pour SL/TP

        self._signals: Optional[pd.DataFrame] = None
        # self.log_prefix = f"[{self.__class__.__name__}]"
        logger.info(f"{self.log_prefix} Stratégie initialisée. Paramètres: {self.params}")

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie la présence des colonnes d'indicateurs PSAR*_strat et ATR_strat.
        """
        df = data.copy()
        # logger.debug(f"{self.log_prefix} Entrée _calculate_indicators. Colonnes reçues: {df.columns.tolist()}")

        expected_strat_cols = [self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat]
        for col_name in expected_strat_cols:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur attendue '{col_name}' manquante. Ajoutée avec NaN.")
                df[col_name] = np.nan
        
        required_ohlc = ['open', 'high', 'low', 'close']
        for col in required_ohlc:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne OHLC de base '{col}' manquante. Ajoutée avec NaN.")
                df[col] = np.nan
        
        # logger.debug(f"{self.log_prefix} Sortie _calculate_indicators. Colonnes présentes: {df.columns.tolist()}")
        return df

    def generate_signals(self, data: pd.DataFrame) -> None:
        """
        Génère les signaux de trading basés sur les renversements de PSAR.
        """
        # logger.debug(f"{self.log_prefix} Génération des signaux...")
        df_with_indicators = self._calculate_indicators(data)

        required_cols_for_signal = [self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat, 'close']
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

        close_prev = df_with_indicators['close'].shift(1)
        close_curr = df_with_indicators['close']
        
        # PSARl_strat est non-NaN quand le PSAR est en dessous du prix (tendance haussière)
        # PSARs_strat est non-NaN quand le PSAR est au-dessus du prix (tendance baissière)
        psarl_curr = df_with_indicators[self.psarl_col_strat]
        psars_curr = df_with_indicators[self.psars_col_strat]
        atr_curr = df_with_indicators[self.atr_col_strat]

        # Un signal d'achat se produit quand le PSAR passe de "au-dessus" (PSARs était actif) à "en-dessous" (PSARl devient actif)
        # Condition d'achat : Le PSAR actuel (psarl_curr) est valide (non-NaN) ET le PSAR précédent était un PSAR court (psars_prev non-NaN)
        # ou, plus simplement, si psarl_curr est non-NaN et que psars_curr.shift(1) était non-NaN
        # La logique originale est : (close_prev <= psars_prev_filled) & (close_curr > psarl_curr)
        # Cela signifie que la clôture précédente était en dessous du PSAR baissier, et la clôture actuelle est au-dessus du PSAR haussier.
        # C'est un signal de retournement à la hausse.

        # psarl_curr.notna() signifie que la tendance actuelle est haussière (PSAR est un support)
        # psars_curr.shift(1).notna() signifie que la tendance précédente était baissière
        entry_long_trigger = psarl_curr.notna() & psars_curr.shift(1).notna()

        # entry_short_trigger : La tendance actuelle est baissière (PSAR est une résistance) ET la tendance précédente était haussière
        entry_short_trigger = psars_curr.notna() & psarl_curr.shift(1).notna()
        
        # Appliquer la condition de "cassure" de la clôture par rapport au PSAR pertinent pour confirmer le retournement
        # (Alternative: utiliser les conditions de la stratégie originale si elles sont préférées)
        # Signal Long: clôture actuelle > PSAR Long actuel ET clôture précédente < PSAR Short précédent
        # Cependant, pandas_ta pour PSAR donne soit PSARl, soit PSARs (l'un est NaN, l'autre une valeur).
        # Un retournement haussier se produit lorsque PSARl devient non-NaN.
        # Un retournement baissier se produit lorsque PSARs devient non-NaN.

        signals_df = pd.DataFrame(index=df_with_indicators.index)
        signals_df['entry_long'] = entry_long_trigger & valid_data_for_signal
        signals_df['entry_short'] = entry_short_trigger & valid_data_for_signal
        
        signals_df['exit_long'] = entry_short_trigger & valid_data_for_signal # Sortie de long sur signal de short
        signals_df['exit_short'] = entry_long_trigger & valid_data_for_signal # Sortie de short sur signal de long

        signals_df['sl'] = np.nan
        signals_df['tp'] = np.nan
        
        valid_data_for_signal = atr_curr.notna() & close_curr.notna()

        signals_df.loc[signals_df['entry_long'], 'sl'] = close_curr - sl_atr_mult * atr_curr
        signals_df.loc[signals_df['entry_long'], 'tp'] = close_curr + tp_atr_mult * atr_curr

        signals_df.loc[signals_df['entry_short'], 'sl'] = close_curr + sl_atr_mult * atr_curr
        signals_df.loc[signals_df['entry_short'], 'tp'] = close_curr - tp_atr_mult * atr_curr

        self._signals = signals_df[['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']].reindex(data.index)
        # logger.debug(f"{self.log_prefix} Signaux générés. Longs: {self._signals['entry_long'].sum()}, Shorts: {self._signals['entry_short'].sum()}")

    def generate_order_request(self,
                               data: pd.DataFrame,
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        # logger.debug(f"{self.log_prefix} Appel de generate_order_request. Position: {current_position}, Capital: {available_capital:.2f}")
        if current_position != 0:
            # logger.debug(f"{self.log_prefix} Position existante. Pas de nouvelle requête d'ordre.")
            return None

        if data.empty or len(data) < 2:
            logger.warning(f"{self.log_prefix} Données d'entrée vides ou insuffisantes pour generate_order_request.")
            return None

        df_verified_indicators = self._calculate_indicators(data.copy())

        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2]

        required_cols = ['close', 'open', self.psarl_col_strat, self.psars_col_strat, self.atr_col_strat]
        if latest_data[required_cols].isnull().any() or \
           previous_data[[self.psarl_col_strat, self.psars_col_strat]].isnull().any(): # PSAR peut être NaN sur une des deux lignes au moment du flip
            # logger.debug(f"{self.log_prefix} Données indicateur manquantes (NaN) sur dernière/avant-dernière ligne. Pas de requête.")
            return None

        psarl_curr = latest_data[self.psarl_col_strat]
        psars_curr = latest_data[self.psars_col_strat]
        psarl_prev = previous_data[self.psarl_col_strat]
        psars_prev = previous_data[self.psars_col_strat]
        
        entry_price_theoretical = latest_data['open'] # Entrée à l'ouverture de la nouvelle bougie
        atr_value = latest_data[self.atr_col_strat]

        if pd.isna(atr_value) or atr_value <= 1e-9:
            # logger.debug(f"{self.log_prefix} Valeur ATR invalide ({atr_value}). Pas de requête d'ordre.")
            return None

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        side: Optional[str] = None
        sl_price_raw: Optional[float] = None
        tp_price_raw: Optional[float] = None

        # Signal d'achat: PSAR était au-dessus (psars_prev notna) et passe en-dessous (psarl_curr notna)
        if pd.notna(psars_prev) and pd.notna(psarl_curr):
            side = 'BUY'
            sl_price_raw = entry_price_theoretical - sl_atr_mult * atr_value
            tp_price_raw = entry_price_theoretical + tp_atr_mult * atr_value
            # logger.info(f"{self.log_prefix} Signal d'ACHAT (PSAR flip) détecté pour requête d'ordre.")
        # Signal de vente: PSAR était en-dessous (psarl_prev notna) et passe au-dessus (psars_curr notna)
        elif pd.notna(psarl_prev) and pd.notna(psars_curr):
            side = 'SELL'
            sl_price_raw = entry_price_theoretical + sl_atr_mult * atr_value
            tp_price_raw = entry_price_theoretical - tp_atr_mult * atr_value
            # logger.info(f"{self.log_prefix} Signal de VENTE (PSAR flip) détecté pour requête d'ordre.")

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
                entry_price=entry_price_theoretical,
                available_capital=available_capital,
                qty_precision=qty_precision,
                symbol_info=symbol_info,
                symbol=symbol
            )

            if quantity is None or quantity <= 1e-9:
                # logger.debug(f"{self.log_prefix} Quantité calculée invalide ({quantity}). Pas de requête d'ordre.")
                return None

            entry_price_for_order_request = adjust_precision(entry_price_theoretical, price_precision, round)
            if entry_price_for_order_request is None:
                logger.error(f"{self.log_prefix} Échec de l'ajustement du prix d'entrée pour la requête.")
                return None
                
            entry_price_str = f"{entry_price_for_order_request:.{price_precision}f}"
            quantity_str = f"{quantity:.{qty_precision}f}"
            
            order_type_pref = self.get_param('order_type_preference', "LIMIT")
            entry_order_params = self._build_entry_params_formatted(
                symbol=symbol,
                side=side,
                quantity_str=quantity_str,
                entry_price_str=entry_price_str if order_type_pref == "LIMIT" else None,
                order_type=order_type_pref
            )

            if not entry_order_params:
                # logger.error(f"{self.log_prefix} Échec de la construction des paramètres d'ordre formatés.")
                return None

            sl_tp_raw_prices = {'sl_price': sl_price_raw, 'tp_price': tp_price_raw}
            # logger.info(f"{self.log_prefix} Requête d'ordre générée: {entry_order_params}, SL/TP bruts: {sl_tp_raw_prices}")
            return entry_order_params, sl_tp_raw_prices

        # logger.debug(f"{self.log_prefix} Aucune condition d'entrée PSAR remplie pour une nouvelle requête d'ordre.")
        return None