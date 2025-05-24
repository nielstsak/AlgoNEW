import logging
import math # Ajouté pour math.isclose si besoin pour les flottants
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

# pandas_ta n'est plus importé ici, car les indicateurs MME/ATR sont calculés en amont.
# Le calcul de pente est conservé comme une logique interne à la stratégie si besoin.

from src.strategies.base import BaseStrategy # Importation corrigée
from src.utils.exchange_utils import (adjust_precision,
                                      get_precision_from_filter)

logger = logging.getLogger(__name__)

class TripleMAAnticipationStrategy(BaseStrategy):
    """
    Stratégie de suivi de tendance utilisant trois moyennes mobiles (MME), avec SL/TP basés sur l'ATR.
    Peut optionnellement anticiper les croisements de MME.
    Les indicateurs (MME courte/moyenne/longue, ATR, et potentiellement Pentes des MME)
    sont fournis à la stratégie via des colonnes suffixées par '_strat'.
    """
    REQUIRED_PARAMS = [
        'ma_short_period', 'ma_medium_period', 'ma_long_period',
        'indicateur_frequence_mms', # Fréquence pour MA Courte
        'indicateur_frequence_mmm', # Fréquence pour MA Moyenne
        'indicateur_frequence_mml', # Fréquence pour MA Longue
        'atr_period_sl_tp', 'atr_base_frequency_sl_tp',
        'sl_atr_mult', 'tp_atr_mult',
        'allow_shorting', 'order_type_preference',
        'anticipate_crossovers', 
        # Les paramètres d'anticipation ne sont requis que si anticipate_crossovers est True
        # La validation plus fine peut être dans __init__
    ]

    def __init__(self, params: dict):
        super().__init__(params)
        # self._validate_strategy_params() # Appelée par super().__init__

        # Noms des colonnes d'indicateurs attendues
        self.ma_short_col_strat = "MA_SHORT_strat"
        self.ma_medium_col_strat = "MA_MEDIUM_strat"
        self.ma_long_col_strat = "MA_LONG_strat"
        self.atr_col_strat = "ATR_strat"

        self.allow_shorting = bool(self.get_param('allow_shorting', False))
        self.order_type_preference = self.get_param('order_type_preference', "MARKET") # Défaut à MARKET

        self.anticipate_crossovers = bool(self.get_param('anticipate_crossovers', False))
        if self.anticipate_crossovers:
            self.slope_ma_short_col_strat = "SLOPE_MA_SHORT_strat"
            self.slope_ma_medium_col_strat = "SLOPE_MA_MEDIUM_strat"
            self.anticipation_slope_period = int(self.get_param('anticipation_slope_period', 3))
            if self.anticipation_slope_period < 2:
                logger.warning(f"{self.log_prefix} anticipation_slope_period ({self.anticipation_slope_period}) < 2. Forcé à 2.")
                self.anticipation_slope_period = 2
            self.anticipation_convergence_threshold_pct = float(self.get_param('anticipation_convergence_threshold_pct', 0.005))
            # Ajouter les paramètres d'anticipation aux requis si anticipation activée
            if 'anticipation_slope_period' not in self.params or self.get_param('anticipation_slope_period') is None:
                 raise ValueError("Missing 'anticipation_slope_period' required when 'anticipate_crossovers' is true.")
            if 'anticipation_convergence_threshold_pct' not in self.params or self.get_param('anticipation_convergence_threshold_pct') is None:
                 raise ValueError("Missing 'anticipation_convergence_threshold_pct' required when 'anticipate_crossovers' is true.")
        else:
            self.slope_ma_short_col_strat = None
            self.slope_ma_medium_col_strat = None

        self._signals: Optional[pd.DataFrame] = None
        logger.info(f"{self.log_prefix} Stratégie initialisée. Anticipation: {self.anticipate_crossovers}. Paramètres: {self.params}")

    def _calculate_slope(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calcule la pente d'une série en utilisant une régression linéaire sur une fenêtre glissante.
        Helper local si les pentes ne sont pas fournies en tant que colonnes _strat.
        """
        if not isinstance(series, pd.Series) or series.empty or series.isnull().all() or len(series) < window or window < 2:
            return pd.Series([np.nan] * len(series), index=series.index, name=f"{series.name}_slope{window}" if series.name else f"slope{window}")
        
        # Fonction pour calculer la pente sur une fenêtre
        def get_slope_value(y_values_window):
            y_clean = y_values_window.dropna()
            if len(y_clean) < 2: # Nécessite au moins 2 points pour une pente
                return np.nan
            x_clean = np.arange(len(y_clean))
            try:
                # polyfit retourne [pente, ordonnée_origine]
                slope = np.polyfit(x_clean, y_clean, 1)[0]
                return slope
            except (np.linalg.LinAlgError, TypeError, ValueError): # Gérer les erreurs de polyfit
                return np.nan

        slopes = series.rolling(window=window, min_periods=window).apply(get_slope_value, raw=False)
        return slopes.rename(f"{series.name}_slope{window}" if series.name else f"slope{window}")


    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie la présence des colonnes d'indicateurs *_strat.
        Si l'anticipation est activée et que les pentes ne sont pas fournies en tant que *_strat,
        cette méthode peut les calculer à partir des MME *_strat.
        """
        df = data.copy()
        # logger.debug(f"{self.log_prefix} Entrée _calculate_indicators. Colonnes reçues: {df.columns.tolist()}")

        expected_base_strat_cols = [
            self.ma_short_col_strat, self.ma_medium_col_strat,
            self.ma_long_col_strat, self.atr_col_strat
        ]
        for col_name in expected_base_strat_cols:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur de base attendue '{col_name}' manquante. Ajoutée avec NaN.")
                df[col_name] = np.nan

        if self.anticipate_crossovers:
            # Vérifier si les colonnes de pente _strat sont fournies
            if self.slope_ma_short_col_strat and self.slope_ma_short_col_strat not in df.columns:
                logger.info(f"{self.log_prefix} Colonne '{self.slope_ma_short_col_strat}' non fournie. Tentative de calcul à partir de '{self.ma_short_col_strat}'.")
                if self.ma_short_col_strat in df and df[self.ma_short_col_strat].notna().any():
                    df[self.slope_ma_short_col_strat] = self._calculate_slope(df[self.ma_short_col_strat], self.anticipation_slope_period)
                else:
                    df[self.slope_ma_short_col_strat] = np.nan
            
            if self.slope_ma_medium_col_strat and self.slope_ma_medium_col_strat not in df.columns:
                logger.info(f"{self.log_prefix} Colonne '{self.slope_ma_medium_col_strat}' non fournie. Tentative de calcul à partir de '{self.ma_medium_col_strat}'.")
                if self.ma_medium_col_strat in df and df[self.ma_medium_col_strat].notna().any():
                     df[self.slope_ma_medium_col_strat] = self._calculate_slope(df[self.ma_medium_col_strat], self.anticipation_slope_period)
                else:
                    df[self.slope_ma_medium_col_strat] = np.nan
        
        required_ohlc = ['open', 'high', 'low', 'close']
        for col in required_ohlc:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne OHLC de base '{col}' manquante. Ajoutée avec NaN.")
                df[col] = np.nan
        
        # logger.debug(f"{self.log_prefix} Sortie _calculate_indicators. Colonnes présentes: {df.columns.tolist()}")
        return df

    def generate_signals(self, data: pd.DataFrame) -> None:
        # logger.debug(f"{self.log_prefix} Génération des signaux...")
        df_with_indicators = self._calculate_indicators(data) # Assure que toutes les colonnes _strat (et pentes si besoin) sont là

        required_cols_for_signal = [
            self.ma_short_col_strat, self.ma_medium_col_strat, self.ma_long_col_strat,
            self.atr_col_strat, 'close'
        ]
        if self.anticipate_crossovers and self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat:
            required_cols_for_signal.extend([self.slope_ma_short_col_strat, self.slope_ma_medium_col_strat])

        if df_with_indicators.empty or \
           any(col not in df_with_indicators.columns for col in required_cols_for_signal) or \
           df_with_indicators[[self.ma_short_col_strat, self.ma_medium_col_strat, self.atr_col_strat, 'close']].isnull().all().all() or \
           len(df_with_indicators) < 2:
            logger.warning(f"{self.log_prefix} Données/colonnes insuffisantes pour générer les signaux. Signaux vides générés.")
            self._signals = pd.DataFrame(index=data.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            self._signals[self._signals.select_dtypes(include=['object', 'bool']).columns] = False
            return

        df = df_with_indicators
        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))

        ma_short = df[self.ma_short_col_strat]
        ma_medium = df[self.ma_medium_col_strat]
        ma_long = df[self.ma_long_col_strat]
        atr_val = df[self.atr_col_strat]
        close_price = df['close']

        # Conditions de croisement réelles
        actual_long_entry_cross = (ma_short > ma_medium) & (ma_short.shift(1) <= ma_medium.shift(1))
        actual_long_exit_cross = (ma_short < ma_medium) & (ma_short.shift(1) >= ma_medium.shift(1))
        
        actual_short_entry_cross = pd.Series(False, index=df.index)
        actual_short_exit_cross = pd.Series(False, index=df.index)
        if self.allow_shorting:
            actual_short_entry_cross = (ma_short < ma_medium) & (ma_short.shift(1) >= ma_medium.shift(1))
            actual_short_exit_cross = (ma_short > ma_medium) & (ma_short.shift(1) <= ma_medium.shift(1))

        # Conditions d'anticipation
        anticipated_long_entry = pd.Series(False, index=df.index)
        anticipated_long_exit = pd.Series(False, index=df.index)
        anticipated_short_entry = pd.Series(False, index=df.index)
        anticipated_short_exit = pd.Series(False, index=df.index)

        if self.anticipate_crossovers and self.slope_ma_short_col_strat and self.slope_ma_medium_col_strat and \
           self.slope_ma_short_col_strat in df and self.slope_ma_medium_col_strat in df and \
           df[self.slope_ma_short_col_strat].notna().any() and df[self.slope_ma_medium_col_strat].notna().any():
            
            slope_short = df[self.slope_ma_short_col_strat]
            slope_medium = df[self.slope_ma_medium_col_strat]
            
            convergence_distance = ma_medium * self.anticipation_convergence_threshold_pct
            ma_diff_abs = abs(ma_short - ma_medium)

            # Anticipation Achat Long
            is_converging_up = slope_short > slope_medium
            is_below_and_closing_for_long = (ma_short < ma_medium) & (ma_diff_abs < convergence_distance)
            main_trend_bullish = ma_medium > ma_long # Filtre de tendance principale
            anticipated_long_entry = is_converging_up & is_below_and_closing_for_long & main_trend_bullish

            # Anticipation Sortie Long (ou Entrée Short si anticipation_exit_is_entry_short)
            is_converging_down_for_exit_long = slope_short < slope_medium
            is_above_and_closing_for_long_exit = (ma_short > ma_medium) & (ma_diff_abs < convergence_distance)
            anticipated_long_exit = is_converging_down_for_exit_long & is_above_and_closing_for_long_exit

            if self.allow_shorting:
                # Anticipation Entrée Short
                is_converging_down_for_entry_short = slope_short < slope_medium
                is_above_and_closing_for_short_entry = (ma_short > ma_medium) & (ma_diff_abs < convergence_distance)
                main_trend_bearish = ma_medium < ma_long # Filtre de tendance principale
                anticipated_short_entry = is_converging_down_for_entry_short & is_above_and_closing_for_short_entry & main_trend_bearish
                
                # Anticipation Sortie Short
                is_converging_up_for_exit_short = slope_short > slope_medium
                is_below_and_closing_for_short_exit = (ma_short < ma_medium) & (ma_diff_abs < convergence_distance)
                anticipated_short_exit = is_converging_up_for_exit_short & is_below_and_closing_for_short_exit
        
        # Combiner les signaux réels et anticipés
        signals_df = pd.DataFrame(index=df.index)
        signals_df['entry_long'] = actual_long_entry_cross | anticipated_long_entry
        signals_df['exit_long'] = actual_long_exit_cross | anticipated_long_exit
        signals_df['entry_short'] = actual_short_entry_cross | anticipated_short_entry
        signals_df['exit_short'] = actual_short_exit_cross | anticipated_short_exit
        
        # Calcul SL/TP
        signals_df['sl'] = np.nan
        signals_df['tp'] = np.nan
        valid_sltp_data = atr_val.notna() & close_price.notna()

        signals_df.loc[signals_df['entry_long'] & valid_sltp_data, 'sl'] = close_price - (atr_val * sl_atr_mult)
        signals_df.loc[signals_df['entry_long'] & valid_sltp_data, 'tp'] = close_price + (atr_val * tp_atr_mult)
        if self.allow_shorting:
            signals_df.loc[signals_df['entry_short'] & valid_sltp_data, 'sl'] = close_price + (atr_val * sl_atr_mult)
            signals_df.loc[signals_df['entry_short'] & valid_sltp_data, 'tp'] = close_price - (atr_val * tp_atr_mult)

        self._signals = signals_df[['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']].reindex(data.index)
        # logger.debug(f"{self.log_prefix} Signaux générés. Longs: {self._signals['entry_long'].sum()}, Shorts: {self._signals['entry_short'].sum()}")


    def generate_order_request(self,
                               data: pd.DataFrame,
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        # logger.debug(f"{self.log_prefix} Appel generate_order_request. Pos: {current_position}, Capital: {available_capital:.2f}")

        if current_position != 0: # Gère uniquement les entrées
            # logger.debug(f"{self.log_prefix} Position existante. Pas de nouvelle requête d'ordre d'entrée.")
            return None
        
        if data.empty or len(data) < 2: # Besoin d'au moins 2 points pour les MME et pentes.
            logger.warning(f"{self.log_prefix} Données d'entrée vides ou insuffisantes (lignes: {len(data)}) pour generate_order_request.")
            return None

        # S'assurer que les indicateurs sont présents/calculés (surtout les pentes si anticipation)
        df_verified_indicators = self._calculate_indicators(data.copy())
        
        latest_signal_info = self.get_signals() # Récupérer les signaux déjà générés pour la dernière barre
        if latest_signal_info is None or latest_signal_info.empty:
            logger.warning(f"{self.log_prefix} Aucun signal disponible via get_signals(). Pas de requête d'ordre.")
            return None
        
        # Utiliser la dernière ligne de signaux pour la décision
        latest_signals = latest_signal_info.iloc[-1]
        # Utiliser la dernière ligne de données (avec indicateurs vérifiés) pour les prix
        latest_data_row = df_verified_indicators.iloc[-1]

        entry_price_theoretical = latest_data_row['open'] # Entrée à l'ouverture de la bougie
        atr_value_for_sltp = latest_data_row.get(self.atr_col_strat) # ATR_strat

        if pd.isna(entry_price_theoretical) or pd.isna(atr_value_for_sltp) or atr_value_for_sltp <= 1e-9:
            # logger.debug(f"{self.log_prefix} Prix d'entrée théorique ou ATR invalide. Entrée: {entry_price_theoretical}, ATR: {atr_value_for_sltp}. Pas de requête.")
            return None

        sl_atr_mult = float(self.get_param('sl_atr_mult'))
        tp_atr_mult = float(self.get_param('tp_atr_mult'))
        
        side: Optional[str] = None
        sl_price_raw: Optional[float] = None
        tp_price_raw: Optional[float] = None

        if latest_signals.get('entry_long', False):
            side = 'BUY'
            sl_price_raw = entry_price_theoretical - (atr_value_for_sltp * sl_atr_mult)
            tp_price_raw = entry_price_theoretical + (atr_value_for_sltp * tp_atr_mult)
            # logger.info(f"{self.log_prefix} Signal d'ACHAT (TripleMA) détecté pour requête d'ordre.")
        elif self.allow_shorting and latest_signals.get('entry_short', False):
            side = 'SELL'
            sl_price_raw = entry_price_theoretical + (atr_value_for_sltp * sl_atr_mult)
            tp_price_raw = entry_price_theoretical - (atr_value_for_sltp * tp_atr_mult)
            # logger.info(f"{self.log_prefix} Signal de VENTE (TripleMA) détecté pour requête d'ordre.")

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
            
            entry_order_params = self._build_entry_params_formatted(
                symbol=symbol, side=side, quantity_str=quantity_str,
                entry_price_str=entry_price_str if self.order_type_preference == "LIMIT" else None,
                order_type=self.order_type_preference
            )
            if not entry_order_params:
                # logger.error(f"{self.log_prefix} Échec de la construction des paramètres d'ordre formatés.")
                return None

            sl_tp_raw_prices = {'sl_price': sl_price_raw, 'tp_price': tp_price_raw}
            # logger.info(f"{self.log_prefix} Requête d'ordre générée: {entry_order_params}, SL/TP bruts: {sl_tp_raw_prices}")
            return entry_order_params, sl_tp_raw_prices

        # logger.debug(f"{self.log_prefix} Aucune condition d'entrée TripleMA remplie pour une nouvelle requête d'ordre.")
        return None