# src/strategies/ma_crossover_strategy.py
"""
Stratégie de trading basée sur le croisement de deux moyennes mobiles (MA),
avec gestion du Stop-Loss (SL) et Take-Profit (TP) basée sur l'Average True Range (ATR).
"""
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
# Les utilitaires d'exchange_utils sont accessibles via les méthodes de BaseStrategy
# ou directement si nécessaire pour des logiques très spécifiques non couvertes.

logger = logging.getLogger(__name__)

class MaCrossoverStrategy(BaseStrategy):
    """
    Stratégie de croisement de moyennes mobiles.
    Génère des signaux d'achat lorsque la MA rapide croise au-dessus de la MA lente,
    et des signaux de vente (ou de sortie de long) lorsque la MA rapide croise
    en dessous de la MA lente. Le SL et le TP sont basés sur l'ATR.
    """

    REQUIRED_PARAMS: List[str] = [
        'fast_ma_period', 'slow_ma_period', 'ma_type',
        'atr_period_sl_tp', 'sl_atr_multiplier', 'tp_atr_multiplier',
        'indicateur_frequence_ma_rapide', 'indicateur_frequence_ma_lente',
        'atr_base_frequency_sl_tp',
        'capital_allocation_pct',
        'order_type_preference'
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        """
        Initialise la stratégie MaCrossoverStrategy.

        Args:
            strategy_name (str): Nom de la stratégie (clé de configuration).
            symbol (str): Symbole de la paire de trading (ex: BTCUSDT).
            params (Dict[str, Any]): Paramètres spécifiques à cette instance de stratégie.
        """
        super().__init__(strategy_name, symbol, params)
        # self.log_prefix est déjà défini dans BaseStrategy

        # Noms des colonnes pour les indicateurs calculés qui seront dans le DataFrame
        self.fast_ma_col_strat: str = "MA_FAST_strat"
        self.slow_ma_col_strat: str = "MA_SLOW_strat"
        self.atr_col_strat: str = "ATR_strat"
        
        # Le message d'initialisation est déjà dans BaseStrategy après _validate_params
        # logger.info(f"{self.log_prefix} Stratégie MaCrossoverStrategy initialisée.")

    def _validate_params(self) -> None:
        """
        Valide les paramètres spécifiques à MaCrossoverStrategy.
        Lève une ValueError si un paramètre est invalide.
        """
        missing_params = [p for p in self.REQUIRED_PARAMS if self.get_param(p) is None]
        if missing_params:
            raise ValueError(f"{self.log_prefix} Paramètres requis manquants : {', '.join(missing_params)}")

        fast_ma_p = self.get_param('fast_ma_period')
        slow_ma_p = self.get_param('slow_ma_period')
        if not (isinstance(fast_ma_p, int) and fast_ma_p > 0):
            raise ValueError(f"{self.log_prefix} 'fast_ma_period' ({fast_ma_p}) doit être un entier positif.")
        if not (isinstance(slow_ma_p, int) and slow_ma_p > 0):
            raise ValueError(f"{self.log_prefix} 'slow_ma_period' ({slow_ma_p}) doit être un entier positif.")
        if fast_ma_p >= slow_ma_p:
            # Ceci est souvent un avertissement plutôt qu'une erreur bloquante,
            # mais pour une stratégie de croisement, c'est généralement une erreur de logique.
            raise ValueError(f"{self.log_prefix} 'fast_ma_period' ({fast_ma_p}) doit être inférieur à 'slow_ma_period' ({slow_ma_p}).")

        ma_type_val = self.get_param('ma_type')
        supported_ma_types = ['sma', 'ema', 'wma', 'hma', 'tema', 'dema', 'rma', 'vwma', 'zlma'] # Liste étendue
        if not isinstance(ma_type_val, str) or ma_type_val.lower() not in supported_ma_types:
            raise ValueError(f"{self.log_prefix} 'ma_type' ({ma_type_val}) invalide. Supportés : {supported_ma_types}")

        atr_p = self.get_param('atr_period_sl_tp')
        if not (isinstance(atr_p, int) and atr_p > 0):
            raise ValueError(f"{self.log_prefix} 'atr_period_sl_tp' ({atr_p}) doit être un entier positif.")

        sl_mult = self.get_param('sl_atr_multiplier')
        tp_mult = self.get_param('tp_atr_multiplier')
        if not (isinstance(sl_mult, (int, float)) and sl_mult > 0):
            raise ValueError(f"{self.log_prefix} 'sl_atr_multiplier' ({sl_mult}) doit être un nombre positif.")
        if not (isinstance(tp_mult, (int, float)) and tp_mult > 0):
            raise ValueError(f"{self.log_prefix} 'tp_atr_multiplier' ({tp_mult}) doit être un nombre positif.")

        cap_alloc = self.get_param('capital_allocation_pct')
        if not (isinstance(cap_alloc, (int, float)) and 0 < cap_alloc <= 1.0):
            raise ValueError(f"{self.log_prefix} 'capital_allocation_pct' ({cap_alloc}) doit être entre 0 (exclusif) et 1 (inclusif).")

        order_type_pref_val = self.get_param('order_type_preference')
        if order_type_pref_val not in ["MARKET", "LIMIT"]:
            raise ValueError(f"{self.log_prefix} 'order_type_preference' ({order_type_pref_val}) doit être 'MARKET' ou 'LIMIT'.")
        
        # Valider les fréquences (doivent être des chaînes non vides si fournies)
        for freq_param_name in ['indicateur_frequence_ma_rapide', 'indicateur_frequence_ma_lente', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' ("
                                 f"{freq_val}) doit être une chaîne de caractères non vide.")
        
        logger.debug(f"{self.log_prefix} Validation des paramètres terminée avec succès.")


    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        """
        Déclare les indicateurs requis par MaCrossoverStrategy et leurs configurations.
        Ces configurations seront utilisées par IndicatorCalculator.
        """
        configs = [
            {
                'indicator_name': str(self.params['ma_type']).lower(), # Ex: 'sma', 'ema'
                'params': {'length': int(self.params['fast_ma_period'])},
                # 'inputs' est implicite pour les MAs simples (close),
                # IndicatorCalculator utilisera 'source_kline_frequency_param_name'
                # pour déterminer la colonne source (ex: 'Klines_5min_close' ou 'close').
                'source_kline_frequency_param_name': 'indicateur_frequence_ma_rapide',
                'output_column_name': self.fast_ma_col_strat
            },
            {
                'indicator_name': str(self.params['ma_type']).lower(),
                'params': {'length': int(self.params['slow_ma_period'])},
                'source_kline_frequency_param_name': 'indicateur_frequence_ma_lente',
                'output_column_name': self.slow_ma_col_strat
            },
            {
                'indicator_name': 'atr',
                'params': {'length': int(self.params['atr_period_sl_tp'])},
                # Pour ATR, IndicatorCalculator s'attendra à 'high', 'low', 'close'
                # de la fréquence spécifiée par 'atr_base_frequency_sl_tp'.
                'source_kline_frequency_param_name': 'atr_base_frequency_sl_tp',
                'output_column_name': self.atr_col_strat
            }
        ]
        logger.debug(f"{self.log_prefix} Configurations d'indicateurs requises : {configs}")
        return configs

    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie la présence des colonnes d'indicateurs `_strat` attendues dans data_feed
        (qui proviennent de IndicatorCalculator). Applique un ffill si nécessaire.

        Args:
            data_feed (pd.DataFrame): DataFrame avec OHLCV et indicateurs calculés.

        Returns:
            pd.DataFrame: DataFrame prêt pour la génération de signaux.
        """
        df = data_feed.copy()
        
        # Colonnes _strat attendues par cette stratégie
        expected_strat_cols = [self.fast_ma_col_strat, self.slow_ma_col_strat, self.atr_col_strat]
        
        missing_cols_added_nan = False
        for col_name in expected_strat_cols:
            if col_name not in df.columns:
                logger.warning(f"{self.log_prefix} Colonne indicateur attendue '{col_name}' manquante dans data_feed. Ajout avec NaN.")
                df[col_name] = np.nan
                missing_cols_added_nan = True
        
        # S'assurer que les colonnes OHLCV de base sont présentes (elles devraient l'être)
        base_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        for col in base_ohlcv:
            if col not in df.columns:
                logger.error(f"{self.log_prefix} Colonne OHLCV de base '{col}' manquante dans data_feed. Ceci est critique.")
                # Retourner un DataFrame potentiellement modifié mais loguer l'erreur grave.
                # Ou lever une exception si c'est bloquant.
                df[col] = np.nan # Pour éviter les KeyErrors plus tard, mais c'est un problème.
                missing_cols_added_nan = True

        if missing_cols_added_nan:
             logger.debug(f"{self.log_prefix} Après vérification des colonnes _calculate_indicators, colonnes actuelles: {df.columns.tolist()}")


        # Appliquer ffill aux colonnes _strat pour propager les valeurs des indicateurs
        # Cela est utile car les indicateurs calculés sur des timeframes plus longs
        # n'auront de nouvelles valeurs qu'à la clôture de leur barre agrégée.
        # Le ffill propage cette valeur sur les barres 1-min suivantes.
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
        Génère les signaux de trading pour la stratégie de croisement de MA.
        """
        signal_type: int = 0  # 0: HOLD, 1: BUY, -1: SELL, 2: EXIT
        limit_price: Optional[float] = None
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        # S'assurer qu'il y a assez de données pour comparer la barre actuelle et la précédente
        if len(data_with_indicators) < 2:
            logger.debug(f"{self.log_prefix} Pas assez de données ({len(data_with_indicators)}) pour générer des signaux.")
            return signal_type, self.get_param("order_type_preference"), limit_price, sl_price, tp_price, self.get_param('capital_allocation_pct')

        latest_row = data_with_indicators.iloc[-1]
        previous_row = data_with_indicators.iloc[-2]

        # Récupérer les valeurs des indicateurs et du prix de clôture
        ma_fast_curr = latest_row.get(self.fast_ma_col_strat)
        ma_slow_curr = latest_row.get(self.slow_ma_col_strat)
        ma_fast_prev = previous_row.get(self.fast_ma_col_strat)
        ma_slow_prev = previous_row.get(self.slow_ma_col_strat)
        
        close_price_curr = latest_row.get('close')
        atr_value_curr = latest_row.get(self.atr_col_strat)

        # Vérifier si les données nécessaires sont valides (non-NaN)
        essential_values = [ma_fast_curr, ma_slow_curr, ma_fast_prev, ma_slow_prev, close_price_curr, atr_value_curr]
        if any(pd.isna(val) for val in essential_values):
            logger.debug(f"{self.log_prefix} Valeurs d'indicateur ou de prix manquantes (NaN) à {latest_row.name}. Signal HOLD.")
            # Tenter de loguer quelles valeurs sont NaN
            nan_details = {
                "ma_fast_curr": ma_fast_curr, "ma_slow_curr": ma_slow_curr,
                "ma_fast_prev": ma_fast_prev, "ma_slow_prev": ma_slow_prev,
                "close_price_curr": close_price_curr, "atr_value_curr": atr_value_curr
            }
            logger.debug(f"{self.log_prefix} Détails des valeurs (NaN check): {nan_details}")
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        # Logique de croisement
        bullish_crossover = (ma_fast_curr > ma_slow_curr) and (ma_fast_prev <= ma_slow_prev) # type: ignore
        bearish_crossover = (ma_fast_curr < ma_slow_curr) and (ma_fast_prev >= ma_slow_prev) # type: ignore

        sl_multiplier = float(self.get_param('sl_atr_multiplier'))
        tp_multiplier = float(self.get_param('tp_atr_multiplier'))

        if not current_position_open:
            if bullish_crossover:
                signal_type = 1 # Signal d'achat (LONG)
                if atr_value_curr > 0: # type: ignore
                    sl_price = close_price_curr - (atr_value_curr * sl_multiplier) # type: ignore
                    tp_price = close_price_curr + (atr_value_curr * tp_multiplier) # type: ignore
                logger.info(f"{self.log_prefix} Signal BUY @ {close_price_curr:.4f}. SL={sl_price}, TP={tp_price}")
            elif bearish_crossover:
                signal_type = -1 # Signal de vente (SHORT)
                if atr_value_curr > 0: # type: ignore
                    sl_price = close_price_curr + (atr_value_curr * sl_multiplier) # type: ignore
                    tp_price = close_price_curr - (atr_value_curr * tp_multiplier) # type: ignore
                logger.info(f"{self.log_prefix} Signal SELL @ {close_price_curr:.4f}. SL={sl_price}, TP={tp_price}")
        else: # Une position est ouverte
            if current_position_direction == 1 and bearish_crossover: # Position LONG et croisement baissier
                signal_type = 2 # Signal de sortie de position
                logger.info(f"{self.log_prefix} Signal EXIT LONG (croisement baissier) @ {close_price_curr:.4f}")
            elif current_position_direction == -1 and bullish_crossover: # Position SHORT et croisement haussier
                signal_type = 2 # Signal de sortie de position
                logger.info(f"{self.log_prefix} Signal EXIT SHORT (croisement haussier) @ {close_price_curr:.4f}")
        
        order_type_preference = str(self.get_param("order_type_preference", "MARKET"))
        if signal_type != 0 and order_type_preference == "LIMIT":
            # Pour un ordre limite, on peut suggérer le prix de clôture actuel
            # ou un prix légèrement ajusté (ex: pour s'assurer d'être preneur ou placeur)
            # Pour la simplicité, utilisons le prix de clôture.
            limit_price = float(close_price_curr)

        position_size_pct = float(self.get_param('capital_allocation_pct', 1.0))

        return signal_type, order_type_preference, limit_price, sl_price, tp_price, position_size_pct

    def generate_order_request(self,
                               data: pd.DataFrame, # Reçoit les données avec indicateurs déjà calculés par IndicatorCalculator
                               current_position: int, # 0: pas de pos, 1: long, -1: short
                               available_capital: float,
                               symbol_info: Dict[str, Any] # C'est le pair_config
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre d'ENTRÉE pour le trading en direct.
        Ne génère un ordre que si current_position est 0.
        """
        if current_position != 0:
            logger.debug(f"{self.log_prefix} [Live] Position déjà ouverte (état: {current_position}). Pas de nouvelle requête d'ordre d'entrée générée.")
            return None

        # 1. Préparer/Vérifier les données d'indicateurs
        # _calculate_indicators s'assure que les colonnes _strat sont prêtes et ffillées
        data_with_indicators = self._calculate_indicators(data.copy())
        if data_with_indicators.empty or len(data_with_indicators) < 2: # Besoin d'au moins 2 barres pour la logique de croisement
            logger.warning(f"{self.log_prefix} [Live] Données insuffisantes après _calculate_indicators pour generate_order_request. Shape: {data_with_indicators.shape}")
            return None

        # 2. Générer le signal de base et les suggestions SL/TP
        # Pour une nouvelle entrée, current_position_open=False, current_position_direction=0, current_entry_price=0.0
        signal, order_type, limit_price_sugg, sl_price_raw, tp_price_raw, pos_size_pct = \
            self._generate_signals(data_with_indicators, False, 0, 0.0)

        if signal not in [1, -1]: # Pas de signal d'entrée (BUY ou SELL)
            logger.debug(f"{self.log_prefix} [Live] Aucun signal d'entrée (1 ou -1) généré. Signal actuel : {signal}")
            return None
        
        # 3. Déterminer le prix d'entrée théorique pour le calcul de la quantité
        latest_bar = data_with_indicators.iloc[-1]
        # Pour un ordre MARKET, le prix d'entrée sera proche de l'ouverture de la prochaine barre (simulé par le close actuel ou l'open actuel)
        # Pour un ordre LIMIT, ce sera le limit_price_sugg.
        entry_price_theoretical: float
        if order_type == "LIMIT" and limit_price_sugg is not None:
            entry_price_theoretical = limit_price_sugg
        else: # Pour MARKET, utiliser le prix de clôture de la dernière barre disponible comme estimation
            entry_price_theoretical = float(latest_bar.get('close', 0.0))
            if entry_price_theoretical <= 0: # Fallback si close est invalide
                entry_price_theoretical = float(latest_bar.get('open', 0.0))
        
        if pd.isna(entry_price_theoretical) or entry_price_theoretical <= 0:
            logger.error(f"{self.log_prefix} [Live] Prix d'entrée théorique invalide ({entry_price_theoretical}). Impossible de calculer la quantité.")
            return None

        # 4. Calculer la quantité en utilisant la méthode de BaseStrategy
        # self.quantity_precision et self.pair_config sont définis par set_backtest_context,
        # qui doit être appelé par LiveTradingManager avant d'utiliser cette méthode en live.
        if self.quantity_precision is None or self.pair_config is None:
            logger.error(f"{self.log_prefix} [Live] Contexte de backtest (précisions, pair_config) non défini. "
                         "Appelez set_backtest_context sur l'instance de stratégie. Impossible de calculer la quantité.")
            return None
            
        quantity_base = self._calculate_quantity(
            entry_price=entry_price_theoretical,
            available_capital=available_capital,
            qty_precision=self.quantity_precision, # Vient de set_backtest_context
            symbol_info=self.pair_config,          # Vient de set_backtest_context
            symbol=self.symbol,
            position_size_pct=pos_size_pct
        )

        if quantity_base is None or quantity_base <= 1e-9: # Utiliser un petit seuil pour les flottants
            logger.warning(f"{self.log_prefix} [Live] Quantité calculée nulle ou invalide ({quantity_base}). Pas d'ordre généré.")
            return None

        # 5. Formater les paramètres de l'ordre d'entrée
        # S'assurer que price_precision est défini
        if self.price_precision is None:
            logger.error(f"{self.log_prefix} [Live] price_precision non défini. Impossible de formater le prix limite.")
            return None

        entry_price_for_order_str: Optional[str] = None
        if order_type == "LIMIT" and limit_price_sugg is not None:
            # ajuster_precision est hérité ou importé
            adjusted_limit_price = adjust_precision(limit_price_sugg, self.price_precision, get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize'))
            if adjusted_limit_price is None:
                logger.error(f"{self.log_prefix} [Live] Échec de l'ajustement du prix limite {limit_price_sugg}.")
                return None
            entry_price_for_order_str = f"{adjusted_limit_price:.{self.price_precision}f}"
        
        # Utiliser les méthodes de BaseStrategy pour construire les paramètres
        # La quantité doit être une chaîne formatée avec la bonne précision
        quantity_str_formatted = f"{quantity_base:.{self.quantity_precision}f}"

        entry_order_params = self._build_entry_params_formatted(
            side="BUY" if signal == 1 else "SELL",
            quantity_str=quantity_str_formatted,
            order_type=str(order_type), # S'assurer que c'est une chaîne
            entry_price_str=entry_price_for_order_str
            # time_in_force peut être ajouté si nécessaire
        )
        if not entry_order_params: # Si la construction échoue pour une raison quelconque
            logger.error(f"{self.log_prefix} [Live] Échec de la construction des paramètres de l'ordre d'entrée.")
            return None

        # Préparer le dictionnaire pour SL/TP bruts (non ajustés ici, l'ajustement se fait dans OrderExecutionClient ou LiveManager)
        sl_tp_raw_prices_dict: Dict[str, float] = {}
        if sl_price_raw is not None:
            sl_tp_raw_prices_dict['sl_price'] = float(sl_price_raw)
        if tp_price_raw is not None:
            sl_tp_raw_prices_dict['tp_price'] = float(tp_price_raw)
        
        logger.info(f"{self.log_prefix} [Live] Requête d'ordre d'entrée générée : {entry_order_params}. "
                    f"SL/TP bruts suggérés : {sl_tp_raw_prices_dict}")
        return entry_order_params, sl_tp_raw_prices_dict

