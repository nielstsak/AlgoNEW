# src/strategies/ma_crossover_strategy.py
"""
Stratégie de trading basée sur le croisement de deux moyennes mobiles (MA),
avec gestion du Stop-Loss (SL) et Take-Profit (TP) basée sur l'Average True Range (ATR).
"""
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd # Assurer l'import pour Pylance

from src.strategies.base import BaseStrategy # BaseStrategy a été refactorisé
from src.data.data_utils import get_kline_prefix_effective
from src.utils.exchange_utils import adjust_precision, get_filter_value


logger = logging.getLogger(__name__)

class MaCrossoverStrategy(BaseStrategy):
    """
    Stratégie de croisement de moyennes mobiles.
    """

    REQUIRED_PARAMS: List[str] = [
        'fast_ma_period', 'slow_ma_period', 'ma_type',
        'atr_period_sl_tp', 'sl_atr_multiplier', 'tp_atr_multiplier',
        'indicateur_frequence_ma_rapide', 'indicateur_frequence_ma_lente',
        'atr_base_frequency_sl_tp',
        'capital_allocation_pct',
        'order_type_preference',
        'margin_leverage'
    ]

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        super().__init__(strategy_name, symbol, params)
        self.fast_ma_col_strat: str = "MA_FAST_strat"
        self.slow_ma_col_strat: str = "MA_SLOW_strat"
        self.atr_col_strat: str = "ATR_strat"
        self._log("MaCrossoverStrategy instance créée.", level=2)


    def validate_params(self) -> None: # Implémentation de la méthode abstraite
        """Valide les paramètres spécifiques à MaCrossoverStrategy."""
        # Sourcery: Use named expression (walrus operator)
        if missing_params := [
            p for p in self.REQUIRED_PARAMS if self.get_param(p) is None
        ]:
            raise ValueError(f"{self.log_prefix} Paramètres requis manquants : {', '.join(missing_params)}")

        fast_ma_p = self.get_param('fast_ma_period')
        slow_ma_p = self.get_param('slow_ma_period')
        if not (isinstance(fast_ma_p, int) and fast_ma_p > 0):
            raise ValueError(f"{self.log_prefix} 'fast_ma_period' ({fast_ma_p}) doit être un entier positif.")
        if not (isinstance(slow_ma_p, int) and slow_ma_p > 0):
            raise ValueError(f"{self.log_prefix} 'slow_ma_period' ({slow_ma_p}) doit être un entier positif.")
        if fast_ma_p >= slow_ma_p:
            raise ValueError(f"{self.log_prefix} 'fast_ma_period' ({fast_ma_p}) doit être inférieur à 'slow_ma_period' ({slow_ma_p}).")

        ma_type_val = self.get_param('ma_type')
        supported_ma_types = ['sma', 'ema', 'wma', 'hma', 'tema', 'dema', 'rma', 'vwma', 'zlma']
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

        margin_lev = self.get_param('margin_leverage')
        if not (isinstance(margin_lev, (int, float)) and margin_lev >= 1.0):
            raise ValueError(f"{self.log_prefix} 'margin_leverage' ({margin_lev}) doit être >= 1.0.")

        order_type_pref_val = self.get_param('order_type_preference')
        if order_type_pref_val not in ["MARKET", "LIMIT"]:
            raise ValueError(f"{self.log_prefix} 'order_type_preference' ({order_type_pref_val}) doit être 'MARKET' ou 'LIMIT'.")

        for freq_param_name in ['indicateur_frequence_ma_rapide', 'indicateur_frequence_ma_lente', 'atr_base_frequency_sl_tp']:
            freq_val = self.get_param(freq_param_name)
            if not isinstance(freq_val, str) or not freq_val.strip():
                raise ValueError(f"{self.log_prefix} Paramètre de fréquence '{freq_param_name}' ("
                                 f"{freq_val}) doit être une chaîne de caractères non vide.")

        self._log("Validation des paramètres de MaCrossoverStrategy terminée avec succès.", level=2)


    def get_required_indicator_configs(self) -> List[Dict[str, Any]]: # Implémentation de la méthode abstraite
        freq_ma_rapide_param = str(self.params['indicateur_frequence_ma_rapide'])
        kline_prefix_ma_rapide = get_kline_prefix_effective(freq_ma_rapide_param)
        source_col_close_ma_rapide = f"{kline_prefix_ma_rapide}_close" if kline_prefix_ma_rapide else "close"

        freq_ma_lente_param = str(self.params['indicateur_frequence_ma_lente'])
        kline_prefix_ma_lente = get_kline_prefix_effective(freq_ma_lente_param)
        source_col_close_ma_lente = f"{kline_prefix_ma_lente}_close" if kline_prefix_ma_lente else "close"

        freq_atr_param = str(self.params['atr_base_frequency_sl_tp'])
        kline_prefix_atr = get_kline_prefix_effective(freq_atr_param)
        source_col_high_atr = f"{kline_prefix_atr}_high" if kline_prefix_atr else "high"
        source_col_low_atr = f"{kline_prefix_atr}_low" if kline_prefix_atr else "low"
        source_col_close_atr = f"{kline_prefix_atr}_close" if kline_prefix_atr else "close"

        configs = [
            {
                'indicator_name': str(self.params['ma_type']).lower(),
                'params': {'length': int(self.params['fast_ma_period'])},
                'inputs': {'close': source_col_close_ma_rapide},
                'outputs': self.fast_ma_col_strat
            },
            {
                'indicator_name': str(self.params['ma_type']).lower(),
                'params': {'length': int(self.params['slow_ma_period'])},
                'inputs': {'close': source_col_close_ma_lente},
                'outputs': self.slow_ma_col_strat
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
        self._log(f"Configurations d'indicateurs requises: {configs}", level=3)
        return configs

    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame: # Implémentation de la méthode abstraite
        df = data_feed.copy() # Travailler sur une copie pour éviter de modifier l'original
        expected_strat_cols = [self.fast_ma_col_strat, self.slow_ma_col_strat, self.atr_col_strat]

        base_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if missing_ohlcv := [col for col in base_ohlcv if col not in df.columns]:
            msg = f"Colonnes OHLCV de base manquantes dans data_feed: {missing_ohlcv}."
            self._log(msg, level=1, is_error=True)
            raise ValueError(msg)

        # Vérifier si les colonnes d'indicateurs (définies par self.xxx_col_strat)
        # sont déjà présentes (calculées par IndicatorCalculator).
        # Si elles manquent, c'est une erreur de configuration ou de flux de données.
        # La stratégie elle-même ne devrait pas recalculer les indicateurs ici si
        # IndicatorCalculator est censé le faire en amont.
        # Cette méthode est plus pour des transformations finales ou des indicateurs composites.
        missing_strat_cols = [col for col in expected_strat_cols if col not in df.columns]
        if missing_strat_cols:
            # Loguer un avertissement sévère ou lever une erreur.
            # Pour la robustesse, on peut ajouter des colonnes NaN, mais cela cache un problème.
            msg = f"Colonnes indicateur attendues manquantes après IndicatorCalculator: {missing_strat_cols}. " \
                  f"Vérifiez get_required_indicator_configs et IndicatorCalculator."
            self._log(msg, level=1, is_error=True)
            # Optionnel: Ajouter les colonnes manquantes avec NaN pour éviter des KeyError plus loin,
            # mais la logique de _generate_signals devra gérer ces NaNs.
            for col_name in missing_strat_cols:
                df[col_name] = np.nan
            # raise ValueError(msg) # Ou lever une erreur pour arrêter

        # Si des post-traitements étaient nécessaires sur les indicateurs, ils iraient ici.
        # Ex: Normalisation, combinaisons simples non gérées par pandas_ta directement.
        # Pour MaCrossover, les indicateurs de base sont généralement suffisants.

        # Assurer le ffill des indicateurs pour éviter les NaNs en cours de route si la source en avait
        cols_to_ffill_present = [col for col in expected_strat_cols if col in df.columns]
        if cols_to_ffill_present:
            if df[cols_to_ffill_present].isnull().values.any():
                 self._log(f"Application de ffill aux colonnes indicateur: {cols_to_ffill_present}", level=3)
            df[cols_to_ffill_present] = df[cols_to_ffill_present].ffill()
        
        self._log(f"Indicateurs prêts pour _generate_signals. Colonnes: {df.columns.tolist()}", level=3)
        return df

    def _generate_signals(self,
                          data_with_indicators: pd.DataFrame,
                          current_position_open: bool,
                          current_position_direction: int,
                          current_entry_price: float
                         ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]: # Implémentation
        signal_type: int = 0
        limit_price: Optional[float] = None
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        if len(data_with_indicators) < 2:
            self._log("Pas assez de données (< 2 barres) pour générer des signaux.", level=2, is_warning=True)
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        latest_row = data_with_indicators.iloc[-1]
        previous_row = data_with_indicators.iloc[-2]

        # Accès sécurisé aux colonnes d'indicateurs
        ma_fast_curr = latest_row.get(self.fast_ma_col_strat)
        ma_slow_curr = latest_row.get(self.slow_ma_col_strat)
        ma_fast_prev = previous_row.get(self.fast_ma_col_strat)
        ma_slow_prev = previous_row.get(self.slow_ma_col_strat)
        close_price_curr = latest_row.get('close')
        atr_value_curr = latest_row.get(self.atr_col_strat)

        essential_values = [ma_fast_curr, ma_slow_curr, ma_fast_prev, ma_slow_prev, close_price_curr, atr_value_curr]
        if any(pd.isna(val) for val in essential_values):
            nan_details = {
                "ma_fast_c": ma_fast_curr, "ma_slow_c": ma_slow_curr,
                "ma_fast_p": ma_fast_prev, "ma_slow_p": ma_slow_prev,
                "close_c": close_price_curr, "atr_c": atr_value_curr
            }
            self._log(f"Valeurs d'indicateur/prix manquantes (NaN) à {latest_row.name}. Détails: {nan_details}. Signal HOLD.", level=2, is_warning=True)
            return 0, self.get_param("order_type_preference"), None, None, None, self.get_param('capital_allocation_pct')

        # Logique de croisement
        bullish_crossover = (ma_fast_curr > ma_slow_curr) and (ma_fast_prev <= ma_slow_prev)
        bearish_crossover = (ma_fast_curr < ma_slow_curr) and (ma_fast_prev >= ma_slow_prev)

        sl_multiplier = float(self.get_param('sl_atr_multiplier'))
        tp_multiplier = float(self.get_param('tp_atr_multiplier'))

        if current_position_open:
            if current_position_direction == 1 and bearish_crossover: # Position LONG, croisement baissier -> EXIT
                signal_type = 2 # EXIT
                self._log(f"Signal EXIT LONG (croisement baissier) @ {close_price_curr:.{self.price_precision or 4}f}", level=1)
            # Pas de gestion de sortie de position SHORT ici car cette stratégie n'entre pas en SHORT
        else: # Pas de position ouverte
            if bullish_crossover:
                signal_type = 1 # BUY
                if atr_value_curr > 0:
                    sl_price = close_price_curr - (atr_value_curr * sl_multiplier)
                    tp_price = close_price_curr + (atr_value_curr * tp_multiplier)
                self._log(f"Signal BUY @ {close_price_curr:.{self.price_precision or 4}f}. SL={sl_price}, TP={tp_price or 'N/A'}", level=1)
            elif bearish_crossover:
                # Cette stratégie de base n'initie pas de positions SHORT sur un croisement baissier.
                # Elle ne fait que sortir des positions LONG.
                self._log("Croisement baissier détecté, mais pas de position LONG ouverte et shorting non implémenté pour entrée. Signal HOLD.", level=2)
                signal_type = 0 # HOLD

        order_type_preference = str(self.get_param("order_type_preference", "MARKET"))
        if signal_type != 0 and order_type_preference == "LIMIT":
            limit_price = float(close_price_curr)

        position_size_pct = float(self.get_param('capital_allocation_pct', 1.0))

        return signal_type, order_type_preference, limit_price, sl_price, tp_price, position_size_pct

    def generate_order_request(self,
                               data: pd.DataFrame,
                               current_position: int,
                               available_capital: float,
                               symbol_info: Dict[str, Any] # C'est le pair_config
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]: # Implémentation
        """
        Génère une requête d'ordre d'ENTRÉE pour le trading en direct ou une simulation détaillée.
        """
        self._log(f"[Live] Appel de generate_order_request. Position actuelle: {current_position}, Capital: {available_capital:.2f}", level=2)

        if current_position != 0: # Si déjà en position (long ou short)
            self._log(f"[Live] Position déjà ouverte ({current_position}). Pas de nouvelle requête d'ordre d'entrée.", level=2)
            return None # Ne génère pas de nouvel ordre d'entrée si déjà en position

        # 1. S'assurer que le contexte de trading est défini pour la stratégie
        #    Normalement, le LiveTradingManager appelle set_trading_context avant.
        if not self.trading_context:
            self._log("[Live] TradingContext non défini dans la stratégie. Impossible de générer un ordre.", level=1, is_error=True)
            # Essayer de le configurer avec les infos fournies si possible (fallback)
            # Ceci est une rustine, le contexte devrait être setté en amont.
            from src.strategies.base import TradingContext # Import local
            temp_context = TradingContext(
                pair_config=symbol_info,
                is_futures=False, # Supposition, à adapter
                leverage=int(self.get_param('margin_leverage', 1)),
                initial_equity=available_capital, # Utiliser le capital dispo comme équité "initiale" pour ce check
                account_type="MARGIN" # Supposition
            )
            val_res = self.set_trading_context(temp_context)
            if not val_res.is_valid:
                self._log(f"[Live] Échec de la configuration du contexte de fallback: {val_res.messages}", level=1, is_error=True)
                return None

        # 2. Calculer les indicateurs sur les données fournies
        try:
            data_with_indicators = self._calculate_indicators(data.copy()) # Utilise les données live les plus récentes
        except Exception as e_calc_live:
            self._log(f"[Live] Erreur _calculate_indicators: {e_calc_live}", level=1, is_error=True)
            return None
        
        if data_with_indicators.empty or len(data_with_indicators) < 2:
            self._log("[Live] Données insuffisantes après _calculate_indicators pour generate_order_request.", level=2, is_warning=True)
            return None

        # 3. Générer le signal basé sur les indicateurs
        # Pour un ordre d'entrée, current_position_open = False, current_position_direction = 0
        try:
            signal, order_type, limit_price_sugg, sl_price_raw, tp_price_raw, pos_size_pct = \
                self._generate_signals(data_with_indicators, False, 0, 0.0)
        except Exception as e_sig_live:
            self._log(f"[Live] Erreur _generate_signals: {e_sig_live}", level=1, is_error=True)
            return None

        # 4. Si signal d'entrée (1 pour BUY, -1 pour SELL/SHORT)
        if signal not in [1, -1]:
            self._log(f"[Live] Aucun signal d'entrée (1 ou -1) généré. Signal actuel: {signal}", level=2)
            return None # Pas d'ordre d'entrée

        # Cette stratégie de base gère seulement les entrées LONG (signal == 1)
        if signal == -1: # Si un signal SHORT était généré (ex: par une sous-classe)
            self._log("[Live] Signal SHORT (-1) généré, mais MaCrossoverStrategy de base n'initie pas de shorts. Pas d'ordre.", level=2)
            return None

        # 5. Déterminer le prix d'entrée théorique pour le calcul de la quantité
        latest_bar = data_with_indicators.iloc[-1]
        entry_price_theoretical: float
        if order_type == "LIMIT" and limit_price_sugg is not None:
            entry_price_theoretical = limit_price_sugg
        else: # Pour MARKET ou si limit_price_sugg est None
            entry_price_theoretical = float(latest_bar.get('close', 0.0)) # Utiliser le close actuel comme estimation
            if entry_price_theoretical <= 0: # Fallback sur 'open' si 'close' est invalide
                entry_price_theoretical = float(latest_bar.get('open', 0.0))
        
        if pd.isna(entry_price_theoretical) or entry_price_theoretical <= 0:
            self._log(f"[Live] Prix d'entrée théorique pour calcul de quantité invalide ({entry_price_theoretical}).", level=1, is_error=True)
            return None

        # 6. Calculer la quantité
        # Les précisions (self.quantity_precision, self.price_precision) et self.pair_config
        # sont supposées être settées par set_trading_context.
        if self.quantity_precision is None or self.pair_config is None:
            self._log("[Live] Contexte de précision ou pair_config non défini. Impossible de calculer la quantité.", level=1, is_error=True)
            return None

        quantity_base = self._calculate_quantity(
            entry_price=entry_price_theoretical,
            available_capital=available_capital,
            qty_precision=self.quantity_precision,
            symbol_info=self.pair_config, # C'est le symbol_info de l'exchange
            symbol=self.symbol,
            position_size_pct=pos_size_pct
        )

        if quantity_base is None or quantity_base <= 1e-9: # Seuil pour quantité non nulle
            self._log(f"[Live] Quantité calculée nulle ou invalide ({quantity_base}). Pas d'ordre.", level=2, is_warning=True)
            return None

        # 7. Formater les paramètres de l'ordre
        if self.price_precision is None:
            self._log("[Live] price_precision non défini. Impossible de formater le prix limite.", level=1, is_error=True)
            return None

        entry_price_for_order_str: Optional[str] = None
        if order_type == "LIMIT" and limit_price_sugg is not None:
            # Le prix limite suggéré par la stratégie doit être ajusté au tickSize de l'exchange
            tick_size_price = get_filter_value(self.pair_config, 'PRICE_FILTER', 'tickSize')
            adjusted_limit_price = adjust_precision(limit_price_sugg, self.price_precision, tick_size=tick_size_price)
            if adjusted_limit_price is None:
                self._log(f"[Live] Échec de l'ajustement du prix limite suggéré {limit_price_sugg}.", level=1, is_error=True)
                return None
            entry_price_for_order_str = f"{adjusted_limit_price:.{self.price_precision}f}"

        quantity_str_formatted = f"{quantity_base:.{self.quantity_precision}f}"

        # Construire les paramètres de l'ordre d'entrée
        # Utiliser les méthodes helper de BaseStrategy
        entry_order_params = self._build_entry_params_formatted(
            side="BUY", # Pour MaCrossover, on n'entre que LONG
            quantity_str=quantity_str_formatted,
            order_type=str(order_type),
            entry_price_str=entry_price_for_order_str
            # time_in_force peut être ajouté ici si nécessaire
        )
        if not entry_order_params: # Si la construction échoue (ex: prix manquant pour LIMIT)
            self._log("[Live] Échec de la construction des paramètres de l'ordre d'entrée.", level=1, is_error=True)
            return None

        # Préparer les prix SL/TP bruts (non ajustés) retournés par la stratégie
        sl_tp_raw_prices_dict: Dict[str, float] = {}
        if sl_price_raw is not None: sl_tp_raw_prices_dict['sl_price'] = float(sl_price_raw)
        if tp_price_raw is not None: sl_tp_raw_prices_dict['tp_price'] = float(tp_price_raw)

        self._log(f"[Live] Requête d'ordre d'entrée générée: {entry_order_params}. SL/TP bruts suggérés: {sl_tp_raw_prices_dict}", level=1)
        return entry_order_params, sl_tp_raw_prices_dict

