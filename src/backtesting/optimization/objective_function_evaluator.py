# src/backtesting/optimization/objective_function_evaluator.py
"""
Ce module définit ObjectiveFunctionEvaluator, la fonction objectif pour Optuna.
Elle évalue un ensemble d'hyperparamètres (un "trial") en exécutant un backtest
et en retournant les métriques de performance qui servent d'objectifs pour
l'optimisation. Elle est utilisée pour l'optimisation In-Sample (IS) et
la validation Out-of-Sample (OOS).
"""
import logging
import time
import importlib
import uuid # Pour les logs OOS détaillés si besoin
from typing import Any, Dict, Optional, Tuple, List, Type, Union, TYPE_CHECKING
from datetime import timezone # Pour s'assurer que les timestamps sont UTC

import numpy as np
import pandas as pd
import pandas_ta as ta # type: ignore
import optuna

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import ParamDetail, SimulationDefaults
    from src.strategies.base import BaseStrategy
    # from src.backtesting.core_simulator import BacktestRunner # Sera défini plus tard

# Imports depuis l'application
try:
    from src.data import data_utils # Pour get_kline_prefix_effective
    from src.config.definitions import ParamDetail, SimulationDefaults # Assurer que ParamDetail est importé
    # Le BacktestRunner sera l'évolution du BacktestSimulator
    # from src.backtesting.core_simulator import BacktestRunner # Sera défini plus tard
    from src.backtesting.simulator import BacktestSimulator # Utilisation temporaire de l'ancien simulateur
    from src.backtesting.performance_analyzer import calculate_performance_metrics_from_inputs
    from src.strategies.base import BaseStrategy # Pour le typage et l'instanciation
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(
        f"ObjectiveFunctionEvaluator: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH."
    )
    raise

logger = logging.getLogger(__name__)

# --- Placeholder pour BacktestRunner ---
# Remplacer par l'importation réelle une fois src.backtesting.core_simulator.BacktestRunner défini
try:
    from src.backtesting.core_simulator import BacktestRunner
except ImportError:
    logger.warning(
        "ObjectiveFunctionEvaluator: src.backtesting.core_simulator.BacktestRunner non trouvé. "
        "Utilisation de src.backtesting.simulator.BacktestSimulator comme placeholder."
    )
    BacktestRunner = BacktestSimulator # Utiliser l'ancien comme placeholder
# --- Fin Placeholder ---


class ObjectiveFunctionEvaluator:
    """
    Fonction objectif pour Optuna. Évalue un ensemble d'hyperparamètres
    en exécutant un backtest et retourne les métriques de performance.
    """
    def __init__(self,
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any],
                 df_enriched_slice: pd.DataFrame,
                 optuna_objectives_config: Dict[str, Any],
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any], # C'est le pair_config
                 app_config: 'AppConfig',
                 run_id: str, # ID du run WFO global
                 is_oos_eval: bool = False,
                 is_trial_number_for_oos_log: Optional[int] = None):
        """
        Initialise ObjectiveFunctionEvaluator.

        Args:
            strategy_name (str): Nom clé de la stratégie.
            strategy_config_dict (Dict[str, Any]): Dictionnaire de configuration pour la stratégie.
            df_enriched_slice (pd.DataFrame): Données enrichies (IS ou OOS) pour cette évaluation.
            optuna_objectives_config (Dict[str, Any]): Config des objectifs Optuna.
            pair_symbol (str): Symbole de la paire.
            symbol_info_data (Dict[str, Any]): Informations de l'exchange pour la paire.
            app_config (AppConfig): Configuration globale de l'application.
            run_id (str): ID du run WFO global.
            is_oos_eval (bool): True si évaluation OOS.
            is_trial_number_for_oos_log (Optional[int]): Numéro du trial IS pour log OOS.
        """
        self.strategy_name_key = strategy_name
        self.strategy_config_dict = strategy_config_dict
        self.df_enriched_slice = df_enriched_slice.copy() # Travailler sur une copie
        self.optuna_objectives_config = optuna_objectives_config
        self.pair_symbol = pair_symbol.upper()
        self.symbol_info_data = symbol_info_data # C'est le pair_config
        self.app_config = app_config
        self.run_id = run_id
        self.is_oos_eval = is_oos_eval
        self.is_trial_number_for_oos_log = is_trial_number_for_oos_log

        self.log_prefix = (
            f"[{self.strategy_name_key}/{self.pair_symbol}]"
            f"[Run:{self.run_id}]"
            f"[{'OOS' if self.is_oos_eval else 'IS'}]"
            f"{f'[IS_Trial:{self.is_trial_number_for_oos_log}]' if self.is_oos_eval and self.is_trial_number_for_oos_log is not None else ''}"
        )

        self.script_reference: str = self.strategy_config_dict.get('script_reference', '')
        self.class_name: str = self.strategy_config_dict.get('class_name', '')
        if not self.script_reference or not self.class_name:
            logger.error(f"{self.log_prefix} 'script_reference' ou 'class_name' manquant dans strategy_config_dict pour {self.strategy_name_key}.")
            raise ValueError("Référence de script ou nom de classe manquant pour la stratégie.")

        # Extraire params_space (ParamDetail)
        self.params_space_details: Dict[str, ParamDetail] = {}
        raw_params_space = self.strategy_config_dict.get('params_space', {})
        if isinstance(raw_params_space, dict):
            for param_key, param_value_dict in raw_params_space.items():
                if isinstance(param_value_dict, dict):
                    try:
                        self.params_space_details[param_key] = ParamDetail(**param_value_dict)
                    except Exception as e_pd:
                        logger.error(f"{self.log_prefix} Erreur lors de la création de ParamDetail pour '{param_key}' : {e_pd}")
                        # Gérer l'erreur, peut-être lever une exception ou ignorer ce paramètre
                else:
                    logger.warning(f"{self.log_prefix} Valeur inattendue pour '{param_key}' dans params_space. Attendu un dict, reçu {type(param_value_dict)}.")
        else:
            logger.error(f"{self.log_prefix} 'params_space' pour la stratégie '{self.strategy_name_key}' n'est pas un dictionnaire.")
        
        if not self.params_space_details and not self.is_oos_eval:
             logger.warning(f"{self.log_prefix} params_space_details est VIDE pour l'évaluation IS. "
                            "L'optimisation pourrait utiliser des paramètres par défaut ou échouer si aucun paramètre n'est suggéré.")


        # Validation et préparation de df_enriched_slice
        if not isinstance(self.df_enriched_slice.index, pd.DatetimeIndex):
            if 'timestamp' in self.df_enriched_slice.columns:
                self.df_enriched_slice['timestamp'] = pd.to_datetime(self.df_enriched_slice['timestamp'], errors='coerce', utc=True)
                self.df_enriched_slice.dropna(subset=['timestamp'], inplace=True)
                self.df_enriched_slice = self.df_enriched_slice.set_index('timestamp')
            else:
                msg = "df_enriched_slice doit avoir un DatetimeIndex ou une colonne 'timestamp'."
                logger.error(f"{self.log_prefix} {msg}")
                raise ValueError(msg)

        if self.df_enriched_slice.index.tz is None:
            logger.warning(f"{self.log_prefix} L'index de df_enriched_slice n'a pas de timezone. Localisation en UTC.")
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_localize('UTC')
        elif self.df_enriched_slice.index.tz.utcoffset(self.df_enriched_slice.index[0]) != timezone.utc.utcoffset(self.df_enriched_slice.index[0]): # type: ignore
            logger.warning(f"{self.log_prefix} L'index de df_enriched_slice n'est pas en UTC. Conversion en UTC.")
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_convert('UTC') # type: ignore

        if not self.df_enriched_slice.index.is_monotonic_increasing:
            self.df_enriched_slice.sort_index(inplace=True)
        if not self.df_enriched_slice.index.is_unique:
            logger.warning(f"{self.log_prefix} Timestamps dupliqués dans df_enriched_slice. Conservation de la première occurrence.")
            self.df_enriched_slice = self.df_enriched_slice[~self.df_enriched_slice.index.duplicated(keep='first')]

        self.last_backtest_results: Optional[Dict[str, Any]] = None
        logger.info(f"{self.log_prefix} ObjectiveFunctionEvaluator initialisé. Données IS/OOS shape: {self.df_enriched_slice.shape}")

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggère des valeurs pour chaque hyperparamètre défini dans params_space.
        Appelé uniquement si self.is_oos_eval est False.
        """
        params_for_trial: Dict[str, Any] = {}
        log_prefix_suggest = f"{self.log_prefix}[Trial:{trial.number}][SuggestParams]"

        if not self.params_space_details:
            default_params = self.strategy_config_dict.get('default_params', {})
            if default_params and isinstance(default_params, dict):
                logger.warning(f"{log_prefix_suggest} params_space est vide. Utilisation des default_params : {default_params}")
                return default_params.copy()
            logger.error(f"{log_prefix_suggest} params_space est vide et aucun default_params n'est disponible. Élagueage du trial.")
            raise optuna.exceptions.TrialPruned("params_space vide et pas de default_params.")

        for param_name, p_detail in self.params_space_details.items():
            try:
                if p_detail.type == 'int':
                    # S'assurer que low et high sont des entiers pour suggest_int
                    low = int(p_detail.low) if p_detail.low is not None else 0
                    high = int(p_detail.high) if p_detail.high is not None else low + 1
                    step = int(p_detail.step or 1)
                    params_for_trial[param_name] = trial.suggest_int(param_name, low, high, step=step)
                elif p_detail.type == 'float':
                    low_f = float(p_detail.low) if p_detail.low is not None else 0.0
                    high_f = float(p_detail.high) if p_detail.high is not None else low_f + 1.0
                    step_f = float(p_detail.step) if p_detail.step is not None else None
                    params_for_trial[param_name] = trial.suggest_float(param_name, low_f, high_f, step=step_f, log=p_detail.log_scale)
                elif p_detail.type == 'categorical' and p_detail.choices:
                    params_for_trial[param_name] = trial.suggest_categorical(param_name, p_detail.choices)
                else:
                    logger.warning(f"{log_prefix_suggest} Type de ParamDetail non supporté '{p_detail.type}' ou choices manquant pour '{param_name}'. Utilisation de la valeur par défaut ou de 'low'.")
                    params_for_trial[param_name] = p_detail.default if p_detail.default is not None else p_detail.low
            except Exception as e_sug:
                logger.error(f"{log_prefix_suggest} Erreur lors de la suggestion du paramètre '{param_name}': {e_sug}", exc_info=True)
                # Fallback à la valeur par défaut si la suggestion échoue
                if p_detail.default is not None:
                    params_for_trial[param_name] = p_detail.default
                elif p_detail.type == 'categorical' and p_detail.choices:
                     params_for_trial[param_name] = p_detail.choices[0]
                else: # Si pas de défaut, cela pourrait être un problème
                    logger.critical(f"{log_prefix_suggest} Impossible de suggérer ou de trouver un fallback pour le paramètre '{param_name}'. Élagueage.")
                    raise optuna.exceptions.TrialPruned(f"Échec de la suggestion pour {param_name}")
        
        logger.debug(f"{log_prefix_suggest} Paramètres suggérés : {params_for_trial}")
        return params_for_trial

    def _calculate_indicator_on_selected_klines(self,
                                                df_source_enriched: pd.DataFrame, # df_enriched_slice
                                                indicator_type: str,
                                                indicator_params: Dict[str, Any],
                                                kline_ohlc_prefix: str # Ex: "Klines_5min" ou "" pour 1min
                                                ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        Calcule un indicateur technique en utilisant pandas-ta sur les colonnes OHLCV spécifiées
        par kline_ohlc_prefix à partir de df_source_enriched.
        """
        # Cette fonction est une adaptation de celle de l'ancien ObjectiveEvaluator.
        # Elle sera utilisée par _prepare_data_with_dynamic_indicators.
        log_prefix_calc = f"{self.log_prefix}[CalcIndicator:{indicator_type}:{kline_ohlc_prefix}]"

        if df_source_enriched.empty:
            logger.warning(f"{log_prefix_calc} df_source_enriched est vide.")
            return None

        open_col = f"{kline_ohlc_prefix}_open" if kline_ohlc_prefix else "open"
        high_col = f"{kline_ohlc_prefix}_high" if kline_ohlc_prefix else "high"
        low_col = f"{kline_ohlc_prefix}_low" if kline_ohlc_prefix else "low"
        close_col = f"{kline_ohlc_prefix}_close" if kline_ohlc_prefix else "close"
        volume_col = f"{kline_ohlc_prefix}_volume" if kline_ohlc_prefix else "volume"

        # Déterminer les colonnes sources nécessaires pour l'indicateur
        required_ta_cols_map: Dict[str, str] = {'close': close_col} # 'close' est presque toujours nécessaire
        indicator_type_lower = indicator_type.lower()

        # Logique pour déterminer les colonnes H, L, O, V basées sur l'indicateur (simplifié)
        # Une approche plus robuste utiliserait inspect ou une table de mapping pour les besoins de chaque indicateur TA-Lib/pandas-ta.
        if indicator_type_lower in ['atr', 'bbands', 'donchian', 'kc', 'ichimoku', 'supertrend', 'psar', 'stoch', 'rsi', 'cci', 'roc', 'mom', 'adx']: # Liste non exhaustive
            required_ta_cols_map['high'] = high_col
            required_ta_cols_map['low'] = low_col
        if indicator_type_lower in ['ichimoku', 'heikinashi']: # HeikinAshi via ta.ha()
            required_ta_cols_map['open'] = open_col
        if indicator_type_lower in ['obv', 'vwap', 'ad', 'adosc', 'cmf', 'efi', 'mfi', 'adx']: # ADX a besoin de HLC, mais pandas-ta peut aussi utiliser volume pour certains calculs internes ou liés
            required_ta_cols_map['volume'] = volume_col


        ta_inputs: Dict[str, pd.Series] = {}
        all_required_present_and_valid = True
        for ta_input_name, source_col_name in required_ta_cols_map.items():
            if source_col_name not in df_source_enriched.columns:
                logger.warning(f"{log_prefix_calc} Colonne source '{source_col_name}' pour l'entrée TA '{ta_input_name}' non trouvée.")
                if ta_input_name == 'close': all_required_present_and_valid = False; break # 'close' est critique
                continue # Pour les autres, on essaie sans
            
            series_data = df_source_enriched[source_col_name]
            if series_data.isnull().all():
                logger.warning(f"{log_prefix_calc} Colonne source '{source_col_name}' (pour '{ta_input_name}') est entièrement NaN.")
                if ta_input_name == 'close': all_required_present_and_valid = False; break
                continue
            ta_inputs[ta_input_name] = series_data.astype(float) # pandas-ta attend souvent des float

        if not all_required_present_and_valid:
            logger.error(f"{log_prefix_calc} Données sources critiques (ex: close) manquantes ou invalides. Calcul de '{indicator_type}' impossible.")
            return None
        
        if not ta_inputs.get('close', pd.Series(dtype=float)).notna().any(): # type: ignore
            logger.error(f"{log_prefix_calc} La série 'close' pour le calcul de '{indicator_type}' est vide ou entièrement NaN après préparation.")
            return None


        try:
            # Trouver la fonction indicateur dans pandas_ta
            # pandas_ta a une structure plate pour la plupart des indicateurs maintenant.
            indicator_function = getattr(ta, indicator_type_lower, None)
            if indicator_function is None:
                 # Essayer de chercher dans les sous-modules si certains sont encore structurés ainsi (moins courant)
                for category in [ta.trend, ta.momentum, ta.overlap, ta.volume, ta.volatility, ta.cycles, ta.statistics, ta.transform, ta.utils]:
                    if hasattr(category, indicator_type_lower):
                        indicator_function = getattr(category, indicator_type_lower)
                        break
            
            if indicator_function is None:
                logger.error(f"{log_prefix_calc} Fonction indicateur '{indicator_type_lower}' non trouvée dans pandas_ta.")
                return None
            
            logger.debug(f"{log_prefix_calc} Calcul de {indicator_type_lower} avec params: {indicator_params}. Entrées TA fournies: {list(ta_inputs.keys())}")
            # L'option `append=False` est importante pour obtenir seulement l'indicateur
            result = indicator_function(**ta_inputs, **indicator_params, append=False) # type: ignore
            
            if isinstance(result, (pd.DataFrame, pd.Series)):
                logger.debug(f"{log_prefix_calc} Indicateur '{indicator_type_lower}' calculé. Type résultat: {type(result)}")
                return result
            else:
                logger.warning(f"{log_prefix_calc} Indicateur '{indicator_type_lower}' n'a pas retourné une Series/DataFrame. Reçu: {type(result)}")
                return None
        except Exception as e:
            logger.error(f"{log_prefix_calc} Erreur lors du calcul de {indicator_type} avec params {indicator_params}: {e}", exc_info=True)
            return None


    def _prepare_data_with_dynamic_indicators(self,
                                              trial_params: Dict[str, Any],
                                              trial_number_for_log: Optional[Union[int, str]] = None
                                              ) -> pd.DataFrame:
        """
        Prépare le DataFrame pour la simulation en calculant/sélectionnant dynamiquement
        les indicateurs requis par la stratégie en fonction des trial_params.
        """
        # Le log_prefix de la classe contient déjà le numéro de trial IS si en mode OOS.
        # Si en mode IS, trial_number_for_log sera le numéro du trial Optuna.
        current_eval_id = trial_number_for_log if trial_number_for_log is not None else "N/A"
        log_prefix_prep = f"{self.log_prefix}[EvalID:{current_eval_id}][PrepData]"

        logger.info(f"{log_prefix_prep} Préparation des données avec indicateurs dynamiques.")
        logger.debug(f"{log_prefix_prep} Paramètres du trial : {trial_params}")

        # Commencer avec les colonnes OHLCV de base (1-minute) du slice enrichi
        # S'assurer que ces colonnes existent dans self.df_enriched_slice
        required_base_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.df_enriched_slice.columns for col in required_base_cols):
            msg = f"Colonnes OHLCV de base manquantes dans df_enriched_slice: {self.df_enriched_slice.columns.tolist()}"
            logger.error(f"{log_prefix_prep} {msg}")
            raise ValueError(msg)
            
        df_for_simulation = self.df_enriched_slice[required_base_cols].copy()

        # 1. Calcul de l'ATR_strat (pour SL/TP, souvent basé sur une fréquence spécifique)
        atr_period_key = 'atr_period_sl_tp' # Nom standard attendu
        atr_freq_key = 'atr_base_frequency_sl_tp' # Nom standard attendu
        
        atr_period_val = trial_params.get(atr_period_key)
        atr_freq_raw_val = trial_params.get(atr_freq_key)

        if atr_period_val is not None and atr_freq_raw_val is not None:
            atr_period_int = int(atr_period_val)
            kline_prefix_atr_source = data_utils.get_kline_prefix_effective(str(atr_freq_raw_val))
            
            # Nom de la colonne ATR pré-calculée dans df_enriched_slice (si elle existe)
            precalc_atr_col_name = f"{kline_prefix_atr_source}_ATR_{atr_period_int}" if kline_prefix_atr_source else f"ATR_{atr_period_int}"

            if precalc_atr_col_name in self.df_enriched_slice.columns:
                df_for_simulation['ATR_strat'] = self.df_enriched_slice[precalc_atr_col_name].reindex(df_for_simulation.index, method='ffill')
                logger.info(f"{log_prefix_prep} ATR_strat chargé depuis la colonne pré-calculée '{precalc_atr_col_name}'.")
            else: # Calculer dynamiquement si non pré-calculé
                logger.info(f"{log_prefix_prep} Colonne ATR pré-calculée '{precalc_atr_col_name}' non trouvée. Calcul dynamique de ATR_strat (source: {kline_prefix_atr_source or '1min'}).")
                atr_series_calc = self._calculate_indicator_on_selected_klines(
                    self.df_enriched_slice, 'atr', {'length': atr_period_int}, kline_prefix_atr_source
                )
                if isinstance(atr_series_calc, pd.Series):
                    df_for_simulation['ATR_strat'] = atr_series_calc.reindex(df_for_simulation.index, method='ffill')
                    if df_for_simulation['ATR_strat'].notna().any():
                         logger.info(f"{log_prefix_prep} ATR_strat calculé dynamiquement.")
                    else:
                         logger.warning(f"{log_prefix_prep} Calcul dynamique de ATR_strat a résulté en une série de NaN.")
                         df_for_simulation['ATR_strat'] = np.nan # Assurer NaN si tout est NaN
                else:
                    logger.warning(f"{log_prefix_prep} Échec du calcul dynamique de ATR_strat. Sera NaN.")
                    df_for_simulation['ATR_strat'] = np.nan
        else:
            logger.warning(f"{log_prefix_prep} Paramètres '{atr_period_key}' ou '{atr_freq_key}' manquants dans trial_params. ATR_strat sera NaN.")
            df_for_simulation['ATR_strat'] = np.nan

        # 2. Logique spécifique à chaque stratégie pour calculer/sélectionner ses indicateurs
        # Cette section doit être adaptée pour chaque stratégie que l'évaluateur supporte.
        # Elle utilise les `trial_params` pour déterminer les périodes, fréquences sources, etc.
        # Les indicateurs finaux doivent être nommés avec le suffixe `_strat`.

        # Exemple pour MaCrossoverStrategy (à adapter avec les noms de colonnes exacts de df_enriched_slice)
        if self.class_name == "MaCrossoverStrategy":
            params_needed = ['fast_ma_period', 'slow_ma_period', 'ma_type', 
                             'indicateur_frequence_ma_rapide', 'indicateur_frequence_ma_lente']
            if not all(p in trial_params for p in params_needed):
                logger.error(f"{log_prefix_prep} Paramètres manquants pour MaCrossoverStrategy. Requis: {params_needed}")
            else:
                # MA Rapide
                kline_prefix_fast = data_utils.get_kline_prefix_effective(str(trial_params['indicateur_frequence_ma_rapide']))
                ma_fast_series = self._calculate_indicator_on_selected_klines(
                    self.df_enriched_slice, str(trial_params['ma_type']).lower(), 
                    {'length': int(trial_params['fast_ma_period'])}, kline_prefix_fast
                )
                df_for_simulation['MA_FAST_strat'] = ma_fast_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ma_fast_series, pd.Series) else np.nan

                # MA Lente
                kline_prefix_slow = data_utils.get_kline_prefix_effective(str(trial_params['indicateur_frequence_ma_lente']))
                ma_slow_series = self._calculate_indicator_on_selected_klines(
                    self.df_enriched_slice, str(trial_params['ma_type']).lower(),
                    {'length': int(trial_params['slow_ma_period'])}, kline_prefix_slow
                )
                df_for_simulation['MA_SLOW_strat'] = ma_slow_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ma_slow_series, pd.Series) else np.nan
                logger.info(f"{log_prefix_prep} Indicateurs MA_FAST_strat et MA_SLOW_strat préparés pour MaCrossoverStrategy.")

        elif self.class_name == "PsarReversalOtocoStrategy":
            params_needed = ['psar_step', 'psar_max_step', 'indicateur_frequence_psar']
            if not all(p in trial_params for p in params_needed):
                 logger.error(f"{log_prefix_prep} Paramètres manquants pour PsarReversalOtocoStrategy. Requis: {params_needed}")
            else:
                kline_prefix_psar = data_utils.get_kline_prefix_effective(str(trial_params['indicateur_frequence_psar']))
                psar_params_ta = {'af': float(trial_params['psar_step']), 'max_af': float(trial_params['psar_max_step'])}
                psar_df_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'psar', psar_params_ta, kline_prefix_psar)
                
                if psar_df_result is not None and isinstance(psar_df_result, pd.DataFrame) and not psar_df_result.empty:
                    # Les noms de colonnes de pandas-ta.psar peuvent varier (ex: PSARl_af_max_af, PSARs_af_max_af)
                    # On cherche des noms contenant 'psarl' et 'psars'
                    long_col = next((col for col in psar_df_result.columns if 'psarl' in col.lower()), None)
                    short_col = next((col for col in psar_df_result.columns if 'psars' in col.lower()), None)
                    
                    df_for_simulation['PSARl_strat'] = psar_df_result[long_col].reindex(df_for_simulation.index, method='ffill') if long_col else np.nan
                    df_for_simulation['PSARs_strat'] = psar_df_result[short_col].reindex(df_for_simulation.index, method='ffill') if short_col else np.nan
                else:
                    df_for_simulation['PSARl_strat'] = np.nan
                    df_for_simulation['PSARs_strat'] = np.nan
                logger.info(f"{log_prefix_prep} Indicateurs PSARl_strat et PSARs_strat préparés pour PsarReversalOtocoStrategy.")

        # Ajouter d'autres logiques pour d'autres stratégies ici...
        # elif self.class_name == "AutreStrategie":
        #    ...
        
        else:
            logger.warning(f"{log_prefix_prep} Logique de préparation d'indicateurs non implémentée pour la stratégie : {self.class_name}. "
                           "Le DataFrame pour la simulation pourrait ne pas avoir tous les indicateurs '_strat' nécessaires.")


        # 3. Assurer le ffill final pour toutes les colonnes _strat
        indicator_strat_cols = [col for col in df_for_simulation.columns if col.endswith('_strat')]
        if indicator_strat_cols:
            df_for_simulation[indicator_strat_cols] = df_for_simulation[indicator_strat_cols].ffill()
            # Optionnel: bfill pour le début si des NaNs persistent et que c'est souhaité
            # df_for_simulation[indicator_strat_cols] = df_for_simulation[indicator_strat_cols].bfill()
            logger.debug(f"{log_prefix_prep} ffill appliqué aux colonnes _strat : {indicator_strat_cols}")

        # Vérifier si des colonnes _strat essentielles sont entièrement NaN
        for strat_col in indicator_strat_cols:
            if df_for_simulation[strat_col].isnull().all():
                logger.warning(f"{log_prefix_prep} La colonne indicateur '{strat_col}' est entièrement NaN après préparation.")
        
        # Vérifier les colonnes OHLCV de base
        if df_for_simulation[BASE_OHLCV_COLUMNS].isnull().values.any():
            logger.warning(f"{log_prefix_prep} Des NaNs existent dans les colonnes OHLCV de base de df_for_simulation après préparation. Cela pourrait affecter le simulateur.")
            # Le simulateur devrait gérer cela, mais c'est un avertissement utile.

        logger.info(f"{log_prefix_prep} Préparation des données terminée. Shape de df_for_simulation : {df_for_simulation.shape}")
        return df_for_simulation


    def __call__(self, trial: optuna.Trial) -> Tuple[float, ...]:
        """
        Fonction objectif appelée par Optuna pour chaque trial.
        """
        start_time_trial = time.time()
        # Déterminer le numéro de trial pour le logging
        trial_log_id: str
        trial_number_for_prepare_log: Optional[Union[int, str]]
        
        if self.is_oos_eval:
            # En OOS, trial.params contient les paramètres fixés par l'appelant (OOSValidator)
            # is_trial_number_for_oos_log est le numéro du trial IS d'origine
            trial_log_id = f"OOS_for_IS_Trial_{self.is_trial_number_for_oos_log}" if self.is_trial_number_for_oos_log is not None else f"OOS_FixedParams_{trial.number if hasattr(trial, 'number') else uuid.uuid4().hex[:6]}"
            trial_number_for_prepare_log = self.is_trial_number_for_oos_log
        else:
            # En IS, trial.number est le numéro du trial Optuna actuel
            trial_log_id = str(trial.number)
            trial_number_for_prepare_log = trial.number

        current_log_prefix = f"{self.log_prefix}[Trial:{trial_log_id}]"
        logger.info(f"{current_log_prefix} Démarrage de l'évaluation du trial.")

        # 1. Obtenir les paramètres du trial
        current_trial_params: Dict[str, Any]
        if self.is_oos_eval:
            # Pour OOS, les paramètres sont fixés et passés via trial.params par l'appelant (OOSValidator)
            # qui les aura mis là en utilisant study.enqueue_trial(params) ou similaire.
            current_trial_params = trial.params # Ces params sont ceux du meilleur essai IS
            logger.info(f"{current_log_prefix} Évaluation OOS avec paramètres fixes (issus de IS trial {self.is_trial_number_for_oos_log}): {current_trial_params}")
        else: # Évaluation IS, suggérer les paramètres
            try:
                current_trial_params = self._suggest_params(trial)
                if not current_trial_params: # Si _suggest_params retourne None ou vide (ex: default_params non trouvés)
                    logger.error(f"{current_log_prefix} Aucun paramètre n'a pu être suggéré. Élagueage du trial.")
                    raise optuna.exceptions.TrialPruned("Aucun paramètre suggéré.")
                logger.info(f"{current_log_prefix} Paramètres IS suggérés : {current_trial_params}")
            except optuna.exceptions.TrialPruned:
                raise # Laisser Optuna gérer l'élagage
            except Exception as e_suggest:
                logger.error(f"{current_log_prefix} Erreur lors de la suggestion des paramètres : {e_suggest}", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Erreur de suggestion de paramètres : {e_suggest}")


        # 2. Préparer les données avec les indicateurs dynamiques
        try:
            df_for_simulation = self._prepare_data_with_dynamic_indicators(current_trial_params, trial_number_for_log=trial_number_for_prepare_log)
            if df_for_simulation.empty or df_for_simulation[BASE_OHLCV_COLUMNS].isnull().all().all():
                logger.error(f"{current_log_prefix} La préparation des données a résulté en un DataFrame inutilisable (vide ou OHLCV entièrement NaN).")
                raise optuna.exceptions.TrialPruned("Données inutilisables après préparation des indicateurs.")
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e_prepare:
            logger.error(f"{current_log_prefix} Erreur lors de la préparation des données : {e_prepare}", exc_info=True)
            # Retourner la pire valeur possible pour chaque objectif
            return tuple([-float('inf') if d == "maximize" else float('inf')
                          for d in self.optuna_objectives_config.get('objectives_directions', ['maximize'])])

        # 3. Charger dynamiquement la classe de stratégie
        StrategyClass: Type['BaseStrategy']
        try:
            module = importlib.import_module(self.script_reference)
            StrategyClass = getattr(module, self.class_name) # type: ignore
            if not issubclass(StrategyClass, BaseStrategy): # type: ignore
                raise TypeError(f"{self.class_name} n'est pas une sous-classe de BaseStrategy.")
        except Exception as e_load_strat:
            logger.error(f"{current_log_prefix} Échec du chargement de la classe de stratégie {self.class_name} depuis {self.script_reference}: {e_load_strat}", exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Échec du chargement de la classe de stratégie: {e_load_strat}")

        # 4. Instancier la stratégie et configurer son contexte de backtest
        strategy_instance = StrategyClass(
            strategy_name=self.strategy_name_key,
            symbol=self.pair_symbol,
            params=current_trial_params
        )
        
        sim_defaults: SimulationDefaults = self.app_config.global_config.simulation_defaults
        leverage_to_use = int(current_trial_params.get('margin_leverage', sim_defaults.margin_leverage))

        strategy_instance.set_backtest_context(
            pair_config=self.symbol_info_data, # C'est le pair_config
            is_futures=sim_defaults.is_futures_trading,
            leverage=leverage_to_use,
            initial_equity=sim_defaults.initial_capital
        )

        # 5. Instancier et exécuter le simulateur de backtest (BacktestRunner)
        # Utilisation de BacktestRunner (ou son placeholder BacktestSimulator)
        simulator = BacktestRunner( # type: ignore
            df_ohlcv=df_for_simulation,
            strategy_instance=strategy_instance,
            initial_equity=sim_defaults.initial_capital,
            leverage=leverage_to_use,
            symbol=self.pair_symbol,
            pair_config=self.symbol_info_data, # Passer le pair_config
            trading_fee_bps=sim_defaults.trading_fee_bps,
            slippage_config=sim_defaults.slippage_config.__dict__ if hasattr(sim_defaults.slippage_config, '__dict__') else sim_defaults.slippage_config, # Passer en dict si c'est une dataclass
            is_futures=sim_defaults.is_futures_trading,
            run_id=self.run_id, # ID du WFOManager/Task
            is_oos_simulation=self.is_oos_eval, # Important pour le logging détaillé OOS
            verbosity=0 if not self.is_oos_eval else sim_defaults.backtest_verbosity # Moins verbeux pour IS
        )

        trades: List[Dict[str, Any]]
        equity_curve_df: pd.DataFrame
        # daily_equity: Dict[Any, float] # Non utilisé directement pour les objectifs Optuna
        oos_detailed_log: List[Dict[str, Any]] # Spécifique au simulateur

        try:
            logger.info(f"{current_log_prefix} Démarrage de la simulation de backtest...")
            trades, equity_curve_df, _, oos_detailed_log = simulator.run_simulation() # type: ignore
            
            # Stocker les résultats pour un accès externe (par OOSValidator par exemple)
            self.last_backtest_results = {
                "params": current_trial_params,
                "trades": trades,
                "equity_curve_df": equity_curve_df,
                "oos_detailed_trades_log": oos_detailed_log, # Peut être vide si IS
                "metrics": {} # Sera peuplé ci-dessous
            }
            logger.info(f"{current_log_prefix} Simulation terminée. Nombre de trades: {len(trades)}")

        except optuna.exceptions.TrialPruned as e_pruned_sim: # Si le simulateur élague
            logger.info(f"{current_log_prefix} Trial élagué pendant la simulation : {e_pruned_sim}")
            raise
        except Exception as e_sim:
            logger.error(f"{current_log_prefix} Erreur durant l'exécution de BacktestRunner : {e_sim}", exc_info=True)
            return tuple([-float('inf') if d == "maximize" else float('inf')
                          for d in self.optuna_objectives_config.get('objectives_directions', ['maximize'])])

        # 6. Calculer les métriques de performance
        equity_series_for_metrics = pd.Series(dtype=float)
        if not equity_curve_df.empty and 'timestamp' in equity_curve_df.columns and 'equity' in equity_curve_df.columns:
            ec_df_copy = equity_curve_df.copy()
            ec_df_copy['timestamp'] = pd.to_datetime(ec_df_copy['timestamp'], errors='coerce', utc=True)
            ec_df_copy.dropna(subset=['timestamp', 'equity'], inplace=True)
            if not ec_df_copy.empty:
                equity_series_for_metrics = ec_df_copy.set_index('timestamp')['equity'].sort_index()
        
        if equity_series_for_metrics.empty:
            logger.warning(f"{current_log_prefix} La série d'équité pour le calcul des métriques est vide. Utilisation du capital initial.")
            # Créer une série minimale si vide pour éviter les erreurs dans calculate_performance_metrics
            start_ts_data = df_for_simulation.index.min() if not df_for_simulation.empty else pd.Timestamp.now(tz='UTC')
            equity_series_for_metrics = pd.Series([sim_defaults.initial_capital], index=[start_ts_data])


        metrics = calculate_performance_metrics_from_inputs(
            trades_df=pd.DataFrame(trades),
            equity_curve_series=equity_series_for_metrics,
            initial_capital=sim_defaults.initial_capital,
            risk_free_rate_daily=(1 + sim_defaults.risk_free_rate)**(1/252) - 1, # Taux journalier
            periods_per_year=252 # Jours de trading typiques
        )
        # Assurer que les métriques clés pour les objectifs sont présentes, même si 0/NaN
        metrics['Total Trades'] = metrics.get('Total Trades', len(trades))
        metrics['Final Equity USDC'] = metrics.get('Final Equity USDC', equity_series_for_metrics.iloc[-1] if not equity_series_for_metrics.empty else sim_defaults.initial_capital)
        
        if self.last_backtest_results: # Mettre à jour les métriques dans les résultats stockés
            self.last_backtest_results["metrics"] = metrics.copy()

        # 7. Enregistrer les métriques comme user_attrs pour Optuna (si IS)
        if not self.is_oos_eval:
            for key, value in metrics.items():
                # Optuna ne gère pas bien tous les types (ex: Timedelta). Convertir ou ignorer.
                attr_value_for_optuna: Any
                if isinstance(value, (int, float, str, bool)) and pd.notna(value) and not (isinstance(value, float) and (np.isinf(value) or np.isnan(value))):
                    attr_value_for_optuna = value
                elif pd.notna(value): # Pour d'autres types, essayer de convertir en str
                    attr_value_for_optuna = str(value)
                else: # Si NaN ou None
                    attr_value_for_optuna = None # Optuna gère None
                
                if attr_value_for_optuna is not None:
                    try:
                        trial.set_user_attr(key, attr_value_for_optuna)
                    except TypeError: # Si Optuna ne peut toujours pas gérer le type après conversion
                        logger.debug(f"{current_log_prefix} Impossible de définir user_attr '{key}' avec valeur '{attr_value_for_optuna}' (type: {type(attr_value_for_optuna)}). Ignoré.")
                        trial.set_user_attr(key, str(attr_value_for_optuna)) # Tenter une dernière fois avec str

        # 8. Extraire et retourner les valeurs des objectifs
        objective_values_list: List[float] = []
        obj_names: List[str] = self.optuna_objectives_config.get('objectives_names', [])
        obj_dirs: List[str] = self.optuna_objectives_config.get('objectives_directions', [])

        if not obj_names: # Fallback si la config des objectifs est vide
            logger.warning(f"{current_log_prefix} 'objectives_names' est vide dans optuna_objectives_config. "
                           "Utilisation de 'Total Net PnL USDC' par défaut.")
            obj_names = ["Total Net PnL USDC"]
            obj_dirs = ["maximize"] # Supposer maximize pour le PnL

        for i, metric_name in enumerate(obj_names):
            value = metrics.get(metric_name)
            direction = obj_dirs[i] if i < len(obj_dirs) else "maximize" # Fallback direction

            if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
                logger.warning(f"{current_log_prefix} L'objectif '{metric_name}' a une valeur invalide : {value}. "
                               "Assignation d'une valeur très mauvaise pour guider Optuna.")
                # Attribuer une valeur très mauvaise pour que Optuna évite ces paramètres
                # Gérer le cas où il n'y a pas de trades, ce qui peut rendre certaines métriques (Sharpe) NaN.
                if metric_name == "Total Trades" and (not trades or len(trades) == 0):
                    value = 0.0 # 0 trade est une valeur valide
                elif "Ratio" in metric_name and (not trades or len(trades) == 0):
                    value = -10.0 # Un Sharpe Ratio très mauvais
                elif ("PnL" in metric_name or "Profit" in metric_name) and (not trades or len(trades) == 0):
                    value = 0.0 # PnL de 0 si pas de trades
                else: # Pour d'autres métriques NaN/inf, assigner la pire valeur
                    value = -float('inf') if direction.lower() == "maximize" else float('inf')
            
            objective_values_list.append(float(value))

        # Logguer les métriques clés et les objectifs retournés
        log_metrics_summary = {
            "Total Net PnL USDC": metrics.get("Total Net PnL USDC"),
            "Sharpe Ratio": metrics.get("Sharpe Ratio"),
            "Max Drawdown Pct": metrics.get("Max Drawdown Pct"),
            "Total Trades": metrics.get("Total Trades"),
            "Win Rate Pct": metrics.get("Win Rate Pct"),
            "Profit Factor": metrics.get("Profit Factor")
        }
        logger.info(f"{current_log_prefix} Métriques clés du trial : { {k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in log_metrics_summary.items()} }")
        
        end_time_trial = time.time()
        logger.info(f"{current_log_prefix} Évaluation du trial terminée en {end_time_trial - start_time_trial:.2f}s. "
                    f"Objectifs ({obj_names}) retournés : {objective_values_list}")
        
        return tuple(objective_values_list)

