# src/data/preprocessing_live.py
"""
Ce module est responsable du prétraitement des données brutes 1-minute acquises
en direct pour une stratégie spécifique. Il charge les données brutes, simule des
K-lines agrégées si nécessaire, calcule les indicateurs requis par la stratégie
(avec les paramètres optimisés), et sauvegarde le DataFrame traité prêt à être
utilisé par LiveTradingManager.
"""
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Union # Union ajouté
import re

import pandas as pd
import numpy as np
import pandas_ta as ta # type: ignore

# Tentative d'importation depuis src.data.data_utils
try:
    from src.data import data_utils
except ImportError:
    # Fallback si l'importation directe échoue
    logging.getLogger(__name__).critical(
        "CRITICAL (preprocessing_live): Échec de l'import de data_utils. "
        "Assurez-vous qu'il est accessible via PYTHONPATH."
    )
    # Définir une fonction factice pour data_utils.get_kline_prefix_effective
    class data_utils: # type: ignore
        @staticmethod
        def get_kline_prefix_effective(freq_param_value: Optional[str]) -> str:
            # Logique de fallback très simplifiée
            if freq_param_value and freq_param_value.lower() not in ["1min", "1m", None, ""]:
                return f"Klines_{freq_param_value}_" # Note: le vrai ajoute 'min' et gère h/d
            return ""
        @staticmethod
        def calculate_taker_pressure_ratio(df: pd.DataFrame, buy_col: str, sell_col: str, out_col: str) -> pd.DataFrame:
            df[out_col] = np.nan
            return df
        @staticmethod
        def calculate_taker_pressure_delta(df: pd.DataFrame, buy_col: str, sell_col: str, out_col: str) -> pd.DataFrame:
            df[out_col] = np.nan
            return df


logger = logging.getLogger(__name__)

# Colonnes OHLCV de base attendues dans le fichier brut 1-minute
BASE_OHLCV_COLS: List[str] = ['open', 'high', 'low', 'close', 'volume']

# Colonnes Taker de base attendues dans le fichier brut 1-minute
BASE_TAKER_COLS: List[str] = [
    'taker_buy_base_asset_volume', 'taker_sell_base_asset_volume',
    'taker_buy_quote_asset_volume', 'taker_sell_quote_asset_volume',
    'quote_asset_volume', 'number_of_trades'
]

def _ensure_required_columns(df: pd.DataFrame, required_cols: List[str], df_name: str = "DataFrame", log_prefix: str = "") -> bool:
    """Vérifie si toutes les colonnes requises sont présentes dans le DataFrame."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"{log_prefix} Le {df_name} manque les colonnes requises : {missing_cols}. Colonnes disponibles : {df.columns.tolist()}")
        return False
    return True

# --- Fonctions de calcul d'indicateurs (conservées temporairement, pourraient migrer vers IndicatorCalculator/data_utils) ---
# Note: pandas_ta.ema et pandas_ta.atr sont généralement préférés.
# Ces versions sont gardées pour correspondre au code existant et être appelées par la logique ci-dessous.

def _calculate_ema_rolling(series: pd.Series, period: int,
                           series_name_debug: str = "Series") -> pd.Series:
    """Calcule l'EMA en utilisant ewm.mean()."""
    log_prefix_calc = f"[_calculate_ema_rolling for {series_name_debug}({period})]"
    if not isinstance(series, pd.Series):
        logger.warning(f"{log_prefix_calc} L'entrée n'est pas une pandas Series. Type: {type(series)}")
        return pd.Series(np.nan, name=f"EMA_{period}", index=getattr(series, 'index', None)) # type: ignore
    if series.empty:
        logger.debug(f"{log_prefix_calc} La série d'entrée est vide.")
        return pd.Series(np.nan, index=series.index, name=f"EMA_{period}") # type: ignore
    if series.isnull().all():
        logger.debug(f"{log_prefix_calc} La série d'entrée est entièrement NaN.")
        return pd.Series(np.nan, index=series.index, name=f"EMA_{period}") # type: ignore
    if not isinstance(period, int) or period <= 0:
        logger.warning(f"{log_prefix_calc} Période invalide : {period}. Doit être un entier positif.")
        return pd.Series(np.nan, index=series.index, name=f"EMA_{period}") # type: ignore
    
    return series.ewm(span=period, adjust=False, min_periods=max(1, period)).mean() # min_periods=period

def _calculate_atr_rolling(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series,
                           period: int, series_name_debug: str = "SeriesATR") -> pd.Series:
    """Calcule l'ATR en utilisant pandas-ta."""
    log_prefix_calc = f"[_calculate_atr_rolling for {series_name_debug}({period})]"
    # Utiliser l'index de close_series comme référence si les autres sont vides ou d'un type différent
    valid_index = getattr(close_series, 'index', None)

    if not all(isinstance(s, pd.Series) for s in [high_series, low_series, close_series]):
        logger.warning(f"{log_prefix_calc} Une ou plusieurs entrées ne sont pas des pandas Series.")
        return pd.Series(np.nan, name=f"ATR_{period}", index=valid_index) # type: ignore
    if high_series.empty or low_series.empty or close_series.empty:
        logger.debug(f"{log_prefix_calc} Une ou plusieurs séries HLC d'entrée sont vides.")
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}") # type: ignore
    if high_series.isnull().all() or low_series.isnull().all() or close_series.isnull().all():
        logger.debug(f"{log_prefix_calc} Une ou plusieurs séries HLC d'entrée sont entièrement NaN.")
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}") # type: ignore
    if not isinstance(period, int) or period <= 0:
        logger.warning(f"{log_prefix_calc} Période invalide : {period}. Doit être un entier positif.")
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}") # type: ignore

    try:
        atr_series = ta.atr(high=high_series.astype(float), low=low_series.astype(float), close=close_series.astype(float), length=period, append=False)
        if isinstance(atr_series, pd.Series):
            return atr_series
        else:
            logger.warning(f"{log_prefix_calc} pandas_ta.atr n'a pas retourné une Series. Type résultat : {type(atr_series)}")
            return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}") # type: ignore
    except Exception as e:
        logger.error(f"{log_prefix_calc} Erreur lors du calcul de l'ATR avec pandas_ta : {e}", exc_info=True)
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}") # type: ignore
# --- Fin des fonctions de calcul d'indicateurs ---

def preprocess_live_data_for_strategy(
    raw_data_path: Path,
    processed_output_path: Path,
    strategy_params: Dict[str, Any],
    strategy_name: str # Nom de la stratégie pour le logging et la logique spécifique (temporaire)
) -> Optional[pd.DataFrame]:
    """
    Prétraite les données brutes 1-minute acquises en direct pour une stratégie spécifique.
    Simule des K-lines agrégées, calcule les indicateurs requis par la stratégie
    (avec les paramètres optimisés), et sauvegarde le DataFrame traité.

    Args:
        raw_data_path (Path): Chemin vers le fichier CSV des données brutes 1-minute.
        processed_output_path (Path): Chemin où sauvegarder le DataFrame traité en CSV.
        strategy_params (Dict[str, Any]): Dictionnaire des paramètres optimisés pour la stratégie.
        strategy_name (str): Nom de la stratégie (utilisé pour le logging et la logique
                             de calcul d'indicateurs spécifique si IndicatorCalculator
                             n'est pas encore utilisé).

    Returns:
        Optional[pd.DataFrame]: Le DataFrame final traité et sauvegardé, ou None en cas d'échec majeur.
    """
    pair_symbol_from_filename = raw_data_path.name.split('_')[0] # Extraire la paire du nom de fichier
    log_prefix = f"[PreprocLive][{pair_symbol_from_filename}][{strategy_name}]"
    logger.info(f"{log_prefix} Démarrage du prétraitement des données live. Sortie vers : {processed_output_path.name}")
    logger.debug(f"{log_prefix} Paramètres de la stratégie : {strategy_params}")
    processing_start_time = time.time()

    # 1. Chargement et Nettoyage des Données Brutes 1-minute
    if not raw_data_path.exists() or raw_data_path.stat().st_size == 0:
        logger.warning(f"{log_prefix} Fichier de données brutes 1-minute non trouvé ou vide : {raw_data_path}.")
        return None
    
    df_1m_raw: pd.DataFrame
    try:
        df_1m_raw = pd.read_csv(raw_data_path, low_memory=False)
        if df_1m_raw.empty:
            logger.warning(f"{log_prefix} Le fichier de données brutes {raw_data_path.name} est vide après chargement.")
            return None

        if 'timestamp' not in df_1m_raw.columns:
            logger.error(f"{log_prefix} Colonne 'timestamp' manquante dans {raw_data_path.name}.")
            return None
        
        df_1m_raw['timestamp'] = pd.to_datetime(df_1m_raw['timestamp'], errors='coerce', utc=True)
        df_1m_raw.dropna(subset=['timestamp'], inplace=True)
        if df_1m_raw.empty:
            logger.warning(f"{log_prefix} DataFrame vide après conversion/dropna du timestamp pour {raw_data_path.name}.")
            return None
        
        df_1m_raw = df_1m_raw.set_index('timestamp').sort_index()
        if not df_1m_raw.index.is_unique:
            logger.info(f"{log_prefix} Timestamps dupliqués trouvés dans les données brutes. Conservation de la dernière occurrence.")
            df_1m_raw = df_1m_raw[~df_1m_raw.index.duplicated(keep='last')]
        
        # S'assurer que les colonnes OHLCV de base et Taker sont présentes et numériques
        cols_to_check_and_convert = BASE_OHLCV_COLS + BASE_TAKER_COLS
        if not _ensure_required_columns(df_1m_raw, cols_to_check_and_convert, f"{log_prefix} Données Brutes", log_prefix):
             logger.error(f"{log_prefix} Colonnes OHLCV ou Taker de base essentielles manquantes dans les données brutes. Prétraitement annulé.")
             return None

        for col in cols_to_check_and_convert:
            df_1m_raw[col] = pd.to_numeric(df_1m_raw[col], errors='coerce')
        
        # Gestion des NaNs pour OHLCV et Taker
        df_1m_raw[BASE_OHLCV_COLS] = df_1m_raw[BASE_OHLCV_COLS].ffill().bfill()
        df_1m_raw.dropna(subset=BASE_OHLCV_COLS, inplace=True) # Les OHLCV sont critiques
        
        if df_1m_raw.empty:
            logger.warning(f"{log_prefix} DataFrame 1-minute vide après nettoyage des NaNs OHLCV.")
            return None
        
        df_1m_raw[BASE_TAKER_COLS] = df_1m_raw[BASE_TAKER_COLS].ffill() # ffill pour les données Taker

        # df_processed contiendra les OHLCV 1min + Klines agrégées + indicateurs _strat
        df_processed = df_1m_raw[BASE_OHLCV_COLS + BASE_TAKER_COLS].copy() # Commencer avec 1-min OHLCV et Taker
        logger.info(f"{log_prefix} Données brutes 1-minute chargées et nettoyées. Shape : {df_1m_raw.shape}")

    except Exception as e_load:
        logger.error(f"{log_prefix} Erreur lors du chargement/nettoyage des données brutes 1-minute depuis {raw_data_path}: {e_load}", exc_info=True)
        return None

    # 2. Simulation des K-lines Agrégées (si nécessaire)
    # Identifier les fréquences uniques nécessaires pour les indicateurs de la stratégie
    required_frequencies_from_params: set[str] = set()
    for param_key, param_value in strategy_params.items():
        if "indicateur_frequence" in param_key and isinstance(param_value, str) and param_value.lower() not in ["1min", "1m", "none", ""]:
            required_frequencies_from_params.add(param_value)
    
    # Ajouter la fréquence de base pour l'ATR SL/TP si elle est différente de 1min
    atr_base_freq_sl_tp = strategy_params.get('atr_base_frequency_sl_tp', strategy_params.get('atr_base_frequency')) # Fallback
    if atr_base_freq_sl_tp and isinstance(atr_base_freq_sl_tp, str) and atr_base_freq_sl_tp.lower() not in ["1min", "1m", "none", ""]:
        required_frequencies_from_params.add(atr_base_freq_sl_tp)

    logger.info(f"{log_prefix} Fréquences d'indicateurs requises (hors 1min) : {required_frequencies_from_params}")

    # Calculer les K-lines agrégées par rolling window
    for freq_str_config in sorted(list(required_frequencies_from_params)): # Trier pour un ordre cohérent
        tf_minutes = data_utils.parse_timeframe_to_seconds(freq_str_config) # Convertir en secondes puis en minutes
        if tf_minutes is None:
            logger.warning(f"{log_prefix} Impossible de parser la fréquence '{freq_str_config}' en minutes. Agrégation pour cette fréquence ignorée.")
            continue
        tf_minutes //= 60 # Convertir secondes en minutes

        if tf_minutes <= 1: # Déjà géré par les données 1-min de base
            continue

        logger.debug(f"{log_prefix} Simulation des K-lines agrégées pour {freq_str_config} (fenêtre : {tf_minutes} minutes)...")
        
        # Utiliser min_periods = window_size_minutes pour s'assurer que les barres agrégées sont complètes
        min_p_agg = tf_minutes
        
        kline_prefix_agg = data_utils.get_kline_prefix_effective(freq_str_config) # Ex: "Klines_5min"

        # Agrégation OHLCV
        df_processed[f"{kline_prefix_agg}_open"] = df_1m_raw['open'].rolling(window=min_p_agg, min_periods=min_p_agg).apply(lambda x: x[0] if len(x) >= min_p_agg else np.nan, raw=True)
        df_processed[f"{kline_prefix_agg}_high"] = df_1m_raw['high'].rolling(window=min_p_agg, min_periods=min_p_agg).max()
        df_processed[f"{kline_prefix_agg}_low"] = df_1m_raw['low'].rolling(window=min_p_agg, min_periods=min_p_agg).min()
        df_processed[f"{kline_prefix_agg}_close"] = df_1m_raw['close'].rolling(window=min_p_agg, min_periods=min_p_agg).apply(lambda x: x[-1] if len(x) >= min_p_agg else np.nan, raw=True)
        df_processed[f"{kline_prefix_agg}_volume"] = df_1m_raw['volume'].rolling(window=min_p_agg, min_periods=min_p_agg).sum()

        # Agrégation des volumes Taker et autres métadonnées
        for taker_col in BASE_TAKER_COLS:
            if taker_col in df_1m_raw.columns: # Vérifier si la colonne source existe
                df_processed[f"{kline_prefix_agg}_{taker_col}"] = df_1m_raw[taker_col].rolling(window=min_p_agg, min_periods=min_p_agg).sum()
            else: # Ne devrait pas arriver si le nettoyage initial est correct
                df_processed[f"{kline_prefix_agg}_{taker_col}"] = np.nan
    
    logger.debug(f"{log_prefix} Colonnes dans df_processed après simulation des K-lines agrégées : {df_processed.columns.tolist()}")


    # 3. Calcul des Indicateurs Spécifiques à la Stratégie (colonnes _strat)
    # Cette section est une adaptation de la logique existante et sera remplacée/améliorée
    # par IndicatorCalculator dans le futur.
    logger.info(f"{log_prefix} Calcul des indicateurs spécifiques à la stratégie...")

    # ATR_strat (pour SL/TP)
    atr_period_sl_tp_key = 'atr_period_sl_tp'
    atr_freq_sl_tp_key = 'atr_base_frequency_sl_tp'
    atr_period_val = strategy_params.get(atr_period_sl_tp_key)
    atr_freq_raw_val = strategy_params.get(atr_freq_sl_tp_key)

    if atr_period_val is not None and atr_freq_raw_val is not None:
        atr_period_int = int(atr_period_val)
        kline_prefix_atr_src = data_utils.get_kline_prefix_effective(str(atr_freq_raw_val))
        
        atr_high_col_src = f"{kline_prefix_atr_src}_high" if kline_prefix_atr_src else "high"
        atr_low_col_src = f"{kline_prefix_atr_src}_low" if kline_prefix_atr_src else "low"
        atr_close_col_src = f"{kline_prefix_atr_src}_close" if kline_prefix_atr_src else "close"
        
        if all(col in df_processed.columns for col in [atr_high_col_src, atr_low_col_src, atr_close_col_src]):
            df_processed['ATR_strat'] = _calculate_atr_rolling(
                df_processed[atr_high_col_src], df_processed[atr_low_col_src], df_processed[atr_close_col_src],
                period=atr_period_int, series_name_debug=f"ATR_SLTP_Src({atr_freq_raw_val})"
            )
            if 'ATR_strat' in df_processed: logger.info(f"{log_prefix} ATR_strat calculé. NaNs: {df_processed['ATR_strat'].isnull().sum()}/{len(df_processed)}")
        else:
            logger.warning(f"{log_prefix} Colonnes sources HLC pour ATR_strat non trouvées dans df_processed (nécessaires: {atr_high_col_src}, {atr_low_col_src}, {atr_close_col_src}). ATR_strat sera NaN.")
            df_processed['ATR_strat'] = np.nan
    else:
        logger.warning(f"{log_prefix} Paramètres ATR_strat ('{atr_period_sl_tp_key}' ou '{atr_freq_sl_tp_key}') manquants. ATR_strat sera NaN.")
        df_processed['ATR_strat'] = np.nan

    # Logique spécifique par stratégie (temporaire, à remplacer par IndicatorCalculator)
    # Note: Cette section est une simplification et devrait être remplacée par un appel
    # à IndicatorCalculator qui utilise strategy.get_required_indicator_configs().
    # Pour l'instant, on garde une logique similaire à l'ancien preprocessing_live.py
    # pour les stratégies connues, en se basant sur les noms de paramètres.

    if "ma_type" in strategy_params: # Indice d'une stratégie de type MA Crossover ou Triple MA
        ma_type_strat = str(strategy_params.get('ma_type', 'ema')).lower()
        
        # Pour MaCrossoverStrategy
        if 'fast_ma_period' in strategy_params and 'indicateur_frequence_ma_rapide' in strategy_params:
            kline_prefix_fast = data_utils.get_kline_prefix_effective(str(strategy_params['indicateur_frequence_ma_rapide']))
            close_col_fast = f"{kline_prefix_fast}_close" if kline_prefix_fast else "close"
            if close_col_fast in df_processed.columns:
                df_processed['MA_FAST_strat'] = _calculate_ema_rolling(df_processed[close_col_fast], int(strategy_params['fast_ma_period']))
            else: df_processed['MA_FAST_strat'] = np.nan
        
        if 'slow_ma_period' in strategy_params and 'indicateur_frequence_ma_lente' in strategy_params:
            kline_prefix_slow = data_utils.get_kline_prefix_effective(str(strategy_params['indicateur_frequence_ma_lente']))
            close_col_slow = f"{kline_prefix_slow}_close" if kline_prefix_slow else "close"
            if close_col_slow in df_processed.columns:
                df_processed['MA_SLOW_strat'] = _calculate_ema_rolling(df_processed[close_col_slow], int(strategy_params['slow_ma_period']))
            else: df_processed['MA_SLOW_strat'] = np.nan

        # Pour TripleMAAnticipationStrategy (ajoute/surcharge les MAs)
        if 'ma_short_period' in strategy_params and 'indicateur_frequence_mms' in strategy_params: # MMS = MA Mobile Short
            k_pref_s = data_utils.get_kline_prefix_effective(str(strategy_params['indicateur_frequence_mms']))
            c_col_s = f"{k_pref_s}_close" if k_pref_s else "close"
            if c_col_s in df_processed.columns:
                df_processed['MA_SHORT_strat'] = _calculate_ema_rolling(df_processed[c_col_s], int(strategy_params['ma_short_period']))
            else: df_processed['MA_SHORT_strat'] = np.nan

        if 'ma_medium_period' in strategy_params and 'indicateur_frequence_mmm' in strategy_params: # MMM = MA Mobile Medium
            k_pref_m = data_utils.get_kline_prefix_effective(str(strategy_params['indicateur_frequence_mmm']))
            c_col_m = f"{k_pref_m}_close" if k_pref_m else "close"
            if c_col_m in df_processed.columns:
                df_processed['MA_MEDIUM_strat'] = _calculate_ema_rolling(df_processed[c_col_m], int(strategy_params['ma_medium_period']))
            else: df_processed['MA_MEDIUM_strat'] = np.nan
        
        if 'ma_long_period' in strategy_params and 'indicateur_frequence_mml' in strategy_params: # MML = MA Mobile Long
            k_pref_l = data_utils.get_kline_prefix_effective(str(strategy_params['indicateur_frequence_mml']))
            c_col_l = f"{k_pref_l}_close" if k_pref_l else "close"
            if c_col_l in df_processed.columns:
                df_processed['MA_LONG_strat'] = _calculate_ema_rolling(df_processed[c_col_l], int(strategy_params['ma_long_period']))
            else: df_processed['MA_LONG_strat'] = np.nan
        
        # Pentes pour TripleMAAnticipation
        if strategy_params.get('anticipate_crossovers', False) and 'anticipation_slope_period' in strategy_params:
            slope_period = int(strategy_params['anticipation_slope_period'])
            if slope_period >= 2:
                if 'MA_SHORT_strat' in df_processed and df_processed['MA_SHORT_strat'].notna().any():
                    df_processed['SLOPE_MA_SHORT_strat'] = ta.slope(df_processed['MA_SHORT_strat'].dropna(), length=slope_period, append=False).reindex(df_processed.index, method='ffill') # type: ignore
                else: df_processed['SLOPE_MA_SHORT_strat'] = np.nan
                
                if 'MA_MEDIUM_strat' in df_processed and df_processed['MA_MEDIUM_strat'].notna().any():
                    df_processed['SLOPE_MA_MEDIUM_strat'] = ta.slope(df_processed['MA_MEDIUM_strat'].dropna(), length=slope_period, append=False).reindex(df_processed.index, method='ffill') # type: ignore
                else: df_processed['SLOPE_MA_MEDIUM_strat'] = np.nan


    if 'psar_step' in strategy_params: # Indice de PsarReversalOtocoStrategy
        kline_prefix_psar = data_utils.get_kline_prefix_effective(str(strategy_params.get('indicateur_frequence_psar')))
        h_col = f"{kline_prefix_psar}_high" if kline_prefix_psar else "high"
        l_col = f"{kline_prefix_psar}_low" if kline_prefix_psar else "low"
        c_col = f"{kline_prefix_psar}_close" if kline_prefix_psar else "close" # PSAR utilise aussi close pour l'initialisation
        if all(col in df_processed.columns for col in [h_col, l_col, c_col]):
            psar_df = ta.psar(high=df_processed[h_col], low=df_processed[l_col], close=df_processed[c_col], # pandas-ta psar a besoin de close
                              af=float(strategy_params['psar_step']), max_af=float(strategy_params['psar_max_step']), append=False)
            if psar_df is not None and isinstance(psar_df, pd.DataFrame):
                # Les noms de colonnes de ta.psar peuvent être PSARl_AF_MAX_AF, PSARs_AF_MAX_AF
                psarl_col_name = next((col for col in psar_df.columns if 'psarl' in col.lower()), None)
                psars_col_name = next((col for col in psar_df.columns if 'psars' in col.lower()), None)
                if psarl_col_name: df_processed['PSARl_strat'] = psar_df[psarl_col_name].reindex(df_processed.index, method='ffill')
                else: df_processed['PSARl_strat'] = np.nan
                if psars_col_name: df_processed['PSARs_strat'] = psar_df[psars_col_name].reindex(df_processed.index, method='ffill')
                else: df_processed['PSARs_strat'] = np.nan
            else:
                df_processed['PSARl_strat'] = np.nan; df_processed['PSARs_strat'] = np.nan
        else:
            df_processed['PSARl_strat'] = np.nan; df_processed['PSARs_strat'] = np.nan


    # 4. Finalisation et Sauvegarde
    strat_cols_final = [col for col in df_processed.columns if col.endswith('_strat')]
    if strat_cols_final:
        df_processed[strat_cols_final] = df_processed[strat_cols_final].ffill()
        logger.debug(f"{log_prefix} ffill appliqué aux colonnes _strat finales : {strat_cols_final}")

    df_final_to_save = df_processed.reset_index() # 'timestamp' redevient une colonne

    # Définir un ordre de colonnes logique pour la sortie
    # Commencer par timestamp et OHLCV de base 1-min
    ordered_output_cols: List[str] = ['timestamp'] + BASE_OHLCV_COLS
    # Ajouter les colonnes Taker 1-min
    ordered_output_cols.extend(col for col in BASE_TAKER_COLS if col in df_final_to_save.columns)
    
    # Ajouter les K-lines agrégées, groupées par timeframe puis par type (OHLCV, Taker)
    # D'abord, extraire tous les préfixes Kline_ (ex: Kline_5min, Kline_1h)
    kline_prefixes_present = sorted(list(set(
        col.split('_')[0] + "_" + col.split('_')[1] # Ex: "Kline_5min"
        for col in df_final_to_save.columns if col.startswith("Kline_")
    )))

    for k_pref in kline_prefixes_present:
        for ohlcv_part in BASE_OHLCV_COLS: # o,h,l,c,v
            col_name = f"{k_pref}_{ohlcv_part}"
            if col_name in df_final_to_save.columns: ordered_output_cols.append(col_name)
        for taker_part in BASE_TAKER_COLS:
            col_name = f"{k_pref}_{taker_part}"
            if col_name in df_final_to_save.columns: ordered_output_cols.append(col_name)
            
    # Ajouter les colonnes _strat (indicateurs finaux)
    ordered_output_cols.extend(sorted(strat_cols_final))
    
    # S'assurer que toutes les colonnes de df_final_to_save sont dans ordered_output_cols
    # et que l'ordre est appliqué.
    final_columns_ordered_existing = [col for col in ordered_output_cols if col in df_final_to_save.columns]
    # Ajouter les colonnes restantes qui n'étaient pas dans l'ordre défini (au cas où)
    remaining_cols = [col for col in df_final_to_save.columns if col not in final_columns_ordered_existing]
    df_final_to_save = df_final_to_save[final_columns_ordered_existing + remaining_cols]

    try:
        processed_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final_to_save.to_csv(processed_output_path, index=False, float_format='%.8f')
        processing_time_secs = time.time() - processing_start_time
        logger.info(f"{log_prefix} Données live prétraitées sauvegardées dans : {processed_output_path} "
                    f"(Shape: {df_final_to_save.shape}, Temps: {processing_time_secs:.3f}s)")
    except Exception as e_save:
        logger.error(f"{log_prefix} Erreur lors de la sauvegarde des données live prétraitées dans {processed_output_path}: {e_save}", exc_info=True)
        return None

    return df_final_to_save

