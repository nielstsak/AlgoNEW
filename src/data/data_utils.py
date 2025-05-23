# Fichier: src/data/data_utils.py
import logging
from typing import Optional, Dict, Any, Union, List
import re

import pandas as pd
import numpy as np
import pandas_ta as ta # type: ignore

logger = logging.getLogger(__name__)

def parse_frequency_to_pandas_offset(freq_str: Optional[str]) -> Optional[str]:
    """
    Convertit une chaîne de fréquence personnalisée (ex: "5min", "1h", "1d")
    en un alias d'offset Pandas (ex: "5T", "1H", "1D").
    Gère "m" comme alias pour "min".

    Args:
        freq_str: La chaîne de fréquence à parser.

    Returns:
        L'alias d'offset Pandas correspondant, ou None si le format n'est pas supporté.
    """
    if not isinstance(freq_str, str) or not freq_str.strip():
        logger.warning("parse_frequency_to_pandas_offset: freq_str est vide, None ou non une chaîne.")
        return None
    
    freq_str_lower = freq_str.lower().strip()
    
    # Standardize 'm' to 'min' for easier regex, unless it's part of a non-minute unit (e.g. 'mon' for month, though not supported here)
    if freq_str_lower.endswith('m') and not freq_str_lower.endswith('min'):
        # Avoid changing '1mon' or similar if those were ever supported
        if not any(freq_str_lower.endswith(unit) for unit in ['mon', 'month']): # Add other month-like units if needed
             freq_str_lower = freq_str_lower[:-1] + 'min'

    match = re.fullmatch(r"(\d+)(min|h|d)", freq_str_lower)
    if match:
        num_str, unit_abbr = match.groups()
        num = int(num_str)
        if unit_abbr == 'min':
            return f"{num}T"
        elif unit_abbr == 'h':
            return f"{num}H"
        elif unit_abbr == 'd':
            return f"{num}D"
            
    logger.error(f"parse_frequency_to_pandas_offset: Format de chaîne de fréquence non supporté: '{freq_str}' (Traité comme '{freq_str_lower}')")
    return None

def get_kline_prefix_effective(freq_param_value: Optional[str]) -> str:
    """
    Standardise la génération du préfixe de colonne K-line (ex: "Klines_5min", 
    "Klines_60min", "" pour 1-min) à partir d'une chaîne de fréquence de paramètre.

    Args:
        freq_param_value: La chaîne de fréquence (ex: "5min", "1h", "1m", None).

    Returns:
        Le préfixe de colonne standardisé. "" si la fréquence est 1-minute ou invalide.
    """
    if not freq_param_value or not isinstance(freq_param_value, str):
        logger.debug("get_kline_prefix_effective: Fréquence None, vide ou non-string. Retour du préfixe vide (pour données 1min).")
        return ""

    freq_lower = freq_param_value.lower().strip()

    if freq_lower in ["1min", "1m"]: # Explicitly 1-minute
        return ""

    match_min = re.fullmatch(r"(\d+)(min|m)", freq_lower)
    if match_min:
        num = int(match_min.group(1))
        if num == 1: # "1min" or "1m" already handled
             return ""
        return f"Klines_{num}min"

    match_h = re.fullmatch(r"(\d+)h", freq_lower)
    if match_h:
        num = int(match_h.group(1))
        return f"Klines_{num * 60}min" # Convert hours to minutes for prefix

    match_d = re.fullmatch(r"(\d+)d", freq_lower)
    if match_d:
        num = int(match_d.group(1))
        return f"Klines_{num * 24 * 60}min" # Convert days to minutes for prefix

    logger.warning(f"get_kline_prefix_effective: Format de fréquence non reconnu '{freq_param_value}'. Retour du préfixe vide.")
    return ""


def aggregate_klines_rolling_for_current_timestamp(
    df_1min_slice: pd.DataFrame,
    window_size_minutes: int,
    aggregation_rules: Optional[Dict[str, Any]] = None
) -> Optional[pd.Series]:
    """
    Agrège les N dernières klines de 1 minute pour obtenir la kline agrégée "actuelle".
    Utile pour des agrégations ponctuelles. Pour générer des colonnes complètes de K-lines
    agrégées, voir `aggregate_klines_to_dataframe` ou l'approche .rolling()
    utilisée dans `preprocessing_live.py`.

    Args:
        df_1min_slice: DataFrame de klines 1-minute, indexé par timestamp.
        window_size_minutes: Nombre de dernières klines 1-minute à agréger.
        aggregation_rules: Dictionnaire optionnel spécifiant comment agréger chaque colonne.

    Returns:
        Une pd.Series représentant la kline agrégée, ou None si l'agrégation échoue.
    """
    log_prefix = f"[agg_klines_rolling_current_ts(w={window_size_minutes})]"
    if not isinstance(df_1min_slice, pd.DataFrame) or df_1min_slice.empty:
        logger.warning(f"{log_prefix} DataFrame d'entrée vide ou invalide.")
        return None
    if not isinstance(df_1min_slice.index, pd.DatetimeIndex):
        logger.error(f"{log_prefix} df_1min_slice doit avoir un DatetimeIndex.")
        return None
    if window_size_minutes <= 0:
        logger.warning(f"{log_prefix} window_size_minutes ({window_size_minutes}) doit être positif.")
        return None
        
    if len(df_1min_slice) < window_size_minutes:
        logger.debug(f"{log_prefix} Pas assez de données ({len(df_1min_slice)}) pour la fenêtre de {window_size_minutes} minutes.")
        return None

    # Default aggregation rules
    if aggregation_rules is None:
        aggregation_rules = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
            'quote_asset_volume': 'sum', 'number_of_trades': 'sum',
            'taker_buy_base_asset_volume': 'sum', 'taker_sell_base_asset_volume': 'sum',
            'taker_buy_quote_asset_volume': 'sum', 'taker_sell_quote_asset_volume': 'sum'
        }
    
    try:
        window_data = df_1min_slice.iloc[-window_size_minutes:]
        
        aggregated_values: Dict[str, Any] = {}
        for col, rule in aggregation_rules.items():
            if col in window_data.columns:
                series_to_agg = pd.to_numeric(window_data[col], errors='coerce') # Ensure numeric for aggregation
                if series_to_agg.isnull().all(): # Skip if all NaNs after conversion
                    aggregated_values[col] = np.nan
                    continue

                if rule == 'first':
                    aggregated_values[col] = series_to_agg.iloc[0]
                elif rule == 'last':
                    aggregated_values[col] = series_to_agg.iloc[-1]
                elif rule == 'max':
                    aggregated_values[col] = series_to_agg.max()
                elif rule == 'min':
                    aggregated_values[col] = series_to_agg.min()
                elif rule == 'sum':
                    aggregated_values[col] = series_to_agg.sum()
                else:
                    logger.warning(f"{log_prefix} Règle d'agrégation inconnue '{rule}' pour la colonne '{col}'. Ignorée.")
                    aggregated_values[col] = np.nan # Ou omettre la clé
            else:
                logger.debug(f"{log_prefix} Colonne '{col}' pour agrégation non trouvée dans window_data.")
        
        if not aggregated_values:
            logger.warning(f"{log_prefix} Aucune valeur agrégée produite.")
            return None

        aggregated_series = pd.Series(aggregated_values)
        aggregated_series.name = window_data.index[-1] # Timestamp de la dernière bougie 1-min incluse
        return aggregated_series
        
    except Exception as e:
        logger.error(f"{log_prefix} Erreur lors de l'agrégation glissante: {e}", exc_info=True)
        return None

def aggregate_klines_to_dataframe(
    df_1min: pd.DataFrame,
    timeframe_minutes: int,
    extra_agg_rules: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Agrège un DataFrame de klines 1-minute (avec DatetimeIndex) à un timeframe supérieur.
    Utilisé principalement pour l'enrichissement des données historiques.

    Args:
        df_1min: DataFrame de klines 1-minute.
        timeframe_minutes: Timeframe cible en minutes pour l'agrégation.
        extra_agg_rules: Règles d'agrégation supplémentaires ou pour surcharger les règles par défaut.

    Returns:
        Un DataFrame avec les klines agrégées.
    """
    log_prefix = f"[agg_klines_to_df(tf={timeframe_minutes}min)]"
    if not isinstance(df_1min.index, pd.DatetimeIndex):
        logger.error(f"{log_prefix} df_1min doit avoir un DatetimeIndex.")
        return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC'))

    if df_1min.empty:
        logger.warning(f"{log_prefix} DataFrame 1-minute vide fourni.")
        return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC'))

    if not isinstance(timeframe_minutes, int) or timeframe_minutes <= 0:
        logger.error(f"{log_prefix} timeframe_minutes ({timeframe_minutes}) doit être un entier positif.")
        return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC'))

    resample_period = f'{timeframe_minutes}T'
    
    agg_rules = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }
    
    common_extra_cols = [
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_sell_base_asset_volume',
        'taker_buy_quote_asset_volume', 'taker_sell_quote_asset_volume'
    ]
    for col in common_extra_cols:
        if col in df_1min.columns:
            agg_rules[col] = 'sum'

    if extra_agg_rules:
        agg_rules.update(extra_agg_rules)

    final_agg_rules = {col: rule for col, rule in agg_rules.items() if col in df_1min.columns}
    
    if not final_agg_rules:
        logger.warning(f"{log_prefix} Aucune règle d'agrégation applicable pour les colonnes de df_1min.")
        try:
            # Retourner un DF avec index resamplé mais sans données
            return pd.DataFrame(index=df_1min.resample(resample_period, label='right', closed='right').first().index)
        except Exception as e_resample_empty:
            logger.error(f"{log_prefix} Erreur lors du resampling pour un index vide: {e_resample_empty}")
            return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC'))


    try:
        # S'assurer que les colonnes à agréger sont numériques
        numeric_cols_to_agg = [col for col in final_agg_rules.keys() if col in df_1min.columns]
        df_1min_numeric = df_1min[numeric_cols_to_agg].apply(pd.to_numeric, errors='coerce')

        df_aggregated = df_1min_numeric.resample(resample_period, label='right', closed='right').agg(final_agg_rules)
        
        ohlc_cols_in_agg = [col for col in ['open', 'high', 'low', 'close'] if col in df_aggregated.columns]
        if ohlc_cols_in_agg:
            df_aggregated.dropna(subset=ohlc_cols_in_agg, how='all', inplace=True)
        
        logger.info(f"{log_prefix} Agrégation terminée. Shape résultant: {df_aggregated.shape}")
        return df_aggregated
    except Exception as e:
        logger.error(f"{log_prefix} Erreur lors du resampling/agrégation: {e}", exc_info=True)
        return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC'))


def calculate_atr_for_dataframe(
    df: pd.DataFrame,
    atr_low: int = 10,
    atr_high: int = 21, # Inclusive
    atr_step: int = 1
) -> pd.DataFrame:
    """
    Calcule l'ATR pour différentes périodes sur un DataFrame (qui doit déjà contenir H, L, C).
    Utilise pandas_ta.atr. Les nouvelles colonnes ATR sont nommées 'ATR_{period}'.

    Args:
        df: DataFrame contenant les colonnes 'high', 'low', 'close'.
        atr_low: Période ATR la plus basse à calculer.
        atr_high: Période ATR la plus haute à calculer.
        atr_step: Pas entre les périodes ATR.

    Returns:
        DataFrame avec les colonnes ATR ajoutées.
    """
    log_prefix = "[calc_atr_for_df]"
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning(f"{log_prefix} DataFrame vide ou invalide fourni.")
        return df.copy()
    
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"{log_prefix} DataFrame doit contenir {required_cols}. Manquant: {missing}. Calcul ATR ignoré.")
        return df.copy()

    df_with_atr = df.copy()
    
    try:
        for col in required_cols: # Ensure numeric types for pandas_ta
            df_with_atr[col] = pd.to_numeric(df_with_atr[col], errors='raise')
        df_with_atr.replace([np.inf, -np.inf], np.nan, inplace=True)
    except Exception as e_conv:
        logger.error(f"{log_prefix} Erreur de conversion des colonnes HLC en numérique: {e_conv}. Calcul ATR ignoré.")
        return df.copy()

    for period in range(atr_low, atr_high + atr_step, atr_step):
        if period <= 0:
            logger.warning(f"{log_prefix} Période ATR invalide ({period}). Ignorée.")
            df_with_atr[f'ATR_{period}'] = np.nan
            continue
        try:
            atr_series = ta.atr(high=df_with_atr['high'], low=df_with_atr['low'], close=df_with_atr['close'],
                                length=period, append=False)
            
            if atr_series is not None and isinstance(atr_series, pd.Series):
                df_with_atr[f'ATR_{period}'] = atr_series
                logger.debug(f"{log_prefix} ATR_{period} calculé. NaNs: {atr_series.isnull().sum()}/{len(atr_series)}")
            else:
                logger.warning(f"{log_prefix} Calcul de l'ATR pour période {period} n'a pas retourné de Series valide.")
                df_with_atr[f'ATR_{period}'] = np.nan
        except Exception as e:
            logger.error(f"{log_prefix} Erreur lors du calcul de l'ATR pour période {period}: {e}", exc_info=True)
            df_with_atr[f'ATR_{period}'] = np.nan
            
    return df_with_atr


def calculate_taker_pressure_ratio(
    df: pd.DataFrame,
    taker_buy_volume_col: str,
    taker_sell_volume_col: str,
    output_col_name: str
) -> pd.DataFrame:
    """
    Calcule le ratio de pression Taker (achats Taker / ventes Taker).
    Retourne np.nan si le dénominateur (ventes Taker) est zéro.
    Retourne np.nan si les deux volumes sont zéro (0/0).

    Args:
        df: DataFrame d'entrée.
        taker_buy_volume_col: Nom de la colonne des volumes d'achat Taker.
        taker_sell_volume_col: Nom de la colonne des volumes de vente Taker.
        output_col_name: Nom de la colonne pour stocker le ratio calculé.

    Returns:
        DataFrame avec la colonne de ratio ajoutée.
    """
    df_copy = df.copy()
    log_prefix = f"[calc_taker_pressure_ratio(out='{output_col_name}')]"

    if not all(col in df_copy.columns for col in [taker_buy_volume_col, taker_sell_volume_col]):
        missing_cols = [col for col in [taker_buy_volume_col, taker_sell_volume_col] if col not in df_copy.columns]
        logger.warning(f"{log_prefix} Colonnes de volume Taker manquantes: {missing_cols}.")
        df_copy[output_col_name] = np.nan
        return df_copy

    try:
        buy_vol = pd.to_numeric(df_copy[taker_buy_volume_col], errors='coerce')
        sell_vol = pd.to_numeric(df_copy[taker_sell_volume_col], errors='coerce')
    except Exception as e_conv:
        logger.error(f"{log_prefix} Erreur de conversion des colonnes de volume Taker en numérique: {e_conv}")
        df_copy[output_col_name] = np.nan
        return df_copy
        
    # Utiliser np.divide pour gérer la division, puis remplacer inf par nan.
    # 0/0 -> nan par np.divide
    # X/0 (X!=0) -> inf par np.divide
    with np.errstate(divide='ignore', invalid='ignore'): # Supprimer les avertissements de division
        ratio = np.divide(buy_vol, sell_vol)
    
    # Remplacer les infinis (résultat de X/0) par NaN. Les NaN (résultat de 0/0 ou NaN/Y) restent NaN.
    df_copy[output_col_name] = np.where(np.isinf(ratio), np.nan, ratio)
    
    logger.debug(f"{log_prefix} Ratio de pression Taker calculé. NaNs: {df_copy[output_col_name].isnull().sum()}/{len(df_copy)}")
    return df_copy


def calculate_taker_pressure_delta(
    df: pd.DataFrame,
    taker_buy_volume_col: str,
    taker_sell_volume_col: str,
    output_col_name: str
) -> pd.DataFrame:
    """
    Calcule le delta de pression Taker (achats Taker - ventes Taker).

    Args:
        df: DataFrame d'entrée.
        taker_buy_volume_col: Nom de la colonne des volumes d'achat Taker.
        taker_sell_volume_col: Nom de la colonne des volumes de vente Taker.
        output_col_name: Nom de la colonne pour stocker le delta calculé.

    Returns:
        DataFrame avec la colonne de delta ajoutée.
    """
    df_copy = df.copy()
    log_prefix = f"[calc_taker_pressure_delta(out='{output_col_name}')]"

    if not all(col in df_copy.columns for col in [taker_buy_volume_col, taker_sell_volume_col]):
        missing_cols = [col for col in [taker_buy_volume_col, taker_sell_volume_col] if col not in df_copy.columns]
        logger.warning(f"{log_prefix} Colonnes de volume Taker manquantes: {missing_cols}.")
        df_copy[output_col_name] = np.nan
        return df_copy

    try:
        buy_vol = pd.to_numeric(df_copy[taker_buy_volume_col], errors='coerce')
        sell_vol = pd.to_numeric(df_copy[taker_sell_volume_col], errors='coerce')
    except Exception as e_conv:
        logger.error(f"{log_prefix} Erreur de conversion des colonnes de volume Taker en numérique: {e_conv}")
        df_copy[output_col_name] = np.nan
        return df_copy
        
    df_copy[output_col_name] = buy_vol.sub(sell_vol, fill_value=np.nan) # fill_value=np.nan si l'un est NaN
    
    logger.debug(f"{log_prefix} Delta de pression Taker calculé. NaNs: {df_copy[output_col_name].isnull().sum()}/{len(df_copy)}")
    return df_copy

