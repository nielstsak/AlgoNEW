# src/data/data_utils.py
"""
Ce module fournit des fonctions utilitaires pour la manipulation, l'agrégation
et la transformation des données de trading, en particulier les K-lines.
Il est utilisé notamment lors des processus d'enrichissement de données historiques.
"""
import logging
import re
from typing import Optional, Dict, Any, List 
from datetime import timezone # <<< IMPORTATION AJOUTÉE ICI

import pandas as pd
import numpy as np
import pandas_ta as ta # type: ignore 

logger = logging.getLogger(__name__)

# Colonnes OHLCV de base qui doivent être présentes dans le DataFrame de sortie final
# avec les indicateurs _strat. Ces colonnes sont extraites de df_source_enriched.
BASE_OHLCV_OUTPUT_COLUMNS: List[str] = ['open', 'high', 'low', 'close', 'volume']

def parse_frequency_to_pandas_offset(freq_str: Optional[str]) -> Optional[str]:
    """
    Convertit une chaîne de fréquence personnalisée (ex: "1s", "5min", "1h", "1d", "1w")
    en un alias d'offset Pandas (ex: "1S", "5min", "1H", "1D", "1W").
    Gère "m" comme alias pour "min". La casse est ignorée.

    Args:
        freq_str (Optional[str]): La chaîne de fréquence à parser.
            Exemples valides : "1s", "30S", "1m", "5min", "2H", "1d", "1W".

    Returns:
        Optional[str]: L'alias d'offset Pandas correspondant (ex: "5min", "1H"),
                       ou None si le format de `freq_str` n'est pas supporté ou invalide.
    """
    if not isinstance(freq_str, str) or not freq_str.strip():
        logger.debug("parse_frequency_to_pandas_offset: freq_str est vide, None ou non une chaîne. Retourne None.")
        return None

    freq_str_cleaned = freq_str.lower().strip()

    match = re.fullmatch(r"(\d+)\s*(s|m|min|h|d|w)", freq_str_cleaned)
    if not match:
        logger.warning(f"parse_frequency_to_pandas_offset: Format de chaîne de fréquence non supporté : '{freq_str_cleaned}' (original: '{freq_str}').")
        return None

    try:
        num_str, unit_abbr = match.groups()
        num = int(num_str)
        if num <= 0:
            logger.warning(f"parse_frequency_to_pandas_offset: La valeur numérique du timeframe doit être positive. Reçu : {num} dans '{freq_str}'.")
            return None
    except ValueError:
        logger.error(f"parse_frequency_to_pandas_offset: Erreur de conversion de la valeur numérique '{num_str}' dans '{freq_str}'.")
        return None


    if unit_abbr == 's':
        return f"{num}S"
    elif unit_abbr in ('m', 'min'):
        return f"{num}min" # Correction: 'T' est déprécié, utiliser 'min'
    elif unit_abbr == 'h':
        return f"{num}H"
    elif unit_abbr == 'd':
        return f"{num}D"
    elif unit_abbr == 'w':
        return f"{num}W" 
    else:
        logger.error(f"parse_frequency_to_pandas_offset: Unité d'abréviation non gérée '{unit_abbr}' (ceci est inattendu).")
        return None

def get_kline_prefix_effective(freq_param_value: Optional[str]) -> str:
    """
    Standardise la génération du préfixe de colonne K-line (ex: "Klines_5min",
    "Klines_60min" pour 1h, "" pour 1-min) à partir d'une chaîne de fréquence.
    Utilisé pour nommer les colonnes des K-lines agrégées.

    Args:
        freq_param_value (Optional[str]): La chaîne de fréquence du paramètre
            (ex: "5min", "1h", "1m", None).

    Returns:
        str: Le préfixe de colonne standardisé. Retourne une chaîne vide ("")
             si la fréquence est 1-minute, None, vide, ou invalide.
    """
    if not freq_param_value or not isinstance(freq_param_value, str):
        logger.debug("get_kline_prefix_effective: Fréquence None, vide ou non-string. Retour du préfixe vide (pour données 1min).")
        return ""

    freq_lower = freq_param_value.lower().strip()

    if freq_lower in ["1min", "1m"]: 
        return ""

    match = re.fullmatch(r"(\d+)\s*(s|m|min|h|d|w)", freq_lower)
    if not match:
        logger.warning(f"get_kline_prefix_effective: Format de fréquence non reconnu '{freq_param_value}'. Retour du préfixe vide.")
        return ""
    
    try:
        num_val_str, unit_abbr = match.groups()
        num_val = int(num_val_str)
        if num_val <=0: return "" 
    except ValueError:
        return ""


    total_minutes = 0
    if unit_abbr == 's':
        if num_val == 60: total_minutes = 1 
        else: return f"Klines_{num_val}s" 
    elif unit_abbr in ["m", "min"]:
        total_minutes = num_val
    elif unit_abbr == "h":
        total_minutes = num_val * 60
    elif unit_abbr == "d":
        total_minutes = num_val * 24 * 60
    elif unit_abbr == "w":
        total_minutes = num_val * 7 * 24 * 60
    
    if total_minutes == 1: 
        return ""
    elif total_minutes > 0 :
        return f"Klines_{total_minutes}min" 
    
    logger.warning(f"get_kline_prefix_effective: Conversion de '{freq_param_value}' en minutes a échoué ou a donné 0. Retour du préfixe vide.")
    return ""


def aggregate_klines_rolling_for_current_timestamp(
    df_1min_slice: pd.DataFrame,
    window_size_minutes: int,
    aggregation_rules: Optional[Dict[str, Any]] = None
) -> Optional[pd.Series]:
    """
    Agrège les `window_size_minutes` dernières klines de 1 minute d'un slice pour
    obtenir la kline agrégée "actuelle" se terminant au dernier timestamp du slice.

    Args:
        df_1min_slice (pd.DataFrame): DataFrame de klines 1-minute, indexé par
                                      timestamp (DatetimeIndex UTC attendu).
        window_size_minutes (int): Nombre de dernières klines 1-minute à agréger.
        aggregation_rules (Optional[Dict[str, Any]]): Dictionnaire spécifiant comment
            agréger chaque colonne.

    Returns:
        Optional[pd.Series]: Une Série pandas représentant la kline agrégée.
    """
    log_prefix = f"[AggRollCurrent][Win:{window_size_minutes}min]"
    if not isinstance(df_1min_slice, pd.DataFrame) or df_1min_slice.empty:
        logger.warning(f"{log_prefix} DataFrame d'entrée (df_1min_slice) vide ou invalide.")
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

    default_agg_rules = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
        'quote_asset_volume': 'sum', 'number_of_trades': 'sum',
        'taker_buy_base_asset_volume': 'sum', 'taker_sell_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum', 'taker_sell_quote_asset_volume': 'sum'
    }
    final_agg_rules = aggregation_rules if aggregation_rules is not None else default_agg_rules

    try:
        window_data = df_1min_slice.iloc[-window_size_minutes:]
        aggregated_values: Dict[str, Any] = {}
        for col, rule in final_agg_rules.items():
            if col in window_data.columns:
                series_to_agg = pd.to_numeric(window_data[col], errors='coerce')
                if series_to_agg.isnull().all():
                    aggregated_values[col] = np.nan
                    continue
                if rule == 'first': aggregated_values[col] = series_to_agg.iloc[0]
                elif rule == 'last': aggregated_values[col] = series_to_agg.iloc[-1]
                elif rule == 'max': aggregated_values[col] = series_to_agg.max()
                elif rule == 'min': aggregated_values[col] = series_to_agg.min()
                elif rule == 'sum': aggregated_values[col] = series_to_agg.sum()
                else:
                    aggregated_values[col] = np.nan
        
        if not aggregated_values: return None
        aggregated_series = pd.Series(aggregated_values)
        aggregated_series.name = window_data.index[-1]
        logger.debug(f"{log_prefix} Agrégation ponctuelle réussie. Timestamp: {aggregated_series.name}")
        return aggregated_series
        
    except Exception as e:
        logger.error(f"{log_prefix} Erreur lors de l'agrégation glissante ponctuelle : {e}", exc_info=True)
        return None

def aggregate_klines_to_dataframe(
    df_1min: pd.DataFrame,
    timeframe_minutes: int,
    extra_agg_rules: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Agrège un DataFrame de klines 1-minute (avec DatetimeIndex UTC) à un timeframe supérieur.

    Args:
        df_1min (pd.DataFrame): DataFrame de klines 1-minute, avec un DatetimeIndex UTC.
        timeframe_minutes (int): Timeframe cible en minutes (ex: 5, 15, 60). Doit être > 1.
        extra_agg_rules (Optional[Dict[str, str]]): Règles d'agrégation supplémentaires.

    Returns:
        pd.DataFrame: DataFrame avec les klines agrégées.
    """
    log_prefix = f"[AggToDF][{timeframe_minutes}min]"
    
    if not isinstance(df_1min.index, pd.DatetimeIndex):
        logger.error(f"{log_prefix} df_1min doit avoir un DatetimeIndex.")
        return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC')) # type: ignore
    
    if df_1min.index.tz is None:
        logger.warning(f"{log_prefix} L'index de df_1min n'a pas de timezone. Localisation en UTC.")
        try:
            df_1min.index = df_1min.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
        except Exception as e_tz_loc:
            logger.error(f"{log_prefix} Échec de la localisation de l'index en UTC: {e_tz_loc}. Agrégation annulée.")
            return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC')) # type: ignore
    elif df_1min.index.tz.utcoffset(df_1min.index[0] if not df_1min.empty else None) != timezone.utc.utcoffset(None): # type: ignore
        logger.warning(f"{log_prefix} L'index de df_1min n'est pas en UTC (actuel: {df_1min.index.tz}). Conversion en UTC.")
        try:
            df_1min.index = df_1min.index.tz_convert('UTC')
        except Exception as e_tz_conv:
            logger.error(f"{log_prefix} Échec de la conversion de l'index en UTC: {e_tz_conv}. Agrégation annulée.")
            return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC')) # type: ignore


    if df_1min.empty:
        logger.warning(f"{log_prefix} DataFrame 1-minute vide fourni pour agrégation.")
        return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC')) # type: ignore

    if not isinstance(timeframe_minutes, int) or timeframe_minutes <= 1:
        logger.error(f"{log_prefix} timeframe_minutes ({timeframe_minutes}) doit être un entier > 1.")
        return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC')) # type: ignore

    resample_period_str = f'{timeframe_minutes}min' # Correction: 'T' est déprécié, utiliser 'min'
    
    agg_rules_default = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    common_extra_cols_to_sum = [
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_sell_base_asset_volume',
        'taker_buy_quote_asset_volume', 'taker_sell_quote_asset_volume'
    ]
    for col in common_extra_cols_to_sum:
        if col in df_1min.columns:
            agg_rules_default[col] = 'sum'

    final_agg_rules = agg_rules_default.copy()
    if extra_agg_rules:
        final_agg_rules.update(extra_agg_rules)

    actual_rules_to_apply = {col: rule for col, rule in final_agg_rules.items() if col in df_1min.columns}
    
    if not actual_rules_to_apply:
        logger.warning(f"{log_prefix} Aucune règle d'agrégation applicable pour les colonnes de df_1min : {df_1min.columns.tolist()}")
        try:
            return pd.DataFrame(index=df_1min.resample(resample_period_str, label='right', closed='right').first().index)
        except Exception as e_resample_empty_rules:
            logger.error(f"{log_prefix} Erreur lors du resampling pour un index vide: {e_resample_empty_rules}")
            return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC')) # type: ignore

    try:
        numeric_cols_for_agg = list(actual_rules_to_apply.keys())
        df_1min_numeric_subset = df_1min[numeric_cols_for_agg].apply(pd.to_numeric, errors='coerce')
        df_aggregated = df_1min_numeric_subset.resample(resample_period_str, label='right', closed='right').agg(actual_rules_to_apply)
        
        ohlc_cols_in_aggregated = [col for col in ['open', 'high', 'low', 'close'] if col in df_aggregated.columns]
        if ohlc_cols_in_aggregated:
            df_aggregated.dropna(subset=ohlc_cols_in_aggregated, how='all', inplace=True)
        
        logger.info(f"{log_prefix} Agrégation vers {timeframe_minutes}min terminée. Shape résultant : {df_aggregated.shape}")
        return df_aggregated
    except Exception as e:
        logger.error(f"{log_prefix} Erreur lors du resampling/agrégation vers {timeframe_minutes}min : {e}", exc_info=True)
        return pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp', tz='UTC')) # type: ignore


def calculate_atr_for_dataframe(
    df: pd.DataFrame,
    atr_low: int = 10,
    atr_high: int = 21, 
    atr_step: int = 1
) -> pd.DataFrame:
    """
    Calcule l'Average True Range (ATR) pour différentes périodes sur un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant 'high', 'low', 'close'.
        atr_low (int): Période ATR la plus basse.
        atr_high (int): Période ATR la plus haute (inclusive).
        atr_step (int): Pas entre les périodes ATR.

    Returns:
        pd.DataFrame: DataFrame original avec les colonnes ATR calculées ajoutées.
    """
    log_prefix = "[CalcATR]"
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning(f"{log_prefix} DataFrame vide ou invalide. Retour du DataFrame original.")
        return df.copy()
    
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"{log_prefix} Colonnes HLC manquantes : {missing}. Calcul ATR ignoré.")
        return df.copy()

    df_with_atr = df.copy()
    
    try:
        for col in required_cols:
            df_with_atr[col] = pd.to_numeric(df_with_atr[col], errors='raise')
        df_with_atr.replace([np.inf, -np.inf], np.nan, inplace=True)
    except Exception as e_conv:
        logger.error(f"{log_prefix} Erreur de conversion HLC ou gestion infinis : {e_conv}. Calcul ATR ignoré.")
        return df.copy()

    logger.info(f"{log_prefix} Calcul des ATRs de {atr_low} à {atr_high} (pas: {atr_step}).")
    for period in range(atr_low, atr_high + atr_step, atr_step):
        if period <= 0:
            logger.warning(f"{log_prefix} Période ATR invalide ({period}). Ignorée.")
            df_with_atr[f'ATR_{period}'] = np.nan
            continue
        try:
            atr_series = ta.atr(
                high=df_with_atr['high'], low=df_with_atr['low'], close=df_with_atr['close'],
                length=period, append=False
            )
            if isinstance(atr_series, pd.Series):
                df_with_atr[f'ATR_{period}'] = atr_series
            else:
                df_with_atr[f'ATR_{period}'] = np.nan
        except Exception as e:
            logger.error(f"{log_prefix} Erreur calcul ATR période {period} : {e}", exc_info=True)
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

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        taker_buy_volume_col (str): Nom de la colonne des volumes d'achat Taker.
        taker_sell_volume_col (str): Nom de la colonne des volumes de vente Taker.
        output_col_name (str): Nom de la colonne pour stocker le ratio.

    Returns:
        pd.DataFrame: DataFrame original avec la colonne de ratio ajoutée.
    """
    df_copy = df.copy()
    log_prefix = f"[CalcTakerRatio][Out:'{output_col_name}']"

    if not all(col in df_copy.columns for col in [taker_buy_volume_col, taker_sell_volume_col]):
        missing_cols = [col for col in [taker_buy_volume_col, taker_sell_volume_col] if col not in df_copy.columns]
        logger.warning(f"{log_prefix} Colonnes Taker manquantes : {missing_cols}. '{output_col_name}' sera NaN.")
        df_copy[output_col_name] = np.nan
        return df_copy

    try:
        buy_vol = pd.to_numeric(df_copy[taker_buy_volume_col], errors='coerce')
        sell_vol = pd.to_numeric(df_copy[taker_sell_volume_col], errors='coerce')
    except Exception as e_conv: 
        logger.error(f"{log_prefix} Erreur conversion volumes Taker : {e_conv}. '{output_col_name}' sera NaN.")
        df_copy[output_col_name] = np.nan
        return df_copy
        
    with np.errstate(divide='ignore', invalid='ignore'): 
        ratio = np.divide(buy_vol, sell_vol)
    df_copy[output_col_name] = ratio
    
    logger.debug(f"{log_prefix} Ratio Taker calculé. Infinis: {np.isinf(df_copy[output_col_name]).sum()}, NaNs: {df_copy[output_col_name].isnull().sum()}.")
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
        df (pd.DataFrame): DataFrame d'entrée.
        taker_buy_volume_col (str): Nom de la colonne des volumes d'achat Taker.
        taker_sell_volume_col (str): Nom de la colonne des volumes de vente Taker.
        output_col_name (str): Nom de la colonne pour stocker le delta.

    Returns:
        pd.DataFrame: DataFrame original avec la colonne de delta ajoutée.
    """
    df_copy = df.copy()
    log_prefix = f"[CalcTakerDelta][Out:'{output_col_name}']"

    if not all(col in df_copy.columns for col in [taker_buy_volume_col, taker_sell_volume_col]):
        missing_cols = [col for col in [taker_buy_volume_col, taker_sell_volume_col] if col not in df_copy.columns]
        logger.warning(f"{log_prefix} Colonnes Taker manquantes : {missing_cols}. '{output_col_name}' sera NaN.")
        df_copy[output_col_name] = np.nan
        return df_copy

    try:
        buy_vol = pd.to_numeric(df_copy[taker_buy_volume_col], errors='coerce')
        sell_vol = pd.to_numeric(df_copy[taker_sell_volume_col], errors='coerce')
    except Exception as e_conv: 
        logger.error(f"{log_prefix} Erreur conversion volumes Taker : {e_conv}. '{output_col_name}' sera NaN.")
        df_copy[output_col_name] = np.nan
        return df_copy
        
    df_copy[output_col_name] = buy_vol.sub(sell_vol)
    
    logger.debug(f"{log_prefix} Delta Taker calculé. NaNs: {df_copy[output_col_name].isnull().sum()}.")
    return df_copy

