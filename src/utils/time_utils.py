# src/utils/time_utils.py
"""
Ce module fournit des fonctions utilitaires pour la manipulation des dates,
des heures et des timestamps, essentielles pour la gestion des données temporelles
dans l'application de trading.
"""
import logging
from datetime import datetime, timezone, timedelta
import re
from typing import Optional, Union

import pandas as pd
import numpy as np # Ajouté pour np.isnan, np.isinf

logger = logging.getLogger(__name__)

def get_current_utc_timestamp_ms() -> int:
    """
    Retourne le timestamp Unix actuel en millisecondes, en UTC.

    Returns:
        int: Timestamp Unix UTC en millisecondes.
    """
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def get_current_utc_datetime() -> datetime:
    """
    Retourne l'objet datetime actuel, conscient du fuseau horaire UTC.

    Returns:
        datetime: Objet datetime actuel en UTC.
    """
    return datetime.now(timezone.utc)

def parse_timeframe_to_timedelta(tf_string: str) -> Optional[pd.Timedelta]:
    """
    Convertit une chaîne de timeframe (ex: "1s", "1m", "5min", "1h", "1d", "1w")
    en un objet pd.Timedelta. Gère les variations de casse et les espaces.

    Args:
        tf_string (str): La chaîne de timeframe à parser.

    Returns:
        Optional[pd.Timedelta]: L'objet Timedelta correspondant, ou None si le
                                format est inconnu ou invalide.
    """
    if not isinstance(tf_string, str) or not tf_string.strip():
        logger.warning(f"parse_timeframe_to_timedelta: tf_string invalide ou vide : '{tf_string}'")
        return None

    tf_string_cleaned = tf_string.lower().strip()

    # Regex pour capturer la valeur numérique et l'unité (s, m, min, h, d, w)
    match = re.fullmatch(r"(\d+)\s*(s|m|min|h|d|w)", tf_string_cleaned)
    if not match:
        logger.warning(f"parse_timeframe_to_timedelta: Format de timeframe inconnu pour '{tf_string_cleaned}' (original: '{tf_string}').")
        return None

    try:
        value_str, unit_str = match.groups()
        value = int(value_str)
        if value <= 0:
            logger.warning(f"parse_timeframe_to_timedelta: La valeur numérique du timeframe doit être positive. Reçu : {value} dans '{tf_string}'.")
            return None
    except ValueError:
        logger.error(f"parse_timeframe_to_timedelta: Erreur de conversion de la valeur numérique '{value_str}' dans '{tf_string}'.")
        return None


    if unit_str == "s":
        return pd.Timedelta(seconds=value)
    elif unit_str in ("m", "min"):
        return pd.Timedelta(minutes=value)
    elif unit_str == "h":
        return pd.Timedelta(hours=value)
    elif unit_str == "d":
        return pd.Timedelta(days=value)
    elif unit_str == "w":
        return pd.Timedelta(weeks=value)
    else:
        # Ce cas ne devrait théoriquement pas être atteint grâce au regex, mais par sécurité.
        logger.error(f"parse_timeframe_to_timedelta: Unité de timeframe non gérée '{unit_str}' dans '{tf_string_cleaned}'. Ceci est inattendu.")
        return None

def parse_timeframe_to_seconds(tf_string: str) -> Optional[int]:
    """
    Convertit une chaîne de timeframe en un nombre total de secondes.

    Args:
        tf_string (str): La chaîne de timeframe à parser.

    Returns:
        Optional[int]: Le nombre total de secondes, ou None si la conversion échoue.
    """
    timedelta_obj = parse_timeframe_to_timedelta(tf_string)
    if timedelta_obj:
        return int(timedelta_obj.total_seconds())
    return None

def round_datetime_to_minute(dt: datetime, direction: str = "floor") -> datetime:
    """
    Arrondit un objet datetime à la minute la plus proche (floor, ceil, ou round),
    en préservant le fuseau horaire de l'objet datetime d'entrée.

    Args:
        dt (datetime): L'objet datetime à arrondir.
        direction (str): Méthode d'arrondi ("floor", "ceil", "round").
                         Par défaut "floor".

    Returns:
        datetime: L'objet datetime arrondi.

    Raises:
        TypeError: Si 'dt' n'est pas un objet datetime.
        ValueError: Si 'direction' n'est pas une des valeurs attendues.
    """
    if not isinstance(dt, datetime):
        logger.error(f"round_datetime_to_minute: L'entrée 'dt' doit être un objet datetime. Reçu : {type(dt)}")
        raise TypeError("L'entrée 'dt' doit être un objet datetime.")

    if direction == "floor":
        return dt.replace(second=0, microsecond=0)
    elif direction == "ceil":
        if dt.second > 0 or dt.microsecond > 0:
            # Ajoute une minute et tronque les secondes/microsecondes
            return (dt + timedelta(minutes=1)).replace(second=0, microsecond=0)
        return dt.replace(second=0, microsecond=0) # Déjà à la minute exacte
    elif direction == "round":
        if dt.second >= 30: # Arrondi à la minute supérieure si secondes >= 30
            return (dt + timedelta(minutes=1)).replace(second=0, microsecond=0)
        return dt.replace(second=0, microsecond=0) # Arrondi à la minute inférieure
    else:
        logger.error(f"round_datetime_to_minute: Direction d'arrondi inconnue '{direction}'. "
                     "Les valeurs valides sont 'floor', 'ceil', 'round'.")
        raise ValueError("Direction d'arrondi invalide. Utilisez 'floor', 'ceil', ou 'round'.")

def convert_to_utc_datetime(
    dt_input: Union[str, datetime, int, float],
    unit_if_timestamp: Optional[str] = 'ms'
) -> Optional[datetime]:
    """
    Convertit diverses représentations de temps en un objet datetime conscient
    du fuseau horaire UTC.

    Args:
        dt_input: Peut être une chaîne de caractères (format ISO ou parsable par
                  pd.to_datetime), un objet datetime existant, ou un timestamp
                  Unix (int ou float).
        unit_if_timestamp (Optional[str]): Si dt_input est un nombre, spécifie
            l'unité du timestamp ('s' pour secondes, 'ms' pour millisecondes).
            Par défaut 'ms'.

    Returns:
        Optional[datetime]: Un objet datetime en UTC, ou None en cas d'échec de
                            parsing ou de conversion.
    """
    if isinstance(dt_input, datetime):
        if dt_input.tzinfo is None:
            # Pour un datetime naïf, on suppose qu'il est en UTC et on le rend conscient.
            logger.debug(f"convert_to_utc_datetime: datetime naïf fourni ({dt_input}), en le considérant comme UTC.")
            return dt_input.replace(tzinfo=timezone.utc)
        else:
            # Si déjà conscient, convertir en UTC.
            return dt_input.astimezone(timezone.utc)
    elif isinstance(dt_input, str):
        try:
            # pd.to_datetime est flexible. utc=True le convertit directement en UTC.
            # errors='raise' pour être informé des problèmes de parsing.
            dt_obj_pandas = pd.to_datetime(dt_input, utc=True, errors='raise')
            if isinstance(dt_obj_pandas, pd.Timestamp):
                return dt_obj_pandas.to_pydatetime() # Convertir en objet datetime standard de Python
            # Normalement, pd.to_datetime avec une seule chaîne retourne un Timestamp.
            # Si ce n'est pas le cas, c'est inattendu.
            logger.warning(f"convert_to_utc_datetime: pd.to_datetime n'a pas retourné un Timestamp pour la chaîne '{dt_input}'. Type retourné: {type(dt_obj_pandas)}")
            return None # Ou gérer ce cas différemment
        except Exception as e:
            logger.warning(f"convert_to_utc_datetime: Échec du parsing de la chaîne '{dt_input}' en datetime UTC : {e}")
            return None
    elif isinstance(dt_input, (int, float)):
        if np.isnan(dt_input) or np.isinf(dt_input): # type: ignore
            logger.warning(f"convert_to_utc_datetime: Timestamp Unix invalide (NaN/Inf) : {dt_input}")
            return None
        try:
            ts_seconds: float
            if unit_if_timestamp == 'ms':
                ts_seconds = float(dt_input) / 1000.0
            elif unit_if_timestamp == 's':
                ts_seconds = float(dt_input)
            else:
                logger.warning(f"convert_to_utc_datetime: Unité de timestamp inconnue '{unit_if_timestamp}'. "
                               "Les unités valides sont 's' ou 'ms'. Tentative avec 's'.")
                ts_seconds = float(dt_input) # Fallback à secondes

            return datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
        except (ValueError, OverflowError, OSError) as e_ts: # Erreurs possibles avec fromtimestamp
            logger.warning(f"convert_to_utc_datetime: Échec de la conversion du timestamp Unix {dt_input} "
                           f"(unité supposée: {unit_if_timestamp}, valeur en secondes: {ts_seconds if 'ts_seconds' in locals() else 'N/A'}) en datetime UTC : {e_ts}")
            return None
    else:
        logger.warning(f"convert_to_utc_datetime: Type d'entrée non supporté : {type(dt_input)}")
        return None

def format_datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """
    Formate un objet datetime en une chaîne ISO 8601 standard (avec 'Z' pour UTC).
    Inclut les millisecondes si elles sont présentes et non nulles (3 chiffres).

    Args:
        dt (Optional[datetime]): L'objet datetime à formater.

    Returns:
        Optional[str]: La chaîne formatée ISO, ou None si l'entrée est None.
                       Si l'entrée n'est pas un datetime, tente une conversion str.
    """
    if dt is None:
        return None
    if not isinstance(dt, datetime):
        logger.warning(f"format_datetime_to_iso: L'entrée n'est pas un objet datetime. Reçu : {type(dt)}. Tentative de conversion str.")
        return str(dt) # Comportement de fallback

    dt_utc: datetime
    if dt.tzinfo is None:
        logger.debug(f"format_datetime_to_iso: datetime naïf fourni ({dt}), en le considérant comme UTC.")
        dt_utc = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        dt_utc = dt.astimezone(timezone.utc)
    else:
        dt_utc = dt

    # Formater avec millisecondes (3 chiffres) si non nulles, sinon sans.
    if dt_utc.microsecond > 0:
        # strftime '%f' donne 6 chiffres pour microsecondes. On prend les 3 premiers pour millisecondes.
        return dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    else:
        return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

if __name__ == '__main__':
    # Configuration du logging pour les tests directs de ce module
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Tests pour time_utils ---")

    # Tests pour get_current_utc_timestamp_ms et get_current_utc_datetime
    ts_ms_now = get_current_utc_timestamp_ms()
    dt_now = get_current_utc_datetime()
    logger.info(f"Timestamp UTC actuel (ms) : {ts_ms_now}")
    logger.info(f"Datetime UTC actuel : {dt_now} (TZ: {dt_now.tzinfo})")
    assert dt_now.tzinfo == timezone.utc

    # Tests pour parse_timeframe_to_timedelta et parse_timeframe_to_seconds
    timeframes_to_test = ["1s", "30s", "1m", "5min", "1H", "4h", "1D", "2w", "10 d", "  3m  ", "1MIN", "1x"]
    expected_deltas = [
        pd.Timedelta(seconds=1), pd.Timedelta(seconds=30), pd.Timedelta(minutes=1), pd.Timedelta(minutes=5),
        pd.Timedelta(hours=1), pd.Timedelta(hours=4), pd.Timedelta(days=1), pd.Timedelta(weeks=2),
        pd.Timedelta(days=10), pd.Timedelta(minutes=3), pd.Timedelta(minutes=1), None
    ]
    for tf_str, expected_td in zip(timeframes_to_test, expected_deltas):
        td = parse_timeframe_to_timedelta(tf_str)
        secs = parse_timeframe_to_seconds(tf_str)
        logger.info(f"Timeframe '{tf_str}': Timedelta = {td}, Secondes = {secs}")
        assert td == expected_td
        if expected_td:
            assert secs == expected_td.total_seconds()
        else:
            assert secs is None
    assert parse_timeframe_to_timedelta("0m") is None # Test valeur 0
    assert parse_timeframe_to_timedelta("-5m") is None # Test valeur négative

    # Tests pour round_datetime_to_minute
    dt_test_rounding = datetime(2023, 10, 26, 14, 35, 45, 123456, tzinfo=timezone.utc)
    logger.info(f"Arrondi pour {dt_test_rounding}:")
    logger.info(f"  Floor: {round_datetime_to_minute(dt_test_rounding, 'floor')} (Attendu: ...14:35:00)")
    assert round_datetime_to_minute(dt_test_rounding, 'floor') == datetime(2023, 10, 26, 14, 35, 0, tzinfo=timezone.utc)
    logger.info(f"  Ceil:  {round_datetime_to_minute(dt_test_rounding, 'ceil')} (Attendu: ...14:36:00)")
    assert round_datetime_to_minute(dt_test_rounding, 'ceil') == datetime(2023, 10, 26, 14, 36, 0, tzinfo=timezone.utc)
    logger.info(f"  Round (45s): {round_datetime_to_minute(dt_test_rounding, 'round')} (Attendu: ...14:36:00)")
    assert round_datetime_to_minute(dt_test_rounding, 'round') == datetime(2023, 10, 26, 14, 36, 0, tzinfo=timezone.utc)
    dt_test_rounding_20s = datetime(2023, 10, 26, 14, 35, 20, tzinfo=timezone.utc)
    logger.info(f"  Round (20s): {round_datetime_to_minute(dt_test_rounding_20s, 'round')} (Attendu: ...14:35:00)")
    assert round_datetime_to_minute(dt_test_rounding_20s, 'round') == datetime(2023, 10, 26, 14, 35, 0, tzinfo=timezone.utc)
    try:
        round_datetime_to_minute(dt_test_rounding, "invalid_direction") # type: ignore
        assert False, "ValueError non levée pour direction invalide"
    except ValueError:
        logger.info("Test d'erreur (direction d'arrondi invalide) RÉUSSI.")


    # Tests pour convert_to_utc_datetime
    logger.info("Tests pour convert_to_utc_datetime:")
    logger.info(f"  String ISO Z: {convert_to_utc_datetime('2023-10-26T12:30:00Z')}")
    logger.info(f"  String ISO avec ms Z: {convert_to_utc_datetime('2023-10-26T12:30:00.500Z')}")
    logger.info(f"  String ISO avec offset: {convert_to_utc_datetime('2023-10-26T14:30:00+02:00')}")
    logger.info(f"  Datetime naïf: {convert_to_utc_datetime(datetime(2023, 1, 1, 10, 0, 0))}")
    tz_plus_2 = timezone(timedelta(hours=2))
    logger.info(f"  Datetime conscient non-UTC: {convert_to_utc_datetime(datetime(2023, 1, 1, 10, 0, 0, tzinfo=tz_plus_2))}")
    ts_ms = get_current_utc_timestamp_ms()
    logger.info(f"  Timestamp ms ({ts_ms}): {convert_to_utc_datetime(ts_ms, 'ms')}")
    ts_s = int(ts_ms / 1000)
    logger.info(f"  Timestamp s ({ts_s}): {convert_to_utc_datetime(ts_s, 's')}")
    logger.info(f"  Timestamp float s : {convert_to_utc_datetime(float(ts_s) + 0.5, 's')}")
    logger.info(f"  Timestamp NaN: {convert_to_utc_datetime(np.nan, 's')}")
    assert convert_to_utc_datetime(np.nan, 's') is None
    logger.info(f"  Timestamp Inf: {convert_to_utc_datetime(np.inf, 's')}")
    assert convert_to_utc_datetime(np.inf, 's') is None

    # Tests pour format_datetime_to_iso
    logger.info("Tests pour format_datetime_to_iso:")
    dt_with_ms = datetime(2023, 5, 15, 10, 20, 30, 123456, tzinfo=timezone.utc)
    dt_without_ms = dt_with_ms.replace(microsecond=0)
    dt_naive_for_iso = datetime(2023, 5, 15, 10, 20, 30)
    logger.info(f"  Avec ms: {format_datetime_to_iso(dt_with_ms)} (Attendu: ...T10:20:30.123Z)")
    assert format_datetime_to_iso(dt_with_ms) == "2023-05-15T10:20:30.123Z"
    logger.info(f"  Sans ms: {format_datetime_to_iso(dt_without_ms)} (Attendu: ...T10:20:30Z)")
    assert format_datetime_to_iso(dt_without_ms) == "2023-05-15T10:20:30Z"
    logger.info(f"  Naïf (devrait être traité comme UTC): {format_datetime_to_iso(dt_naive_for_iso)}")
    assert format_datetime_to_iso(dt_naive_for_iso) == "2023-05-15T10:20:30Z"
    logger.info(f"  None: {format_datetime_to_iso(None)}")
    assert format_datetime_to_iso(None) is None

    logger.info("--- Fin des tests pour time_utils ---")

