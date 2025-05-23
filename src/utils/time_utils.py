# Fichier: src/utils/time_utils.py
"""
Fonctions utilitaires pour la manipulation des dates, heures, et timestamps.
"""
import logging
from datetime import datetime, timezone, timedelta
import pandas as pd # Pour pd.Timedelta et pd.to_datetime
from typing import Optional, Union
import re

logger = logging.getLogger(__name__)

def get_current_utc_timestamp_ms() -> int:
    """
    Obtient le timestamp Unix actuel en millisecondes, en UTC.

    Returns:
        int: Timestamp Unix UTC en millisecondes.
    """
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def get_current_utc_datetime() -> datetime:
    """
    Obtient l'objet datetime actuel, conscient du timezone UTC.

    Returns:
        datetime: Objet datetime actuel en UTC.
    """
    return datetime.now(timezone.utc)

def parse_timeframe_to_timedelta(tf_string: str) -> Optional[pd.Timedelta]:
    """
    Convertit une chaîne de timeframe (ex: "1m", "5min", "1h", "1d")
    en un objet pd.Timedelta.

    Args:
        tf_string (str): La chaîne de timeframe à parser.

    Returns:
        Optional[pd.Timedelta]: L'objet Timedelta correspondant ou None si le format est inconnu.
    """
    if not isinstance(tf_string, str) or not tf_string.strip():
        logger.warning(f"parse_timeframe_to_timedelta: tf_string invalide ou vide: '{tf_string}'")
        return None

    tf_string_lower = tf_string.lower().strip()
    
    # Regex pour capturer le nombre et l'unité (m/min, h, d, s, w)
    match = re.fullmatch(r"(\d+)\s*(m|min|h|d|s|w)", tf_string_lower)
    if not match:
        logger.warning(f"parse_timeframe_to_timedelta: Format de timeframe inconnu pour '{tf_string}'.")
        return None

    value_str, unit_str = match.groups()
    value = int(value_str)

    if unit_str in ("m", "min"):
        return pd.Timedelta(minutes=value)
    elif unit_str == "h":
        return pd.Timedelta(hours=value)
    elif unit_str == "d":
        return pd.Timedelta(days=value)
    elif unit_str == "s":
        return pd.Timedelta(seconds=value)
    elif unit_str == "w":
        return pd.Timedelta(weeks=value)
    else:
        # Ce cas ne devrait pas être atteint à cause du regex, mais par sécurité
        logger.warning(f"parse_timeframe_to_timedelta: Unité de timeframe non gérée '{unit_str}' dans '{tf_string}'.")
        return None

def parse_timeframe_to_seconds(tf_string: str) -> Optional[int]:
    """
    Convertit une chaîne de timeframe en un nombre total de secondes.

    Args:
        tf_string (str): La chaîne de timeframe à parser.

    Returns:
        Optional[int]: Le nombre total de secondes ou None si la conversion échoue.
    """
    timedelta_obj = parse_timeframe_to_timedelta(tf_string)
    if timedelta_obj:
        return int(timedelta_obj.total_seconds())
    return None

def round_datetime_to_minute(dt: datetime, direction: str = "floor") -> datetime:
    """
    Arrondit un objet datetime à la minute la plus proche (floor, ceil, ou round).
    Préserve le timezone de l'objet datetime d'entrée.

    Args:
        dt (datetime): L'objet datetime à arrondir.
        direction (str): Méthode d'arrondi ("floor", "ceil", "round"). Défaut "floor".

    Returns:
        datetime: L'objet datetime arrondi.
    """
    if not isinstance(dt, datetime):
        logger.error(f"round_datetime_to_minute: L'entrée 'dt' doit être un objet datetime. Reçu: {type(dt)}")
        raise TypeError("L'entrée 'dt' doit être un objet datetime.")

    if direction == "floor":
        return dt.replace(second=0, microsecond=0)
    elif direction == "ceil":
        if dt.second > 0 or dt.microsecond > 0:
            return (dt.replace(second=0, microsecond=0) + timedelta(minutes=1))
        return dt.replace(second=0, microsecond=0) # Déjà à la minute
    elif direction == "round":
        if dt.second >= 30:
            return (dt.replace(second=0, microsecond=0) + timedelta(minutes=1))
        return dt.replace(second=0, microsecond=0)
    else:
        logger.warning(f"round_datetime_to_minute: Direction d'arrondi inconnue '{direction}'. Utilisation de 'floor'.")
        return dt.replace(second=0, microsecond=0)

def convert_to_utc_datetime(
    dt_input: Union[str, datetime, int, float],
    unit_if_timestamp: Optional[str] = 'ms'
) -> Optional[datetime]:
    """
    Convertit diverses représentations de temps en un objet datetime conscient du timezone UTC.

    Args:
        dt_input: Peut être une chaîne (ISO format ou formats parsables par pd.to_datetime),
                  un objet datetime existant, ou un timestamp Unix (int/float).
        unit_if_timestamp (Optional[str]): Si dt_input est un nombre, spécifie l'unité
                                            du timestamp ('s' pour secondes, 'ms' pour millisecondes).
                                            Défaut 'ms'.

    Returns:
        Optional[datetime]: Un objet datetime en UTC, ou None en cas d'échec de parsing.
    """
    if isinstance(dt_input, datetime):
        if dt_input.tzinfo is None:
            logger.debug(f"convert_to_utc_datetime: datetime naïf fourni ({dt_input}), en supposant UTC et le rendant conscient.")
            return dt_input.replace(tzinfo=timezone.utc)
        else: # Déjà conscient, convertir en UTC
            return dt_input.astimezone(timezone.utc)
    elif isinstance(dt_input, str):
        try:
            # pd.to_datetime est assez flexible pour parser divers formats ISO et autres.
            # utc=True le convertit directement en UTC.
            dt_obj = pd.to_datetime(dt_input, utc=True, errors='raise')
            if isinstance(dt_obj, pd.Timestamp): # pd.to_datetime retourne Timestamp
                return dt_obj.to_pydatetime() # Convertir en objet datetime standard
            return dt_obj # Si c'est déjà un datetime (moins probable avec errors='raise')
        except Exception as e:
            logger.warning(f"convert_to_utc_datetime: Échec du parsing de la chaîne '{dt_input}' en datetime: {e}")
            return None
    elif isinstance(dt_input, (int, float)):
        if np.isnan(dt_input) or np.isinf(dt_input):
            logger.warning(f"convert_to_utc_datetime: Timestamp Unix invalide (NaN/Inf): {dt_input}")
            return None
        try:
            ts_seconds = dt_input
            if unit_if_timestamp == 'ms':
                ts_seconds = dt_input / 1000.0
            elif unit_if_timestamp != 's':
                logger.warning(f"convert_to_utc_datetime: Unité de timestamp inconnue '{unit_if_timestamp}'. En supposant secondes.")
            
            return datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
        except Exception as e:
            logger.warning(f"convert_to_utc_datetime: Échec de la conversion du timestamp Unix {dt_input} (unité: {unit_if_timestamp}) en datetime: {e}")
            return None
    else:
        logger.warning(f"convert_to_utc_datetime: Type d'entrée non supporté: {type(dt_input)}")
        return None

def format_datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """
    Formate un objet datetime en une chaîne ISO 8601 standard avec 'Z' pour UTC.
    Inclut les millisecondes si présentes et non nulles.

    Args:
        dt (Optional[datetime]): L'objet datetime à formater.

    Returns:
        Optional[str]: La chaîne formatée ISO, ou None si l'entrée est None.
    """
    if dt is None:
        return None
    if not isinstance(dt, datetime):
        logger.warning(f"format_datetime_to_iso: L'entrée n'est pas un objet datetime. Reçu: {type(dt)}")
        return str(dt) # Tenter une conversion str en fallback

    dt_utc = dt
    if dt_utc.tzinfo is None:
        logger.debug(f"format_datetime_to_iso: datetime naïf fourni ({dt_utc}), en supposant UTC.")
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    elif dt_utc.tzinfo != timezone.utc:
        dt_utc = dt_utc.astimezone(timezone.utc)

    # Formater avec millisecondes si elles sont non nulles, sinon sans.
    if dt_utc.microsecond > 0:
        return dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z' # Format avec 3 chiffres pour ms
    else:
        return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

if __name__ == '__main__':
    # Exemples d'utilisation pour tester le module
    logging.basicConfig(level=logging.DEBUG)

    logger.info(f"Timestamp UTC actuel (ms): {get_current_utc_timestamp_ms()}")
    logger.info(f"Datetime UTC actuel: {get_current_utc_datetime()}")
    logger.info(f"Datetime UTC actuel (formaté ISO): {format_datetime_to_iso(get_current_utc_datetime())}")
    
    timeframes = ["5m", "15min", "1h", "4H", "1d", "1w", "30s", "1mon"]
    for tf in timeframes:
        td = parse_timeframe_to_timedelta(tf)
        secs = parse_timeframe_to_seconds(tf)
        logger.info(f"Timeframe '{tf}': Timedelta = {td}, Secondes = {secs}")

    dt_test = datetime(2023, 10, 26, 14, 35, 45, 123456, tzinfo=timezone.utc)
    logger.info(f"Datetime original: {dt_test}")
    logger.info(f"  Arrondi floor: {round_datetime_to_minute(dt_test, 'floor')}")
    logger.info(f"  Arrondi ceil:  {round_datetime_to_minute(dt_test, 'ceil')}")
    logger.info(f"  Arrondi round: {round_datetime_to_minute(dt_test, 'round')}")
    dt_test_on_minute = datetime(2023, 10, 26, 14, 35, 0, 0, tzinfo=timezone.utc)
    logger.info(f"Datetime sur minute: {dt_test_on_minute}")
    logger.info(f"  Arrondi ceil (sur minute): {round_datetime_to_minute(dt_test_on_minute, 'ceil')}")


    logger.info(f"Conversion de string ISO: {convert_to_utc_datetime('2023-10-26T12:30:00Z')}")
    logger.info(f"Conversion de string ISO avec ms: {convert_to_utc_datetime('2023-10-26T12:30:00.500Z')}")
    logger.info(f"Conversion de string (autre format): {convert_to_utc_datetime('2023/10/26 14:30:00 +02:00')}")
    logger.info(f"Conversion de datetime naïf: {convert_to_utc_datetime(datetime(2023,1,1,10,0,0))}")
    logger.info(f"Conversion de datetime conscient (non-UTC): {convert_to_utc_datetime(datetime(2023,1,1,10,0,0, tzinfo=timezone(timedelta(hours=2))))}")
    ts_ms = get_current_utc_timestamp_ms()
    logger.info(f"Conversion de timestamp ms ({ts_ms}): {convert_to_utc_datetime(ts_ms, 'ms')}")
    ts_s = int(ts_ms / 1000)
    logger.info(f"Conversion de timestamp s ({ts_s}): {convert_to_utc_datetime(ts_s, 's')}")
    logger.info(f"Conversion de float timestamp s: {convert_to_utc_datetime(time.time(), 's')}")

    dt_with_ms = datetime.now(timezone.utc)
    dt_without_ms = dt_with_ms.replace(microsecond=0)
    logger.info(f"Format ISO (avec ms): {format_datetime_to_iso(dt_with_ms)}")
    logger.info(f"Format ISO (sans ms): {format_datetime_to_iso(dt_without_ms)}")
    logger.info(f"Format ISO (None): {format_datetime_to_iso(None)}")
