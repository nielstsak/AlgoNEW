# src/utils/validation_utils.py
"""
Ce module fournit des fonctions utilitaires génériques pour la validation
de données, de types, de formats, et d'autres conditions courantes.
"""
import logging
import re
from typing import Any, Optional, List, Union, Type, Pattern, Tuple, Dict
import pandas as pd

logger = logging.getLogger(__name__)

# Constante pour un motif regex d'email simple (peut être affiné si nécessaire)
# Source: HTML5 W3C Recommendation (https://html.spec.whatwg.org/multipage/input.html#valid-e-mail-address)
# Ce regex est une simplification commune et peut ne pas couvrir tous les cas extrêmes d'emails valides selon RFC 5322.
# Pour une validation plus stricte, une bibliothèque dédiée pourrait être envisagée.
EMAIL_REGEX_PATTERN = r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"


def is_valid_type(value: Any, expected_type: Union[Type[Any], Tuple[Type[Any], ...]]) -> bool:
    """
    Vérifie si la valeur est du type attendu ou l'un des types dans un tuple.

    Args:
        value (Any): La valeur à vérifier.
        expected_type (Union[Type[Any], Tuple[Type[Any], ...]]): Le type attendu
            (ex: `int`, `str`) ou un tuple de types attendus (ex: `(int, float)`).

    Returns:
        bool: `True` si la valeur est d'un des types attendus, `False` sinon.
              Retourne également `False` si `expected_type` n'est pas un type ou un tuple de types.
    """
    if not isinstance(expected_type, (type, tuple)):
        # Ceci est une erreur d'utilisation de la fonction, donc un log est approprié.
        logger.warning(
            f"is_valid_type: expected_type doit être un type ou un tuple de types. Reçu : {type(expected_type)}"
        )
        return False
    return isinstance(value, expected_type)


def is_within_range(
    value: Any,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True
) -> bool:
    """
    Vérifie si une valeur numérique se situe dans une plage spécifiée.

    Args:
        value (Any): La valeur à vérifier. Doit être un nombre (int ou float)
                     pour que la validation de plage ait un sens.
        min_val (Optional[Union[int, float]]): La valeur minimale de la plage.
                                               Si None, aucune limite inférieure n'est vérifiée.
                                               Par défaut None.
        max_val (Optional[Union[int, float]]): La valeur maximale de la plage.
                                               Si None, aucune limite supérieure n'est vérifiée.
                                               Par défaut None.
        inclusive_min (bool): Si `True`, la valeur minimale est incluse dans la plage (>=).
                              Si `False`, elle est exclue (>). Par défaut `True`.
        inclusive_max (bool): Si `True`, la valeur maximale est incluse dans la plage (<=).
                              Si `False`, elle est exclue (<). Par défaut `True`.

    Returns:
        bool: `True` si la valeur est dans la plage spécifiée (ou si les limites ne
              sont pas définies), `False` sinon (y compris si la valeur n'est pas
              un `int` ou `float`).
    """
    if not isinstance(value, (int, float)):
        # La fonction est conçue pour les nombres ; si ce n'est pas un nombre, ce n'est pas dans la plage.
        # Pas besoin de logger ici car c'est un résultat de validation normal.
        return False

    if min_val is not None:
        if inclusive_min:
            if value < min_val:
                return False
        else: # Exclusif
            if value <= min_val:
                return False

    if max_val is not None:
        if inclusive_max:
            if value > max_val:
                return False
        else: # Exclusif
            if value >= max_val:
                return False
    return True


def is_not_empty_string(value: Any, allow_whitespace: bool = False) -> bool:
    """
    Vérifie si la valeur est une chaîne de caractères non vide.

    Args:
        value (Any): La valeur à vérifier.
        allow_whitespace (bool): Si `False` (par défaut), une chaîne contenant
                                 uniquement des espaces est considérée comme vide
                                 (après `strip()`). Si `True`, une telle chaîne
                                 est considérée comme non vide.

    Returns:
        bool: `True` si la valeur est une chaîne non vide (selon le critère
              `allow_whitespace`), `False` sinon (y compris si `value` n'est pas
              une chaîne).
    """
    if not isinstance(value, str):
        return False

    if not allow_whitespace:
        return bool(value.strip())
    return bool(value)


def is_not_empty_list(value: Any) -> bool:
    """
    Vérifie si la valeur est une liste non vide.

    Args:
        value (Any): La valeur à vérifier.

    Returns:
        bool: `True` si la valeur est une liste et qu'elle contient au moins
              un élément, `False` sinon (y compris si `value` n'est pas une liste).
    """
    return isinstance(value, list) and bool(value)


def matches_regex(text: Any, pattern: Union[str, Pattern[str]]) -> bool:
    """
    Vérifie si une chaîne de caractères correspond (entièrement) à une expression régulière.

    Args:
        text (Any): La chaîne de caractères à vérifier.
        pattern (Union[str, Pattern[str]]): L'expression régulière, soit sous forme
                                            de chaîne, soit un objet `Pattern` compilé.

    Returns:
        bool: `True` si la chaîne `text` correspond entièrement au `pattern`,
              `False` sinon (y compris si `text` n'est pas une chaîne ou si
              le `pattern` est invalide).
    """
    if not isinstance(text, str):
        return False
    try:
        # Utiliser re.fullmatch pour s'assurer que toute la chaîne correspond,
        # pas seulement une partie.
        if isinstance(pattern, str):
            match_object = re.fullmatch(pattern, text)
        elif isinstance(pattern, re.Pattern): # type: ignore # mypy peut avoir du mal avec re.Pattern avant Py3.7+ typing.Pattern
            match_object = pattern.fullmatch(text)
        else:
            logger.warning(
                f"matches_regex: Type de pattern non supporté : {type(pattern)}. "
                "Le pattern doit être une chaîne ou un objet re.Pattern compilé."
            )
            return False
        return bool(match_object)
    except re.error as e:
        # Erreur dans la compilation ou l'utilisation du regex lui-même.
        logger.error(f"matches_regex: Erreur de Regex avec le pattern '{pattern}' sur le texte '{text[:50]}...': {e}")
        return False


def is_valid_choice(value: Any, choices: List[Any]) -> bool:
    """
    Vérifie si la valeur est l'une des options valides dans une liste de choix.

    Args:
        value (Any): La valeur à vérifier.
        choices (List[Any]): Une liste d'options valides.

    Returns:
        bool: `True` si `value` est présente dans `choices`, `False` sinon.
              Retourne `False` si `choices` n'est pas une liste.
    """
    if not isinstance(choices, list):
        logger.warning(f"is_valid_choice: L'argument 'choices' doit être une liste. Reçu : {type(choices)}")
        return False
    return value in choices


def are_all_elements_unique(input_list: List[Any]) -> bool:
    """
    Vérifie si tous les éléments d'une liste sont uniques.
    Les éléments non hashables ne peuvent pas être vérifiés de cette manière et
    provoqueront une TypeError.

    Args:
        input_list (List[Any]): La liste à vérifier.

    Returns:
        bool: `True` si tous les éléments sont uniques ou si la liste est vide.
              `False` s'il y a des doublons.
              Retourne `False` si `input_list` n'est pas une liste.

    Raises:
        TypeError: Si la liste contient des éléments non hashables (ex: des listes imbriquées).
                   L'appelant devrait gérer cette exception si de tels éléments sont possibles.
    """
    if not isinstance(input_list, list):
        # logger.debug("are_all_elements_unique: L'entrée n'est pas une liste.") # Optionnel, car c'est un résultat de validation
        return False
    if not input_list:  # Une liste vide est considérée comme ayant des éléments uniques
        return True
    # La conversion en set puis la comparaison des longueurs est une manière idiomatique
    # de vérifier l'unicité pour les éléments hashables.
    try:
        return len(input_list) == len(set(input_list))
    except TypeError as e:
        logger.error(f"are_all_elements_unique: La liste contient des éléments non hashables, impossible de vérifier l'unicité avec set(). Erreur: {e}")
        raise # Renvoyer l'erreur car la fonction ne peut pas remplir son contrat


def validate_dict_keys(
    data: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    allowed_keys: Optional[List[str]] = None
) -> List[str]:
    """
    Valide les clés d'un dictionnaire par rapport à des listes de clés
    requises et/ou autorisées.

    Args:
        data (Dict[str, Any]): Le dictionnaire à valider.
        required_keys (Optional[List[str]]): Une liste optionnelle de clés qui
                                             doivent être présentes dans `data`.
        allowed_keys (Optional[List[str]]): Une liste optionnelle de clés qui sont
                                            les seules autorisées dans `data`. Si fournie,
                                            toute clé dans `data` non présente dans
                                            `allowed_keys` sera signalée comme une erreur.

    Returns:
        List[str]: Une liste de messages d'erreur (chaînes). Une liste vide
                   signifie que la validation des clés est réussie.
                   Si `data` n'est pas un dictionnaire, la liste contiendra un message d'erreur.
    """
    if not isinstance(data, dict):
        return ["L'entrée 'data' doit être un dictionnaire."]

    errors: List[str] = []
    data_keys_set = set(data.keys())

    if required_keys:
        required_keys_set = set(required_keys)
        missing_keys = sorted(list(required_keys_set - data_keys_set))
        if missing_keys:
            errors.append(f"Clés requises manquantes : {', '.join(missing_keys)}")

    if allowed_keys:
        allowed_keys_set = set(allowed_keys)
        extra_keys = sorted(list(data_keys_set - allowed_keys_set))
        if extra_keys:
            errors.append(
                f"Clés non autorisées trouvées : {', '.join(extra_keys)}. "
                f"Clés autorisées : {sorted(list(allowed_keys_set))}"
            )
    return errors

# --- Nouvelles fonctions suggérées ---

def is_positive_number(value: Any) -> bool:
    """
    Vérifie si la valeur est un nombre (entier ou flottant) strictement positif.

    Args:
        value (Any): La valeur à vérifier.

    Returns:
        bool: `True` si `value` est un nombre et `value > 0`, `False` sinon.
    """
    return isinstance(value, (int, float)) and value > 0


def is_non_negative_number(value: Any) -> bool:
    """
    Vérifie si la valeur est un nombre (entier ou flottant) positif ou nul.

    Args:
        value (Any): La valeur à vérifier.

    Returns:
        bool: `True` si `value` est un nombre et `value >= 0`, `False` sinon.
    """
    return isinstance(value, (int, float)) and value >= 0


def is_valid_email(email_string: Any) -> bool:
    """
    Vérifie si une chaîne de caractères semble être une adresse e-mail valide
    en utilisant une expression régulière.

    Note : Les expressions régulières pour la validation d'e-mails peuvent être
    complexes et ne garantissent pas à 100% la validité ou l'existence
    d'une adresse e-mail. Ce regex est une approximation courante.

    Args:
        email_string (Any): La chaîne de caractères à valider.

    Returns:
        bool: `True` si `email_string` est une chaîne et correspond au motif
              d'e-mail, `False` sinon.
    """
    if not isinstance(email_string, str):
        return False
    return matches_regex(email_string, EMAIL_REGEX_PATTERN)


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str], df_name: str = "DataFrame") -> None:
    """
    Valide qu'un DataFrame contient toutes les colonnes requises.

    Args:
        df (pd.DataFrame): Le DataFrame à valider.
        required_columns (List[str]): La liste des noms de colonnes requis.
        df_name (str): Un nom descriptif pour le DataFrame (utilisé dans les messages d'erreur).

    Raises:
        TypeError: Si df n'est pas un DataFrame pandas.
        ValueError: Si une ou plusieurs colonnes requises sont manquantes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{df_name} doit être un DataFrame pandas. Reçu : {type(df)}")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{df_name} ne contient pas les colonnes requises : {', '.join(missing_columns)}. "
            f"Colonnes disponibles : {', '.join(df.columns.tolist())}"
        )

def validate_non_empty_dataframe(df: pd.DataFrame, df_name: str = "DataFrame") -> None:
    """
    Valide qu'un DataFrame n'est pas None et n'est pas vide.

    Args:
        df (pd.DataFrame): Le DataFrame à valider.
        df_name (str): Un nom descriptif pour le DataFrame (utilisé dans les messages d'erreur).

    Raises:
        TypeError: Si df n'est pas un DataFrame pandas.
        ValueError: Si le DataFrame est None ou vide.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{df_name} doit être un DataFrame pandas. Reçu : {type(df)}")
    if df is None: # Redondant avec le check de type mais explicit
        raise ValueError(f"{df_name} ne peut pas être None.")
    if df.empty:
        raise ValueError(f"{df_name} ne peut pas être vide.")

def validate_series_not_empty_or_all_na(series: pd.Series, series_name: str = "Series") -> None:
    """
    Valide qu'une Series pandas n'est pas None, n'est pas vide, et ne contient pas que des NA.

    Args:
        series (pd.Series): La Series à valider.
        series_name (str): Un nom descriptif pour la Series (utilisé dans les messages d'erreur).

    Raises:
        TypeError: Si series n'est pas une Series pandas.
        ValueError: Si la Series est None, vide, ou ne contient que des valeurs NA.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"{series_name} doit être une Series pandas. Reçu : {type(series)}")
    if series is None: # Redondant
        raise ValueError(f"{series_name} ne peut pas être None.")
    if series.empty:
        raise ValueError(f"{series_name} ne peut pas être vide.")
    if series.isnull().all():
        raise ValueError(f"{series_name} ne doit pas contenir que des valeurs NA.")

