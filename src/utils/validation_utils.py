# Fichier: src/utils/validation_utils.py
"""
Fonctions utilitaires génériques pour la validation de données, types et formats.
"""
import logging
from typing import Any, Optional, List, Union, Type, Pattern, Tuple, Dict
import re
# from numbers import Number # isinstance(value, (int, float)) est généralement suffisant

logger = logging.getLogger(__name__)

def is_valid_type(value: Any, expected_type: Union[Type[Any], Tuple[Type[Any], ...]]) -> bool:
    """
    Vérifie si la valeur est du type attendu (ou l'un des types dans un tuple).

    Args:
        value: La valeur à vérifier.
        expected_type: Le type attendu ou un tuple de types attendus.

    Returns:
        True si la valeur est du type attendu, False sinon.
    """
    if not isinstance(expected_type, (type, tuple)):
        logger.warning(f"is_valid_type: expected_type doit être un type ou un tuple de types. Reçu: {type(expected_type)}")
        # Optionnel: lever une erreur ou retourner False basé sur la criticité
        # For now, let's assume it's a usage error and return False
        return False
    return isinstance(value, expected_type)

def is_within_range(
    value: Any, # Peut être autre chose qu'un nombre, la fonction doit le gérer
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True
) -> bool:
    """
    Vérifie si une valeur numérique est dans une plage spécifiée.

    Args:
        value: La valeur numérique à vérifier.
        min_val: La valeur minimale de la plage (optionnelle).
        max_val: La valeur maximale de la plage (optionnelle).
        inclusive_min: True si la valeur minimale est incluse dans la plage.
        inclusive_max: True si la valeur maximale est incluse dans la plage.

    Returns:
        True si la valeur est dans la plage, False sinon (y compris si la valeur n'est pas numérique).
    """
    if not isinstance(value, (int, float)):
        logger.debug(f"is_within_range: La valeur '{value}' n'est pas un nombre. Validation échouée.")
        return False

    if min_val is not None:
        if inclusive_min:
            if value < min_val:
                return False
        else:
            if value <= min_val:
                return False
    
    if max_val is not None:
        if inclusive_max:
            if value > max_val:
                return False
        else:
            if value >= max_val:
                return False
    return True

def is_not_empty_string(value: Any, allow_whitespace: bool = False) -> bool:
    """
    Vérifie si la valeur est une chaîne non vide.

    Args:
        value: La valeur à vérifier.
        allow_whitespace: Si True, une chaîne contenant uniquement des espaces est considérée comme non vide.
                          Si False, la chaîne est "stripée" avant la vérification.

    Returns:
        True si la valeur est une chaîne non vide (selon allow_whitespace), False sinon.
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
        value: La valeur à vérifier.

    Returns:
        True si la valeur est une liste non vide, False sinon.
    """
    return isinstance(value, list) and bool(value)

def matches_regex(text: Any, pattern: Union[str, Pattern[str]]) -> bool:
    """
    Vérifie si une chaîne correspond (entièrement) à une expression régulière.

    Args:
        text: La chaîne à vérifier.
        pattern: L'expression régulière (chaîne ou objet Pattern compilé).

    Returns:
        True si la chaîne correspond entièrement au motif, False sinon (y compris si text n'est pas une chaîne).
    """
    if not isinstance(text, str):
        return False
    try:
        if isinstance(pattern, str):
            match_object = re.fullmatch(pattern, text)
        elif isinstance(pattern, re.Pattern): # type: ignore # re.Pattern n'est pas reconnu par mypy avant Python 3.7+ avec typing.Pattern
            match_object = pattern.fullmatch(text)
        else:
            logger.warning(f"matches_regex: Type de motif non supporté: {type(pattern)}")
            return False
        return bool(match_object)
    except re.error as e:
        logger.error(f"matches_regex: Erreur de Regex avec le motif '{pattern}': {e}")
        return False


def is_valid_choice(value: Any, choices: List[Any]) -> bool:
    """
    Vérifie si la valeur est l'une des options valides dans une liste de choix.

    Args:
        value: La valeur à vérifier.
        choices: Une liste d'options valides.

    Returns:
        True si la valeur est dans la liste de choix, False sinon.
    """
    if not isinstance(choices, list):
        logger.warning("is_valid_choice: 'choices' doit être une liste.")
        return False # Ou lever une erreur selon la politique de gestion des erreurs
    return value in choices

def are_all_elements_unique(input_list: List[Any]) -> bool:
    """
    Vérifie si tous les éléments d'une liste sont uniques.

    Args:
        input_list: La liste à vérifier.

    Returns:
        True si tous les éléments sont uniques ou si la liste est vide/None.
             False si des doublons existent.
             False si input_list n'est pas une liste.
    """
    if not isinstance(input_list, list):
        logger.debug("are_all_elements_unique: L'entrée n'est pas une liste.")
        return False
    if not input_list: # Liste vide est considérée comme ayant des éléments uniques
        return True
    return len(input_list) == len(set(input_list))

def validate_dict_keys(
    data: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    allowed_keys: Optional[List[str]] = None
) -> List[str]:
    """
    Valide les clés d'un dictionnaire par rapport aux clés requises et/ou autorisées.

    Args:
        data: Le dictionnaire à valider.
        required_keys: Une liste optionnelle de clés qui doivent être présentes.
        allowed_keys: Une liste optionnelle de clés qui sont les seules autorisées.
                      Si fournie, toute clé dans 'data' non présente dans 'allowed_keys'
                      sera considérée comme une erreur.

    Returns:
        Une liste de messages d'erreur (strings). Une liste vide signifie que la validation est réussie.
    """
    if not isinstance(data, dict):
        return ["L'entrée 'data' n'est pas un dictionnaire."]

    errors: List[str] = []
    data_keys_set = set(data.keys())

    if required_keys:
        required_keys_set = set(required_keys)
        missing_keys = list(required_keys_set - data_keys_set)
        if missing_keys:
            errors.append(f"Clés requises manquantes: {', '.join(sorted(missing_keys))}")

    if allowed_keys:
        allowed_keys_set = set(allowed_keys)
        extra_keys = list(data_keys_set - allowed_keys_set)
        if extra_keys:
            errors.append(f"Clés non autorisées trouvées: {', '.join(sorted(extra_keys))}. Clés autorisées: {sorted(list(allowed_keys_set))}")
            
    return errors

# Vous pouvez ajouter d'autres fonctions de validation génériques ici au besoin.
# Par exemple :
# def is_positive_number(value: Any) -> bool:
#     return isinstance(value, (int, float)) and value > 0

# def is_non_negative_number(value: Any) -> bool:
#     return isinstance(value, (int, float)) and value >= 0

# def is_valid_email(email_string: str) -> bool:
#     # Utiliser un regex plus complet pour la validation d'email
#     pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
#     return matches_regex(email_string, pattern)

