# src/utils/exchange_utils.py
"""
Ce module fournit des fonctions utilitaires pour gérer les filtres, les précisions
(prix, quantité) et les validations spécifiques aux exchanges (par exemple, Binance),
en se basant sur les informations de l'exchange.
"""

import logging
import math
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP, ROUND_CEILING, ROUND_FLOOR, Context as DecimalContext, Inexact, Rounded
from typing import Dict, Optional, Any, Union, List

# numpy n'est pas strictement nécessaire ici si on utilise Decimal partout pour la précision.
# import numpy as np # Pour np.isnan, np.isinf

logger = logging.getLogger(__name__)

# Contexte décimal pour les opérations de haute précision, par exemple 15 décimales.
# Cela peut être ajusté en fonction des besoins de précision maximale rencontrés.
DECIMAL_CONTEXT = DecimalContext(prec=28) # Précision suffisante pour la plupart des cryptos

# Une petite valeur epsilon pour les comparaisons de flottants généraux, si nécessaire.
# Pour les vérifications de multiples de tick/step, Decimal est privilégié.
FLOAT_COMPARISON_EPSILON = Decimal('1e-9')

def get_pair_config_for_symbol(
    pair_symbol: str,
    exchange_info_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extrait la configuration spécifique d'une paire (symbole) à partir des
    données d'information de l'exchange (préalablement chargées).

    Args:
        pair_symbol (str): Le symbole de la paire à rechercher (ex: "BTCUSDT").
                           La casse est ignorée pour la comparaison.
        exchange_info_data (Dict[str, Any]): Un dictionnaire contenant les informations
            de l'exchange, typiquement la sortie de `load_exchange_config` ou de
            l'API `get_exchange_info()`. Doit contenir une clé "symbols" qui est
            une liste de dictionnaires, chaque dictionnaire représentant un symbole.

    Returns:
        Optional[Dict[str, Any]]: Un dictionnaire contenant la configuration de la paire
                                  si trouvée, sinon None.
    """
    log_prefix = f"[GetPairCfg({pair_symbol.upper()})]"
    if not isinstance(exchange_info_data, dict):
        logger.error(f"{log_prefix} exchange_info_data n'est pas un dictionnaire. Reçu : {type(exchange_info_data)}")
        return None

    symbols_list = exchange_info_data.get("symbols")
    if not isinstance(symbols_list, list):
        logger.error(f"{log_prefix} La clé 'symbols' est manquante ou n'est pas une liste dans exchange_info_data.")
        return None

    pair_symbol_upper = pair_symbol.upper()
    for symbol_details in symbols_list:
        if isinstance(symbol_details, dict) and symbol_details.get("symbol", "").upper() == pair_symbol_upper:
            logger.debug(f"{log_prefix} Configuration trouvée pour la paire.")
            return symbol_details

    logger.warning(f"{log_prefix} Aucune configuration trouvée pour la paire dans les données de l'exchange fournies.")
    return None


def adjust_precision(
    value: Optional[Union[float, str, Decimal]],
    precision: Optional[int] = None,
    rounding_mode_str: str = "ROUND_HALF_UP", # Chaîne correspondant aux constantes Decimal.ROUND_*
    tick_size: Optional[Union[float, str, Decimal]] = None
) -> Optional[float]:
    """
    Ajuste une valeur numérique à une précision décimale spécifiée ou à un multiple
    d'un `tick_size` donné. Utilise la classe `Decimal` pour une meilleure précision.

    Si `tick_size` est fourni, il a la priorité sur `precision` pour l'ajustement.
    La valeur sera ajustée pour être un multiple du `tick_size`.

    Args:
        value (Optional[Union[float, str, Decimal]]): La valeur à ajuster.
        precision (Optional[int]): Le nombre de décimales souhaité si `tick_size`
            n'est pas utilisé. Ignoré si `tick_size` est fourni.
        rounding_mode_str (str): La méthode d'arrondi à utiliser (ex: "ROUND_FLOOR",
            "ROUND_HALF_UP"). Doit correspondre à une constante de la classe Decimal.
        tick_size (Optional[Union[float, str, Decimal]]): La taille du pas (tick).
            Si fourni, la valeur sera ajustée pour être un multiple de cette taille.

    Returns:
        Optional[float]: La valeur ajustée en float, ou None si l'entrée `value`
                         est None ou ne peut pas être convertie en Decimal.
                         Retourne la valeur originale (convertie en float) si NaN ou Inf.
    """
    if value is None:
        return None

    try:
        # Utiliser le contexte décimal pour contrôler la précision des calculs intermédiaires
        with DECIMAL_CONTEXT:
            value_decimal = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        logger.warning(f"adjust_precision: Impossible de convertir la valeur '{value}' en Decimal.")
        return None

    if value_decimal.is_nan() or value_decimal.is_infinite():
        logger.debug(f"adjust_precision: La valeur est NaN ou Inf ({value_decimal}). Retour de la valeur float originale.")
        return float(value)

    actual_rounding_mode = getattr(Decimal, rounding_mode_str.upper(), None)
    if actual_rounding_mode is None:
        logger.warning(f"adjust_precision: Mode d'arrondi '{rounding_mode_str}' invalide. Utilisation de ROUND_HALF_UP par défaut.")
        actual_rounding_mode = ROUND_HALF_UP

    if tick_size is not None:
        try:
            with DECIMAL_CONTEXT:
                tick_size_decimal = Decimal(str(tick_size))
                if tick_size_decimal.is_zero() or not tick_size_decimal.is_finite():
                    logger.warning(f"adjust_precision: tick_size '{tick_size}' est zéro ou non fini. "
                                   "L'ajustement par tick_size est impossible. Utilisation de la précision si fournie, sinon retour de la valeur originale.")
                    # Fallback à la logique de précision si tick_size est invalide
                    if precision is not None and isinstance(precision, int) and precision >= 0:
                        quantizer = Decimal('1e-' + str(precision))
                        adjusted_value_decimal = value_decimal.quantize(quantizer, rounding=actual_rounding_mode)
                        return float(adjusted_value_decimal)
                    return float(value_decimal) # Retourner la valeur originale si pas de précision non plus

                # (valeur / tick_size) -> arrondir à l'entier -> multiplier par tick_size
                adjusted_value_decimal = (value_decimal / tick_size_decimal).quantize(Decimal('1'), rounding=actual_rounding_mode) * tick_size_decimal
                return float(adjusted_value_decimal)
        except (InvalidOperation, ValueError, TypeError) as e_tick:
            logger.error(f"adjust_precision: Erreur lors de l'ajustement avec tick_size Decimal ('{tick_size}'): {e_tick}. "
                         "Retour de la valeur originale (float).")
            return float(value_decimal)

    if precision is None or not isinstance(precision, int) or precision < 0:
        logger.debug(f"adjust_precision: Précision invalide ou manquante ({precision}) et pas de tick_size. "
                     f"Retour de la valeur originale (float) : {float(value_decimal)}.")
        return float(value_decimal)

    quantizer = Decimal('1e-' + str(precision))
    try:
        with DECIMAL_CONTEXT: # Assurer que quantize utilise aussi le contexte de haute précision
            adjusted_value_decimal = value_decimal.quantize(quantizer, rounding=actual_rounding_mode)
        return float(adjusted_value_decimal)
    except (OverflowError, Inexact, Rounded) as e_quantize: # Inexact et Rounded sont des signaux Decimal, pas toujours des erreurs
        if isinstance(e_quantize, OverflowError):
            logger.error(f"adjust_precision: OverflowError lors de l'ajustement de {value_decimal} avec précision {precision}. "
                         "Retour de la valeur originale (float).")
            return float(value_decimal)
        # Pour Inexact ou Rounded, le résultat de quantize est généralement ce qu'on veut.
        # Si value_decimal.quantize a déjà été assigné, on peut le retourner.
        # Cependant, si l'exception est levée avant l'assignation, il faut être prudent.
        # Le code actuel assigne après quantize, donc si quantize lève une exception non gérée ici,
        # cela signifie un problème plus profond.
        # Le bloc try/except global gérera les autres exceptions Decimal.
        # Pour l'instant, on logue et on retourne la valeur quantifiée si disponible.
        logger.debug(f"adjust_precision: Signal Decimal '{type(e_quantize).__name__}' lors de quantize pour {value_decimal} à {precision} décimales. "
                     "Le résultat de quantize sera utilisé s'il est disponible.")
        # Si quantize a réussi mais a signalé Inexact/Rounded, adjusted_value_decimal devrait être correct.
        # Si quantize a échoué plus fondamentalement, le except externe le prendra.
        # Pour être sûr, on recalcule dans le bloc except si nécessaire, ou on retourne la valeur originale.
        # Ici, on suppose que si on atteint ce point, c'est une "exception" de signal Decimal, pas une erreur fatale de quantize.
        # La valeur ajustée est déjà dans adjusted_value_decimal si l'exception n'est pas OverflowError.
        # On va supposer que si on arrive ici, c'est que `quantize` a quand même produit un résultat.
        # Pour être plus sûr, on pourrait refaire le quantize dans un try/except plus fin.
        # Mais pour l'instant, on va considérer que si ce n'est pas Overflow, le résultat est utilisable.
        # Le code actuel ne stocke pas le résultat de quantize avant de le retourner, donc on doit le refaire.
        try:
            with DECIMAL_CONTEXT:
                adjusted_value_final_attempt = value_decimal.quantize(quantizer, rounding=actual_rounding_mode)
            return float(adjusted_value_final_attempt)
        except Exception: # Si même la tentative finale échoue
             logger.error(f"adjust_precision: Échec final de quantize après signal {type(e_quantize).__name__}. Retour de la valeur originale.")
             return float(value_decimal)

    except Exception as e_prec: # pylint: disable=broad-except
        logger.error(f"adjust_precision: Erreur inattendue lors de l'ajustement à la précision {precision}: {e_prec}. "
                      "Retour de la valeur originale (float).")
        return float(value_decimal)


def get_precision_from_filter(symbol_info: Dict[str, Any], filter_type: str, filter_key: str) -> Optional[int]:
    """
    Extrait la précision (nombre de décimales) d'un filtre de symbole
    en se basant sur la valeur de `tickSize` ou `stepSize` (qui est une chaîne).

    Args:
        symbol_info (Dict[str, Any]): Dictionnaire des informations du symbole de l'exchange.
        filter_type (str): Le type de filtre (ex: 'LOT_SIZE', 'PRICE_FILTER').
        filter_key (str): La clé dans le filtre contenant la valeur de précision
                          (typiquement 'stepSize' ou 'tickSize').

    Returns:
        Optional[int]: Le nombre de décimales, ou None si non trouvé, invalide,
                       ou si le tick/step est zéro.
    """
    log_prefix = f"[GetPrecision({symbol_info.get('symbol', 'N/A')}/{filter_type}/{filter_key})]"
    try:
        if not isinstance(symbol_info, dict):
            logger.warning(f"{log_prefix} symbol_info n'est pas un dictionnaire.")
            return None
        filters = symbol_info.get('filters', [])
        if not isinstance(filters, list):
            logger.warning(f"{log_prefix} 'filters' n'est pas une liste ou est manquant dans symbol_info.")
            return None

        target_filter = next((f for f in filters if isinstance(f, dict) and f.get('filterType') == filter_type), None)
        if not target_filter:
            logger.debug(f"{log_prefix} Filtre de type '{filter_type}' non trouvé.")
            return None

        size_str_raw = target_filter.get(filter_key)
        if not isinstance(size_str_raw, str) or not size_str_raw.strip():
            logger.warning(f"{log_prefix} Clé '{filter_key}' non trouvée, vide ou non-string dans le filtre '{filter_type}'. Valeur: '{size_str_raw}'")
            return None
        
        size_str_cleaned = size_str_raw.strip()

        try:
            d_value = Decimal(size_str_cleaned)
        except InvalidOperation:
            logger.warning(f"{log_prefix} Impossible de convertir '{filter_key}' ('{size_str_cleaned}') en Decimal.")
            return None

        if d_value.is_zero() or not d_value.is_finite():
            logger.warning(f"{log_prefix} {filter_key} '{size_str_cleaned}' est zéro, NaN ou Inf. Précision indéfinie.")
            return None
        
        # La méthode as_tuple().exponent donne le nombre de chiffres après la virgule pour les nombres < 1
        # et 0 pour les entiers. Pour les nombres comme "1.0", "1.00", on se base sur la chaîne.
        # Si la chaîne contient un '.', la précision est le nombre de chiffres après.
        # Sinon (ex: "1", "100"), la précision est 0.
        # Pour "1E-5", l'exposant est -5, donc précision 5.
        if '.' in size_str_cleaned:
            # Compter les chiffres après le point, sans enlever les zéros de fin.
            return len(size_str_cleaned.split('.')[-1])
        else: # Pas de point décimal dans la chaîne originale (ex: "1", "1000", ou "1e-5")
            # Utiliser l'exposant du Decimal normalisé.
            # d_value.normalize() enlève les zéros de fin (ex: Decimal('1.00') -> Decimal('1')).
            # as_tuple().exponent pour Decimal('1') est 0. Pour Decimal('0.1') est -1. Pour Decimal('0.001') est -3.
            exponent = d_value.normalize().as_tuple().exponent
            if isinstance(exponent, int):
                return abs(exponent) if exponent < 0 else 0
            else: # Should not happen if d_value is finite and non-zero
                logger.warning(f"{log_prefix} Exponent non entier pour Decimal normalisé de '{size_str_cleaned}'.")
                return None

    except Exception as e: # pylint: disable=broad-except
        logger.error(f"{log_prefix} Erreur inattendue : {e}", exc_info=True)
        return None


def get_filter_value(symbol_info: Dict[str, Any], filter_type: str, filter_key: str) -> Optional[float]:
    """
    Extrait une valeur numérique spécifique (convertie en float) d'un filtre de symbole.

    Args:
        symbol_info (Dict[str, Any]): Dictionnaire contenant les informations du symbole.
        filter_type (str): Le type de filtre (ex: 'LOT_SIZE', 'MIN_NOTIONAL').
        filter_key (str): La clé dans le filtre contenant la valeur à extraire.

    Returns:
        Optional[float]: La valeur flottante du filtre si trouvée et convertible, None sinon.
    """
    log_prefix = f"[GetFilterVal({symbol_info.get('symbol', 'N/A')}/{filter_type}/{filter_key})]"
    try:
        if not isinstance(symbol_info, dict):
            logger.warning(f"{log_prefix} symbol_info n'est pas un dictionnaire.")
            return None
        filters = symbol_info.get('filters', [])
        if not isinstance(filters, list):
            logger.warning(f"{log_prefix} 'filters' n'est pas une liste ou est manquant.")
            return None

        target_filter = next((f for f in filters if isinstance(f,dict) and f.get('filterType') == filter_type), None)
        if target_filter:
            value_raw = target_filter.get(filter_key)
            if value_raw is not None: # Accepter 0 ou "0"
                try:
                    return float(value_raw)
                except (ValueError, TypeError):
                     logger.warning(f"{log_prefix} Impossible de convertir '{filter_key}' ('{value_raw}') en float.")
                     return None
        return None
    except Exception as e:
        logger.error(f"{log_prefix} Erreur inattendue : {e}", exc_info=True)
        return None

def adjust_quantity_to_step_size(
    quantity: float,
    symbol_info: Dict[str, Any],
    # qty_precision: Optional[int], # La précision est implicite dans le step_size
    rounding_mode_str: str = "ROUND_FLOOR" # Typiquement ROUND_FLOOR pour les quantités
) -> float:
    """
    Ajuste une quantité pour qu'elle soit un multiple de `stepSize` du filtre `LOT_SIZE`.
    Utilise typiquement `ROUND_FLOOR` pour les quantités afin d'être conservateur.

    Args:
        quantity (float): La quantité à ajuster.
        symbol_info (Dict[str, Any]): Informations du symbole de l'exchange.
        rounding_mode_str (str): Méthode d'arrondi (ex: "ROUND_FLOOR").

    Returns:
        float: La quantité ajustée. Retourne la quantité originale arrondie à une
               précision par défaut si `stepSize` n'est pas trouvé/valide ou si une erreur se produit.
    """
    log_prefix = f"[AdjustQtyToStep][{symbol_info.get('symbol', 'N/A')}]"
    step_size_val = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize')

    if step_size_val is None or step_size_val <= 0:
        default_precision_for_qty = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize') or 8
        logger.warning(f"{log_prefix} stepSize non trouvé ou invalide ({step_size_val}). "
                       f"Retour de la quantité originale arrondie à {default_precision_for_qty} décimales.")
        # Utiliser adjust_precision avec la précision si step_size est invalide
        adjusted_qty_fallback = adjust_precision(quantity, default_precision_for_qty, rounding_mode_str)
        return adjusted_qty_fallback if adjusted_qty_fallback is not None else round(quantity, default_precision_for_qty)


    adjusted_quantity_float = adjust_precision(
        value=quantity,
        precision=None, # La précision est déterminée par le step_size
        rounding_mode_str=rounding_mode_str,
        tick_size=step_size_val # step_size agit comme un tick_size pour la quantité
    )
    
    final_quantity = adjusted_quantity_float if adjusted_quantity_float is not None else quantity
    if adjusted_quantity_float is None:
         logger.warning(f"{log_prefix} adjust_precision a retourné None pour la quantité. "
                        f"Quantité originale: {quantity}, StepSize: {step_size_val}. "
                        "Retour de la quantité originale non modifiée.")
    else:
        logger.debug(f"{log_prefix} Quantité brute: {quantity:.8f}, StepSize: {step_size_val}, "
                     f"Mode d'arrondi: {rounding_mode_str}, Quantité ajustée: {final_quantity:.8f}")

    return final_quantity


def validate_notional(
    quantity: float,
    price: float,
    symbol_info: Dict[str, Any],
    pair_symbol_for_log: Optional[str] = None
) -> bool:
    """
    Valide si la valeur notionnelle d'un ordre respecte le filtre MIN_NOTIONAL.
    """
    log_sym = pair_symbol_for_log or symbol_info.get('symbol', 'N/A_SYMBOL')
    log_prefix = f"[{log_sym}][ValidateNotional]"

    try:
        q_decimal = Decimal(str(quantity))
        p_decimal = Decimal(str(price))
        if q_decimal.is_nan() or q_decimal.is_infinite() or p_decimal.is_nan() or p_decimal.is_infinite():
            logger.warning(f"{log_prefix} Quantité ({quantity}) ou prix ({price}) invalide (NaN/Inf).")
            return False
    except InvalidOperation:
        logger.warning(f"{log_prefix} Quantité ({quantity}) ou prix ({price}) non convertible en Decimal.")
        return False

    min_notional_val = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
    if min_notional_val is None: # Essayer le filtre NOTIONAL comme fallback
        min_notional_val = get_filter_value(symbol_info, 'NOTIONAL', 'minNotional')

    if min_notional_val is not None and min_notional_val > 0:
        min_notional_decimal = Decimal(str(min_notional_val))
        current_notional = q_decimal.copy_abs() * p_decimal.copy_abs() # abs(q*p)

        if current_notional < min_notional_decimal - FLOAT_COMPARISON_EPSILON: # Comparaison avec tolérance
            logger.debug(f"{log_prefix} Échec : Notionnel actuel {current_notional} < Min requis {min_notional_decimal}")
            return False
        logger.debug(f"{log_prefix} Succès : Notionnel actuel {current_notional} >= Min requis {min_notional_decimal}")
        return True

    logger.debug(f"{log_prefix} Filtre MIN_NOTIONAL/NOTIONAL non trouvé ou non applicable. Validation passée.")
    return True


def validate_order_parameters(
    order_params: Dict[str, Any],
    symbol_info: Dict[str, Any],
    pair_symbol_for_log: Optional[str] = None,
    estimated_market_price_for_market_order: Optional[float] = None
) -> List[str]:
    """
    Valide les paramètres d'un ordre par rapport aux filtres de l'exchange.
    Utilise Decimal pour une meilleure précision dans les comparaisons.
    """
    log_sym = pair_symbol_for_log or symbol_info.get('symbol', 'N/A_SYMBOL')
    log_prefix = f"[{log_sym}][ValidateOrderParams]"
    errors: List[str] = []

    quantity_input = order_params.get('quantity')
    if quantity_input is None:
        errors.append("Paramètre 'quantity' manquant.")
        return errors

    try:
        quantity_decimal = Decimal(str(quantity_input))
        if quantity_decimal.is_nan() or quantity_decimal.is_infinite() or quantity_decimal <= Decimal(0):
            errors.append(f"Quantité invalide '{quantity_input}'. Doit être positive.")
            return errors
    except InvalidOperation:
        errors.append(f"Quantité '{quantity_input}' n'est pas un nombre valide.")
        return errors

    # Validation LOT_SIZE
    min_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
    max_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'maxQty')
    step_size_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize')

    if min_qty_filter is not None and quantity_decimal < Decimal(str(min_qty_filter)) - FLOAT_COMPARISON_EPSILON:
        errors.append(f"Quantité {quantity_decimal} < minQty requis {min_qty_filter}.")
    if max_qty_filter is not None and quantity_decimal > Decimal(str(max_qty_filter)) + FLOAT_COMPARISON_EPSILON:
        errors.append(f"Quantité {quantity_decimal} > maxQty requis {max_qty_filter}.")

    if step_size_qty_filter is not None and step_size_qty_filter > 0:
        step_size_decimal = Decimal(str(step_size_qty_filter))
        # (quantity - minQty) % stepSize == 0.  Si minQty = 0, alors quantity % stepSize == 0.
        # Pour la robustesse, on vérifie si (quantity / stepSize) est un entier à une petite tolérance près.
        if not (quantity_decimal / step_size_decimal).quantize(FLOAT_COMPARISON_EPSILON).is_zero() and \
           abs((quantity_decimal / step_size_decimal) - (quantity_decimal / step_size_decimal).to_integral_value(rounding=ROUND_HALF_UP)) > FLOAT_COMPARISON_EPSILON:
            errors.append(f"Quantité {quantity_decimal} ne respecte pas stepSize {step_size_decimal}.")

    # Validation PRICE_FILTER (si ordre non MARKET et prix fourni)
    order_type = str(order_params.get('type', "LIMIT")).upper()
    price_input = order_params.get('price')
    price_decimal: Optional[Decimal] = None

    if order_type != "MARKET" and price_input is not None:
        try:
            price_decimal = Decimal(str(price_input))
            if price_decimal.is_nan() or price_decimal.is_infinite() or price_decimal <= Decimal(0):
                errors.append(f"Prix '{price_input}' invalide. Doit être positif.")
                price_decimal = None
        except InvalidOperation:
            errors.append(f"Prix '{price_input}' n'est pas un nombre valide.")
            price_decimal = None

        if price_decimal is not None:
            min_price_filter = get_filter_value(symbol_info, 'PRICE_FILTER', 'minPrice')
            max_price_filter = get_filter_value(symbol_info, 'PRICE_FILTER', 'maxPrice')
            tick_size_price_filter = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')

            if min_price_filter is not None and price_decimal < Decimal(str(min_price_filter)) - FLOAT_COMPARISON_EPSILON:
                errors.append(f"Prix {price_decimal} < minPrice requis {min_price_filter}.")
            if max_price_filter is not None and max_price_filter > 0 and price_decimal > Decimal(str(max_price_filter)) + FLOAT_COMPARISON_EPSILON:
                errors.append(f"Prix {price_decimal} > maxPrice requis {max_price_filter}.")
            if tick_size_price_filter is not None and tick_size_price_filter > 0:
                tick_size_decimal = Decimal(str(tick_size_price_filter))
                if not (price_decimal / tick_size_decimal).quantize(FLOAT_COMPARISON_EPSILON).is_zero() and \
                   abs((price_decimal / tick_size_decimal) - (price_decimal / tick_size_decimal).to_integral_value(rounding=ROUND_HALF_UP)) > FLOAT_COMPARISON_EPSILON:
                    errors.append(f"Prix {price_decimal} ne respecte pas tickSize {tick_size_decimal}.")

    # Validation MIN_NOTIONAL
    price_for_notional_check: Optional[float] = None
    if order_type == "MARKET":
        price_for_notional_check = estimated_market_price_for_market_order
        if price_for_notional_check is None:
            logger.debug(f"{log_prefix} Prix de marché estimé non fourni pour ordre MARKET. Validation MIN_NOTIONAL sautée/imprécise.")
    elif price_decimal is not None:
        price_for_notional_check = float(price_decimal)

    if price_for_notional_check is not None:
        if not validate_notional(float(quantity_decimal), price_for_notional_check, symbol_info, log_sym):
            min_notional_report = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional') or get_filter_value(symbol_info, 'NOTIONAL', 'minNotional')
            errors.append(f"Valeur notionnelle ({abs(float(quantity_decimal) * price_for_notional_check):.4f}) "
                          f"< MIN_NOTIONAL/NOTIONAL requis ({min_notional_report if min_notional_report else 'N/A'}).")

    if errors:
        logger.warning(f"{log_prefix} Échecs de validation de l'ordre : {'; '.join(errors)}")
    else:
        logger.debug(f"{log_prefix} Validation de l'ordre réussie.")
    return errors
