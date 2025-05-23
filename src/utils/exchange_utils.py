# Fichier: src/utils/exchange_utils.py
"""
Utilitaires pour gérer les filtres, précisions et validations
spécifiques aux exchanges (ex: Binance).
"""

import logging
import math
import numpy as np
from typing import Dict, Optional, Any, Union, List
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

# Une petite valeur epsilon pour les comparaisons de flottants, si nécessaire.
# Pour les vérifications de multiples de tick/step, il est souvent préférable d'utiliser Decimal.
FLOAT_COMPARISON_EPSILON = 1e-9

def adjust_precision(
    value: Optional[Union[float, str]],
    precision: Optional[int],
    rounding_method: Callable[[float], float] = math.floor, # type: ignore
    tick_size: Optional[Union[float, str]] = None
) -> Optional[float]:
    """
    Ajuste une valeur numérique à une précision spécifiée ou à un multiple d'un tick_size.

    Si tick_size est fourni, il a la priorité sur 'precision' pour l'ajustement.
    Le prix sera ajusté pour être un multiple du tick_size en utilisant la méthode d'arrondi.

    Args:
        value: La valeur à ajuster (peut être float, int, ou str convertible).
        precision: Le nombre de décimales souhaité si tick_size n'est pas utilisé.
                   Ignoré si tick_size est fourni.
        rounding_method: La méthode d'arrondi (math.floor, math.ceil, round).
                         Par défaut math.floor (conservateur pour les quantités).
        tick_size: La taille du tick. Si fourni, la valeur sera ajustée pour être
                   un multiple de cette taille.

    Returns:
        La valeur ajustée en float, ou None si l'entrée est invalide.
    """
    if value is None:
        return None
    
    try:
        val_float = float(value)
    except (ValueError, TypeError):
        logger.warning(f"adjust_precision: Impossible de convertir la valeur '{value}' en float.")
        return None

    if np.isnan(val_float) or np.isinf(val_float):
        logger.debug(f"adjust_precision: La valeur est NaN ou Inf ({val_float}). Retour de la valeur telle quelle.")
        return val_float # Retourner NaN/Inf sans modification

    if tick_size is not None:
        try:
            ts_decimal = Decimal(str(tick_size))
            val_decimal = Decimal(str(val_float))
            
            if ts_decimal == Decimal(0):
                logger.warning("adjust_precision: tick_size est zéro. Retour de la valeur originale.")
                return val_float

            # multiplier = val_decimal / ts_decimal  -> Decimal('xxx.yyy')
            # rounded_multiplier = multiplier.to_integral_value(rounding=...)
            # Cela dépend de la version de Python pour `rounding`.
            # Plus simple :
            # Pour floor: floor(val / tick) * tick
            # Pour ceil: ceil(val / tick) * tick
            # Pour round: round(val / tick) * tick
            
            if rounding_method == math.floor:
                adjusted_val_decimal = math.floor(val_decimal / ts_decimal) * ts_decimal
            elif rounding_method == math.ceil:
                adjusted_val_decimal = math.ceil(val_decimal / ts_decimal) * ts_decimal
            elif rounding_method == round:
                adjusted_val_decimal = round(val_decimal / ts_decimal) * ts_decimal
            else: # Méthode d'arrondi inconnue, fallback
                logger.warning(f"adjust_precision: Méthode d'arrondi inconnue. Utilisation de 'round'.")
                adjusted_val_decimal = round(val_decimal / ts_decimal) * ts_decimal
            
            return float(adjusted_val_decimal)

        except (InvalidOperation, ValueError, TypeError) as e_decimal:
            logger.error(f"adjust_precision: Erreur lors de l'ajustement avec tick_size Decimal ('{tick_size}'): {e_decimal}. "
                         f"Retour de la valeur originale après arrondi standard à 8 décimales.")
            return round(val_float, 8) # Fallback simple

    # Si tick_size n'est pas fourni, utiliser la 'precision' (nombre de décimales)
    if precision is None or not isinstance(precision, int) or precision < 0:
        logger.debug(f"adjust_precision: Précision invalide ou manquante ({precision}) et pas de tick_size. "
                     f"Retour de la valeur originale: {val_float}.")
        return val_float # Retourner la valeur originale si la précision est invalide

    if precision == 0:
        return float(rounding_method(val_float))

    factor = 10**precision
    try:
        # Appliquer la méthode d'arrondi spécifiée
        if rounding_method == math.floor:
            return math.floor(val_float * factor) / factor
        elif rounding_method == math.ceil:
            return math.ceil(val_float * factor) / factor
        elif rounding_method == round: # Python's round (arrondi standard à la décimale la plus proche)
            return round(val_float, precision)
        else: # Fallback si une méthode inconnue est passée (ne devrait pas arriver avec le typage)
            logger.warning(f"adjust_precision: Méthode d'arrondi non reconnue. Utilisation de 'round'.")
            return round(val_float, precision)
    except OverflowError:
        logger.error(f"adjust_precision: OverflowError lors de l'ajustement de {val_float} avec précision {precision}. "
                     "Retour de la valeur originale.")
        return val_float


def get_precision_from_filter(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[int]:
    """
    Extrait la précision (nombre de décimales) d'un filtre de symbole
    en utilisant la représentation string du tickSize ou stepSize.
    Ex: "0.001" -> 3; "1.0" -> 0; "1e-5" -> 5.

    Args:
        symbol_info: Dictionnaire des informations du symbole de l'exchange.
        filter_type: Le type de filtre (ex: 'LOT_SIZE', 'PRICE_FILTER').
        filter_key: La clé dans le filtre contenant la valeur de précision (ex: 'stepSize', 'tickSize').

    Returns:
        Le nombre de décimales, ou None si non trouvé ou invalide.
    """
    log_prefix = "[get_precision_from_filter]"
    try:
        if not isinstance(symbol_info, dict):
            logger.warning(f"{log_prefix} symbol_info n'est pas un dictionnaire.")
            return None
        filters = symbol_info.get('filters', [])
        if not isinstance(filters, list):
            logger.warning(f"{log_prefix} 'filters' n'est pas une liste dans symbol_info.")
            return None

        target_filter = next((f for f in filters if isinstance(f, dict) and f.get('filterType') == filter_type), None)
        if not target_filter:
            logger.debug(f"{log_prefix} Filtre de type '{filter_type}' non trouvé.")
            return None
            
        size_str = target_filter.get(filter_key)
        if not isinstance(size_str, str) or not size_str.strip():
            logger.warning(f"{log_prefix} Clé '{filter_key}' non trouvée ou vide dans le filtre '{filter_type}'. Valeur: '{size_str}'")
            return None

        size_str_cleaned = size_str.strip()
        
        # Utiliser Decimal pour une détermination robuste de la précision
        try:
            d_value = Decimal(size_str_cleaned)
            if d_value.is_zero(): # Un tick/step de zéro n'est pas valide pour la précision
                logger.warning(f"{log_prefix} {filter_key} '{size_str_cleaned}' est zéro, précision indéfinie.")
                return None

            exponent = d_value.as_tuple().exponent
            if isinstance(exponent, int): # Pour les nombres normaux, l'exposant est négatif ou zéro
                # Le nombre de décimales est -exponent pour les valeurs < 1 (ex: 0.01 -> exp -2 -> prec 2)
                # Pour les entiers (1, 10), exponent est 0, precision est 0.
                # Pour "1.0", exponent est -1, precision est 1 (pour formater "X.0")
                # Si on veut le nombre de chiffres *après la virgule pour la mise en forme*:
                if '.' in size_str_cleaned: # Utiliser la string originale pour "1.0" vs "1"
                    return len(size_str_cleaned.split('.')[-1].rstrip('0')) if size_str_cleaned.split('.')[-1].rstrip('0') else 0
                else: # Pas de point décimal dans la string originale (ex: "1", "100")
                    return 0

            # Si l'exposant n'est pas un int (ex: pour 'NaN', 'Infinity'), Decimal lève une erreur avant
            logger.warning(f"{log_prefix} Impossible de déterminer la précision à partir de l'exposant Decimal pour '{size_str_cleaned}'.")
            return None

        except InvalidOperation: # Erreur de conversion en Decimal
            logger.warning(f"{log_prefix} Impossible de convertir '{filter_key}' ('{size_str_cleaned}') en Decimal pour le filtre {filter_type}.")
            return None
            
    except Exception as e:
        logger.error(f"{log_prefix} Erreur inattendue dans get_precision_from_filter ({filter_type}/{filter_key}): {e}", exc_info=True)
        return None

def get_filter_value(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[float]:
    """
    Extrait une valeur numérique spécifique d'un filtre de symbole.

    Args:
        symbol_info: Dictionnaire contenant les informations du symbole.
        filter_type: Le type de filtre (ex: 'LOT_SIZE', 'MIN_NOTIONAL').
        filter_key: La clé dans le filtre contenant la valeur (ex: 'minQty', 'minNotional').

    Returns:
        La valeur flottante du filtre, ou None si non trouvé ou invalide.
    """
    log_prefix = "[get_filter_value]"
    try:
        if not isinstance(symbol_info, dict): return None
        filters = symbol_info.get('filters', [])
        if not isinstance(filters, list): return None

        target_filter = next((f for f in filters if isinstance(f,dict) and f.get('filterType') == filter_type), None)
        if target_filter:
            value_str = target_filter.get(filter_key)
            if value_str is not None: # Accepter 0
                try:
                    return float(value_str)
                except (ValueError, TypeError):
                     logger.warning(f"{log_prefix} Impossible de convertir '{filter_key}' ('{value_str}') en float pour le filtre {filter_type}.")
                     return None
        logger.debug(f"{log_prefix} Filtre '{filter_type}' ou clé '{filter_key}' non trouvé.")
        return None
    except Exception as e:
        logger.error(f"{log_prefix} Erreur inattendue dans get_filter_value ({filter_type}/{filter_key}): {e}", exc_info=True)
        return None

def validate_notional(quantity: float, price: float, symbol_info: dict, pair_symbol_for_log: Optional[str] = None) -> bool:
    """
    Valide si la valeur notionnelle d'un ordre (quantité * prix) respecte
    le filtre MIN_NOTIONAL (ou NOTIONAL) de l'exchange.

    Args:
        quantity: La quantité de l'ordre.
        price: Le prix de l'ordre.
        symbol_info: Dictionnaire des informations du symbole de l'exchange.
        pair_symbol_for_log: Symbole de la paire pour des logs plus clairs.

    Returns:
        True si valide ou si le filtre n'est pas spécifié, False sinon.
    """
    log_sym = pair_symbol_for_log or symbol_info.get('symbol', 'N/A_SYMBOL')
    log_prefix = f"[{log_sym}][validate_notional]"

    if np.isnan(quantity) or np.isinf(quantity) or np.isnan(price) or np.isinf(price):
        logger.warning(f"{log_prefix} Quantité ({quantity}) ou prix ({price}) invalide (NaN/Inf). Validation notionnelle échouée.")
        return False

    min_notional_filter = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
    if min_notional_filter is None: # Fallback pour certains exchanges/endpoints
        min_notional_filter = get_filter_value(symbol_info, 'NOTIONAL', 'minNotional')

    if min_notional_filter is not None and min_notional_filter > 0:
        current_notional = abs(quantity * price)
        # Utiliser une petite tolérance pour la comparaison flottante
        if current_notional < (min_notional_filter - FLOAT_COMPARISON_EPSILON):
            logger.debug(f"{log_prefix} Échec: Notionnel actuel {current_notional:.4f} < Min requis {min_notional_filter:.4f}")
            return False
        logger.debug(f"{log_prefix} Succès: Notionnel actuel {current_notional:.4f} >= Min requis {min_notional_filter:.4f}")
        return True
    
    logger.debug(f"{log_prefix} Filtre MIN_NOTIONAL non trouvé ou non applicable. Validation notionnelle passée par défaut.")
    return True # Si le filtre n'est pas défini, on suppose que c'est valide

def validate_order_parameters(
    order_params: Dict[str, Any],
    symbol_info: dict,
    pair_symbol_for_log: Optional[str] = None,
    estimated_market_price_for_market_order: Optional[float] = None
) -> List[str]:
    """
    Valide les paramètres d'un ordre (quantité, prix pour les ordres LIMIT)
    par rapport aux filtres du symbole de l'exchange (LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL).

    Args:
        order_params: Dictionnaire contenant 'quantity' et optionnellement 'price'.
                      Les valeurs doivent être des nombres ou des strings convertibles en float.
        symbol_info: Dictionnaire des informations du symbole de l'exchange.
        pair_symbol_for_log: Symbole de la paire pour des messages de log plus clairs.
        estimated_market_price_for_market_order: Prix estimé à utiliser pour la validation
                                                 du notionnel des ordres MARKET.

    Returns:
        Une liste de messages d'erreur (strings). Si la liste est vide, l'ordre est valide.
    """
    log_sym = pair_symbol_for_log or symbol_info.get('symbol', 'N/A_SYMBOL')
    log_prefix = f"[{log_sym}][validate_order_params]"
    errors: List[str] = []

    # --- Extraire et valider la quantité ---
    quantity_str = order_params.get('quantity')
    if quantity_str is None:
        errors.append("Paramètre 'quantity' manquant dans order_params.")
        return errors # Erreur bloquante
    try:
        quantity = float(quantity_str)
        if np.isnan(quantity) or np.isinf(quantity) or quantity <= 0: # Quantité doit être positive
            errors.append(f"Quantité invalide: '{quantity_str}'. Doit être un nombre positif.")
            return errors # Erreur bloquante
    except (ValueError, TypeError):
        errors.append(f"Quantité '{quantity_str}' n'est pas un nombre valide.")
        return errors # Erreur bloquante

    # --- Validation de la Quantité (LOT_SIZE) ---
    min_qty = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
    max_qty = get_filter_value(symbol_info, 'LOT_SIZE', 'maxQty')
    step_size_qty = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize')

    if min_qty is not None and quantity < (min_qty - FLOAT_COMPARISON_EPSILON):
        errors.append(f"Quantité {quantity:.8f} < minQty {min_qty:.8f}")
    if max_qty is not None and quantity > (max_qty + FLOAT_COMPARISON_EPSILON):
        errors.append(f"Quantité {quantity:.8f} > maxQty {max_qty:.8f}")
    
    if step_size_qty is not None and step_size_qty > 1e-12: # Ignorer si step_size est effectivement zéro ou trop petit
        # Check: (quantity - minQty) % stepSize == 0
        # minQty_for_step_check = min_qty if min_qty is not None else 0.0 # Binance doc dit (qty-minQty)%stepSize == 0
        # Pour simplifier et être plus général, on vérifie si la quantité est un multiple du step_size,
        # après avoir potentiellement soustrait min_qty si min_qty lui-même n'est pas un multiple de step_size.
        # Cependant, la plupart des exchanges s'attendent à ce que la quantité elle-même soit un multiple de step_size
        # *après* avoir satisfait min_qty.
        # La contrainte est: quantity = minQty + K * stepSize (où K est un entier >= 0)
        # Donc (quantity - minQty) doit être un multiple de stepSize.
        
        val_to_check_step = quantity
        if min_qty is not None and min_qty > 0 : # Si minQty est défini et positif
             val_to_check_step = quantity - min_qty
             if val_to_check_step < -FLOAT_COMPARISON_EPSILON: # quantity < minQty, déjà géré
                 pass # L'erreur minQty sera déjà ajoutée
             elif val_to_check_step < 0: # quantity est très proche de minQty mais légèrement en dessous
                 val_to_check_step = 0 # Traiter comme si c'était minQty

        # Utiliser Decimal pour la vérification du modulo avec des flottants
        try:
            dec_val = Decimal(str(val_to_check_step))
            dec_step = Decimal(str(step_size_qty))
            if not (dec_val % dec_step).is_zero(): # is_zero() gère la précision de Decimal
                 # Petite tolérance pour les erreurs de conversion float -> Decimal initiales
                 if not math.isclose((dec_val % dec_step), Decimal(0), abs_tol=Decimal('1e-10')) and \
                    not math.isclose((dec_val % dec_step), dec_step, abs_tol=Decimal('1e-10')):
                    errors.append(f"Quantité {quantity:.8f} (val_to_check_step: {val_to_check_step:.8f}) "
                                  f"ne respecte pas stepSize {step_size_qty:.8f}. Remainder: {dec_val % dec_step}")
        except InvalidOperation:
            errors.append(f"Erreur de conversion Decimal pour la vérification stepSize de la quantité (val: {val_to_check_step}, step: {step_size_qty}).")


    # --- Validation du Prix (PRICE_FILTER) - Uniquement si ordre LIMIT et prix fourni ---
    order_type = order_params.get('type', "LIMIT").upper() # Supposer LIMIT si non spécifié
    price_str = order_params.get('price')
    price: Optional[float] = None

    if order_type != "MARKET" and price_str is not None:
        try:
            price = float(price_str)
            if np.isnan(price) or np.isinf(price) or price <= 0:
                errors.append(f"Prix invalide: '{price_str}'. Doit être un nombre positif.")
                price = None # Ne pas continuer la validation du prix si invalide
        except (ValueError, TypeError):
            errors.append(f"Prix '{price_str}' n'est pas un nombre valide.")
            price = None
        
        if price is not None:
            min_price = get_filter_value(symbol_info, 'PRICE_FILTER', 'minPrice')
            max_price = get_filter_value(symbol_info, 'PRICE_FILTER', 'maxPrice')
            tick_size_price = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')

            if min_price is not None and price < (min_price - FLOAT_COMPARISON_EPSILON):
                errors.append(f"Prix {price:.8f} < minPrice {min_price:.8f}")
            if max_price is not None and max_price > 0 and price > (max_price + FLOAT_COMPARISON_EPSILON): # maxPrice peut être 0 pour "pas de limite"
                errors.append(f"Prix {price:.8f} > maxPrice {max_price:.8f}")
            
            if tick_size_price is not None and tick_size_price > 1e-12:
                try:
                    dec_price = Decimal(str(price))
                    dec_tick = Decimal(str(tick_size_price))
                    if not (dec_price % dec_tick).is_zero():
                        if not math.isclose((dec_price % dec_tick), Decimal(0), abs_tol=Decimal('1e-10')) and \
                           not math.isclose((dec_price % dec_tick), dec_tick, abs_tol=Decimal('1e-10')):
                            errors.append(f"Prix {price:.8f} ne respecte pas tickSize {tick_size_price:.8f}. Remainder: {dec_price % dec_tick}")
                except InvalidOperation:
                     errors.append(f"Erreur de conversion Decimal pour la vérification tickSize du prix (prix: {price}, tick: {tick_size_price}).")


    # --- Validation du Notionnel (MIN_NOTIONAL) ---
    price_for_notional_check: Optional[float] = None
    if order_type == "MARKET":
        if estimated_market_price_for_market_order is not None:
            price_for_notional_check = estimated_market_price_for_market_order
        else:
            logger.debug(f"{log_prefix} Prix de marché estimé non fourni pour l'ordre MARKET. "
                         "La validation du notionnel MIN_NOTIONAL pourrait être imprécise ou sautée si le filtre existe.")
            # On pourrait essayer de récupérer un prix ticker ici, mais c'est coûteux pour une simple validation.
            # Le filtre MIN_NOTIONAL est souvent appliqué par l'exchange au moment de l'exécution pour les ordres MARKET.
    elif price is not None: # Pour les ordres LIMIT, utiliser le prix de l'ordre
        price_for_notional_check = price

    if price_for_notional_check is not None:
        if not validate_notional(quantity, price_for_notional_check, symbol_info, pair_symbol_for_log):
            min_notional_val = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional') or \
                               get_filter_value(symbol_info, 'NOTIONAL', 'minNotional')
            errors.append(f"Valeur notionnelle ({abs(quantity * price_for_notional_check):.4f}) "
                          f"< MIN_NOTIONAL ({min_notional_val:.4f if min_notional_val else 'N/A'})")
    
    if errors:
        logger.warning(f"{log_prefix} Validation de l'ordre échouée: {errors}")
    else:
        logger.debug(f"{log_prefix} Validation de l'ordre réussie.")
        
    return errors
