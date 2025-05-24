# Fichier: src/utils/exchange_utils.py
"""
Utilitaires pour gérer les filtres, précisions et validations
spécifiques aux exchanges (ex: Binance).
"""

import logging
import math
import numpy as np
from typing import Dict, Optional, Any, Union, List, Callable
from decimal import Decimal, InvalidOperation
from pathlib import Path # Ajouté pour une éventuelle gestion de chemin, bien que non utilisé dans la nouvelle fonction

# Importer load_exchange_config si get_pair_config_for_symbol devait charger le fichier lui-même.
# Cependant, pour éviter les dépendances circulaires et suivre le pattern existant,
# nous supposerons que les données de l'exchange sont passées en tant que dictionnaire.
# from src.config.loader import load_exchange_config # Optionnel, si la fonction doit charger le fichier

logger = logging.getLogger(__name__)

# Une petite valeur epsilon pour les comparaisons de flottants, si nécessaire.
# Pour les vérifications de multiples de tick/step, il est souvent préférable d'utiliser Decimal.
FLOAT_COMPARISON_EPSILON = 1e-9

def get_pair_config_for_symbol(
    pair_symbol: str,
    exchange_info_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extrait la configuration spécifique d'une paire (symbol) à partir des données
    d'information de l'exchange (préalablement chargées).

    Args:
        pair_symbol (str): Le symbole de la paire à rechercher (ex: "BTCUSDT").
        exchange_info_data (Dict[str, Any]): Un dictionnaire contenant les informations
                                             de l'exchange, typiquement la sortie de
                                             `load_exchange_config` ou de l'API get_exchange_info().
                                             Doit contenir une clé "symbols" qui est une liste de dicts.

    Returns:
        Optional[Dict[str, Any]]: Un dictionnaire contenant la configuration de la paire
                                  si trouvée, sinon None.
    """
    log_prefix = f"[get_pair_config_for_symbol({pair_symbol})]"
    if not isinstance(exchange_info_data, dict):
        logger.error(f"{log_prefix} exchange_info_data n'est pas un dictionnaire. Reçu: {type(exchange_info_data)}")
        return None

    symbols_list = exchange_info_data.get("symbols")
    if not isinstance(symbols_list, list):
        logger.error(f"{log_prefix} La clé 'symbols' est manquante ou n'est pas une liste dans exchange_info_data.")
        return None

    for symbol_details in symbols_list:
        if isinstance(symbol_details, dict) and symbol_details.get("symbol") == pair_symbol.upper():
            logger.debug(f"{log_prefix} Configuration trouvée pour la paire.")
            return symbol_details

    logger.warning(f"{log_prefix} Aucune configuration trouvée pour la paire dans les données de l'exchange fournies.")
    return None


def adjust_precision(
    value: Optional[Union[float, str, Decimal]], # Modifié pour accepter Decimal
    precision: Optional[int],
    rounding_method: Callable[[Decimal], Decimal] = lambda x: x.quantize(Decimal('1'), rounding='ROUND_FLOOR'), # type: ignore
    tick_size: Optional[Union[float, str, Decimal]] = None # Modifié pour accepter Decimal
) -> Optional[float]:
    """
    Ajuste une valeur numérique à une précision spécifiée ou à un multiple d'un tick_size.
    Utilise Decimal pour une meilleure précision.

    Si tick_size est fourni, il a la priorité sur 'precision' pour l'ajustement.
    Le prix sera ajusté pour être un multiple du tick_size en utilisant la méthode d'arrondi.

    Args:
        value: La valeur à ajuster (peut être float, int, str convertible, ou Decimal).
        precision: Le nombre de décimales souhaité si tick_size n'est pas utilisé.
                   Ignoré si tick_size est fourni.
        rounding_method: La méthode d'arrondi pour Decimal (ex: ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_UP).
                         Par défaut ROUND_FLOOR (conservateur pour les quantités).
        tick_size: La taille du tick. Si fourni, la valeur sera ajustée pour être
                   un multiple de cette taille.

    Returns:
        La valeur ajustée en float, ou None si l'entrée est invalide.
    """
    if value is None:
        return None

    try:
        val_decimal = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        logger.warning(f"adjust_precision: Impossible de convertir la valeur '{value}' en Decimal.")
        return None

    if val_decimal.is_nan() or val_decimal.is_infinite():
        logger.debug(f"adjust_precision: La valeur est NaN ou Inf ({val_decimal}). Retour de la valeur float originale.")
        return float(value) # Retourner NaN/Inf sans modification, converti en float

    if tick_size is not None:
        try:
            ts_decimal = Decimal(str(tick_size))

            if ts_decimal.is_zero():
                logger.warning("adjust_precision: tick_size est zéro. Retour de la valeur originale.")
                return float(val_decimal)

            # Ajustement pour être un multiple de tick_size
            # (val_decimal / ts_decimal) donne le nombre de ticks.
            # .quantize(Decimal('1'), rounding=...) arrondit ce nombre à l'entier le plus proche selon la méthode.
            # Puis on multiplie par ts_decimal.
            # Assurer que rounding_method est compatible avec Decimal.quantize
            # Les méthodes math.floor, math.ceil, round ne sont pas directement utilisables avec quantize.
            # On doit utiliser les constantes de rounding de Decimal.
            # Le rounding_method passé doit être une string comme 'ROUND_FLOOR', etc.
            # Pour l'instant, la signature attend une Callable. On va simplifier pour utiliser les strings de Decimal.

            # Exemple de mapping simple si on veut garder la signature Callable (plus complexe)
            # Pour cet exemple, on va supposer que rounding_method est déjà une string de Decimal.ROUND_*
            # ou on utilise un mapping interne. Pour l'instant, on va utiliser une approche directe.

            if rounding_method.__name__ == 'floor' or str(rounding_method).lower().endswith("floor'>"): # Approximation
                adjusted_val_decimal = (val_decimal / ts_decimal).to_integral_value(rounding='ROUND_FLOOR') * ts_decimal
            elif rounding_method.__name__ == 'ceil' or str(rounding_method).lower().endswith("ceiling'>"):
                adjusted_val_decimal = (val_decimal / ts_decimal).to_integral_value(rounding='ROUND_CEILING') * ts_decimal
            elif rounding_method.__name__ == 'round':
                adjusted_val_decimal = (val_decimal / ts_decimal).to_integral_value(rounding='ROUND_HALF_UP') * ts_decimal
            else: # Fallback si la fonction passée n'est pas reconnue, ou si c'est déjà une string de Decimal
                try:
                    # Si rounding_method est une string comme 'ROUND_FLOOR'
                    if isinstance(rounding_method, str) and hasattr(Decimal, rounding_method.upper()):
                        actual_rounding_mode = getattr(Decimal, rounding_method.upper())
                        adjusted_val_decimal = (val_decimal / ts_decimal).quantize(Decimal('1'), rounding=actual_rounding_mode) * ts_decimal
                    else: # Si c'est une fonction non mappée, fallback à ROUND_HALF_UP
                        logger.warning(f"adjust_precision: Méthode d'arrondi non standard pour tick_size. Utilisation de ROUND_HALF_UP.")
                        adjusted_val_decimal = (val_decimal / ts_decimal).to_integral_value(rounding='ROUND_HALF_UP') * ts_decimal
                except Exception as e_round_mode:
                    logger.error(f"Erreur avec rounding_method pour tick_size: {e_round_mode}. Fallback ROUND_HALF_UP.")
                    adjusted_val_decimal = (val_decimal / ts_decimal).to_integral_value(rounding='ROUND_HALF_UP') * ts_decimal

            return float(adjusted_val_decimal)

        except (InvalidOperation, ValueError, TypeError) as e_decimal:
            logger.error(f"adjust_precision: Erreur lors de l'ajustement avec tick_size Decimal ('{tick_size}'): {e_decimal}. "
                         f"Retour de la valeur originale après arrondi standard à 8 décimales (float).")
            return round(float(val_decimal), 8) # Fallback simple

    # Si tick_size n'est pas fourni, utiliser la 'precision' (nombre de décimales)
    if precision is None or not isinstance(precision, int) or precision < 0:
        logger.debug(f"adjust_precision: Précision invalide ou manquante ({precision}) et pas de tick_size. "
                     f"Retour de la valeur originale: {float(val_decimal)}.")
        return float(val_decimal)

    # Créer le quantizer pour la précision (ex: Decimal('0.01') pour precision=2)
    quantizer = Decimal('1e-' + str(precision))
    
    # Utiliser la même logique de mapping pour rounding_method que ci-dessus pour la précision
    final_rounding_mode = 'ROUND_HALF_UP' # Default
    if rounding_method.__name__ == 'floor' or str(rounding_method).lower().endswith("floor'>"):
        final_rounding_mode = 'ROUND_FLOOR'
    elif rounding_method.__name__ == 'ceil' or str(rounding_method).lower().endswith("ceiling'>"):
        final_rounding_mode = 'ROUND_CEILING'
    elif isinstance(rounding_method, str) and hasattr(Decimal, rounding_method.upper()):
        final_rounding_mode = getattr(Decimal, rounding_method.upper())


    try:
        adjusted_val_decimal = val_decimal.quantize(quantizer, rounding=final_rounding_mode)
        return float(adjusted_val_decimal)
    except OverflowError:
        logger.error(f"adjust_precision: OverflowError lors de l'ajustement de {val_decimal} avec précision {precision}. "
                     "Retour de la valeur originale.")
        return float(val_decimal)
    except Exception as e_prec:
        logger.error(f"adjust_precision: Erreur lors de l'ajustement à la précision {precision}: {e_prec}. "
                      "Retour de la valeur originale.")
        return float(val_decimal)


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

        size_str_cleaned = size_str.strip().rstrip('0') # Rstrip '0' to handle "1.000" as "1.0" or "1" as "1"

        # Utiliser Decimal pour une détermination robuste de la précision
        try:
            d_value = Decimal(size_str_cleaned)
            if d_value.is_zero(): # Un tick/step de zéro n'est pas valide pour la précision
                logger.warning(f"{log_prefix} {filter_key} '{size_str_cleaned}' est zéro, précision indéfinie.")
                return None

            # La méthode `as_tuple().exponent` de Decimal donne le nombre de chiffres après la virgule
            # pour les nombres < 1 (ex: 0.01 -> exp -2). Pour les entiers (1, 10), exponent est 0.
            # Pour "1.0", exponent est -1.
            # On veut le nombre de chiffres *après la virgule significatifs*.
            if '.' in size_str_cleaned:
                # Compter les chiffres après le point, sans les zéros de fin (déjà fait par rstrip)
                # Si c'est "0.001", on veut 3. Si "1.0", on veut 1. Si "1.120", devient "1.12", on veut 2.
                # Si "1", on veut 0.
                # L'utilisation de `size_str.split('.')[-1]` est plus directe pour cela.
                # Il faut utiliser la string originale avant rstrip pour "1.0"
                original_size_str = target_filter.get(filter_key, "").strip()
                if '.' in original_size_str:
                    return len(original_size_str.split('.')[-1])
                else: # Pas de point décimal dans la string originale (ex: "1", "100")
                    return 0
            else: # Pas de point décimal (ex: "1", "10", ou "1E-5")
                # Si c'est en notation scientifique comme "1E-5", Decimal gère bien ça.
                # L'exposant de Decimal().as_tuple().exponent peut être utilisé.
                # Pour "1E-5" (0.00001), exponent est -5. On veut 5.
                # Pour "1" (1), exponent est 0. On veut 0.
                exponent = d_value.normalize().as_tuple().exponent # Normalize pour enlever les zéros de fin de la partie décimale
                if isinstance(exponent, int):
                    return abs(exponent) if exponent < 0 else 0
                else: # Cas de 'NaN', 'Infinity', etc. (ne devrait pas arriver si d_value est un nombre fini)
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
    step_size_qty_str = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize') # C'est une string

    if min_qty is not None and quantity < (min_qty - FLOAT_COMPARISON_EPSILON):
        errors.append(f"Quantité {quantity:.8f} < minQty {min_qty:.8f}")
    if max_qty is not None and quantity > (max_qty + FLOAT_COMPARISON_EPSILON):
        errors.append(f"Quantité {quantity:.8f} > maxQty {max_qty:.8f}")

    if step_size_qty_str is not None:
        try:
            step_size_qty_dec = Decimal(str(step_size_qty_str))
            if step_size_qty_dec > Decimal(0):
                quantity_dec = Decimal(str(quantity))
                # Vérification: (quantité - minQty) % stepSize == 0 (si minQty est multiple de stepSize)
                # Ou plus simplement: quantité % stepSize == 0 (si minQty est 0 ou multiple de stepSize)
                # Binance doc: (quantity-minQty) % stepSize == 0
                # Pour être plus général, on vérifie si la quantité est un multiple de step_size,
                # après avoir potentiellement soustrait min_qty.
                # La contrainte est souvent que la quantité elle-même doit être un multiple de step_size,
                # et aussi >= min_qty.

                # Si min_qty est défini, la quantité doit être >= min_qty.
                # Ensuite, la quantité (ou quantité - min_qty si min_qty n'est pas un multiple de step_size)
                # doit être un multiple de step_size.
                # Le plus simple est de vérifier si la quantité elle-même est un multiple de step_size.
                # Si min_qty existe, il doit aussi être un multiple de step_size.
                # Binance: (quantity - minQty) % stepSize == 0
                
                val_to_check_step_dec = quantity_dec
                if min_qty is not None and min_qty > 0:
                    val_to_check_step_dec = quantity_dec - Decimal(str(min_qty))
                
                # Gérer les cas où val_to_check_step_dec est négatif à cause d'erreurs de flottants
                if val_to_check_step_dec < Decimal(0) and abs(val_to_check_step_dec) < step_size_qty_dec / Decimal(2):
                    val_to_check_step_dec = Decimal(0) # Si très proche de zéro par en dessous

                if val_to_check_step_dec >= Decimal(0): # Ne vérifier que si positif ou nul
                    remainder = val_to_check_step_dec % step_size_qty_dec
                    # Tolérance pour les erreurs de flottant lors de la comparaison du reste
                    if not (remainder.is_zero() or remainder.is_close(step_size_qty_dec, rel_tol=Decimal('1e-9')) or remainder.is_close(Decimal(0), abs_tol=Decimal('1e-10'))):
                        errors.append(f"Quantité {quantity:.8f} (val relative: {val_to_check_step_dec}) ne respecte pas stepSize {step_size_qty_str}. Reste: {remainder}")
                # Si val_to_check_step_dec < 0, c'est que quantity < min_qty, déjà géré.
        except InvalidOperation:
            errors.append(f"Erreur de conversion Decimal pour stepSize de quantité (val: {quantity}, step: {step_size_qty_str}).")


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
            tick_size_price_str = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')

            if min_price is not None and price < (min_price - FLOAT_COMPARISON_EPSILON):
                errors.append(f"Prix {price:.8f} < minPrice {min_price:.8f}")
            if max_price is not None and max_price > 0 and price > (max_price + FLOAT_COMPARISON_EPSILON): # maxPrice peut être 0 pour "pas de limite"
                errors.append(f"Prix {price:.8f} > maxPrice {max_price:.8f}")

            if tick_size_price_str is not None:
                try:
                    tick_size_price_dec = Decimal(str(tick_size_price_str))
                    if tick_size_price_dec > Decimal(0):
                        price_dec = Decimal(str(price))
                        remainder = price_dec % tick_size_price_dec
                        if not (remainder.is_zero() or remainder.is_close(tick_size_price_dec, rel_tol=Decimal('1e-9')) or remainder.is_close(Decimal(0), abs_tol=Decimal('1e-10'))):
                            errors.append(f"Prix {price:.8f} ne respecte pas tickSize {tick_size_price_str}. Reste: {remainder}")
                except InvalidOperation:
                     errors.append(f"Erreur de conversion Decimal pour tickSize du prix (prix: {price}, tick: {tick_size_price_str}).")


    # --- Validation du Notionnel (MIN_NOTIONAL) ---
    price_for_notional_check: Optional[float] = None
    if order_type == "MARKET":
        if estimated_market_price_for_market_order is not None:
            price_for_notional_check = estimated_market_price_for_market_order
        else:
            logger.debug(f"{log_prefix} Prix de marché estimé non fourni pour l'ordre MARKET. "
                         "La validation du notionnel MIN_NOTIONAL pourrait être imprécise ou sautée si le filtre existe.")
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

# Fonctions utilitaires pour ajuster prix et quantité (peuvent être déplacées si besoin)
def adjust_price_to_tick_size(price: float, symbol_info: Dict[str, Any], price_precision: Optional[int] = None) -> float:
    """Ajuste un prix pour qu'il soit un multiple de tickSize."""
    tick_size_str = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')
    if tick_size_str is None:
        # Si tick_size n'est pas trouvé, utiliser price_precision si disponible
        if price_precision is not None:
            return round(price, price_precision)
        return price # Retourner le prix original si aucune info de précision

    try:
        tick_size = Decimal(str(tick_size_str))
        if tick_size.is_zero(): return price # Éviter division par zéro

        price_dec = Decimal(str(price))
        # Arrondir au multiple le plus proche de tick_size
        adjusted_price_dec = (price_dec / tick_size).quantize(Decimal('1'), rounding='ROUND_HALF_UP') * tick_size
        
        # Assurer que le résultat final respecte aussi la pricePrecision globale si fournie
        if price_precision is not None:
            quantizer_precision = Decimal('1e-' + str(price_precision))
            adjusted_price_dec = adjusted_price_dec.quantize(quantizer_precision, rounding='ROUND_HALF_UP')

        return float(adjusted_price_dec)
    except (InvalidOperation, TypeError):
        logger.warning(f"Erreur lors de l'ajustement du prix {price} avec tick_size {tick_size_str}. Retour du prix original.")
        if price_precision is not None: return round(price, price_precision)
        return price

def adjust_quantity_to_step_size(quantity: float, symbol_info: Dict[str, Any], quantity_precision: Optional[int] = None) -> float:
    """Ajuste une quantité pour qu'elle soit un multiple de stepSize."""
    step_size_str = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize')
    if step_size_str is None:
        if quantity_precision is not None:
            # Arrondir vers le bas à la précision de quantité
            factor = 10**quantity_precision
            return math.floor(quantity * factor) / factor
        return quantity

    try:
        step_size = Decimal(str(step_size_str))
        if step_size.is_zero(): return quantity

        quantity_dec = Decimal(str(quantity))
        # Arrondir vers le bas (floor) au multiple de step_size
        adjusted_quantity_dec = (quantity_dec / step_size).to_integral_value(rounding='ROUND_FLOOR') * step_size
        
        # Assurer que le résultat final respecte aussi la quantityPrecision globale si fournie
        if quantity_precision is not None:
            quantizer_precision = Decimal('1e-' + str(quantity_precision))
            # Pour la quantité, on arrondit généralement vers le bas pour être conservateur
            adjusted_quantity_dec = adjusted_quantity_dec.quantize(quantizer_precision, rounding='ROUND_FLOOR')

        return float(adjusted_quantity_dec)
    except (InvalidOperation, TypeError):
        logger.warning(f"Erreur lors de l'ajustement de la quantité {quantity} avec step_size {step_size_str}. Retour de la quantité originale.")
        if quantity_precision is not None:
            factor = 10**quantity_precision
            return math.floor(quantity * factor) / factor
        return quantity

