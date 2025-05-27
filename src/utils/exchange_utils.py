# src/utils/exchange_utils.py
"""
Utilitaires pour les opérations liées aux exchanges.
Gère la précision des prix/quantités, les filtres d'exchange, et les calculs d'arrondi.
"""
import logging
from typing import Any, Dict, Optional, Union, List
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP, ROUND_FLOOR,  InvalidOperation, Context as DecimalContext, localcontext as decimal_localcontext
import math

logger = logging.getLogger(__name__)

# Contexte décimal pour les opérations de haute précision.
DECIMAL_CONTEXT_HIGH_PRECISION = DecimalContext(prec=28) # Précision suffisante pour la plupart des crypto-monnaies

def get_pair_config_for_symbol(
    pair_symbol: str,
    exchange_info_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extrait la configuration spécifique d'une paire (symbole) à partir des
    données d'information de l'exchange.
    """
    log_prefix = f"[GetPairCfg({pair_symbol.upper()})]" # Utilisation de pair_symbol pour le log
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

    logger.warning(f"{log_prefix} Aucune configuration trouvée pour la paire '{pair_symbol_upper}' dans les données de l'exchange fournies.")
    return None

def _calculate_precision(value_str: str) -> int:
    """
    Calcule la précision (nombre de décimales) à partir d'une chaîne.
    Exemple: "0.001" -> 3, "1" -> 0, "0.100" -> 1 (après suppression des zéros non significatifs à droite).
    Gère la notation scientifique.

    Args:
        value_str: Chaîne représentant un nombre (ex: "0.00010000", "1e-4").

    Returns:
        Nombre de décimales significatives.
    """
    if not isinstance(value_str, str) or not value_str.strip():
        logger.warning(f"_calculate_precision: Chaîne de valeur invalide ou vide : '{value_str}'. Retourne 0.")
        return 0
    
    value_str_cleaned = value_str.strip().lower()

    if 'e' in value_str_cleaned:
        try:
            d_value = Decimal(value_str_cleaned)
            if d_value.is_zero(): return 0 
            exponent = d_value.normalize().as_tuple().exponent
            return abs(exponent) if isinstance(exponent, int) and exponent < 0 else 0
        except InvalidOperation:
            logger.error(f"_calculate_precision: InvalidOperation pour la notation scientifique '{value_str_cleaned}'.")
            return 0 

    if '.' in value_str_cleaned:
        parts = value_str_cleaned.split('.')
        if len(parts) == 2:
            decimal_part = parts[1].rstrip('0') 
            return len(decimal_part)
        else: 
            logger.warning(f"_calculate_precision: Format de nombre décimal invalide '{value_str_cleaned}'.")
            return 0
    return 0 

def get_precision_from_filter(symbol_info: Dict[str, Any], filter_type: str, key: str) -> Optional[int]:
    """
    Récupère la précision (nombre de décimales) depuis un filtre d'exchange.
    """
    log_prefix = f"[GetPrecision][{symbol_info.get('symbol','N/A')}/{filter_type}/{key}]"
    if not symbol_info or 'filters' not in symbol_info or not isinstance(symbol_info['filters'], list):
        logger.warning(f"{log_prefix} symbol_info manquant, invalide ou sans clé 'filters'.")
        return None
    
    for filter_dict in symbol_info['filters']:
        if isinstance(filter_dict, dict) and filter_dict.get('filterType') == filter_type:
            value_str = filter_dict.get(key)
            if value_str is not None and isinstance(value_str, str):
                try:
                    precision = _calculate_precision(value_str)
                    logger.debug(f"{log_prefix} Précision calculée: {precision} depuis la valeur de filtre '{value_str}'.")
                    return precision
                except Exception as e: 
                    logger.error(f"{log_prefix} Erreur lors du calcul de la précision pour la valeur '{value_str}': {e}", exc_info=True)
                    return None 
            else:
                logger.debug(f"{log_prefix} Clé '{key}' non trouvée ou sa valeur n'est pas une chaîne dans le filtre {filter_type}: {filter_dict.get(key)}")
    
    logger.debug(f"{log_prefix} Filtre {filter_type} avec clé {key} non trouvé.")
    return None

def get_filter_value(symbol_info: Dict[str, Any], filter_type: str, key: str) -> Optional[float]:
    """
    Récupère une valeur de filtre comme float.
    """
    log_prefix = f"[GetFilterVal][{symbol_info.get('symbol','N/A')}/{filter_type}/{key}]"
    if not symbol_info or 'filters' not in symbol_info or not isinstance(symbol_info['filters'], list):
        logger.warning(f"{log_prefix} symbol_info manquant, invalide ou sans clé 'filters'.")
        return None
    
    for filter_dict in symbol_info['filters']:
        if isinstance(filter_dict, dict) and filter_dict.get('filterType') == filter_type:
            value_str = filter_dict.get(key)
            if value_str is not None:
                try:
                    val_float = float(str(value_str)) 
                    logger.debug(f"{log_prefix} Valeur de filtre '{value_str}' convertie en float: {val_float}.")
                    return val_float
                except (ValueError, TypeError) as e_conv:
                    logger.error(f"{log_prefix} Impossible de convertir la valeur de filtre '{value_str}' en float pour {filter_type}/{key}: {e_conv}")
                    return None
            else:
                logger.debug(f"{log_prefix} Clé '{key}' non trouvée dans le filtre {filter_type}: {filter_dict}")

    logger.debug(f"{log_prefix} Filtre {filter_type} avec clé {key} non trouvé.")
    return None

def adjust_precision(
    value: Union[float, str, Decimal], 
    precision: Optional[int], 
    tick_size: Optional[Union[float, str, Decimal]] = None,
    rounding_mode_str: str = "ROUND_HALF_UP" 
) -> Optional[float]:
    """
    Ajuste une valeur numérique à une précision décimale spécifiée ou à un multiple
    d'un `tick_size` donné, en utilisant la classe `Decimal` pour la précision.
    """
    if value is None:
        return None

    try:
        with decimal_localcontext(DECIMAL_CONTEXT_HIGH_PRECISION) as ctx:
            value_decimal = ctx.create_decimal(str(value)) 

            if value_decimal.is_nan() or value_decimal.is_infinite():
                logger.debug(f"adjust_precision: Valeur {value} est NaN ou Inf. Retour de la valeur float originale.")
                return float(value) 

            actual_rounding_mode = getattr(Decimal, rounding_mode_str.upper(), None)
            if actual_rounding_mode is None:
                logger.warning(f"adjust_precision: Mode d'arrondi '{rounding_mode_str}' invalide. Utilisation de ROUND_HALF_UP.")
                actual_rounding_mode = ROUND_HALF_UP
            ctx.rounding = actual_rounding_mode 

            adjusted_value_decimal: Decimal

            if tick_size is not None:
                tick_size_decimal = ctx.create_decimal(str(tick_size))
                if tick_size_decimal.is_zero() or not tick_size_decimal.is_finite():
                    logger.warning(f"adjust_precision: tick_size '{tick_size}' invalide. "
                                   "Tentative avec 'precision' si fournie.")
                    # Laisser la logique de précision ci-dessous s'appliquer
                else: 
                    adjusted_value_decimal = (value_decimal / tick_size_decimal).quantize(Decimal('0'), context=ctx) * tick_size_decimal
                    logger.debug(f"adjust_precision avec tick_size: {value_decimal} -> {adjusted_value_decimal} (tick_size={tick_size_decimal}, mode={rounding_mode_str})")
                    # Si 'precision' est aussi fourni, on peut l'utiliser pour un formatage final après l'ajustement au tick_size.
                    # Cela est utile si la précision souhaitée est plus grossière que celle induite par le tick_size.
                    if precision is not None and isinstance(precision, int) and precision >= 0:
                        quantizer_final_format = Decimal('1e-' + str(precision))
                        adjusted_value_decimal = adjusted_value_decimal.quantize(quantizer_final_format, context=ctx)
                        logger.debug(f"adjust_precision (post-tick_size) formatage final à {precision} décimales: -> {adjusted_value_decimal}")
                    return float(adjusted_value_decimal)

            if precision is not None and isinstance(precision, int) and precision >= 0:
                quantizer = Decimal('1e-' + str(precision))
                adjusted_value_decimal = value_decimal.quantize(quantizer, context=ctx)
                logger.debug(f"adjust_precision avec precision: {value_decimal} -> {adjusted_value_decimal} (precision={precision}, mode={rounding_mode_str})")
                return float(adjusted_value_decimal)
            
            logger.debug(f"adjust_precision: Ni tick_size valide ni 'precision' fournie. Retour de la valeur originale (float): {float(value_decimal)}.")
            return float(value_decimal)

    except InvalidOperation as e_inv_op:
        logger.warning(f"adjust_precision: InvalidOperation lors de la conversion de '{value}' ou '{tick_size}' en Decimal: {e_inv_op}. Retour de None.")
        return None
    except Exception as e_unexp:
        logger.error(f"adjust_precision: Erreur inattendue lors de l'ajustement de '{value}': {e_unexp}", exc_info=True)
        try:
            return round(float(str(value)), precision or 8) 
        except:
            return None


def adjust_quantity_to_step_size(
    quantity: Union[float, str, Decimal],
    symbol_info: Dict[str, Any],
    qty_precision: Optional[int] = None, 
    rounding_mode_str: str = "ROUND_FLOOR" 
) -> Optional[float]:
    """
    Ajuste une quantité selon le `stepSize` du filtre `LOT_SIZE` de l'exchange.
    """
    log_prefix = f"[AdjustQtyToStep][{symbol_info.get('symbol','N/A')}]"
    
    try:
        initial_quantity_decimal = Decimal(str(quantity))
    except InvalidOperation:
        logger.error(f"{log_prefix} Quantité initiale '{quantity}' non convertible en Decimal.")
        return None

    if initial_quantity_decimal.is_nan() or initial_quantity_decimal.is_infinite() or initial_quantity_decimal < Decimal(0):
        logger.warning(f"{log_prefix} Quantité initiale invalide: {initial_quantity_decimal}. Retour de 0.0.")
        return 0.0 

    step_size_from_filter_val = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize')
    
    effective_precision = qty_precision
    if effective_precision is None and step_size_from_filter_val is not None:
        effective_precision = _calculate_precision(str(step_size_from_filter_val))
    elif effective_precision is None:
        effective_precision = 8 
        logger.debug(f"{log_prefix} Précision de quantité non fournie et non dérivable de stepSize. Utilisation de {effective_precision} décimales.")

    if step_size_from_filter_val is None or step_size_from_filter_val <= 1e-12: 
        logger.debug(f"{log_prefix} stepSize ('{step_size_from_filter_val}') non valide. "
                       f"Arrondi de la quantité {initial_quantity_decimal} à {effective_precision} décimales avec {rounding_mode_str}.")
        return adjust_precision(initial_quantity_decimal, effective_precision, tick_size=None, rounding_mode_str=rounding_mode_str)

    try:
        with decimal_localcontext(DECIMAL_CONTEXT_HIGH_PRECISION) as ctx:
            step_size_decimal = ctx.create_decimal(str(step_size_from_filter_val))
            
            actual_rounding_mode = getattr(Decimal, rounding_mode_str.upper(), None)
            if actual_rounding_mode is None:
                logger.warning(f"{log_prefix} Mode d'arrondi '{rounding_mode_str}' invalide pour step_size. Utilisation de ROUND_FLOOR.")
                actual_rounding_mode = ROUND_FLOOR
            ctx.rounding = actual_rounding_mode

            if step_size_decimal.is_zero():
                 logger.error(f"{log_prefix} step_size_decimal est zéro. Division par zéro évitée.")
                 return adjust_precision(initial_quantity_decimal, effective_precision, tick_size=None, rounding_mode_str=rounding_mode_str)

            adjusted_qty_decimal = (initial_quantity_decimal / step_size_decimal).quantize(Decimal('0'), context=ctx) * step_size_decimal
            
            if effective_precision is not None:
                quantizer_final = Decimal('1e-' + str(effective_precision))
                adjusted_qty_decimal = adjusted_qty_decimal.quantize(quantizer_final, context=ctx)

            logger.debug(f"{log_prefix} Quantité ajustée: {initial_quantity_decimal} -> {adjusted_qty_decimal} "
                         f"(stepSize={step_size_decimal}, mode={rounding_mode_str}, prec_eff={effective_precision})")
            
            final_float_qty = float(adjusted_qty_decimal)
            min_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
            if min_qty_filter is not None and final_float_qty < min_qty_filter and final_float_qty > 1e-9: 
                logger.warning(f"{log_prefix} Quantité ajustée {final_float_qty} < minQty requis {min_qty_filter}. Retour de 0.0.")
                return 0.0
            elif final_float_qty < 0: 
                 logger.error(f"{log_prefix} Quantité ajustée est devenue négative: {final_float_qty}. Retour de 0.0.")
                 return 0.0

            return final_float_qty

    except InvalidOperation as e_inv_op_qty:
        logger.error(f"{log_prefix} InvalidOperation lors de l'ajustement de la quantité '{quantity}' avec step_size '{step_size_from_filter_val}': {e_inv_op_qty}.")
        return None
    except Exception as e_unexp_qty:
        logger.error(f"{log_prefix} Erreur inattendue lors de l'ajustement de la quantité '{quantity}': {e_unexp_qty}", exc_info=True)
        return None


def calculate_notional_value(quantity: float, price: float) -> float:
    """
    Calcule la valeur notionnelle d'un ordre.
    """
    if not (isinstance(quantity, (int, float)) and isinstance(price, (int, float))):
        logger.warning(f"calculate_notional_value: Entrées invalides pour quantité ({quantity}, type {type(quantity)}) "
                       f"ou prix ({price}, type {type(price)}).")
        return 0.0
    return quantity * price

def validate_order_filters(
    quantity: float,
    price: float,
    symbol_info: Dict[str, Any],
    order_type: str = "LIMIT" 
) -> Dict[str, Any]:
    """
    Valide un ordre contre tous les filtres pertinents de l'exchange.
    """
    errors: List[str] = []
    log_prefix = f"[ValidateOrderFilters][{symbol_info.get('symbol','N/A')}]"

    min_qty = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
    if min_qty is not None and quantity < min_qty:
        errors.append(f"Quantité {quantity:.8f} < minQty requis {min_qty:.8f}")
    
    max_qty = get_filter_value(symbol_info, 'LOT_SIZE', 'maxQty')
    if max_qty is not None and quantity > max_qty:
        errors.append(f"Quantité {quantity:.8f} > maxQty requis {max_qty:.8f}")

    if order_type.upper() not in ["MARKET"]: 
        min_price = get_filter_value(symbol_info, 'PRICE_FILTER', 'minPrice')
        if min_price is not None and price < min_price:
            errors.append(f"Prix {price:.8f} < minPrice requis {min_price:.8f}")
        
        max_price = get_filter_value(symbol_info, 'PRICE_FILTER', 'maxPrice')
        if max_price is not None and price > max_price:
            errors.append(f"Prix {price:.8f} > maxPrice requis {max_price:.8f}")

    notional_value = calculate_notional_value(quantity, price)
    min_notional_val: Optional[float] = None
    notional_filter_type_used = ""

    notional_filter_details = next((f for f in symbol_info.get('filters', []) if f.get('filterType') == 'NOTIONAL'), None)
    min_notional_filter_details = next((f for f in symbol_info.get('filters', []) if f.get('filterType') == 'MIN_NOTIONAL'), None)

    if order_type.upper() == "MARKET":
        if notional_filter_details and notional_filter_details.get('applyMinToMarket', False):
            min_notional_val = get_filter_value(symbol_info, 'NOTIONAL', 'minNotional')
            if min_notional_val is not None: notional_filter_type_used = "NOTIONAL (applyMinToMarket)"
        elif notional_filter_details:
             logger.debug(f"{log_prefix} Filtre NOTIONAL trouvé mais applyMinToMarket n'est pas true.")
        
        if min_notional_val is None and min_notional_filter_details: # Fallback rare si MIN_NOTIONAL s'applique aussi aux market
            min_notional_val = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
            if min_notional_val is not None: notional_filter_type_used = "MIN_NOTIONAL (fallback pour MARKET)"
    else: 
        if min_notional_filter_details:
            min_notional_val = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
            if min_notional_val is not None: notional_filter_type_used = "MIN_NOTIONAL"
        
        if min_notional_val is None and notional_filter_details: # Fallback sur NOTIONAL pour les ordres non-MARKET
            min_notional_val = get_filter_value(symbol_info, 'NOTIONAL', 'minNotional')
            if min_notional_val is not None: notional_filter_type_used = "NOTIONAL (fallback pour non-MARKET)"

    if min_notional_val is not None and notional_value < min_notional_val:
        errors.append(f"Valeur notionnelle {notional_value:.2f} < {notional_filter_type_used} requis {min_notional_val:.2f}")

    if errors:
        logger.warning(f"{log_prefix} Échecs de validation de l'ordre : {'; '.join(errors)}")
    else:
        logger.debug(f"{log_prefix} Validation des filtres d'ordre réussie pour Qty:{quantity}, Px:{price}.")
        
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }
