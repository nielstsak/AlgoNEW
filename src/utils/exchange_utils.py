# src/utils/exchange_utils.py
"""
Ce module fournit des fonctions utilitaires pour gérer les filtres, les précisions
(prix, quantité) et les validations spécifiques aux exchanges (par exemple, Binance),
en se basant sur les informations de l'exchange.
"""

import logging
import math 
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP, ROUND_CEILING, ROUND_FLOOR, Context as DecimalContext, Inexact, Rounded, localcontext as decimal_localcontext # Importation correcte de localcontext
from typing import Dict, Optional, Any, Union, List

logger = logging.getLogger(__name__)

# Contexte décimal pour les opérations de haute précision.
DECIMAL_CONTEXT_HIGH_PRECISION = DecimalContext(prec=28)

# Tolérance pour les comparaisons de Decimals.
DECIMAL_COMPARISON_EPSILON = Decimal('1e-9')

def get_pair_config_for_symbol(
    pair_symbol: str,
    exchange_info_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extrait la configuration spécifique d'une paire (symbole) à partir des
    données d'information de l'exchange.
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
    rounding_mode_str: str = "ROUND_HALF_UP",
    tick_size: Optional[Union[float, str, Decimal]] = None
) -> Optional[float]:
    """
    Ajuste une valeur numérique à une précision décimale spécifiée ou à un multiple
    d'un `tick_size` donné, en utilisant la classe `Decimal` pour la précision.

    Args:
        value: La valeur à ajuster.
        precision: Le nombre de décimales souhaité si `tick_size` n'est pas utilisé.
        rounding_mode_str: La méthode d'arrondi Decimal (ex: "ROUND_DOWN").
        tick_size: La taille du pas (tick). Si fourni, `value` est ajustée à un multiple de `tick_size`.

    Returns:
        La valeur ajustée en float, ou None si l'entrée est invalide.
    """
    if value is None:
        return None

    if not isinstance(rounding_mode_str, str):
        logger.error(f"adjust_precision: 'rounding_mode_str' doit être une chaîne, reçu {type(rounding_mode_str)}. Valeur: {rounding_mode_str}")
        return None

    try:
        value_decimal = DECIMAL_CONTEXT_HIGH_PRECISION.create_decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as e_conv:
        logger.warning(f"adjust_precision: Impossible de convertir la valeur '{value}' (type: {type(value)}) en Decimal: {e_conv}")
        return None

    if value_decimal.is_nan() or value_decimal.is_infinite():
        logger.debug(f"adjust_precision: La valeur est NaN ou Inf ({value_decimal}). Retour de la valeur float originale.")
        return float(value_decimal)

    actual_rounding_mode = getattr(Decimal, rounding_mode_str.upper(), None)
    if actual_rounding_mode is None:
        logger.warning(f"adjust_precision: Mode d'arrondi '{rounding_mode_str}' invalide. Utilisation de ROUND_HALF_UP par défaut.")
        actual_rounding_mode = ROUND_HALF_UP

    adjusted_value_decimal: Decimal
    try:
        with decimal_localcontext(DECIMAL_CONTEXT_HIGH_PRECISION) as ctx: # Utilisation correcte de localcontext
            ctx.rounding = actual_rounding_mode 

            if tick_size is not None:
                tick_size_decimal = ctx.create_decimal(str(tick_size))
                if tick_size_decimal.is_zero() or not tick_size_decimal.is_finite():
                    logger.warning(f"adjust_precision: tick_size '{tick_size}' est zéro ou non fini. "
                                   "Tentative avec précision si fournie.")
                    if precision is not None and isinstance(precision, int) and precision >= 0:
                        quantizer = Decimal('1e-' + str(precision))
                        adjusted_value_decimal = value_decimal.quantize(quantizer) 
                        return float(adjusted_value_decimal)
                    logger.debug("adjust_precision: tick_size invalide et pas de précision. Retour de la valeur originale (float).")
                    return float(value_decimal)

                adjusted_value_decimal = (value_decimal / tick_size_decimal).quantize(Decimal('1')) * tick_size_decimal
            
            elif precision is not None and isinstance(precision, int) and precision >= 0:
                quantizer = Decimal('1e-' + str(precision))
                adjusted_value_decimal = value_decimal.quantize(quantizer)
            
            else: 
                logger.debug(f"adjust_precision: Ni tick_size ni précision valide fournie. Retour de la valeur originale (float): {float(value_decimal)}.")
                return float(value_decimal)
        
        return float(adjusted_value_decimal)

    except (InvalidOperation, OverflowError, Inexact, Rounded) as e_decimal_op:
        logger.error(f"adjust_precision: Erreur/Signal Decimal lors de l'ajustement de {value_decimal} "
                     f"(tick_size: {tick_size}, precision: {precision}): {type(e_decimal_op).__name__} - {e_decimal_op}. "
                     "Retour de la valeur originale (float).")
        return float(value_decimal) # Retourner la valeur originale en cas d'erreur d'opération Decimal spécifique
    except Exception as e_unexp: 
        logger.error(f"adjust_precision: Erreur inattendue lors de l'ajustement: {e_unexp}. "
                      "Retour de la valeur originale (float).", exc_info=True)
        return float(value_decimal)


def get_precision_from_filter(symbol_info: Dict[str, Any], filter_type: str, filter_key: str) -> Optional[int]:
    """
    Extrait la précision (nombre de décimales) d'un filtre de symbole.
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
        
        size_str_cleaned = size_str_raw.strip().rstrip('0') 

        try:
            with decimal_localcontext(DECIMAL_CONTEXT_HIGH_PRECISION): # Utilisation correcte
                d_value = Decimal(size_str_cleaned)
        except InvalidOperation:
            logger.warning(f"{log_prefix} Impossible de convertir '{filter_key}' ('{size_str_cleaned}') en Decimal.")
            return None

        if d_value.is_zero() or not d_value.is_finite():
            logger.warning(f"{log_prefix} {filter_key} '{size_str_cleaned}' est zéro, NaN ou Inf. Précision indéfinie.")
            return None
        
        exponent = d_value.normalize().as_tuple().exponent
        if isinstance(exponent, int):
            return abs(exponent) if exponent < 0 else 0
        else: 
            logger.warning(f"{log_prefix} Exponent non entier pour Decimal normalisé de '{size_str_cleaned}'. Ceci est inattendu.")
            return None

    except Exception as e: 
        logger.error(f"{log_prefix} Erreur inattendue : {e}", exc_info=True)
        return None


def get_filter_value(symbol_info: Dict[str, Any], filter_type: str, filter_key: str) -> Optional[float]:
    """
    Extrait une valeur numérique spécifique (convertie en float) d'un filtre de symbole.
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
            if value_raw is not None: 
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
    quantity: Union[float, Decimal, str],
    symbol_info: Dict[str, Any],
    rounding_mode_str: str = "ROUND_FLOOR" 
) -> float:
    """
    Ajuste une quantité pour qu'elle soit un multiple de `stepSize` du filtre `LOT_SIZE`.
    """
    log_prefix = f"[AdjustQtyToStep][{symbol_info.get('symbol', 'N/A')}]"
    step_size_val_raw = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize')

    if step_size_val_raw is None or step_size_val_raw <= 1e-12: 
        qty_precision_fallback = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
        final_precision_for_fallback = qty_precision_fallback if qty_precision_fallback is not None else 8
        
        logger.warning(f"{log_prefix} stepSize non trouvé ou invalide ({step_size_val_raw}). "
                       f"Retour de la quantité originale arrondie à {final_precision_for_fallback} décimales "
                       f"avec mode {rounding_mode_str}.")
        
        adjusted_qty_fallback = adjust_precision(
            value=quantity, 
            precision=final_precision_for_fallback, 
            rounding_mode_str=rounding_mode_str,
            tick_size=None 
        )
        original_qty_float = 0.0
        try:
            original_qty_float = float(str(quantity))
        except (ValueError, TypeError):
            logger.error(f"{log_prefix} Impossible de convertir la quantité originale '{quantity}' en float pour fallback. Retour de 0.0.")
            return 0.0
            
        return adjusted_qty_fallback if adjusted_qty_fallback is not None else round(original_qty_float, final_precision_for_fallback)

    adjusted_quantity_float = adjust_precision(
        value=quantity,
        precision=None, 
        rounding_mode_str=rounding_mode_str,
        tick_size=step_size_val_raw 
    )
    
    final_quantity = 0.0 
    try:
        final_quantity = float(str(quantity)) # Valeur par défaut si l'ajustement retourne None
    except (ValueError, TypeError):
        logger.error(f"{log_prefix} Impossible de convertir la quantité originale '{quantity}' en float pour fallback initial. Retour de 0.0.")
        return 0.0

    if adjusted_quantity_float is None:
         logger.warning(f"{log_prefix} adjust_precision a retourné None pour la quantité. "
                        f"Quantité originale: {quantity}, StepSize: {step_size_val_raw}. "
                        "Retour de la quantité originale non modifiée (convertie en float).")
    else:
        final_quantity = adjusted_quantity_float
        original_qty_float_for_log = 0.0
        try: original_qty_float_for_log = float(str(quantity))
        except: pass
        logger.debug(f"{log_prefix} Quantité brute: {original_qty_float_for_log:.8f}, StepSize: {step_size_val_raw}, "
                     f"Mode d'arrondi: {rounding_mode_str}, Quantité ajustée: {final_quantity:.8f}")

    return final_quantity


def validate_notional(
    quantity: Union[float, Decimal, str],
    price: Union[float, Decimal, str],
    symbol_info: Dict[str, Any],
    pair_symbol_for_log: Optional[str] = None
) -> bool:
    """
    Valide si la valeur notionnelle d'un ordre respecte le filtre MIN_NOTIONAL.
    """
    log_sym = pair_symbol_for_log or symbol_info.get('symbol', 'N/A_SYMBOL')
    log_prefix = f"[{log_sym}][ValidateNotional]"

    try:
        with decimal_localcontext(DECIMAL_CONTEXT_HIGH_PRECISION): 
            q_decimal = Decimal(str(quantity))
            p_decimal = Decimal(str(price))
    except InvalidOperation:
        logger.warning(f"{log_prefix} Quantité ({quantity}) ou prix ({price}) non convertible en Decimal.")
        return False

    if q_decimal.is_nan() or q_decimal.is_infinite() or p_decimal.is_nan() or p_decimal.is_infinite():
        logger.warning(f"{log_prefix} Quantité ({q_decimal}) ou prix ({p_decimal}) invalide (NaN/Inf).")
        return False

    min_notional_val_float = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
    if min_notional_val_float is None or min_notional_val_float <= 0:
        min_notional_val_float = get_filter_value(symbol_info, 'NOTIONAL', 'minNotional')

    if min_notional_val_float is not None and min_notional_val_float > 0:
        min_notional_decimal = Decimal(str(min_notional_val_float))
        current_notional = q_decimal.copy_abs() * p_decimal.copy_abs()

        if current_notional < (min_notional_decimal - DECIMAL_COMPARISON_EPSILON):
            logger.debug(f"{log_prefix} Échec : Notionnel actuel {current_notional} < Min requis {min_notional_decimal}")
            return False
        logger.debug(f"{log_prefix} Succès : Notionnel actuel {current_notional} >= Min requis {min_notional_decimal}")
        return True

    logger.debug(f"{log_prefix} Filtre MIN_NOTIONAL/NOTIONAL non trouvé, non applicable ou nul. Validation passée.")
    return True


def validate_order_parameters(
    order_params: Dict[str, Any], 
    symbol_info: Dict[str, Any],
    pair_symbol_for_log: Optional[str] = None,
    estimated_market_price_for_market_order: Optional[float] = None
) -> List[str]:
    """
    Valide les paramètres d'un ordre par rapport aux filtres de l'exchange.
    """
    log_sym = pair_symbol_for_log or symbol_info.get('symbol', 'N/A_SYMBOL')
    log_prefix = f"[{log_sym}][ValidateOrderParams]"
    errors: List[str] = []

    quantity_input = order_params.get('quantity')
    if quantity_input is None:
        errors.append("Paramètre 'quantity' manquant.")
        return errors

    try:
        with decimal_localcontext(DECIMAL_CONTEXT_HIGH_PRECISION): 
            quantity_decimal = Decimal(str(quantity_input))
    except InvalidOperation:
        errors.append(f"Quantité '{quantity_input}' n'est pas un nombre valide.")
        return errors

    if quantity_decimal.is_nan() or quantity_decimal.is_infinite() or quantity_decimal <= Decimal(0):
        errors.append(f"Quantité invalide '{quantity_decimal}'. Doit être positive et finie.")
        return errors 

    min_qty_filter_float = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
    max_qty_filter_float = get_filter_value(symbol_info, 'LOT_SIZE', 'maxQty')
    step_size_qty_filter_float = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize')

    if min_qty_filter_float is not None:
        min_qty_decimal = Decimal(str(min_qty_filter_float))
        if quantity_decimal < (min_qty_decimal - DECIMAL_COMPARISON_EPSILON):
            errors.append(f"Quantité {quantity_decimal} < minQty requis {min_qty_decimal}.")
    
    if max_qty_filter_float is not None:
        max_qty_decimal = Decimal(str(max_qty_filter_float))
        if quantity_decimal > (max_qty_decimal + DECIMAL_COMPARISON_EPSILON):
            errors.append(f"Quantité {quantity_decimal} > maxQty requis {max_qty_decimal}.")

    if step_size_qty_filter_float is not None and step_size_qty_filter_float > 1e-12:
        step_size_decimal = Decimal(str(step_size_qty_filter_float))
        remainder = quantity_decimal % step_size_decimal
        if not (remainder.is_zero() or remainder < DECIMAL_COMPARISON_EPSILON or abs(remainder - step_size_decimal) < DECIMAL_COMPARISON_EPSILON):
            errors.append(f"Quantité {quantity_decimal} ne respecte pas stepSize {step_size_decimal} (Reste: {remainder}).")

    order_type = str(order_params.get('type', "LIMIT")).upper()
    price_input = order_params.get('price')
    price_decimal: Optional[Decimal] = None

    if order_type != "MARKET" and price_input is not None:
        try:
            with decimal_localcontext(DECIMAL_CONTEXT_HIGH_PRECISION): 
                price_decimal = Decimal(str(price_input))
        except InvalidOperation:
            errors.append(f"Prix '{price_input}' n'est pas un nombre valide.")
            price_decimal = None 

        if price_decimal is not None: 
            if price_decimal.is_nan() or price_decimal.is_infinite() or price_decimal <= Decimal(0):
                errors.append(f"Prix '{price_decimal}' invalide. Doit être positif et fini.")
            else:
                min_price_filter_float = get_filter_value(symbol_info, 'PRICE_FILTER', 'minPrice')
                max_price_filter_float = get_filter_value(symbol_info, 'PRICE_FILTER', 'maxPrice')
                tick_size_price_filter_float = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')

                if min_price_filter_float is not None:
                    min_price_decimal = Decimal(str(min_price_filter_float))
                    if price_decimal < (min_price_decimal - DECIMAL_COMPARISON_EPSILON):
                        errors.append(f"Prix {price_decimal} < minPrice requis {min_price_decimal}.")
                
                if max_price_filter_float is not None and max_price_filter_float > 0: 
                    max_price_decimal = Decimal(str(max_price_filter_float))
                    if price_decimal > (max_price_decimal + DECIMAL_COMPARISON_EPSILON):
                        errors.append(f"Prix {price_decimal} > maxPrice requis {max_price_decimal}.")
                
                if tick_size_price_filter_float is not None and tick_size_price_filter_float > 1e-12:
                    tick_size_decimal = Decimal(str(tick_size_price_filter_float))
                    remainder_price = price_decimal % tick_size_decimal
                    if not (remainder_price.is_zero() or remainder_price < DECIMAL_COMPARISON_EPSILON or abs(remainder_price - tick_size_decimal) < DECIMAL_COMPARISON_EPSILON):
                        errors.append(f"Prix {price_decimal} ne respecte pas tickSize {tick_size_decimal} (Reste: {remainder_price}).")

    price_for_notional_check_float: Optional[float] = None
    if order_type == "MARKET":
        price_for_notional_check_float = estimated_market_price_for_market_order
        if price_for_notional_check_float is None:
            logger.debug(f"{log_prefix} Prix de marché estimé non fourni pour ordre MARKET. Validation MIN_NOTIONAL/NOTIONAL sautée/imprécise.")
    elif price_decimal is not None and price_decimal.is_finite() and price_decimal > Decimal(0): 
        price_for_notional_check_float = float(price_decimal)

    if price_for_notional_check_float is not None:
        # Assurer que quantity_decimal est défini avant d'appeler validate_notional
        if 'quantity_decimal' in locals() and quantity_decimal is not None:
            if not validate_notional(quantity_decimal, Decimal(str(price_for_notional_check_float)), symbol_info, log_sym):
                min_notional_report_float = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional') or \
                                            get_filter_value(symbol_info, 'NOTIONAL', 'minNotional')
                current_notional_val = float(quantity_decimal.copy_abs() * Decimal(str(price_for_notional_check_float)).copy_abs())
                errors.append(f"Valeur notionnelle ({current_notional_val:.4f}) "
                              f"< MIN_NOTIONAL/NOTIONAL requis ({min_notional_report_float if min_notional_report_float is not None else 'N/A'}).")
        else:
            # Ce cas ne devrait pas se produire si la logique précédente est correcte.
            logger.error(f"{log_prefix} quantity_decimal non défini avant l'appel à validate_notional.")


    if errors:
        logger.warning(f"{log_prefix} Échecs de validation de l'ordre : {'; '.join(errors)}")
    else:
        logger.debug(f"{log_prefix} Validation de l'ordre réussie.")
    return errors
