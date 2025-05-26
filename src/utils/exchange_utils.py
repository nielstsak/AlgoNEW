# src/utils/exchange_utils.py
"""
Ce module fournit des fonctions utilitaires pour gérer les filtres, les précisions
(prix, quantité) et les validations spécifiques aux exchanges (par exemple, Binance),
en se basant sur les informations de l'exchange.
"""

import logging
import math # Pour math.isclose si besoin, bien que Decimal ait sa propre logique
import numpy as np # Pour np.isnan, np.isinf
from typing import Dict, Optional, Any, Union, List, Callable # Callable sera remplacé par str pour rounding_mode
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP, ROUND_CEILING, ROUND_FLOOR # Importer les constantes

# Importer load_exchange_config si get_pair_config_for_symbol devait charger le fichier lui-même.
# Cependant, pour éviter les dépendances circulaires et suivre le pattern existant,
# nous supposerons que les données de l'exchange sont passées en tant que dictionnaire.
# from src.config.loader import load_exchange_config # Optionnel

logger = logging.getLogger(__name__)

# Une petite valeur epsilon pour les comparaisons de flottants généraux, si nécessaire.
# Pour les vérifications de multiples de tick/step, Decimal est privilégié.
FLOAT_COMPARISON_EPSILON = Decimal('1e-9') # Utiliser Decimal pour l'epsilon aussi

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
    precision: Optional[int] = None, # Nombre de décimales si pas de tick_size
    rounding_mode: str = ROUND_HALF_UP, # Constante Decimal.ROUND_* (ex: "ROUND_HALF_UP")
    tick_size: Optional[Union[float, str, Decimal]] = None
) -> Optional[float]:
    """
    Ajuste une valeur numérique à une précision décimale spécifiée ou à un multiple
    d'un `tick_size` donné. Utilise la classe `Decimal` pour une meilleure précision.

    Si `tick_size` est fourni, il a la priorité sur `precision` pour l'ajustement.
    La valeur sera ajustée pour être un multiple du `tick_size`.

    Args:
        value (Optional[Union[float, str, Decimal]]): La valeur à ajuster.
            Peut être float, int, str convertible en Decimal, ou Decimal.
        precision (Optional[int]): Le nombre de décimales souhaité si `tick_size`
            n'est pas utilisé. Ignoré si `tick_size` est fourni.
        rounding_mode (str): La méthode d'arrondi à utiliser, sous forme de chaîne
            correspondant aux constantes `Decimal.ROUND_*` (ex: "ROUND_FLOOR",
            "ROUND_HALF_UP", "ROUND_CEILING"). Par défaut "ROUND_HALF_UP".
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
        value_decimal = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        logger.warning(f"adjust_precision: Impossible de convertir la valeur '{value}' en Decimal.")
        return None # Ou lever une exception selon la politique de gestion des erreurs

    if value_decimal.is_nan() or value_decimal.is_infinite():
        logger.debug(f"adjust_precision: La valeur est NaN ou Inf ({value_decimal}). Retour de la valeur float originale.")
        return float(value) # NaN/Inf sont retournés tels quels

    # Vérifier si le rounding_mode est une constante Decimal valide
    actual_rounding_mode = getattr(Decimal, rounding_mode.upper(), None)
    if actual_rounding_mode is None:
        logger.warning(f"adjust_precision: Mode d'arrondi '{rounding_mode}' invalide. Utilisation de ROUND_HALF_UP par défaut.")
        actual_rounding_mode = ROUND_HALF_UP


    if tick_size is not None:
        try:
            tick_size_decimal = Decimal(str(tick_size))
            if tick_size_decimal.is_zero():
                logger.warning("adjust_precision: tick_size est zéro. L'ajustement par tick_size est impossible. Retour de la valeur originale arrondie à une haute précision.")
                # Retourner la valeur originale, peut-être arrondie à une précision par défaut élevée
                return float(value_decimal.quantize(Decimal('1e-8'), rounding=actual_rounding_mode))


            # Ajustement pour être un multiple de tick_size
            # (valeur / tick_size) -> arrondir à l'entier -> multiplier par tick_size
            # La méthode quantize de Decimal peut être utilisée pour cela.
            # Pour forcer à être un multiple, on peut faire :
            # quotient = (value_decimal / tick_size_decimal)
            # rounded_quotient = quotient.quantize(Decimal('1'), rounding=actual_rounding_mode)
            # adjusted_value_decimal = rounded_quotient * tick_size_decimal
            
            # Ou plus directement, si on veut que la valeur soit un multiple de tick_size,
            # et que le résultat ait la même "échelle" que le tick_size:
            # C'est-à-dire, si tick_size est "0.001", le résultat doit avoir 3 décimales.
            # Si value_decimal = 0.12345, tick_size = 0.001, rounding = ROUND_FLOOR
            # (0.12345 / 0.001) = 123.45. quantize(1, ROUND_FLOOR) -> 123. 123 * 0.001 = 0.123
            if tick_size_decimal.is_finite() and not tick_size_decimal.is_zero():
                 adjusted_value_decimal = (value_decimal / tick_size_decimal).quantize(Decimal('1'), rounding=actual_rounding_mode) * tick_size_decimal
            else: # tick_size invalide
                logger.error(f"adjust_precision: tick_size '{tick_size}' invalide pour l'ajustement. Retour de la valeur originale.")
                return float(value_decimal)

            return float(adjusted_value_decimal)

        except (InvalidOperation, ValueError, TypeError) as e_tick:
            logger.error(f"adjust_precision: Erreur lors de l'ajustement avec tick_size Decimal ('{tick_size}'): {e_tick}. "
                         "Retour de la valeur originale (float).")
            return float(value_decimal) # Fallback simple

    # Si tick_size n'est pas fourni, utiliser 'precision' (nombre de décimales)
    if precision is None or not isinstance(precision, int) or precision < 0:
        logger.debug(f"adjust_precision: Précision invalide ou manquante ({precision}) et pas de tick_size. "
                     f"Retour de la valeur originale (float) : {float(value_decimal)}.")
        return float(value_decimal) # Retourner la valeur originale si pas de règle d'ajustement claire

    # Créer le quantizer pour la précision (ex: Decimal('0.01') pour precision=2)
    quantizer = Decimal('1e-' + str(precision))
    try:
        adjusted_value_decimal = value_decimal.quantize(quantizer, rounding=actual_rounding_mode)
        return float(adjusted_value_decimal)
    except OverflowError: # Peut arriver si la valeur est trop grande pour la précision demandée
        logger.error(f"adjust_precision: OverflowError lors de l'ajustement de {value_decimal} avec précision {precision}. "
                     "Retour de la valeur originale (float).")
        return float(value_decimal)
    except Exception as e_prec: # pylint: disable=broad-except
        logger.error(f"adjust_precision: Erreur inattendue lors de l'ajustement à la précision {precision}: {e_prec}. "
                      "Retour de la valeur originale (float).")
        return float(value_decimal)


def get_precision_from_filter(symbol_info: Dict[str, Any], filter_type: str, filter_key: str) -> Optional[int]:
    """
    Extrait la précision (nombre de décimales) d'un filtre de symbole
    en se basant sur la valeur de `tickSize` ou `stepSize` (qui est une chaîne).

    Exemples :
    - "0.001" (tickSize/stepSize) -> précision 3
    - "1.0"   (tickSize/stepSize) -> précision 1
    - "1.000" (tickSize/stepSize) -> précision 3
    - "1"     (tickSize/stepSize) -> précision 0
    - "1e-5"  (tickSize/stepSize) -> précision 5 (0.00001)

    Args:
        symbol_info (Dict[str, Any]): Dictionnaire des informations du symbole de l'exchange.
        filter_type (str): Le type de filtre (ex: 'LOT_SIZE', 'PRICE_FILTER').
        filter_key (str): La clé dans le filtre contenant la valeur de précision
                          (typiquement 'stepSize' ou 'tickSize').

    Returns:
        Optional[int]: Le nombre de décimales, ou None si non trouvé, invalide,
                       ou si le tick/step est zéro.
    """
    log_prefix = f"[GetPrecision({filter_type}/{filter_key})]"
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
            logger.debug(f"{log_prefix} Filtre de type '{filter_type}' non trouvé pour le symbole '{symbol_info.get('symbol', 'N/A')}'.")
            return None

        size_str_raw = target_filter.get(filter_key)
        if not isinstance(size_str_raw, str) or not size_str_raw.strip():
            logger.warning(f"{log_prefix} Clé '{filter_key}' non trouvée, vide ou non-string dans le filtre '{filter_type}'. Valeur: '{size_str_raw}'")
            return None
        
        size_str_cleaned = size_str_raw.strip()

        # Convertir en Decimal pour gérer la notation scientifique et la normalisation
        try:
            d_value = Decimal(size_str_cleaned)
        except InvalidOperation:
            logger.warning(f"{log_prefix} Impossible de convertir '{filter_key}' ('{size_str_cleaned}') en Decimal pour le filtre {filter_type}.")
            return None

        if d_value.is_zero():
            logger.warning(f"{log_prefix} {filter_key} '{size_str_cleaned}' est zéro. La précision est indéfinie ou infinie pour un pas de zéro. Retourne None.")
            return None
        
        # Si la chaîne originale contient un point décimal, la précision est le nombre de chiffres après le point.
        if '.' in size_str_cleaned:
            # Ne pas utiliser rstrip('0') ici car "1.50" a une précision de 2, pas 1.
            return len(size_str_cleaned.split('.')[-1])
        else:
            # Pas de point décimal dans la chaîne originale (ex: "1", "100", ou "1e-5")
            # Utiliser l'exposant de la forme normalisée pour les cas comme "1e-5"
            # La normalisation enlève les zéros de fin inutiles pour la *valeur* du Decimal,
            # mais pour la *précision* d'un tick/step, la chaîne originale est plus fiable.
            # Cependant, si pas de '.', l'exposant est utile.
            # Pour "1", "10", l'exposant est 0. Pour "1E-5", l'exposant est -5.
            exponent = d_value.normalize().as_tuple().exponent
            if isinstance(exponent, int):
                return abs(exponent) if exponent < 0 else 0
            else: # Cas de NaN, Infinity (ne devrait pas arriver si d_value est un nombre fini non nul)
                logger.warning(f"{log_prefix} Impossible de déterminer la précision à partir de l'exposant Decimal pour '{size_str_cleaned}'.")
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
        filter_key (str): La clé dans le filtre contenant la valeur à extraire
                          (ex: 'minQty', 'minNotional', 'tickSize', 'stepSize').

    Returns:
        Optional[float]: La valeur flottante du filtre si trouvée et convertible,
                         None sinon.
    """
    log_prefix = f"[GetFilterVal({filter_type}/{filter_key})]"
    try:
        if not isinstance(symbol_info, dict):
            logger.warning(f"{log_prefix} symbol_info n'est pas un dictionnaire.")
            return None
        filters = symbol_info.get('filters', [])
        if not isinstance(filters, list):
            logger.warning(f"{log_prefix} 'filters' n'est pas une liste ou est manquant dans symbol_info.")
            return None

        target_filter = next((f for f in filters if isinstance(f,dict) and f.get('filterType') == filter_type), None)
        if target_filter:
            value_str = target_filter.get(filter_key)
            if value_str is not None: # Accepter "0", "0.0" etc.
                try:
                    return float(value_str)
                except (ValueError, TypeError):
                     logger.warning(f"{log_prefix} Impossible de convertir la valeur '{filter_key}' ('{value_str}') en float pour le filtre {filter_type}.")
                     return None
            # else: logger.debug(f"{log_prefix} Clé '{filter_key}' non trouvée dans le filtre '{filter_type}'.")
        # else: logger.debug(f"{log_prefix} Filtre de type '{filter_type}' non trouvé.")
        return None # Si filtre ou clé non trouvé
    except Exception as e: # pylint: disable=broad-except
        logger.error(f"{log_prefix} Erreur inattendue : {e}", exc_info=True)
        return None


def validate_notional(quantity: float, price: float, symbol_info: Dict[str, Any], pair_symbol_for_log: Optional[str] = None) -> bool:
    """
    Valide si la valeur notionnelle d'un ordre (quantité * prix) respecte
    le filtre MIN_NOTIONAL (ou NOTIONAL si MIN_NOTIONAL n'est pas trouvé) de l'exchange.

    Args:
        quantity (float): La quantité de l'ordre (en actif de base).
        price (float): Le prix de l'ordre (en actif de cotation).
        symbol_info (Dict[str, Any]): Dictionnaire des informations du symbole de l'exchange.
        pair_symbol_for_log (Optional[str]): Symbole de la paire pour des logs plus clairs.

    Returns:
        bool: `True` si la valeur notionnelle est valide ou si le filtre n'est pas
              spécifié/applicable, `False` sinon.
    """
    log_sym = pair_symbol_for_log or symbol_info.get('symbol', 'N/A_SYMBOL')
    log_prefix = f"[{log_sym}][ValidateNotional]"

    if np.isnan(quantity) or np.isinf(quantity) or np.isnan(price) or np.isinf(price):
        logger.warning(f"{log_prefix} Quantité ({quantity}) ou prix ({price}) invalide (NaN/Inf). Validation notionnelle échouée.")
        return False

    # Essayer MIN_NOTIONAL d'abord, puis NOTIONAL comme fallback
    min_notional_val = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
    if min_notional_val is None:
        min_notional_val = get_filter_value(symbol_info, 'NOTIONAL', 'minNotional') # Certains exchanges utilisent NOTIONAL

    if min_notional_val is not None and min_notional_val > 0:
        current_notional = abs(quantity * price)
        # Comparaison avec une petite tolérance pour les flottants
        if current_notional < (min_notional_val - float(FLOAT_COMPARISON_EPSILON)):
            logger.debug(f"{log_prefix} Échec : Notionnel actuel {current_notional:.4f} < Min requis {min_notional_val:.4f}")
            return False
        logger.debug(f"{log_prefix} Succès : Notionnel actuel {current_notional:.4f} >= Min requis {min_notional_val:.4f}")
        return True

    logger.debug(f"{log_prefix} Filtre MIN_NOTIONAL/NOTIONAL non trouvé ou non applicable (valeur <= 0). Validation notionnelle passée par défaut.")
    return True # Si le filtre n'est pas défini ou est <=0, on considère que c'est valide


def validate_order_parameters(
    order_params: Dict[str, Any], # Contient 'quantity', optionnellement 'price', 'type'
    symbol_info: Dict[str, Any],
    pair_symbol_for_log: Optional[str] = None,
    estimated_market_price_for_market_order: Optional[float] = None
) -> List[str]:
    """
    Valide les paramètres d'un ordre (quantité, prix pour les ordres LIMIT)
    par rapport aux filtres du symbole de l'exchange (LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL).

    Args:
        order_params (Dict[str, Any]): Dictionnaire contenant 'quantity' (str ou float)
            et optionnellement 'price' (str ou float), 'type' (str, ex: "MARKET", "LIMIT").
        symbol_info (Dict[str, Any]): Dictionnaire des informations du symbole de l'exchange.
        pair_symbol_for_log (Optional[str]): Symbole de la paire pour logs.
        estimated_market_price_for_market_order (Optional[float]): Prix estimé à utiliser
            pour la validation du notionnel des ordres MARKET.

    Returns:
        List[str]: Une liste de messages d'erreur. Si vide, l'ordre est valide.
    """
    log_sym = pair_symbol_for_log or symbol_info.get('symbol', 'N/A_SYMBOL')
    log_prefix = f"[{log_sym}][ValidateOrderParams]"
    errors: List[str] = []

    # --- Extraire et valider la quantité de base ---
    quantity_input = order_params.get('quantity')
    if quantity_input is None:
        errors.append("Paramètre 'quantity' manquant dans order_params.")
        return errors # Erreur bloquante

    try:
        quantity_decimal = Decimal(str(quantity_input))
        if quantity_decimal.is_nan() or quantity_decimal.is_infinite() or quantity_decimal <= Decimal(0):
            errors.append(f"Quantité invalide : '{quantity_input}'. Doit être un nombre positif.")
            return errors # Erreur bloquante
    except InvalidOperation:
        errors.append(f"Quantité '{quantity_input}' n'est pas un nombre valide.")
        return errors

    # --- Validation de la Quantité (LOT_SIZE filter) ---
    min_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
    max_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'maxQty')
    step_size_qty_str = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize') # C'est un float ici
    
    if min_qty_filter is not None and quantity_decimal < Decimal(str(min_qty_filter)) - FLOAT_COMPARISON_EPSILON:
        errors.append(f"Quantité {quantity_decimal} < minQty requis {min_qty_filter}")
    if max_qty_filter is not None and quantity_decimal > Decimal(str(max_qty_filter)) + FLOAT_COMPARISON_EPSILON:
        errors.append(f"Quantité {quantity_decimal} > maxQty requis {max_qty_filter}")

    if step_size_qty_str is not None: # step_size_qty_str est un float
        try:
            step_size_qty_decimal = Decimal(str(step_size_qty_str))
            if step_size_qty_decimal > Decimal(0):
                # Binance doc: (quantity - minQty) % stepSize == 0
                # Si min_qty_filter est None ou 0, on vérifie quantity % stepSize
                val_to_check_remainder = quantity_decimal
                if min_qty_filter is not None and min_qty_filter > 0:
                    # S'assurer que min_qty_filter est lui-même un multiple de step_size,
                    # sinon la formule de Binance est difficile à appliquer directement.
                    # Pour simplifier, on vérifie si la quantité elle-même est un multiple,
                    # après s'être assuré qu'elle est >= min_qty.
                    # La plupart des exchanges s'attendent à ce que la quantité soit un multiple de stepSize.
                    pass # La vérification quantity >= min_qty est déjà faite.

                remainder = val_to_check_remainder % step_size_qty_decimal
                # Vérifier si le reste est "effectivement" zéro en utilisant une tolérance
                # basée sur une fraction du step_size lui-même, ou un epsilon absolu.
                # Decimal.is_zero() est strict.
                # Une tolérance est nécessaire si quantity_decimal vient d'un float.
                if not remainder.is_zero():
                    # Si le reste est très proche de step_size_qty_decimal, c'est aussi acceptable (arrondi)
                    # ex: q=0.20000001, step=0.1, rem=0.00000001. is_zero()=False.
                    # ex: q=0.19999999, step=0.1, rem=0.09999999. is_zero()=False.
                    # On veut que (val_to_check_remainder / step_size_qty_decimal) soit un entier.
                    quotient = val_to_check_remainder / step_size_qty_decimal
                    if not quotient.quantize(Decimal('1e-8')).is_zero() and \
                       abs(quotient - quotient.to_integral_value(rounding=ROUND_HALF_UP)) > Decimal('1e-8'): # Si pas un entier à une petite tolérance près
                        errors.append(f"Quantité {quantity_decimal} (valeur pour modulo: {val_to_check_remainder}) "
                                      f"ne respecte pas stepSize {step_size_qty_decimal}. Reste: {remainder}, Quotient/Step: {quotient}")
        except InvalidOperation:
            errors.append(f"Erreur de conversion Decimal pour stepSize de quantité (val: {quantity_decimal}, step: {step_size_qty_str}).")


    # --- Validation du Prix (PRICE_FILTER) - Uniquement si ordre LIMIT et prix fourni ---
    order_type = str(order_params.get('type', "LIMIT")).upper() # Supposer LIMIT si non spécifié
    price_input = order_params.get('price')
    price_decimal: Optional[Decimal] = None

    if order_type != "MARKET" and price_input is not None:
        try:
            price_decimal = Decimal(str(price_input))
            if price_decimal.is_nan() or price_decimal.is_infinite() or price_decimal <= Decimal(0):
                errors.append(f"Prix invalide : '{price_input}'. Doit être un nombre positif.")
                price_decimal = None # Ne pas continuer la validation du prix si invalide
        except InvalidOperation:
            errors.append(f"Prix '{price_input}' n'est pas un nombre valide.")
            price_decimal = None

        if price_decimal is not None:
            min_price_filter = get_filter_value(symbol_info, 'PRICE_FILTER', 'minPrice')
            max_price_filter = get_filter_value(symbol_info, 'PRICE_FILTER', 'maxPrice')
            tick_size_price_str = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize') # C'est un float

            if min_price_filter is not None and price_decimal < Decimal(str(min_price_filter)) - FLOAT_COMPARISON_EPSILON:
                errors.append(f"Prix {price_decimal} < minPrice requis {min_price_filter}")
            if max_price_filter is not None and max_price_filter > 0 and price_decimal > Decimal(str(max_price_filter)) + FLOAT_COMPARISON_EPSILON:
                errors.append(f"Prix {price_decimal} > maxPrice requis {max_price_filter}")

            if tick_size_price_str is not None:
                try:
                    tick_size_price_decimal = Decimal(str(tick_size_price_str))
                    if tick_size_price_decimal > Decimal(0):
                        remainder_price = price_decimal % tick_size_price_decimal
                        if not remainder_price.is_zero():
                            quotient_price = price_decimal / tick_size_price_decimal
                            if abs(quotient_price - quotient_price.to_integral_value(rounding=ROUND_HALF_UP)) > Decimal('1e-8'):
                                errors.append(f"Prix {price_decimal} ne respecte pas tickSize {tick_size_price_decimal}. "
                                              f"Reste: {remainder_price}, Quotient/Tick: {quotient_price}")
                except InvalidOperation:
                     errors.append(f"Erreur de conversion Decimal pour tickSize du prix (prix: {price_decimal}, tick: {tick_size_price_str}).")

    # --- Validation du Notionnel (MIN_NOTIONAL) ---
    price_for_notional_check_float: Optional[float] = None
    if order_type == "MARKET":
        if estimated_market_price_for_market_order is not None:
            price_for_notional_check_float = estimated_market_price_for_market_order
        else: # Si pas de prix estimé pour MARKET, on ne peut pas valider le notionnel précisément ici.
            logger.debug(f"{log_prefix} Prix de marché estimé non fourni pour l'ordre MARKET. "
                         "La validation du notionnel MIN_NOTIONAL pourrait être imprécise ou sautée si le filtre existe "
                         "et si `applyMinToMarket` est true.")
    elif price_decimal is not None: # Pour les ordres LIMIT, utiliser le prix de l'ordre
        price_for_notional_check_float = float(price_decimal)

    if price_for_notional_check_float is not None:
        if not validate_notional(float(quantity_decimal), price_for_notional_check_float, symbol_info, pair_symbol_for_log):
            min_notional_val_report = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional') or \
                                      get_filter_value(symbol_info, 'NOTIONAL', 'minNotional') # Fallback
            errors.append(f"Valeur notionnelle ({abs(float(quantity_decimal) * price_for_notional_check_float):.4f}) "
                          f"< MIN_NOTIONAL/NOTIONAL requis ({min_notional_val_report:.4f if min_notional_val_report else 'N/A'})")

    if errors:
        logger.warning(f"{log_prefix} Validation de l'ordre échouée : {errors}")
    else:
        logger.debug(f"{log_prefix} Validation de l'ordre réussie.")
    return errors


def adjust_price_to_tick_size(price: float, symbol_info: Dict[str, Any], rounding_mode_str: str = "ROUND_HALF_UP") -> float:
    """
    Ajuste un prix pour qu'il soit un multiple de `tickSize` du filtre `PRICE_FILTER`.

    Args:
        price (float): Le prix à ajuster.
        symbol_info (Dict[str, Any]): Informations du symbole de l'exchange.
        rounding_mode_str (str): Méthode d'arrondi (ex: "ROUND_HALF_UP", "ROUND_FLOOR").

    Returns:
        float: Le prix ajusté. Retourne le prix original si `tickSize` n'est pas trouvé
               ou si une erreur se produit.
    """
    tick_size_val = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')
    if tick_size_val is None or tick_size_val <= 0:
        logger.warning(f"adjust_price_to_tick_size: tickSize non trouvé ou invalide pour {symbol_info.get('symbol','N/A')}. "
                       "Retour du prix original arrondi à une précision par défaut (8).")
        return round(price, 8) # Fallback simple si pas de tick_size

    adjusted_price_float = adjust_precision(
        value=price,
        precision=None, # La précision est implicite dans le tick_size
        rounding_mode=rounding_mode_str,
        tick_size=tick_size_val
    )
    return adjusted_price_float if adjusted_price_float is not None else price


def adjust_quantity_to_step_size(quantity: float, symbol_info: Dict[str, Any], rounding_mode_str: str = "ROUND_FLOOR") -> float:
    """
    Ajuste une quantité pour qu'elle soit un multiple de `stepSize` du filtre `LOT_SIZE`.
    Utilise typiquement `ROUND_FLOOR` pour les quantités afin d'être conservateur.

    Args:
        quantity (float): La quantité à ajuster.
        symbol_info (Dict[str, Any]): Informations du symbole de l'exchange.
        rounding_mode_str (str): Méthode d'arrondi (ex: "ROUND_FLOOR").

    Returns:
        float: La quantité ajustée. Retourne la quantité originale si `stepSize`
               n'est pas trouvé ou si une erreur se produit.
    """
    step_size_val = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize')
    if step_size_val is None or step_size_val <= 0:
        logger.warning(f"adjust_quantity_to_step_size: stepSize non trouvé ou invalide pour {symbol_info.get('symbol','N/A')}. "
                       "Retour de la quantité originale arrondie à une précision par défaut (8).")
        return round(quantity, 8) # Fallback si pas de step_size

    adjusted_quantity_float = adjust_precision(
        value=quantity,
        precision=None, # La précision est implicite dans le step_size
        rounding_mode=rounding_mode_str, # Typiquement ROUND_FLOOR pour les quantités
        tick_size=step_size_val
    )
    return adjusted_quantity_float if adjusted_quantity_float is not None else quantity

