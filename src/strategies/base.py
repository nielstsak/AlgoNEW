import logging
import math
import time # Pour les clientOrderIds uniques
import uuid # Pour les clientOrderIds uniques
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List # List ajouté

import numpy as np
import pandas as pd

# Import des utilitaires d'exchange
# Tentative d'import relatif en premier (si base.py est dans un sous-module de src)
try:
    from ..utils.exchange_utils import (
        adjust_precision,
        get_filter_value,
        get_precision_from_filter
    )
except ImportError:
    # Fallback si l'import relatif échoue (ex: src est directement dans PYTHONPATH)
    try:
        from src.utils.exchange_utils import (
            adjust_precision,
            get_filter_value,
            get_precision_from_filter
        )
    except ImportError as e:
        # Fallback si src.utils introuvable, avec des fonctions factices pour permettre le chargement
        logging.getLogger(__name__).critical(
            f"CRITICAL ERROR: Could not import exchange_utils: {e}. "
            "Using dummy functions. Order precision and validation will be incorrect."
        )
        def adjust_precision(value, precision, method=None): return value # type: ignore
        def get_precision_from_filter(info, ftype, key): return 8 # type: ignore
        def get_filter_value(info, ftype, key): return None # type: ignore

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Classe de base abstraite pour toutes les stratégies de trading.
    Définit l'interface commune et fournit des méthodes utilitaires.
    Les stratégies concrètes doivent hériter de cette classe et implémenter
    les méthodes abstraites.
    """
    REQUIRED_PARAMS: List[str] = [] # Les sous-classes doivent définir cette liste

    def __init__(self, params: dict):
        """
        Initialise la stratégie avec les paramètres fournis.

        Args:
            params (dict): Dictionnaire des paramètres de la stratégie.
        """
        if not isinstance(params, dict):
            raise TypeError("Strategy parameters must be provided as a dictionary.")
        self.params = params
        self._signals: Optional[pd.DataFrame] = None
        
        # Validation des paramètres requis
        self._validate_strategy_params()
        
        logger.info(f"Strategy {self.__class__.__name__} initialized with params: {self.params}")

    def _validate_strategy_params(self):
        """
        Valide que tous les paramètres listés dans REQUIRED_PARAMS sont présents
        dans self.params et ne sont pas None.
        """
        # Utilise l'attribut de classe ou d'instance REQUIRED_PARAMS
        required_params_list = getattr(self, "REQUIRED_PARAMS", [])
        if not isinstance(required_params_list, list):
            logger.warning(f"REQUIRED_PARAMS for {self.__class__.__name__} is not a list. Skipping validation.")
            return
            
        missing = [p for p in required_params_list if self.get_param(p) is None]
        if missing:
            err_msg = f"Strategy {self.__class__.__name__}: Required parameters missing or None: {missing}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        logger.debug(f"Strategy {self.__class__.__name__}: All required parameters validated successfully.")


    @abstractmethod
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare le DataFrame pour la génération de signaux par la stratégie.

        Rôle Clarifié:
        Cette méthode reçoit un DataFrame `data` qui est censé contenir les colonnes
        d'indicateurs finales déjà calculées et nommées avec le suffixe `_strat`
        (ex: EMA_short_strat, ATR_strat). Ces calculs sont effectués en amont par
        ObjectiveEvaluator (backtesting) ou preprocessing_live.py (live), en utilisant
        les paramètres de fréquence et de période de la stratégie.

        La responsabilité principale de cette méthode dans une stratégie concrète est de :
        1. Vérifier la présence des colonnes `*_strat` attendues dans le `data` fourni.
        2. Si une colonne `*_strat` est manquante, elle DOIT être ajoutée avec `np.nan` pour
           éviter les `KeyError` en aval (dans `generate_signals` ou `generate_order_request`),
           et un avertissement DOIT être logué.
        3. Elle ne DOIT PAS recalculer les indicateurs primaires (comme EMA, MACD, ATR sur
           les K-lines agrégées) à partir des colonnes OHLCV de base.
        4. Elle PEUT effectuer des calculs très mineurs basés sur les colonnes `*_strat`
           déjà présentes si nécessaire (ex: une différence simple, un flag booléen).

        Args:
            data (pd.DataFrame): DataFrame d'entrée, typiquement avec un DatetimeIndex UTC.
                                 Attendues : OHLCV 1-min de base et colonnes d'indicateurs `*_strat`.

        Returns:
            pd.DataFrame: Le DataFrame `data` potentiellement modifié (colonnes `*_strat`
                          manquantes ajoutées avec NaN).
        """
        raise NotImplementedError("Subclasses must implement _calculate_indicators.")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> None:
        """
        Génère les signaux de trading (entry_long, entry_short, exit_long, exit_short)
        et les niveaux de Stop Loss (sl) et Take Profit (tp) théoriques.

        Le DataFrame `data` en entrée est celui retourné par `_calculate_indicators`
        (donc avec les colonnes `*_strat` vérifiées ou ajoutées comme NaN).
        Les signaux et SL/TP sont calculés en utilisant ces colonnes `*_strat` et les
        OHLCV 1-minute de base si nécessaire (ex: `close` pour référence de prix).
        Le résultat doit être stocké dans `self._signals` (un DataFrame avec l'index de `data`
        et les colonnes de signaux/sl/tp).

        Args:
            data (pd.DataFrame): DataFrame d'entrée préparé par `_calculate_indicators`.
        """
        raise NotImplementedError("Subclasses must implement generate_signals.")

    @abstractmethod
    def generate_order_request(self,
                               data: pd.DataFrame, # Données les plus récentes
                               symbol: str,
                               current_position: int, # 0:flat, >0:long, <0:short
                               available_capital: float,
                               symbol_info: dict # Infos de l'exchange (filtres, précisions)
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre d'entrée et les prix SL/TP bruts associés,
        basée sur la dernière situation de marché.

        Le DataFrame `data` contient les dernières données, incluant les colonnes `*_strat`
        (après appel à `_calculate_indicators` sur ces dernières données).
        La logique de décision doit être cohérente avec `generate_signals`.

        Args:
            data: DataFrame des données de marché les plus récentes.
            symbol: Symbole de trading.
            current_position: Position actuelle sur le marché.
            available_capital: Capital disponible pour le trading.
            symbol_info: Informations sur le symbole de l'exchange.

        Returns:
            Un tuple `(entry_order_params_dict, sl_tp_raw_prices_dict)` si un ordre
            doit être placé, où `sl_tp_raw_prices_dict` contient `{"sl_price": float, "tp_price": float}`.
            Retourne `None` si aucun ordre n'est généré.
        """
        raise NotImplementedError("Subclasses must implement generate_order_request.")

    def get_signals(self) -> pd.DataFrame:
        """
        Retourne le DataFrame des signaux générés.
        S'assure que les colonnes de signaux booléens sont bien booléennes et
        que les colonnes SL/TP sont numériques.
        """
        if self._signals is None:
            logger.error(f"Strategy {self.__class__.__name__}: generate_signals() must be called before get_signals(). Returning empty DataFrame.")
            return pd.DataFrame(columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])

        # Ensure boolean columns are boolean and fill NaNs with False
        bool_cols = ['entry_long', 'exit_long', 'entry_short', 'exit_short']
        for col in bool_cols:
            if col not in self._signals.columns:
                self._signals[col] = False
            else:
                self._signals.loc[:, col] = self._signals[col].fillna(False).astype(bool)
        
        # Ensure float columns (sl, tp) are numeric, coercing errors to NaN
        float_cols = ['sl', 'tp']
        for col in float_cols:
            if col not in self._signals.columns:
                self._signals[col] = np.nan
            else:
                self._signals.loc[:, col] = pd.to_numeric(self._signals[col], errors='coerce')
        return self._signals

    def get_params(self) -> Dict[str, Any]:
        """Retourne le dictionnaire des paramètres de la stratégie."""
        return self.params

    def get_param(self, key: str, default: Any = None) -> Any:
        """Retourne la valeur d'un paramètre spécifique, ou une valeur par défaut."""
        return self.params.get(key, default)

    def _calculate_quantity(self,
                            entry_price: float,
                            available_capital: float,
                            qty_precision: Optional[int],
                            symbol_info: dict,
                            symbol: Optional[str] = None
                           ) -> Optional[float]:
        """
        Calcule la quantité à trader en fonction du capital, du prix, du levier,
        de l'allocation de capital, et des filtres de l'exchange.

        Args:
            entry_price: Prix d'entrée théorique.
            available_capital: Capital disponible (équité actuelle).
            qty_precision: Précision requise pour la quantité de l'actif.
            symbol_info: Dictionnaire des informations du symbole de l'exchange.
            symbol: Symbole de trading (pour logging).

        Returns:
            La quantité ajustée, ou None si le calcul échoue ou ne respecte pas les filtres.
        """
        log_sym = symbol or self.get_param('symbol', 'N/A_SYMBOL') # Fallback
        log_prefix = f"[{self.__class__.__name__}][{log_sym}][_calc_qty]"

        if available_capital <= 0 or entry_price <= 0:
            logger.debug(f"{log_prefix} Capital disponible ({available_capital}) ou prix d'entrée ({entry_price}) invalide.")
            return None
        
        # Récupérer les paramètres de risque/levier depuis les paramètres de la stratégie
        # Ces paramètres devraient avoir été fusionnés avec les `simulation_defaults` par `ObjectiveEvaluator` ou `LiveTradingManager`
        leverage = float(self.get_param('margin_leverage', 1.0)) # Default à 1 si non trouvé
        capital_alloc_pct = float(self.get_param('capital_allocation_pct', 0.90)) # Default à 90% si non trouvé

        if not (0 < capital_alloc_pct <= 1.0):
            logger.warning(f"{log_prefix} capital_allocation_pct ({capital_alloc_pct}) invalide. Utilisation de 0.90.")
            capital_alloc_pct = 0.90
        if leverage <= 0:
            logger.warning(f"{log_prefix} margin_leverage ({leverage}) invalide. Utilisation de 1.0.")
            leverage = 1.0

        notional_target = available_capital * capital_alloc_pct * leverage
        raw_quantity = notional_target / entry_price
        logger.debug(f"{log_prefix} Capital: {available_capital:.2f}, Alloc%: {capital_alloc_pct:.2f}, Lev: {leverage:.1f} -> Notional Target: {notional_target:.2f}, Raw Qty: {raw_quantity:.8f}")

        if qty_precision is None:
            logger.warning(f"{log_prefix} qty_precision non fournie. Tentative de la déduire de LOT_SIZE/stepSize ou défaut à 8.")
            qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            if qty_precision is None: qty_precision = 8 # Fallback si toujours None

        adjusted_quantity = adjust_precision(raw_quantity, qty_precision, math.floor) # Toujours arrondir vers le bas pour la quantité
        
        if adjusted_quantity is None or adjusted_quantity <= 1e-9: # Utiliser un petit seuil pour les floats
            logger.debug(f"{log_prefix} Quantité ajustée ({adjusted_quantity}) est nulle ou invalide après application de la précision.")
            return None
        
        # Vérification des filtres LOT_SIZE
        min_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
        max_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'maxQty')
        step_size_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'stepSize') # Pour vérifier la conformité au pas

        if min_qty_filter is not None and adjusted_quantity < (min_qty_filter - 1e-9): # Tolérance pour float
            logger.debug(f"{log_prefix} Quantité ajustée {adjusted_quantity:.8f} < minQty {min_qty_filter:.8f}. Retour de None.")
            return None # Ou ajuster à minQty si la politique le permet, mais cela change le risque.
        
        if max_qty_filter is not None and adjusted_quantity > (max_qty_filter + 1e-9): # Tolérance pour float
            logger.debug(f"{log_prefix} Quantité ajustée {adjusted_quantity:.8f} > maxQty {max_qty_filter:.8f}. Ajustement à maxQty.")
            adjusted_quantity = adjust_precision(max_qty_filter, qty_precision, math.floor)
            if adjusted_quantity is None or (min_qty_filter is not None and adjusted_quantity < (min_qty_filter-1e-9)):
                logger.debug(f"{log_prefix} Quantité après ajustement à maxQty ({adjusted_quantity}) est invalide ou < minQty. Retour de None.")
                return None
        
        # Vérification de la conformité au stepSize (plus complexe, souvent géré par adjust_precision si bien fait)
        if step_size_filter is not None and step_size_filter > 1e-9:
            remainder = adjusted_quantity % step_size_filter
            # Si le reste n'est pas "proche de zéro" ou "proche de step_size_filter" (dû aux erreurs flottantes)
            if not (math.isclose(remainder, 0.0, abs_tol=1e-9) or math.isclose(remainder, step_size_filter, abs_tol=1e-9)):
                logger.warning(f"{log_prefix} Quantité ajustée {adjusted_quantity:.8f} ne respecte pas stepSize {step_size_filter:.8f}. Remainder: {remainder}. "
                               "Cela peut causer un rejet d'ordre. La fonction adjust_precision devrait idéalement gérer cela.")
                # Pourrait nécessiter un ré-ajustement plus fin ici, ou s'assurer que adjust_precision le fait.

        # Vérification du filtre MIN_NOTIONAL
        price_precision_for_notional = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
        if price_precision_for_notional is None: price_precision_for_notional = 8
        
        # Utiliser le prix d'entrée théorique (non slippé) pour la vérification du notionnel initial
        entry_price_for_notional_check = adjust_precision(entry_price, price_precision_for_notional, round)
        if entry_price_for_notional_check is None or entry_price_for_notional_check <= 0:
            logger.warning(f"{log_prefix} Prix d'entrée pour vérification notionnelle invalide ({entry_price_for_notional_check}).")
            return None

        if not self._validate_notional(adjusted_quantity, entry_price_for_notional_check, symbol_info):
            logger.debug(f"{log_prefix} Échec de la validation MIN_NOTIONAL. Quantité: {adjusted_quantity:.8f}, Prix: {entry_price_for_notional_check:.8f}")
            return None
            
        logger.info(f"{log_prefix} Quantité finale calculée: {adjusted_quantity:.8f}")
        return adjusted_quantity

    def _validate_notional(self, quantity: float, price: float, symbol_info: dict) -> bool:
        """Vérifie si la valeur notionnelle de l'ordre respecte le filtre MIN_NOTIONAL."""
        min_notional_filter = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
        # Certains exchanges utilisent 'NOTIONAL' au lieu de 'MIN_NOTIONAL' pour le filtre de notionnel minimum par ordre.
        if min_notional_filter is None:
            min_notional_filter = get_filter_value(symbol_info, 'NOTIONAL', 'minNotional') # Fallback

        if min_notional_filter is not None:
            notional_value = abs(quantity * price)
            # Utiliser une petite tolérance pour les comparaisons flottantes
            if notional_value < (min_notional_filter - 1e-9):
                logger.debug(f"Validation Notionnel: Échec. Valeur: {notional_value:.4f}, Min requis: {min_notional_filter:.4f}")
                return False
        logger.debug(f"Validation Notionnel: Succès. Valeur: {abs(quantity*price):.4f} (Min: {min_notional_filter or 'N/A'})")
        return True

    def _build_entry_params_formatted(self,
                                      symbol: str,
                                      side: str, # "BUY" ou "SELL"
                                      quantity_str: str, # Quantité déjà formatée en string
                                      entry_price_str: Optional[str] = None, # Prix formaté, requis pour LIMIT
                                      order_type: str = "LIMIT" # Ex: "LIMIT", "MARKET"
                                     ) -> Dict[str, Any]:
        """Construit le dictionnaire de paramètres pour un ordre d'entrée."""
        current_timestamp_ms = int(time.time() * 1000)
        # Générer un clientOrderId unique et conforme aux exigences de Binance (alphanumérique, certains caractères spéciaux)
        # Format typique: web_Abc123Def456 (python-binance le fait automatiquement si non fourni)
        # Pour plus de contrôle et de traçabilité :
        unique_id_part = uuid.uuid4().hex[:8] # Plus court pour ne pas dépasser les limites de longueur
        client_order_id = f"x_entry_{symbol[:3].lower()}{current_timestamp_ms % 100000}_{unique_id_part}"
        # Binance max length for newClientOrderId is 36 characters for some endpoints.
        client_order_id = client_order_id[:32] # Tronquer pour être sûr (certaines API ont 32, d'autres 36)

        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity_str,
            "newClientOrderId": client_order_id,
            "newOrderRespType": "FULL" # Pour obtenir tous les détails de l'ordre en réponse
        }

        if order_type.upper() in ["LIMIT", "LIMIT_MAKER"]:
            if not entry_price_str:
                raise ValueError(f"Prix d'entrée (entry_price_str) requis pour les ordres de type {order_type}.")
            params["price"] = entry_price_str
            params["timeInForce"] = "GTC" # Good Till Cancelled, commun pour les ordres LIMIT
        elif order_type.upper() == "MARKET":
            if "price" in params: del params["price"] # Le prix n'est pas spécifié pour les ordres MARKET
            if "timeInForce" in params: del params["timeInForce"]
        
        # Gestion de sideEffectType pour les comptes MARGIN (Cross ou Isolated)
        # Ceci est spécifique à l'API de marge de Binance.
        # Pour SPOT, ce paramètre n'est pas utilisé.
        # Le OrderExecutionClient devrait savoir s'il opère sur un compte de marge.
        # Ici, on assume que si la stratégie est utilisée avec un compte de marge, ces params sont pertinents.
        account_type_param = self.get_param('account_type_for_orders', 'MARGIN').upper() # Un param de strat ou global

        if account_type_param in ["MARGIN", "ISOLATED_MARGIN"]:
            if side.upper() == 'BUY':
                # MARGIN_BUY: Emprunte si nécessaire pour acheter, puis transfère l'achat vers le compte marge.
                # NO_SIDE_EFFECT: N'emprunte pas, utilise les fonds disponibles.
                # Pour une stratégie de marge typique qui utilise le levier, MARGIN_BUY est souvent utilisé.
                params["sideEffectType"] = "MARGIN_BUY"
            elif side.upper() == 'SELL':
                # Pour ouvrir une position SHORT sur marge: on vend un actif qu'on n'a pas, donc on l'emprunte.
                # AUTO_BORROW_REPAY: Emprunte l'asset de base si nécessaire pour shorter, et rembourse automatiquement
                #                    les dettes lors de la clôture de la position short (avec un ordre BUY).
                # MARGIN_SELL: Autre option pour shorter, peut nécessiter un emprunt manuel préalable selon l'exchange.
                # Pour fermer une position LONG, on utilise "SELL" avec sideEffectType="AUTO_REPAY".
                # Cette fonction est pour _ouvrir_ une position.
                # Si l'intention est d'ouvrir un SHORT:
                params["sideEffectType"] = "AUTO_BORROW_REPAY" # Ou MARGIN_SELL selon la logique de gestion des prêts.
                                                            # AUTO_BORROW_REPAY est plus simple pour le shorting.
        
        logger.debug(f"Paramètres d'ordre d'entrée formatés: {params}")
        return params

    def _build_oco_params(self,
                          symbol: str,
                          position_side: str, # "BUY" (pour position Long) ou "SELL" (pour position Short)
                          executed_qty: float, # Quantité de la position ouverte
                          sl_price_raw: float, # Prix SL brut (non ajusté)
                          tp_price_raw: float, # Prix TP brut (non ajusté)
                          price_precision: int,
                          qty_precision: int,
                          symbol_info: dict # Pourrait être utilisé pour des filtres OCO spécifiques
                         ) -> Optional[Dict[str, Any]]:
        """Construit le dictionnaire de paramètres pour un ordre OCO (One-Cancels-the-Other)."""
        log_prefix = f"[{self.__class__.__name__}][{symbol}][_build_oco]"

        oco_side = "SELL" if position_side.upper() == "BUY" else "BUY"
        quantity_str = f"{executed_qty:.{qty_precision}f}" # Quantité formatée
        
        # Récupérer tick_size pour des ajustements précis des prix SL/TP
        tick_size_str = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')
        if tick_size_str is None:
            logger.error(f"{log_prefix} tickSize non trouvé dans symbol_info. Impossible d'ajuster finement les prix OCO.")
            return None
        tick_size = float(tick_size_str)

        # Ajustement des prix SL et TP
        # Pour un OCO de VENTE (protégeant un LONG):
        #   - Take Profit (LIMIT order): Prix au-dessus du marché.
        #   - Stop Loss (STOP_LOSS_LIMIT order): stopPrice en dessous du marché, limitPrice légèrement inférieur au stopPrice.
        # Pour un OCO d'ACHAT (protégeant un SHORT):
        #   - Take Profit (LIMIT order): Prix en dessous du marché.
        #   - Stop Loss (STOP_LOSS_LIMIT order): stopPrice au-dessus du marché, limitPrice légèrement supérieur au stopPrice.

        tp_price_adjusted: Optional[float] = None
        sl_stop_price_adjusted: Optional[float] = None
        sl_limit_price_adjusted: Optional[float] = None # Pour STOP_LOSS_LIMIT

        # Paramètres OCO de la stratégie (peuvent être None)
        sl_order_type = self.get_param('otoco_params', {}).get('slOrderType', "STOP_LOSS_LIMIT").upper()
        tp_order_type = self.get_param('otoco_params', {}).get('tpOrderType', "LIMIT_MAKER").upper() # LIMIT_MAKER pour TP

        if oco_side == "SELL": # Protéger une position LONGUE
            tp_price_adjusted = adjust_precision(tp_price_raw, price_precision, math.floor) # Vendre HAUT, floor pour être conservateur
            sl_stop_price_adjusted = adjust_precision(sl_price_raw, price_precision, math.ceil) # Déclencher BAS, ceil pour être sûr
            if sl_order_type == "STOP_LOSS_LIMIT":
                # Mettre le limit un peu plus bas que le stop pour augmenter les chances de fill
                sl_limit_price_adjusted = adjust_precision(sl_price_raw - (tick_size * 5), price_precision, math.ceil) # Ex: 5 ticks plus bas
                if sl_limit_price_adjusted is not None and sl_stop_price_adjusted is not None and sl_limit_price_adjusted >= sl_stop_price_adjusted :
                     sl_limit_price_adjusted = adjust_precision(sl_stop_price_adjusted - tick_size, price_precision, math.ceil)


        elif oco_side == "BUY": # Protéger une position COURTE
            tp_price_adjusted = adjust_precision(tp_price_raw, price_precision, math.ceil) # Acheter BAS, ceil pour être conservateur
            sl_stop_price_adjusted = adjust_precision(sl_price_raw, price_precision, math.floor) # Déclencher HAUT, floor pour être sûr
            if sl_order_type == "STOP_LOSS_LIMIT":
                sl_limit_price_adjusted = adjust_precision(sl_price_raw + (tick_size * 5), price_precision, math.floor) # Ex: 5 ticks plus haut
                if sl_limit_price_adjusted is not None and sl_stop_price_adjusted is not None and sl_limit_price_adjusted <= sl_stop_price_adjusted:
                    sl_limit_price_adjusted = adjust_precision(sl_stop_price_adjusted + tick_size, price_precision, math.floor)


        if tp_price_adjusted is None or sl_stop_price_adjusted is None:
            logger.error(f"{log_prefix} Échec de l'ajustement des prix SL/TP pour OCO.")
            return None
        if sl_order_type == "STOP_LOSS_LIMIT" and sl_limit_price_adjusted is None:
            logger.error(f"{log_prefix} Échec de l'ajustement du prix limite SL pour OCO STOP_LOSS_LIMIT.")
            return None


        # Validation de base des prix (ex: TP doit être meilleur que SL par rapport au côté)
        if oco_side == "SELL" and tp_price_adjusted <= sl_stop_price_adjusted:
            logger.warning(f"{log_prefix} OCO Vente: TP ajusté ({tp_price_adjusted}) <= SL Stop ajusté ({sl_stop_price_adjusted}). Ordre potentiellement invalide.")
            # Pourrait retourner None ou tenter un ajustement (ex: écarter SL/TP)
            # return None
        elif oco_side == "BUY" and tp_price_adjusted >= sl_stop_price_adjusted:
            logger.warning(f"{log_prefix} OCO Achat: TP ajusté ({tp_price_adjusted}) >= SL Stop ajusté ({sl_stop_price_adjusted}). Ordre potentiellement invalide.")
            # return None
        
        # S'assurer que les prix limites pour STOP_LOSS_LIMIT sont valides
        if sl_order_type == "STOP_LOSS_LIMIT":
            if oco_side == "SELL" and sl_limit_price_adjusted >= sl_stop_price_adjusted : # Limit doit être <= Stop pour SELL
                logger.warning(f"{log_prefix} OCO Vente (STOP_LOSS_LIMIT): Prix Limite SL ({sl_limit_price_adjusted}) >= Prix Stop SL ({sl_stop_price_adjusted}). Ajustement forcé.")
                sl_limit_price_adjusted = adjust_precision(sl_stop_price_adjusted - tick_size, price_precision, math.ceil)
                if sl_limit_price_adjusted is None or sl_limit_price_adjusted <=0 : # Failsafe
                     logger.error(f"{log_prefix} Ajustement du prix limite SL a échoué ou est devenu invalide.")
                     return None

            elif oco_side == "BUY" and sl_limit_price_adjusted <= sl_stop_price_adjusted: # Limit doit être >= Stop pour BUY
                logger.warning(f"{log_prefix} OCO Achat (STOP_LOSS_LIMIT): Prix Limite SL ({sl_limit_price_adjusted}) <= Prix Stop SL ({sl_stop_price_adjusted}). Ajustement forcé.")
                sl_limit_price_adjusted = adjust_precision(sl_stop_price_adjusted + tick_size, price_precision, math.floor)
                if sl_limit_price_adjusted is None or sl_limit_price_adjusted <=0 : # Failsafe
                     logger.error(f"{log_prefix} Ajustement du prix limite SL a échoué ou est devenu invalide.")
                     return None


        sl_stop_price_str = f"{sl_stop_price_adjusted:.{price_precision}f}"
        tp_price_str = f"{tp_price_adjusted:.{price_precision}f}"
        sl_limit_price_str = f"{sl_limit_price_adjusted:.{price_precision}f}" if sl_limit_price_adjusted is not None else None


        current_timestamp_ms = int(time.time() * 1000)
        unique_id_part = uuid.uuid4().hex[:8]
        list_client_order_id = f"x_oco_{symbol[:3].lower()}{current_timestamp_ms % 100000}_{unique_id_part}"
        list_client_order_id = list_client_order_id[:32] # Tronquer pour être sûr

        oco_api_params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": oco_side, # "SELL" pour protéger un LONG, "BUY" pour protéger un SHORT
            "quantity": quantity_str,
            # Take Profit leg (LIMIT ou LIMIT_MAKER)
            "price": tp_price_str, # C'est le prix de l'ordre LIMIT pour le Take Profit
            # Stop Loss leg (STOP_LOSS, STOP_LOSS_LIMIT)
            "stopPrice": sl_stop_price_str, # Prix de déclenchement pour le Stop Loss
            "listClientOrderId": list_client_order_id,
            "newOrderRespType": "FULL"
        }
        
        # Paramètres spécifiques au type d'ordre Stop Loss
        if sl_order_type == "STOP_LOSS_LIMIT":
            if not sl_limit_price_str:
                logger.error(f"{log_prefix} stopLimitPrice est requis pour STOP_LOSS_LIMIT mais n'a pas pu être formaté.")
                return None
            oco_api_params["stopLimitPrice"] = sl_limit_price_str
            oco_api_params["stopLimitTimeInForce"] = "GTC" # Requis pour STOP_LOSS_LIMIT OCO
        # Si STOP_LOSS (ordre MARKET au déclenchement), stopLimitPrice n'est pas nécessaire.
        # Le SDK python-binance peut gérer cela implicitement si stopLimitPrice n'est pas fourni
        # pour un type d'ordre stop qui devient market. Vérifier la doc de l'API Binance pour OCO.
        # Typiquement, un OCO est un LIMIT (TP) vs un STOP_LOSS_LIMIT (SL).

        # Gestion de sideEffectType pour les comptes MARGIN
        account_type_param = self.get_param('account_type_for_orders', 'MARGIN').upper()
        if account_type_param in ["MARGIN", "ISOLATED_MARGIN"]:
            # Un OCO est pour clôturer une position, donc on veut rembourser tout prêt.
            oco_api_params["sideEffectType"] = "AUTO_REPAY"
        
        logger.debug(f"Paramètres OCO formatés: {oco_api_params}")
        return oco_api_params
