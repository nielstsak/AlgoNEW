# src/strategies/base.py
"""
Ce module définit la classe de base abstraite pour toutes les stratégies de trading.
Elle implémente l'interface IStrategy, fournit une structure standardisée,
facilite la gestion des paramètres et du contexte, et intègre les
fonctionnalités de gestion d'état et d'événements.
"""
import logging
import uuid
import copy
import enum # Pour les types d'événements ou de régimes de marché
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List, Union, TypeVar, Generic, Callable

import pandas as pd
import numpy as np

# Imports depuis les modules du projet (s'attendre à ce qu'ils existent)
try:
    from src.core.interfaces import IStrategy, IEventDispatcher # IStrategy est maintenant un Protocol
    from src.utils.exchange_utils import (
        adjust_precision,
        adjust_quantity_to_step_size,
        get_precision_from_filter,
        get_filter_value
    )
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger(__name__).critical(
        f"BaseStrategy: Échec de l'importation des dépendances critiques (IStrategy, IEventDispatcher, exchange_utils): {e}. "
        "Vérifiez PYTHONPATH et l'existence des modules dans src.core et src.utils."
    )
    # Définir des placeholders pour permettre au module de se charger en cas d'erreur d'import
    class IStrategy: pass # type: ignore
    class IEventDispatcher: # type: ignore
        def register_listener(self, event_type: str, listener: Callable[[Dict[str, Any]], None]) -> None: ...
        def dispatch(self, event_type: str, event_data: Dict[str, Any]) -> None: ...

    def get_precision_from_filter(pair_config: Dict, filter_type: str, key: str) -> Optional[int]: return 8
    def adjust_precision(value: Union[float,str,Any], precision: Optional[int], tick_size: Optional[Union[float,str,Any]] = None, rounding_mode_str: str = "ROUND_HALF_UP") -> Optional[float]:
        if value is None: return None
        try: return round(float(str(value)), precision or 8)
        except: return None # pylint: disable=bare-except
    def adjust_quantity_to_step_size(quantity: Union[float,str,Any], symbol_info: Dict, qty_precision: Optional[int] = None, rounding_mode_str: str = "ROUND_FLOOR") -> Optional[float]:
        try: return round(float(str(quantity)), qty_precision or 8)
        except: return None # pylint: disable=bare-except
    def get_filter_value(symbol_info: Dict, filter_type: str, key: str) -> Optional[float]: return 0.0
    # Renvoyer l'erreur pour indiquer un problème de configuration du projet
    raise


logger = logging.getLogger(__name__)

# --- Dataclasses pour le contexte et la validation ---
@dataclass
class TradingContext:
    """
    Représente le contexte d'exécution d'une stratégie (pour backtest ou live).
    """
    pair_config: Dict[str, Any] # Informations de l'exchange pour la paire
    is_futures: bool
    leverage: int
    initial_equity: float # Ou capital disponible actuel pour le live
    account_type: Optional[str] = "SPOT" # Ex: SPOT, MARGIN, ISOLATED_MARGIN
    # D'autres champs contextuels peuvent être ajoutés (ex: current_time_utc)

@dataclass
class ValidationResult:
    """
    Résultat d'une opération de validation.
    """
    is_valid: bool = True
    messages: List[str] = field(default_factory=list)

    def add_error(self, message: str):
        self.is_valid = False
        self.messages.append(f"ERROR: {message}")

    def add_warning(self, message: str):
        # Les avertissements n'invalident pas nécessairement le résultat global
        self.messages.append(f"WARNING: {message}")

# --- Types d'événements (exemple) ---
class StrategyEventType(enum.Enum):
    """Types d'événements que la stratégie peut émettre ou écouter."""
    PARAMETER_CHANGED = "PARAMETER_CHANGED"
    MARKET_REGIME_SHIFT = "MARKET_REGIME_SHIFT"
    EXTERNAL_SIGNAL_RECEIVED = "EXTERNAL_SIGNAL_RECEIVED"
    # Ajouter d'autres types d'événements pertinents


class BaseStrategy(IStrategy, ABC): # Hérite maintenant de IStrategy
    """
    Classe de base abstraite pour toutes les stratégies de trading.
    Implémente l'interface IStrategy et fournit des fonctionnalités communes.
    """
    REQUIRED_PARAMS: List[str] = [] # Doit être défini par les sous-classes

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        """
        Initialise la stratégie de base.

        Args:
            strategy_name (str): Le nom clé de la stratégie (ex: "ma_crossover_strategy").
            symbol (str): Le symbole de la paire de trading (ex: BTCUSDT).
            params (Dict[str, Any]): Les paramètres spécifiques à cette instance de stratégie.
        """
        self.strategy_name: str = strategy_name
        self.symbol: str = symbol.upper()
        self.params: Dict[str, Any] = params.copy() # Travailler sur une copie
        self.log_prefix: str = f"[{self.strategy_name}][{self.symbol}]"

        # Attributs de contexte, initialisés par set_trading_context
        self.trading_context: Optional[TradingContext] = None
        self.price_precision: Optional[int] = None
        self.quantity_precision: Optional[int] = None
        self.base_asset: str = ""
        self.quote_asset: str = ""

        # Pour la gestion d'état
        self._internal_state: Dict[str, Any] = {} # Pour les variables d'état spécifiques à la stratégie

        # Pour la souscription aux événements
        self._event_dispatcher_ref: Optional[weakref.ReferenceType[IEventDispatcher]] = None
        self._subscribed_event_types: List[str] = []
        self._event_callbacks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}


        # Valider les paramètres spécifiques à la stratégie lors de l'initialisation
        try:
            self.validate_params() # Appel à la méthode (potentiellement surchargée)
            logger.info(f"{self.log_prefix} Stratégie initialisée avec les paramètres : {self.params}")
        except ValueError as e_val:
            logger.error(f"{self.log_prefix} Erreur de validation des paramètres lors de l'initialisation : {e_val}")
            raise # Renvoyer l'erreur pour arrêter la création si les params sont mauvais

    def get_param(self, param_name: str, default: Optional[Any] = None) -> Any:
        """Récupère une valeur de paramètre pour la stratégie."""
        return self.params.get(param_name, default)

    def set_trading_context(self, context: TradingContext) -> ValidationResult:
        """
        Configure le contexte de trading (backtest ou live).
        Cette méthode remplace et étend set_backtest_context.
        Elle valide également le contexte fourni.
        """
        self._log(f"Configuration du contexte de trading: {context}", level=2)
        validation_res = self.validate_context(context)
        if not validation_res.is_valid:
            self._log(f"Échec de la validation du contexte de trading: {validation_res.messages}", level=1, is_error=True)
            self.trading_context = None # Assurer que le contexte n'est pas partiellement setté
            return validation_res

        self.trading_context = context
        
        if self.trading_context.pair_config:
            self.base_asset = self.trading_context.pair_config.get('baseAsset', '')
            self.quote_asset = self.trading_context.pair_config.get('quoteAsset', '')
            self.price_precision = get_precision_from_filter(self.trading_context.pair_config, 'PRICE_FILTER', 'tickSize')
            self.quantity_precision = get_precision_from_filter(self.trading_context.pair_config, 'LOT_SIZE', 'stepSize')
            
            if self.price_precision is None:
                 self._log(f"Price precision (tickSize) non trouvée dans pair_config. Utilisation d'une valeur par défaut ou arrondi standard.", level=2, is_warning=True)
            if self.quantity_precision is None:
                 self._log(f"Quantity precision (stepSize) non trouvée dans pair_config. Utilisation d'une valeur par défaut ou arrondi standard.", level=2, is_warning=True)
            self._log(f"Contexte de trading défini. PricePrec: {self.price_precision}, QtyPrec: {self.quantity_precision}, Leverage: {self.trading_context.leverage}", level=2)
        else:
            self._log(f"pair_config non fourni dans le contexte. Les précisions ne peuvent pas être déterminées.", level=1, is_error=True)
            validation_res.add_error("pair_config manquant dans TradingContext.")
        
        return validation_res

    def validate_context(self, context: TradingContext) -> ValidationResult:
        """
        Valide le contexte de trading fourni.
        Les sous-classes peuvent surcharger pour ajouter des validations spécifiques.
        """
        self._log(f"Validation du contexte de trading: {context}", level=3)
        report = ValidationResult()
        if not isinstance(context, TradingContext):
            report.add_error("Le contexte fourni n'est pas une instance de TradingContext.")
            return report # Erreur bloquante

        if not context.pair_config or not isinstance(context.pair_config, dict):
            report.add_error("'pair_config' est manquant ou invalide dans le contexte.")
        if not (isinstance(context.leverage, int) and context.leverage >= 1):
            report.add_error(f"'leverage' ({context.leverage}) doit être un entier >= 1.")
        if not (isinstance(context.initial_equity, (int, float)) and context.initial_equity > 0):
            report.add_error(f"'initial_equity' ({context.initial_equity}) doit être un nombre positif.")
        
        # Vérifier la cohérence du symbole si possible
        if context.pair_config and context.pair_config.get('symbol', '').upper() != self.symbol:
            report.add_warning(f"Le symbole dans pair_config ('{context.pair_config.get('symbol')}') "
                               f"ne correspond pas au symbole de la stratégie ('{self.symbol}').")
        return report

    def get_meta_parameters(self) -> Dict[str, Any]:
        """
        Retourne des méta-paramètres descriptifs de la stratégie.
        Les sous-classes devraient surcharger pour fournir des valeurs significatives.
        """
        # Valeurs par défaut ou dérivées des self.params
        risk_level = "medium" # Peut être "low", "medium", "high"
        time_horizon = "short_term" # "intraday", "short_term", "medium_term", "long_term"
        market_regime_assumption = "any" # "trending", "ranging", "volatile", "any"

        # Exemple de dérivation (très basique)
        if "atr_period" in self.params and self.params["atr_period"] > 20:
            time_horizon = "medium_term"
        if "stop_loss_mult" in self.params and self.params["stop_loss_mult"] < 1.0:
            risk_level = "high"
            
        return {
            "risk_level_profile": risk_level,
            "typical_time_horizon": time_horizon,
            "market_regime_assumption": market_regime_assumption,
            "strategy_class_name": self.__class__.__name__
        }

    def subscribe_to_events(self, event_dispatcher: 'IEventDispatcher', event_types: List[Union[str, StrategyEventType]]) -> None:
        """
        Permet à la stratégie de s'abonner à des types d'événements spécifiques
        via un EventDispatcher.
        """
        self._log(f"Demande de souscription aux événements: {event_types}", level=2)
        if not hasattr(event_dispatcher, 'register_listener') or not callable(event_dispatcher.register_listener):
            self._log("EventDispatcher fourni n'a pas de méthode 'register_listener'. Souscription impossible.", level=1, is_error=True)
            return

        self._event_dispatcher_ref = weakref.ref(event_dispatcher)
        self._subscribed_event_types = []
        for et_raw in event_types:
            event_type_str = et_raw.value if isinstance(et_raw, StrategyEventType) else str(et_raw)
            self._subscribed_event_types.append(event_type_str)
            # La stratégie doit avoir une méthode on_event(self, event_type_str, event_data)
            # ou des méthodes spécifiques comme on_parameter_changed_event(...)
            # Pour cet exemple, on suppose une méthode on_event générique.
            if hasattr(self, 'on_event') and callable(getattr(self, 'on_event')):
                # Le listener est une méthode liée de cette instance de stratégie
                bound_listener_method = functools.partial(self.on_event, event_type_str)
                
                # Enregistrer le listener auprès du dispatcher
                # Le dispatcher appellera bound_listener_method(event_data)
                # event_dispatcher.register_listener(event_type_str, bound_listener_method) # type: ignore
                # Pour une meilleure gestion, on pourrait stocker les callbacks par type d'événement
                if event_type_str not in self._event_callbacks:
                    self._event_callbacks[event_type_str] = []
                self._event_callbacks[event_type_str].append(self.on_event) # Stocker la méthode non liée
                
                # L'enregistrement réel se fait via le dispatcher, qui appellera on_event
                # avec les bons arguments. La logique ici est plus pour savoir à quoi on est abonné.
                # Le mécanisme d'appel réel dépend de l'implémentation de IEventDispatcher.
                # Si le dispatcher appelle une méthode fixe (ex: handle_event(type, data)),
                # alors la stratégie doit implémenter cela.
                # Ici, on suppose que la stratégie a une méthode `on_event(self, event_type: str, event_data: Dict[str, Any])`
                # et que le dispatcher sait comment l'appeler.
                self._log(f"Souscription (conceptuelle) à '{event_type_str}'. La stratégie doit implémenter on_event.", level=3)
            else:
                self._log(f"La stratégie n'a pas de méthode 'on_event' pour gérer '{event_type_str}'.", level=2, is_warning=True)
        
        if self._subscribed_event_types:
            self._log(f"Stratégie configurée pour écouter les types d'événements: {self._subscribed_event_types}", level=2)


    def on_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Méthode de callback générique appelée par l'EventDispatcher lorsqu'un
        événement souscrit se produit. Les sous-classes peuvent surcharger ceci
        ou implémenter des handlers plus spécifiques.
        """
        self._log(f"Événement reçu: Type='{event_type}', Données='{str(event_data)[:100]}...'", level=2)
        # Logique de traitement de l'événement ici
        if event_type == StrategyEventType.PARAMETER_CHANGED.value:
            changed_params = event_data.get("changed_parameters", {})
            self._log(f"Notification de changement de paramètres reçue: {changed_params}", level=1)
            # Mettre à jour self.params et potentiellement invalider des caches internes
            self.params.update(changed_params)
            self.validate_params() # Revalider après changement
            # Invalider des caches internes si nécessaire
            # self._cached_property_value = None
        # Gérer d'autres types d'événements...

    def get_state(self) -> Dict[str, Any]:
        """
        Sérialise l'état interne pertinent de la stratégie pour la persistance ou la reprise.
        Les sous-classes doivent surcharger pour ajouter leur état spécifique.
        """
        state = {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "params": copy.deepcopy(self.params), # Copie profonde des paramètres
            "internal_state_variables": copy.deepcopy(self._internal_state),
            # Ne pas sérialiser trading_context directement car il contient des DataFrames
            # et est généralement reconstruit. On pourrait sérialiser des éléments clés du contexte si nécessaire.
            "last_signal_details": self._internal_state.get("last_signal_info_for_state") # Exemple
        }
        self._log("Récupération de l'état de la stratégie.", level=3)
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restaure l'état interne de la stratégie à partir d'un dictionnaire.
        Les sous-classes doivent surcharger pour restaurer leur état spécifique.
        """
        self._log(f"Configuration de l'état de la stratégie à partir de: {list(state.keys())}", level=3)
        if state.get("strategy_name") != self.strategy_name or state.get("symbol") != self.symbol:
            msg = "Tentative de restauration d'un état pour une stratégie/symbole différent."
            self._log(msg, level=1, is_error=True)
            raise ValueError(msg)
        
        self.params = copy.deepcopy(state.get("params", self.params))
        self._internal_state = copy.deepcopy(state.get("internal_state_variables", {}))
        
        # Revalider les paramètres après restauration
        try:
            self.validate_params()
        except ValueError as e_reval:
            self._log(f"Erreur de validation des paramètres après restauration de l'état: {e_reval}", level=1, is_error=True)
            # Laisser l'exception se propager ou gérer plus finement
            raise
        self._log("État de la stratégie restauré et paramètres revalidés.", level=2)


    @abstractmethod
    def validate_params(self) -> None:
        """Valide les paramètres spécifiques à la stratégie (self.params)."""
        pass

    @abstractmethod
    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        """Déclare les indicateurs requis par la stratégie et leurs configurations."""
        return []

    @abstractmethod
    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        """
        Assure que le data_feed contient les colonnes d'indicateurs finales
        nécessaires pour _generate_signals. Peut effectuer des calculs ou des
        transformations supplémentaires sur les indicateurs déjà présents.
        """
        pass

    @abstractmethod
    def _generate_signals(
        self,
        data_with_indicators: pd.DataFrame,
        current_position_open: bool,
        current_position_direction: int,
        current_entry_price: float
    ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Génère les signaux de trading (achat, vente, sortie, hold).
        C'est le cœur de la logique de la stratégie.
        """
        pass

    def get_signal(
        self,
        data_feed: pd.DataFrame, # Données OHLCV brutes (ou déjà avec certains indicateurs de base)
        current_position_open: bool,
        current_position_direction: int,
        current_entry_price: float,
        current_equity: float # Ajouté pour permettre des calculs de taille de position basés sur l'équité
    ) -> Dict[str, Any]:
        """
        Méthode principale appelée par le simulateur pour obtenir les décisions de trading.
        """
        self._log(f"Appel de get_signal. Position ouverte: {current_position_open}, Dir: {current_position_direction}, "
                  f"Px Entrée: {current_entry_price:.4f}, Équité: {current_equity:.2f}", level=3)
        
        # Vérification du contexte (supposé avoir été setté via set_trading_context)
        if not self.trading_context:
            msg = "TradingContext non défini. Appelez set_trading_context avant get_signal."
            self._log(msg, level=1, is_error=True)
            # Retourner un signal HOLD par défaut en cas d'erreur de contexte
            return self._get_default_signal_response(reason="Contexte de trading non initialisé.")

        # Valider le contexte à chaque appel peut être coûteux.
        # Alternative: le valider une fois lors de set_trading_context.
        # Pour la robustesse, on peut le revalider ici ou avoir un flag.
        # context_validation = self.validate_context(self.trading_context)
        # if not context_validation.is_valid:
        #     self._log(f"Contexte de trading invalide pour get_signal: {context_validation.messages}", level=1, is_error=True)
        #     return self._get_default_signal_response(reason=f"Contexte invalide: {context_validation.messages[0]}")

        if data_feed.empty:
            self._log("data_feed vide reçu dans get_signal. Retour de HOLD.", level=2, is_warning=True)
            return self._get_default_signal_response(reason="Data feed vide.")

        # 1. Calculer/assurer les indicateurs (méthode (potentiellement) surchargée par la sous-classe)
        #    _calculate_indicators est responsable de s'assurer que toutes les colonnes
        #    nécessaires pour _generate_signals sont présentes et correctes.
        #    Elle reçoit data_feed qui peut déjà contenir des indicateurs pré-calculés
        #    par un IndicatorCalculator externe.
        try:
            # Utiliser une vue pour les opérations de lecture si possible, copier si modification
            data_ready_for_signals = self._calculate_indicators(data_feed) # Pas de .copy() ici, la méthode doit gérer
        except Exception as e_calc:
            self._log(f"Erreur dans _calculate_indicators : {e_calc}", level=1, is_error=True)
            return self._get_default_signal_response(reason=f"Erreur calcul indicateurs: {str(e_calc)[:100]}")

        if data_ready_for_signals.empty:
            self._log("DataFrame vide après _calculate_indicators. Retour de HOLD.", level=2, is_warning=True)
            return self._get_default_signal_response(reason="Données vides post-calcul indicateurs.")

        # 2. Générer les signaux (méthode (potentiellement) surchargée par la sous-classe)
        try:
            signal_type, order_type_pref, limit_sugg, sl_sugg, tp_sugg, pos_size_pct_sugg = \
                self._generate_signals(data_ready_for_signals,
                                       current_position_open,
                                       current_position_direction,
                                       current_entry_price)
        except Exception as e_gen_sig:
            self._log(f"Erreur dans _generate_signals : {e_gen_sig}", level=1, is_error=True)
            return self._get_default_signal_response(reason=f"Erreur génération signaux: {str(e_gen_sig)[:100]}")

        # Préparer la réponse
        # Utiliser les valeurs par défaut de la config si non spécifiées par _generate_signals
        final_order_type = order_type_pref if order_type_pref is not None else self.get_param("order_type_preference", "MARKET")
        final_pos_size_pct = pos_size_pct_sugg if pos_size_pct_sugg is not None else self.get_param('capital_allocation_pct', 1.0)

        # Ajuster les prix SL/TP à la précision de l'exchange
        # Le prix limite est déjà supposé être "ajustable" ou est un prix de marché
        final_sl = None
        if sl_sugg is not None and self.trading_context and self.price_precision is not None:
            tick_size_sl = get_filter_value(self.trading_context.pair_config, 'PRICE_FILTER', 'tickSize')
            final_sl = adjust_precision(sl_sugg, self.price_precision, tick_size=tick_size_sl)
        elif sl_sugg is not None: # Fallback si pas de contexte complet
            final_sl = round(sl_sugg, self.price_precision or 8)

        final_tp = None
        if tp_sugg is not None and self.trading_context and self.price_precision is not None:
            tick_size_tp = get_filter_value(self.trading_context.pair_config, 'PRICE_FILTER', 'tickSize')
            final_tp = adjust_precision(tp_sugg, self.price_precision, tick_size=tick_size_tp)
        elif tp_sugg is not None:
            final_tp = round(tp_sugg, self.price_precision or 8)
            
        final_limit = None
        if limit_sugg is not None and self.trading_context and self.price_precision is not None:
            tick_size_limit = get_filter_value(self.trading_context.pair_config, 'PRICE_FILTER', 'tickSize')
            final_limit = adjust_precision(limit_sugg, self.price_precision, tick_size=tick_size_limit)
        elif limit_sugg is not None:
            final_limit = round(limit_sugg, self.price_precision or 8)

        signal_response = {
            "signal": signal_type, # 1: BUY, -1: SELL (short), 2: EXIT, 0: HOLD
            "order_type": str(final_order_type),
            "limit_price": final_limit,
            "sl_price": final_sl,
            "tp_price": final_tp,
            "position_size_pct": float(final_pos_size_pct), # Assurer float
            # Les champs suivants sont pour le logging OOS détaillé par le simulateur
            "entry_order_params_theoretical_for_oos_log": None,
            "oco_params_theoretical_for_oos_log": None
        }
        # Stocker pour get_state si besoin
        self._internal_state["last_signal_info_for_state"] = signal_response.copy()
        
        self._log(f"Signal généré: {signal_response}", level=3)
        return signal_response

    def _get_default_signal_response(self, reason: str) -> Dict[str, Any]:
        """Retourne une réponse de signal HOLD par défaut avec une raison."""
        self._log(f"Retour de HOLD par défaut. Raison: {reason}", level=2, is_warning=True)
        return {
            "signal": 0, "order_type": "MARKET", "limit_price": None,
            "sl_price": None, "tp_price": None, "position_size_pct": 1.0,
            "entry_order_params_theoretical_for_oos_log": None,
            "oco_params_theoretical_for_oos_log": None,
            "info": reason
        }

    def _calculate_quantity(self,
                            entry_price: float,
                            available_capital: float,
                            qty_precision: Optional[int], # Précision de la quantité (nombre de décimales)
                            symbol_info: Dict[str, Any], # pair_config de l'exchange
                            symbol: str, # Pour le logging
                            position_size_pct: Optional[float] = None # % du capital à risquer/allouer
                           ) -> Optional[float]:
        """
        Calcule la quantité d'actifs de base à trader, en respectant les filtres de l'exchange.
        Optimisé pour éviter les boucles Python.
        """
        calc_log_prefix = f"{self.log_prefix}[CalcQty]"
        if entry_price <= 1e-9: # Seuil pour éviter division par zéro ou prix invalides
            self._log(f"Prix d'entrée ({entry_price}) invalide pour calcul de quantité.", level=1, is_error=True)
            return None
        if available_capital <= 1e-9:
            self._log(f"Capital disponible ({available_capital}) nul ou négatif. Quantité nulle.", level=2, is_warning=True)
            return 0.0

        # Utiliser le % de taille de position du signal, sinon celui des params de la strat
        actual_pos_size_pct = position_size_pct if position_size_pct is not None else self.get_param('capital_allocation_pct', 1.0)
        
        if not (isinstance(actual_pos_size_pct, (float, int)) and 0 < actual_pos_size_pct <= 1.0):
            self._log(f"position_size_pct ({actual_pos_size_pct}) invalide. Utilisation de 100% (1.0).", level=2, is_warning=True)
            actual_pos_size_pct = 1.0
        
        # Le levier est maintenant dans self.trading_context
        leverage_to_use = self.trading_context.leverage if self.trading_context else 1
        
        capital_for_this_trade = available_capital * actual_pos_size_pct
        total_position_value_quote = capital_for_this_trade * leverage_to_use # Valeur notionnelle
        
        quantity_base_raw = total_position_value_quote / entry_price
        
        self._log(f"Calcul Qty: CapitalDisp={available_capital:.2f}, Alloc%={actual_pos_size_pct:.2%}, "
                  f"Levier={leverage_to_use}x, CapPourTrade={capital_for_this_trade:.2f}, "
                  f"ValPosNotionnelleQt={total_position_value_quote:.2f}, PxEntrée={entry_price:.{self.price_precision or 8}f}, "
                  f"QtyBaseBrute={quantity_base_raw:.8f}", level=3)

        if qty_precision is None:
            self._log(f"qty_precision non disponible pour {symbol}. La quantité ne sera pas finement ajustée.", level=2, is_warning=True)
            # Fallback sur un arrondi standard si la précision n'est pas connue
            # return round(quantity_base_raw, 8) # Exemple de fallback, mais adjust_quantity_to_step_size est mieux

        # Utiliser la fonction utilitaire pour appliquer stepSize et minQty
        adjusted_quantity_base = adjust_quantity_to_step_size(
            quantity_base_raw,
            symbol_info, # C'est le pair_config
            qty_precision=qty_precision # Passer la précision pour l'arrondi final si stepSize n'est pas le facteur limitant
        )
        
        if adjusted_quantity_base is None or adjusted_quantity_base <= 1e-9:
            self._log(f"Quantité ajustée ({adjusted_quantity_base}) nulle ou invalide après step_size. Qty brute: {quantity_base_raw:.8f}", level=2, is_warning=True)
            return 0.0

        # Vérifier minNotional
        min_notional_filter = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional') or \
                              get_filter_value(symbol_info, 'NOTIONAL', 'minNotional') # NOTIONAL est aussi utilisé par Binance pour minNotional
        if min_notional_filter is not None and min_notional_filter > 0:
            current_notional_value = adjusted_quantity_base * entry_price
            if current_notional_value < min_notional_filter:
                self._log(f"Valeur notionnelle de l'ordre ({current_notional_value:.2f}) "
                          f"< MIN_NOTIONAL requis ({min_notional_filter:.2f}). Quantité mise à 0.", level=2, is_warning=True)
                return 0.0
            
        self._log(f"Quantité finale calculée pour {symbol}: {adjusted_quantity_base:.{qty_precision or 8}f} "
                  f"{self.base_asset if self.base_asset else ''}", level=2)
        return adjusted_quantity_base

    def _build_entry_params_formatted(self,
                                      side: str,
                                      quantity_str: str,
                                      order_type: str,
                                      entry_price_str: Optional[str] = None,
                                      time_in_force: Optional[str] = None,
                                      new_client_order_id: Optional[str] = None
                                     ) -> Dict[str, Any]:
        """Construit le dictionnaire de paramètres pour un ordre d'entrée (pour log OOS ou exécution réelle)."""
        # ... (logique existante, s'assurer qu'elle est propre et utilise self.symbol) ...
        if new_client_order_id is None:
            # Générer un ID client unique et informatif
            strat_short = self.strategy_name[:min(len(self.strategy_name), 7)].lower().replace("_", "")
            sym_short = self.symbol[:min(len(self.symbol),5)].lower().replace("/", "")
            ts_short = str(int(pd.Timestamp.now(tz='UTC').timestamp() * 1000))[-7:] # Plus de chiffres pour unicité
            uuid_short = str(uuid.uuid4().hex)[:6] # Plus long pour unicité
            new_client_order_id = f"sim_{strat_short}_{sym_short}_{ts_short}_{uuid_short}"
            new_client_order_id = new_client_order_id[:36] # Limite de certains exchanges

        params: Dict[str, Any] = {
            "symbol": self.symbol, # Utiliser self.symbol
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity_str,
            "newClientOrderId": new_client_order_id
        }
        if order_type.upper() == "LIMIT":
            if entry_price_str is None:
                msg = f"Le prix (entry_price_str) doit être spécifié pour un ordre LIMIT. Reçu: {entry_price_str}"
                self._log(msg, level=1, is_error=True)
                raise ValueError(msg)
            params["price"] = entry_price_str
        
        if time_in_force: # Ex: "GTC", "IOC", "FOK"
            params["timeInForce"] = time_in_force

        self._log(f"Paramètres d'ordre d'entrée formatés : {params}", level=3)
        return params

    def _build_oco_params_formatted(self,
                                    entry_side: str, # "BUY" ou "SELL" (pour l'ordre d'entrée qui a créé la position)
                                    quantity_str: str, # Quantité pour les ordres SL/TP
                                    sl_price_str: str, # Prix de déclenchement du StopLoss
                                    tp_price_str: str, # Prix limite du TakeProfit
                                    stop_limit_price_str: Optional[str] = None, # Prix limite pour STOP_LOSS_LIMIT
                                    stop_limit_time_in_force: Optional[str] = "GTC",
                                    list_client_order_id: Optional[str] = None
                                   ) -> Dict[str, Any]:
        """Construit le dictionnaire de paramètres pour un ordre OCO (pour log OOS ou exécution réelle)."""
        # ... (logique existante, s'assurer qu'elle est propre et utilise self.symbol) ...
        exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"
        
        if list_client_order_id is None:
            strat_short = self.strategy_name[:min(len(self.strategy_name), 6)].lower().replace("_", "")
            sym_short = self.symbol[:min(len(self.symbol),4)].lower().replace("/", "")
            ts_short = str(int(pd.Timestamp.now(tz='UTC').timestamp() * 1000))[-6:]
            uuid_short = str(uuid.uuid4().hex)[:4]
            list_client_order_id = f"oco_{strat_short}_{sym_short}_{ts_short}_{uuid_short}"
            list_client_order_id = list_client_order_id[:32] # Limite de certains exchanges pour listClientOrderId

        oco_params: Dict[str, Any] = {
            "symbol": self.symbol, # Utiliser self.symbol
            "side": exit_side, # L'OCO est pour sortir de la position
            "quantity": quantity_str, # La même quantité que l'entrée
            "price": tp_price_str,  # Prix de l'ordre LIMIT (Take Profit)
            "stopPrice": sl_price_str, # Prix de déclenchement du STOP_LOSS / STOP_LOSS_LIMIT
            # "stopLimitPrice": stop_limit_price_str if stop_limit_price_str else sl_price_str, # Prix limite du STOP_LOSS_LIMIT
            "listClientOrderId": list_client_order_id
        }
        
        # stopLimitPrice n'est pertinent que si le type d'ordre stop est STOP_LOSS_LIMIT
        # Pour un OCO simple avec STOP_LOSS (qui devient MARKET) et LIMIT (TP),
        # stopLimitPrice n'est pas toujours requis par l'API.
        # L'API Binance pour OCO sur marge (POST /sapi/v1/margin/order/oco)
        # a `price` (pour le LIMIT TP) et `stopPrice` (pour le STOP_LOSS).
        # Si `stopLimitPrice` est fourni, il implique un STOP_LOSS_LIMIT.
        if stop_limit_price_str:
            oco_params["stopLimitPrice"] = stop_limit_price_str
            if stop_limit_time_in_force: # Time in force pour l'ordre stop limit
                oco_params["stopLimitTimeInForce"] = stop_limit_time_in_force.upper()
        
        self._log(f"Paramètres d'ordre OCO formatés : {oco_params}", level=3)
        return oco_params

    @abstractmethod
    def generate_order_request(
        self,
        data: pd.DataFrame, # Données OHLCV + indicateurs ( potentiellement sur une fréquence différente de 1min)
        current_position: int, # 0 pas de pos, 1 long, -1 short (état actuel du compte réel/simulé)
        available_capital: float, # Capital dispo en quote asset pour ce trade
        symbol_info: Dict[str, Any] # Infos de l'exchange pour la paire (pair_config)
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre d'ENTRÉE pour le trading en direct ou une simulation détaillée.
        Cette méthode est appelée par le LiveTradingManager.

        Args:
            data (pd.DataFrame): Les données de marché les plus récentes, incluant
                                 les indicateurs nécessaires calculés par la stratégie
                                 sur la (ou les) fréquence(s) de temps appropriée(s).
            current_position (int): État actuel de la position sur le compte
                                    (0: pas de position, 1: long, -1: short).
            available_capital (float): Capital disponible (en actif de cotation)
                                       pour cette opération de trading.
            symbol_info (Dict[str, Any]): Informations de l'exchange pour la paire
                                          (filtres, précisions). C'est le `pair_config`.

        Returns:
            Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
                - Si un ordre d'ENTRÉE doit être placé:
                    - Un tuple contenant:
                        - entry_order_params (Dict[str, Any]): Dictionnaire des paramètres
                          pour l'ordre d'entrée principal (ex: symbol, side, type, quantity, price).
                          Prêt à être envoyé à l'API de l'exchange.
                        - sl_tp_raw_prices (Dict[str, float]): Dictionnaire optionnel avec les prix
                          Stop-Loss et Take-Profit bruts suggérés (non ajustés à la précision de l'exchange)
                          par la stratégie, ex: `{"sl_price": 123.45, "tp_price": 128.90}`.
                          Ces prix seront ajustés par le LiveTradingManager avant de placer l'OCO.
                - Si aucun ordre d'entrée n'est généré, retourne None.
        """
        pass

    def _log(self, message: str, level: int = 1, is_error: bool = False, is_warning:bool = False) -> None:
        """Gère le logging basé sur un niveau de verbosité conceptuel."""
        # Cette méthode est un helper interne, la verbosité du backtest est gérée par BacktestRunner.
        # Pour une stratégie, on peut utiliser les niveaux de logging standards.
        if is_error:
            logger.error(f"{self.log_prefix} {message}")
        elif is_warning:
            logger.warning(f"{self.log_prefix} {message}")
        elif level <= 1: # Messages importants
            logger.info(f"{self.log_prefix} {message}")
        elif level == 2: # Messages de debug plus détaillés
            logger.debug(f"{self.log_prefix} {message}")
        else: # level >= 3, messages très détaillés
            # Pourrait utiliser un niveau de trace si configuré, sinon debug
            logger.debug(f"{self.log_prefix} {message}") # Ou logger.trace si un niveau TRACE est défini

