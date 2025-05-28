# src/strategies/base.py
"""
Ce module définit la classe de base abstraite pour toutes les stratégies de trading.
Elle implémente l'interface IStrategy, fournit une structure standardisée,
facilite la gestion des paramètres et du contexte, et intègre les
fonctionnalités de gestion d'état et d'événements, tout en supportant
les intégrations avec des outils comme vectorbtpro.
"""
import logging
import uuid
import copy
import enum
import weakref
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List, Union, TypeVar, Generic, Callable

import pandas as pd
import numpy as np
import vectorbtpro as vbt

from src.config.definitions import StrategyConfig # Maintenu pour la compatibilité avec le nouveau framework
from src.core.interfaces import IStrategy # IStrategy est maintenant un Protocol

# Tentative d'import des utilitaires exchange et IEventDispatcher comme dans l'ancienne version
try:
    from src.core.interfaces import IEventDispatcher # Assumant que IEventDispatcher est défini ici
    from src.utils.exchange_utils import (
        adjust_precision,
        adjust_quantity_to_step_size,
        get_precision_from_filter,
        get_filter_value
    )
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger(__name__).critical(
        f"BaseStrategy: Échec de l'importation des dépendances (IEventDispatcher, exchange_utils): {e}. "
        "Vérifiez PYTHONPATH et l'existence des modules."
    )
    # Définir des placeholders pour permettre au module de se charger en cas d'erreur d'import
    class IEventDispatcher: # type: ignore
        def register_listener(self, event_type: str, listener: Callable[[Dict[str, Any]], None]) -> None: ...
        def dispatch(self, event_type: str, event_data: Dict[str, Any]) -> None: ...

    def get_precision_from_filter(pair_config: Dict, filter_type: str, key: str) -> Optional[int]: return 8
    def adjust_precision(value: Union[float,str,Any], precision: Optional[int], tick_size: Optional[Union[float,str,Any]] = None, rounding_mode_str: str = "ROUND_HALF_UP") -> Optional[float]:
        if value is None: return None
        try: return round(float(str(value)), precision or 8)
        except: return None # pylint: disable=bare-except
    def adjust_quantity_to_step_size(quantity: Union[float,str,Any], symbol_info: Dict, qty_precision: Optional[int] = None, rounding_mode_str: str = "ROUND_FLOOR") -> Optional[float]:
        if quantity is None: return None
        try: return round(float(str(quantity)), qty_precision or 8) # Simplifié, la vraie fonction est plus complexe
        except: return None # pylint: disable=bare-except
    def get_filter_value(symbol_info: Dict, filter_type: str, key: str) -> Optional[float]: return 0.0
    # Ne pas renvoyer l'erreur ici pour permettre au reste du module de se charger pour inspection
    # raise

from src.utils.validation_utils import (
    validate_dataframe_columns,
    validate_non_empty_dataframe,
    validate_series_not_empty_or_all_na, # Utilisation d'un nom de fonction plausible
)

logger = logging.getLogger(__name__)

# --- Dataclasses pour le contexte et la validation (repris de l'ancienne version) ---
@dataclass
class TradingContext:
    """
    Représente le contexte d'exécution d'une stratégie (pour backtest ou live).
    """
    pair_config: Dict[str, Any] # Informations de l'exchange pour la paire
    is_futures: bool
    leverage: int
    initial_equity: float
    account_type: Optional[str] = "SPOT"
    # Ajout pour compatibilité avec set_backtest_context si nécessaire
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
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
        self.messages.append(f"WARNING: {message}")

# --- Types d'événements (repris de l'ancienne version) ---
class StrategyEventType(enum.Enum):
    """Types d'événements que la stratégie peut émettre ou écouter."""
    PARAMETER_CHANGED = "PARAMETER_CHANGED"
    MARKET_REGIME_SHIFT = "MARKET_REGIME_SHIFT"
    EXTERNAL_SIGNAL_RECEIVED = "EXTERNAL_SIGNAL_RECEIVED"

class BaseStrategy(IStrategy, ABC):
    """
    Classe de base abstraite pour toutes les stratégies de trading.
    Fusionne les concepts de l'ancienne et de la nouvelle version.
    """
    REQUIRED_PARAMS: List[str] = [] # Doit être défini par les sous-classes (ancienne version)
    REQUIRED_INPUT_COLUMNS = ["Open", "High", "Low", "Close", "Volume"] # Nouvelle version
    EXPECTED_SIGNAL_COLUMNS = ["entries", "exits", "short_entries", "short_exits"] # Nouvelle version

    def __init__(self, strategy_config: StrategyConfig, params: Optional[Dict[str, Any]] = None):
        """
        Initialise la stratégie.
        Utilise StrategyConfig pour la configuration principale, mais permet la surcharge des paramètres.
        Initialise également les attributs de l'ancienne version.
        """
        self.strategy_config = strategy_config
        self.strategy_id = strategy_config.strategy_id # Équivalent à strategy_name ou identifiant unique
        self.name = strategy_config.name # Nom lisible de la stratégie

        # Logique de fusion des paramètres de la nouvelle version
        current_params = strategy_config.parameters.copy()
        if params:
            current_params.update(params)
        
        # Valider et définir les paramètres via la méthode (potentiellement surchargée)
        # _validate_and_set_params est de la nouvelle version, validate_params de l'ancienne.
        # Nous allons utiliser _validate_and_set_params comme méthode principale d'initialisation des params.
        self.params = self._validate_and_set_params(current_params)

        # Attributs de la nouvelle version (pour vectorbtpro)
        self.data: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        self.indicator_data: Dict[str, Union[pd.Series, pd.DataFrame]] = {}

        # Attributs pour le contexte du backtest (symbol, timeframe) - Nouvelle version
        # Ces attributs sont ceux que set_backtest_context (Ticket 2) va setter.
        self._current_symbol: Optional[str] = strategy_config.default_symbol
        self._current_timeframe: Optional[str] = strategy_config.default_timeframe

        # Attributs de l'ancienne version
        self.symbol: str = strategy_config.default_symbol # Dérivé de strategy_config
        self.log_prefix: str = f"[{self.strategy_id}][{self.symbol}]"

        self.trading_context: Optional[TradingContext] = None
        self.price_precision: Optional[int] = None
        self.quantity_precision: Optional[int] = None
        self.base_asset: str = ""
        self.quote_asset: str = ""
        self._internal_state: Dict[str, Any] = {}

        self._event_dispatcher_ref: Optional[weakref.ReferenceType[IEventDispatcher]] = None # type: ignore
        self._subscribed_event_types: List[str] = []
        self._event_callbacks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

        try:
            self.validate_params() # Appel à la méthode de validation de l'ancienne version (peut être surchargée)
            self._log(f"Stratégie initialisée. Params: {self.params}", level=logging.INFO)
        except ValueError as e_val:
            self._log(f"Erreur de validation des paramètres lors de l'initialisation : {e_val}", level=logging.ERROR, is_error=True)
            raise

    # --- Méthodes de la nouvelle version (adaptées ou conservées) ---
    @abstractmethod
    def _validate_and_set_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide et définit les paramètres spécifiques à la stratégie. (Nouvelle version)
        Cette méthode est appelée par __init__.
        """
        # Les sous-classes doivent implémenter cela pour valider et stocker leurs paramètres spécifiques.
        # Exemple: self.short_window = params.get("short_window")
        #          if not isinstance(self.short_window, int) or self.short_window <=0:
        #              raise ValueError("short_window doit être un entier positif")
        return params


    def set_data(self, data: pd.DataFrame) -> None:
        """Définit les données de marché pour la stratégie (Nouvelle version)."""
        validate_non_empty_dataframe(data, "Les données d'entrée (data)")
        validate_dataframe_columns(data, self.REQUIRED_INPUT_COLUMNS, "Les données d'entrée (data)")
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError("L'index des données d'entrée doit être DatetimeIndex.") from e
        self.data = data.copy()
        self._log(f"Données définies. Shape: {self.data.shape}", level=logging.DEBUG)
        self.signals = None
        self.indicator_data = {}

    def set_backtest_context(self, symbol: str, timeframe: str) -> None:
        """
        Définit le contexte actuel du backtest (symbole et timeframe). (Nouvelle version - Ticket 2)
        Cette méthode est appelée par l'optimiseur.
        Elle met à jour _current_symbol, _current_timeframe et self.symbol pour la cohérence.
        Elle peut aussi initialiser un TradingContext simplifié.
        """
        self._current_symbol = symbol
        self._current_timeframe = timeframe
        self.symbol = symbol # Mettre à jour le self.symbol de l'ancienne version aussi
        self.log_prefix = f"[{self.strategy_id}][{self.symbol}]" # Mettre à jour le log_prefix
        self._log(f"Contexte de backtest défini: Symbole='{symbol}', Timeframe='{timeframe}'", level=logging.DEBUG)

        # Optionnel: créer un TradingContext partiel si d'autres méthodes en dépendent.
        # Cela nécessiterait d'avoir accès à plus d'informations (pair_config, equity, etc.)
        # Pour l'instant, on se contente de setter symbol et timeframe.
        # Si un TradingContext complet est nécessaire, la méthode set_trading_context devrait être appelée.
        if self.trading_context: # Si un contexte existait, le mettre à jour ou le réinitialiser
            if self.trading_context.pair_config and self.trading_context.pair_config.get('symbol') != symbol:
                 self._log(f"Le symbole du contexte de backtest ({symbol}) diffère du symbole dans pair_config existant. Réinitialisation partielle du contexte.", level=logging.WARNING)
                 # Idéalement, il faudrait un pair_config pour le nouveau symbole.
                 # Pour l'instant, on met juste à jour le symbole dans le contexte existant s'il y en a un.
                 self.trading_context.symbol = symbol
                 self.trading_context.timeframe = timeframe
            elif not self.trading_context.pair_config:
                 self.trading_context.symbol = symbol
                 self.trading_context.timeframe = timeframe


    @abstractmethod
    def _calculate_indicators(self) -> None:
        """Calcule les indicateurs. (Nouvelle version - pour vbt)"""
        if self.data is None:
            raise ValueError("Données non définies pour _calculate_indicators.")
        pass

    @abstractmethod
    def _generate_signals(self) -> pd.DataFrame:
        """Génère les signaux. (Nouvelle version - pour vbt)"""
        if self.data is None or not self.indicator_data:
            raise ValueError("Données ou indicateurs non prêts pour _generate_signals.")
        pass

    def run(self) -> pd.DataFrame:
        """Exécute la stratégie et retourne les signaux (Nouvelle version - pour vbt)."""
        if self.data is None:
            raise ValueError("Données non définies pour run.")
        self._log("Exécution de la stratégie (mode vbt)...", level=logging.DEBUG)
        self._calculate_indicators()
        self.signals = self._generate_signals()
        self._validate_signals(self.signals) # Méthode de la nouvelle version
        self._log("Stratégie (mode vbt) exécutée.", level=logging.INFO)
        return self.signals

    def _validate_signals(self, signals_df: pd.DataFrame) -> None:
        """Valide le DataFrame des signaux (Nouvelle version)."""
        validate_non_empty_dataframe(signals_df, "Le DataFrame des signaux")
        validate_dataframe_columns(signals_df, self.EXPECTED_SIGNAL_COLUMNS, "Le DataFrame des signaux")
        for col in self.EXPECTED_SIGNAL_COLUMNS:
            if not pd.api.types.is_bool_dtype(signals_df[col]):
                try:
                    signals_df[col] = signals_df[col].fillna(False).astype(bool)
                except Exception as e:
                    raise ValueError(f"Colonne de signaux '{col}' doit être booléenne. Erreur conversion: {e}") from e
        if self.data is not None and not signals_df.index.equals(self.data.index):
            raise ValueError("L'index des signaux doit correspondre à l'index des données.")
        self._log("Validation des signaux (mode vbt) terminée.", level=logging.DEBUG)

    def get_portfolio_from_signals(
        self, close_prices: pd.Series, entries: pd.Series, exits: pd.Series,
        short_entries: Optional[pd.Series] = None, short_exits: Optional[pd.Series] = None,
        init_cash: float = 100000.0, fees: float = 0.001, slippage: float = 0.0005,
        freq: str = "D", log_level_vbt: int = logging.WARNING, **kwargs
    ) -> Optional[vbt.Portfolio]:
        """Génère un portefeuille VectorBT (Nouvelle version)."""
        validate_series_not_empty_or_all_na(close_prices, "close_prices")
        validate_series_not_empty_or_all_na(entries, "entries") # Utilisation de la fonction corrigée
        validate_series_not_empty_or_all_na(exits, "exits")

        # ... (le reste de la logique de get_portfolio_from_signals est conservée) ...
        # S'assure que les séries de short sont fournies si l'une l'est, ou crée des séries vides
        if short_entries is None: # Création par défaut si non fourni
            short_entries = pd.Series(False, index=close_prices.index, name="short_entries")
        if short_exits is None: # Création par défaut si non fourni
            short_exits = pd.Series(False, index=close_prices.index, name="short_exits")

        validate_series_not_empty_or_all_na(short_entries, "short_entries")
        validate_series_not_empty_or_all_na(short_exits, "short_exits")
        
        # Vérification de la cohérence des index
        if not (close_prices.index.equals(entries.index) and
                entries.index.equals(exits.index) and
                exits.index.equals(short_entries.index) and
                short_entries.index.equals(short_exits.index)):
            self._log("Les index de close_prices et de toutes les séries de signaux doivent correspondre.", level=logging.ERROR, is_error=True)
            raise ValueError("Les index des séries de prix et de signaux doivent être identiques.")

        vbt_logger = logging.getLogger("vectorbt")
        original_vbt_level = vbt_logger.level
        vbt_logger.setLevel(log_level_vbt)
        try:
            portfolio = vbt.Portfolio.from_signals(
                close=close_prices, entries=entries.astype(bool), exits=exits.astype(bool),
                short_entries=short_entries.astype(bool), short_exits=short_exits.astype(bool),
                init_cash=init_cash, fees=fees, slippage=slippage, freq=freq, log=True, **kwargs
            )
            self._log(f"Portefeuille VectorBT généré avec {len(portfolio.trades)} trades.", level=logging.INFO)
            return portfolio
        except Exception as e:
            self._log(f"Erreur lors de la génération du portefeuille VectorBT: {e}", level=logging.ERROR, is_error=True, exc_info=True)
            return None
        finally:
            vbt_logger.setLevel(original_vbt_level)

    def update_params(self, params: Dict[str, Any]) -> None:
        """Met à jour les paramètres de la stratégie (Nouvelle version)."""
        current_params = self.params.copy()
        current_params.update(params)
        self.params = self._validate_and_set_params(current_params) # Utilise la méthode de la nouvelle version
        self._log(f"Paramètres mis à jour : {self.params}", level=logging.INFO)
        self.signals = None
        self.indicator_data = {}
        try:
            self.validate_params() # Appelle aussi la validation de l'ancienne version
        except ValueError as e_val:
            self._log(f"Erreur de validation des params (ancienne méthode) après update: {e_val}", level=logging.ERROR, is_error=True)
            raise


    # --- Méthodes de l'ancienne version (réintégrées et adaptées) ---
    def get_param(self, param_name: str, default: Optional[Any] = None) -> Any:
        """Récupère une valeur de paramètre pour la stratégie."""
        return self.params.get(param_name, default)

    def set_trading_context(self, context: TradingContext) -> ValidationResult:
        """Configure le contexte de trading (ancienne version)."""
        self._log(f"Configuration du contexte de trading: {context}", level=logging.DEBUG)
        validation_res = self.validate_context(context)
        if not validation_res.is_valid:
            self._log(f"Échec de la validation du contexte: {validation_res.messages}", level=logging.ERROR, is_error=True)
            self.trading_context = None
            return validation_res

        self.trading_context = context
        self.symbol = context.symbol or self.symbol # Mettre à jour le symbole principal de la strat
        self._current_symbol = self.symbol # Synchroniser avec l'attribut de la nouvelle version
        if context.timeframe:
             self._current_timeframe = context.timeframe

        if self.trading_context.pair_config:
            self.base_asset = self.trading_context.pair_config.get('baseAsset', '')
            self.quote_asset = self.trading_context.pair_config.get('quoteAsset', '')
            # S'assurer que exchange_utils est importé correctement pour ces fonctions
            try:
                self.price_precision = get_precision_from_filter(self.trading_context.pair_config, 'PRICE_FILTER', 'tickSize')
                self.quantity_precision = get_precision_from_filter(self.trading_context.pair_config, 'LOT_SIZE', 'stepSize')
            except NameError: # exchange_utils n'a pas pu être importé
                self._log("exchange_utils non disponible, impossible de déterminer les précisions.", level=logging.WARNING, is_warning=True)
                self.price_precision = self.price_precision or 8 # Fallback
                self.quantity_precision = self.quantity_precision or 8 # Fallback

            self._log(f"Contexte de trading défini. PricePrec: {self.price_precision}, QtyPrec: {self.quantity_precision}", level=logging.DEBUG)
        else:
            validation_res.add_error("pair_config manquant dans TradingContext.")
            self._log("pair_config manquant dans TradingContext.", level=logging.ERROR, is_error=True)
        return validation_res

    def validate_context(self, context: TradingContext) -> ValidationResult:
        """Valide le contexte de trading fourni (ancienne version)."""
        report = ValidationResult()
        if not isinstance(context, TradingContext):
            report.add_error("Le contexte n'est pas une instance de TradingContext.")
            return report
        if not context.pair_config or not isinstance(context.pair_config, dict):
            report.add_error("'pair_config' est manquant ou invalide.")
        # ... (autres validations de l'ancienne version) ...
        if context.symbol and context.pair_config and context.pair_config.get('symbol', '').upper() != context.symbol.upper():
             report.add_warning(f"Symbole du contexte ({context.symbol}) != symbole pair_config ({context.pair_config.get('symbol')})")
        return report

    def get_meta_parameters(self) -> Dict[str, Any]:
        """Retourne des méta-paramètres descriptifs (ancienne version)."""
        return {
            "risk_level_profile": "medium",
            "typical_time_horizon": "short_term",
            "market_regime_assumption": "any",
            "strategy_class_name": self.__class__.__name__
        }

    def subscribe_to_events(self, event_dispatcher: 'IEventDispatcher', event_types: List[Union[str, StrategyEventType]]) -> None: # type: ignore
        """S'abonne à des événements (ancienne version)."""
        self._log(f"Demande de souscription aux événements: {event_types}", level=logging.DEBUG)
        if not hasattr(event_dispatcher, 'register_listener') or not callable(event_dispatcher.register_listener):
            self._log("EventDispatcher invalide.", level=logging.ERROR, is_error=True)
            return
        # ... (logique de souscription de l'ancienne version) ...
        self._event_dispatcher_ref = weakref.ref(event_dispatcher) # type: ignore
        # ...

    def on_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Callback générique pour les événements (ancienne version)."""
        self._log(f"Événement reçu: Type='{event_type}', Données='{str(event_data)[:100]}...'", level=logging.DEBUG)
        # ... (logique de traitement de l'ancienne version) ...

    def get_state(self) -> Dict[str, Any]:
        """Sérialise l'état interne (ancienne version)."""
        state = {
            "strategy_name": self.strategy_id, # Utilise strategy_id
            "symbol": self.symbol,
            "params": copy.deepcopy(self.params),
            "internal_state_variables": copy.deepcopy(self._internal_state),
            "_current_symbol": self._current_symbol,
            "_current_timeframe": self._current_timeframe,
        }
        # Ne pas inclure self.data ou self.signals ici car ils sont volumineux et transitoires
        self._log("Récupération de l'état de la stratégie.", level=logging.DEBUG)
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restaure l'état interne (ancienne version)."""
        self._log(f"Configuration de l'état à partir de: {list(state.keys())}", level=logging.DEBUG)
        if state.get("strategy_name") != self.strategy_id or state.get("symbol") != self.symbol:
            # Permettre une certaine flexibilité si le symbol a changé via set_backtest_context
            if state.get("strategy_name") != self.strategy_id:
                 msg = "Tentative de restauration d'un état pour une stratégie différente."
                 self._log(msg, level=logging.ERROR, is_error=True)
                 raise ValueError(msg)
            self._log(f"Restauration d'état pour {self.strategy_id} mais le symbole stocké ({state.get('symbol')}) diffère du symbole actuel ({self.symbol}). Cela peut être dû à set_backtest_context.", level=logging.WARNING, is_warning=True)


        self.params = copy.deepcopy(state.get("params", self.params))
        self._internal_state = copy.deepcopy(state.get("internal_state_variables", {}))
        self._current_symbol = state.get("_current_symbol", self._current_symbol)
        self._current_timeframe = state.get("_current_timeframe", self._current_timeframe)
        
        # Mettre à jour le symbole principal si présent dans l'état et différent
        # Cela peut être important si l'état est restauré avant un set_backtest_context
        loaded_symbol = state.get("symbol")
        if loaded_symbol and self.symbol != loaded_symbol:
            self._log(f"Mise à jour du symbole principal de la stratégie de '{self.symbol}' à '{loaded_symbol}' à partir de l'état.", level=logging.INFO)
            self.symbol = loaded_symbol
            # Mettre à jour le log_prefix si le symbole change
            self.log_prefix = f"[{self.strategy_id}][{self.symbol}]"


        try:
            self.validate_params() # Valider les params de l'ancienne méthode
            self.params = self._validate_and_set_params(self.params) # Revalider avec la nouvelle méthode
        except ValueError as e_reval:
            self._log(f"Erreur de validation des params après restauration: {e_reval}", level=logging.ERROR, is_error=True)
            raise
        self._log("État de la stratégie restauré.", level=logging.INFO)

    @abstractmethod
    def validate_params(self) -> None:
        """Valide les paramètres self.params (ancienne version)."""
        # Les sous-classes doivent vérifier si self.params contient REQUIRED_PARAMS
        # et si les types/valeurs sont corrects.
        # Exemple:
        # for req_param in self.REQUIRED_PARAMS:
        # if req_param not in self.params:
        # raise ValueError(f"Paramètre requis '{req_param}' manquant.")
        pass

    @abstractmethod
    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        """Déclare les indicateurs requis (ancienne version)."""
        # Cette méthode était pour un système où les indicateurs étaient calculés en externe
        # et déclarés ici. Dans le flux vbt, _calculate_indicators les gère.
        # Peut retourner une liste vide si non utilisé dans le flux actuel.
        return []

    # _calculate_indicators et _generate_signals sont déjà définis pour le flux vbt.
    # Les signatures de l'ancienne version étaient différentes:
    # @abstractmethod
    # def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame: ...
    # @abstractmethod
    # def _generate_signals(self, data_with_indicators: pd.DataFrame, ...) -> Tuple[int, ...]: ...
    # Pour éviter les conflits, nous gardons les signatures pour vbt.
    # Si l'ancien flux doit coexister, des noms de méthodes différents seraient nécessaires.

    @abstractmethod
    def get_signal(
        self, data_feed: pd.DataFrame, current_position_open: bool,
        current_position_direction: int, current_entry_price: float, current_equity: float
    ) -> Dict[str, Any]:
        """
        Méthode principale pour obtenir les décisions de trading dans un simulateur pas à pas (ancienne version).
        Les stratégies concrètes doivent implémenter cela si elles sont utilisées avec ce type de simulateur.
        """
        # Doit retourner un dictionnaire comme dans l'ancienne version.
        # Exemple: {"signal": 0, "order_type": "MARKET", ...}
        self._log("get_signal (ancienne méthode) appelée. Les sous-classes doivent implémenter.", level=logging.WARNING, is_warning=True)
        return self._get_default_signal_response("Non implémenté dans la classe de base (version fusionnée).")


    def _get_default_signal_response(self, reason: str) -> Dict[str, Any]:
        """Retourne une réponse de signal HOLD par défaut (ancienne version)."""
        return {
            "signal": 0, "order_type": "MARKET", "limit_price": None,
            "sl_price": None, "tp_price": None, "position_size_pct": 1.0,
            "info": reason
        }

    def _calculate_quantity(self, entry_price: float, available_capital: float,
                            qty_precision: Optional[int], symbol_info: Dict[str, Any],
                            symbol_log_name: str, position_size_pct: Optional[float] = None
                           ) -> Optional[float]:
        """Calcule la quantité (ancienne version)."""
        self._log(f"Appel de _calculate_quantity pour {symbol_log_name}", level=logging.DEBUG)
        if not self.trading_context:
            self._log("TradingContext non défini, impossible de calculer la quantité précisément.", level=logging.WARNING, is_warning=True)
            # Fallback simple si pas de contexte pour levier etc.
            if entry_price <= 1e-9: return 0.0
            raw_qty = (available_capital * (position_size_pct or 1.0)) / entry_price
            return round(raw_qty, qty_precision or 8)

        # Utilisation de la logique de l'ancienne version avec self.trading_context pour le levier
        leverage_to_use = self.trading_context.leverage
        capital_for_this_trade = available_capital * (position_size_pct if position_size_pct is not None else self.get_param('capital_allocation_pct', 1.0))
        total_position_value_quote = capital_for_this_trade * leverage_to_use
        
        if entry_price <= 1e-9: return 0.0
        quantity_base_raw = total_position_value_quote / entry_price
        
        try:
            adjusted_quantity = adjust_quantity_to_step_size(
                quantity_base_raw, symbol_info, qty_precision
            )
            # ... (ajouter la vérification minNotional etc. de l'ancienne version) ...
            min_notional = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
            if adjusted_quantity is not None and min_notional is not None and (adjusted_quantity * entry_price < min_notional):
                self._log(f"Notionnel {adjusted_quantity * entry_price} < minNotional {min_notional}. Qty=0.", level=logging.WARNING, is_warning=True)
                return 0.0
            return adjusted_quantity
        except NameError:
            self._log("exchange_utils non disponibles pour adjust_quantity_to_step_size.", level=logging.WARNING, is_warning=True)
            return round(quantity_base_raw, qty_precision or 8)


    # _build_entry_params_formatted et _build_oco_params_formatted sont conservées de l'ancienne version
    # Elles pourraient être utiles si la stratégie doit générer des paramètres d'ordre détaillés.
    def _build_entry_params_formatted(self, side: str, quantity_str: str, order_type: str,
                                     entry_price_str: Optional[str] = None,
                                     time_in_force: Optional[str] = None,
                                     new_client_order_id: Optional[str] = None) -> Dict[str, Any]:
        if new_client_order_id is None:
            new_client_order_id = f"sim_{self.strategy_id[:10]}_{str(uuid.uuid4())[:8]}"
        
        params: Dict[str, Any] = {
            "symbol": self.symbol, "side": side.upper(), "type": order_type.upper(),
            "quantity": quantity_str, "newClientOrderId": new_client_order_id
        }
        if order_type.upper() == "LIMIT":
            if entry_price_str is None: raise ValueError("Prix requis pour ordre LIMIT.")
            params["price"] = entry_price_str
        if time_in_force: params["timeInForce"] = time_in_force
        return params

    def _build_oco_params_formatted(self, entry_side: str, quantity_str: str, sl_price_str: str,
                                   tp_price_str: str, stop_limit_price_str: Optional[str] = None,
                                   stop_limit_time_in_force: Optional[str] = "GTC",
                                   list_client_order_id: Optional[str] = None) -> Dict[str, Any]:
        exit_side = "SELL" if entry_side.upper() == "BUY" else "BUY"
        if list_client_order_id is None:
            list_client_order_id = f"oco_{self.strategy_id[:10]}_{str(uuid.uuid4())[:8]}"

        oco_params: Dict[str, Any] = {
            "symbol": self.symbol, "side": exit_side, "quantity": quantity_str,
            "price": tp_price_str, "stopPrice": sl_price_str,
            "listClientOrderId": list_client_order_id
        }
        if stop_limit_price_str:
            oco_params["stopLimitPrice"] = stop_limit_price_str
            if stop_limit_time_in_force:
                oco_params["stopLimitTimeInForce"] = stop_limit_time_in_force.upper()
        return oco_params

    @abstractmethod
    def generate_order_request(
        self, data: pd.DataFrame, current_position: int,
        available_capital: float, symbol_info: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre d'ENTRÉE pour trading live ou simulation détaillée (ancienne version).
        Les stratégies concrètes doivent implémenter cela si elles sont utilisées avec ce type de moteur.
        """
        self._log("generate_order_request (ancienne méthode) appelée. Les sous-classes doivent implémenter.", level=logging.WARNING, is_warning=True)
        return None


    def _log(self, message: str, level: int = logging.DEBUG, is_error: bool = False, is_warning:bool = False, exc_info=False) -> None:
        """Gère le logging pour la stratégie."""
        # Utilise le logger standard avec le préfixe de la stratégie.
        # Le niveau de verbosité est géré par la configuration globale du logger.
        full_message = f"{self.log_prefix} {message}"
        if is_error:
            logger.error(full_message, exc_info=exc_info)
        elif is_warning:
            logger.warning(full_message, exc_info=exc_info)
        elif level == logging.INFO: # Mappage simple des niveaux
             logger.info(full_message)
        elif level == logging.DEBUG:
             logger.debug(full_message)
        elif level >= logging.ERROR: # Si un niveau numérique est passé
             logger.log(level, full_message, exc_info=exc_info)
        else: # Par défaut, debug pour les autres entiers ou niveaux custom bas
             logger.debug(full_message)


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(strategy_id='{self.strategy_id}', "
            f"name='{self.name}', symbol='{self.symbol}', params={self.params})"
        )

