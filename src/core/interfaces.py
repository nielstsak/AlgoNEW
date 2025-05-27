# src/core/interfaces.py
"""
Ce module définit les interfaces (Protocoles Python) pour les composants majeurs
du système de trading algorithmique. Ces protocoles servent de contrats clairs
entre les différents modules, facilitant le duck typing, la validation statique
avec mypy, et le développement de nouvelles implémentations conformes.
"""
import logging # Ajout de l'import logging
from typing import (
    Protocol,
    runtime_checkable,
    Optional,
    Dict,
    Any,
    List,
    Tuple,
    Generic,
    Callable,
    TypeVar, # Pour les types génériques
    Union # Ajouté pour Union[str, Any] dans IStrategy
)
from pathlib import Path
import pandas as pd
import numpy as np
# Importer optuna.Trial pour IErrorHandler si besoin, ou le garder générique
# import optuna

# Type générique pour les valeurs de retour ou d'entrée
T = TypeVar('T')
V = TypeVar('V') # Pour la valeur dans CacheEntry (utilisé par ICacheManager)
R = TypeVar('R') # Pour le résultat dans TaskResult (utilisé par IParallelExecutor)


@runtime_checkable
class IStrategy(Protocol):
    """
    Interface (Protocole) pour les stratégies de trading.
    Définit les méthodes essentielles qu'une stratégie doit implémenter
    pour interagir avec le moteur de backtesting et d'autres composants.
    """
    # Attributs attendus (pour information et introspection par la factory)
    REQUIRED_PARAMS: List[str]
    VERSION: str

    def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]):
        ...

    def validate_params(self) -> None:
        """
        Valide les paramètres spécifiques fournis à l'instance de la stratégie.
        Doit lever une ValueError si les paramètres sont invalides.
        """
        ...

    def get_required_indicator_configs(self) -> List[Dict[str, Any]]:
        """
        Retourne une liste de configurations pour les indicateurs techniques
        requis par la stratégie.
        """
        ...

    def _calculate_indicators(self, data_feed: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule ou assure la présence des indicateurs techniques nécessaires
        directement sur le DataFrame fourni. Appelé après IndicatorCalculator.
        """
        ...

    def _generate_signals(
        self,
        data_with_indicators: pd.DataFrame,
        current_position_open: bool,
        current_position_direction: int,
        current_entry_price: float
    ) -> Tuple[int, Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Génère les signaux de trading (achat, vente, sortie, hold).
        """
        ...

    def get_signal(
        self,
        data_feed: pd.DataFrame,
        current_position_open: bool,
        current_position_direction: int,
        current_entry_price: float,
        current_equity: float
    ) -> Dict[str, Any]:
        """
        Méthode principale appelée par le simulateur pour obtenir la décision de trading.
        """
        ...

    def set_trading_context(self, context: Any) -> Any: # Type de context et ValidationResult serait mieux
        """Configure le contexte de trading (backtest ou live)."""
        ...

    def validate_context(self, context: Any) -> Any: # Type de context et ValidationResult serait mieux
        """Valide le contexte de trading fourni."""
        ...

    def get_meta_parameters(self) -> Dict[str, Any]:
        """Retourne des méta-paramètres descriptifs de la stratégie."""
        ...

    def subscribe_to_events(self, event_dispatcher: 'IEventDispatcher', event_types: List[Union[str, Any]]) -> None:
        """Permet à la stratégie de s'abonner à des types d'événements spécifiques."""
        ...

    def on_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Méthode de callback générique pour les événements souscrits."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Sérialise l'état interne pertinent de la stratégie."""
        ...

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restaure l'état interne de la stratégie à partir d'un dictionnaire."""
        ...

    def generate_order_request(
        self,
        data: pd.DataFrame,
        current_position: int,
        available_capital: float,
        symbol_info: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """Génère une requête d'ordre d'ENTRÉE pour le trading en direct."""
        ...


@runtime_checkable
class IIndicatorCalculator(Protocol):
    """
    Interface (Protocole) pour les calculateurs d'indicateurs techniques.
    Définit comment les indicateurs sont calculés, mis en cache et invalidés.
    Aligné avec la Feuille de Route section 2.1.1.
    """

    def calculate(
        self,
        data_feed: pd.DataFrame, # DataFrame source (OHLCV et autres données de base)
        indicator_configs: List[Dict[str, Any]], # Configs des indicateurs à calculer
        # Les strategy_params pourraient être passés ici si certains indicateurs
        # en dépendent directement et ne sont pas définis dans indicator_configs.
        # Ou, le CacheManager utilisé par l'implémentation pourrait prendre en compte
        # un hash des params pertinents pour sa clé de cache.
        # Pour l'instant, on garde la signature simple.
    ) -> pd.DataFrame:
        """
        Calcule les indicateurs spécifiés et les ajoute/retourne dans un DataFrame.

        Args:
            data_feed (pd.DataFrame): DataFrame source.
            indicator_configs (List[Dict[str, Any]]): Liste des configurations d'indicateurs.

        Returns:
            pd.DataFrame: DataFrame enrichi avec les indicateurs calculés.
                          Peut être une copie du data_feed avec des colonnes ajoutées,
                          ou un nouveau DataFrame contenant seulement les indicateurs.
                          L'implémentation `calculate_indicators_for_trial` actuelle
                          retourne un DF qui inclut les colonnes OHLCV de base + les indicateurs.
        """
        ...

    def get_cached(self, cache_key: str) -> Optional[pd.DataFrame]: # Le cache contiendra souvent des DataFrames
        """
        Récupère un résultat d'indicateur (DataFrame d'indicateurs) depuis le cache.
        """
        ...

    def invalidate_cache(self, pattern: str, recursive: bool = False) -> int:
        """
        Invalide (supprime) des entrées du cache basées sur un pattern de clé.
        """
        ...


@runtime_checkable
class IBacktestEngine(Protocol):
    """
    Interface (Protocole) pour les moteurs de backtesting.
    """
    # La signature de run_simulation est alignée avec l'implémentation de BacktestRunner
    def run_simulation(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[Any, float], List[Dict[str, Any]]]:
        """Exécute une simulation de backtest complète."""
        ...

    def get_results(self) -> Dict[str, Any]:
        """Retourne les résultats complets de la dernière simulation."""
        ...

    # La signature de get_metrics est alignée avec PerformanceAnalyzer
    def get_metrics(self,
                    trades_df: pd.DataFrame,
                    equity_curve_series: pd.Series,
                    initial_capital: float,
                    risk_free_rate_daily: float = 0.0,
                    periods_per_year: int = 252,
                    market_data_for_regimes: Optional[pd.DataFrame] = None,
                    cache_manager: Optional['ICacheManager'] = None,
                    base_cache_key_prefix: Optional[str] = None
                   ) -> Dict[str, Any]:
        """Calcule et retourne les métriques de performance."""
        ...


@runtime_checkable
class IDataValidator(Protocol):
    """
    Interface (Protocole) pour les validateurs de données.
    """
    def validate_ohlcv_data(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        expected_frequency: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Any]: # Any pour ValidationReport
        """Valide la structure, les types, et la cohérence des données OHLCV."""
        ...

    def ensure_datetime_index(
        self,
        df: pd.DataFrame,
        target_tz_str: str = "UTC"
    ) -> pd.DataFrame:
        """S'assure que le DataFrame a un DatetimeIndex UTC, trié, et unique."""
        ...

    def validate_indicators(
        self,
        df: pd.DataFrame, # DataFrame contenant déjà les indicateurs calculés
        indicator_configs: List[Dict[str, Any]] # Configurations des indicateurs attendus
    ) -> Any: # Any pour ValidationReport
        """Valide la présence et la validité de base des colonnes d'indicateurs."""
        ...

    def detect_data_anomalies(
        self,
        df: pd.DataFrame,
        anomaly_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, Any]: # Any pour AnomalyReport
        """Détecte les anomalies communes dans les données."""
        ...


@runtime_checkable
class ICacheManager(Protocol, Generic[V]): # V est le type de la valeur cachée
    """
    Interface (Protocole) pour un gestionnaire de cache.
    """
    def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], V], # La fonction retourne un type V
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> V: # Retourne un type V
        """Récupère ou calcule, met en cache, et retourne une valeur."""
        ...

    def invalidate(self, pattern: str, recursive: bool = False) -> int:
        """Invalide (supprime) des entrées du cache basées sur un pattern de clé."""
        ...

    def get_statistics(self) -> Any: # Any pour CacheStatistics (dataclass)
        """Retourne des statistiques sur l'utilisation et la performance du cache."""
        ...

    def persist_to_disk(self, path: Optional[Path] = None, compress: bool = True) -> None:
        """Sauvegarde l'état actuel du cache sur disque."""
        ...

    def load_from_disk(self, path: Optional[Path] = None) -> bool:
        """Charge le cache depuis le disque."""
        ...

    def clear(self) -> None:
        """Vide complètement le cache."""
        ...


# --- Interfaces pour les dépendances injectées (nécessaires pour les modules refactorisés) ---
@runtime_checkable
class IStrategyLoader(Protocol):
    """Interface pour un chargeur de stratégie."""
    def load_strategy(self,
                      strategy_name_key: str, # Nom de la config de stratégie (ex: "ma_crossover_strategy")
                      params_for_strategy: Dict[str, Any], # Paramètres du trial Optuna
                      strategy_script_ref: str, # Ex: "src/strategies/ma_crossover_strategy.py"
                      strategy_class_name: str, # Ex: "MaCrossoverStrategy"
                      pair_symbol: str # La paire pour laquelle cette instance est créée
                     ) -> IStrategy: # Doit retourner une instance de IStrategy
        ...

@runtime_checkable
class IErrorHandler(Protocol):
    """Interface pour un gestionnaire d'erreurs."""
    def handle_evaluation_error(self,
                                exception: Exception,
                                context: Dict[str, Any],
                                trial: Optional[Any] = None # optuna.trial.Trial si optuna est une dépendance ici
                               ) -> Any: # Any pour ErrorResult (dataclass)
        """Gère une erreur survenue durant l'évaluation d'un trial."""
        ...

@runtime_checkable
class IEventDispatcher(Protocol):
    """Interface pour un distributeur d'événements."""
    def register_listener(self, event_type: str, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Enregistre un listener pour un type d'événement spécifique."""
        ...

    def dispatch(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Distribue un événement à tous les listeners enregistrés pour ce type."""
        ...

@runtime_checkable
class IParallelExecutor(Protocol, Generic[T, R]): # T: Task type, R: Result type
    """Interface pour un exécuteur parallèle."""
    def execute_parallel(
        self,
        tasks: List[Any], # Devrait être List[Task[T, R]] si Task est défini ici ou importé
        executor_type: Optional[Any] = None, # Devrait être Optional[ExecutorType]
        max_workers: Optional[int] = None,
        description: Optional[str] = None,
        show_progress: bool = False
    ) -> List[Any]: # Devrait être List[TaskResult[R]]
        """Exécute une liste de tâches en parallèle."""
        ...
    # Ajouter d'autres méthodes si définies dans l'implémentation de ParallelExecutor
    # def execute_with_progress(...): ...
    # def monitor_resource_usage(...): ...
