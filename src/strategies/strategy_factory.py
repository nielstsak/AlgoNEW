# src/strategies/strategy_factory.py
"""
Factory pattern pour la création dynamique de stratégies de trading,
avec auto-discovery, enregistrement, validation de configuration, et gestion de version.
"""
import importlib
import inspect
import logging
import pkgutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast, Tuple
from abc import ABCMeta # Pour vérifier si c'est une classe abstraite (comme IStrategy)

# Tentative d'importation de jsonschema pour la validation avancée (optionnel)
try:
    import jsonschema
    from jsonschema import validate as jsonschema_validate
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    jsonschema = None # type: ignore
    jsonschema_validate = None # type: ignore
    JSONSCHEMA_AVAILABLE = False

# Imports depuis les modules du projet
try:
    from src.core.interfaces import IStrategy
    from src.strategies.base import BaseStrategy, TradingContext # TradingContext pour type hint
    # Importer IAppConfig et IDependencyContainer si la factory doit interagir avec eux.
    # Pour l'instant, on garde la factory relativement autonome.
    # from src.config.loader import AppConfig
    # from src.utils.dependency_injector import DependencyContainer
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger(__name__).critical(
        f"StrategyFactory: Échec de l'importation des dépendances (IStrategy, BaseStrategy): {e}. "
        "Vérifiez PYTHONPATH et l'existence des modules."
    )
    # Définir des placeholders pour permettre au module de se charger
    class IStrategy: pass # type: ignore
    class BaseStrategy(IStrategy): # type: ignore
        VERSION: str = "unknown"
        REQUIRED_PARAMS: List[str] = []
        def __init__(self, strategy_name: str, symbol: str, params: Dict[str, Any]): pass
        def set_trading_context(self, context: Any) -> Any: pass # Placeholder
    class TradingContext: pass # type: ignore
    JSONSCHEMA_AVAILABLE = False # Assumer non disponible si les imports de base échouent
    # Renvoyer l'erreur pour indiquer un problème de configuration du projet
    raise

logger = logging.getLogger(__name__)

@dataclass
class RegisteredStrategyInfo:
    """Informations sur une stratégie enregistrée."""
    name: str # Nom unique (souvent nom de la classe ou clé de config)
    strategy_class: Type[IStrategy]
    module_path: str
    version: str = "0.0.0"
    description: Optional[str] = None # Extrait du docstring de la classe
    required_params_list: List[str] = field(default_factory=list)
    # Optionnel: schéma JSON pour les paramètres de cette stratégie
    params_json_schema: Optional[Dict[str, Any]] = None
    # Optionnel: chemin vers le fichier de configuration par défaut de la stratégie
    default_config_path: Optional[Path] = None


class StrategyFactory:
    """
    Factory singleton pour la découverte, l'enregistrement et la création
    d'instances de stratégies de trading.
    """
    _instance: Optional['StrategyFactory'] = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs) -> 'StrategyFactory':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False # Pour s'assurer que __init__ n'est appelé qu'une fois
        return cls._instance

    def __init__(self,
                 strategies_package_path: str = "src.strategies",
                 project_root: Optional[Union[str, Path]] = None,
                 auto_discover: bool = True,
                 dev_mode_hot_reload: bool = False): # Hot reload non implémenté dans cette version
        """
        Initialise la StrategyFactory.

        Args:
            strategies_package_path (str): Chemin Python du package où découvrir
                                           automatiquement les stratégies (ex: "src.strategies").
            project_root (Optional[Union[str, Path]]): Chemin racine du projet,
                                                       utilisé pour résoudre les chemins relatifs.
                                                       Si None, tente de le déduire.
            auto_discover (bool): Si True, lance l'auto-découverte à l'initialisation.
            dev_mode_hot_reload (bool): Activer le rechargement à chaud (non implémenté).
        """
        if hasattr(self, '_initialized') and self._initialized:
            return # Singleton déjà initialisé

        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return

            self.log_prefix = "[StrategyFactory]"
            logger.info(f"{self.log_prefix} Initialisation du singleton StrategyFactory...")

            self._registry: Dict[str, RegisteredStrategyInfo] = {}
            self._instance_cache: Dict[Any, IStrategy] = {} # Pour le mode test
            self._test_mode_caching_active: bool = False

            if project_root is None:
                try:
                    self._project_root = Path(__file__).resolve().parent.parent.parent # src/strategies -> src -> project_root
                except NameError:
                    self._project_root = Path(".").resolve()
                    logger.warning(f"{self.log_prefix} __file__ non défini, project_root déduit comme CWD: {self._project_root}")
            else:
                self._project_root = Path(project_root).resolve()

            self.strategies_package_path_str = strategies_package_path
            self.dev_mode_hot_reload = dev_mode_hot_reload # Non utilisé pour l'instant

            if auto_discover:
                self.auto_discover_strategies(package_python_path=self.strategies_package_path_str)

            self._initialized = True
            logger.info(f"{self.log_prefix} StrategyFactory initialisée. {len(self._registry)} stratégie(s) initialement enregistrée(s).")

    def _get_module_path_from_class(self, strategy_class: Type[IStrategy]) -> str:
        """Tente de déduire le chemin Python du module à partir de la classe."""
        module_name = strategy_class.__module__
        # Si le module est dans un sous-package de self._project_root/src, on peut le rendre relatif à src
        # Exemple: si module_name est "project_root_name.src.strategies.my_strategy"
        # et strategies_package_path_str est "src.strategies", on veut "src.strategies.my_strategy"
        if module_name.startswith(self.strategies_package_path_str):
            return module_name
        # Fallback simple si la logique ci-dessus ne suffit pas
        return module_name


    def register_strategy(self,
                          strategy_class: Type[IStrategy],
                          name_override: Optional[str] = None,
                          version_override: Optional[str] = None,
                          params_schema: Optional[Dict[str, Any]] = None
                         ) -> bool:
        """
        Enregistre manuellement une classe de stratégie.

        Args:
            strategy_class (Type[IStrategy]): La classe de stratégie à enregistrer.
            name_override (Optional[str]): Nom à utiliser pour enregistrer la stratégie.
                                           Si None, utilise strategy_class.__name__.
            version_override (Optional[str]): Version à utiliser. Si None, tente de lire
                                              l'attribut de classe `VERSION`.
            params_schema (Optional[Dict[str, Any]]): Schéma JSON optionnel pour valider
                                                      les paramètres de cette stratégie.

        Returns:
            bool: True si l'enregistrement a réussi, False sinon.
        """
        with self._lock:
            if not inspect.isclass(strategy_class) or not issubclass(strategy_class, IStrategy): # type: ignore
                logger.error(f"{self.log_prefix} Tentative d'enregistrement de '{strategy_class}' qui n'est pas une sous-classe de IStrategy.")
                return False
            # Éviter d'enregistrer la classe de base elle-même ou les interfaces/protocoles
            if strategy_class is BaseStrategy or strategy_class is IStrategy or inspect.isabstract(strategy_class):
                logger.debug(f"{self.log_prefix} Tentative d'enregistrement d'une classe de base/abstraite '{strategy_class.__name__}'. Ignorée.")
                return False

            name = name_override if name_override else strategy_class.__name__
            if name in self._registry:
                # Gérer le rechargement à chaud: si la classe est différente, mettre à jour.
                # Pour l'instant, on logue un avertissement si le nom est déjà pris par une classe différente.
                if self._registry[name].strategy_class is not strategy_class:
                    logger.warning(f"{self.log_prefix} Stratégie '{name}' déjà enregistrée avec une classe différente. "
                                   "Le nouvel enregistrement va l'écraser (utile pour hot-reload, sinon c'est une collision de noms).")
                # else: # Même nom, même classe, pas besoin de ré-enregistrer
                #     logger.debug(f"{self.log_prefix} Stratégie '{name}' déjà enregistrée avec la même classe. Pas de mise à jour.")
                #     return True


            version = version_override if version_override else getattr(strategy_class, 'VERSION', '0.0.1-dev')
            description = inspect.getdoc(strategy_class)
            module_path = self._get_module_path_from_class(strategy_class)
            required_params = getattr(strategy_class, 'REQUIRED_PARAMS', [])

            reg_info = RegisteredStrategyInfo(
                name=name,
                strategy_class=strategy_class,
                module_path=module_path,
                version=version,
                description=description.strip() if description else None,
                required_params_list=required_params,
                params_json_schema=params_schema
            )
            self._registry[name] = reg_info
            logger.info(f"{self.log_prefix} Stratégie '{name}' (v{version}) enregistrée depuis module '{module_path}'.")
            return True

    def auto_discover_strategies(self, package_python_path: str = "src.strategies") -> int:
        """
        Découvre et enregistre automatiquement les stratégies dans le package spécifié.
        Recharge les modules si dev_mode_hot_reload est True (conceptuel).

        Args:
            package_python_path (str): Chemin Python du package (ex: "src.strategies").

        Returns:
            int: Nombre de stratégies nouvellement découvertes et enregistrées.
        """
        with self._lock: # Protéger l'accès au registre
            log_prefix_discover = f"{self.log_prefix}[AutoDiscover({package_python_path})]"
            logger.info(f"{log_prefix_discover} Démarrage de l'auto-découverte des stratégies...")
            
            discovered_count = 0
            try:
                package = importlib.import_module(package_python_path)
                package_file_path = getattr(package, '__file__', None)
                if not package_file_path:
                    logger.error(f"{log_prefix_discover} Impossible de déterminer le chemin du package '{package_python_path}'.")
                    return 0
                
                package_dir = Path(package_file_path).parent

                for (_, module_name, is_pkg) in pkgutil.iter_modules([str(package_dir)]):
                    if is_pkg: # Ne pas descendre dans les sous-packages pour l'instant
                        continue
                    
                    full_module_name = f"{package_python_path}.{module_name}"
                    try:
                        module = importlib.import_module(full_module_name)
                        # Si hot-reload était activé, on pourrait faire importlib.reload(module) ici
                        # mais cela a des effets de bord complexes.

                        for name_in_module, obj_in_module in inspect.getmembers(module, inspect.isclass):
                            # Vérifier si c'est une sous-classe de IStrategy (ou BaseStrategy)
                            # et qu'elle n'est pas IStrategy/BaseStrategy elle-même, ni abstraite.
                            if issubclass(obj_in_module, IStrategy) and \
                               obj_in_module is not IStrategy and \
                               obj_in_module is not BaseStrategy and \
                               not inspect.isabstract(obj_in_module):
                                
                                # Utiliser le nom de la classe comme nom par défaut pour l'enregistrement
                                if self.register_strategy(obj_in_module): # register_strategy gère les doublons
                                    discovered_count +=1
                                    
                    except ImportError as e_mod_import:
                        logger.error(f"{log_prefix_discover} Échec de l'import du module de stratégie '{full_module_name}': {e_mod_import}")
                    except Exception as e_mod_inspect:
                        logger.error(f"{log_prefix_discover} Erreur lors de l'inspection du module '{full_module_name}': {e_mod_inspect}", exc_info=False) # exc_info=False pour ne pas polluer avec des erreurs mineures d'import

            except ImportError as e_pkg_import:
                logger.error(f"{log_prefix_discover} Package de stratégies '{package_python_path}' non trouvé ou erreur d'import: {e_pkg_import}")
                return 0
            except Exception as e_discover_general:
                logger.error(f"{log_prefix_discover} Erreur inattendue durant l'auto-découverte: {e_discover_general}", exc_info=True)
                return 0

            logger.info(f"{log_prefix_discover} Auto-découverte terminée. {discovered_count} nouvelle(s) stratégie(s) enregistrée(s). "
                        f"Total stratégies enregistrées: {len(self._registry)}.")
            return discovered_count

    def validate_strategy_config(self, strategy_name: str, config_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valide les paramètres de configuration pour une stratégie nommée.
        Utilise le schéma JSON si disponible, sinon des règles de base.
        """
        log_prefix_validate = f"{self.log_prefix}[ValidateConfig({strategy_name})]"
        if strategy_name not in self._registry:
            logger.error(f"{log_prefix_validate} Stratégie '{strategy_name}' non enregistrée.")
            return False, [f"Stratégie '{strategy_name}' non enregistrée."]

        reg_info = self._registry[strategy_name]
        errors: List[str] = []

        # 1. Validation par schéma JSON (si disponible et jsonschema installé)
        if reg_info.params_json_schema and JSONSCHEMA_AVAILABLE and jsonschema_validate:
            try:
                jsonschema_validate(instance=config_params, schema=reg_info.params_json_schema)
                logger.debug(f"{log_prefix_validate} Validation JSONSchema réussie.")
            except jsonschema.exceptions.ValidationError as e_schema:
                errors.append(f"Échec validation JSONSchema: {e_schema.message} (Path: {list(e_schema.path)})")
            except Exception as e_schema_other:
                errors.append(f"Erreur inattendue durant validation JSONSchema: {e_schema_other}")
        elif reg_info.params_json_schema and not JSONSCHEMA_AVAILABLE:
            logger.warning(f"{log_prefix_validate} Schéma JSON défini pour '{strategy_name}' mais la bibliothèque 'jsonschema' n'est pas installée. Validation par schéma sautée.")

        # 2. Validation des règles métier de base (ex: présence des REQUIRED_PARAMS)
        #    Ceci est redondant si la stratégie elle-même valide ses params dans son __init__
        #    via sa propre méthode validate_params(). Mais peut servir de première passe.
        if reg_info.required_params_list:
            for req_param in reg_info.required_params_list:
                if req_param not in config_params:
                    errors.append(f"Paramètre requis '{req_param}' manquant dans la configuration fournie.")
        
        # On pourrait ajouter ici des appels à une méthode statique `validate_business_rules(params)`
        # sur la strategy_class si une telle convention était établie.
        # Exemple: if hasattr(reg_info.strategy_class, 'validate_specific_params_static'):
        #             errors.extend(reg_info.strategy_class.validate_specific_params_static(config_params))

        if errors:
            logger.warning(f"{log_prefix_validate} Validation de configuration échouée pour '{strategy_name}': {errors}")
            return False, errors
        
        logger.info(f"{log_prefix_validate} Validation de configuration réussie pour '{strategy_name}'.")
        return True, []


    def create_strategy(self,
                        name: str, # Nom de la stratégie enregistrée
                        symbol: str,
                        params: Dict[str, Any],
                        trading_context: Optional[TradingContext] = None,
                        dependency_container: Optional[Any] = None # Placeholder pour DI avancée
                       ) -> IStrategy:
        """
        Crée une instance d'une stratégie enregistrée.

        Args:
            name (str): Nom de la stratégie (doit être enregistrée).
            symbol (str): Symbole de la paire pour cette instance.
            params (Dict[str, Any]): Paramètres d'hyperparamètres pour cette instance.
            trading_context (Optional[TradingContext]): Contexte de trading à appliquer.
            dependency_container (Optional[Any]): Conteneur DI à utiliser pour
                                                  injecter des dépendances dans la stratégie
                                                  (fonctionnalité avancée, non pleinement utilisée ici).

        Returns:
            IStrategy: Une instance configurée de la stratégie.

        Raises:
            ValueError: Si la stratégie n'est pas enregistrée ou si la configuration est invalide.
            TypeError: Si l'instanciation échoue.
        """
        with self._lock: # Protéger la création si le cache est actif
            log_prefix_create = f"{self.log_prefix}[CreateStrategy({name})]"
            logger.info(f"{log_prefix_create} Demande de création pour {symbol} avec params: {str(params)[:100]}...")

            if name not in self._registry:
                msg = f"Stratégie '{name}' non enregistrée dans la factory."
                logger.error(f"{log_prefix_create} {msg}")
                raise ValueError(msg)

            reg_info = self._registry[name]
            
            # Valider la configuration des paramètres fournis
            is_config_valid, config_errors = self.validate_strategy_config(name, params)
            if not is_config_valid:
                error_summary = f"Configuration des paramètres invalide pour la stratégie '{name}': {'; '.join(config_errors)}"
                logger.error(f"{log_prefix_create} {error_summary}")
                raise ValueError(error_summary)

            # Gestion du cache d'instance pour le mode test
            if self._test_mode_caching_active:
                # Créer une clé de cache basée sur le nom, symbole, et params (stables)
                # frozenset pour rendre le dict de params hashable
                cache_key_instance = (name, symbol.upper(), frozenset(sorted(params.items())))
                if cache_key_instance in self._instance_cache:
                    logger.info(f"{log_prefix_create} Instance de stratégie (mode test) récupérée du cache pour clé: {name}/{symbol}.")
                    # Retourner une copie si on veut éviter les effets de bord entre tests,
                    # ou l'instance elle-même si c'est acceptable.
                    # Pour un vrai test d'isolation, une nouvelle instance est mieux,
                    # donc le cache ici est plus pour la performance si les instances sont coûteuses à créer.
                    # Ici, on retourne l'instance cachée.
                    return self._instance_cache[cache_key_instance]

            try:
                # Instanciation de la stratégie
                # Le nom passé à la stratégie est le nom d'enregistrement (la clé)
                strategy_instance = reg_info.strategy_class(
                    strategy_name=name, # Utiliser le nom d'enregistrement
                    symbol=symbol.upper(),
                    params=params.copy() # Passer une copie des params
                )
                logger.info(f"{log_prefix_create} Instance de '{reg_info.strategy_class.__name__}' (v{reg_info.version}) créée avec succès pour {symbol}.")

                # Appliquer le contexte de trading si fourni
                if trading_context:
                    if hasattr(strategy_instance, 'set_trading_context') and callable(getattr(strategy_instance, 'set_trading_context')):
                        validation_res_ctx = strategy_instance.set_trading_context(trading_context) # type: ignore
                        if not validation_res_ctx.is_valid:
                            logger.warning(f"{log_prefix_create} Le contexte de trading fourni pour '{name}' a des problèmes de validation: {validation_res_ctx.messages}")
                            # Décider si c'est une erreur bloquante ou juste un avertissement.
                            # Pour l'instant, on continue mais on logue.
                    else:
                        logger.warning(f"{log_prefix_create} Stratégie '{name}' n'a pas de méthode 'set_trading_context'. Contexte non appliqué par la factory.")

                # Mettre en cache si en mode test
                if self._test_mode_caching_active and 'cache_key_instance' in locals():
                    self._instance_cache[cache_key_instance] = strategy_instance # type: ignore

                return strategy_instance

            except ValueError as e_val_init: # Erreurs de validation dans __init__ de la stratégie
                logger.error(f"{log_prefix_create} Erreur de valeur (validation des params) lors de l'instanciation de '{name}': {e_val_init}", exc_info=True)
                raise
            except TypeError as e_type_init: # Mauvais arguments pour __init__
                logger.error(f"{log_prefix_create} Erreur de type (arguments __init__) lors de l'instanciation de '{name}': {e_type_init}", exc_info=True)
                raise
            except Exception as e_inst_general:
                logger.error(f"{log_prefix_create} Erreur inattendue lors de l'instanciation de '{name}': {e_inst_general}", exc_info=True)
                raise

    def list_available_strategies(self, filters: Optional[Dict[str, Any]] = None) -> List[RegisteredStrategyInfo]:
        """
        Liste les stratégies enregistrées, avec option de filtrage.
        """
        with self._lock: # Accès en lecture au registre
            if not filters:
                return list(self._registry.values())
            
            filtered_list: List[RegisteredStrategyInfo] = []
            for reg_info in self._registry.values():
                match = True
                for key, value in filters.items():
                    if key == "name" and reg_info.name != value: match = False; break
                    if key == "version" and reg_info.version != value: match = False; break
                    # Ajouter d'autres filtres si nécessaire (ex: par tag, par module_path)
                if match:
                    filtered_list.append(reg_info)
            return filtered_list

    def get_strategy_info(self, name: str) -> Optional[RegisteredStrategyInfo]:
        """Récupère les informations d'une stratégie enregistrée par son nom."""
        with self._lock:
            return self._registry.get(name)

    def clear_registry(self) -> None:
        """Vide le registre des stratégies (principalement pour les tests)."""
        with self._lock:
            self._registry.clear()
            self._instance_cache.clear()
            logger.info(f"{self.log_prefix} Registre des stratégies et cache d'instances vidés.")

    def set_test_mode_caching(self, active: bool) -> None:
        """Active ou désactive la mise en cache des instances pour le mode test."""
        with self._lock:
            self._test_mode_caching_active = active
            if not active:
                self._instance_cache.clear()
            logger.info(f"{self.log_prefix} Mise en cache des instances en mode test {'activée' if active else 'désactivée'}.")

# Instance globale du singleton (peut être importée et utilisée directement)
# GlobalStrategyFactory = StrategyFactory()
