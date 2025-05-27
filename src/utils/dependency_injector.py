# src/utils/dependency_injector.py
"""
Container d'injection de dépendances (IoC) léger pour gérer les dépendances
entre les composants de l'application et faciliter les tests.
"""
import inspect
import logging
import threading
import weakref # Pour les proxies de lazy loading si implémenté
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import (Any, Callable, Dict, List, Optional, Type, TypeVar, Union,
                    Generic, cast)
from pathlib import Path

# Tentative d'importation de PyYAML pour la configuration YAML, optionnel
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None # type: ignore
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Définition des scopes supportés
SCOPE_SINGLETON = "singleton"
SCOPE_PROTOTYPE = "prototype"
SCOPE_THREAD = "thread"
# SCOPE_REQUEST = "request" # Plus complexe, omis pour l'instant

VALID_SCOPES = [SCOPE_SINGLETON, SCOPE_PROTOTYPE, SCOPE_THREAD]

class CircularDependencyError(Exception):
    """Exception levée lors de la détection d'une dépendance circulaire."""
    pass

class DependencyResolutionError(Exception):
    """Exception levée lorsqu'une dépendance ne peut pas être résolue."""
    pass

@dataclass
class Registration:
    """Stocke les informations d'enregistrement pour une dépendance."""
    implementation: Union[Type[Any], Callable[..., Any]]
    scope: str = SCOPE_SINGLETON
    # Pour les singletons et thread-scoped instances
    cached_instances: Dict[Any, Any] = field(default_factory=lambda: weakref.WeakValueDictionary() if False else {}) # weakref pour lazy singletons plus tard
    # Pour les thread-scoped instances, la clé du dict ci-dessus serait l'id du thread
    # Ou utiliser threading.local pour les thread-scoped instances directement.

    # Factory pour lazy singletons (non utilisé dans la première passe)
    # lazy_factory: Optional[Callable[[], Any]] = None


class _LazyProxy(Generic[T]):
    """
    Un proxy simple pour le chargement paresseux des singletons.
    Note: Cette implémentation est basique. Une bibliothèque de proxy plus robuste
    serait meilleure pour une couverture complète des cas d'usage.
    """
    _instance: Optional[T] = None
    _factory: Optional[Callable[[], T]] = None
    _lock: Optional[threading.RLock] = None

    def __init__(self, factory: Callable[[], T], lock: threading.RLock):
        # Ne pas appeler __setattr__ directement ici pour éviter la récursion
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_lock", lock)

    def _ensure_instance(self) -> T:
        if self._instance is None:
            if self._lock is None or self._factory is None: # Should not happen if initialized correctly
                 raise RuntimeError("LazyProxy non initialisé correctement (lock ou factory manquant).")
            with self._lock:
                if self._instance is None: # Double-check locking
                    logger.debug(f"LazyProxy: Instanciation paresseuse de {self._factory}")
                    object.__setattr__(self, "_instance", self._factory())
        if self._instance is None: # Should be set by factory
            raise RuntimeError(f"LazyProxy: La factory {self._factory} n'a pas retourné d'instance.")
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensure_instance(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._ensure_instance(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._ensure_instance(), name)

    # Ajouter d'autres méthodes magiques si nécessaire (ex: __str__, __call__, etc.)
    def __repr__(self) -> str:
        if self._instance is not None:
            return repr(self._instance)
        return f"<LazyProxy factory={self._factory}>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        instance = self._ensure_instance()
        if callable(instance):
            return instance(*args, **kwargs)
        raise TypeError(f"Instance de LazyProxy (type: {type(instance)}) non appelable.")


class DependencyContainer:
    """
    Container d'injection de dépendances (IoC) singleton.
    """
    _instance: Optional['DependencyContainer'] = None
    _global_lock = threading.RLock() # Lock pour la création du singleton

    def __new__(cls, *args, **kwargs) -> 'DependencyContainer':
        if cls._instance is None:
            with cls._global_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Initialiser les attributs d'instance ici pour éviter la ré-initialisation
                    cls._instance._initialized_for_singleton = False
        return cls._instance

    def __init__(self, parent: Optional['DependencyContainer'] = None):
        """
        Initialise le conteneur. Si parent est fourni, ce conteneur devient un enfant.
        """
        with self._global_lock: # S'assurer que l'init est aussi thread-safe pour le singleton
            if hasattr(self, '_initialized_for_singleton') and self._initialized_for_singleton and parent is None:
                # Si c'est le singleton global et qu'il est déjà initialisé, ne rien faire.
                # Ceci est pour le cas où __init__ est appelé plusieurs fois sur le singleton.
                return

            self.log_prefix = f"[DepContainer(id:{hex(id(self))})]"
            logger.info(f"{self.log_prefix} Initialisation du conteneur de dépendances. Parent: {parent is not None}")
            
            self._registrations: Dict[Type[Any], Registration] = {}
            self._resolution_lock = threading.RLock() # Lock pour les opérations de résolution/enregistrement
            self._currently_resolving: threading.local = threading.local() # Pour la détection de dépendances circulaires par thread
            
            self._parent: Optional['DependencyContainer'] = parent
            self._child_containers: List[weakref.ReferenceType['DependencyContainer']] = []

            # Pour les scopes 'thread'
            self._thread_scoped_storage = threading.local()

            # Pour la configuration via fichier
            self._config_loaded_path: Optional[Path] = None
            
            if parent is None: # Seulement pour le conteneur racine
                self._initialized_for_singleton = True

    def register(self,
                 interface: Type[T],
                 implementation: Optional[Union[Type[T], Callable[..., T]]] = None,
                 scope: str = SCOPE_SINGLETON,
                 instance: Optional[T] = None) -> None:
        """
        Enregistre une dépendance dans le conteneur.

        Args:
            interface (Type[T]): Le type (interface, classe abstraite, ou classe concrète)
                                 à enregistrer.
            implementation (Optional[Union[Type[T], Callable[..., T]]]):
                                 La classe concrète ou la fonction factory qui fournit
                                 l'instance. Si None, `interface` est supposée être
                                 une classe concrète à instancier directement.
            scope (str): Le scope de la dépendance ('singleton', 'prototype', 'thread').
                         Par défaut 'singleton'.
            instance (Optional[T]): Une instance pré-créée à utiliser (sera toujours singleton).
                                    Si fourni, `implementation` et `scope` sont ignorés.
        """
        with self._resolution_lock:
            log_prefix_reg = f"{self.log_prefix}[Register({interface.__name__ if hasattr(interface, '__name__') else str(interface)})]"
            logger.debug(f"{log_prefix_reg} Enregistrement avec impl: {implementation}, scope: {scope}, instance: {instance is not None}")

            if not isinstance(interface, type) and not callable(interface): # ABCMeta est une 'type'
                 raise TypeError(f"{log_prefix_reg} L'interface doit être un type ou un protocole appelable.")
            if scope not in VALID_SCOPES:
                raise ValueError(f"{log_prefix_reg} Scope '{scope}' invalide. Valides: {VALID_SCOPES}")

            if instance is not None:
                # Si une instance est fournie, elle est enregistrée comme singleton
                reg = Registration(implementation=type(instance), scope=SCOPE_SINGLETON)
                reg.cached_instances['singleton_instance'] = instance # Clé spéciale pour instance pré-fournie
                self._registrations[interface] = reg
                logger.info(f"{log_prefix_reg} Instance pré-fournie enregistrée comme singleton.")
                return

            impl_to_register = implementation if implementation is not None else interface
            if not (isinstance(impl_to_register, type) or callable(impl_to_register)):
                raise TypeError(f"{log_prefix_reg} L'implémentation doit être un type ou une factory appelable.")

            self._registrations[interface] = Registration(implementation=impl_to_register, scope=scope)
            logger.info(f"{log_prefix_reg} Enregistrement terminé. Impl: {impl_to_register.__name__ if hasattr(impl_to_register, '__name__') else str(impl_to_register)}, Scope: {scope}.")

    def _resolve_dependencies_for_init(self,
                                       concrete_class_or_factory: Union[Type[T], Callable[..., T]],
                                       current_resolution_path: List[Type[Any]]
                                      ) -> Dict[str, Any]:
        """Inspecte __init__ ou la signature de la factory et résout ses dépendances."""
        deps_to_inject: Dict[str, Any] = {}
        try:
            sig = inspect.signature(concrete_class_or_factory)
        except ValueError: # Peut arriver pour certains built-ins ou objets C
            logger.debug(f"{self.log_prefix} Impossible d'inspecter la signature de {concrete_class_or_factory}. "
                         "Supposition d'aucun argument pour l'auto-wiring.")
            return deps_to_inject
            
        for param_name, param_obj in sig.parameters.items():
            if param_name == 'self' or param_name == 'cls':
                continue
            
            param_type_annotation = param_obj.annotation
            if param_type_annotation is inspect.Parameter.empty:
                if param_obj.default is inspect.Parameter.empty:
                    # Pas d'annotation de type et pas de valeur par défaut -> Ne peut pas auto-wire
                    raise DependencyResolutionError(
                        f"Impossible de résoudre le paramètre '{param_name}' pour "
                        f"'{concrete_class_or_factory.__name__ if hasattr(concrete_class_or_factory, '__name__') else str(concrete_class_or_factory)}': "
                        "pas d'annotation de type ni de valeur par défaut."
                    )
                # S'il y a une valeur par défaut mais pas de type, on ne tente pas de résoudre.
                continue

            # Tenter de résoudre basé sur l'annotation de type
            try:
                # Utiliser _resolve_internal pour gérer le chemin de résolution et les parents
                deps_to_inject[param_name] = self._resolve_internal(param_type_annotation, current_resolution_path)
            except DependencyResolutionError as e_dep_res:
                # Si la dépendance n'est pas enregistrée et qu'il n'y a pas de valeur par défaut
                if param_obj.default is inspect.Parameter.empty:
                    logger.error(f"{self.log_prefix} Échec de résolution de la dépendance '{param_name}: {param_type_annotation}' "
                                 f"pour '{concrete_class_or_factory.__name__ if hasattr(concrete_class_or_factory, '__name__') else str(concrete_class_or_factory)}'. Cause: {e_dep_res}")
                    raise
                # Si une valeur par défaut existe, on ne lève pas d'erreur ici, Python l'utilisera.
                # Le paramètre ne sera juste pas injecté par le DI container.
                logger.debug(f"{self.log_prefix} Dépendance '{param_name}: {param_type_annotation}' non résolue, "
                             "mais une valeur par défaut existe. Python l'utilisera.")
            except CircularDependencyError: # Renvoyer directement
                raise

        return deps_to_inject

    def _create_instance(self,
                         registration: Registration,
                         current_resolution_path: List[Type[Any]]
                        ) -> Any:
        """Crée une instance, résolvant les dépendances pour __init__."""
        impl = registration.implementation
        log_prefix_create = f"{self.log_prefix}[CreateInstance({impl.__name__ if hasattr(impl, '__name__') else str(impl)})]"
        logger.debug(f"{log_prefix_create} Création d'instance...")

        # Auto-wiring: inspecter __init__ (ou la factory elle-même si c'est une fonction)
        # et résoudre les dépendances basées sur les annotations de type.
        init_args = self._resolve_dependencies_for_init(impl, current_resolution_path)
        
        logger.debug(f"{log_prefix_create} Appel de {impl.__name__ if hasattr(impl, '__name__') else str(impl)} avec les dépendances injectées: {list(init_args.keys())}")
        try:
            return impl(**init_args)
        except Exception as e_create:
            logger.error(f"{log_prefix_create} Erreur lors de l'instanciation de {impl} avec args {init_args}: {e_create}", exc_info=True)
            raise DependencyResolutionError(f"Échec de l'instanciation de {impl}: {e_create}") from e_create


    def _resolve_internal(self, interface: Type[T], current_resolution_path: List[Type[Any]]) -> T:
        """Méthode de résolution interne pour gérer la récursion et les conteneurs parents."""
        log_prefix_resolve = f"{self.log_prefix}[ResolveInternal({interface.__name__ if hasattr(interface, '__name__') else str(interface)})]"

        if interface in current_resolution_path:
            path_str = ' -> '.join([getattr(p, '__name__', str(p)) for p in current_resolution_path]) + \
                       f' -> {getattr(interface, "__name__", str(interface))}'
            logger.error(f"{log_prefix_resolve} Dépendance circulaire détectée : {path_str}")
            raise CircularDependencyError(f"Dépendance circulaire détectée : {path_str}")

        new_resolution_path = current_resolution_path + [interface]

        registration = self._registrations.get(interface)
        if not registration:
            if self._parent: # Tenter de résoudre depuis le parent
                logger.debug(f"{log_prefix_resolve} Non trouvé localement, tentative de résolution depuis le parent.")
                try:
                    return self._parent._resolve_internal(interface, new_resolution_path)
                except DependencyResolutionError as e_parent_res: # Si le parent ne peut pas résoudre non plus
                    logger.warning(f"{log_prefix_resolve} Non résolu par le parent: {e_parent_res}")
                    # Continuer pour voir si c'est une classe concrète auto-enregistrable
                except CircularDependencyError: # Renvoyer si le parent détecte un cycle
                    raise

            # Si non enregistré et pas de parent, ou si le parent n'a pas pu résoudre,
            # vérifier si 'interface' est une classe concrète que l'on peut auto-enregistrer et instancier.
            if inspect.isclass(interface):
                logger.info(f"{log_prefix_resolve} '{interface.__name__}' non enregistré explicitement. "
                            "Tentative d'auto-enregistrement et résolution comme classe concrète (scope prototype par défaut).")
                # Auto-enregistrer avec scope prototype par défaut pour les classes concrètes non enregistrées
                self.register(interface, implementation=interface, scope=SCOPE_PROTOTYPE)
                registration = self._registrations.get(interface)
                if not registration: # Ne devrait pas arriver après un register réussi
                    raise DependencyResolutionError(f"Auto-enregistrement de {interface.__name__} a échoué de manière inattendue.")
            else:
                raise DependencyResolutionError(f"Aucun enregistrement trouvé pour le type/interface '{interface.__name__ if hasattr(interface, '__name__') else str(interface)}' et ce n'est pas une classe concrète auto-enregistrable.")

        # Gérer les scopes
        if registration.scope == SCOPE_SINGLETON:
            # Gérer les instances pré-fournies
            if 'singleton_instance' in registration.cached_instances:
                logger.debug(f"{log_prefix_resolve} Retour de l'instance singleton pré-fournie.")
                return cast(T, registration.cached_instances['singleton_instance'])
            
            # Gérer les singletons normaux (créés par le conteneur)
            # La clé 'instance' est utilisée pour le singleton créé par le conteneur.
            if 'instance' not in registration.cached_instances:
                with self._resolution_lock: # Protéger la création du singleton
                    if 'instance' not in registration.cached_instances: # Double-check
                        logger.debug(f"{log_prefix_resolve} Création de l'instance singleton...")
                        instance = self._create_instance(registration, new_resolution_path)
                        registration.cached_instances['instance'] = instance
            return cast(T, registration.cached_instances['instance'])

        elif registration.scope == SCOPE_PROTOTYPE:
            logger.debug(f"{log_prefix_resolve} Création d'une nouvelle instance (scope prototype)...")
            return cast(T, self._create_instance(registration, new_resolution_path))

        elif registration.scope == SCOPE_THREAD:
            thread_id = threading.get_ident()
            if not hasattr(self._thread_scoped_storage, 'instances'):
                self._thread_scoped_storage.instances = {} # type: ignore
            
            thread_cache_key = (interface, thread_id)
            if thread_cache_key not in self._thread_scoped_storage.instances: # type: ignore
                logger.debug(f"{log_prefix_resolve} Création d'une instance thread-local pour thread {thread_id}...")
                instance = self._create_instance(registration, new_resolution_path)
                self._thread_scoped_storage.instances[thread_cache_key] = instance # type: ignore
            return cast(T, self._thread_scoped_storage.instances[thread_cache_key]) # type: ignore

        else: # Ne devrait pas arriver si la validation du scope est correcte
            raise DependencyResolutionError(f"Scope inconnu '{registration.scope}' pour {interface.__name__ if hasattr(interface, '__name__') else str(interface)}")


    def resolve(self, interface: Type[T]) -> T:
        """
        Résout une dépendance et retourne une instance.
        Gère les dépendances transitives et la détection de cycles.

        Args:
            interface (Type[T]): Le type (interface ou classe concrète) à résoudre.

        Returns:
            T: Une instance de l'implémentation enregistrée.

        Raises:
            DependencyResolutionError: Si la dépendance ne peut pas être résolue.
            CircularDependencyError: Si une dépendance circulaire est détectée.
        """
        # Initialiser le chemin de résolution pour ce thread si ce n'est pas déjà fait
        if not hasattr(self._currently_resolving, 'path'):
            self._currently_resolving.path = [] # type: ignore

        # `path` est une liste pour suivre la chaîne de résolution pour ce thread
        current_path_for_thread = cast(List[Type[Any]], self._currently_resolving.path) # type: ignore
        
        try:
            instance = self._resolve_internal(interface, current_path_for_thread)
            return instance
        finally:
            # Nettoyer le chemin de résolution pour ce thread seulement si on est revenu
            # à l'appel initial de resolve() pour ce thread.
            # Ceci est géré par le fait que _resolve_internal passe une copie modifiée du chemin.
            # La vraie gestion de la pile de résolution par thread est plus complexe
            # si on veut la vider correctement.
            # Pour une détection de cycle simple, ajouter/retirer du path dans _resolve_internal est plus direct.
            
            # Simplification: si le path est vide après cet appel, c'est qu'on a fini la résolution racine.
            # La gestion de current_resolution_path dans _resolve_internal (en passant une copie new_resolution_path)
            # signifie que le `path` original de `self._currently_resolving` n'est pas directement modifié par les appels récursifs.
            # La détection de cycle se fait en vérifiant si `interface` est déjà dans `current_resolution_path` *passé* à _resolve_internal.
            pass # Le nettoyage du path est implicitement géré par la portée des appels récursifs.


    def configure(self, config: Union[Dict[str, Any], Path, str]) -> None:
        """
        Configure le conteneur à partir d'un dictionnaire ou d'un fichier YAML/JSON.

        Le format attendu du dictionnaire de configuration est :
        {
            "registrations": [
                {
                    "interface": "module.path.to.InterfaceClass", // ou un type direct
                    "implementation": "module.path.to.ConcreteClass", // ou type/factory
                    "scope": "singleton" // optionnel, défaut singleton
                },
                // ... autres enregistrements ...
            ]
        }

        Args:
            config (Union[Dict[str, Any], Path, str]): Un dictionnaire de configuration
                ou un chemin vers un fichier de configuration (JSON ou YAML).
        """
        with self._resolution_lock:
            log_prefix_conf = f"{self.log_prefix}[Configure]"
            config_data: Dict[str, Any]

            if isinstance(config, (Path, str)):
                config_path = Path(config)
                if not config_path.is_file():
                    raise FileNotFoundError(f"{log_prefix_conf} Fichier de configuration non trouvé : {config_path}")
                
                logger.info(f"{log_prefix_conf} Chargement de la configuration depuis le fichier : {config_path}")
                file_ext = config_path.suffix.lower()
                if file_ext == ".json":
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                elif file_ext in [".yaml", ".yml"]:
                    if not YAML_AVAILABLE:
                        raise ImportError(f"{log_prefix_conf} La bibliothèque PyYAML est requise pour charger les fichiers YAML. "
                                          "Veuillez l'installer (`pip install PyYAML`).")
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f) # type: ignore
                else:
                    raise ValueError(f"{log_prefix_conf} Format de fichier de configuration non supporté : {file_ext}. "
                                     "Utilisez JSON ou YAML.")
                self._config_loaded_path = config_path
            elif isinstance(config, dict):
                config_data = config
                self._config_loaded_path = None # Configuration depuis un dict en mémoire
            else:
                raise TypeError(f"{log_prefix_conf} Le type de configuration doit être dict, Path, ou str.")

            if not isinstance(config_data, dict) or "registrations" not in config_data:
                raise ValueError(f"{log_prefix_conf} La configuration doit être un dictionnaire avec une clé 'registrations'.")
            
            registrations_list = config_data.get("registrations", [])
            if not isinstance(registrations_list, list):
                raise ValueError(f"{log_prefix_conf} La clé 'registrations' doit contenir une liste.")

            for reg_entry in registrations_list:
                if not isinstance(reg_entry, dict):
                    logger.warning(f"{log_prefix_conf} Entrée d'enregistrement invalide (non-dict) : {reg_entry}. Ignorée.")
                    continue
                
                interface_repr = reg_entry.get("interface")
                implementation_repr = reg_entry.get("implementation")
                scope = reg_entry.get("scope", SCOPE_SINGLETON)
                
                if not interface_repr:
                    logger.warning(f"{log_prefix_conf} 'interface' manquante dans l'entrée d'enregistrement : {reg_entry}. Ignorée.")
                    continue

                try:
                    interface_type = self._resolve_type_from_string(interface_repr, "interface")
                    implementation_obj: Optional[Union[Type[Any], Callable[..., Any]]] = None
                    if implementation_repr:
                        implementation_obj = self._resolve_type_from_string(implementation_repr, "implementation")
                    
                    self.register(interface_type, implementation_obj, scope)
                except (TypeError, ValueError, ImportError, AttributeError) as e_reg_entry:
                    logger.error(f"{log_prefix_conf} Échec de l'enregistrement de l'entrée {reg_entry}: {e_reg_entry}", exc_info=True)
            
            logger.info(f"{log_prefix_conf} Configuration appliquée. {len(registrations_list)} enregistrements traités.")

    def _resolve_type_from_string(self, type_repr: Union[str, Type[Any], Callable[..., Any]], context_name: str) -> Union[Type[Any], Callable[..., Any]]:
        """Tente de résoudre un type ou une factory à partir d'une chaîne (ex: 'module.Classe')."""
        if isinstance(type_repr, str):
            try:
                module_name, class_or_func_name = type_repr.rsplit('.', 1)
                module = importlib.import_module(module_name)
                resolved_obj = getattr(module, class_or_func_name)
                if not (isinstance(resolved_obj, type) or callable(resolved_obj)):
                     raise TypeError(f"L'objet résolu '{type_repr}' n'est ni un type ni une factory appelable.")
                return resolved_obj
            except (ValueError, ImportError, AttributeError) as e_resolve_str:
                logger.error(f"{self.log_prefix} Impossible de résoudre '{type_repr}' pour {context_name}: {e_resolve_str}")
                raise ImportError(f"Impossible de charger {context_name} '{type_repr}': {e_resolve_str}") from e_resolve_str
        elif isinstance(type_repr, type) or callable(type_repr):
            return type_repr # Déjà un type ou une factory
        else:
            raise TypeError(f"{context_name} doit être une chaîne, un type, ou une factory appelable. Reçu: {type(type_repr)}")


    def create_child_container(self, overrides: Optional[Dict[Type[Any], Dict[str, Any]]] = None) -> 'DependencyContainer':
        """
        Crée un conteneur enfant qui hérite des enregistrements du parent
        et peut avoir ses propres enregistrements ou surcharges.

        Args:
            overrides (Optional[Dict[Type[Any], Dict[str, Any]]]): Un dictionnaire
                d'enregistrements à surcharger ou ajouter dans le conteneur enfant.
                Le format est {InterfaceType: {"implementation": ImplType, "scope": "scope_val"}}.

        Returns:
            DependencyContainer: Une nouvelle instance de conteneur enfant.
        """
        with self._resolution_lock: # Protéger la modification de _child_containers si nécessaire
            child = DependencyContainer(parent=self)
            # Les enregistrements du parent sont accessibles via la chaîne de résolution.
            # Le conteneur enfant commence avec ses propres enregistrements vides,
            # mais sa méthode `resolve` consultera le parent si une dépendance n'est pas
            # trouvée localement.
            
            if overrides:
                logger.info(f"{self.log_prefix} Création d'un conteneur enfant avec {len(overrides)} surcharge(s).")
                for interface, override_config in overrides.items():
                    impl = override_config.get("implementation")
                    scope = override_config.get("scope", SCOPE_SINGLETON) # Utiliser le scope de l'override
                    instance = override_config.get("instance") # Pour surcharger avec une instance spécifique
                    
                    # Enregistrer l'override dans l'enfant
                    child.register(interface, implementation=impl, scope=scope, instance=instance)
            else:
                logger.info(f"{self.log_prefix} Création d'un conteneur enfant sans surcharges initiales.")
            
            # Garder une référence faible à l'enfant (optionnel, pour suivi ou nettoyage)
            # self._child_containers.append(weakref.ref(child))
            return child

    def dispose(self) -> None:
        """
        Nettoie les ressources du conteneur.
        Pour les singletons qui pourraient implémenter une méthode `dispose()` ou `close()`,
        cette méthode pourrait les appeler.
        Vide également les enregistrements et les instances cachées.
        """
        with self._resolution_lock:
            logger.info(f"{self.log_prefix} Nettoyage du conteneur de dépendances...")
            # Appeler dispose sur les singletons qui le supportent (conceptuel)
            for reg in self._registrations.values():
                if reg.scope == SCOPE_SINGLETON and 'instance' in reg.cached_instances:
                    instance = reg.cached_instances.get('instance')
                    if hasattr(instance, 'dispose') and callable(getattr(instance, 'dispose')):
                        try:
                            logger.debug(f"{self.log_prefix} Appel de dispose() sur le singleton {type(instance).__name__}")
                            getattr(instance, 'dispose')()
                        except Exception as e_dispose:
                            logger.warning(f"{self.log_prefix} Erreur lors de dispose() sur {type(instance).__name__}: {e_dispose}")
            
            self._registrations.clear()
            if hasattr(self._thread_scoped_storage, 'instances'):
                self._thread_scoped_storage.instances.clear() # type: ignore
            
            # Si ce conteneur est le singleton global, le "réinitialiser" pour un usage futur potentiel
            # (bien qu'un vrai singleton ne devrait pas être "disposé" et réutilisé de cette manière).
            # Il est plus probable que `dispose` soit pour les conteneurs enfants.
            if DependencyContainer._instance is self:
                logger.debug(f"{self.log_prefix} Le conteneur global a été disposé (enregistrements vidés).")
                # Ne pas mettre _instance à None ici, car __new__ le gère.
                # Mais on peut réinitialiser son état interne.
                # self._initialized_for_singleton = False # Permettrait une ré-initialisation, mais attention.
            
            logger.info(f"{self.log_prefix} Nettoyage terminé.")


# Accès global au singleton (si nécessaire en dehors d'une injection explicite)
# GlobalDIContainer = DependencyContainer()
