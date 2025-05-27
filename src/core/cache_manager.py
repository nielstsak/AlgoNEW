# src/core/cache_manager.py
"""
Gestionnaire de cache LRU (Least Recently Used) intelligent, thread-safe,
avec support pour la persistance sur disque, limites de mémoire, et TTL.
"""
import collections
import functools
import hashlib
import logging
import pickle
import re
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type, Union,
                    Generic, TypeVar)

import pandas as pd
import psutil # Pour surveiller l'utilisation mémoire du processus

# Tentative d'importation de l'interface ICacheManager
try:
    from src.core.interfaces import ICacheManager
except ImportError:
    # Protocole factice si l'interface réelle n'est pas trouvée
    class ICacheManager: # type: ignore
        def get_or_compute(self, key: str, compute_func: Callable[[], Any],
                           ttl: Optional[int] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Any: ...
        def invalidate(self, pattern: str, recursive: bool = False) -> int: ...
        def get_statistics(self) -> Dict[str, Any]: ...
        def persist_to_disk(self, path: Path, compress: bool = True) -> None: ...
    logging.getLogger(__name__).warning(
        "ICacheManager interface not found. Using a placeholder. "
        "Ensure src.core.interfaces is in PYTHONPATH for full type safety."
    )

logger = logging.getLogger(__name__)

# Type générique pour la valeur stockée dans le cache
V = TypeVar('V')

DEFAULT_MAX_MEMORY_MB = 1024  # 1 GB
DEFAULT_TTL_SECONDS = 3600  # 1 hour
BYTES_IN_MB = 1024 * 1024

@dataclass
class CacheEntry(Generic[V]):
    """Représente une entrée dans le cache."""
    key: str
    value: V
    expiry_time: Optional[float] = None  # Timestamp d'expiration (time.time() + ttl)
    created_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    size_bytes: int = 0
    hit_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def is_expired(self) -> bool:
        """Vérifie si l'entrée du cache a expiré."""
        if self.expiry_time is None:
            return False  # Pas de TTL, n'expire jamais explicitement par temps
        return time.time() > self.expiry_time

@dataclass
class CacheStatistics:
    """Statistiques d'utilisation et de performance du cache."""
    total_items: int
    current_memory_usage_bytes: int
    max_memory_bytes: int
    hits: int
    misses: int
    expirations_on_get: int # Nombre de fois où un get a trouvé un item expiré
    evictions_for_space: int # Nombre d'items évincés pour faire de la place
    hit_rate: float
    process_memory_mb: float # Utilisation mémoire totale du processus Python
    top_n_keys_by_hits: List[Tuple[str, int]] = field(default_factory=list)
    top_n_keys_by_size: List[Tuple[str, int]] = field(default_factory=list)


class CacheManager(ICacheManager):
    """
    Gestionnaire de cache singleton implémentant une politique LRU avec
    limites de mémoire, TTL, et persistance sur disque.
    """
    _instance: Optional['CacheManager'] = None
    _lock = threading.RLock() # RLock pour la réentrance si nécessaire

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None: # Double-check locking
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 max_memory_mb: int = DEFAULT_MAX_MEMORY_MB,
                 default_ttl_seconds: Optional[int] = DEFAULT_TTL_SECONDS,
                 persist_path: Optional[Union[str, Path]] = None,
                 auto_load_persist: bool = True,
                 auto_persist_on_change: bool = False, # Peut être coûteux
                 eviction_check_interval_seconds: int = 60): # Pour un thread de nettoyage optionnel
        """
        Initialise le CacheManager. L'initialisation ne se produit qu'une fois
        pour le singleton.

        Args:
            max_memory_mb (int): Taille maximale du cache en mégaoctets.
            default_ttl_seconds (Optional[int]): Durée de vie par défaut des entrées
                                                 en secondes. None pour pas de TTL par défaut.
            persist_path (Optional[Union[str, Path]]): Chemin vers le fichier pour la
                                                       persistance du cache.
            auto_load_persist (bool): Si True, charge automatiquement le cache depuis
                                      persist_path à l'initialisation s'il existe.
            auto_persist_on_change (bool): Si True, sauvegarde le cache sur disque après
                                           chaque modification (peut impacter la performance).
            eviction_check_interval_seconds (int): Intervalle pour le thread de vérification
                                                   de la mémoire et d'éviction (non implémenté
                                                   dans cette version, l'éviction est sur get/set).
        """
        # L'initialisation du singleton est délicate. On s'assure qu'elle n'est faite qu'une fois.
        if hasattr(self, '_initialized') and self._initialized:
            return

        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return

            self.log_prefix = "[CacheManager]"
            logger.info(f"{self.log_prefix} Initialisation du singleton CacheManager...")

            self._cache: collections.OrderedDict[str, CacheEntry[Any]] = collections.OrderedDict()
            self._max_memory_bytes: int = max_memory_mb * BYTES_IN_MB
            self._default_ttl_seconds: Optional[int] = default_ttl_seconds
            self._current_memory_bytes: int = 0

            # Statistiques
            self._hits: int = 0
            self._misses: int = 0
            self._expirations_on_get: int = 0
            self._evictions_for_space: int = 0

            self._persist_path: Optional[Path] = Path(persist_path) if persist_path else None
            self._auto_persist_on_change: bool = auto_persist_on_change

            if self._persist_path and auto_load_persist:
                self.load_from_disk(self._persist_path)

            self._process = psutil.Process() # Pour l'utilisation mémoire du processus

            self._initialized: bool = True # Marquer comme initialisé
            logger.info(f"{self.log_prefix} CacheManager initialisé. Max Memory: {max_memory_mb}MB, "
                        f"Default TTL: {default_ttl_seconds}s, Persist Path: {self._persist_path}")

    def _generate_cache_key(self, item: Any) -> str:
        """Génère une clé de cache SHA256 pour des objets complexes (comme des dicts de config)."""
        if isinstance(item, str):
            # Si c'est déjà une chaîne, on peut l'utiliser directement ou la hasher si on veut uniformiser
            # Pour l'instant, on suppose que les clés string sont déjà bien formées.
            # On pourrait ajouter un hash si on s'attend à des clés string très longues.
            return item
        try:
            # Tenter de sérialiser avec pickle pour obtenir une représentation binaire stable
            # des objets Python, puis hasher.
            # Utiliser sort_keys=True pour les dictionnaires via json.dumps avant pickle
            # pour s'assurer que l'ordre des clés n'affecte pas le hash.
            if isinstance(item, dict):
                # Tenter une sérialisation JSON stable pour les dictionnaires
                # avant de pickler, car l'ordre des clés dans un dict peut varier.
                # Cela suppose que le contenu du dict est sérialisable en JSON.
                try:
                    stable_representation = json.dumps(item, sort_keys=True, default=str).encode('utf-8')
                except TypeError:
                    # Fallback sur pickle directement si le JSON échoue (ex: objets non sérialisables en JSON)
                    stable_representation = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                stable_representation = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)

            return hashlib.sha256(stable_representation).hexdigest()
        except (pickle.PicklingError, TypeError) as e:
            logger.warning(f"{self.log_prefix} Échec de la génération de clé de cache pour l'objet {type(item)}: {e}. "
                           f"Utilisation de repr() comme fallback (moins fiable).")
            # Fallback moins fiable mais qui évite un crash
            return hashlib.sha256(repr(item).encode('utf-8')).hexdigest()


    def _get_item_size_bytes(self, item_value: Any) -> int:
        """Estime la taille en octets d'un élément du cache."""
        if isinstance(item_value, pd.DataFrame):
            return int(item_value.memory_usage(deep=True).sum())
        if isinstance(item_value, (str, bytes)):
            return len(item_value)
        try:
            # Pour d'autres objets, une estimation via pickle peut être faite.
            # C'est coûteux et approximatif.
            return len(pickle.dumps(item_value, protocol=pickle.HIGHEST_PROTOCOL))
        except (pickle.PicklingError, TypeError):
            # Fallback très basique si pickle échoue
            import sys
            return sys.getsizeof(item_value)

    def _evict_lru_for_space(self, required_space: int) -> None:
        """Évince les éléments les moins récemment utilisés (LRU) jusqu'à libérer l'espace requis."""
        with self._lock:
            logger.debug(f"{self.log_prefix} Tentative d'éviction LRU pour libérer {required_space} octets. "
                         f"Mémoire actuelle: {self._current_memory_bytes}/{self._max_memory_bytes}")
            freed_space = 0
            # L'OrderedDict maintient l'ordre d'insertion. Pour LRU, popitem(last=False) enlève le plus ancien.
            while self._cache and (self._current_memory_bytes + required_space - freed_space > self._max_memory_bytes or freed_space < required_space):
                try:
                    old_key, old_entry = self._cache.popitem(last=False) # FIFO simule LRU si on déplace à la fin lors du get
                    freed_this_item = old_entry.size_bytes
                    self._current_memory_bytes -= freed_this_item
                    freed_space += freed_this_item
                    self._evictions_for_space += 1
                    logger.info(f"{self.log_prefix} Élément LRU '{old_key}' (taille: {freed_this_item}B) évincé pour faire de la place. "
                                f"Espace libéré total: {freed_space}B. Mémoire actuelle: {self._current_memory_bytes}B.")
                    if self._current_memory_bytes < 0: self._current_memory_bytes = 0 # Sanity check
                except KeyError: # Devrait pas arriver avec un OrderedDict non vide
                    break
            if freed_space < required_space:
                 logger.warning(f"{self.log_prefix} Éviction LRU n'a pas pu libérer assez d'espace. "
                                f"Requis: {required_space}B, Libéré: {freed_space}B.")


    def get_or_compute(
        self,
        key: str, # La clé est déjà une chaîne (peut-être générée par _generate_cache_key en amont)
        compute_func: Callable[[], V],
        ttl: Optional[int] = None, # TTL spécifique pour cet item
        metadata: Optional[Dict[str, Any]] = None
    ) -> V:
        """Récupère ou calcule, met en cache, et retourne une valeur."""
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                if entry.is_expired():
                    logger.info(f"{self.log_prefix} Clé '{key}' trouvée mais expirée. Recalcul.")
                    self._misses += 1
                    self._expirations_on_get += 1
                    # Supprimer l'entrée expirée explicitement pour recalculer la taille
                    self._current_memory_bytes -= entry.size_bytes
                    if self._current_memory_bytes < 0: self._current_memory_bytes = 0
                    del self._cache[key]
                    # Ne pas retourner, continuer pour recalculer
                else:
                    logger.debug(f"{self.log_prefix} Cache HIT pour la clé '{key}'.")
                    self._hits += 1
                    entry.last_access_time = time.time()
                    entry.hit_count += 1
                    self._cache.move_to_end(key) # Marquer comme récemment utilisé pour LRU
                    return entry.value

            # Cache MISS ou item expiré
            logger.debug(f"{self.log_prefix} Cache MISS pour la clé '{key}'. Appel de compute_func.")
            self._misses += 1
            
            # Libérer le verrou pendant le calcul pour permettre à d'autres opérations de cache
            # non liées à cette clé de se produire. C'est complexe si compute_func
            # interagit elle-même avec le cache. Pour un compute_func simple, c'est ok.
            # Pour l'instant, on garde le verrou pour simplicité.
            # self._lock.release()
            try:
                value_to_cache = compute_func()
            except Exception as e_compute:
                logger.error(f"{self.log_prefix} Erreur lors de l'exécution de compute_func pour la clé '{key}': {e_compute}", exc_info=True)
                raise # Renvoyer l'exception de compute_func
            # finally:
            #     self._lock.acquire() # Ré-acquérir le verrou

            item_size = self._get_item_size_bytes(value_to_cache)

            # Vérifier si l'espace est suffisant, évincer si nécessaire
            if self._current_memory_bytes + item_size > self._max_memory_bytes:
                self._evict_lru_for_space(item_size - (self._max_memory_bytes - self._current_memory_bytes))
            
            # Vérifier à nouveau si l'espace est suffisant après éviction
            if self._current_memory_bytes + item_size <= self._max_memory_bytes:
                actual_ttl = ttl if ttl is not None else self._default_ttl_seconds
                expiry_timestamp = (time.time() + actual_ttl) if actual_ttl is not None else None
                
                new_entry = CacheEntry(
                    key=key,
                    value=value_to_cache,
                    expiry_time=expiry_timestamp,
                    size_bytes=item_size,
                    metadata=metadata
                )
                self._cache[key] = new_entry
                self._current_memory_bytes += item_size
                self._cache.move_to_end(key) # Marquer comme récemment ajouté/utilisé
                logger.info(f"{self.log_prefix} Clé '{key}' ajoutée au cache. Taille: {item_size}B, TTL: {actual_ttl}s. "
                            f"Mémoire actuelle: {self._current_memory_bytes / BYTES_IN_MB:.2f}MB / {self._max_memory_bytes / BYTES_IN_MB:.2f}MB.")
                if self._auto_persist_on_change and self._persist_path:
                    self.persist_to_disk(self._persist_path)
            else:
                logger.warning(f"{self.log_prefix} Pas assez d'espace pour la clé '{key}' (taille: {item_size}B) même après éviction. "
                               "L'élément ne sera pas mis en cache.")
            
            return value_to_cache

    def invalidate(self, pattern: str, recursive: bool = False) -> int:
        """Invalide les entrées du cache correspondant à un pattern regex."""
        # `recursive` n'est pas directement applicable à une structure de cache plate,
        # mais le pattern regex peut simuler une invalidation "hiérarchique" si les clés sont structurées.
        with self._lock:
            keys_to_delete = [k for k in self._cache if re.search(pattern, k)]
            invalidated_count = 0
            for k_del in keys_to_delete:
                try:
                    entry_to_del = self._cache.pop(k_del)
                    self._current_memory_bytes -= entry_to_del.size_bytes
                    invalidated_count += 1
                    logger.info(f"{self.log_prefix} Clé '{k_del}' invalidée (pattern: '{pattern}').")
                except KeyError:
                    pass # Déjà supprimée par une autre opération/thread
            
            if self._current_memory_bytes < 0: self._current_memory_bytes = 0
            
            if invalidated_count > 0 and self._auto_persist_on_change and self._persist_path:
                self.persist_to_disk(self._persist_path)
            
            logger.info(f"{self.log_prefix} {invalidated_count} entrée(s) invalidée(s) pour le pattern '{pattern}'.")
            return invalidated_count

    def get_statistics(self) -> CacheStatistics: # Remplacer Dict[str, Any] par CacheStatistics
        """Retourne les statistiques du cache."""
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate_calc = (self._hits / total_accesses * 100.0) if total_accesses > 0 else 0.0
            
            # Obtenir l'utilisation mémoire du processus Python
            process_mem_mb = self._process.memory_info().rss / BYTES_IN_MB

            # Obtenir les top N clés par hits et par taille
            top_n = 10 # Configurable si besoin
            sorted_by_hits = sorted(self._cache.values(), key=lambda e: e.hit_count, reverse=True)
            top_hits_list = [(e.key, e.hit_count) for e in sorted_by_hits[:top_n]]

            sorted_by_size = sorted(self._cache.values(), key=lambda e: e.size_bytes, reverse=True)
            top_size_list = [(e.key, e.size_bytes) for e in sorted_by_size[:top_n]]

            stats = CacheStatistics(
                total_items=len(self._cache),
                current_memory_usage_bytes=self._current_memory_bytes,
                max_memory_bytes=self._max_memory_bytes,
                hits=self._hits,
                misses=self._misses,
                expirations_on_get=self._expirations_on_get,
                evictions_for_space=self._evictions_for_space,
                hit_rate=hit_rate_calc,
                process_memory_mb=process_mem_mb,
                top_n_keys_by_hits=top_hits_list,
                top_n_keys_by_size=top_size_list
            )
            logger.debug(f"{self.log_prefix} Statistiques du cache récupérées : {stats}")
            return stats

    def persist_to_disk(self, path: Optional[Path] = None, compress: bool = True) -> None:
        """Sauvegarde le cache sur disque."""
        target_path = path if path else self._persist_path
        if not target_path:
            logger.error(f"{self.log_prefix} Aucun chemin de persistance fourni ou configuré. Sauvegarde annulée.")
            return

        with self._lock:
            logger.info(f"{self.log_prefix} Tentative de sauvegarde du cache sur disque : {target_path} (Compression: {compress})")
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # On sauvegarde l'OrderedDict _cache et les compteurs de stats pour une reprise plus fidèle
                data_to_persist = {
                    'cache_data': self._cache,
                    'stats_hits': self._hits,
                    'stats_misses': self._misses,
                    'stats_expirations': self._expirations_on_get,
                    'stats_evictions': self._evictions_for_space,
                    'current_memory_bytes': self._current_memory_bytes # Sauvegarder pour cohérence au chargement
                }
                pickled_data = pickle.dumps(data_to_persist, protocol=pickle.HIGHEST_PROTOCOL)
                if compress:
                    pickled_data = zlib.compress(pickled_data)
                
                with open(target_path, 'wb') as f:
                    f.write(pickled_data)
                logger.info(f"{self.log_prefix} Cache sauvegardé avec succès sur {target_path}.")
            except Exception as e_persist:
                logger.error(f"{self.log_prefix} Échec de la sauvegarde du cache sur {target_path}: {e_persist}", exc_info=True)

    def load_from_disk(self, path: Optional[Path] = None) -> bool:
        """Charge le cache depuis le disque."""
        target_path = path if path else self._persist_path
        if not target_path or not target_path.is_file():
            logger.warning(f"{self.log_prefix} Fichier de persistance du cache non trouvé à {target_path}. Chargement annulé.")
            return False

        with self._lock:
            logger.info(f"{self.log_prefix} Tentative de chargement du cache depuis : {target_path}")
            try:
                with open(target_path, 'rb') as f:
                    persisted_data_compressed = f.read()
                
                # Tenter de décompresser (si zlib.error, supposer non compressé pour rétrocompatibilité)
                try:
                    persisted_data_raw = zlib.decompress(persisted_data_compressed)
                except zlib.error:
                    logger.info(f"{self.log_prefix} Données non compressées détectées (ou erreur zlib). Tentative de dépickling direct.")
                    persisted_data_raw = persisted_data_compressed
                
                loaded_content = pickle.loads(persisted_data_raw)

                if isinstance(loaded_content, dict) and 'cache_data' in loaded_content:
                    # Restaurer le cache et les statistiques
                    self._cache = loaded_content.get('cache_data', collections.OrderedDict())
                    self._hits = loaded_content.get('stats_hits', 0)
                    self._misses = loaded_content.get('stats_misses', 0)
                    self._expirations_on_get = loaded_content.get('stats_expirations', 0)
                    self._evictions_for_space = loaded_content.get('stats_evictions', 0)
                    
                    # Recalculer _current_memory_bytes basé sur les entrées chargées
                    self._current_memory_bytes = sum(entry.size_bytes for entry in self._cache.values())
                    logger.info(f"{self.log_prefix} Cache chargé avec succès depuis {target_path}. "
                                f"{len(self._cache)} éléments. Mémoire recalculée: {self._current_memory_bytes / BYTES_IN_MB:.2f}MB.")
                    return True
                elif isinstance(loaded_content, collections.OrderedDict): # Ancien format de sauvegarde (juste le dict _cache)
                    self._cache = loaded_content
                    self._current_memory_bytes = sum(entry.size_bytes for entry in self._cache.values())
                    # Les stats ne sont pas restaurées dans ce cas, elles repartent de zéro.
                    self._hits = self._misses = self._expirations_on_get = self._evictions_for_space = 0
                    logger.info(f"{self.log_prefix} Cache (ancien format) chargé depuis {target_path}. "
                                f"{len(self._cache)} éléments. Stats réinitialisées. Mémoire: {self._current_memory_bytes / BYTES_IN_MB:.2f}MB.")
                    return True
                else:
                    logger.error(f"{self.log_prefix} Format de données invalide dans le fichier cache {target_path}.")
                    return False
            except FileNotFoundError:
                logger.warning(f"{self.log_prefix} Fichier cache {target_path} non trouvé (peut-être supprimé entre-temps).")
                return False
            except Exception as e_load:
                logger.error(f"{self.log_prefix} Échec du chargement du cache depuis {target_path}: {e_load}", exc_info=True)
                # Optionnel: invalider/supprimer le fichier cache corrompu
                # try: target_path.unlink(missing_ok=True)
                # except OSError: pass
                return False

    def clear(self) -> None:
        """Vide complètement le cache."""
        with self._lock:
            self._cache.clear()
            self._current_memory_bytes = 0
            self._hits = 0
            self._misses = 0
            self._expirations_on_get = 0
            self._evictions_for_space = 0
            logger.info(f"{self.log_prefix} Cache vidé.")
            if self._auto_persist_on_change and self._persist_path:
                self.persist_to_disk(self._persist_path)

# Pour obtenir l'instance singleton :
# cache_instance = CacheManager()
