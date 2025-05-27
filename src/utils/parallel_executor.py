# src/utils/parallel_executor.py
"""
Gestionnaire d'exécution parallèle optimisé, offrant des fonctionnalités
telles que le choix de l'exécuteur (Thread/Process), les réessais,
le circuit breaker, la barre de progression, et le monitoring des ressources.
"""
import asyncio
import concurrent.futures
import functools
import logging
import os
import queue # Pour la communication inter-processus/thread si nécessaire
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (Any, Callable, Dict, Iterable, List, Optional, TypeVar,
                    Generic, Union, Coroutine)

import psutil # Pour le monitoring des ressources
from tqdm import tqdm # Pour la barre de progression

logger = logging.getLogger(__name__)

T = TypeVar('T') # Type de la tâche (fonction)
R = TypeVar('R') # Type du résultat de la tâche

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5 # Nombre d'échecs consécutifs pour ouvrir le circuit
DEFAULT_CIRCUIT_BREAKER_TIMEOUT_SECONDS = 60 # Temps avant de retenter après ouverture du circuit
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30


class ExecutorType(Enum):
    """Type d'exécuteur à utiliser."""
    THREAD = "thread"
    PROCESS = "process"
    ASYNCIO = "asyncio" # Pour les tâches purement asyncio

@dataclass
class Task(Generic[T, R]):
    """
    Représente une tâche à exécuter.
    Permet de stocker la fonction, ses arguments, et des métadonnées.
    """
    func: Callable[..., R] # La fonction à exécuter
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    # Pour les réessais et le circuit breaker
    retries_left: int = DEFAULT_MAX_RETRIES
    # Pour le type d'exécuteur si on veut le spécifier par tâche (plus complexe)
    # executor_preference: Optional[ExecutorType] = None

@dataclass
class TaskResult(Generic[R]):
    """
    Représente le résultat d'une tâche, incluant le statut et les erreurs potentielles.
    """
    task_id: str
    success: bool
    result: Optional[R] = None
    exception: Optional[Exception] = None
    traceback_str: Optional[str] = None
    retries_attempted: int = 0

@dataclass
class ResourceStats:
    """Statistiques sur l'utilisation des ressources."""
    cpu_percent_system: float
    cpu_percent_process: float
    memory_percent_system: float
    memory_rss_process_mb: float
    memory_vms_process_mb: float
    active_threads: int
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CircuitBreaker:
    """Implémentation simple d'un circuit breaker."""
    STATE_CLOSED = "CLOSED"
    STATE_OPEN = "OPEN"
    STATE_HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int, recovery_timeout_seconds: int, name: str = "DefaultCircuit"):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._lock = threading.RLock()
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_attempt_allowed = True # Permet une tentative en half-open

        logger.info(f"[CircuitBreaker:{self.name}] Initialisé. Seuil: {failure_threshold}, Timeout: {recovery_timeout_seconds}s")

    @property
    def state(self) -> str:
        with self._lock:
            # Vérifier si on doit passer de OPEN à HALF_OPEN
            if self._state == self.STATE_OPEN and \
               self._last_failure_time is not None and \
               (time.monotonic() - self._last_failure_time) > self.recovery_timeout_seconds:
                self._state = self.STATE_HALF_OPEN
                self._half_open_attempt_allowed = True # Permettre une nouvelle tentative
                logger.info(f"[CircuitBreaker:{self.name}] Transition vers l'état HALF_OPEN après timeout.")
            return self._state

    def allow_request(self) -> bool:
        """Vérifie si une requête est autorisée selon l'état du circuit."""
        current_state = self.state # Propriété qui peut transitionner vers HALF_OPEN
        if current_state == self.STATE_OPEN:
            return False
        if current_state == self.STATE_HALF_OPEN:
            with self._lock: # Protéger l'accès à _half_open_attempt_allowed
                if self._half_open_attempt_allowed:
                    self._half_open_attempt_allowed = False # Autoriser une seule tentative
                    return True
                return False # Bloquer les autres tentatives en HALF_OPEN
        return True # CLOSED

    def record_success(self) -> None:
        """Enregistre un succès, réinitialise le circuit si en HALF_OPEN."""
        with self._lock:
            if self._state == self.STATE_HALF_OPEN:
                self._state = self.STATE_CLOSED
                self._failure_count = 0
                self._half_open_attempt_allowed = True # Réinitialiser pour la prochaine fois
                logger.info(f"[CircuitBreaker:{self.name}] Succès en HALF_OPEN. Transition vers CLOSED.")
            # Si déjà CLOSED, réinitialiser le compteur d'échecs est une bonne pratique
            # bien que ce ne soit pas strictement nécessaire si on ne l'incrémente que sur échec.
            self._failure_count = 0

    def record_failure(self) -> None:
        """Enregistre un échec. Peut ouvrir le circuit."""
        with self._lock:
            if self._state == self.STATE_HALF_OPEN:
                # L'échec en HALF_OPEN remet immédiatement en OPEN
                self._state = self.STATE_OPEN
                self._last_failure_time = time.monotonic()
                self._failure_count = self.failure_threshold # Assurer qu'il reste ouvert
                logger.warning(f"[CircuitBreaker:{self.name}] Échec en HALF_OPEN. Transition vers OPEN pour {self.recovery_timeout_seconds}s.")
                return

            self._failure_count += 1
            if self._failure_count >= self.failure_threshold and self._state == self.STATE_CLOSED:
                self._state = self.STATE_OPEN
                self._last_failure_time = time.monotonic()
                logger.warning(f"[CircuitBreaker:{self.name}] Seuil d'échec atteint ({self._failure_count}). "
                               f"Transition vers OPEN pour {self.recovery_timeout_seconds}s.")


class ParallelExecutor:
    """
    Gestionnaire d'exécution parallèle optimisé.
    """
    def __init__(self,
                 default_executor_type: ExecutorType = ExecutorType.THREAD,
                 default_max_workers: Optional[int] = None,
                 max_retries_per_task: int = DEFAULT_MAX_RETRIES,
                 retry_base_delay_s: float = DEFAULT_RETRY_DELAY_SECONDS,
                 use_circuit_breaker: bool = True,
                 circuit_failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
                 circuit_recovery_timeout_s: int = DEFAULT_CIRCUIT_BREAKER_TIMEOUT_SECONDS,
                 shutdown_timeout_s: int = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS):
        
        self.log_prefix = "[ParallelExecutor]"
        logger.info(f"{self.log_prefix} Initialisation. Exécuteur par défaut: {default_executor_type.value}, "
                    f"Max Workers par défaut: {default_max_workers or 'auto'}")

        self.default_executor_type = default_executor_type
        self.default_max_workers = default_max_workers if default_max_workers is not None else os.cpu_count() or 1
        
        self.max_retries_per_task = max_retries_per_task
        self.retry_base_delay_s = retry_base_delay_s
        self.shutdown_timeout_s = shutdown_timeout_s

        self.use_circuit_breaker = use_circuit_breaker
        if self.use_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=circuit_failure_threshold,
                recovery_timeout_seconds=circuit_recovery_timeout_s,
                name=f"ParallelExecutorCB"
            )
        else:
            self.circuit_breaker = None # type: ignore

        self._current_executor: Optional[Union[concurrent.futures.ThreadPoolExecutor, concurrent.futures.ProcessPoolExecutor]] = None
        self._resource_monitor_thread: Optional[threading.Thread] = None
        self._resource_monitor_stop_event = threading.Event()

    def _get_optimal_workers(self, executor_type: ExecutorType) -> int:
        """Détermine le nombre optimal de workers."""
        if self.default_max_workers and self.default_max_workers > 0:
            return self.default_max_workers
        
        cpu_cores = os.cpu_count() or 1
        if executor_type == ExecutorType.PROCESS:
            return cpu_cores
        elif executor_type == ExecutorType.THREAD:
            # Pour I/O-bound, on peut avoir plus de threads que de cœurs
            return min(32, cpu_cores * 5) # Limite raisonnable
        return cpu_cores # Fallback

    def _execute_single_task_with_retry(self, task: Task[Any, Any]) -> TaskResult[Any]:
        """Exécute une seule tâche avec logique de réessai."""
        task_log_prefix = f"{self.log_prefix}[Task:{task.task_id}]"
        retries_done = 0
        current_delay = self.retry_base_delay_s

        while task.retries_left >= 0: # task.retries_left est le nombre de réessais restants (0 = dernière tentative)
            attempt_number = self.max_retries_per_task - task.retries_left + 1
            logger.debug(f"{task_log_prefix} Tentative {attempt_number}/{self.max_retries_per_task + 1}...")
            
            if self.circuit_breaker and not self.circuit_breaker.allow_request():
                logger.warning(f"{task_log_prefix} Circuit breaker est OUVERT. La tâche ne sera pas exécutée.")
                # Ne pas décrémenter retries_left ici, car la tâche n'a pas été tentée.
                # On pourrait retourner une erreur spécifique ou attendre que le circuit se ferme.
                # Pour l'instant, on la considère comme un échec immédiat si le circuit est ouvert.
                return TaskResult(task_id=task.task_id, success=False, exception=RuntimeError("Circuit breaker ouvert"), retries_attempted=retries_done)

            try:
                result_val = task.func(*task.args, **task.kwargs)
                if self.circuit_breaker: self.circuit_breaker.record_success()
                return TaskResult(task_id=task.task_id, success=True, result=result_val, retries_attempted=retries_done)
            except Exception as e:
                logger.warning(f"{task_log_prefix} Échec tentative {attempt_number}: {e}", exc_info=self.verbosity >= 2 if hasattr(self, 'verbosity') else False)
                if self.circuit_breaker: self.circuit_breaker.record_failure()
                
                task.retries_left -= 1
                retries_done += 1
                if task.retries_left < 0: # Plus de réessais
                    logger.error(f"{task_log_prefix} Échec final après {retries_done} tentative(s): {e}")
                    import traceback
                    return TaskResult(task_id=task.task_id, success=False, exception=e, traceback_str=traceback.format_exc(), retries_attempted=retries_done)
                
                logger.info(f"{task_log_prefix} Réessai dans {current_delay:.2f}s... ({task.retries_left} réessai(s) restant(s))")
                time.sleep(current_delay)
                current_delay *= 2 # Backoff exponentiel
        
        # Ne devrait pas être atteint si la logique de la boucle est correcte
        return TaskResult(task_id=task.task_id, success=False, exception=RuntimeError("Logique de réessai inattendue"), retries_attempted=retries_done)


    async def _execute_single_async_task_with_retry(self, task: Task[Any, Coroutine[Any, Any, Any]]) -> TaskResult[Any]:
        """Exécute une seule tâche asyncio avec logique de réessai."""
        task_log_prefix = f"{self.log_prefix}[AsyncTask:{task.task_id}]"
        retries_done = 0
        current_delay = self.retry_base_delay_s
        
        # Assurer que task.func est bien une coroutine function
        if not asyncio.iscoroutinefunction(task.func):
            logger.error(f"{task_log_prefix} La fonction de tâche n'est pas une coroutine. Exécution synchrone tentée.")
            # Fallback sur l'exécution synchrone si ce n'est pas une coroutine
            # Cela suppose que _execute_single_task_with_retry peut gérer une func non-coroutine.
            # Idéalement, on lèverait une erreur ou on utiliserait une autre méthode.
            # Pour cet exemple, on va juste loguer et laisser _execute_single_task_with_retry gérer.
            # Ceci n'est pas idéal car le type de retour de Task est Coroutine.
            # On devrait avoir un TaskAsync distinct ou une validation plus stricte.
            return self._execute_single_task_with_retry(cast(Task[Any, Any], task))


        while task.retries_left >= 0:
            attempt_number = self.max_retries_per_task - task.retries_left + 1
            logger.debug(f"{task_log_prefix} Tentative async {attempt_number}/{self.max_retries_per_task + 1}...")

            if self.circuit_breaker and not self.circuit_breaker.allow_request():
                logger.warning(f"{task_log_prefix} Circuit breaker OUVERT (async). Tâche non exécutée.")
                return TaskResult(task_id=task.task_id, success=False, exception=RuntimeError("Circuit breaker ouvert (async)"), retries_attempted=retries_done)

            try:
                # task.func est une coroutine function, donc il faut l'appeler pour obtenir une coroutine
                coro = task.func(*task.args, **task.kwargs)
                result_val = await coro # Exécuter la coroutine
                if self.circuit_breaker: self.circuit_breaker.record_success()
                return TaskResult(task_id=task.task_id, success=True, result=result_val, retries_attempted=retries_done)
            except Exception as e:
                logger.warning(f"{task_log_prefix} Échec tentative async {attempt_number}: {e}", exc_info=self.verbosity >= 2 if hasattr(self, 'verbosity') else False)
                if self.circuit_breaker: self.circuit_breaker.record_failure()
                
                task.retries_left -= 1
                retries_done += 1
                if task.retries_left < 0:
                    logger.error(f"{task_log_prefix} Échec final async après {retries_done} tentative(s): {e}")
                    import traceback
                    return TaskResult(task_id=task.task_id, success=False, exception=e, traceback_str=traceback.format_exc(), retries_attempted=retries_done)
                
                logger.info(f"{task_log_prefix} Réessai async dans {current_delay:.2f}s...")
                await asyncio.sleep(current_delay)
                current_delay *= 2
        
        return TaskResult(task_id=task.task_id, success=False, exception=RuntimeError("Logique de réessai async inattendue"), retries_attempted=retries_done)


    def execute_parallel(
        self,
        tasks: Iterable[Task[Any, Any]],
        executor_type: Optional[ExecutorType] = None,
        max_workers: Optional[int] = None,
        description: Optional[str] = None, # Pour tqdm
        show_progress: bool = False
    ) -> List[TaskResult[Any]]:
        """
        Exécute une liste de tâches en parallèle.
        Choisit automatiquement entre ThreadPoolExecutor et ProcessPoolExecutor
        si executor_type n'est pas spécifié.
        """
        exec_type_to_use = executor_type if executor_type is not None else self.default_executor_type
        num_workers = max_workers if max_workers is not None else self._get_optimal_workers(exec_type_to_use)
        
        tasks_list = list(tasks) # Convertir l'itérable en liste pour tqdm et len
        if not tasks_list:
            return []

        log_prefix_exec = f"{self.log_prefix}[ExecuteParallel][{exec_type_to_use.value}][W:{num_workers}]"
        logger.info(f"{log_prefix_exec} Démarrage de l'exécution de {len(tasks_list)} tâche(s).")

        results: List[TaskResult[Any]] = []
        
        # Gérer le cas ASYNCIO séparément
        if exec_type_to_use == ExecutorType.ASYNCIO:
            # Nécessite une boucle d'événements asyncio en cours d'exécution ou la création d'une nouvelle.
            # Ceci est une implémentation simplifiée pour les tâches asyncio.
            async def _run_async_tasks():
                async_tasks_to_run = [
                    self._execute_single_async_task_with_retry(cast(Task[Any, Coroutine[Any, Any, Any]], t)) 
                    for t in tasks_list
                ]
                # Utiliser tqdm pour asyncio si disponible et souhaité
                if show_progress and hasattr(tqdm, 'asyncio'):
                    return [await f for f in tqdm.asyncio.tqdm.as_completed(async_tasks_to_run, total=len(async_tasks_to_run), desc=description or "Async Tasks")]
                else:
                    return await asyncio.gather(*async_tasks_to_run, return_exceptions=False) # gather gère les exceptions si return_exceptions=False (elles seront dans TaskResult)

            try:
                # S'assurer qu'on est dans une boucle asyncio ou en créer une temporaire
                loop = asyncio.get_event_loop_policy().get_event_loop()
                if loop.is_running():
                    # Si déjà dans une boucle (ex: Jupyter), on ne peut pas utiliser loop.run_until_complete
                    # On pourrait créer une tâche et l'attendre, ou juste exécuter directement si c'est le thread principal.
                    # Pour la simplicité, on suppose que si c'est appelé depuis un contexte async, c'est géré.
                    # Si appelé depuis un contexte sync, asyncio.run() est plus simple.
                    # Cette partie est complexe à rendre universelle.
                    # Pour un appel synchrone à des tâches async, asyncio.run est le plus simple.
                    results = asyncio.run(_run_async_tasks())
                else:
                    results = loop.run_until_complete(_run_async_tasks())
            except RuntimeError as e_async_loop: # Ex: si pas de boucle courante et run() est appelé depuis une coroutine
                logger.error(f"{log_prefix_exec} Erreur de boucle asyncio: {e_async_loop}. "
                             "Assurez-vous d'appeler depuis un contexte approprié ou utilisez asyncio.run().")
                results = [TaskResult(task_id=t.task_id, success=False, exception=e_async_loop) for t in tasks_list]
            
            logger.info(f"{log_prefix_exec} Exécution asyncio terminée pour {len(tasks_list)} tâche(s).")
            return results


        # Pour ThreadPoolExecutor et ProcessPoolExecutor
        ExecutorClass = concurrent.futures.ThreadPoolExecutor \
            if exec_type_to_use == ExecutorType.THREAD \
            else concurrent.futures.ProcessPoolExecutor

        with ExecutorClass(max_workers=num_workers) as executor:
            self._current_executor = executor # Pour le graceful shutdown via signal
            futures_map: Dict[concurrent.futures.Future[TaskResult[Any]], Task[Any, Any]] = {
                executor.submit(self._execute_single_task_with_retry, task_item): task_item
                for task_item in tasks_list
            }
            
            iterable_futures = concurrent.futures.as_completed(futures_map)
            if show_progress:
                iterable_futures = tqdm(iterable_futures, total=len(tasks_list), desc=description or "Processing Tasks")

            for future in iterable_futures:
                original_task = futures_map[future]
                try:
                    result_obj = future.result(timeout=self.shutdown_timeout_s) # Récupérer le TaskResult
                    results.append(result_obj)
                except concurrent.futures.TimeoutError:
                    logger.error(f"{self.log_prefix}[Task:{original_task.task_id}] Timeout lors de l'attente du résultat.")
                    results.append(TaskResult(task_id=original_task.task_id, success=False, exception=TimeoutError("Timeout d'attente du résultat")))
                except Exception as exc: # Erreur non gérée dans _execute_single_task_with_retry (ne devrait pas arriver)
                    logger.error(f"{self.log_prefix}[Task:{original_task.task_id}] Erreur inattendue lors de la récupération du résultat : {exc}", exc_info=True)
                    import traceback
                    results.append(TaskResult(task_id=original_task.task_id, success=False, exception=exc, traceback_str=traceback.format_exc()))
            
            self._current_executor = None # Réinitialiser après usage

        logger.info(f"{log_prefix_exec} Exécution parallèle terminée pour {len(tasks_list)} tâche(s).")
        return results

    def execute_with_progress(
        self,
        tasks: Iterable[Task[Any, Any]],
        executor_type: Optional[ExecutorType] = None,
        max_workers: Optional[int] = None,
        description: Optional[str] = "Tasks"
    ) -> List[TaskResult[Any]]:
        """Exécute des tâches en parallèle avec une barre de progression tqdm."""
        return self.execute_parallel(
            tasks,
            executor_type=executor_type,
            max_workers=max_workers,
            description=description,
            show_progress=True
        )

    def adaptive_batch_size(self,
                            num_total_items: int,
                            task_complexity_score: float, # Ex: 1.0 (simple) à 10.0 (complexe)
                            available_memory_mb: Optional[float] = None,
                            max_batch_size: int = 1000,
                            min_batch_size: int = 10
                           ) -> int:
        """
        Calcule une taille de batch adaptative (conceptuel).
        Une implémentation réelle nécessiterait un profilage ou un historique.
        """
        log_prefix_adaptive = f"{self.log_prefix}[AdaptiveBatchSize]"
        if available_memory_mb is None:
            try:
                available_memory_mb = psutil.virtual_memory().available / BYTES_IN_MB
            except Exception:
                available_memory_mb = 2048 # Fallback à 2GB si psutil échoue

        # Logique très simplifiée :
        # Plus la complexité est élevée, plus le batch est petit.
        # Plus la mémoire est disponible, plus le batch peut être grand.
        
        # Facteur basé sur la complexité (inverse)
        complexity_factor = 1.0 / max(1.0, task_complexity_score) # Entre 0.1 et 1.0
        
        # Facteur basé sur la mémoire (supposons qu'une tâche complexe utilise ~50MB, simple ~5MB)
        # Et qu'on ne veut pas utiliser plus de 50% de la mémoire dispo pour les batches.
        estimated_mem_per_item_mb = (task_complexity_score / 10.0) * 45 + 5 # Echelle de 5MB à 50MB
        
        # Nombre max d'items basé sur la mémoire (très approximatif)
        if estimated_mem_per_item_mb > 0:
            max_items_by_memory = (available_memory_mb * 0.5) / estimated_mem_per_item_mb
        else:
            max_items_by_memory = max_batch_size

        # Taille de batch suggérée
        suggested_size = int(max_items_by_memory * complexity_factor * 0.5) # 0.5 est un facteur d'ajustement
        
        # Clamper entre min et max
        final_batch_size = max(min_batch_size, min(suggested_size, max_batch_size))
        # S'assurer que ce n'est pas plus grand que le nombre total d'items
        final_batch_size = min(final_batch_size, num_total_items)
        
        if final_batch_size <= 0 and num_total_items > 0 : final_batch_size = 1 # Au moins 1 si des items existent

        logger.info(f"{log_prefix_adaptive} Calcul taille batch: TotalItems={num_total_items}, "
                    f"Complexité={task_complexity_score:.1f}, MemDispoMB={available_memory_mb:.0f} -> BatchSize={final_batch_size}")
        return final_batch_size


    def _monitor_resources_loop(self, interval_s: float, stats_queue: queue.Queue) -> None:
        """Boucle pour le thread de monitoring des ressources."""
        process = psutil.Process(os.getpid()) # Processus Python courant
        log_prefix_monitor_loop = f"{self.log_prefix}[ResourceMonitorLoop]"
        logger.info(f"{log_prefix_monitor_loop} Démarrage du thread de monitoring des ressources (PID: {process.pid}). Intervalle: {interval_s}s.")
        
        while not self._resource_monitor_stop_event.is_set():
            try:
                cpu_system = psutil.cpu_percent(interval=None) # Utilisation CPU globale
                cpu_process = process.cpu_percent(interval=None) # Utilisation CPU de ce processus
                
                mem_system_info = psutil.virtual_memory()
                mem_system_percent = mem_system_info.percent
                
                mem_process_info = process.memory_info()
                mem_process_rss_mb = mem_process_info.rss / BYTES_IN_MB
                mem_process_vms_mb = mem_process_info.vms / BYTES_IN_MB
                
                active_threads_count = threading.active_count()

                stats = ResourceStats(
                    cpu_percent_system=cpu_system,
                    cpu_percent_process=cpu_process / psutil.cpu_count(), # Normaliser par nombre de coeurs
                    memory_percent_system=mem_system_percent,
                    memory_rss_process_mb=mem_process_rss_mb,
                    memory_vms_process_mb=mem_process_vms_mb,
                    active_threads=active_threads_count
                )
                stats_queue.put(stats) # Envoyer les stats via la queue
            except psutil.NoSuchProcess:
                logger.warning(f"{log_prefix_monitor_loop} Le processus (PID: {process.pid}) n'existe plus. Arrêt du monitoring.")
                break
            except Exception as e_monitor:
                logger.error(f"{log_prefix_monitor_loop} Erreur dans la boucle de monitoring: {e_monitor}", exc_info=True)
            
            # Attendre l'intervalle ou l'événement d'arrêt
            self._resource_monitor_stop_event.wait(timeout=interval_s)
        
        logger.info(f"{log_prefix_monitor_loop} Arrêt du thread de monitoring des ressources.")


    @contextlib.contextmanager
    def monitor_resource_usage(self, interval_s: float = 5.0) -> Callable[[], Optional[ResourceStats]]:
        """
        Context manager pour monitorer l'utilisation des ressources pendant son exécution.
        Retourne une fonction qui, appelée, donne les dernières statistiques collectées.
        """
        log_prefix_ctx_monitor = f"{self.log_prefix}[ResourceUsageCtxManager]"
        stats_q: queue.Queue[ResourceStats] = queue.Queue()
        latest_stats_lock = threading.Lock()
        _latest_stats_value: Optional[ResourceStats] = None # Stocker la dernière stat

        def get_latest_stats() -> Optional[ResourceStats]:
            nonlocal _latest_stats_value
            # Vider la queue et prendre la dernière valeur
            last_val_from_q = None
            while not stats_q.empty():
                try:
                    last_val_from_q = stats_q.get_nowait()
                except queue.Empty:
                    break # Devrait pas arriver avec not empty() mais par sécurité
            
            with latest_stats_lock:
                if last_val_from_q is not None:
                    _latest_stats_value = last_val_from_q
                return _latest_stats_value

        self._resource_monitor_stop_event.clear()
        monitor_thread = threading.Thread(
            target=self._monitor_resources_loop,
            args=(interval_s, stats_q),
            daemon=True, # Le thread s'arrêtera si le programme principal se termine
            name="ResourceMonitorThread"
        )
        
        try:
            logger.info(f"{log_prefix_ctx_monitor} Démarrage du monitoring des ressources...")
            monitor_thread.start()
            yield get_latest_stats # Retourner la fonction pour récupérer les stats
        finally:
            logger.info(f"{log_prefix_ctx_monitor} Arrêt du monitoring des ressources...")
            self._resource_monitor_stop_event.set()
            monitor_thread.join(timeout=interval_s + 1) # Attendre que le thread se termine
            if monitor_thread.is_alive():
                logger.warning(f"{log_prefix_ctx_monitor} Le thread de monitoring ne s'est pas arrêté proprement.")
            final_stats_summary = get_latest_stats() # Obtenir les toutes dernières stats
            logger.info(f"{log_prefix_ctx_monitor} Monitoring terminé. Dernières stats (si dispo): CPU Sys {final_stats_summary.cpu_percent_system if final_stats_summary else 'N/A'}%, "
                        f"Mem Proc RSS {final_stats_summary.memory_rss_process_mb if final_stats_summary else 'N/A'}MB")

    def shutdown(self, wait: bool = True) -> None:
        """Arrête proprement l'exécuteur (si un pool est actif)."""
        # Cette méthode est plus pour un exécuteur persistant.
        # Pour l'instant, les exécuteurs sont créés et détruits avec le `with` statement.
        # Si on avait un _current_executor persistant, on l'arrêterait ici.
        logger.info(f"{self.log_prefix} Demande d'arrêt de l'exécuteur (non implémenté pour les exécuteurs contextuels).")
        if self._current_executor:
            logger.warning(f"{self.log_prefix} Tentative d'arrêt d'un _current_executor, mais il devrait être géré par son contexte 'with'.")
            try:
                self._current_executor.shutdown(wait=wait, cancel_futures=True if sys.version_info >= (3,9) else True) # type: ignore
            except Exception as e_shutdown:
                 logger.error(f"{self.log_prefix} Erreur lors de l'arrêt manuel de _current_executor: {e_shutdown}")
            self._current_executor = None

# --- Fonctions utilitaires globales (si nécessaire en dehors de la classe) ---
# (Exemple: une fonction pour créer des objets Task plus facilement)
def create_task(func: Callable[..., R], *args: Any, **kwargs: Any) -> Task[Any, R]:
    """Crée un objet Task."""
    # On pourrait ajouter de la logique ici, comme la détection du type de func (async ou non)
    # pour définir un executor_preference.
    return Task(func=func, args=args, kwargs=kwargs)

# Nécessaire pour Task dataclass si uuid n'est pas importé globalement dans ce fichier
import uuid
from datetime import datetime, timezone # Pour ResourceStats

