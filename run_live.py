import argparse
import logging
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any # Union retiré car non utilisé directement ici
from datetime import datetime, timezone, timedelta # timedelta ajouté
import signal
import json # Pour charger le live_config.json spécifique au déploiement

# Détection de la racine du projet
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    # print(f"Added {str(SRC_PATH)} to sys.path for module imports.")

# Configuration initiale du logging (sera surchargée par load_all_configs)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger spécifique pour ce script

try:
    from src.config.loader import load_all_configs, AppConfig
    from src.live.manager import LiveTradingManager
    from src.config.definitions import StrategyDeployment, AccountConfig # Pour le typage
    # setup_logging est appelé à l'intérieur de load_all_configs
except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE: Impossible d'importer les modules nécessaires: {e}. "
                    f"Vérifiez PYTHONPATH et les installations. CWD: {Path.cwd()}, sys.path: {sys.path}", exc_info=True)
    sys.exit(1)
except Exception as e_imp:
    logger.critical(f"ÉCHEC CRITIQUE: Erreur inattendue lors des imports initiaux: {e_imp}", exc_info=True)
    sys.exit(1)


# Variables Globales pour la Gestion des Threads/Managers
active_managers: Dict[str, LiveTradingManager] = {} # Clé: manager_unique_id, Valeur: instance
active_threads: Dict[str, threading.Thread] = {}   # Clé: manager_unique_id, Valeur: thread
global_shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Gère SIGINT (Ctrl+C) et SIGTERM pour un arrêt propre."""
    logger.info(f"Signal {signal.Signals(signum).name if hasattr(signal, 'Signals') else signum} reçu. Demande d'arrêt global de l'orchestrateur...")
    global_shutdown_event.set()

def _run_single_trading_session_cycle(
        app_config_initial: AppConfig, # Passer la config initiale pour le rechargement
        cli_args: argparse.Namespace,
        orchestrator_shutdown_event: threading.Event
    ):
    """
    Gère un cycle de démarrage/vérification des LiveTradingManager.
    Peut être appelée périodiquement par main_loop_orchestrator.
    """
    global active_managers, active_threads # S'assurer qu'on utilise les globales
    cycle_log_prefix = "[OrchestratorCycle]"
    logger.info(f"{cycle_log_prefix} --- Démarrage du nouveau cycle de session de trading à {datetime.now(timezone.utc).isoformat()} ---")

    # 1. Rechargement de Configuration (Optionnel mais recommandé)
    app_config_reloaded: AppConfig
    try:
        current_project_root = cli_args.root or str(PROJECT_ROOT) # Utiliser la racine CLI ou détectée
        app_config_reloaded = load_all_configs(project_root=current_project_root)
        # Le logging est reconfiguré par load_all_configs si la config de log a changé.
        logger.info(f"{cycle_log_prefix} Configuration de l'application (re)chargée avec succès.")
    except Exception as e_conf_reload:
        logger.critical(f"{cycle_log_prefix} ERREUR CRITIQUE lors du rechargement de la configuration: {e_conf_reload}. "
                        "L'orchestrateur pourrait s'arrêter ou continuer avec l'ancienne configuration.", exc_info=True)
        # Décider si on arrête tout ou si on continue avec app_config_initial
        # Pour la robustesse, on pourrait continuer avec app_config_initial si le rechargement échoue après le premier cycle.
        # Pour ce cycle, si le rechargement échoue, on utilise la config passée.
        app_config_reloaded = app_config_initial # Fallback à la config du cycle précédent ou initial
        # orchestrator_shutdown_event.set() # Optionnel: arrêter si le rechargement est critique
        # return


    # 2. Nettoyage des Threads Inactifs
    inactive_manager_ids = [mid for mid, th in list(active_threads.items()) if not th.is_alive()] # list() pour éviter RuntimeError si dict change
    if inactive_manager_ids:
        logger.info(f"{cycle_log_prefix} Nettoyage de {len(inactive_manager_ids)} thread(s) de manager inactif(s)...")
        for manager_id in inactive_manager_ids:
            logger.warning(f"{cycle_log_prefix} Le thread du Manager '{manager_id}' n'est plus actif.")
            if manager_id in active_managers:
                try:
                    # L'appel à stop_trading est pour s'assurer que le manager sait qu'il doit s'arrêter
                    # même si son thread s'est terminé de manière inattendue.
                    active_managers[manager_id].stop_trading()
                except Exception as e_stop_cleanup:
                    logger.error(f"{cycle_log_prefix} Erreur lors de l'appel à stop_trading pour le manager inactif '{manager_id}': {e_stop_cleanup}")
                del active_managers[manager_id]
            if manager_id in active_threads:
                del active_threads[manager_id]
        logger.info(f"{cycle_log_prefix} Nettoyage des managers inactifs terminé.")


    # 3. Détermination des Instances à Lancer/Gérer
    # Utiliser les paires spécifiées en CLI, sinon celles de live_fetch config
    pairs_from_cli = [p.upper() for p in cli_args.pair] if cli_args.pair else None
    pairs_to_process_this_cycle = pairs_from_cli if pairs_from_cli else app_config_reloaded.live_config.live_fetch.crypto_pairs
    
    if not pairs_to_process_this_cycle:
        logger.warning(f"{cycle_log_prefix} Aucune paire de trading spécifiée (CLI ou config). Cycle de session inactif.")
        return

    logger.info(f"{cycle_log_prefix} Paires à traiter pour ce cycle: {pairs_to_process_this_cycle}")

    managers_started_or_found_active_this_cycle = 0
    for account_cfg in app_config_reloaded.accounts_config:
        if orchestrator_shutdown_event.is_set(): break # Vérifier avant de traiter chaque compte
        
        logger.debug(f"{cycle_log_prefix} Évaluation du compte: {account_cfg.account_alias}")
        for deployment_cfg in app_config_reloaded.live_config.strategy_deployments:
            if orchestrator_shutdown_event.is_set(): break

            if not deployment_cfg.active:
                logger.debug(f"{cycle_log_prefix} Déploiement '{deployment_cfg.strategy_id}' inactif. Ignoré.")
                continue

            if deployment_cfg.account_alias_to_use != account_cfg.account_alias:
                continue # Ce déploiement n'est pas pour ce compte

            # Charger le fichier live_config.json spécifique au déploiement pour obtenir la paire et le contexte des paramètres
            try:
                param_config_path = Path(app_config_reloaded.project_root) / deployment_cfg.results_config_path
                if not param_config_path.is_file():
                    logger.error(f"{cycle_log_prefix} Fichier de paramètres live '{param_config_path}' pour déploiement '{deployment_cfg.strategy_id}' non trouvé. Ignoré.")
                    continue
                with open(param_config_path, 'r', encoding='utf-8') as f_params:
                    specific_live_params_json = json.load(f_params)
                
                pair_for_manager = specific_live_params_json.get("pair_symbol")
                context_for_manager_params = specific_live_params_json.get("timeframe_context") # Ce contexte est lié aux paramètres optimisés

                if not pair_for_manager or not context_for_manager_params:
                    logger.error(f"{cycle_log_prefix} 'pair_symbol' ou 'timeframe_context' manquant dans '{param_config_path}' pour déploiement '{deployment_cfg.strategy_id}'. Ignoré.")
                    continue
                
                pair_for_manager = pair_for_manager.upper()

            except Exception as e_load_spec_cfg:
                logger.error(f"{cycle_log_prefix} Erreur lors du chargement du fichier de paramètres live '{param_config_path}' pour déploiement '{deployment_cfg.strategy_id}': {e_load_spec_cfg}. Ignoré.", exc_info=True)
                continue

            # Filtrer par paire CLI si spécifié
            if pairs_from_cli and pair_for_manager not in pairs_from_cli:
                logger.debug(f"{cycle_log_prefix} Déploiement '{deployment_cfg.strategy_id}' pour paire '{pair_for_manager}' non inclus dans les paires CLI. Ignoré.")
                continue
            
            # Le context_label pour le LiveTradingManager doit être celui qui correspond aux paramètres chargés.
            # Le cli_args.tf peut être utilisé pour un filtrage plus large ou un logging de l'orchestrateur,
            # mais le manager a besoin du contexte de ses paramètres.
            context_label_for_manager_instance = context_for_manager_params
            
            # Si un contexte CLI est fourni ET qu'il est différent du contexte des paramètres,
            # on pourrait choisir d'ignorer ce déploiement ou de loguer un avertissement.
            # Pour l'instant, on priorise le contexte des paramètres pour le manager.
            if cli_args.tf and cli_args.tf != context_label_for_manager_instance:
                logger.warning(f"{cycle_log_prefix} Contexte CLI '{cli_args.tf}' fourni, mais le déploiement '{deployment_cfg.strategy_id}' "
                               f"utilise des paramètres pour le contexte '{context_label_for_manager_instance}'. Le manager utilisera '{context_label_for_manager_instance}'.")


            # ID unique pour le manager: strategy_id du déploiement + alias du compte.
            # Le context_label est déjà implicitement dans le strategy_id ou son results_config_path.
            manager_unique_id = f"{deployment_cfg.strategy_id}@@{account_cfg.account_alias}"
            
            if manager_unique_id in active_threads and active_threads[manager_unique_id].is_alive():
                logger.debug(f"{cycle_log_prefix} Manager '{manager_unique_id}' déjà actif et vivant. Ignoré.")
                managers_started_or_found_active_this_cycle += 1
                continue
            elif manager_unique_id in active_threads and not active_threads[manager_unique_id].is_alive():
                logger.warning(f"{cycle_log_prefix} Thread du Manager '{manager_unique_id}' trouvé inactif. Tentative de redémarrage.")
                if manager_unique_id in active_managers: del active_managers[manager_unique_id]
                if manager_unique_id in active_threads: del active_threads[manager_unique_id]
            
            logger.info(f"{cycle_log_prefix} --- Initialisation du Manager pour déploiement: '{deployment_cfg.strategy_id}', "
                        f"Compte: '{account_cfg.account_alias}', Paire: '{pair_for_manager}', Contexte Params: '{context_label_for_manager_instance}' ---")
            try:
                manager = LiveTradingManager(
                    app_config=app_config_reloaded, # Utiliser la config (re)chargée
                    strategy_deployment_config=deployment_cfg,
                    account_config=account_cfg,
                    pair_to_trade=pair_for_manager,
                    context_label_from_deployment=context_label_for_manager_instance
                )
                
                thread = threading.Thread(target=manager.run, name=manager_unique_id, daemon=True) # daemon=True pour que les threads s'arrêtent si le main s'arrête brusquement
                active_managers[manager_unique_id] = manager
                active_threads[manager_unique_id] = thread
                thread.start()
                logger.info(f"{cycle_log_prefix} Thread LiveTradingManager démarré pour '{manager_unique_id}'")
                managers_started_or_found_active_this_cycle += 1
            except Exception as e_mgr_init:
                logger.critical(f"{cycle_log_prefix} ÉCHEC CRITIQUE lors de l'initialisation de LiveTradingManager pour '{manager_unique_id}': {e_mgr_init}", exc_info=True)
                # Ne pas ajouter aux dictionnaires actifs si l'initialisation échoue

        if orchestrator_shutdown_event.is_set(): break # Sortir de la boucle des comptes

    if managers_started_or_found_active_this_cycle == 0 and \
       any(d.active for d in app_config_reloaded.live_config.strategy_deployments):
        logger.warning(f"{cycle_log_prefix} Aucun manager n'a pu être démarré ou trouvé actif ce cycle, "
                       "bien que des déploiements actifs soient configurés. Vérifiez la correspondance des paires/comptes.")
    elif managers_started_or_found_active_this_cycle > 0:
        logger.info(f"{cycle_log_prefix} {managers_started_or_found_active_this_cycle} thread(s) de manager sont actifs ou ont été démarrés ce cycle.")
    else:
        logger.info(f"{cycle_log_prefix} Aucun déploiement actif trouvé ou aucune correspondance avec les arguments CLI. Orchestrateur inactif pour ce cycle.")


def main_loop_orchestrator(args: argparse.Namespace):
    """Boucle principale de l'orchestrateur pour gérer les sessions de trading."""
    global global_shutdown_event, active_managers, active_threads
    
    app_config_initial: AppConfig
    try:
        current_project_root = args.root if args.root else str(PROJECT_ROOT)
        app_config_initial = load_all_configs(project_root=current_project_root)
        # Le logging global est configuré par load_all_configs.
        logger.info("[Orchestrator] Configuration initiale chargée et logging configuré.")
    except Exception as e_init_conf:
        logger.critical(f"[Orchestrator] ERREUR CRITIQUE lors du chargement de la configuration initiale: {e_init_conf}. L'orchestrateur ne peut pas démarrer.", exc_info=True)
        return

    session_cycle_interval_seconds = getattr(app_config_initial.live_config.global_live_settings, 'session_cycle_interval_seconds', 60)
    if not isinstance(session_cycle_interval_seconds, (int, float)) or session_cycle_interval_seconds <= 5: # Minimum 5s
        logger.warning(f"[Orchestrator] session_cycle_interval_seconds invalide ({session_cycle_interval_seconds}). Utilisation de 60 secondes.")
        session_cycle_interval_seconds = 60
    logger.info(f"[Orchestrator] Intervalle du cycle de session: {session_cycle_interval_seconds} secondes.")

    try:
        while not global_shutdown_event.is_set():
            _run_single_trading_session_cycle(app_config_initial, args, global_shutdown_event)
            
            if global_shutdown_event.is_set():
                logger.info("[Orchestrator] Événement d'arrêt reçu pendant le cycle. Sortie de la boucle principale.")
                break
            
            logger.info(f"[Orchestrator] Prochain cycle de session de trading dans {session_cycle_interval_seconds} secondes (sauf arrêt demandé)...")
            # Attendre l'intervalle ou jusqu'à ce que l'événement d'arrêt soit signalé
            shutdown_during_wait = global_shutdown_event.wait(timeout=session_cycle_interval_seconds)
            if shutdown_during_wait:
                logger.info("[Orchestrator] Événement d'arrêt reçu pendant l'attente. Sortie de la boucle principale.")
                break
    except KeyboardInterrupt: # Gérer Ctrl+C ici aussi pour l'orchestrateur lui-même
        logger.info("[Orchestrator] Interruption clavier (Ctrl+C) reçue. Demande d'arrêt global...")
        global_shutdown_event.set()
    except Exception as e_main_loop:
        logger.critical(f"[Orchestrator] Erreur non gérée dans la boucle principale de l'orchestrateur: {e_main_loop}", exc_info=True)
        global_shutdown_event.set() # Déclencher l'arrêt en cas d'erreur grave

    logger.info("[Orchestrator] Boucle principale terminée. Procédure d'arrêt des managers actifs...")
    active_manager_ids_at_shutdown = list(active_managers.keys())
    if not active_manager_ids_at_shutdown:
        logger.info("[Orchestrator] Aucun manager actif à arrêter.")
    else:
        logger.info(f"[Orchestrator] Arrêt de {len(active_manager_ids_at_shutdown)} manager(s) actif(s)...")

    for manager_id in active_manager_ids_at_shutdown:
        manager = active_managers.get(manager_id)
        thread = active_threads.get(manager_id)
        if manager:
            logger.info(f"[Orchestrator] Demande d'arrêt pour le manager: {manager_id}")
            try:
                manager.stop_trading() # Signale l'arrêt au manager
            except Exception as e_stop_mgr:
                logger.error(f"[Orchestrator] Erreur lors de la demande d'arrêt au manager {manager_id}: {e_stop_mgr}")
        
        if thread and thread.is_alive():
            logger.info(f"[Orchestrator] Attente de la fin du thread du manager {manager_id} (timeout 30s)...")
            thread.join(timeout=30)
            if thread.is_alive():
                logger.warning(f"[Orchestrator] Le thread du Manager {manager_id} ne s'est pas terminé dans le délai imparti.")
            else:
                logger.info(f"[Orchestrator] Le thread du Manager {manager_id} s'est terminé.")
        
        if manager_id in active_managers: del active_managers[manager_id]
        if manager_id in active_threads: del active_threads[manager_id]

    logger.info("[Orchestrator] --- Tous les managers actifs ont été priés de s'arrêter. Arrêt de l'orchestrateur terminé. ---")


def main():
    """Point d'entrée principal pour lancer l'orchestrateur de trading en direct."""
    run_start_time = time.time()
    # Le logging sera configuré par load_all_configs dans main_loop_orchestrator

    parser = argparse.ArgumentParser(description="Orchestrateur du Bot de Trading Live.")
    parser.add_argument(
        "--pair",
        type=str,
        action='append', # Permet --pair BTCUSDT --pair ETHUSDT
        help="Paire(s) de trading spécifique(s) à traiter (ex: BTCUSDT). Si non fourni, utilise les paires de config_live.json."
    )
    parser.add_argument(
        "--tf", "--context", dest="tf",
        type=str,
        help="Label de contexte global pour ce run de l'orchestrateur (peut influencer la sélection des déploiements ou le logging). "
             "Note: Chaque LiveTradingManager utilisera le contexte spécifié dans ses paramètres de déploiement pour charger sa config."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Répertoire racine du projet si le script n'est pas exécuté depuis la racine du projet.",
    )
    args = parser.parse_args()

    # Configurer les gestionnaires de signaux globaux pour un arrêt propre
    # Note: signal.SIGINT est Ctrl+C. signal.SIGTERM est un signal d'arrêt plus général.
    # Sur Windows, SIGTERM n'est pas toujours disponible ou gérable de la même manière.
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, signal_handler)
        except OSError: # Peut arriver sur Windows
             logger.warning("Impossible de configurer le gestionnaire pour SIGTERM sur cette plateforme.")
    logger.info("[Main] Gestionnaires de signaux globaux configurés (SIGINT, SIGTERM si disponible).")

    logger.info("--- Démarrage du Script run_live.py (Orchestrateur) ---")
    
    try:
        main_loop_orchestrator(args)
    except SystemExit: # Attraper sys.exit() pour un logging final propre
        logger.info("[Main] Sortie de l'application demandée par sys.exit().")
    except Exception as e_main:
        logger.critical(f"[Main] ERREUR CRITIQUE non gérée dans l'exécution principale de run_live.py: {e_main}", exc_info=True)
    finally:
        run_end_time = time.time()
        total_duration_seconds = run_end_time - run_start_time
        logger.info(f"--- Script run_live.py Terminé --- Temps d'Exécution Total: {total_duration_seconds:.2f} secondes ({timedelta(seconds=total_duration_seconds)})")
        logging.shutdown() # S'assurer que tous les logs sont écrits

if __name__ == "__main__":
    main()
