# run_live.py
"""
Orchestrateur principal pour le trading en direct.
Ce script gère le démarrage, l'arrêt et la supervision des instances de
LiveTradingManager pour les différentes stratégies et paires de trading actives,
en se basant sur la configuration de déploiement.
"""
import argparse
import logging
import sys
import time
import threading
from pathlib import Path
import signal
import json # Pour charger le live_config.json spécifique au déploiement
from typing import Dict, List, Optional, Any, cast
from datetime import datetime, timezone, timedelta

# --- Configuration initiale du logging et du PYTHONPATH ---
# Sera reconfiguré par load_all_configs, mais utile pour les erreurs précoces.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__) # Logger spécifique pour ce script

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    logger.debug(f"Ajouté {SRC_PATH} au PYTHONPATH par run_live.py")

# --- Imports des modules de l'application ---
try:
    from src.config.loader import load_all_configs, AppConfig
    from src.live.manager import LiveTradingManager
    from src.config.definitions import StrategyDeployment, AccountConfig # Pour le typage
    # setup_logging est appelé à l'intérieur de load_all_configs
except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE (run_live.py): Impossible d'importer les modules nécessaires: {e}. "
                    f"Vérifiez PYTHONPATH et les installations. CWD: {Path.cwd()}, sys.path: {sys.path}", exc_info=True)
    sys.exit(1)
except Exception as e_imp: # pylint: disable=broad-except
    logger.critical(f"ÉCHEC CRITIQUE (run_live.py): Erreur inattendue lors des imports initiaux: {e_imp}", exc_info=True)
    sys.exit(1)


# --- Variables Globales pour la Gestion des Threads/Managers ---
active_managers: Dict[str, LiveTradingManager] = {} # Clé: manager_unique_id, Valeur: instance
active_threads: Dict[str, threading.Thread] = {}   # Clé: manager_unique_id, Valeur: thread
global_shutdown_event = threading.Event() # Événement global pour signaler l'arrêt

def signal_handler(signum, frame):
    """
    Gère les signaux SIGINT (Ctrl+C) et SIGTERM pour un arrêt propre de l'orchestrateur.
    Positionne l'événement global d'arrêt.
    """
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') and isinstance(signum, int) else str(signum)
    logger.info(f"Signal {signal_name} reçu. Demande d'arrêt global de l'orchestrateur...")
    global_shutdown_event.set()

def _run_single_trading_session_cycle(
        app_config_current_cycle: AppConfig,
        cli_args: argparse.Namespace,
        orchestrator_shutdown_event: threading.Event
    ) -> AppConfig:
    """
    Gère un cycle de démarrage, de vérification et de nettoyage des LiveTradingManagers.
    Cette fonction est appelée périodiquement par la boucle principale de l'orchestrateur.

    Args:
        app_config_current_cycle (AppConfig): L'instance AppConfig à utiliser pour ce cycle
                                              (peut être fraîchement rechargée ou celle du cycle précédent).
        cli_args (argparse.Namespace): Arguments de la ligne de commande passés à l'orchestrateur.
        orchestrator_shutdown_event (threading.Event): L'événement global d'arrêt.

    Returns:
        AppConfig: L'instance AppConfig qui a été utilisée ou rechargée durant ce cycle.
                   Permet de propager une configuration rechargée au cycle suivant.
    """
    global active_managers, active_threads # Utiliser les variables globales
    cycle_log_prefix = "[OrchestratorCycle]"
    logger.info(f"{cycle_log_prefix} --- Démarrage du cycle de session de trading à {datetime.now(timezone.utc).isoformat()} ---")

    # 1. Rechargement de Configuration (Optionnel mais recommandé pour la flexibilité)
    # Si le rechargement échoue, on continue avec la configuration du cycle précédent.
    app_config_for_this_cycle = app_config_current_cycle
    try:
        # Le chemin racine du projet est déterminé une fois au démarrage de l'orchestrateur.
        # Si la config change, load_all_configs reconfigure aussi le logging.
        # Note: Si des chemins absolus sont dans la config (après un premier load_all_configs),
        # le rechargement devrait bien fonctionner.
        reloaded_config = load_all_configs(project_root=app_config_current_cycle.project_root)
        app_config_for_this_cycle = reloaded_config
        logger.info(f"{cycle_log_prefix} Configuration de l'application (re)chargée avec succès pour ce cycle.")
    except Exception as e_conf_reload: # pylint: disable=broad-except
        logger.error(f"{cycle_log_prefix} ERREUR lors du rechargement de la configuration: {e_conf_reload}. "
                        "Utilisation de la configuration du cycle précédent.", exc_info=True)
        # En cas d'échec de rechargement, on continue avec app_config_current_cycle.

    # 2. Nettoyage des Threads de Managers Inactifs
    # Utiliser list() pour créer une copie des clés avant de modifier le dictionnaire
    inactive_manager_ids = [manager_id for manager_id, thread_instance in list(active_threads.items()) if not thread_instance.is_alive()]
    if inactive_manager_ids:
        logger.info(f"{cycle_log_prefix} Nettoyage de {len(inactive_manager_ids)} thread(s) de manager inactif(s)...")
        for manager_id in inactive_manager_ids:
            logger.warning(f"{cycle_log_prefix} Le thread du Manager '{manager_id}' n'est plus actif (terminé ou planté).")
            if manager_id in active_managers:
                try:
                    # Tenter d'appeler stop_trading pour un nettoyage interne du manager,
                    # même si le thread est déjà mort (au cas où des ressources seraient à libérer).
                    active_managers[manager_id].stop_trading()
                except Exception as e_stop_cleanup: # pylint: disable=broad-except
                    logger.error(f"{cycle_log_prefix} Erreur lors de l'appel à stop_trading pour le manager inactif '{manager_id}': {e_stop_cleanup}")
                del active_managers[manager_id]
            if manager_id in active_threads:
                del active_threads[manager_id]
        logger.info(f"{cycle_log_prefix} Nettoyage des managers inactifs terminé.")

    # 3. Détermination des Instances de LiveTradingManager à Lancer ou Gérer
    # Utiliser les paires spécifiées en CLI (si fournies), sinon toutes les paires des déploiements actifs.
    cli_target_pairs_upper: Optional[List[str]] = [p.upper() for p in cli_args.pair] if cli_args.pair else None
    
    if not app_config_for_this_cycle.live_config.strategy_deployments:
        logger.info(f"{cycle_log_prefix} Aucun déploiement de stratégie configuré dans config_live.json. Cycle inactif.")
        return app_config_for_this_cycle

    managers_considered_this_cycle = 0
    for account_cfg_instance in app_config_for_this_cycle.accounts_config:
        if orchestrator_shutdown_event.is_set(): break # Vérifier avant de traiter chaque compte
        
        logger.debug(f"{cycle_log_prefix} Évaluation du compte : {account_cfg_instance.account_alias}")
        for deployment_cfg_instance in app_config_for_this_cycle.live_config.strategy_deployments:
            if orchestrator_shutdown_event.is_set(): break

            if not deployment_cfg_instance.active:
                logger.debug(f"{cycle_log_prefix} Déploiement '{deployment_cfg_instance.strategy_id}' inactif. Ignoré.")
                continue

            if deployment_cfg_instance.account_alias_to_use != account_cfg_instance.account_alias:
                continue # Ce déploiement n'est pas pour ce compte en cours d'itération

            managers_considered_this_cycle +=1
            # Charger le fichier live_config.json spécifique au déploiement pour obtenir la paire et le contexte des paramètres
            try:
                if not app_config_for_this_cycle.project_root:
                     raise ValueError("project_root non défini dans AppConfig, impossible de résoudre results_config_path.")
                
                params_config_file_path = Path(app_config_for_this_cycle.project_root) / deployment_cfg_instance.results_config_path
                if not params_config_file_path.is_file():
                    logger.error(f"{cycle_log_prefix} Fichier de paramètres live '{params_config_file_path}' pour déploiement "
                                 f"'{deployment_cfg_instance.strategy_id}' non trouvé. Déploiement ignoré.")
                    continue
                with open(params_config_file_path, 'r', encoding='utf-8') as f_params_json:
                    deployment_specific_params_json = json.load(f_params_json)
                
                pair_symbol_for_manager = deployment_specific_params_json.get("pair_symbol")
                # Le "context_label" dans ce JSON est le contexte sous lequel les paramètres ont été optimisés.
                context_label_for_manager_parameters = deployment_specific_params_json.get("context_label") # ou "timeframe_context"

                if not pair_symbol_for_manager or not context_label_for_manager_parameters:
                    logger.error(f"{cycle_log_prefix} 'pair_symbol' ou 'context_label' (ou 'timeframe_context') manquant dans "
                                 f"'{params_config_file_path}' pour déploiement '{deployment_cfg_instance.strategy_id}'. Ignoré.")
                    continue
                
                pair_symbol_for_manager = pair_symbol_for_manager.upper()

            except Exception as e_load_deploy_json: # pylint: disable=broad-except
                logger.error(f"{cycle_log_prefix} Erreur lors du chargement du fichier de paramètres live '{params_config_file_path}' "
                             f"pour déploiement '{deployment_cfg_instance.strategy_id}': {e_load_deploy_json}. Ignoré.", exc_info=True)
                continue

            # Filtrer par paire(s) spécifiée(s) en ligne de commande, si applicable
            if cli_target_pairs_upper and pair_symbol_for_manager not in cli_target_pairs_upper:
                logger.debug(f"{cycle_log_prefix} Déploiement '{deployment_cfg_instance.strategy_id}' pour paire '{pair_symbol_for_manager}' "
                               "non inclus dans les paires CLI spécifiées. Ignoré.")
                continue
            
            # Filtrer par contexte CLI global si spécifié (optionnel, car le contexte des params est prioritaire pour le manager)
            if cli_args.tf and cli_args.tf != context_label_for_manager_parameters:
                logger.info(f"{cycle_log_prefix} Contexte CLI global '{cli_args.tf}' fourni, mais le déploiement "
                              f"'{deployment_cfg_instance.strategy_id}' (paire {pair_symbol_for_manager}) utilise des paramètres optimisés "
                              f"pour le contexte '{context_label_for_manager_parameters}'. Ce déploiement sera quand même traité "
                              f"avec son contexte de paramètres '{context_label_for_manager_parameters}'.")
                # Aucune action de filtrage ici, juste un log. Le manager doit utiliser son contexte de paramètres.

            # ID unique pour le manager : strategy_id du déploiement + alias du compte.
            # Le contexte est implicite dans le strategy_id ou son results_config_path.
            manager_unique_id = f"{deployment_cfg_instance.strategy_id}@@{account_cfg_instance.account_alias}"
            
            if manager_unique_id in active_threads and active_threads[manager_unique_id].is_alive():
                logger.debug(f"{cycle_log_prefix} Manager '{manager_unique_id}' déjà actif et vivant. Ignoré pour ce cycle de démarrage.")
                continue
            elif manager_unique_id in active_threads: # Thread inactif, besoin de nettoyage et redémarrage
                logger.warning(f"{cycle_log_prefix} Thread du Manager '{manager_unique_id}' trouvé inactif. Nettoyage avant tentative de redémarrage.")
                if manager_unique_id in active_managers: del active_managers[manager_unique_id]
                del active_threads[manager_unique_id] # Retirer l'ancien thread mort
            
            logger.info(f"{cycle_log_prefix} --- Initialisation d'un nouveau LiveTradingManager ---")
            logger.info(f"{cycle_log_prefix}   Déploiement ID       : {deployment_cfg_instance.strategy_id}")
            logger.info(f"{cycle_log_prefix}   Compte Alias         : {account_cfg_instance.account_alias}")
            logger.info(f"{cycle_log_prefix}   Paire de Trading     : {pair_symbol_for_manager}")
            logger.info(f"{cycle_log_prefix}   Contexte des Params  : {context_label_for_manager_parameters}")
            logger.info(f"{cycle_log_prefix}   Manager Unique ID    : {manager_unique_id}")
            
            try:
                manager_instance = LiveTradingManager(
                    app_config=app_config_for_this_cycle, # Utiliser la config (re)chargée
                    strategy_deployment_config=deployment_cfg_instance,
                    account_config=account_cfg_instance,
                    pair_to_trade=pair_symbol_for_manager,
                    context_label_from_deployment=context_label_for_manager_parameters # Crucial: passer le contexte des params
                )
                
                # Utiliser daemon=True pour que les threads s'arrêtent si le processus principal est tué brusquement.
                # Cependant, pour un arrêt propre, on se fie à shutdown_event.
                thread_instance = threading.Thread(target=manager_instance.run, name=manager_unique_id, daemon=True)
                
                active_managers[manager_unique_id] = manager_instance
                active_threads[manager_unique_id] = thread_instance
                thread_instance.start()
                logger.info(f"{cycle_log_prefix} Thread LiveTradingManager démarré pour '{manager_unique_id}'.")
            except Exception as e_mgr_init_start: # pylint: disable=broad-except
                logger.critical(f"{cycle_log_prefix} ÉCHEC CRITIQUE lors de l'initialisation ou du démarrage de "
                                f"LiveTradingManager pour '{manager_unique_id}': {e_mgr_init_start}", exc_info=True)
                # Ne pas ajouter aux dictionnaires actifs si l'initialisation/démarrage échoue

        if orchestrator_shutdown_event.is_set(): break # Sortir de la boucle des comptes

    active_manager_count = sum(1 for th in active_threads.values() if th.is_alive())
    if managers_considered_this_cycle == 0 and any(d.active for d in app_config_for_this_cycle.live_config.strategy_deployments):
        logger.warning(f"{cycle_log_prefix} Aucun déploiement actif n'a correspondu aux filtres ou n'a pu être traité ce cycle, "
                       "bien que des déploiements actifs soient configurés.")
    elif active_manager_count > 0:
        logger.info(f"{cycle_log_prefix} {active_manager_count} thread(s) de manager sont actuellement actifs.")
    else:
        logger.info(f"{cycle_log_prefix} Aucun manager actif à la fin de ce cycle.")
        
    return app_config_for_this_cycle # Retourner la config utilisée (potentiellement rechargée)


def main_loop_orchestrator(cli_args: argparse.Namespace) -> None:
    """
    Boucle principale de l'orchestrateur pour gérer les sessions de trading en direct.
    Charge la configuration initiale, puis entre dans une boucle qui appelle
    `_run_single_trading_session_cycle` à intervalles réguliers.
    Gère l'arrêt propre de tous les managers actifs lorsque `global_shutdown_event` est signalé.

    Args:
        cli_args (argparse.Namespace): Arguments de la ligne de commande.
    """
    global global_shutdown_event, active_managers, active_threads
    
    app_config_initial_load: AppConfig
    try:
        # Déterminer la racine du projet à partir des arguments CLI ou par défaut
        current_project_root = cli_args.root if cli_args.root else str(PROJECT_ROOT)
        app_config_initial_load = load_all_configs(project_root=current_project_root)
        # Le logging global de l'application est maintenant configuré par load_all_configs.
        logger.info("[OrchestratorLoop] Configuration initiale chargée et logging configuré.")
    except Exception as e_init_conf_loop: # pylint: disable=broad-except
        logger.critical(f"[OrchestratorLoop] ERREUR CRITIQUE lors du chargement de la configuration initiale: {e_init_conf_loop}. "
                        "L'orchestrateur ne peut pas démarrer.", exc_info=True)
        return # Arrêter si la config initiale ne peut pas être chargée

    # Utiliser la configuration chargée pour l'intervalle de cycle
    session_cycle_interval_s = app_config_initial_load.live_config.global_live_settings.session_cycle_interval_seconds
    if not isinstance(session_cycle_interval_s, (int, float)) or session_cycle_interval_s <= 5:
        logger.warning(f"[OrchestratorLoop] session_cycle_interval_seconds ({session_cycle_interval_s}) invalide. "
                       "Utilisation de 60 secondes par défaut.")
        session_cycle_interval_s = 60
    logger.info(f"[OrchestratorLoop] Intervalle du cycle de session de l'orchestrateur : {session_cycle_interval_s} secondes.")

    app_config_for_next_cycle = app_config_initial_load

    try:
        while not global_shutdown_event.is_set():
            app_config_for_next_cycle = _run_single_trading_session_cycle(
                app_config_current_cycle=app_config_for_next_cycle, # Passer la config du cycle précédent (ou initiale)
                cli_args=cli_args,
                orchestrator_shutdown_event=global_shutdown_event
            )
            
            if global_shutdown_event.is_set():
                logger.info("[OrchestratorLoop] Événement d'arrêt global reçu pendant l'exécution du cycle. Sortie de la boucle principale.")
                break
            
            logger.info(f"[OrchestratorLoop] Prochain cycle de session de trading dans {session_cycle_interval_s} secondes "
                        "(sauf si un arrêt est demandé)...")
            
            # Attendre l'intervalle ou jusqu'à ce que l'événement d'arrêt soit signalé
            shutdown_requested_during_wait = global_shutdown_event.wait(timeout=session_cycle_interval_s)
            if shutdown_requested_during_wait:
                logger.info("[OrchestratorLoop] Événement d'arrêt global reçu pendant la période d'attente. Sortie de la boucle principale.")
                break
    except KeyboardInterrupt:
        logger.info("[OrchestratorLoop] Interruption clavier (Ctrl+C) reçue par l'orchestrateur. Demande d'arrêt global...")
        global_shutdown_event.set() # S'assurer que l'événement est positionné
    except Exception as e_main_loop_exc: # pylint: disable=broad-except
        logger.critical(f"[OrchestratorLoop] Erreur non gérée dans la boucle principale de l'orchestrateur : {e_main_loop_exc}", exc_info=True)
        global_shutdown_event.set() # Déclencher l'arrêt en cas d'erreur grave

    # --- Procédure d'Arrêt ---
    logger.info("[OrchestratorShutdown] Boucle principale terminée. Démarrage de la procédure d'arrêt des managers actifs...")
    
    # Créer une copie des IDs pour itérer, car le dictionnaire peut changer
    active_manager_ids_at_shutdown_time = list(active_managers.keys())
    if not active_manager_ids_at_shutdown_time:
        logger.info("[OrchestratorShutdown] Aucun manager actif à arrêter.")
    else:
        logger.info(f"[OrchestratorShutdown] Arrêt de {len(active_manager_ids_at_shutdown_time)} manager(s) actif(s)...")

    for manager_id_to_stop in active_manager_ids_at_shutdown_time:
        manager_instance_to_stop = active_managers.get(manager_id_to_stop)
        thread_instance_to_join = active_threads.get(manager_id_to_stop)
        
        if manager_instance_to_stop:
            logger.info(f"[OrchestratorShutdown] Envoi de la demande d'arrêt au manager : {manager_id_to_stop}")
            try:
                manager_instance_to_stop.stop_trading() # Signale au manager de s'arrêter
            except Exception as e_stop_mgr_call: # pylint: disable=broad-except
                logger.error(f"[OrchestratorShutdown] Erreur lors de l'appel à stop_trading() pour le manager {manager_id_to_stop}: {e_stop_mgr_call}")
        
        if thread_instance_to_join and thread_instance_to_join.is_alive():
            logger.info(f"[OrchestratorShutdown] Attente de la fin du thread du manager {manager_id_to_stop} (timeout 30s)...")
            thread_instance_to_join.join(timeout=30) # Attendre que le thread se termine
            if thread_instance_to_join.is_alive():
                logger.warning(f"[OrchestratorShutdown] Le thread du Manager {manager_id_to_stop} ne s'est pas terminé dans le délai de 30s.")
            else:
                logger.info(f"[OrchestratorShutdown] Le thread du Manager {manager_id_to_stop} s'est terminé proprement.")
        elif thread_instance_to_join: # Thread existe mais n'est plus vivant
             logger.info(f"[OrchestratorShutdown] Le thread du Manager {manager_id_to_stop} était déjà terminé.")

        # Nettoyer les références après avoir tenté l'arrêt
        if manager_id_to_stop in active_managers: del active_managers[manager_id_to_stop]
        if manager_id_to_stop in active_threads: del active_threads[manager_id_to_stop]

    logger.info("[OrchestratorShutdown] --- Tous les managers actifs ont été priés de s'arrêter. Arrêt de l'orchestrateur terminé. ---")


def main():
    """
    Point d'entrée principal pour lancer l'orchestrateur de trading en direct.
    Configure les arguments CLI, les gestionnaires de signaux, et lance la boucle principale.
    """
    run_script_start_time = time.time()
    # Le logging sera configuré par load_all_configs dans main_loop_orchestrator la première fois.
    # Les logs avant cela utiliseront la config basicConfig définie au début du fichier.

    parser = argparse.ArgumentParser(description="Orchestrateur du Bot de Trading en Direct.")
    parser.add_argument(
        "--pair",
        type=str,
        action='append', # Permet de spécifier --pair X --pair Y
        default=None, # Si non fourni, sera None, et toutes les paires des déploiements actifs seront considérées
        help="Paire(s) de trading spécifique(s) à traiter (ex: BTCUSDT). "
             "Si non fourni, toutes les paires des déploiements actifs seront considérées."
    )
    parser.add_argument(
        "--tf", "--context", dest="tf", # Alias --tf et --context, stocké dans args.tf
        type=str,
        default=None, # Optionnel
        help="Label de contexte global optionnel pour cette exécution de l'orchestrateur. "
             "Peut être utilisé pour le logging ou un filtrage de haut niveau. "
             "Chaque LiveTradingManager utilisera le contexte spécifié dans ses propres "
             "paramètres de déploiement (lus depuis le fichier JSON de résultats WFO)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None, # Sera déterminé par PROJECT_ROOT si non fourni
        help="Chemin vers la racine du projet si le script n'est pas exécuté depuis la racine du projet.",
    )
    cli_args = parser.parse_args()

    # Configurer les gestionnaires de signaux globaux pour un arrêt propre
    # SIGINT est typiquement Ctrl+C. SIGTERM est un signal d'arrêt plus général.
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"): # SIGTERM n'est pas disponible sur toutes les plateformes (ex: Windows natif)
        try:
            signal.signal(signal.SIGTERM, signal_handler)
        except (OSError, AttributeError, ValueError) as e_sigterm: # Attraper les erreurs possibles
             logger.warning(f"Impossible de configurer le gestionnaire pour SIGTERM sur cette plateforme : {e_sigterm}")
    logger.info("[MainEntry] Gestionnaires de signaux globaux (SIGINT, SIGTERM si disponible) configurés.")

    logger.info("--- Démarrage du Script run_live.py (Orchestrateur de Trading en Direct) ---")
    
    try:
        main_loop_orchestrator(cli_args)
    except SystemExit: # Attraper sys.exit() pour un logging final propre si appelé ailleurs
        logger.info("[MainEntry] Sortie de l'application demandée par sys.exit().")
    except Exception as e_main_entry: # pylint: disable=broad-except
        logger.critical(f"[MainEntry] ERREUR CRITIQUE non gérée dans l'exécution principale de run_live.py : {e_main_entry}", exc_info=True)
    finally:
        run_script_end_time = time.time()
        total_duration_seconds = run_script_end_time - run_script_start_time
        logger.info(f"--- Script run_live.py Terminé --- Temps d'Exécution Total de l'Orchestrateur : {total_duration_seconds:.2f} secondes "
                    f"({timedelta(seconds=total_duration_seconds)})")
        logging.shutdown() # S'assurer que tous les messages de log sont écrits avant la sortie

if __name__ == "__main__":
    # Cette condition est cruciale pour le bon fonctionnement de multiprocessing sur Windows.
    # Elle assure que le code de création de processus n'est exécuté que lorsque le script
    # est le module principal, et non lorsqu'il est importé par un processus fils.
    main()
