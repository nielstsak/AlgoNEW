# src/live/manager.py
"""
Ce module définit LiveTradingManager, qui gère une session de trading en direct
pour une stratégie, une paire, et une configuration de compte spécifiques.
Il orchestre l'acquisition des données, la génération de signaux, l'exécution
des ordres et la gestion de l'état.
"""
import importlib
import logging
import os # Pour os.path.join, bien que pathlib soit préféré
import threading
import time
import json
import uuid # Pour les trade_cycle_id
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Tuple, Union, cast
from datetime import datetime, timezone, timedelta
import re # Pour nettoyer les noms de fichiers pour les logs

import pandas as pd
import numpy as np # Pour np.isnan et autres opérations numériques

# Imports depuis le projet src
try:
    from src.config.loader import AppConfig # Pour le typage
    from src.config.definitions import (
        StrategyDeployment, AccountConfig, GlobalLiveSettings,
        PathsConfig, ApiKeys, LiveFetchConfig, LiveLoggingConfig, LoggingConfig
    )
    from src.data import acquisition_live # Pour initialize_pair_data et update_pair_data_rest (à créer/adapter)
    from src.data import preprocessing_live # Pour preprocess_live_data_for_strategy
    from src.strategies.base import BaseStrategy
    from src.live.state import (
        LiveTradingState, STATUT_1_NO_TRADE_NO_OCO,
        STATUT_2_ENTRY_FILLED_OCO_PENDING, STATUT_3_OCO_ACTIVE
    )
    from src.live.execution import OrderExecutionClient
    from src.utils.exchange_utils import get_precision_from_filter, adjust_precision # Pour ajuster les prix OCO
    from src.utils.logging_setup import setup_logging # Pour configurer le logger de trade spécifique
    from src.utils.run_utils import _sanitize_path_component # Pour nettoyer les noms de fichiers/dossiers
except ImportError as e:
    # Ce log est un fallback, le logging principal sera configuré plus tard
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger(__name__).critical(
        f"LiveTradingManager: Erreur d'importation critique : {e}. Vérifiez PYTHONPATH.",
        exc_info=True
    )
    raise # Renvoyer l'erreur car le manager ne peut pas fonctionner

logger = logging.getLogger(__name__) # Logger principal pour ce module

# --- Constantes pour le LiveTradingManager ---
DEFAULT_QUOTE_ASSET = "USDC" # Actif de cotation par défaut (ex: pour les soldes)
MIN_USDC_BALANCE_FOR_TRADE = 10.0 # Solde USDC minimum pour initier un nouveau trade
SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT = 5.0 # Seuil pour considérer un prêt comme significatif
API_CALL_DELAY_SECONDS = 0.35 # Délai entre les appels API pour éviter le rate limiting
MAIN_LOOP_SLEEP_SECONDS = 5 # Temps de pause de la boucle principale
KLINE_FETCH_LIMIT_LIVE_UPDATE = 10 # Nombre de K-lines 1-min à récupérer lors des mises à jour
ORDER_STATUS_CHECK_DELAY_SECONDS = 3 # Délai avant de vérifier le statut d'un ordre fraîchement placé
MAX_OCO_CONFIRMATION_ATTEMPTS = 10 # Nombre max de tentatives pour confirmer un OCO
FULL_STATE_SYNC_INTERVAL_MINUTES = 5 # Intervalle pour une synchronisation complète avec l'exchange
MIN_EXECUTED_QTY_THRESHOLD = 1e-9 # Seuil pour considérer une quantité comme non nulle

# Placeholder pour acquisition_live.update_pair_data_rest
# Cette fonction devra être implémentée dans acquisition_live.py
def update_pair_data_rest_placeholder(
    pair: str,
    raw_file_path: Path,
    account_type: str,
    limit_fetch: int,
    last_known_timestamp_ms: Optional[int]
) -> bool:
    """
    Placeholder pour mettre à jour le fichier de données brutes avec les nouvelles K-lines.
    Retourne True si de nouvelles données ont été ajoutées, False sinon.
    """
    logger.warning(f"Utilisation de update_pair_data_rest_placeholder pour {pair}. "
                   "Implémentez la vraie fonction dans acquisition_live.py.")
    # Simuler une tentative de fetch et un ajout potentiel de données
    # Dans une vraie implémentation, appellerait get_binance_klines_rest avec start_timestamp_ms
    # et fusionnerait avec le fichier existant.
    time.sleep(0.1) # Simuler un appel API
    # Pour le test, on peut simuler un ajout de données une fois sur deux
    # if np.random.rand() > 0.5:
    #     logger.debug(f"Placeholder: Simulé ajout de nouvelles données pour {pair}.")
    #     return True
    return False # Simuler pas de nouvelles données pour l'instant


class LiveTradingManager:
    """
    Gère une session de trading en direct pour une stratégie, une paire, et une
    configuration de compte spécifiques.
    """
    def __init__(self,
                 app_config: 'AppConfig',
                 strategy_deployment_config: StrategyDeployment,
                 account_config: AccountConfig,
                 pair_to_trade: str,
                 context_label_from_deployment: str): # Contexte des paramètres optimisés
        """
        Initialise le LiveTradingManager.

        Args:
            app_config (AppConfig): Configuration globale de l'application.
            strategy_deployment_config (StrategyDeployment): Configuration du déploiement
                spécifique de la stratégie.
            account_config (AccountConfig): Configuration du compte de trading à utiliser.
            pair_to_trade (str): Symbole de la paire de trading (ex: BTCUSDT).
            context_label_from_deployment (str): Label de contexte associé aux paramètres
                optimisés de la stratégie (ex: "5min_rsi_filter").
        """
        self.app_config = app_config
        self.strategy_deployment_config = strategy_deployment_config
        self.account_config = account_config
        self.pair_symbol = pair_to_trade.upper()
        # Le context_label ici est celui des paramètres optimisés, utilisé pour nommer les fichiers de données traitées.
        self.context_label_params = context_label_from_deployment

        self.log_prefix = (f"[LiveMgr][{self.pair_symbol}]"
                           f"[Acc:{self.account_config.account_alias}]"
                           f"[StratDepID:{_sanitize_path_component(self.strategy_deployment_config.strategy_id)}]"
                           f"[CtxParams:{_sanitize_path_component(self.context_label_params)}]")
        
        logger.info(f"{self.log_prefix} Initialisation du LiveTradingManager...")

        self.shutdown_event = threading.Event()
        self.strategy: Optional[BaseStrategy] = None
        self.state_manager: Optional[LiveTradingState] = None
        self.execution_client: Optional[OrderExecutionClient] = None
        self.trade_event_logger: Optional[logging.Logger] = None # Logger dédié aux événements de trade

        # Configuration des chemins
        if not self.app_config.project_root:
            msg = "project_root n'est pas défini dans AppConfig. Impossible de résoudre les chemins."
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        self.project_root: Path = Path(self.app_config.project_root)
        
        paths_cfg: PathsConfig = self.app_config.global_config.paths
        self.raw_data_dir: Path = self.project_root / paths_cfg.data_live_raw
        self.processed_data_dir: Path = self.project_root / paths_cfg.data_live_processed
        self.state_dir: Path = self.project_root / paths_cfg.live_state
        self.trades_log_output_dir: Path = self.project_root / paths_cfg.logs_live # Répertoire de base pour les logs de trade

        # Nom du fichier de données brutes 1-min (standard)
        self.raw_1min_data_file_path = self.raw_data_dir / f"{self.pair_symbol}_1min_live_raw.csv"
        # Nom du fichier de données traitées (spécifique au contexte des paramètres)
        s_context_params = _sanitize_path_component(self.context_label_params)
        self.processed_data_file_path = self.processed_data_dir / f"{self.pair_symbol}_{s_context_params}_processed_live.csv"

        # Attributs de suivi
        self.last_1m_kline_open_timestamp_ms: Optional[int] = None
        self.is_isolated_margin_trading: bool = (self.account_config.account_type == "ISOLATED_MARGIN")
        self.base_asset: str = ""
        self.quote_asset: str = ""
        self.oco_confirmation_attempts: int = 0
        self.last_full_state_sync_time: Optional[datetime] = None
        self.current_trade_cycle_id: Optional[str] = None # Géré par le state_manager

        self._initialize_components()
        logger.info(f"{self.log_prefix} Initialisation du LiveTradingManager terminée.")

    def _initialize_components(self) -> None:
        """Initialise tous les composants critiques du manager."""
        logger.info(f"{self.log_prefix} Initialisation des composants critiques...")
        
        self._setup_trade_event_logger() # Configurer le logger de trade en premier
        self._load_strategy_and_params() # Charge self.strategy
        self._initialize_execution_client() # Init self.execution_client, self.base_asset, self.quote_asset
        self._initialize_state_manager() # Init self.state_manager
        
        # S'assurer que les composants essentiels sont initialisés
        if not all([self.strategy, self.execution_client, self.state_manager, self.trade_event_logger]):
            msg = "Un ou plusieurs composants critiques (stratégie, client d'exécution, gestionnaire d'état, logger de trade) n'ont pas pu être initialisés."
            logger.critical(f"{self.log_prefix} {msg}")
            raise RuntimeError(msg)

        # Initialisation des données et de l'état initial
        logger.info(f"{self.log_prefix} Initialisation des données (fetch et preprocessing)...")
        self._fetch_initial_1min_klines()
        self._run_initial_preprocessing()
        
        logger.info(f"{self.log_prefix} Détermination de l'état initial depuis l'exchange...")
        self._determine_initial_status(is_periodic_sync=False)
        self.last_full_state_sync_time = datetime.now(timezone.utc)
        
        # Passer le contexte au state_manager pour enregistrement (si souhaité)
        self.state_manager.update_specific_fields({
            "instance_context_label": self.context_label_params,
            "instance_account_alias": self.account_config.account_alias
        })
        
        logger.info(f"{self.log_prefix} Initialisation des composants réussie. État actuel : {self.state_manager.get_current_status_name()}")

    def _setup_trade_event_logger(self) -> None:
        """Configure un logger dédié pour les événements de trade de cette instance."""
        s_pair = _sanitize_path_component(self.pair_symbol)
        s_account = _sanitize_path_component(self.account_config.account_alias)
        s_strat_id = _sanitize_path_component(self.strategy_deployment_config.strategy_id)
        s_context = _sanitize_path_component(self.context_label_params)

        # Nom du logger unique pour éviter les conflits si plusieurs managers tournent
        logger_name = f"trade_events.{s_pair}.{s_account}.{s_strat_id}.{s_context}"
        self.trade_event_logger = logging.getLogger(logger_name)
        
        # Éviter d'ajouter des handlers multiples si ce logger est déjà configuré
        if self.trade_event_logger.hasHandlers():
            logger.debug(f"{self.log_prefix} Le logger d'événements de trade '{logger_name}' est déjà configuré.")
            return

        self.trade_event_logger.setLevel(logging.INFO) # Niveau par défaut pour les logs de trade
        self.trade_event_logger.propagate = False # Ne pas propager au root logger si on veut des logs séparés

        # Créer le répertoire de log spécifique à cette session de trading
        # Structure: logs_live / PAIR / ACCOUNT_ALIAS / STRATEGY_DEPLOYMENT_ID / CONTEXT_PARAMS / trade_events.log
        session_specific_log_dir = self.trades_log_output_dir / s_pair / s_account / s_strat_id / s_context
        ensure_dir_exists(session_specific_log_dir)
        
        trade_log_file_path = session_specific_log_dir / "trade_events.log"

        # Handler pour le fichier de log des trades
        fh = logging.handlers.RotatingFileHandler(
            trade_log_file_path,
            maxBytes=5*1024*1024, # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        # Utiliser un format simple pour les logs de trade, ou un format JSON structuré
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.trade_event_logger.addHandler(fh)
        
        logger.info(f"{self.log_prefix} Logger d'événements de trade configuré. Fichier : {trade_log_file_path}")


    def _load_strategy_and_params(self) -> None:
        """Charge la classe de stratégie et ses paramètres optimisés."""
        logger.info(f"{self.log_prefix} Chargement de la stratégie et des paramètres optimisés...")
        
        results_config_rel_path_str = self.strategy_deployment_config.results_config_path
        if not results_config_rel_path_str:
            msg = f"'results_config_path' est vide dans StrategyDeployment '{self.strategy_deployment_config.strategy_id}'."
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)

        # results_config_path est relatif à project_root
        optimized_params_file = self.project_root / results_config_rel_path_str
        if not optimized_params_file.is_file():
            msg = f"Fichier de paramètres optimisés (live_config.json) non trouvé à : {optimized_params_file}"
            logger.critical(f"{self.log_prefix} {msg}")
            raise FileNotFoundError(msg)

        try:
            with open(optimized_params_file, 'r', encoding='utf-8') as f:
                live_params_from_file = json.load(f)
        except Exception as e_load_json:
            msg = f"Échec du chargement ou du parsing du fichier de paramètres optimisés {optimized_params_file}: {e_load_json}"
            logger.critical(f"{self.log_prefix} {msg}", exc_info=True)
            raise ValueError(msg) from e_load_json

        # Extraire les informations nécessaires du live_config.json
        strategy_name_base = live_params_from_file.get("strategy_name_base")
        strategy_script_ref_from_file = live_params_from_file.get("strategy_script_reference")
        strategy_class_name_from_file = live_params_from_file.get("strategy_class_name")
        optimized_params_for_strategy = live_params_from_file.get("best_params") # 'best_params' est la clé attendue

        if not all([strategy_name_base, strategy_script_ref_from_file, strategy_class_name_from_file, isinstance(optimized_params_for_strategy, dict)]):
            msg = (f"Champs requis ('strategy_name_base', 'strategy_script_reference', 'strategy_class_name', "
                   f"ou 'best_params') manquants ou invalides dans {optimized_params_file.name}.")
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        
        # Vérifier la cohérence du contexte (optionnel, mais bon à savoir)
        file_pair = live_params_from_file.get("pair_symbol")
        file_context_params = live_params_from_file.get("context_label") # Le contexte des paramètres
        if file_pair and file_pair.upper() != self.pair_symbol:
            logger.warning(f"{self.log_prefix} Discordance de paire : Manager pour '{self.pair_symbol}', "
                           f"fichier de params pour '{file_pair}'. Utilisation des params du fichier.")
        if file_context_params and file_context_params != self.context_label_params:
            logger.warning(f"{self.log_prefix} Discordance de contexte des paramètres : Manager pour '{self.context_label_params}', "
                           f"fichier de params pour '{file_context_params}'. Utilisation des params du fichier avec contexte '{file_context_params}'.")
            # Si on veut forcer l'utilisation du contexte du fichier de params pour le manager:
            # self.context_label_params = file_context_params
            # self.log_prefix = f"[LiveMgr][{self.pair_symbol}][Acc:{self.account_config.account_alias}][StratDepID:{_sanitize_path_component(self.strategy_deployment_config.strategy_id)}][CtxParams:{_sanitize_path_component(self.context_label_params)}]"
            # s_context_params = _sanitize_path_component(self.context_label_params)
            # self.processed_data_file_path = self.processed_data_dir / f"{self.pair_symbol}_{s_context_params}_processed_live.csv"


        # Charger dynamiquement la classe de stratégie
        try:
            # Le script_reference est relatif à la racine du projet (ex: src/strategies/my_strat.py)
            # Convertir en chemin d'import Python (ex: src.strategies.my_strat)
            module_import_path = strategy_script_ref_from_file.replace('.py', '').replace(os.sep, '.')
            
            StrategyClass = getattr(importlib.import_module(module_import_path), strategy_class_name_from_file)
            self.strategy = StrategyClass(
                strategy_name=strategy_name_base, # Utiliser le nom de base pour l'instance
                symbol=self.pair_symbol,
                params=optimized_params_for_strategy
            )
            logger.info(f"{self.log_prefix} Stratégie '{strategy_class_name_from_file}' (base: {strategy_name_base}) "
                        f"chargée avec les paramètres de {optimized_params_file.name}.")
        except Exception as e_strat_load:
            msg = f"Erreur lors de l'importation ou de l'instanciation de la stratégie {strategy_class_name_from_file} depuis {strategy_script_ref_from_file}: {e_strat_load}"
            logger.critical(f"{self.log_prefix} {msg}", exc_info=True)
            raise RuntimeError(msg) from e_strat_load

    def _initialize_execution_client(self) -> None:
        """Initialise le client d'exécution des ordres."""
        logger.info(f"{self.log_prefix} Initialisation de OrderExecutionClient pour le compte : {self.account_config.account_alias}")
        
        # Récupérer les clés API depuis AppConfig
        api_creds_tuple = self.app_config.api_keys.credentials.get(self.account_config.account_alias)
        if not api_creds_tuple or not api_creds_tuple[0] or not api_creds_tuple[1]:
            msg = (f"Clés API (key/secret) non trouvées ou incomplètes pour l'alias de compte '{self.account_config.account_alias}'. "
                   f"Vérifiez les variables d'environnement '{self.account_config.api_key_env_var}' et "
                   f"'{self.account_config.api_secret_env_var}' et la configuration des comptes.")
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)
        
        api_key_val, api_secret_val = api_creds_tuple
        
        try:
            self.execution_client = OrderExecutionClient(
                api_key=api_key_val,
                api_secret=api_secret_val,
                account_type=self.account_config.account_type,
                is_testnet=self.account_config.is_testnet
            )
            if not self.execution_client.test_connection():
                msg = f"Échec du test de connexion API pour le compte {self.account_config.account_alias}."
                logger.critical(f"{self.log_prefix} {msg}")
                raise ConnectionError(msg)
            
            # Récupérer et stocker base_asset et quote_asset
            symbol_info = self.execution_client.get_symbol_info(self.pair_symbol)
            if not symbol_info or not symbol_info.get('baseAsset') or not symbol_info.get('quoteAsset'):
                msg = f"Informations de symbole invalides ou manquantes pour {self.pair_symbol} depuis l'exchange."
                logger.critical(f"{self.log_prefix} {msg}")
                raise ValueError(msg)
            self.base_asset = symbol_info['baseAsset']
            self.quote_asset = symbol_info['quoteAsset']
            
            # Configurer le contexte de backtest/live pour l'instance de stratégie
            if self.strategy:
                self.strategy.set_backtest_context( # Renommer cette méthode ou clarifier son usage pour le live
                    pair_config=symbol_info, # C'est le symbol_info de l'exchange
                    is_futures=self.app_config.global_config.simulation_defaults.is_futures_trading, # Ou basé sur account_type
                    leverage=int(self.strategy.get_param('margin_leverage', self.app_config.global_config.simulation_defaults.margin_leverage)),
                    initial_equity=0, # L'équité initiale n'est pas gérée par la stratégie elle-même en live, mais par le manager/state
                    account_type=self.account_config.account_type
                )
            logger.info(f"{self.log_prefix} OrderExecutionClient initialisé et connecté. "
                        f"Paire: {self.pair_symbol} (Base: {self.base_asset}, Quote: {self.quote_asset}).")

        except Exception as e_exec_client:
            msg = f"Échec de l'initialisation de OrderExecutionClient : {e_exec_client}"
            logger.critical(f"{self.log_prefix} {msg}", exc_info=True)
            raise RuntimeError(msg) from e_exec_client

    def _initialize_state_manager(self) -> None:
        """Initialise le gestionnaire d'état."""
        # Créer un nom de fichier d'état unique pour cette session
        s_pair = _sanitize_path_component(self.pair_symbol)
        s_account = _sanitize_path_component(self.account_config.account_alias)
        s_strat_id = _sanitize_path_component(self.strategy_deployment_config.strategy_id)
        s_context_params = _sanitize_path_component(self.context_label_params)
        
        # Structure du nom de fichier : PAIR_ACCOUNT_STRATID_CONTEXT_state.json
        state_file_name = f"{s_pair}_{s_account}_{s_strat_id}_{s_context_params}_state.json"
        state_file_path = self.state_dir / state_file_name
        
        logger.info(f"{self.log_prefix} Initialisation de LiveTradingState. Fichier d'état : {state_file_path}")
        try:
            # Déterminer l'actif de cotation pour les commissions (généralement l'actif de cotation de la paire)
            quote_asset_for_state = self.quote_asset if self.quote_asset else DEFAULT_QUOTE_ASSET
            self.state_manager = LiveTradingState(
                pair_symbol=self.pair_symbol,
                state_file_path=state_file_path,
                quote_asset_for_commission=quote_asset_for_state
            )
            # Récupérer le cycle de trade ID depuis l'état chargé, s'il existe
            self.current_trade_cycle_id = self.state_manager.get_state_snapshot().get("current_trade_cycle_id")
            logger.info(f"{self.log_prefix} LiveTradingState initialisé. Statut : {self.state_manager.get_current_status_name()}. "
                        f"Cycle ID actuel (depuis état) : {self.current_trade_cycle_id}")
        except Exception as e_state_mgr:
            msg = f"Échec de l'initialisation de LiveTradingState : {e_state_mgr}"
            logger.critical(f"{self.log_prefix} {msg}", exc_info=True)
            raise RuntimeError(msg) from e_state_mgr

    def _fetch_initial_1min_klines(self) -> None:
        """S'assure que le fichier de données brutes 1-minute est initialisé."""
        if not self.state_manager: return # Should not happen if _initialize_components is called in order
        logger.debug(f"{self.log_prefix} Vérification/Initialisation des K-lines 1-minute initiales pour {self.raw_1min_data_file_path}")
        
        live_fetch_settings: LiveFetchConfig = self.app_config.live_config.live_fetch
        limit_history_needed = live_fetch_settings.limit_init_history
        
        try:
            acquisition_live.initialize_pair_data(
                pair=self.pair_symbol,
                raw_path=self.raw_1min_data_file_path,
                total_klines_to_fetch=limit_history_needed,
                account_type=self.account_config.account_type # Pour déterminer l'endpoint API (spot/futures)
            )
            # Mettre à jour le timestamp de la dernière kline 1m connue
            if self.raw_1min_data_file_path.exists() and self.raw_1min_data_file_path.stat().st_size > 0:
                df_raw_check = pd.read_csv(self.raw_1min_data_file_path, usecols=['timestamp'])
                if not df_raw_check.empty:
                    timestamps_series = pd.to_datetime(df_raw_check['timestamp'], errors='coerce', utc=True).dropna()
                    if not timestamps_series.empty:
                        self.last_1m_kline_open_timestamp_ms = int(timestamps_series.iloc[-1].timestamp() * 1000)
                        logger.info(f"{self.log_prefix} Dernier timestamp 1-min connu après init : "
                                    f"{pd.to_datetime(self.last_1m_kline_open_timestamp_ms, unit='ms', utc=True)}")
        except Exception as e_fetch_init:
            logger.error(f"{self.log_prefix} Erreur lors de l'initialisation des données K-lines 1-minute : {e_fetch_init}", exc_info=True)
            # Continuer peut être risqué si les données de base manquent.

    def _run_initial_preprocessing(self) -> None:
        """Exécute le prétraitement initial pour créer le fichier de données traitées."""
        if not self.strategy or not self.processed_data_file_path or not self.raw_1min_data_file_path.exists():
            logger.warning(f"{self.log_prefix} Prérequis manquants pour le prétraitement initial. Saut.")
            return
        logger.debug(f"{self.log_prefix} Exécution du prétraitement initial des données pour la stratégie...")
        try:
            preprocessing_live.preprocess_live_data_for_strategy(
                raw_data_path=self.raw_1min_data_file_path,
                processed_output_path=self.processed_data_file_path,
                strategy_params=self.strategy.params, # Utiliser les paramètres de l'instance de stratégie
                strategy_name=self.strategy.strategy_name # Utiliser le nom de base de l'instance
            )
        except Exception as e_preproc_init:
            logger.error(f"{self.log_prefix} Erreur lors du prétraitement initial des données : {e_preproc_init}", exc_info=True)
            # Selon la criticité, on pourrait arrêter ici.

    def _run_current_preprocessing_cycle(self) -> None:
        """Exécute un cycle de prétraitement (appelé après mise à jour des données brutes)."""
        if not self.strategy or not self.processed_data_file_path or not self.raw_1min_data_file_path.exists():
            logger.warning(f"{self.log_prefix} Prérequis manquants pour le cycle de prétraitement actuel. Saut.")
            return
        logger.debug(f"{self.log_prefix} Exécution du cycle de prétraitement des données...")
        try:
            preprocessing_live.preprocess_live_data_for_strategy(
                raw_data_path=self.raw_1min_data_file_path,
                processed_output_path=self.processed_data_file_path,
                strategy_params=self.strategy.params,
                strategy_name=self.strategy.strategy_name
            )
        except Exception as e_preproc_cycle:
            logger.error(f"{self.log_prefix} Erreur lors du cycle de prétraitement des données : {e_preproc_cycle}", exc_info=True)


    def _get_latest_price_for_loan_check(self) -> float:
        """
        Obtient le dernier prix de clôture à partir des données traitées.
        Si indisponible, tente de récupérer via ticker API. Utilisé pour évaluer la valeur des prêts.
        """
        if self.processed_data_file_path and self.processed_data_file_path.exists() and self.processed_data_file_path.stat().st_size > 0:
            try:
                # Lire seulement la dernière ligne pour l'efficacité si le fichier est grand
                # Ceci est une approximation, read_csv complet puis iloc[-1] est plus sûr mais plus lent.
                # Pour un fichier CSV qui grandit, il faut une méthode plus robuste pour lire la dernière ligne.
                # Pour l'instant, on charge les colonnes nécessaires.
                df = pd.read_csv(self.processed_data_file_path, usecols=['close'])
                if not df.empty and 'close' in df.columns and pd.notna(df['close'].iloc[-1]):
                    return float(df['close'].iloc[-1])
            except Exception as e_read_price:
                logger.warning(f"{self.log_prefix} Erreur lors de la lecture du dernier prix depuis {self.processed_data_file_path}: {e_read_price}. Tentative via API.")
        
        # Fallback à l'API si le fichier n'est pas disponible ou si la lecture échoue
        if self.execution_client:
            try:
                # Utiliser get_symbol_ticker du client SDK Binance sous-jacent
                if hasattr(self.execution_client.client, 'get_symbol_ticker'):
                    ticker_info = self.execution_client.client.get_symbol_ticker(symbol=self.pair_symbol) # type: ignore
                    if ticker_info and 'price' in ticker_info:
                        return float(ticker_info['price'])
                    else:
                        logger.warning(f"{self.log_prefix} Réponse du ticker API invalide pour {self.pair_symbol}: {ticker_info}")
                else:
                    logger.warning(f"{self.log_prefix} Méthode get_symbol_ticker non disponible sur le client SDK.")
            except Exception as e_api_ticker:
                logger.error(f"{self.log_prefix} Erreur lors de la récupération du ticker API pour {self.pair_symbol}: {e_api_ticker}")
        
        logger.error(f"{self.log_prefix} Impossible d'obtenir le dernier prix pour l'évaluation des prêts. Retour de 0.0.")
        return 0.0

    # --- Implémentation des méthodes de logique de trading et de gestion d'état ---
    # (Les méthodes _determine_initial_status, _check_and_process_orders_via_rest,
    # _handle_status_X, _handle_trade_closure_and_loan_repayment, etc.
    # seraient implémentées ici, en s'inspirant fortement du code existant fourni
    # dans le fichier manager.py original, mais en adaptant les appels et la logique
    # aux nouvelles structures de configuration et aux membres de cette classe.)

    # --- Exemple simplifié de la structure des méthodes de gestion d'état ---
    def _determine_initial_status(self, is_periodic_sync: bool = False) -> None:
        """Détermine l'état initial ou synchronise périodiquement avec l'exchange."""
        # Logique complexe impliquant des appels à self.execution_client pour :
        # - get_margin_usdc_balance
        # - get_active_margin_loans
        # - get_all_open_margin_orders
        # - get_open_margin_oco_orders
        # Puis, comparaison avec self.state_manager.get_state_snapshot()
        # et appel des méthodes de transition du state_manager appropriées
        # (transition_to_status_1, transition_to_status_2, transition_to_status_3,
        #  _deduce_position_from_loan_and_set_status2, _handle_loan_without_clear_position)
        # et potentiellement _cancel_all_orders_for_pair.
        # Cette méthode est cruciale et sa logique doit être très robuste.
        # Pour ce prompt, on suppose que la logique de l'ancien manager.py est réutilisée ici.
        sync_type_log = "[SYNC]" if is_periodic_sync else "[INIT_STATUS]"
        logger.info(f"{self.log_prefix}{sync_type_log} Début de la détermination/synchronisation de l'état avec l'exchange.")
        
        if not self.execution_client or not self.state_manager:
            logger.error(f"{self.log_prefix}{sync_type_log} Client d'exécution ou gestionnaire d'état non initialisé. Impossible de déterminer l'état.")
            return

        # ... (logique détaillée de récupération d'infos exchange et de décision) ...
        # Cette logique est complexe et dépend fortement des détails de l'API et des états.
        # Elle est omise ici pour la brièveté mais serait une partie substantielle.
        # Le code existant dans le fichier manager.py fourni initialement contient une bonne base.
        
        # Exemple très simplifié :
        current_bot_status = self.state_manager.get_current_status()
        logger.info(f"{self.log_prefix}{sync_type_log} Statut actuel du bot avant synchro : {current_bot_status}")
        # ... (appels API) ...
        # if (pas d'ordres ouverts et pas de prêts significatifs) and current_bot_status != STATUT_1_NO_TRADE_NO_OCO:
        #     self.state_manager.transition_to_status_1(f"{sync_type_log}_NO_POS_NO_OCO_DETECTED")
        #     self.current_trade_cycle_id = None # Assurer la réinitialisation
        
        # Placeholder pour la logique de l'ancien manager.py
        # Cette méthode est très dépendante de l'état actuel et des réponses de l'exchange.
        # Elle doit être implémentée avec soin en s'inspirant du code existant.
        logger.warning(f"{self.log_prefix}{sync_type_log} La logique détaillée de _determine_initial_status n'est pas entièrement réimplémentée dans cette version.")
        
        # À la fin, loguer le statut après synchro
        self.state_manager.update_last_successful_sync_timestamp()
        logger.info(f"{self.log_prefix}{sync_type_log} Détermination/synchronisation de l'état terminée. Statut actuel du bot : {self.state_manager.get_current_status_name()}")


    def _check_new_1min_kline_and_trigger_preprocessing(self) -> bool:
        """
        Vérifie si une nouvelle bougie 1-minute est disponible, met à jour les données brutes,
        et déclenche le prétraitement.
        Retourne True si de nouvelles données ont été traitées, False sinon.
        """
        log_ctx = "[DataUpdateCycle]"
        # La logique exacte pour savoir si une "nouvelle" bougie 1-min est "fermée"
        # dépend de la granularité. Si on tourne toutes les 5s, on peut vérifier si
        # current_time_utc.second est proche de 0 après le début d'une nouvelle minute.
        # Pour l'instant, on simule un appel à une fonction qui mettrait à jour le fichier.
        
        # Utiliser le placeholder pour update_pair_data_rest
        # Dans une vraie implémentation, cette fonction devrait :
        # 1. Lire le dernier timestamp du fichier raw_1min_data_file_path.
        # 2. Appeler get_binance_klines_rest avec start_timestamp_ms = dernier_ts + 1min.
        # 3. Ajouter les nouvelles klines au fichier CSV.
        # 4. Retourner True si de nouvelles klines ont été ajoutées.
        
        logger.debug(f"{self.log_prefix}{log_ctx} Vérification de nouvelles données K-line 1-minute...")
        new_data_fetched = update_pair_data_rest_placeholder(
            pair=self.pair_symbol,
            raw_file_path=self.raw_1min_data_file_path,
            account_type=self.account_config.account_type,
            limit_fetch=KLINE_FETCH_LIMIT_LIVE_UPDATE,
            last_known_timestamp_ms=self.last_1m_kline_open_timestamp_ms
        )

        if new_data_fetched:
            logger.info(f"{self.log_prefix}{log_ctx} Nouvelles données 1-minute récupérées. Lancement du prétraitement.")
            self._run_current_preprocessing_cycle()
            # Mettre à jour self.last_1m_kline_open_timestamp_ms après le fetch/preprocessing réussi
            if self.raw_1min_data_file_path.exists() and self.raw_1min_data_file_path.stat().st_size > 0:
                try:
                    # Lire seulement la dernière ligne pour obtenir le timestamp le plus récent
                    # Pour un CSV, cela peut être inefficace. Idéalement, update_pair_data_rest retournerait le dernier timestamp.
                    df_temp = pd.read_csv(self.raw_1min_data_file_path, usecols=['timestamp'])
                    if not df_temp.empty:
                        last_ts_in_file = pd.to_datetime(df_temp['timestamp'].iloc[-1], errors='coerce', utc=True)
                        if pd.notna(last_ts_in_file):
                            self.last_1m_kline_open_timestamp_ms = int(last_ts_in_file.timestamp() * 1000)
                except Exception as e_ts_update:
                    logger.error(f"{self.log_prefix}{log_ctx} Erreur lors de la mise à jour de last_1m_kline_open_timestamp_ms: {e_ts_update}")
            return True
        else:
            logger.debug(f"{self.log_prefix}{log_ctx} Aucune nouvelle donnée K-line 1-minute détectée ou récupérée.")
            return False

    def _check_and_process_orders_via_rest(self) -> None:
        """Vérifie et traite les ordres en attente via des appels REST."""
        # Logique similaire à celle de l'ancien manager.py, adaptée pour utiliser
        # self.execution_client et self.state_manager.
        # Vérifie les pending_entry_order, puis les pending_oco_order, puis les active_oco.
        # Appelle les méthodes de transition du state_manager en conséquence.
        logger.warning(f"{self.log_prefix} La logique détaillée de _check_and_process_orders_via_rest n'est pas entièrement réimplémentée dans cette version.")
        pass

    def _handle_status_1_no_trade(self) -> None:
        """Gère la logique de trading lorsque l'état est STATUT_1_NO_TRADE_NO_OCO."""
        # Logique de l'ancien manager.py :
        # - Vérifier le solde.
        # - Charger les données traitées (self.processed_data_file_path).
        # - Appeler self.strategy.generate_order_request().
        # - Si signal d'entrée, placer l'ordre via self.execution_client.place_margin_order().
        # - Mettre à jour self.state_manager (prepare_for_entry_order, record_placed_entry_order).
        logger.warning(f"{self.log_prefix} La logique détaillée de _handle_status_1_no_trade n'est pas entièrement réimplémentée dans cette version.")
        pass

    def _handle_status_2_oco_pending(self) -> None:
        """Gère la logique lorsque l'état est STATUT_2_ENTRY_FILLED_OCO_PENDING."""
        # Logique de l'ancien manager.py :
        # - Construire les paramètres OCO (ex: en appelant une méthode de self.strategy).
        # - Placer l'ordre OCO via self.execution_client.place_margin_oco_order().
        # - Mettre à jour self.state_manager (prepare_for_oco_order, puis attendre confirmation).
        logger.warning(f"{self.log_prefix} La logique détaillée de _handle_status_2_oco_pending n'est pas entièrement réimplémentée dans cette version.")
        pass

    def _handle_status_3_oco_active(self) -> None:
        """Gère la logique lorsque l'état est STATUT_3_OCO_ACTIVE."""
        # Principalement du logging ou des ajustements d'OCO (non prévus pour l'instant).
        # La détection de remplissage OCO se fait dans _check_and_process_orders_via_rest.
        logger.debug(f"{self.log_prefix} État STATUT_3_OCO_ACTIVE. Surveillance des ordres OCO (gérée par _check_and_process_orders_via_rest).")
        pass
        
    def _handle_trade_closure_and_loan_repayment(self,
                                                 closed_trade_state_snapshot: Dict[str, Any],
                                                 exit_reason: str,
                                                 closed_order_details_api: Optional[Dict[str, Any]]) -> None:
        """Gère la clôture d'un trade et le remboursement du prêt sur marge."""
        # Logique de l'ancien manager.py :
        # - Rembourser le prêt via self.execution_client.repay_margin_loan().
        # - Enregistrer le trade clôturé via self.state_manager.record_closed_trade().
        # - Transitionner vers STATUT_1 via self.state_manager.transition_to_status_1().
        logger.warning(f"{self.log_prefix} La logique détaillée de _handle_trade_closure_and_loan_repayment n'est pas entièrement réimplémentée.")
        # Exemple simplifié de la fin:
        # self.state_manager.record_closed_trade(...)
        # self.state_manager.transition_to_status_1(exit_reason, closed_order_details_api)
        # self.current_trade_cycle_id = None # Réinitialiser
        pass

    def _periodic_full_state_sync(self) -> None:
        """Effectue une synchronisation complète de l'état avec l'exchange."""
        logger.info(f"{self.log_prefix} Exécution de la synchronisation périodique complète de l'état...")
        try:
            self._determine_initial_status(is_periodic_sync=True)
            self.last_full_state_sync_time = datetime.now(timezone.utc)
            logger.info(f"{self.log_prefix} Synchronisation périodique complète de l'état terminée.")
        except Exception as e_sync:
            logger.error(f"{self.log_prefix} Erreur durant la synchronisation périodique complète de l'état : {e_sync}", exc_info=True)
            if self.state_manager:
                self.state_manager.set_last_error(f"Échec de la synchronisation périodique : {str(e_sync)[:100]}")

    def _log_trade_event(self, event_type: str, event_details: Dict[str, Any]) -> None:
        """
        Logue un événement de trading important.
        Utilise le logger dédié self.trade_event_logger.
        """
        if not self.trade_event_logger:
            logger.error(f"{self.log_prefix} trade_event_logger non initialisé. Impossible de loguer l'événement : {event_type}")
            return

        log_message = {"event_type": event_type, "details": event_details}
        try:
            # Utiliser json.dumps pour une sérialisation robuste des détails, même complexes.
            # Le logger de trade pourrait être configuré pour écrire du JSON directement.
            self.trade_event_logger.info(json.dumps(log_message, default=str))
        except Exception as e_log: # pylint: disable=broad-except
            # Fallback si json.dumps échoue (ne devrait pas avec default=str)
            self.trade_event_logger.error(f"Erreur lors de la sérialisation de l'événement de trade '{event_type}': {e_log}. Détails bruts: {event_details}")


    def run(self) -> None:
        """Boucle principale d'opération du LiveTradingManager."""
        logger.info(f"{self.log_prefix} Démarrage de la boucle principale du LiveTradingManager...")
        try:
            while not self.shutdown_event.is_set():
                current_time_utc = datetime.now(timezone.utc)
                
                # 1. Synchronisation Périodique Complète de l'État
                if self.last_full_state_sync_time is None or \
                   (current_time_utc - self.last_full_state_sync_time) >= timedelta(minutes=FULL_STATE_SYNC_INTERVAL_MINUTES):
                    self._periodic_full_state_sync()
                    if self.shutdown_event.is_set(): break

                # 2. Acquisition et Prétraitement des Données
                new_data_processed_this_cycle = self._check_new_1min_kline_and_trigger_preprocessing()
                time.sleep(API_CALL_DELAY_SECONDS) # Pause après opérations de données/fichiers
                if self.shutdown_event.is_set(): break

                # 3. Vérification et Traitement des Ordres (via REST)
                self._check_and_process_orders_via_rest()
                if self.shutdown_event.is_set(): break

                # Vérifier si les composants sont toujours valides
                if not all([self.state_manager, self.strategy, self.execution_client, self.trade_event_logger]):
                    logger.critical(f"{self.log_prefix} Un composant critique est devenu None. Arrêt du manager.")
                    self.shutdown_event.set()
                    break
                
                # 4. Logique de Trading Basée sur l'État Actuel
                current_status = self.state_manager.get_current_status()
                self.current_trade_cycle_id = self.state_manager.get_state_snapshot().get("current_trade_cycle_id") # Mettre à jour
                
                # Agir seulement si de nouvelles données ont été traitées OU si l'état nécessite une action immédiate
                # (ex: placer un OCO après une entrée remplie, même sans nouvelle bougie).
                needs_action_due_to_state = (current_status == STATUT_2_ENTRY_FILLED_OCO_PENDING and \
                                             not self.state_manager.get_state_snapshot().get("pending_oco_list_client_order_id"))
                
                if new_data_processed_this_cycle or needs_action_due_to_state:
                    logger.info(f"{self.log_prefix} Cycle d'action déclenché. Nouvelles données: {new_data_processed_this_cycle}, "
                                f"Action d'état requise: {needs_action_due_to_state}. Statut actuel: {current_status}")
                    if current_status == STATUT_1_NO_TRADE_NO_OCO:
                        # Vérifier si un ordre d'entrée n'est pas déjà en attente de confirmation de placement
                        if not self.state_manager.get_state_snapshot().get("pending_entry_order_id_api"):
                            self._handle_status_1_no_trade()
                        else:
                            logger.debug(f"{self.log_prefix} Statut 1 mais un ordre d'entrée ({self.state_manager.get_state_snapshot().get('pending_entry_order_id_api')}) "
                                         "est en attente de confirmation de remplissage. Pas de nouvelle action d'entrée.")
                    elif current_status == STATUT_2_ENTRY_FILLED_OCO_PENDING:
                        # Vérifier si un OCO n'est pas déjà en attente de confirmation de placement
                        if not self.state_manager.get_state_snapshot().get("pending_oco_list_client_order_id"):
                             self._handle_status_2_oco_pending()
                        else:
                            logger.debug(f"{self.log_prefix} Statut 2 mais un ordre OCO ({self.state_manager.get_state_snapshot().get('pending_oco_list_client_order_id')}) "
                                         "est en attente de confirmation de placement. Pas de nouvelle action OCO.")
                    elif current_status == STATUT_3_OCO_ACTIVE:
                        self._handle_status_3_oco_active()
                    else:
                        logger.warning(f"{self.log_prefix} Statut inconnu ou non géré dans la boucle principale : {current_status}")
                else:
                    logger.debug(f"{self.log_prefix} Pas de nouvelles données traitées et pas d'action d'état immédiate requise. "
                                 f"Statut: {current_status}. Cycle ID: {self.current_trade_cycle_id}")

                # Pause avant le prochain cycle de la boucle principale
                if self.shutdown_event.wait(timeout=MAIN_LOOP_SLEEP_SECONDS):
                    logger.info(f"{self.log_prefix} Événement d'arrêt reçu pendant la pause. Sortie de la boucle.")
                    break
        
        except Exception as e_loop: # pylint: disable=broad-except
            logger.critical(f"{self.log_prefix} ERREUR CRITIQUE dans la boucle principale du manager : {e_loop}", exc_info=True)
            if self.state_manager:
                self.state_manager.set_last_error(f"Erreur critique de boucle : {str(e_loop)[:250]}")
        finally:
            self.stop_trading() # Assurer l'appel à stop_trading à la fin

    def stop_trading(self) -> None:
        """Arrête proprement le LiveTradingManager."""
        if self.shutdown_event.is_set():
            logger.info(f"{self.log_prefix} Événement d'arrêt déjà positionné.")
            return # Déjà en cours d'arrêt ou arrêté

        logger.info(f"{self.log_prefix} Arrêt du LiveTradingManager demandé...")
        self.shutdown_event.set()

        # Actions de nettoyage optionnelles ici, si nécessaire avant que le thread ne se termine.
        # Par exemple, tenter une dernière annulation d'ordres ou un remboursement de prêt.
        # Cependant, cela peut être complexe à gérer de manière synchrone ici.
        # Il est souvent préférable que la boucle `run` gère son propre état final.

        if self.execution_client and hasattr(self.execution_client, 'close') and callable(self.execution_client.close):
            try:
                self.execution_client.close()
                logger.info(f"{self.log_prefix} Session OrderExecutionClient fermée.")
            except Exception as e_close_client: # pylint: disable=broad-except
                logger.error(f"{self.log_prefix} Erreur lors de la fermeture de OrderExecutionClient : {e_close_client}")
        
        logger.info(f"{self.log_prefix} LiveTradingManager arrêté.")

