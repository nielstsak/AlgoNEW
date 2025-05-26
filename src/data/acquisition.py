# src/data/acquisition.py
"""
Ce module est responsable de la récupération des données historiques de K-lines
1-minute depuis Binance, de leur nettoyage initial (y compris le calcul des
volumes Taker vendeurs) et de leur sauvegarde aux formats CSV et Parquet.
"""
import logging
import os
import sys # Ajouté pour sys.exit en cas d'erreur critique d'import
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, List, Union, TYPE_CHECKING
from pathlib import Path

import pandas as pd
import numpy as np

# Tentative d'importation du client Binance et des exceptions
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    BINANCE_CLIENT_AVAILABLE = True
    # KLINE_INTERVAL_1MINUTE est une constante de classe de binance.Client
    KLINE_INTERVAL_1MINUTE = Client.KLINE_INTERVAL_1MINUTE
except ImportError:
    BINANCE_CLIENT_AVAILABLE = False
    KLINE_INTERVAL_1MINUTE = "1m" # Fallback si la bibliothèque n'est pas installée
    # Définir des exceptions factices pour permettre au module de se charger
    # mais les opérations réelles échoueront.
    class BinanceAPIException(Exception): pass
    class BinanceRequestException(Exception): pass
    class Client: # Client factice pour le typage et l'import initial
        def __init__(self, api_key=None, api_secret=None, tld='com', testnet=False):
            pass
        def get_historical_klines(self, symbol, interval, start_str, end_str=None, limit=500) -> list: # type: ignore
            return []
        def ping(self) -> dict: # type: ignore
            return {}

    # Logger un avertissement critique si la bibliothèque est manquante
    logging.basicConfig(level=logging.CRITICAL) # Configurer un logger basique pour ce message
    logging.getLogger(__name__).critical(
        "ÉCHEC CRITIQUE : La bibliothèque python-binance n'a pas été trouvée ou n'a pas pu être importée. "
        "La récupération des données historiques ne fonctionnera pas. "
        "Veuillez installer python-binance (ex: pip install python-binance)."
    )
    # Optionnel : sys.exit(1) ici si la bibliothèque est absolument critique pour toute utilisation du module.
    # Pour l'instant, on permet au module de se charger pour que d'autres parties puissent être testées.

if TYPE_CHECKING:
    from src.config.loader import AppConfig # Pour le typage de AppConfig

logger = logging.getLogger(__name__)

# Colonnes telles que reçues de l'API Binance pour les klines
BINANCE_KLINES_COLS: List[str] = [
    'kline_open_time', 'open', 'high', 'low', 'close', 'volume',
    'kline_close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

# Colonnes OHLCV de base que nous voulons assurer et typer correctement
OUTPUT_OHLCV_COLS: List[str] = ['open', 'high', 'low', 'close', 'volume']

# Colonnes finales dans les fichiers de sortie, incluant 'timestamp' et les données Taker calculées
FINAL_OUTPUT_COLS: List[str] = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', # OHLCV de base + timestamp
    'quote_asset_volume', 'number_of_trades',             # Métadonnées de trade
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', # Volumes Taker acheteur
    'taker_sell_base_asset_volume', 'taker_sell_quote_asset_volume' # Volumes Taker vendeur (calculés)
]

# Constantes pour les limites de l'API et les re-essais
API_SPOT_MARGIN_LIMIT_PER_REQUEST = 1000
API_FUTURES_LIMIT_PER_REQUEST = 1500 # Pour référence, si le client était pour les futures
MAX_API_CALL_RETRIES = 5
API_RETRY_BASE_DELAY_SECONDS = 3 # Délai de base pour le backoff exponentiel


def _parse_and_clean_binance_klines(klines_data: List[List[Any]], pair_symbol: str) -> pd.DataFrame:
    """
    Parse les données K-lines brutes de l'API Binance, convertit les types,
    calcule les volumes Taker vendeurs, et sélectionne/ordonne les colonnes finales.

    Args:
        klines_data (List[List[Any]]): Données brutes de l'API Binance (liste de listes).
        pair_symbol (str): Le symbole de la paire de trading (pour le logging).

    Returns:
        pd.DataFrame: Un DataFrame pandas nettoyé avec les colonnes définies
                      dans FINAL_OUTPUT_COLS. Retourne un DataFrame vide si
                      les données d'entrée sont vides ou si une erreur critique se produit.
    """
    log_prefix = f"[{pair_symbol}][_parseClean]"
    if not klines_data:
        logger.warning(f"{log_prefix} Aucune donnée K-line fournie pour le parsing.")
        return pd.DataFrame(columns=FINAL_OUTPUT_COLS)

    df = pd.DataFrame(klines_data, columns=BINANCE_KLINES_COLS)
    logger.debug(f"{log_prefix} Shape initial du DataFrame brut : {df.shape}")

    # Conversion de kline_open_time en 'timestamp' (datetime UTC)
    df['timestamp'] = pd.to_datetime(df['kline_open_time'], unit='ms', utc=True, errors='coerce')

    # Conversion des colonnes numériques
    numeric_cols_to_convert = OUTPUT_OHLCV_COLS + [
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
    ]
    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.warning(f"{log_prefix} Colonne numérique attendue '{col}' non trouvée dans les données brutes. Sera initialisée à NaN.")
            df[col] = np.nan

    # Calcul des volumes Taker vendeurs
    # taker_sell_base_volume = total_base_volume - taker_buy_base_volume
    # taker_sell_quote_volume = total_quote_volume - taker_buy_quote_volume
    if 'volume' in df.columns and 'taker_buy_base_asset_volume' in df.columns:
        # La soustraction propage les NaNs correctement.
        df['taker_sell_base_asset_volume'] = df['volume'] - df['taker_buy_base_asset_volume']
    else:
        logger.warning(f"{log_prefix} Colonnes sources ('volume' ou 'taker_buy_base_asset_volume') manquantes "
                       "pour calculer 'taker_sell_base_asset_volume'. Sera NaN.")
        df['taker_sell_base_asset_volume'] = np.nan

    if 'quote_asset_volume' in df.columns and 'taker_buy_quote_asset_volume' in df.columns:
        df['taker_sell_quote_asset_volume'] = df['quote_asset_volume'] - df['taker_buy_quote_asset_volume']
    else:
        logger.warning(f"{log_prefix} Colonnes sources ('quote_asset_volume' ou 'taker_buy_quote_asset_volume') manquantes "
                       "pour calculer 'taker_sell_quote_asset_volume'. Sera NaN.")
        df['taker_sell_quote_asset_volume'] = np.nan

    # S'assurer que toutes les colonnes de FINAL_OUTPUT_COLS sont présentes
    for col in FINAL_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan # Ajouter la colonne avec des NaNs si elle manque
            logger.debug(f"{log_prefix} Colonne de sortie finale '{col}' ajoutée avec NaNs car non présente.")

    df = df[FINAL_OUTPUT_COLS] # Sélectionner et ordonner les colonnes

    # Supprimer les lignes avec des NaNs dans les colonnes essentielles (timestamp + OHLCV)
    essential_cols_for_dropna = ['timestamp'] + OUTPUT_OHLCV_COLS
    rows_before_dropna = len(df)
    df.dropna(subset=essential_cols_for_dropna, how='any', inplace=True)
    if rows_before_dropna > len(df):
        logger.debug(f"{log_prefix} {rows_before_dropna - len(df)} lignes supprimées en raison de NaNs dans les colonnes essentielles.")

    if df.empty:
        logger.warning(f"{log_prefix} DataFrame devenu vide après suppression des NaNs dans les colonnes essentielles.")
        return df # Retourner le DataFrame vide

    # Trier par timestamp et supprimer les doublons, en gardant la dernière occurrence
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    if df.duplicated(subset=['timestamp']).any():
        rows_before_drop_duplicates = len(df)
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        logger.debug(f"{log_prefix} {rows_before_drop_duplicates - len(df)} entrées de timestamp dupliquées supprimées (conservation de la dernière).")

    # Réinitialiser l'index après le tri et la déduplication
    df = df.reset_index(drop=True)
    logger.debug(f"{log_prefix} Données parsées et nettoyées. Shape final : {df.shape}")
    return df


def _fetch_single_pair_1min_history_and_clean(
    client: Client,
    pair: str,
    start_date_str: str,
    end_date_str: Optional[str] = None,
    asset_type_for_limit: str = "SPOT", # "SPOT", "MARGIN", "FUTURES_UM" (pour déterminer la limite de l'API)
    api_batch_limit: int = API_SPOT_MARGIN_LIMIT_PER_REQUEST # Limite par requête API
) -> Optional[pd.DataFrame]:
    """
    Récupère l'historique complet des K-lines 1-minute pour une seule paire en utilisant
    la pagination, puis nettoie les données.

    Args:
        client (Client): Client API Binance initialisé.
        pair (str): Symbole de la paire de trading (ex: "BTCUSDT").
        start_date_str (str): Date de début pour la récupération (format accepté par Binance API).
        end_date_str (Optional[str]): Date de fin optionnelle. Si None, récupère jusqu'au
                                      moment présent (ou la dernière donnée disponible).
        asset_type_for_limit (str): Type d'actif/marché pour déterminer la limite de l'API
                                    si `api_batch_limit` n'est pas explicitement fourni.
                                    Principalement "SPOT" ou "FUTURES_UM".
        api_batch_limit (int): Nombre maximal de klines à récupérer par appel API.

    Returns:
        Optional[pd.DataFrame]: Un DataFrame pandas nettoyé contenant les données historiques,
                                ou None si la récupération échoue de manière critique.
                                Retourne un DataFrame vide si aucune donnée n'est trouvée pour la période.
    """
    log_prefix = f"[{pair}][{asset_type_for_limit}]" # asset_type_for_limit est plus pour le contexte de logging ici
    logger.info(f"{log_prefix} Récupération de l'historique 1-minute de {start_date_str} à {end_date_str or 'maintenant'}...")

    all_klines_raw_data: List[List[Any]] = []
    current_start_fetch_str = start_date_str # Timestamp de début pour l'appel API actuel

    # Déterminer la limite de batch correcte si non fournie explicitement
    if api_batch_limit <= 0: # Si une valeur invalide est passée
        if "FUTURES" in asset_type_for_limit.upper():
            api_batch_limit = API_FUTURES_LIMIT_PER_REQUEST
        else: # SPOT, MARGIN
            api_batch_limit = API_SPOT_MARGIN_LIMIT_PER_REQUEST
        logger.debug(f"{log_prefix} api_batch_limit ajusté à {api_batch_limit} basé sur asset_type_for_limit.")


    while True:
        klines_batch_raw: List[List[Any]] = []
        fetch_batch_successful = False
        
        for attempt in range(MAX_API_CALL_RETRIES):
            try:
                logger.debug(f"{log_prefix} Requête API (limite: {api_batch_limit}) à partir de : {current_start_fetch_str}, tentative {attempt + 1}/{MAX_API_CALL_RETRIES}")
                
                klines_batch_raw = client.get_historical_klines( # type: ignore
                    pair,
                    KLINE_INTERVAL_1MINUTE,
                    current_start_fetch_str,
                    end_str=end_date_str, # Passer end_date_str à chaque appel
                    limit=api_batch_limit
                )
                logger.debug(f"{log_prefix} Reçu {len(klines_batch_raw)} klines dans le batch à partir de {current_start_fetch_str}.")
                fetch_batch_successful = True
                break # Sortir de la boucle de re-essai si succès
            except BinanceAPIException as e_api:
                logger.error(f"{log_prefix} Exception API Binance (tentative {attempt + 1}) pour start_str {current_start_fetch_str}: Code={e_api.code}, Msg='{e_api.message}'")
                if e_api.code == -1121: # Symbole invalide
                    logger.error(f"{log_prefix} Symbole '{pair}' invalide pour le point d'accès API. Abandon de la récupération pour cette paire.")
                    return pd.DataFrame(columns=FINAL_OUTPUT_COLS) # Retourner DF vide
                
                # Gérer les rate limits (429, 418) ou IP ban (-1003)
                if e_api.status_code in [429, 418] or e_api.code == -1003:
                    if attempt < MAX_API_CALL_RETRIES - 1:
                        sleep_duration = API_RETRY_BASE_DELAY_SECONDS * (2 ** attempt) # Backoff exponentiel
                        logger.warning(f"{log_prefix} Rate limit atteint (HTTP {e_api.status_code}, Code {e_api.code}). "
                                       f"Nouvelle tentative dans {sleep_duration:.2f}s...")
                        time.sleep(sleep_duration)
                    else:
                        logger.error(f"{log_prefix} Nombre maximum de re-essais atteint après rate limit pour start_str {current_start_fetch_str}.")
                        break # Sortir de la boucle de re-essai
                elif attempt < MAX_API_CALL_RETRIES - 1: # Pour d'autres erreurs API retryables
                    time.sleep(API_RETRY_BASE_DELAY_SECONDS)
                # Si c'est la dernière tentative, l'erreur sera gérée après la boucle
            except BinanceRequestException as e_req: # Erreur dans la requête, généralement non retryable
                logger.error(f"{log_prefix} Exception de Requête Binance pour start_str {current_start_fetch_str}: {e_req}. Abandon pour cette paire.")
                return pd.DataFrame(columns=FINAL_OUTPUT_COLS) # Retourner DF vide
            except Exception as e_unexp: # pylint: disable=broad-except
                logger.error(f"{log_prefix} Erreur inattendue (tentative {attempt + 1}) pour start_str {current_start_fetch_str}: {e_unexp}", exc_info=True)
                if attempt < MAX_API_CALL_RETRIES - 1:
                    time.sleep(API_RETRY_BASE_DELAY_SECONDS * (1.5 ** attempt)) # Backoff plus doux
                # Si c'est la dernière tentative, l'erreur sera gérée après la boucle
        
        if not fetch_batch_successful or not klines_batch_raw:
            if not fetch_batch_successful: # Si toutes les tentatives ont échoué
                 logger.error(f"{log_prefix} Échec de la récupération du batch après {MAX_API_CALL_RETRIES} tentatives pour start_str {current_start_fetch_str}.")
            else: # fetch_successful est True, mais klines_batch_raw est vide
                 logger.info(f"{log_prefix} Plus aucune K-line reçue (batch vide) à partir de {current_start_fetch_str}. Fin de la pagination.")
            break # Sortir de la boucle while principale

        all_klines_raw_data.extend(klines_batch_raw)
        
        # Préparer le timestamp de début pour le prochain appel API
        try:
            # Le dernier kline du batch actuel est klines_batch_raw[-1].
            # Son timestamp d'ouverture est klines_batch_raw[-1][0].
            # Le prochain appel doit commencer APRÈS ce kline.
            last_kline_open_time_ms = int(klines_batch_raw[-1][0])
            next_start_fetch_ms = last_kline_open_time_ms + 60000 # Ajouter 1 minute (en ms)
            current_start_fetch_str = str(next_start_fetch_ms)
        except (IndexError, TypeError, ValueError) as e_ts_next:
            logger.error(f"{log_prefix} Erreur lors du traitement du timestamp pour le prochain batch : {e_ts_next}. "
                         f"Dernier kline du batch (si existant) : {klines_batch_raw[-1] if klines_batch_raw else 'batch vide'}. "
                         "Fin de la pagination.")
            break

        # Vérifier si la fin de la période demandée (end_date_str) est atteinte
        if end_date_str:
            try:
                # Convertir end_date_str en timestamp ms pour comparaison
                # S'assurer que end_date_str est parsable et en UTC
                end_datetime_utc = pd.to_datetime(end_date_str, errors='raise', utc=True)
                end_timestamp_target_ms = int(end_datetime_utc.timestamp() * 1000)
                
                if last_kline_open_time_ms >= end_timestamp_target_ms:
                    logger.info(f"{log_prefix} Date de fin '{end_date_str}' (timestamp {end_timestamp_target_ms}) atteinte ou dépassée. "
                                f"Dernier kline ouvert à {last_kline_open_time_ms} ({pd.to_datetime(last_kline_open_time_ms, unit='ms', utc=True)}). "
                                "Fin de la pagination.")
                    break
            except Exception as e_parse_end_dt: # pylint: disable=broad-except
                 logger.error(f"{log_prefix} Impossible de parser end_date_str '{end_date_str}' ou de comparer les timestamps : {e_parse_end_dt}. "
                              "La pagination pourrait continuer indéfiniment si non gérée. Arrêt de la pagination par sécurité.")
                 break # Arrêter par sécurité

        # Respecter les limites de taux de l'API
        time.sleep(0.2) # Petit délai entre les appels paginés

    if not all_klines_raw_data:
        logger.warning(f"{log_prefix} Aucune K-line historique 1-minute n'a été récupérée au total pour la période demandée.")
        return pd.DataFrame(columns=FINAL_OUTPUT_COLS) # Retourner un DF vide avec les bonnes colonnes

    logger.info(f"{log_prefix} Récupération de {len(all_klines_raw_data)} K-lines brutes 1-minute terminée. Parsing et nettoyage...")
    df_cleaned_all = _parse_and_clean_binance_klines(all_klines_raw_data, pair)

    # Filtrage final par end_date_str si celui-ci a été fourni, car l'API peut retourner des klines
    # dont le open_time est avant end_date_str mais le close_time est après.
    # On veut les klines dont l'open_time est < end_date_str.
    if end_date_str and not df_cleaned_all.empty:
        try:
            end_datetime_filter_utc = pd.to_datetime(end_date_str, errors='raise', utc=True)
            # S'assurer que la colonne 'timestamp' est bien de type datetime UTC
            if not pd.api.types.is_datetime64_any_dtype(df_cleaned_all['timestamp']) or \
               df_cleaned_all['timestamp'].dt.tz != timezone.utc:
                 df_cleaned_all['timestamp'] = pd.to_datetime(df_cleaned_all['timestamp'], errors='coerce', utc=True)
            
            original_rows_before_final_filter = len(df_cleaned_all)
            df_cleaned_all = df_cleaned_all[df_cleaned_all['timestamp'] < end_datetime_filter_utc]
            if len(df_cleaned_all) < original_rows_before_final_filter:
                logger.debug(f"{log_prefix} {original_rows_before_final_filter - len(df_cleaned_all)} lignes filtrées "
                             f"car leur timestamp d'ouverture était >= à la date de fin spécifiée {end_date_str}.")
        except Exception as e_filter_final: # pylint: disable=broad-except
            logger.error(f"{log_prefix} Erreur lors du filtrage final par date de fin '{end_date_str}': {e_filter_final}.")


    logger.info(f"{log_prefix} Récupération et nettoyage des données 1-minute terminés. "
                f"Nombre final de lignes : {len(df_cleaned_all)}")
    return df_cleaned_all


def fetch_all_historical_data(config: 'AppConfig') -> str:
    """
    Orchestre la récupération des données historiques 1-minute pour toutes les paires
    configurées. Sauvegarde les données nettoyées aux formats CSV et Parquet.

    Args:
        config (AppConfig): L'objet de configuration global de l'application.

    Returns:
        str: Chemin vers le répertoire de run contenant les fichiers CSV "bruts-nettoyés"
             pour cette exécution. Lève une exception en cas d'échec critique.
    """
    main_log_prefix = "[FetchAllHistorical]"
    logger.info(f"{main_log_prefix} --- Démarrage du processus de récupération et nettoyage des données historiques (K-lines 1-minute) ---")

    if not BINANCE_CLIENT_AVAILABLE:
        msg = "La bibliothèque python-binance n'est pas disponible. Impossible de récupérer les données historiques."
        logger.critical(f"{main_log_prefix} {msg}")
        raise ImportError(msg)

    # Sélectionner les clés API. Priorité au premier compte Binance non-testnet, sinon premier compte Binance.
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    selected_account_alias: Optional[str] = None

    if config.accounts_config and config.api_keys and config.api_keys.credentials:
        # Chercher un compte Binance live non-testnet
        for acc_cfg in config.accounts_config:
            if acc_cfg.exchange.lower() == "binance" and not acc_cfg.is_testnet:
                creds = config.api_keys.credentials.get(acc_cfg.account_alias)
                if creds and creds[0] and creds[1]:
                    api_key, api_secret = creds
                    selected_account_alias = acc_cfg.account_alias
                    logger.info(f"{main_log_prefix} Utilisation des clés API du compte live non-testnet : '{selected_account_alias}'.")
                    break
        # Si non trouvé, chercher n'importe quel compte Binance (y compris testnet)
        if not selected_account_alias:
            for acc_cfg in config.accounts_config:
                if acc_cfg.exchange.lower() == "binance":
                    creds = config.api_keys.credentials.get(acc_cfg.account_alias)
                    if creds and creds[0] and creds[1]:
                        api_key, api_secret = creds
                        selected_account_alias = acc_cfg.account_alias
                        logger.warning(f"{main_log_prefix} Aucun compte Binance live non-testnet avec clés complètes trouvé. "
                                       f"Utilisation des clés du compte (potentiellement testnet) : '{selected_account_alias}'.")
                        break
    
    # Fallback aux variables d'environnement si aucune clé n'a été trouvée via la config
    if not selected_account_alias:
        logger.warning(f"{main_log_prefix} Aucune clé API trouvée via la configuration des comptes. "
                       "Tentative de chargement depuis les variables d'environnement BINANCE_API_KEY et BINANCE_SECRET_KEY.")
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET_KEY")
        if api_key and api_secret:
            selected_account_alias = "ENV_DEFAULT"
            logger.info(f"{main_log_prefix} Utilisation des clés API des variables d'environnement.")


    if not api_key or not api_secret:
        msg = "Clés API Binance (key ou secret) manquantes. Vérifiez la configuration des comptes ou les variables d'environnement."
        logger.critical(f"{main_log_prefix} {msg}")
        raise ValueError(msg)

    try:
        # Le client est initialisé pour le domaine principal de Binance (tld='com').
        # Pour le testnet, il faudrait `testnet=True` si les clés sont pour le testnet.
        # On suppose ici que les clés sélectionnées correspondent à l'environnement principal.
        client = Client(api_key, api_secret)
        client.ping() # Test de connexion
        logger.info(f"{main_log_prefix} Client API Binance initialisé et connexion testée avec succès (compte utilisé pour init: '{selected_account_alias}').")
    except Exception as client_err:
        logger.critical(f"{main_log_prefix} Échec de l'initialisation ou de la connexion du client API Binance : {client_err}", exc_info=True)
        raise ConnectionError(f"Échec du client API Binance : {client_err}")


    pairs_to_fetch = config.data_config.assets_and_timeframes.pairs
    start_date_str = config.data_config.historical_period.start_date
    end_date_config_str = config.data_config.historical_period.end_date

    end_date_to_use_for_api: Optional[str]
    if end_date_config_str is None or str(end_date_config_str).lower() == "now" or str(end_date_config_str).strip() == "":
        # Pour get_historical_klines, end_str=None signifie "jusqu'aux données les plus récentes".
        end_date_to_use_for_api = None
        logger.info(f"{main_log_prefix} Date de fin non spécifiée ou 'now', récupération jusqu'aux données les plus récentes.")
    else:
        end_date_to_use_for_api = str(end_date_config_str)
        logger.info(f"{main_log_prefix} Date de fin pour la récupération API : {end_date_to_use_for_api}")


    asset_type_context = config.data_config.source_details.asset_type # Utilisé pour le logging et potentiellement la limite de batch
    max_workers = config.data_config.fetching_options.max_workers
    api_batch_limit_from_config = config.data_config.fetching_options.batch_size

    run_timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Utiliser project_root pour construire les chemins absolus
    project_root_path = Path(config.project_root) if config.project_root else Path(".").resolve()

    raw_data_base_path_str = config.global_config.paths.data_historical_raw
    raw_run_output_dir = project_root_path / raw_data_base_path_str / run_timestamp_str
    
    cleaned_data_output_path_str = config.global_config.paths.data_historical_processed_cleaned
    cleaned_parquet_output_dir = project_root_path / cleaned_data_output_path_str

    try:
        raw_run_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"{main_log_prefix} Répertoire de run pour les CSV bruts (nettoyés) : {raw_run_output_dir}")
        cleaned_parquet_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"{main_log_prefix} Répertoire pour les fichiers Parquet nettoyés : {cleaned_parquet_output_dir}")
    except OSError as e_mkdir:
        logger.critical(f"{main_log_prefix} Échec de la création des répertoires de sortie : {e_mkdir}", exc_info=True)
        raise

    valid_pairs_to_fetch: List[str] = [p for p in pairs_to_fetch if p and isinstance(p, str)]
    if not valid_pairs_to_fetch:
        logger.warning(f"{main_log_prefix} Aucune paire valide à récupérer dans la configuration. Arrêt.")
        return str(raw_run_output_dir) # Retourner le chemin du répertoire de run même s'il est vide

    results_dataframes: Dict[str, Optional[pd.DataFrame]] = {}

    logger.info(f"{main_log_prefix} Démarrage du téléchargement parallèle et nettoyage pour {len(valid_pairs_to_fetch)} paire(s) "
                f"en utilisant jusqu'à {max_workers} workers.")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair_map = {
            executor.submit(
                _fetch_single_pair_1min_history_and_clean,
                client,
                pair,
                start_date_str,
                end_date_to_use_for_api,
                asset_type_context,
                api_batch_limit_from_config
            ): pair
            for pair in valid_pairs_to_fetch
        }

        for future in as_completed(future_to_pair_map):
            pair_symbol_completed = future_to_pair_map[future]
            try:
                result_df = future.result() # Peut lever une exception si la tâche a échoué
                results_dataframes[pair_symbol_completed] = result_df

                if result_df is not None and not result_df.empty:
                    # Sauvegarde CSV dans le répertoire de run spécifique
                    raw_csv_file_name = f"{pair_symbol_completed}_1min_raw_with_taker.csv"
                    raw_csv_file_path = raw_run_output_dir / raw_csv_file_name
                    
                    # Sauvegarde Parquet dans le répertoire "cleaned" global
                    cleaned_parquet_file_name = f"{pair_symbol_completed}_1min_cleaned_with_taker.parquet"
                    cleaned_parquet_file_path = cleaned_parquet_output_dir / cleaned_parquet_file_name
                    
                    try:
                        result_df.to_csv(raw_csv_file_path, index=False)
                        logger.info(f"[{pair_symbol_completed}] Données 1-min nettoyées (avec taker) sauvegardées en CSV : {raw_csv_file_path}")
                        
                        result_df.to_parquet(cleaned_parquet_file_path, index=False, engine='pyarrow')
                        logger.info(f"[{pair_symbol_completed}] Données 1-min nettoyées (avec taker) sauvegardées en Parquet : {cleaned_parquet_file_path}")
                    except IOError as e_io:
                        logger.error(f"[{pair_symbol_completed}] Échec de la sauvegarde du/des fichier(s) de données : {e_io}", exc_info=True)
                    except Exception as e_save: # pylint: disable=broad-except
                        logger.error(f"[{pair_symbol_completed}] Erreur inattendue lors de la sauvegarde du/des fichier(s) de données : {e_save}", exc_info=True)

                elif result_df is not None and result_df.empty:
                    logger.warning(f"[{pair_symbol_completed}] Aucune donnée 1-minute retournée ou traitée. Fichiers non sauvegardés.")
                else: # result_df is None
                    logger.error(f"[{pair_symbol_completed}] La tâche de récupération et nettoyage a échoué (retourné None). Fichiers non sauvegardés.")
            except Exception as exc_future: # pylint: disable=broad-except
                logger.error(f"La tâche pour {pair_symbol_completed} a généré une exception : {exc_future}", exc_info=True)
                results_dataframes[pair_symbol_completed] = None # Marquer comme échoué

    successful_fetches_count = sum(1 for df in results_dataframes.values() if df is not None and not df.empty)
    failed_fetches_count = len(valid_pairs_to_fetch) - successful_fetches_count
    logger.info(f"{main_log_prefix} Récupération et nettoyage des données historiques 1-minute terminés.")
    logger.info(f"{main_log_prefix} Paires traitées avec succès : {successful_fetches_count}")
    logger.info(f"{main_log_prefix} Paires échouées ou sans données : {failed_fetches_count}")

    if successful_fetches_count == 0 and valid_pairs_to_fetch:
        logger.critical(f"{main_log_prefix} ÉCHEC CRITIQUE : Aucune donnée historique 1-minute n'a pu être récupérée "
                        "et nettoyée avec succès pour aucune des paires configurées.")
        # On pourrait lever une exception ici si c'est bloquant pour la suite.

    return str(raw_run_output_dir.resolve())


if __name__ == '__main__':
    # Configuration du logging pour exécution directe (tests)
    # Note: Dans une exécution normale via run_fetch_data.py, load_all_configs
    # s'occuperait de la configuration du logging.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger.info("Exécution de src/data/acquisition.py directement pour tests...")
    
    # Pour tester, il faudrait créer un objet AppConfig factice ou charger une config réelle.
    # Exemple (nécessiterait que les fichiers de config existent et que .env soit configuré) :
    # try:
    #     config_test_root = Path(__file__).resolve().parent.parent.parent # Supposer la racine du projet
    #     app_config_instance = load_all_configs(project_root=str(config_test_root))
    #     logger.info("AppConfig chargée pour le test.")
    #     output_dir = fetch_all_historical_data(app_config_instance)
    #     logger.info(f"Test de fetch_all_historical_data terminé. Données brutes (nettoyées) dans : {output_dir}")
    # except ImportError as e_imp_test:
    #     logger.critical(f"Erreur d'importation lors du test direct (manque peut-être src.config.loader ou ses dépendances): {e_imp_test}")
    # except FileNotFoundError as e_fnf_test:
    #     logger.critical(f"Fichier de configuration non trouvé lors du test direct: {e_fnf_test}")
    # except ValueError as e_val_test:
    #     logger.critical(f"Erreur de valeur (ex: clés API manquantes) lors du test direct: {e_val_test}")
    # except Exception as e_test_main: # pylint: disable=broad-except
    #     logger.critical(f"Erreur inattendue lors du test direct de acquisition.py: {e_test_main}", exc_info=True)
    
    logger.info("Fin du test direct de src/data/acquisition.py.")
    # Note: Un test réel nécessiterait des clés API valides (même pour testnet) et une connexion internet.
    # Le placeholder ci-dessus est juste pour illustrer comment on pourrait l'appeler.
    pass
