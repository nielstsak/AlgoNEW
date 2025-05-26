# src/data/acquisition_live.py
"""
Ce module est responsable de la récupération et de la mise à jour des données
de K-lines 1-minute pour le trading en direct, en utilisant l'API REST de Binance.
Il gère l'initialisation des fichiers de données brutes pour les paires configurées.
"""
import json
import logging
import os # Maintenu pour une éventuelle utilisation, bien que pathlib soit préféré
import sys # Pour la configuration du logging de fallback
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING

import pandas as pd
import numpy as np
import requests # Pour les appels REST directs

if TYPE_CHECKING:
    # Utilisation de AppConfig pour une configuration plus complète si disponible
    from src.config.loader import AppConfig
    from src.config.definitions import LiveFetchConfig, PathsConfig, GlobalLiveSettings

# KLINE_INTERVAL_1MINUTE peut être défini ici ou importé depuis binance.client
# Pour ce module qui utilise REST directement, nous définissons la constante.
KLINE_INTERVAL_1MINUTE = "1m"

logger = logging.getLogger(__name__)

# Colonnes telles que reçues de l'API Binance pour les klines
BINANCE_KLINE_COLUMNS: List[str] = [
    'kline_open_time', 'open', 'high', 'low', 'close', 'volume',
    'kline_close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

# Colonnes finales dans les fichiers de sortie, incluant le timestamp et les données Taker calculées.
# Standardisées pour les fichiers bruts (historiques ou live).
BASE_COLUMNS_RAW: List[str] = [
    'timestamp', 'kline_close_time', 'open', 'high', 'low', 'close', 'volume',
    'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'taker_sell_base_asset_volume', 'taker_sell_quote_asset_volume' # Calculées
]

# Limites de l'API Binance pour les klines par requête
API_BATCH_LIMIT_SPOT_MARGIN = 1000
API_BATCH_LIMIT_FUTURES = 1500 # Pour USD-M Futures
API_BATCH_LIMIT_FUTURES_COIN_M = 500 # Pour COIN-M Futures

# Constantes pour les re-essais API
MAX_API_FETCH_RETRIES = 3
API_FETCH_RETRY_DELAY_SECONDS = 2.0


def get_binance_klines_rest(
    symbol: str,
    limit: int = 100,
    account_type: str = "SPOT", # Ou "MARGIN", "FUTURES_USDT", "FUTURES_COIN"
    end_timestamp_ms: Optional[int] = None,
    start_timestamp_ms: Optional[int] = None # Ajouté pour plus de flexibilité
) -> Optional[pd.DataFrame]:
    """
    Récupère les K-lines 1-minute via l'API REST publique de Binance.
    Cette fonction ne nécessite pas de clés API car elle accède à des points d'accès publics.

    Args:
        symbol (str): Symbole de la paire de trading (ex: "BTCUSDT").
        limit (int): Nombre de K-lines à récupérer (max dépendant de l'endpoint).
        account_type (str): Type de compte/marché pour déterminer l'URL de base de l'API
                            et la limite de batch. Options : "SPOT", "MARGIN",
                            "FUTURES_USDT" (pour USD-M), "FUTURES_COIN" (pour COIN-M).
        end_timestamp_ms (Optional[int]): Timestamp de fin (en millisecondes, exclusif)
                                          pour récupérer les K-lines *avant* ce temps.
                                          Si None, récupère les plus récentes.
        start_timestamp_ms (Optional[int]): Timestamp de début (en millisecondes, inclusif)
                                            pour récupérer les K-lines *à partir de* ce temps.

    Returns:
        Optional[pd.DataFrame]: DataFrame pandas avec les K-lines, ou None en cas d'erreur critique.
                                Retourne un DataFrame vide si aucune donnée n'est trouvée pour les paramètres.
    """
    actual_fetch_interval = KLINE_INTERVAL_1MINUTE
    log_ctx = f"[{symbol.upper()}][{actual_fetch_interval}][{account_type.upper()}]"

    if end_timestamp_ms:
        logger.debug(f"{log_ctx} Requête de {limit} K-lines se terminant avant {pd.to_datetime(end_timestamp_ms, unit='ms', utc=True)} via API REST.")
    elif start_timestamp_ms:
        logger.debug(f"{log_ctx} Requête de {limit} K-lines commençant à {pd.to_datetime(start_timestamp_ms, unit='ms', utc=True)} via API REST.")
    else:
        logger.debug(f"{log_ctx} Requête des {limit} K-lines les plus récentes via API REST.")

    account_type_upper = account_type.upper()
    base_url: str
    endpoint: str = "/klines" # Commun pour la plupart, mais les préfixes d'API changent
    api_max_limit_for_endpoint: int

    if account_type_upper in ["SPOT", "MARGIN"]: # Binance Spot/Margin API
        base_url = "https://api.binance.com/api/v3"
        api_max_limit_for_endpoint = API_BATCH_LIMIT_SPOT_MARGIN
    elif account_type_upper == "FUTURES_USDT": # USD-M Futures API
        base_url = "https://fapi.binance.com/fapi/v1"
        api_max_limit_for_endpoint = API_BATCH_LIMIT_FUTURES
    elif account_type_upper == "FUTURES_COIN": # COIN-M Futures API
        base_url = "https://dapi.binance.com/dapi/v1"
        api_max_limit_for_endpoint = API_BATCH_LIMIT_FUTURES_COIN_M
    else:
        logger.error(f"{log_ctx} Type de compte non supporté pour la récupération de K-lines : {account_type}")
        return None

    # S'assurer que la limite demandée ne dépasse pas la limite de l'API pour cet endpoint
    current_fetch_limit = min(limit, api_max_limit_for_endpoint)
    if limit > api_max_limit_for_endpoint:
        logger.debug(f"{log_ctx} Limite demandée ({limit}) > limite API ({api_max_limit_for_endpoint}). Utilisation de {api_max_limit_for_endpoint}.")


    url = base_url + endpoint
    params: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": actual_fetch_interval,
        "limit": current_fetch_limit
    }
    if end_timestamp_ms:
        params["endTime"] = end_timestamp_ms
    if start_timestamp_ms:
        params["startTime"] = start_timestamp_ms

    try:
        response = requests.get(url, params=params, timeout=15) # Timeout augmenté à 15s
        response.raise_for_status() # Lève une HTTPError pour les codes d'erreur HTTP (4XX, 5XX)
        data_raw = response.json()

        if not data_raw: # Si l'API retourne une liste vide
            logger.info(f"{log_ctx} Aucune donnée K-line reçue de l'API Binance pour les paramètres : {params}")
            return pd.DataFrame(columns=BASE_COLUMNS_RAW)

        df = pd.DataFrame(data_raw, columns=BINANCE_KLINE_COLUMNS)
        # Renommer 'kline_open_time' en 'timestamp' pour la cohérence
        df.rename(columns={'kline_open_time': 'timestamp'}, inplace=True)

        # Conversion des timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df['kline_close_time'] = pd.to_datetime(df['kline_close_time'], unit='ms', utc=True, errors='coerce')

        # Conversion des colonnes numériques
        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                        'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else: # Devrait être rare si BINANCE_KLINE_COLUMNS est correct
                df[col] = np.nan

        # Calcul des volumes Taker vendeurs
        # taker_sell_base = total_base_volume - taker_buy_base_volume
        # taker_sell_quote = total_quote_volume - taker_buy_quote_volume
        if 'volume' in df.columns and 'taker_buy_base_asset_volume' in df.columns:
            df['taker_sell_base_asset_volume'] = df['volume'].sub(df['taker_buy_base_asset_volume'], fill_value=np.nan)
        else:
            df['taker_sell_base_asset_volume'] = np.nan
        
        if 'quote_asset_volume' in df.columns and 'taker_buy_quote_asset_volume' in df.columns:
            df['taker_sell_quote_asset_volume'] = df['quote_asset_volume'].sub(df['taker_buy_quote_asset_volume'], fill_value=np.nan)
        else:
            df['taker_sell_quote_asset_volume'] = np.nan

        # S'assurer que toutes les colonnes de BASE_COLUMNS_RAW sont présentes et dans le bon ordre
        for col in BASE_COLUMNS_RAW:
            if col not in df.columns:
                df[col] = np.nan # Ajouter la colonne avec NaNs si elle manque
        df = df[BASE_COLUMNS_RAW]

        # Supprimer les lignes où les timestamps ou OHLC essentiels sont NaNs après conversion
        essential_cols_for_dropna = ['timestamp', 'open', 'high', 'low', 'close', 'kline_close_time']
        df.dropna(subset=essential_cols_for_dropna, how='any', inplace=True)
        
        if df.empty:
            logger.info(f"{log_ctx} DataFrame vide après nettoyage des K-lines (NaNs dans colonnes essentielles).")
        else:
            logger.info(f"{log_ctx} Récupéré et traité {len(df)} K-lines via API REST.")
        return df

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"{log_ctx} Erreur HTTP lors de la récupération des K-lines : {http_err}. "
                     f"Réponse : {http_err.response.text if http_err.response else 'Pas de corps de réponse'}")
    except requests.exceptions.RequestException as req_err: # Timeout, ConnectionError, etc.
        logger.error(f"{log_ctx} Erreur de requête lors de la récupération des K-lines : {req_err}")
    except json.JSONDecodeError as json_err:
        logger.error(f"{log_ctx} Erreur de décodage JSON lors du traitement de la réponse des K-lines : {json_err}")
    except Exception as e: # pylint: disable=broad-except
        logger.error(f"{log_ctx} Erreur inattendue lors de la récupération/traitement des K-lines : {e}", exc_info=True)
    return None


def initialize_pair_data(
    pair: str,
    raw_path: Path, # Chemin vers le fichier CSV brut (ex: PAIR_1min_live_raw.csv)
    total_klines_to_fetch: int, # Nombre total de K-lines 1-min souhaité dans le fichier
    account_type: str # Pour get_binance_klines_rest (SPOT, MARGIN, FUTURES_USDT, etc.)
) -> None:
    """
    Initialise le fichier de données brutes 1-minute pour une paire.
    Si le fichier existe, il est complété avec des données plus anciennes pour atteindre
    `total_klines_to_fetch`. Si le fichier n'existe pas, il est créé avec
    `total_klines_to_fetch` K-lines historiques.

    Args:
        pair (str): Symbole de la paire (ex: BTCUSDT).
        raw_path (Path): Chemin complet vers le fichier CSV de données brutes 1-minute.
        total_klines_to_fetch (int): Nombre total de K-lines 1-minute à avoir dans le fichier.
        account_type (str): Type de compte/marché pour l'appel API.
    """
    pair_upper = pair.upper()
    log_ctx = f"[{pair_upper}][1min][InitData][{account_type.upper()}]"
    logger.info(f"{log_ctx} Initialisation du fichier de données brutes live : {raw_path}")
    logger.info(f"{log_ctx} Nombre total de K-lines 1-minute visé : {total_klines_to_fetch}")

    df_existing = pd.DataFrame(columns=BASE_COLUMNS_RAW)
    if raw_path.exists() and raw_path.stat().st_size > 0:
        try:
            df_existing = pd.read_csv(raw_path)
            if 'timestamp' in df_existing.columns:
                df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'], errors='coerce', utc=True)
                df_existing.dropna(subset=['timestamp'], inplace=True) # Important
            else:
                logger.warning(f"{log_ctx} Fichier existant {raw_path} sans colonne 'timestamp'. Traité comme vide.")
                df_existing = pd.DataFrame(columns=BASE_COLUMNS_RAW)
            logger.info(f"{log_ctx} Fichier existant chargé. Contient {len(df_existing)} lignes valides.")
        except pd.errors.EmptyDataError:
            logger.warning(f"{log_ctx} Fichier brut {raw_path} est vide. Récupération complète de l'historique.")
        except Exception as e_read: # pylint: disable=broad-except
            logger.error(f"{log_ctx} Erreur lors de la lecture du fichier brut existant {raw_path}: {e_read}. "
                         "Tentative de récupération complète de l'historique.", exc_info=True)
    else:
        logger.info(f"{log_ctx} Fichier brut {raw_path} non trouvé ou vide. Récupération de l'historique.")
        # S'assurer que le répertoire parent existe
        try:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e_mkdir:
            logger.error(f"{log_ctx} Échec de la création du répertoire parent {raw_path.parent} : {e_mkdir}. Abandon.")
            return

    num_existing_valid_lines = len(df_existing)
    klines_still_needed = total_klines_to_fetch - num_existing_valid_lines
    logger.info(f"{log_ctx} Lignes valides existantes : {num_existing_valid_lines}. K-lines encore nécessaires : {klines_still_needed}")

    if klines_still_needed > 0:
        logger.info(f"{log_ctx} Récupération de {klines_still_needed} K-lines 1-minute historiques supplémentaires...")
        
        all_newly_fetched_klines_list: List[pd.DataFrame] = []
        
        # Déterminer le point de départ pour la récupération des données plus anciennes
        # Si des données existent, on part du timestamp le plus ancien de ces données.
        current_api_end_timestamp_ms: Optional[int] = None
        if not df_existing.empty and 'timestamp' in df_existing.columns and df_existing['timestamp'].is_monotonic_increasing:
            oldest_existing_ts: pd.Timestamp = df_existing['timestamp'].iloc[0]
            current_api_end_timestamp_ms = int(oldest_existing_ts.timestamp() * 1000)
            logger.debug(f"{log_ctx} Récupération des données antérieures à : {oldest_existing_ts} (TS ms: {current_api_end_timestamp_ms})")
        else:
            logger.debug(f"{log_ctx} Aucune donnée existante valide ou timestamp le plus ancien non déterminable. Récupération des données les plus récentes en premier, puis vers le passé.")
            # Si pas de données, current_api_end_timestamp_ms reste None, get_binance_klines_rest prendra les plus récentes.

        # Déterminer la limite de batch pour l'API
        batch_limit_for_api: int
        if account_type.upper() in ["SPOT", "MARGIN"]: batch_limit_for_api = API_BATCH_LIMIT_SPOT_MARGIN
        elif account_type.upper() == "FUTURES_USDT": batch_limit_for_api = API_BATCH_LIMIT_FUTURES
        elif account_type.upper() == "FUTURES_COIN": batch_limit_for_api = API_BATCH_LIMIT_FUTURES_COIN_M
        else: batch_limit_for_api = API_BATCH_LIMIT_SPOT_MARGIN # Fallback

        fetch_attempts_current_history_fill = 0
        while klines_still_needed > 0 and fetch_attempts_current_history_fill < MAX_API_FETCH_RETRIES:
            num_to_fetch_this_batch = min(klines_still_needed, batch_limit_for_api)
            logger.info(f"{log_ctx} Tentative de récupération d'un batch de {num_to_fetch_this_batch} K-lines. "
                        f"Fin avant (ms) : {current_api_end_timestamp_ms}. Restant à récupérer : {klines_still_needed}.")

            df_batch_fetched = get_binance_klines_rest(
                symbol=pair_upper,
                limit=num_to_fetch_this_batch,
                account_type=account_type,
                end_timestamp_ms=current_api_end_timestamp_ms
            )

            if df_batch_fetched is not None and not df_batch_fetched.empty:
                all_newly_fetched_klines_list.append(df_batch_fetched)
                klines_still_needed -= len(df_batch_fetched)
                
                # Mettre à jour end_timestamp_ms pour le prochain batch (plus ancien)
                # S'assurer que la colonne 'timestamp' est du bon type et triée
                if not pd.api.types.is_datetime64_any_dtype(df_batch_fetched['timestamp']):
                    df_batch_fetched['timestamp'] = pd.to_datetime(df_batch_fetched['timestamp'], errors='coerce', utc=True)
                
                df_batch_fetched.sort_values(by='timestamp', ascending=True, inplace=True) # Nécessaire pour prendre le .iloc[0] correct
                
                if df_batch_fetched.empty or df_batch_fetched['timestamp'].isnull().all():
                    logger.warning(f"{log_ctx} Batch devenu vide ou timestamps invalides après conversion. Arrêt de la récupération pour ce cycle.")
                    break
                
                oldest_ts_in_this_batch: pd.Timestamp = df_batch_fetched['timestamp'].iloc[0]
                current_api_end_timestamp_ms = int(oldest_ts_in_this_batch.timestamp() * 1000)
                # Pas besoin de soustraire 1ms ici, car endTime est exclusif.
                # Si on veut les klines strictement AVANT ce timestamp, c'est correct.

                if len(df_batch_fetched) < num_to_fetch_this_batch:
                    logger.info(f"{log_ctx} L'API a retourné moins de K-lines ({len(df_batch_fetched)}) que demandé ({num_to_fetch_this_batch}). "
                                "Supposition de la fin de l'historique disponible dans cette direction.")
                    break # Fin de l'historique disponible
                time.sleep(0.3) # Petit délai pour respecter les limites de taux API
            else: # Échec de la récupération du batch ou batch vide
                logger.warning(f"{log_ctx} Échec de la récupération d'un batch ou batch vide. Tentative {fetch_attempts_current_history_fill + 1}/{MAX_API_FETCH_RETRIES}.")
                fetch_attempts_current_history_fill += 1
                if fetch_attempts_current_history_fill < MAX_API_FETCH_RETRIES:
                    logger.info(f"{log_ctx} Nouvelle tentative de récupération de l'historique dans {API_FETCH_RETRY_DELAY_SECONDS} secondes.")
                    time.sleep(API_FETCH_RETRY_DELAY_SECONDS)
                else:
                    logger.error(f"{log_ctx} Nombre maximum de tentatives atteint pour la récupération des batches historiques.")
                    break # Sortir de la boucle while
        
        if all_newly_fetched_klines_list:
            df_newly_fetched_total = pd.concat(all_newly_fetched_klines_list, ignore_index=True)
            
            # Combiner avec les données existantes, trier, et supprimer les doublons
            df_combined_final = pd.concat([df_existing, df_newly_fetched_total], ignore_index=True)
            
            if 'timestamp' in df_combined_final.columns: # Devrait toujours être vrai
                # Re-convertir et nettoyer au cas où des types non-datetime seraient introduits par concat
                if not pd.api.types.is_datetime64_any_dtype(df_combined_final['timestamp']):
                    df_combined_final['timestamp'] = pd.to_datetime(df_combined_final['timestamp'], errors='coerce', utc=True)
                df_combined_final.dropna(subset=['timestamp'], inplace=True)
                df_combined_final.sort_values(by='timestamp', ascending=True, inplace=True)
                df_combined_final.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            
            # S'assurer que toutes les colonnes BASE_COLUMNS_RAW sont présentes avant la sauvegarde
            for col in BASE_COLUMNS_RAW:
                if col not in df_combined_final.columns:
                    df_combined_final[col] = np.nan
            
            # Sauvegarder le DataFrame combiné et nettoyé
            try:
                df_combined_final[BASE_COLUMNS_RAW].to_csv(raw_path, index=False, float_format='%.8f')
                logger.info(f"{log_ctx} Fichier de données brutes initialisé/mis à jour avec {len(df_combined_final)} lignes : {raw_path}")
            except Exception as e_save: # pylint: disable=broad-except
                logger.error(f"{log_ctx} Erreur lors de la sauvegarde du fichier de données combinées {raw_path} : {e_save}", exc_info=True)

        elif num_existing_valid_lines == 0: # Si la récupération a échoué et que le fichier était initialement vide
            logger.warning(f"{log_ctx} La récupération de l'historique initial a échoué et le fichier était vide. "
                           f"Le fichier {raw_path} reste vide (ou avec seulement les en-têtes s'il vient d'être créé).")
            # Créer un fichier vide avec les en-têtes s'il n'existe pas ou est complètement vide
            if not raw_path.exists() or raw_path.stat().st_size == 0:
                 pd.DataFrame(columns=BASE_COLUMNS_RAW).to_csv(raw_path, index=False)
        else: # Si la récupération a échoué mais qu'il y avait des données existantes
             logger.warning(f"{log_ctx} La récupération de l'historique supplémentaire a échoué. "
                            f"Les {num_existing_valid_lines} lignes existantes dans {raw_path} sont préservées.")
    else: # klines_still_needed <= 0
        logger.info(f"{log_ctx} Données suffisantes ({num_existing_valid_lines} lignes >= cible {total_klines_to_fetch}) "
                    f"déjà présentes dans {raw_path}. Aucune récupération supplémentaire nécessaire.")

    # Vérification finale du fichier après toutes les opérations
    if raw_path.exists() and raw_path.stat().st_size > 0:
        try:
            df_final_check = pd.read_csv(raw_path)
            logger.info(f"{log_ctx} Vérification finale : {raw_path} contient {len(df_final_check)} lignes.")
        except Exception: # pylint: disable=broad-except
            logger.error(f"{log_ctx} Échec de la lecture du fichier final {raw_path} pour vérification.")
    elif not raw_path.exists():
        logger.error(f"{log_ctx} Fichier final {raw_path} non trouvé après le processus d'initialisation.")


def run_initialization(app_config_obj_or_dict: Union['AppConfig', Dict[str, Any]]) -> None:
    """
    Orchestre l'initialisation des fichiers de données brutes 1-minute pour toutes
    les paires configurées dans `AppConfig.live_config.live_fetch.crypto_pairs`.

    Args:
        app_config_obj_or_dict (Union[AppConfig, Dict[str, Any]]): Soit une instance
            complète de AppConfig, soit un dictionnaire la représentant (ex: chargé d'un JSON).
    """
    main_log_prefix = "[LiveInitOrchestrator]"
    logger.info(f"{main_log_prefix} --- Démarrage de l'Initialisation des Données Live (API REST pour historique 1-minute) ---")

    # Extraction robuste de la configuration nécessaire
    live_fetch_cfg: Optional['LiveFetchConfig'] = None
    global_live_settings_cfg: Optional['GlobalLiveSettings'] = None
    paths_cfg: Optional['PathsConfig'] = None
    project_root: Optional[str] = None

    if hasattr(app_config_obj_or_dict, 'live_config') and \
       hasattr(app_config_obj_or_dict, 'global_config') and \
       hasattr(app_config_obj_or_dict, 'project_root'): # Probablement une instance AppConfig
        
        app_config_instance = cast('AppConfig', app_config_obj_or_dict)
        live_fetch_cfg = app_config_instance.live_config.live_fetch
        global_live_settings_cfg = app_config_instance.live_config.global_live_settings
        paths_cfg = app_config_instance.global_config.paths
        project_root = app_config_instance.project_root
    elif isinstance(app_config_obj_or_dict, dict):
        # Tentative d'extraire d'un dictionnaire
        live_config_dict = app_config_obj_or_dict.get('live_config', {})
        live_fetch_cfg_dict = live_config_dict.get('live_fetch', {})
        global_live_settings_dict = live_config_dict.get('global_live_settings', {})
        
        global_config_dict = app_config_obj_or_dict.get('global_config', {})
        paths_cfg_dict = global_config_dict.get('paths', {})
        project_root = app_config_obj_or_dict.get('project_root')

        # Créer des objets factices ou des adaptateurs si nécessaire pour accéder aux attributs
        # Pour simplifier, on accède directement aux clés du dict, en supposant qu'elles existent
        # ou en utilisant .get() avec des valeurs par défaut.
        # Ceci est moins sûr que d'utiliser des dataclasses typées.
        class _TempLiveFetchConfig:
            crypto_pairs = live_fetch_cfg_dict.get('crypto_pairs', [])
            limit_init_history = live_fetch_cfg_dict.get('limit_init_history', 2000)
        live_fetch_cfg = cast('LiveFetchConfig', _TempLiveFetchConfig())

        class _TempGlobalLiveSettings:
            # account_type est dans GlobalLiveSettings dans definitions.py
            # mais dans le JSON config_live.json, il est au même niveau que run_live_trading.
            # La structure de AppConfig doit être respectée.
            account_type = global_live_settings_dict.get('account_type', 'SPOT') # Ajuster si la structure diffère
        global_live_settings_cfg = cast('GlobalLiveSettings', _TempGlobalLiveSettings())

        class _TempPathsConfig:
            data_live_raw = paths_cfg_dict.get('data_live_raw', 'data/live/raw')
        paths_cfg = cast('PathsConfig', _TempPathsConfig())
    
    if not all([live_fetch_cfg, global_live_settings_cfg, paths_cfg]):
        logger.critical(f"{main_log_prefix} Configuration essentielle manquante (live_fetch, global_live_settings, ou paths). Impossible de continuer.")
        return

    if not project_root:
        try:
            project_root = str(Path(__file__).resolve().parent.parent.parent) # src/data -> src -> project_root
            logger.warning(f"{main_log_prefix} project_root non trouvé dans la config, déduction : {project_root}")
        except NameError: # __file__ non défini
            project_root = str(Path(".").resolve())
            logger.warning(f"{main_log_prefix} project_root non trouvé et __file__ non défini, utilisation du répertoire courant : {project_root}")
    
    project_root_path = Path(project_root)
    raw_dir_path = project_root_path / paths_cfg.data_live_raw

    pairs_to_init = live_fetch_cfg.crypto_pairs
    total_klines_target = live_fetch_cfg.limit_init_history
    # L'account_type pour l'API REST publique est moins critique que pour les ordres,
    # mais il détermine l'URL de base (spot vs futures) et les limites de taux.
    # On prend celui de global_live_settings qui devrait refléter le marché principal visé.
    api_account_type = global_live_settings_cfg.account_type # type: ignore

    if not pairs_to_init:
        logger.error(f"{main_log_prefix} Aucune crypto_pairs définie dans live_fetch_config. Initialisation impossible.")
        return

    logger.info(f"{main_log_prefix} Paires à initialiser : {pairs_to_init}. "
                f"Type de compte API pour fetch : {api_account_type}. "
                f"Répertoire des données brutes live : {raw_dir_path}")

    for pair_symbol in pairs_to_init:
        if not pair_symbol or not isinstance(pair_symbol, str):
            logger.warning(f"{main_log_prefix} Symbole de paire invalide trouvé dans la configuration : '{pair_symbol}'. Ignoré.")
            continue
            
        # Nom de fichier standardisé pour les données brutes live 1-minute
        raw_file_name_for_pair = f"{pair_symbol.upper()}_1min_live_raw.csv"
        full_raw_path_for_pair = raw_dir_path / raw_file_name_for_pair
        
        logger.info(f"{main_log_prefix} Initialisation des données brutes 1-minute pour {pair_symbol.upper()} -> {full_raw_path_for_pair}...")
        try:
            initialize_pair_data(
                pair=pair_symbol.upper(),
                raw_path=full_raw_path_for_pair,
                total_klines_to_fetch=total_klines_target,
                account_type=api_account_type
            )
        except Exception as e_init_pair: # pylint: disable=broad-except
            logger.error(f"{main_log_prefix} Échec de l'initialisation des données brutes live pour {pair_symbol.upper()} "
                         f"(fichier: {full_raw_path_for_pair}): {e_init_pair}", exc_info=True)

    logger.info(f"{main_log_prefix} --- Initialisation des Données Live (API REST) Terminée ---")


if __name__ == '__main__':
    # Configuration du logging pour exécution directe (tests)
    logging.basicConfig(
        level=logging.DEBUG, # Mettre à DEBUG pour voir plus de détails pendant les tests
        format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger.info("Exécution de src/data/acquisition_live.py directement pour tests (API REST uniquement)...")

    # Créer une configuration factice pour le test
    # S'assurer que les chemins sont corrects par rapport à l'endroit où le test est lancé
    try:
        _test_project_root_main = Path(__file__).resolve().parent.parent.parent
    except NameError: # Si __file__ n'est pas défini (ex: REPL)
        _test_project_root_main = Path(".").resolve()

    _test_data_live_raw_path_main = _test_project_root_main / "temp_test_data_live" / "raw"
    _test_data_live_raw_path_main.mkdir(parents=True, exist_ok=True) # Créer le répertoire de test

    mock_app_config_dict_main = {
        "project_root": str(_test_project_root_main),
        "global_config": {
            "paths": {
                "data_live_raw": str(_test_data_live_raw_path_main.relative_to(_test_project_root_main))
            }
            # D'autres champs de GlobalConfig peuvent être nécessaires si les fonctions internes les utilisent
        },
        "live_config": {
            "live_fetch": {
                "crypto_pairs": ["BTCUSDT", "ETHUSDT"], # Paires pour le test
                "intervals": ["1m"], # Non directement utilisé par initialize_pair_data mais bon à avoir
                "limit_init_history": 500, # Nombre plus petit pour un test rapide
            },
            "global_live_settings": {
                "account_type": "SPOT" # Ou MARGIN, FUTURES_USDT, FUTURES_COIN selon l'API à tester
            }
        }
        # Les autres sections de AppConfig (data_config, strategies_config, etc.) ne sont pas
        # directement utilisées par run_initialization ici, mais une AppConfig complète serait passée normalement.
    }

    logger.info(f"Utilisation de la racine de projet de test : {_test_project_root_main}")
    logger.info(f"Les données brutes live de test seront stockées dans : {_test_data_live_raw_path_main}")

    try:
        run_initialization(cast('AppConfig', mock_app_config_dict_main)) # cast pour le typage
        logger.info("Exécution de test de run_initialization terminée.")
    except ImportError as e_imp_test_main:
         logger.critical(f"Bibliothèque manquante pour le test: {e_imp_test}. "
                         "Veuillez installer les paquets requis (ex: requests, pandas, numpy).")
    except Exception as main_err_test: # pylint: disable=broad-except
         logger.critical(f"Erreur critique lors de l'exécution du test principal : {main_err_test}", exc_info=True)

    # Optionnel : Nettoyage après test
    # import shutil
    # if _test_data_live_raw_path_main.parent.exists(): # Supprimer temp_test_data_live
    #     logger.info(f"Nettoyage du répertoire de test : {_test_data_live_raw_path_main.parent}")
    #     shutil.rmtree(_test_data_live_raw_path_main.parent)
