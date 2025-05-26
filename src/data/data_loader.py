# src/data/data_loader.py
"""
Ce module est responsable du chargement des fichiers de données historiques
prétraitées et enrichies (fichiers Parquet) pour une paire de trading spécifique.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional # Ajout de Optional

import pandas as pd
import numpy as np # Pour la gestion des NaN/inf

if TYPE_CHECKING:
    from src.config.loader import AppConfig

logger = logging.getLogger(__name__)

# Colonnes OHLCV de base attendues dans le DataFrame enrichi
BASE_OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

def load_enriched_historical_data(
    pair_symbol: str,
    app_config: 'AppConfig'
) -> pd.DataFrame:
    """
    Charge, valide et nettoie les données historiques enrichies à partir d'un fichier Parquet
    pour une paire de trading spécifique.

    Args:
        pair_symbol (str): Le symbole de la paire de trading (ex: BTCUSDT).
        app_config (AppConfig): L'instance de configuration globale de l'application.

    Returns:
        pd.DataFrame: Un DataFrame pandas nettoyé et prêt à l'emploi,
                      indexé par timestamp UTC. Retourne un DataFrame vide
                      en cas d'erreur critique empêchant le chargement ou la validation.

    Raises:
        FileNotFoundError: Si le fichier de données enrichies n'est pas trouvé.
        ValueError: Si les données sont invalides (ex: colonne timestamp manquante,
                    colonnes OHLCV manquantes).
    """
    log_prefix = f"[DataLoader][{pair_symbol.upper()}]"
    logger.info(f"{log_prefix} Tentative de chargement des données historiques enrichies.")

    if not app_config.project_root:
        logger.error(f"{log_prefix} Le chemin racine du projet (project_root) n'est pas défini dans AppConfig.")
        raise ValueError("project_root non défini dans AppConfig.")

    enriched_data_path_str = app_config.global_config.paths.data_historical_processed_enriched
    if not enriched_data_path_str:
        logger.error(f"{log_prefix} Le chemin vers data_historical_processed_enriched n'est pas défini dans PathsConfig.")
        raise ValueError("Chemin data_historical_processed_enriched non configuré.")

    # Construire le chemin complet vers le fichier Parquet
    # Le nom du fichier est supposé être {PAIR_SYMBOL}_enriched.parquet
    enriched_file_name = f"{pair_symbol.upper()}_enriched.parquet"
    enriched_file_path = Path(app_config.project_root) / enriched_data_path_str / enriched_file_name

    logger.debug(f"{log_prefix} Chemin du fichier Parquet enrichi : {enriched_file_path}")

    # 1. Vérifier l'existence du fichier
    if not enriched_file_path.is_file():
        logger.error(f"{log_prefix} Fichier de données enrichies non trouvé : {enriched_file_path}")
        raise FileNotFoundError(f"Fichier de données enrichies non trouvé : {enriched_file_path}")

    # 2. Charger le fichier Parquet
    try:
        df = pd.read_parquet(enriched_file_path)
        logger.info(f"{log_prefix} Fichier Parquet chargé avec succès. Shape initial : {df.shape}")
    except Exception as e:
        logger.error(f"{log_prefix} Erreur lors du chargement du fichier Parquet {enriched_file_path}: {e}", exc_info=True)
        # Retourner un DataFrame vide en cas d'échec de chargement
        return pd.DataFrame()

    if df.empty:
        logger.warning(f"{log_prefix} Le fichier Parquet {enriched_file_path} est vide.")
        return pd.DataFrame()

    # 3. Validation et Nettoyage du DataFrame
    # 3.a. Colonne 'timestamp'
    if 'timestamp' not in df.columns:
        logger.error(f"{log_prefix} Colonne 'timestamp' manquante dans le fichier {enriched_file_path}.")
        raise ValueError(f"Colonne 'timestamp' manquante dans {enriched_file_path}.")

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        # Supprimer les lignes où la conversion du timestamp a échoué
        rows_before_ts_dropna = len(df)
        df.dropna(subset=['timestamp'], inplace=True)
        if len(df) < rows_before_ts_dropna:
            logger.warning(f"{log_prefix} {rows_before_ts_dropna - len(df)} lignes supprimées en raison de timestamps invalides.")
        
        if df.empty:
            logger.error(f"{log_prefix} DataFrame vide après suppression des timestamps invalides.")
            return pd.DataFrame()

        df = df.set_index('timestamp')
        logger.debug(f"{log_prefix} Colonne 'timestamp' convertie en DatetimeIndex UTC.")
    except Exception as e:
        logger.error(f"{log_prefix} Erreur lors de la conversion ou de la définition de l'index 'timestamp': {e}", exc_info=True)
        raise ValueError(f"Erreur de traitement du timestamp dans {enriched_file_path}: {e}")


    # 3.b. Unicité et tri de l'index
    if not df.index.is_unique:
        logger.warning(f"{log_prefix} L'index DatetimeIndex contient des timestamps dupliqués. Conservation de la première occurrence.")
        df = df[~df.index.duplicated(keep='first')] # Garder la première
    
    if not df.index.is_monotonic_increasing:
        logger.info(f"{log_prefix} L'index DatetimeIndex n'est pas trié. Tri en cours...")
        df = df.sort_index()

    if df.empty: # Au cas où la déduplication viderait le DataFrame
        logger.error(f"{log_prefix} DataFrame vide après déduplication/tri de l'index.")
        return pd.DataFrame()

    # 3.c. Présence des colonnes OHLCV de base
    missing_ohlcv_cols = [col for col in BASE_OHLCV_COLUMNS if col not in df.columns]
    if missing_ohlcv_cols:
        logger.error(f"{log_prefix} Colonnes OHLCV de base manquantes : {missing_ohlcv_cols} dans {enriched_file_path}.")
        raise ValueError(f"Colonnes OHLCV de base manquantes : {missing_ohlcv_cols} dans {enriched_file_path}.")

    # 3.d. Conversion des colonnes OHLCV en types numériques
    try:
        for col in BASE_OHLCV_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='raise') # 'raise' pour attraper les erreurs de conversion
        logger.debug(f"{log_prefix} Colonnes OHLCV converties en types numériques.")
    except Exception as e:
        logger.error(f"{log_prefix} Erreur lors de la conversion des colonnes OHLCV en numérique : {e}", exc_info=True)
        raise ValueError(f"Erreur de conversion numérique des colonnes OHLCV dans {enriched_file_path}: {e}")

    # 3.e. Gestion des valeurs NaN ou infinies dans les colonnes OHLCV
    # Remplacer infini par NaN d'abord
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Compter les NaNs avant ffill/bfill pour le logging
    nan_counts_before_fill = df[BASE_OHLCV_COLUMNS].isnull().sum()

    # Appliquer ffill puis bfill pour les colonnes OHLCV
    # Cela remplit les NaNs avec la dernière valeur valide, puis avec la prochaine si le début a des NaNs.
    df[BASE_OHLCV_COLUMNS] = df[BASE_OHLCV_COLUMNS].ffill().bfill()
    
    nan_counts_after_fill = df[BASE_OHLCV_COLUMNS].isnull().sum()
    for col in BASE_OHLCV_COLUMNS:
        if nan_counts_before_fill[col] > 0 and nan_counts_after_fill[col] < nan_counts_before_fill[col]:
            logger.info(f"{log_prefix} NaNs remplis dans la colonne '{col}'. Avant: {nan_counts_before_fill[col]}, Après: {nan_counts_after_fill[col]}.")
        elif nan_counts_after_fill[col] > 0:
            # Si des NaNs persistent, cela signifie que toute la colonne était NaN (ou des sections entières au début/fin)
            logger.warning(f"{log_prefix} Des NaNs persistent dans la colonne OHLCV '{col}' après ffill/bfill. "
                           f"Cela peut indiquer des données manquantes importantes.")
            # Optionnel : supprimer les lignes où des colonnes OHLCV critiques sont encore NaN
            # df.dropna(subset=[col], inplace=True)
            # Pour l'instant, on les laisse, mais les stratégies devront les gérer.

    if df.empty: # Si dropna était activé et a vidé le DataFrame
        logger.error(f"{log_prefix} DataFrame vide après gestion des NaNs dans OHLCV.")
        return pd.DataFrame()

    logger.info(f"{log_prefix} Données historiques enrichies chargées et nettoyées avec succès. Shape final : {df.shape}")
    return df

if __name__ == '__main__':
    # Section pour des tests directs du module (optionnel)
    # Nécessite une instance AppConfig factice ou un chargement partiel pour tester.
    
    # Configuration basique du logging pour les tests directs
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Exécution de data_loader.py directement pour tests (nécessite une configuration factice).")

    # --- Création d'une AppConfig factice pour le test ---
    # Dans un vrai scénario, AppConfig serait chargée par src.config.loader.load_all_configs()
    from src.config.definitions import GlobalConfig, PathsConfig, DataConfig, SourceDetails, AssetsAndTimeframes, HistoricalPeriod, FetchingOptions, LoggingConfig, SimulationDefaults, WfoSettings, OptunaSettings, StrategiesConfig, LiveConfig, AccountConfig, ApiKeys, ExchangeSettings
    
    # Créer un répertoire de test et un fichier Parquet factice
    test_project_root = Path("./temp_data_loader_test_project").resolve()
    test_project_root.mkdir(parents=True, exist_ok=True)
    
    test_enriched_path = test_project_root / "data" / "historical" / "processed" / "enriched"
    test_enriched_path.mkdir(parents=True, exist_ok=True)
    
    dummy_pair = "BTCUSDT"
    dummy_file_path = test_enriched_path / f"{dummy_pair}_enriched.parquet"

    # Créer des données factices
    timestamps = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', None, '2023-01-01 00:02:00', '2023-01-01 00:02:00'], utc=True)
    data = {
        'timestamp': timestamps,
        'open': [100, 101, 102, 100.5, 100.6],
        'high': [105, 102, 103, 101.5, 101.6],
        'low': [99, 100, 101, 99.5, 99.6],
        'close': [101, 101.5, np.nan, 101, 101.1],
        'volume': [10, 20, 5, 15, 16],
        'some_other_indicator': [1, 2, 3, 4, 5]
    }
    dummy_df_to_save = pd.DataFrame(data)
    try:
        dummy_df_to_save.to_parquet(dummy_file_path)
        logger.info(f"Fichier Parquet factice créé : {dummy_file_path}")

        # Créer une instance AppConfig factice minimale
        mock_paths = PathsConfig(
            data_historical_raw="data/historical/raw",
            data_historical_processed_cleaned="data/historical/processed/cleaned",
            data_historical_processed_enriched=str(test_enriched_path.relative_to(test_project_root)), # Chemin relatif
            logs_backtest_optimization="logs/backtest_optimization",
            logs_live="logs/live_trading",
            results="results",
            data_live_raw="data/live/raw",
            data_live_processed="data/live/processed",
            live_state="data/live_state"
        )
        # Créer d'autres dataclasses nécessaires pour GlobalConfig et AppConfig avec des valeurs par défaut ou factices
        mock_logging = LoggingConfig(level="INFO", format="", log_to_file=False, log_filename_global="") # Simplifié
        mock_sim_defaults = SimulationDefaults(initial_capital=1000, margin_leverage=1, trading_fee_bps=7.0) # Simplifié
        mock_wfo = WfoSettings(n_splits=3, oos_period_days=30, min_is_period_days=90, metric_to_optimize="Sharpe", optimization_direction="maximize") # Simplifié
        mock_optuna = OptunaSettings(n_trials=10) # Simplifié
        
        mock_global_config = GlobalConfig(
            project_name="TestProject", paths=mock_paths, logging=mock_logging,
            simulation_defaults=mock_sim_defaults, wfo_settings=mock_wfo, optuna_settings=mock_optuna
        )
        # ... (créer d'autres instances factices pour DataConfig, StrategiesConfig, etc. si nécessaire pour AppConfig)
        # Pour ce test, seul global_config.paths.data_historical_processed_enriched et project_root sont cruciaux.
        
        mock_app_config = AppConfig(
            global_config=mock_global_config,
            data_config=DataConfig(SourceDetails("",""), AssetsAndTimeframes([],[]), HistoricalPeriod("2020-01-01"), FetchingOptions(1,100)), # Factice
            strategies_config=StrategiesConfig(strategies={}), # Factice
            live_config=LiveConfig(GlobalLiveSettings(False,1,0.01,0.1),[],LiveFetchConfig([],[],1,1),LiveLoggingConfig()), # Factice
            accounts_config=[], # Factice
            api_keys=ApiKeys(), # Factice
            exchange_settings=ExchangeSettings(), # Factice
            project_root=str(test_project_root)
        )

        logger.info(f"Test de load_enriched_historical_data pour la paire : {dummy_pair}")
        loaded_df = load_enriched_historical_data(dummy_pair, mock_app_config)

        if not loaded_df.empty:
            logger.info(f"DataFrame chargé avec succès. Shape : {loaded_df.shape}")
            logger.info(f"Index : {loaded_df.index.name}, Type : {type(loaded_df.index)}, Unique : {loaded_df.index.is_unique}, Monotonique : {loaded_df.index.is_monotonic_increasing}")
            logger.info(f"Colonnes : {loaded_df.columns.tolist()}")
            logger.info("Premières lignes du DataFrame chargé et nettoyé :")
            print(loaded_df.head())
            logger.info("Dernières lignes du DataFrame chargé et nettoyé :")
            print(loaded_df.tail())
            # Vérifier les NaNs restants
            logger.info(f"NaNs restants dans OHLCV : \n{loaded_df[BASE_OHLCV_COLUMNS].isnull().sum()}")

        else:
            logger.error("Échec du chargement du DataFrame de test.")

    except Exception as e_test:
        logger.error(f"Erreur lors de l'exécution des tests directs de data_loader : {e_test}", exc_info=True)
    finally:
        # Nettoyage optionnel du répertoire de test
        import shutil
        if test_project_root.exists():
            # shutil.rmtree(test_project_root)
            # logger.info(f"Répertoire de test nettoyé : {test_project_root}")
            pass # Laisser les fichiers pour inspection
