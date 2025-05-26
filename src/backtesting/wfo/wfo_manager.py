# src/backtesting/wfo/wfo_manager.py
"""
Ce module définit WFOManager, responsable de la gestion complète du processus
d'Optimisation Walk-Forward (WFO) pour une seule combinaison
stratégie/paire/contexte CLI.
"""
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, cast
from datetime import datetime, timezone

# Imports depuis l'application
try:
    from src.config.loader import AppConfig, load_exchange_config # load_exchange_config pour le placeholder
    from src.config.definitions import (
        WfoSettings,
        StrategyParamsConfig,
        PathsConfig,
        ExchangeSettings,
        HistoricalPeriod,
        LoggingConfig # Ajouté pour la configuration du logging ad-hoc
    )
    from src.utils.run_utils import generate_run_id, create_wfo_task_dirs, _sanitize_path_component
    from src.utils.logging_setup import setup_logging # Pour configurer le logging de la tâche
    # Les fonctions/classes suivantes seront définies dans d'autres modules.
    # Des placeholders seront utilisés pour l'instant.
    # from src.data.data_loader import load_enriched_historical_data
    # from src.common.exchange_info_provider import get_symbol_exchange_info
    # from src.backtesting.wfo.fold_generator import FoldGenerator
    # from src.backtesting.optimization.fold_orchestrator import run_fold_optimization_and_validation
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(f"WFOManager: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH.")
    raise

# Logger pour ce module. Il sera configuré par __init__.
logger = logging.getLogger(__name__)

# --- Placeholders pour les modules/fonctions qui seront définis ultérieurement ---

def load_enriched_historical_data_placeholder(
    pair_symbol: str,
    app_config: AppConfig
) -> pd.DataFrame:
    """
    Placeholder pour charger les données historiques enrichies pour une paire.
    Lit depuis app_config.global_config.paths.data_historical_processed_enriched.
    """
    placeholder_log_prefix = f"[WFOManager][LoadDataPlaceholder][{pair_symbol}]"
    logger.warning(f"{placeholder_log_prefix} Utilisation de load_enriched_historical_data_placeholder.")
    
    if not app_config.project_root:
        logger.error(f"{placeholder_log_prefix} project_root n'est pas défini dans AppConfig.")
        raise ValueError("project_root non défini dans AppConfig pour le chargement des données placeholder.")

    enriched_file_path = Path(app_config.project_root) / app_config.global_config.paths.data_historical_processed_enriched / f"{pair_symbol.upper()}_enriched.parquet"
    
    if not enriched_file_path.is_file():
        logger.error(f"{placeholder_log_prefix} Fichier de données enrichies non trouvé : {enriched_file_path}")
        raise FileNotFoundError(f"Fichier de données enrichies non trouvé : {enriched_file_path}")
    try:
        df = pd.read_parquet(enriched_file_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"{placeholder_log_prefix} Les données de {enriched_file_path} doivent avoir une colonne 'timestamp' ou un DatetimeIndex.")
            raise ValueError("Les données enrichies doivent avoir une colonne 'timestamp' ou un DatetimeIndex.")
        
        # S'assurer que l'index est UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz.utcoffset(df.index[0]) != timezone.utc.utcoffset(df.index[0]): # type: ignore
            df.index = df.index.tz_convert('UTC')

        df = df.sort_index()
        if not df.index.is_unique:
            logger.warning(f"{placeholder_log_prefix} Timestamps dupliqués trouvés dans {enriched_file_path}. Conservation du premier.")
            df = df[~df.index.duplicated(keep='first')]
        logger.info(f"{placeholder_log_prefix} Données enrichies chargées pour {pair_symbol}, shape: {df.shape}, de {df.index.min()} à {df.index.max()}")
        return df
    except Exception as e:
        logger.error(f"{placeholder_log_prefix} Erreur lors du chargement des données enrichies pour {pair_symbol} depuis {enriched_file_path}: {e}", exc_info=True)
        raise

def get_symbol_exchange_info_placeholder(
    pair_symbol: str,
    app_config: AppConfig
) -> Dict[str, Any]:
    """
    Placeholder pour obtenir les informations de l'exchange pour un symbole.
    Utilise app_config.exchange_settings.exchange_info_file_path.
    """
    placeholder_log_prefix = f"[WFOManager][ExchangeInfoPlaceholder][{pair_symbol}]"
    logger.warning(f"{placeholder_log_prefix} Utilisation de get_symbol_exchange_info_placeholder.")
    
    if not app_config.project_root:
        logger.error(f"{placeholder_log_prefix} project_root n'est pas défini dans AppConfig.")
        raise ValueError("project_root non défini dans AppConfig pour le chargement des infos exchange placeholder.")

    exchange_info_file = Path(app_config.project_root) / app_config.exchange_settings.exchange_info_file_path
    try:
        exchange_info_full = load_exchange_config(exchange_info_file, project_root=app_config.project_root) # type: ignore
        if exchange_info_full and "symbols" in exchange_info_full:
            for symbol_data in exchange_info_full["symbols"]:
                if symbol_data.get("symbol") == pair_symbol.upper():
                    logger.info(f"{placeholder_log_prefix} Informations d'exchange trouvées pour {pair_symbol}.")
                    return symbol_data
        logger.error(f"{placeholder_log_prefix} Informations d'exchange non trouvées pour {pair_symbol} dans {exchange_info_file}.")
        raise ValueError(f"Informations d'exchange non trouvées pour {pair_symbol} dans {exchange_info_file}.")
    except Exception as e:
        logger.error(f"{placeholder_log_prefix} Erreur lors de l'obtention des informations d'exchange pour {pair_symbol}: {e}", exc_info=True)
        raise

class FoldGeneratorPlaceholder:
    """
    Placeholder pour le générateur de folds WFO, implémentant la logique
    de fenêtres IS expansives et OOS fixe.
    """
    def __init__(self, wfo_settings: WfoSettings):
        self.wfo_settings = wfo_settings
        self.log_prefix = "[WFOManager][FoldGeneratorPlaceholder]"
        logger.warning(f"{self.log_prefix} Utilisation de FoldGeneratorPlaceholder avec logique de fenêtres IS expansives.")

    def generate_folds(self,
                       df_enriched_data: pd.DataFrame,
                       global_start_date_config: Optional[pd.Timestamp] = None,
                       global_end_date_config: Optional[pd.Timestamp] = None
                       ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        logger.info(f"{self.log_prefix} Génération des folds WFO...")
        if df_enriched_data.empty:
            logger.error(f"{self.log_prefix} DataFrame de données enrichies vide. Impossible de générer les folds.")
            return []
        if not isinstance(df_enriched_data.index, pd.DatetimeIndex) or df_enriched_data.index.tz is None:
            logger.error(f"{self.log_prefix} L'index de df_enriched_data doit être un DatetimeIndex avec timezone (UTC attendu).")
            return []

        # 1. Déterminer les dates de début et de fin effectives pour le WFO
        data_start_date = df_enriched_data.index.min()
        data_end_date = df_enriched_data.index.max()

        wfo_start_date = global_start_date_config if global_start_date_config else data_start_date
        wfo_end_date = global_end_date_config if global_end_date_config else data_end_date
        
        # S'assurer que les dates de config sont compatibles avec les données
        wfo_start_date = max(wfo_start_date, data_start_date)
        wfo_end_date = min(wfo_end_date, data_end_date)

        if wfo_start_date >= wfo_end_date:
            logger.error(f"{self.log_prefix} Période WFO invalide après ajustement aux données : Début={wfo_start_date}, Fin={wfo_end_date}.")
            return []
        
        logger.info(f"{self.log_prefix} Période WFO effective : de {wfo_start_date} à {wfo_end_date}.")

        # 2. Définir la période OOS fixe
        oos_duration = pd.Timedelta(days=self.wfo_settings.oos_period_days)
        oos_end_date_fixed = wfo_end_date
        oos_start_date_fixed = oos_end_date_fixed - oos_duration + pd.Timedelta(days=1) # +1 pour inclure le jour de début

        # Ajuster oos_start_date_fixed pour qu'il ne soit pas avant wfo_start_date
        oos_start_date_fixed = max(oos_start_date_fixed, wfo_start_date + pd.Timedelta(days=self.wfo_settings.min_is_period_days)) # Assurer un IS minimal

        df_oos_fixed = df_enriched_data.loc[oos_start_date_fixed : oos_end_date_fixed]
        if df_oos_fixed.empty:
            logger.error(f"{self.log_prefix} Période OOS fixe vide (de {oos_start_date_fixed} à {oos_end_date_fixed}). Vérifiez oos_period_days et la plage de données.")
            return []
        
        actual_oos_start_ts = df_oos_fixed.index.min()
        actual_oos_end_ts = df_oos_fixed.index.max()
        logger.info(f"{self.log_prefix} Période OOS fixe définie : de {actual_oos_start_ts} à {actual_oos_end_ts} (Durée: {actual_oos_end_ts - actual_oos_start_ts}).")

        # 3. Définir la période IS totale
        is_total_end_date = actual_oos_start_ts - pd.Timedelta(microseconds=1) # Juste avant le début OOS
        is_total_start_date = wfo_start_date
        
        df_is_total = df_enriched_data.loc[is_total_start_date : is_total_end_date]
        if df_is_total.empty:
            logger.error(f"{self.log_prefix} Période IS totale vide (de {is_total_start_date} à {is_total_end_date}). Vérifiez la configuration.")
            return []

        actual_is_total_start_ts = df_is_total.index.min()
        actual_is_total_end_ts = df_is_total.index.max()
        duration_is_total = actual_is_total_end_ts - actual_is_total_start_ts
        logger.info(f"{self.log_prefix} Période IS totale définie : de {actual_is_total_start_ts} à {actual_is_total_end_ts} (Durée: {duration_is_total}).")

        if duration_is_total < pd.Timedelta(days=self.wfo_settings.min_is_period_days):
            logger.error(f"{self.log_prefix} Durée IS totale ({duration_is_total}) est inférieure à min_is_period_days ({self.wfo_settings.min_is_period_days} jours).")
            return []

        # 4. Segmenter la période IS totale et construire les folds IS expansifs
        n_splits = self.wfo_settings.n_splits
        if n_splits <= 0:
            logger.error(f"{self.log_prefix} n_splits ({n_splits}) doit être positif.")
            return []

        # Créer les points de segmentation pour les débuts des périodes IS
        # T_N est actual_is_total_end_ts
        # T_0 est actual_is_total_start_ts
        # Les segments sont entre ces points.
        
        # timestamps_is_total = pd.date_range(start=actual_is_total_start_ts, end=actual_is_total_end_ts, freq='D') # Approximatif si données non journalières
        # Pour une division plus précise basée sur le nombre de points de données :
        is_total_indices = df_is_total.index
        num_is_points = len(is_total_indices)
        
        if num_is_points < n_splits: # Pas assez de points pour créer les segments distincts
            logger.error(f"{self.log_prefix} Pas assez de points de données ({num_is_points}) dans la période IS totale pour {n_splits} splits.")
            return []

        segment_indices = np.linspace(0, num_is_points -1 , n_splits + 1, dtype=int)
        # T_0_idx = segment_indices[0] -> is_total_indices[0]
        # T_1_idx = segment_indices[1]
        # ...
        # T_N_idx = segment_indices[n_splits] -> is_total_indices[-1]

        # Les points d'ancrage T_k pour le début des segments IS des folds
        # T_k est le début du (k+1)-ième segment en partant de la fin
        # T_N (fin IS totale) est is_total_indices[segment_indices[n_splits]]
        # T_{N-1} (début du dernier segment IS) est is_total_indices[segment_indices[n_splits-1]]
        # T_0 (début IS totale) est is_total_indices[segment_indices[0]]

        folds_generated: List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

        for i in range(n_splits): # i va de 0 à n_splits - 1
            # Fold i utilise les segments de N-(i+1) à N
            # Le début de la période IS pour le fold i est T_{N-(i+1)}
            # L'index du point de début dans segment_indices est n_splits - (i + 1)
            start_segment_idx_in_array = n_splits - (i + 1)
            
            is_fold_start_ts = is_total_indices[segment_indices[start_segment_idx_in_array]]
            is_fold_end_ts = actual_is_total_end_ts # Toujours la fin de la période IS totale

            df_is_fold = df_is_total.loc[is_fold_start_ts : is_fold_end_ts]

            if df_is_fold.empty:
                logger.warning(f"{self.log_prefix} Fold {i}: Période IS vide (de {is_fold_start_ts} à {is_fold_end_ts}). Saut.")
                continue
            
            duration_is_fold_actual = df_is_fold.index.max() - df_is_fold.index.min()
            if duration_is_fold_actual < pd.Timedelta(days=self.wfo_settings.min_is_period_days):
                logger.warning(f"{self.log_prefix} Fold {i}: Durée IS ({duration_is_fold_actual}) < min_is_period_days ({self.wfo_settings.min_is_period_days} jours).")

            folds_generated.append((
                df_is_fold.copy(),
                df_oos_fixed.copy(),
                i, # fold_id
                df_is_fold.index.min(),
                df_is_fold.index.max(),
                actual_oos_start_ts,
                actual_oos_end_ts
            ))
            logger.info(f"{self.log_prefix} Fold {i} généré. IS: [{df_is_fold.index.min()} - {df_is_fold.index.max()}], OOS: [{actual_oos_start_ts} - {actual_oos_end_ts}]")

        if not folds_generated:
            logger.error(f"{self.log_prefix} Aucun fold valide n'a pu être généré avec la méthode des fenêtres expansives.")
        return folds_generated


def run_fold_optimization_and_validation_placeholder(
    app_config: AppConfig,
    strategy_name: str,
    pair_symbol: str,
    cli_context_label: str, # Ajouté pour la cohérence
    orchestrator_run_id: str, # Ajouté
    wfo_task_run_id: str, # Ajouté
    fold_id: int,
    df_is_enriched: pd.DataFrame,
    df_oos_enriched: pd.DataFrame,
    symbol_exchange_info: Dict[str, Any],
    fold_output_dir: Path # Répertoire de sortie spécifique à ce fold
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Placeholder pour l'orchestration de l'optimisation IS et de la validation OOS pour un fold.
    Dans une implémentation réelle, appellerait StudyManager, ObjectiveEvaluator, ResultsAnalyzer.
    """
    log_prefix_fold_orch = f"[{strategy_name}/{pair_symbol}/{cli_context_label}][Task:{wfo_task_run_id}][Fold_{fold_id}][FoldOrchPlaceholder]"
    logger.warning(f"{log_prefix_fold_orch} Utilisation de run_fold_optimization_and_validation_placeholder.")

    # Simuler une optimisation et une validation réussies
    # Générer des paramètres factices basés sur la config de la stratégie si possible
    selected_params: Dict[str, Any] = {"placeholder_param": f"val_fold_{fold_id}"}
    strat_cfg = app_config.strategies_config.strategies.get(strategy_name)
    if strat_cfg and strat_cfg.default_params:
        selected_params.update(strat_cfg.default_params) # Commencer avec les défauts
        if strat_cfg.params_space: # Modifier quelques params si espace défini
            for p_name, p_detail in list(strat_cfg.params_space.items())[:2]: # Prendre les 2 premiers
                if p_detail.type == "int" and p_detail.low is not None:
                    selected_params[p_name] = p_detail.low + fold_id # Varier par fold
                elif p_detail.type == "float" and p_detail.low is not None:
                    selected_params[p_name] = p_detail.low + fold_id * 0.1
                elif p_detail.type == "categorical" and p_detail.choices:
                    selected_params[p_name] = p_detail.choices[fold_id % len(p_detail.choices)]

    oos_metrics = {
        "Total Net PnL USDC": 100.0 + (fold_id * 10) - (len(pair_symbol) * 5), # Métrique factice
        "Sharpe Ratio": 1.0 + (fold_id * 0.1) - (len(cli_context_label) * 0.01),
        "Max Drawdown Pct": 10.0 - fold_id + (len(strategy_name) * 0.5),
        "Total Trades": 5 + fold_id
    }

    # Simuler la création de fichiers d'artefacts que le vrai module créerait
    try:
        # Fichiers liés à Optuna (IS)
        (fold_output_dir / f"optuna_is_study_{strategy_name}_{pair_symbol}_fold_{fold_id}.db").touch()
        optuna_summary_data = {
            "study_name": f"{strategy_name}_{pair_symbol}_{cli_context_label}_fold_{fold_id}_is_opt",
            "best_params_is_placeholder": selected_params,
            "best_value_is_placeholder": oos_metrics["Total Net PnL USDC"] * 0.8 # Simuler une valeur IS
        }
        with open(fold_output_dir / f"optuna_study_summary_fold_{fold_id}.json", "w") as f:
            json.dump(optuna_summary_data, f, indent=4, default=str)
        
        # Fichier de résumé de la validation OOS
        oos_validation_summary_data = [{
            "is_trial_number_placeholder": 0,
            "is_trial_params": selected_params,
            "oos_metrics": oos_metrics,
            "oos_detailed_trades_log_placeholder": [{"trade": 1}, {"trade": 2}]
        }]
        with open(fold_output_dir / f"oos_validation_summary_TOP_N_TRIALS_fold_{fold_id}.json", "w") as f:
            json.dump(oos_validation_summary_data, f, indent=4, default=str)
        logger.info(f"{log_prefix_fold_orch} Fichiers d'artefacts factices créés pour le fold {fold_id} dans {fold_output_dir}")
    except Exception as e_write_dummy:
        logger.error(f"{log_prefix_fold_orch} Erreur lors de la création des fichiers factices pour le fold {fold_id}: {e_write_dummy}")

    return selected_params, oos_metrics

# --- Fin des Placeholders ---


class WFOManager:
    """
    Gère le processus complet d'Optimisation Walk-Forward (WFO) pour une seule
    combinaison stratégie/paire/contexte CLI.
    """
    def __init__(self,
                 app_config: AppConfig,
                 strategy_name: str,
                 pair_symbol: str,
                 cli_context_label: str,
                 orchestrator_run_id: str,
                 # Les répertoires sont maintenant passés par le task_executor
                 wfo_task_log_run_dir: Path,
                 wfo_task_results_root_dir: Path,
                 wfo_task_run_id: str
                ):
        self.app_config = app_config
        self.strategy_name = strategy_name
        self.pair_symbol = pair_symbol.upper()
        self.cli_context_label = cli_context_label
        self.orchestrator_run_id = orchestrator_run_id
        self.wfo_task_run_id = wfo_task_run_id

        self.wfo_task_log_run_dir = wfo_task_log_run_dir
        self.wfo_task_results_root_dir = wfo_task_results_root_dir

        self.log_prefix = f"[{self.strategy_name}/{self.pair_symbol}/{self.cli_context_label}][Task:{self.wfo_task_run_id}][WFOManager]"

        # Récupération des configurations spécifiques
        self.wfo_settings: WfoSettings = self.app_config.global_config.wfo_settings
        self.strategy_params_config: Optional[StrategyParamsConfig] = self.app_config.strategies_config.strategies.get(self.strategy_name)
        
        if not self.strategy_params_config:
            logger.critical(f"{self.log_prefix} Configuration pour la stratégie '{self.strategy_name}' non trouvée.")
            raise ValueError(f"Configuration pour la stratégie '{self.strategy_name}' non trouvée.")

        logger.info(f"{self.log_prefix} WFOManager initialisé.")
        logger.debug(f"{self.log_prefix} Répertoire de log de la tâche : {self.wfo_task_log_run_dir}")
        logger.debug(f"{self.log_prefix} Répertoire de résultats de la tâche : {self.wfo_task_results_root_dir}")
        logger.debug(f"{self.log_prefix} WFO Settings: {self.wfo_settings}")
        logger.debug(f"{self.log_prefix} Strategy Params Config: {self.strategy_params_config.params_space if self.strategy_params_config else 'N/A'}")


    def run_single_wfo_task(self) -> Dict[str, Any]:
        """
        Exécute le processus WFO complet pour la stratégie, la paire et le contexte configurés.

        Returns:
            Dict[str, Any]: Un dictionnaire contenant le statut de la tâche et les chemins
                            vers les artefacts principaux générés.
        """
        logger.info(f"{self.log_prefix} Démarrage de l'exécution de la tâche WFO.")
        task_final_status = "FAILURE_INIT"
        wfo_summary_file_path_str: Optional[str] = None
        # Les chemins pour live_config et performance_report seront déterminés par le générateur de rapports.
        # WFOManager se concentre sur la génération du wfo_strategy_pair_summary.json.

        try:
            # 1. Charger les données historiques enrichies
            logger.info(f"{self.log_prefix} Chargement des données historiques enrichies pour {self.pair_symbol}...")
            df_enriched = load_enriched_historical_data_placeholder(self.pair_symbol, self.app_config)
            if df_enriched.empty:
                raise ValueError(f"Les données enrichies pour {self.pair_symbol} sont vides.")
            logger.info(f"{self.log_prefix} Données historiques enrichies chargées. Shape: {df_enriched.shape}")

            # 2. Obtenir les informations de l'exchange pour la paire
            logger.info(f"{self.log_prefix} Obtention des informations de l'exchange pour {self.pair_symbol}...")
            symbol_exchange_info = get_symbol_exchange_info_placeholder(self.pair_symbol, self.app_config)
            if not symbol_exchange_info:
                raise ValueError(f"Informations de l'exchange non trouvées pour {self.pair_symbol}.")
            logger.info(f"{self.log_prefix} Informations de l'exchange obtenues pour {self.pair_symbol}.")

            # 3. Générer les folds WFO
            logger.info(f"{self.log_prefix} Génération des folds WFO...")
            fold_generator = FoldGeneratorPlaceholder(self.wfo_settings)
            
            hist_period_config: HistoricalPeriod = self.app_config.data_config.historical_period
            global_start_ts = pd.to_datetime(hist_period_config.start_date, utc=True, errors='coerce') if hist_period_config.start_date else None
            global_end_ts = pd.to_datetime(hist_period_config.end_date, utc=True, errors='coerce') if hist_period_config.end_date else None

            folds = fold_generator.generate_folds(df_enriched, global_start_date_config=global_start_ts, global_end_date_config=global_end_ts)
            if not folds:
                raise ValueError("Aucun fold WFO n'a pu être généré.")
            logger.info(f"{self.log_prefix} {len(folds)} folds WFO générés.")

            # 4. Traiter chaque fold
            all_fold_summaries: List[Dict[str, Any]] = []
            for df_is_fold, df_oos_fold, fold_id, is_start_ts, is_end_ts, oos_start_ts, oos_end_ts in folds:
                fold_log_prefix = f"{self.log_prefix}[Fold_{fold_id}]"
                logger.info(f"{fold_log_prefix} Traitement du fold. IS: {is_start_ts} à {is_end_ts}, OOS: {oos_start_ts} à {oos_end_ts}.")

                fold_output_dir = self.wfo_task_log_run_dir / f"fold_{fold_id}"
                fold_output_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"{fold_log_prefix} Appel de run_fold_optimization_and_validation...")
                selected_params, oos_metrics = run_fold_optimization_and_validation_placeholder(
                    app_config=self.app_config,
                    strategy_name=self.strategy_name,
                    pair_symbol=self.pair_symbol,
                    cli_context_label=self.cli_context_label,
                    orchestrator_run_id=self.orchestrator_run_id,
                    wfo_task_run_id=self.wfo_task_run_id,
                    fold_id=fold_id,
                    df_is_enriched=df_is_fold,
                    df_oos_enriched=df_oos_fold,
                    symbol_exchange_info=symbol_exchange_info,
                    fold_output_dir=fold_output_dir
                )
                
                fold_status_msg = "COMPLETED_SUCCESS" if selected_params and oos_metrics else "COMPLETED_NO_VALID_PARAMS_FROM_OOS"
                if selected_params is None and oos_metrics is None:
                    fold_status_msg = "FAILED_FOLD_OPTIMIZATION_OR_VALIDATION"
                
                logger.info(f"{fold_log_prefix} Traitement du fold terminé. Statut: {fold_status_msg}")
                all_fold_summaries.append({
                    "fold_id": fold_id,
                    "status": fold_status_msg,
                    "is_period_start_utc": is_start_ts.isoformat() if pd.notna(is_start_ts) else None,
                    "is_period_end_utc": is_end_ts.isoformat() if pd.notna(is_end_ts) else None,
                    "oos_period_start_utc": oos_start_ts.isoformat() if pd.notna(oos_start_ts) else None,
                    "oos_period_end_utc": oos_end_ts.isoformat() if pd.notna(oos_end_ts) else None,
                    "selected_is_params_for_oos": selected_params, # Paramètres IS qui ont donné les meilleurs résultats OOS
                    "representative_oos_metrics": oos_metrics # Métriques OOS pour ces paramètres
                })

            # 5. Sauvegarder le résumé WFO pour cette tâche
            wfo_task_summary_content = {
                "strategy_name": self.strategy_name,
                "pair_symbol": self.pair_symbol,
                "cli_context_label": self.cli_context_label,
                "orchestrator_run_id": self.orchestrator_run_id,
                "wfo_task_run_id": self.wfo_task_run_id,
                "wfo_task_execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "wfo_settings_used": self.wfo_settings.__dict__, # Sauvegarder les settings WFO utilisés
                "folds_data": all_fold_summaries
            }
            
            summary_file = self.wfo_task_log_run_dir / "wfo_strategy_pair_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(wfo_task_summary_content, f, indent=4, default=str)
            wfo_summary_file_path_str = str(summary_file.resolve())
            logger.info(f"{self.log_prefix} Résumé WFO de la tâche sauvegardé dans : {wfo_summary_file_path_str}")
            
            task_final_status = "SUCCESS_WFO_COMPLETED"
            
            # Les fichiers live_config.json et performance_report.md sont générés par
            # le module `src.reporting.generator` en utilisant ce fichier de résumé.
            # WFOManager n'est pas directement responsable de leur création.
            # Il fournit le résumé nécessaire à leur génération.

        except FileNotFoundError as e_fnf:
            logger.error(f"{self.log_prefix} Fichier non trouvé durant l'exécution de la tâche WFO : {e_fnf}", exc_info=True)
            task_final_status = f"FAILURE_FILE_NOT_FOUND: {e_fnf}"
        except ValueError as e_val: # Erreurs de configuration ou de données
            logger.error(f"{self.log_prefix} Erreur de valeur durant l'exécution de la tâche WFO : {e_val}", exc_info=True)
            task_final_status = f"FAILURE_VALUE_ERROR: {e_val}"
        except Exception as e_unexp:
            logger.critical(f"{self.log_prefix} Erreur inattendue majeure durant l'exécution de la tâche WFO : {e_unexp}", exc_info=True)
            task_final_status = f"FAILURE_UNEXPECTED_ERROR: {e_unexp}"

        return {
            "status": task_final_status,
            "wfo_strategy_pair_summary_file": wfo_summary_file_path_str,
            # Les chemins suivants sont retournés à titre indicatif, leur création est gérée ailleurs.
            "expected_live_config_file_location": str((self.wfo_task_results_root_dir / "live_config.json").resolve()),
            "expected_performance_report_file_location": str((self.wfo_task_results_root_dir / "performance_report.md").resolve())
        }

