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
    from src.config.loader import AppConfig # AppConfig est maintenant plus riche
    from src.config.definitions import (
        WfoSettings,
        StrategyParamsConfig, # Pour le typage, bien que strategy_config_dict soit utilisé
        HistoricalPeriod
    )
    from src.utils.file_utils import save_json # Utilitaire pour sauvegarder JSON
    # Importer les vraies fonctions/classes
    from src.data.data_loader import load_enriched_historical_data
    from src.common.exchange_info_provider import get_symbol_exchange_info
    from src.backtesting.wfo.fold_generator import WfoFoldGenerator # Le générateur refactorisé
    from src.backtesting.optimization.fold_orchestrator import run_fold_optimization_and_validation # L'orchestrateur de fold refactorisé
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).critical(f"WFOManager: Erreur d'importation critique: {e}. Vérifiez PYTHONPATH.")
    raise

logger = logging.getLogger(__name__)

class WFOManager:
    """
    Gère le processus complet d'Optimisation Walk-Forward (WFO) pour une seule
    combinaison stratégie/paire/contexte CLI.
    Cette classe est instanciée et exécutée par `run_wfo_task_wrapper` dans
    un processus fils.
    """
    def __init__(self,
                 app_config: AppConfig,
                 strategy_name: str,
                 pair_symbol: str,
                 cli_context_label: str,
                 orchestrator_run_id: str,
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

        self.log_prefix = f"[{self.strategy_name}/{self.pair_symbol}/{self.cli_context_label}][Task:{self.wfo_task_run_id}][WFOManagerV2]"

        self.wfo_settings: WfoSettings = self.app_config.global_config.wfo_settings
        
        strategy_params_config_obj = self.app_config.strategies_config.strategies.get(self.strategy_name)
        if strategy_params_config_obj:
            import dataclasses
            self.strategy_config_dict: Dict[str, Any] = dataclasses.asdict(strategy_params_config_obj)
        else:
            msg = f"Configuration pour la stratégie '{self.strategy_name}' non trouvée dans AppConfig."
            logger.critical(f"{self.log_prefix} {msg}")
            raise ValueError(msg)

        logger.info(f"{self.log_prefix} WFOManager V2 initialisé.")
        logger.debug(f"{self.log_prefix} Répertoire de log de la tâche: {self.wfo_task_log_run_dir}")
        logger.debug(f"{self.log_prefix} WFO Settings utilisés: {self.wfo_settings}") # WfoSettings inclut maintenant les nouveaux params

    def run_single_wfo_task(self) -> Dict[str, Any]:
        """
        Exécute le processus WFO complet pour la stratégie, la paire et le contexte configurés.

        Returns:
            Dict[str, Any]: Un dictionnaire contenant le statut de la tâche et le chemin
                            vers le fichier de résumé WFO de la tâche.
        """
        logger.info(f"{self.log_prefix} Démarrage de l'exécution de la tâche WFO V2.")
        task_final_status = "FAILURE_INIT"
        wfo_summary_file_path_str: Optional[str] = None

        try:
            # 1. Charger les données historiques enrichies
            logger.info(f"{self.log_prefix} Chargement des données historiques enrichies pour {self.pair_symbol}...")
            df_enriched = load_enriched_historical_data(self.pair_symbol, self.app_config) # Appel réel
            if df_enriched.empty:
                logger.error(f"{self.log_prefix} Les données enrichies pour {self.pair_symbol} sont vides. Abandon.")
                raise ValueError(f"Données enrichies vides pour {self.pair_symbol}.")
            logger.info(f"{self.log_prefix} Données historiques enrichies chargées. Shape: {df_enriched.shape}, Index: {df_enriched.index.min()} à {df_enriched.index.max()}")

            # 2. Obtenir les informations de l'exchange pour la paire
            logger.info(f"{self.log_prefix} Obtention des informations de l'exchange pour {self.pair_symbol}...")
            symbol_exchange_info = get_symbol_exchange_info(self.pair_symbol, self.app_config) # Appel réel
            if not symbol_exchange_info:
                logger.error(f"{self.log_prefix} Informations de l'exchange non trouvées pour {self.pair_symbol}. Abandon.")
                raise ValueError(f"Informations d'exchange non trouvées pour {self.pair_symbol}.")
            logger.info(f"{self.log_prefix} Informations de l'exchange obtenues pour {self.pair_symbol}.")

            # 3. Générer les folds WFO
            logger.info(f"{self.log_prefix} Génération des folds WFO avec WfoFoldGenerator refactorisé...")
            fold_generator = WfoFoldGenerator(self.wfo_settings) # Utilise les WfoSettings étendus
            
            hist_period_config: HistoricalPeriod = self.app_config.data_config.historical_period
            global_start_ts_cfg = pd.to_datetime(hist_period_config.start_date, utc=True, errors='coerce') if hist_period_config.start_date else None
            global_end_ts_cfg = pd.to_datetime(hist_period_config.end_date, utc=True, errors='coerce') if hist_period_config.end_date and str(hist_period_config.end_date).lower() != "now" else None

            folds = fold_generator.generate_folds( # Cette méthode est maintenant un dispatcher
                df_enriched_data=df_enriched,
                is_total_start_ts_config=global_start_ts_cfg,
                oos_total_end_ts_config=global_end_ts_cfg
            )
            if not folds:
                logger.error(f"{self.log_prefix} Aucun fold WFO n'a pu être généré. Abandon.")
                raise ValueError("Aucun fold WFO généré.")
            logger.info(f"{self.log_prefix} {len(folds)} folds WFO générés (type: {self.wfo_settings.fold_type.value}).")

            # 4. Traiter chaque fold
            all_fold_summaries: List[Dict[str, Any]] = []
            for df_is_fold, df_oos_fold, fold_id_num, is_start_ts, is_end_ts, oos_start_ts, oos_end_ts in folds:
                fold_log_prefix_loop = f"{self.log_prefix}[Fold_{fold_id_num}]"
                logger.info(f"{fold_log_prefix_loop} Traitement du fold. IS: [{is_start_ts} - {is_end_ts}], OOS: [{oos_start_ts} - {oos_end_ts}].")

                fold_specific_output_dir = self.wfo_task_log_run_dir / f"fold_{fold_id_num}"
                fold_specific_output_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"{fold_log_prefix_loop} Appel de run_fold_optimization_and_validation (refactorisé)...")
                # run_fold_optimization_and_validation prend maintenant AppConfig
                selected_params_for_fold, representative_oos_metrics_for_fold = run_fold_optimization_and_validation(
                    app_config=self.app_config, # Passer l'AppConfig complète
                    strategy_name=self.strategy_name,
                    strategy_config_dict=self.strategy_config_dict,
                    data_1min_cleaned_is_slice=df_is_fold,
                    data_1min_cleaned_oos_slice=df_oos_fold,
                    output_dir_fold=fold_specific_output_dir,
                    pair_symbol=self.pair_symbol,
                    symbol_info_data=symbol_exchange_info,
                    run_id=self.wfo_task_run_id,
                    context_label=self.cli_context_label,
                    fold_id_numeric=fold_id_num
                )
                
                fold_status_msg = "COMPLETED" # Statut de base
                if selected_params_for_fold is None or representative_oos_metrics_for_fold is None:
                    fold_status_msg = "COMPLETED_NO_VALID_PARAMS_OR_METRICS"
                elif isinstance(representative_oos_metrics_for_fold, dict) and \
                     representative_oos_metrics_for_fold.get("status_oos", "").startswith("FAILURE"):
                    fold_status_msg = f"COMPLETED_WITH_OOS_ISSUES: {representative_oos_metrics_for_fold['status_oos']}"
                
                logger.info(f"{fold_log_prefix_loop} Traitement du fold terminé. Statut: {fold_status_msg}")
                all_fold_summaries.append({
                    "fold_id": fold_id_num,
                    "status": fold_status_msg,
                    "is_period_start_utc": is_start_ts.isoformat() if pd.notna(is_start_ts) else None,
                    "is_period_end_utc": is_end_ts.isoformat() if pd.notna(is_end_ts) else None,
                    "oos_period_start_utc": oos_start_ts.isoformat() if pd.notna(oos_start_ts) else None,
                    "oos_period_end_utc": oos_end_ts.isoformat() if pd.notna(oos_end_ts) else None,
                    "selected_is_params_for_oos": selected_params_for_fold,
                    "representative_oos_metrics": representative_oos_metrics_for_fold
                })

            # 5. Sauvegarder le résumé WFO pour cette tâche
            wfo_task_summary_content = {
                "strategy_name": self.strategy_name,
                "pair_symbol": self.pair_symbol,
                "cli_context_label": self.cli_context_label,
                "orchestrator_run_id": self.orchestrator_run_id,
                "wfo_task_run_id": self.wfo_task_run_id,
                "wfo_task_execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "wfo_settings_used": dataclasses.asdict(self.wfo_settings), # Sauvegarder WfoSettings
                "strategy_params_space_used": self.strategy_config_dict.get('params_space', {}),
                "folds_data": all_fold_summaries
            }
            
            summary_file = self.wfo_task_log_run_dir / "wfo_strategy_pair_summary.json"
            if not save_json(summary_file, wfo_task_summary_content, indent=4, default_serializer=str):
                logger.error(f"{self.log_prefix} Échec de la sauvegarde du fichier de résumé WFO de la tâche via save_json.")
                # Tenter un fallback simple
                try:
                    with open(summary_file, 'w', encoding='utf-8') as f_fallback:
                        json.dump(wfo_task_summary_content, f_fallback, indent=4, default=str)
                    logger.info(f"{self.log_prefix} Résumé WFO de la tâche sauvegardé (via fallback) dans : {summary_file}")
                except Exception as e_save_fallback:
                    logger.critical(f"{self.log_prefix} Échec critique de la sauvegarde du résumé WFO: {e_save_fallback}", exc_info=True)
                    raise IOError(f"Impossible de sauvegarder le résumé WFO: {e_save_fallback}") from e_save_fallback
            
            wfo_summary_file_path_str = str(summary_file.resolve())
            logger.info(f"{self.log_prefix} Résumé WFO de la tâche sauvegardé avec succès dans : {wfo_summary_file_path_str}")
            
            task_final_status = "SUCCESS_WFO_COMPLETED"
            
        except FileNotFoundError as e_fnf:
            logger.error(f"{self.log_prefix} Fichier non trouvé durant l'exécution de la tâche WFO : {e_fnf}", exc_info=True)
            task_final_status = f"FAILURE_FILE_NOT_FOUND: {str(e_fnf)}"
        except ValueError as e_val:
            logger.error(f"{self.log_prefix} Erreur de valeur durant l'exécution de la tâche WFO : {e_val}", exc_info=True)
            task_final_status = f"FAILURE_VALUE_ERROR: {str(e_val)}"
        except Exception as e_unexp:
            logger.critical(f"{self.log_prefix} Erreur inattendue majeure durant l'exécution de la tâche WFO : {e_unexp}", exc_info=True)
            task_final_status = f"FAILURE_UNEXPECTED_ERROR: {str(e_unexp)}"

        # Les chemins pour live_config et performance_report sont retournés à titre indicatif.
        # Leur création effective est gérée par le module de reporting (master_report_generator)
        # basé sur le summary_file.
        return {
            "status": task_final_status,
            "wfo_strategy_pair_summary_file": wfo_summary_file_path_str,
            # Ces chemins sont où le master_report_generator s'attend à les créer
            "expected_live_config_file_location": str((self.wfo_task_results_root_dir / "live_config.json").resolve()),
            "expected_performance_report_file_location": str((self.wfo_task_results_root_dir / "performance_report.md").resolve())
        }
