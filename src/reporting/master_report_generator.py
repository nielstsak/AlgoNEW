# src/reporting/master_report_generator.py
"""
Ce module agrège les résultats de toutes les tâches WFO d'un run d'orchestrateur
et génère les rapports finaux, y compris les fichiers live_config.json
et un rapport Markdown consolidé.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple, TYPE_CHECKING
from datetime import datetime, timezone

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.config.definitions import WfoSettings, StrategyParamsConfig

# Tentative d'importation d'Optuna pour les visualisations
try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_pareto_front
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None # type: ignore
    plot_optimization_history = plot_param_importances = plot_slice = plot_pareto_front = None # type: ignore
    OPTUNA_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Optuna non trouvé. Les visualisations Optuna ne seront pas générées."
    )

# Utilisation de file_utils pour sauvegarder JSON si disponible
try:
    from src.utils.file_utils import ensure_dir_exists, save_json # save_json est hypothétique
except ImportError:
    def ensure_dir_exists(path: Path) -> bool: # type: ignore
        path.mkdir(parents=True, exist_ok=True)
        return True
    def save_json(filepath: Path, data: Any) -> None: # type: ignore
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

logger = logging.getLogger(__name__)

def _format_metric(value: Any, precision: int = 4, is_pnl: bool = False) -> str:
    """Formate une métrique pour l'affichage dans le rapport Markdown."""
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating, np.float64)): # type: ignore
        if np.isnan(value): # type: ignore
            return "NaN"
        if np.isinf(value): # type: ignore
            return "Infinity" if value > 0 else "-Infinity"
        return f"{value:.{2 if is_pnl else precision}f}"
    if isinstance(value, (int, np.integer)): # type: ignore
        return str(value)
    return str(value)

def _generate_optuna_visualizations_for_fold(
    optuna_db_path: Path,
    output_viz_dir_for_fold: Path,
    study_name: str, # Nom complet de l'étude pour ce fold
    app_config: 'AppConfig' # Pour accéder à OptunaSettings si besoin
) -> None:
    """Génère et sauvegarde les visualisations Optuna pour une étude de fold donnée."""
    log_prefix = f"[OptunaViz][{study_name}]"
    if not OPTUNA_AVAILABLE:
        logger.warning(f"{log_prefix} Optuna non disponible, saut de la génération des visualisations.")
        return
    if not optuna_db_path.exists():
        logger.warning(f"{log_prefix} Fichier DB Optuna non trouvé à {optuna_db_path}. Pas de visualisations.")
        return

    ensure_dir_exists(output_viz_dir_for_fold)
    storage_url = f"sqlite:///{optuna_db_path.resolve()}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e_load_study:
        logger.error(f"{log_prefix} Échec du chargement de l'étude Optuna '{study_name}' depuis {storage_url}: {e_load_study}", exc_info=True)
        # Essayer de lister les études pour aider au débogage
        try:
            summaries = optuna.study.get_all_study_summaries(storage=storage_url)
            logger.info(f"{log_prefix} Études disponibles dans {optuna_db_path}: {[s.study_name for s in summaries]}")
        except Exception as e_list:
            logger.error(f"{log_prefix} Impossible de lister les études depuis {optuna_db_path}: {e_list}")
        return

    viz_functions_map = {
        "optimization_history": plot_optimization_history,
        "param_importances": plot_param_importances,
        "slice": plot_slice,
    }
    if len(study.directions) > 1: # Multi-objectifs
        viz_functions_map["pareto_front"] = plot_pareto_front

    for viz_name, viz_func in viz_functions_map.items():
        try:
            fig = None
            if viz_name == "slice" and study.best_trials: # plot_slice nécessite des paramètres
                # Tenter avec les paramètres du meilleur trial ou les plus importants
                important_params = []
                if hasattr(optuna, 'importance') and hasattr(optuna.importance, 'get_param_importances'):
                     try:
                        importances = optuna.importance.get_param_importances(study)
                        if importances:
                            important_params = [p_name for p_name, _ in sorted(importances.items(), key=lambda item: item[1], reverse=True)[:4]] # Top 4
                     except Exception: # Peut échouer si pas assez d'essais, etc.
                        pass
                if not important_params and study.best_trial: # Fallback sur les params du meilleur trial
                    important_params = list(study.best_trial.params.keys())[:4]
                
                if important_params:
                    fig = viz_func(study, params=important_params) # type: ignore
                else:
                    logger.info(f"{log_prefix} Pas de paramètres trouvés pour plot_slice. Saut.")
            elif viz_func: # Pour les autres fonctions ou si plot_slice n'a pas besoin de params spécifiques
                fig = viz_func(study) # type: ignore
            
            if fig:
                output_file = output_viz_dir_for_fold / f"{viz_name}.html"
                fig.write_html(str(output_file)) # type: ignore
                logger.info(f"{log_prefix} Visualisation Optuna '{viz_name}' sauvegardée : {output_file}")
        except Exception as e_viz:
            logger.error(f"{log_prefix} Échec de la génération/sauvegarde de la visualisation '{viz_name}': {e_viz}", exc_info=True)


def _process_single_wfo_task_summary(
    task_summary_path: Path, # Chemin vers wfo_strategy_pair_summary.json
    task_results_output_dir: Path, # Répertoire de sortie pour cette tâche (results/ORCH_ID/STRAT/PAIR/CONTEXT)
    task_log_dir: Path, # Répertoire de log pour cette tâche (logs/.../ORCH_ID/STRAT/PAIR/CONTEXT/TASK_ID)
    app_config: 'AppConfig'
) -> Optional[Dict[str, Any]]:
    """
    Traite le résumé d'une seule tâche WFO : sélectionne les meilleurs paramètres,
    génère live_config.json, un rapport Markdown spécifique à la tâche, et les visualisations Optuna.
    """
    try:
        with open(task_summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except Exception as e:
        logger.error(f"Échec du chargement du résumé de la tâche WFO {task_summary_path}: {e}", exc_info=True)
        return None

    strategy_name = summary_data.get("strategy_name", "UnknownStrategy")
    pair_symbol = summary_data.get("pair_symbol", "UnknownPair")
    cli_context_label = summary_data.get("cli_context_label", "UnknownContext")
    # orchestrator_run_id = summary_data.get("orchestrator_run_id") # Déjà connu
    wfo_task_run_id = summary_data.get("wfo_task_run_id")
    folds_data = summary_data.get("folds_data", [])

    log_prefix_task_proc = f"[ReportGenTask][{strategy_name}/{pair_symbol}/{cli_context_label}][Task:{wfo_task_run_id}]"
    logger.info(f"{log_prefix_task_proc} Traitement du résumé de la tâche WFO.")

    ensure_dir_exists(task_results_output_dir)

    # 1. Sélectionner les meilleurs paramètres et métriques OOS agrégées pour cette tâche
    best_params_for_live: Optional[Dict[str, Any]] = None
    best_oos_metrics_for_live: Optional[Dict[str, Any]] = None
    best_fold_id_for_live: Optional[int] = None
    
    metric_to_optimize = app_config.global_config.wfo_settings.metric_to_optimize
    is_maximize = app_config.global_config.wfo_settings.optimization_direction.lower() == "maximize"
    current_best_metric_value = -float('inf') if is_maximize else float('inf')

    valid_folds_with_oos_results = [
        f for f in folds_data 
        if f.get("status", "").startswith("COMPLETED") and f.get("representative_oos_metrics") and f.get("selected_is_params_for_oos")
    ]

    if not valid_folds_with_oos_results:
        logger.warning(f"{log_prefix_task_proc} Aucun fold avec des résultats OOS valides trouvé pour la sélection live_config.")
    else:
        for fold_summary in valid_folds_with_oos_results:
            oos_metrics = fold_summary["representative_oos_metrics"]
            metric_val = oos_metrics.get(metric_to_optimize)
            if metric_val is not None and isinstance(metric_val, (int, float)) and np.isfinite(metric_val):
                if (is_maximize and metric_val > current_best_metric_value) or \
                   (not is_maximize and metric_val < current_best_metric_value):
                    current_best_metric_value = metric_val
                    best_params_for_live = fold_summary["selected_is_params_for_oos"]
                    best_oos_metrics_for_live = oos_metrics
                    best_fold_id_for_live = fold_summary.get("fold_id")
        
        if best_params_for_live:
            logger.info(f"{log_prefix_task_proc} Meilleurs paramètres pour live_config sélectionnés du Fold {best_fold_id_for_live} "
                        f"basé sur OOS '{metric_to_optimize}': {current_best_metric_value}")
            
            # Générer live_config.json
            strategy_params_config = app_config.strategies_config.strategies.get(strategy_name)
            if strategy_params_config:
                live_config_content = {
                    "strategy_name_base": strategy_name, # Nom de base de la stratégie
                    "strategy_script_reference": strategy_params_config.script_reference,
                    "strategy_class_name": strategy_params_config.class_name,
                    "pair_symbol": pair_symbol,
                    "context_label": cli_context_label, # Contexte CLI utilisé pour ce WFO
                    "best_params": best_params_for_live,
                    "oos_performance_summary_selection": best_oos_metrics_for_live,
                    "source_orchestrator_run_id": summary_data.get("orchestrator_run_id"),
                    "source_wfo_task_run_id": wfo_task_run_id,
                    "source_fold_id_for_params": best_fold_id_for_live,
                    "generation_timestamp_utc": datetime.now(timezone.utc).isoformat()
                }
                live_config_file = task_results_output_dir / "live_config.json"
                save_json(live_config_file, live_config_content)
                logger.info(f"{log_prefix_task_proc} Fichier live_config.json généré : {live_config_file}")
            else:
                logger.error(f"{log_prefix_task_proc} Configuration de la stratégie '{strategy_name}' non trouvée. Impossible de générer live_config.json.")
        else:
            logger.warning(f"{log_prefix_task_proc} Aucun paramètre n'a pu être sélectionné pour live_config.json basé sur OOS.")

    # 2. Générer le rapport Markdown spécifique à la tâche
    task_report_path = task_results_output_dir / "performance_report_task.md"
    with open(task_report_path, "w", encoding="utf-8") as f_task_report:
        f_task_report.write(f"# Rapport WFO pour Tâche : {strategy_name} / {pair_symbol} / {cli_context_label}\n\n")
        f_task_report.write(f"* Run Orchestrateur ID: `{summary_data.get('orchestrator_run_id')}`\n")
        f_task_report.write(f"* Tâche WFO Run ID: `{wfo_task_run_id}`\n")
        f_task_report.write(f"* Fichier résumé source: `{task_summary_path.name}` (dans le répertoire de log de la tâche)\n\n")

        if best_params_for_live:
            f_task_report.write(f"## Paramètres Sélectionnés pour Déploiement (Live Config)\n")
            f_task_report.write(f"* Sélectionnés depuis le Fold ID: {best_fold_id_for_live}\n")
            f_task_report.write(f"* Métrique OOS optimisée ('{metric_to_optimize}'): {_format_metric(current_best_metric_value, is_pnl='PnL' in metric_to_optimize)}\n")
            f_task_report.write("```json\n")
            f_task_report.write(json.dumps(best_params_for_live, indent=2, default=str) + "\n")
            f_task_report.write("```\n\n")
            f_task_report.write(f"[Voir live_config.json](./live_config.json)\n\n")
        else:
            f_task_report.write("## Aucun paramètre sélectionné pour déploiement basé sur OOS.\n\n")

        f_task_report.write("## Performance des Folds\n\n")
        f_task_report.write("| Fold ID | Statut | IS Début | IS Fin | OOS Début | OOS Fin | Métrique OOS Optimisée | Params IS (pour OOS) | Visualisations IS |\n")
        f_task_report.write("|:-------:|:------:|:--------:|:------:|:---------:|:-------:|:--------------------:|:--------------------:|:-----------------:|\n")

        for fold_summary in folds_data:
            fold_id = fold_summary.get("fold_id", "N/A")
            status = fold_summary.get("status", "INCONNU")
            is_s = pd.to_datetime(fold_summary.get("is_period_start_utc")).strftime('%Y-%m-%d') if fold_summary.get("is_period_start_utc") else "N/A"
            is_e = pd.to_datetime(fold_summary.get("is_period_end_utc")).strftime('%Y-%m-%d') if fold_summary.get("is_period_end_utc") else "N/A"
            oos_s = pd.to_datetime(fold_summary.get("oos_period_start_utc")).strftime('%Y-%m-%d') if fold_summary.get("oos_period_start_utc") else "N/A"
            oos_e = pd.to_datetime(fold_summary.get("oos_period_end_utc")).strftime('%Y-%m-%d') if fold_summary.get("oos_period_end_utc") else "N/A"
            
            oos_metrics_fold = fold_summary.get("representative_oos_metrics", {})
            metric_val_fold = oos_metrics_fold.get(metric_to_optimize) if oos_metrics_fold else None
            metric_display = _format_metric(metric_val_fold, is_pnl='PnL' in metric_to_optimize)
            
            params_fold = fold_summary.get("selected_is_params_for_oos", {})
            params_display_parts = [f"{k}: {_format_metric(v,2)}" for k,v in params_fold.items()] if params_fold else ["N/A"]
            params_str = ", ".join(params_display_parts)
            if len(params_str) > 70: params_str = params_str[:67] + "..."

            viz_link = f"[Viz Fold {fold_id}](./optuna_visualizations/fold_{fold_id}/optimization_history.html)"
            
            f_task_report.write(f"| {fold_id} | {status} | {is_s} | {is_e} | {oos_s} | {oos_e} | {metric_display} | `{params_str}` | {viz_link} |\n")
        
        logger.info(f"{log_prefix_task_proc} Rapport Markdown spécifique à la tâche généré : {task_report_path}")

    # 3. Générer les visualisations Optuna pour chaque fold de cette tâche
    if OPTUNA_AVAILABLE:
        optuna_viz_base_dir = task_results_output_dir / "optuna_visualizations"
        for fold_summary in folds_data:
            fold_id = fold_summary.get("fold_id")
            if fold_id is None: continue

            # Chemin vers la DB Optuna du fold (dans le répertoire de log de la tâche)
            # Le nom de la DB est construit par OptunaStudyManager
            fold_log_dir = task_log_dir / f"fold_{fold_id}"
            # Le nom de l'étude est : f"{strategy_name}_{pair_symbol}_{fold_dir_name}_is_opt"
            # où fold_dir_name est "fold_X"
            study_name_fold = f"{strategy_name}_{pair_symbol}_fold_{fold_id}_is_opt"
            optuna_db_file = fold_log_dir / f"optuna_is_study_{study_name_fold.replace('_is_opt', '')}.db" # Reconstruire le nom de la DB
            
            if optuna_db_file.exists():
                output_viz_dir_for_fold = optuna_viz_base_dir / f"fold_{fold_id}"
                _generate_optuna_visualizations_for_fold(
                    optuna_db_path=optuna_db_file,
                    output_viz_dir_for_fold=output_viz_dir_for_fold,
                    study_name=study_name_fold, # Nom complet de l'étude
                    app_config=app_config
                )
            else:
                logger.warning(f"{log_prefix_task_proc}[Fold:{fold_id}] Fichier DB Optuna non trouvé à {optuna_db_file}. Visualisations sautées.")
    
    return {
        "strategy_name": strategy_name,
        "pair_symbol": pair_symbol,
        "cli_context_label": cli_context_label,
        "wfo_task_run_id": wfo_task_run_id,
        "best_overall_oos_metric_value_for_task": current_best_metric_value if best_params_for_live else None,
        "best_params_for_task": best_params_for_live,
        "source_fold_id_for_best_params": best_fold_id_for_live,
        "task_summary_file_log_path": str(task_summary_path.resolve()),
        "task_report_file_results_path": str(task_report_path.resolve()),
        "live_config_file_results_path": str(task_results_output_dir / "live_config.json") if best_params_for_live else None
    }


def generate_all_reports(
    log_dir: Path, # Répertoire du run de l'orchestrateur (ex: logs/backtest_optimization/ORCH_RUN_ID/)
    results_dir: Path, # Répertoire de sortie pour ce run (ex: results/ORCH_RUN_ID/)
    app_config: 'AppConfig'
) -> None:
    """
    Agrège les résultats de toutes les tâches WFO d'un run d'orchestrateur et
    génère les rapports finaux (rapport Markdown consolidé, fichiers live_config.json).

    Args:
        log_dir (Path): Répertoire de log du run de l'orchestrateur.
        results_dir (Path): Répertoire où les rapports et fichiers live_config.json
                            doivent être sauvegardés pour ce run d'orchestrateur.
        app_config (AppConfig): Instance de configuration globale.
    """
    main_report_log_prefix = f"[MasterReportGen][OrchRun:{log_dir.name}]"
    logger.info(f"{main_report_log_prefix} Démarrage de la génération de tous les rapports.")
    logger.info(f"{main_report_log_prefix} Lecture des logs depuis : {log_dir}")
    logger.info(f"{main_report_log_prefix} Écriture des résultats dans : {results_dir}")

    ensure_dir_exists(results_dir)

    # Copier le run_config_orchestrator.json (s'il existe) dans le répertoire results
    orchestrator_config_log_path = log_dir / "run_config_orchestrator.json"
    if orchestrator_config_log_path.is_file():
        try:
            shutil.copy2(orchestrator_config_log_path, results_dir / "run_config_orchestrator.json")
            logger.info(f"{main_report_log_prefix} Copie de {orchestrator_config_log_path.name} vers {results_dir}")
        except Exception as e_copy:
            logger.warning(f"{main_report_log_prefix} Échec de la copie de {orchestrator_config_log_path.name}: {e_copy}")


    # 1. Trouver et agréger tous les résumés de tâches WFO
    all_task_summaries_info: List[Dict[str, Any]] = []
    
    # Structure attendue: log_dir / STRATEGIE / PAIRE / CONTEXTE_CLI / TASK_RUN_ID / wfo_strategy_pair_summary.json
    for strat_dir in log_dir.iterdir():
        if not strat_dir.is_dir(): continue
        for pair_dir in strat_dir.iterdir():
            if not pair_dir.is_dir(): continue
            for context_cli_dir in pair_dir.iterdir():
                if not context_cli_dir.is_dir(): continue
                for task_run_id_dir in context_cli_dir.iterdir():
                    if not task_run_id_dir.is_dir(): continue # S'assurer que c'est bien le répertoire TASK_RUN_ID
                    
                    summary_file = task_run_id_dir / "wfo_strategy_pair_summary.json"
                    if summary_file.is_file():
                        logger.info(f"{main_report_log_prefix} Traitement du fichier résumé : {summary_file}")
                        
                        # Déterminer le répertoire de sortie des résultats pour cette tâche
                        task_results_output_dir = results_dir / strat_dir.name / pair_dir.name / context_cli_dir.name
                        
                        processed_task_info = _process_single_wfo_task_summary(
                            task_summary_path=summary_file,
                            task_results_output_dir=task_results_output_dir,
                            task_log_dir=task_run_id_dir, # Chemin vers .../TASK_RUN_ID/
                            app_config=app_config
                        )
                        if processed_task_info:
                            all_task_summaries_info.append(processed_task_info)
                    else:
                        logger.debug(f"{main_report_log_prefix} Fichier wfo_strategy_pair_summary.json non trouvé dans {task_run_id_dir}")
    
    if not all_task_summaries_info:
        logger.warning(f"{main_report_log_prefix} Aucun résumé de tâche WFO trouvé dans {log_dir}. Le rapport global sera vide.")
        # Créer un rapport global vide ou avec un message
        global_report_path = results_dir / "global_wfo_report.md"
        with open(global_report_path, "w", encoding="utf-8") as f_global_report:
            f_global_report.write(f"# Rapport Global WFO - Run Orchestrateur: {log_dir.name}\n\n")
            f_global_report.write("Aucun résultat de tâche WFO n'a été trouvé ou traité.\n")
        logger.info(f"{main_report_log_prefix} Rapport global vide généré : {global_report_path}")
        return

    # 2. Générer le rapport Markdown consolidé global
    global_report_path = results_dir / "global_wfo_report.md"
    logger.info(f"{main_report_log_prefix} Génération du rapport Markdown global : {global_report_path}")
    
    # Trier les tâches pour le rapport global (par exemple, par meilleure métrique OOS)
    metric_to_sort_global = app_config.global_config.wfo_settings.metric_to_optimize
    is_maximize_global = app_config.global_config.wfo_settings.optimization_direction.lower() == "maximize"
    
    all_task_summaries_info.sort(
        key=lambda x: x.get("best_overall_oos_metric_value_for_task") if isinstance(x.get("best_overall_oos_metric_value_for_task"), (int, float)) else (-float('inf') if is_maximize_global else float('inf')),
        reverse=is_maximize_global
    )

    with open(global_report_path, "w", encoding="utf-8") as f_global:
        f_global.write(f"# Rapport Global d'Optimisation Walk-Forward\n\n")
        f_global.write(f"**Run Orchestrateur ID:** `{log_dir.name}`\n")
        f_global.write(f"**Date de Génération:** {datetime.now(timezone.utc).isoformat()}\n")
        f_global.write(f"**Nombre Total de Tâches WFO Traitées:** {len(all_task_summaries_info)}\n")
        f_global.write(f"**Métrique d'Optimisation Principale (OOS):** `{metric_to_sort_global}` (Direction: {'Maximiser' if is_maximize_global else 'Minimiser'})\n\n")

        f_global.write("## Résumé des Meilleures Tâches WFO (Stratégie/Paire/Contexte)\n\n")
        f_global.write("| Stratégie | Paire | Contexte CLI | Meilleure Métrique OOS | Params Sélectionnés (du Fold ID) | Fichier Live Config | Rapport de Tâche |\n")
        f_global.write("|:----------|:------|:-------------|:-----------------------:|:---------------------------------|:--------------------|:----------------:|\n")

        for task_info in all_task_summaries_info:
            strat = task_info.get("strategy_name", "N/A")
            pair = task_info.get("pair_symbol", "N/A")
            context = task_info.get("cli_context_label", "N/A")
            
            metric_val = task_info.get("best_overall_oos_metric_value_for_task")
            metric_display = _format_metric(metric_val, is_pnl='PnL' in metric_to_sort_global)
            
            params = task_info.get("best_params_for_task")
            source_fold = task_info.get("source_fold_id_for_best_params", "N/A")
            params_str = "N/A"
            if params:
                params_display_parts = [f"{k}: {_format_metric(v,2)}" for k,v in params.items()]
                params_str = f"`{', '.join(params_display_parts)}` (Fold: {source_fold})"
            if len(params_str) > 60: params_str = params_str[:57] + "...`"
            
            live_config_path_rel = None
            if task_info.get("live_config_file_results_path"):
                try:
                    live_config_path_rel = Path(task_info["live_config_file_results_path"]).relative_to(results_dir)
                    live_config_link = f"[{live_config_path_rel.name}](./{str(live_config_path_rel).replace(chr(92), '/')})"
                except ValueError: # Si pas relatif (ne devrait pas arriver)
                    live_config_link = f"[Fichier Live]({Path(task_info['live_config_file_results_path']).name})"
            else:
                live_config_link = "Non généré"

            task_report_path_rel = None
            if task_info.get("task_report_file_results_path"):
                try:
                    task_report_path_rel = Path(task_info["task_report_file_results_path"]).relative_to(results_dir)
                    task_report_link = f"[{task_report_path_rel.name}](./{str(task_report_path_rel).replace(chr(92), '/')})"
                except ValueError:
                    task_report_link = f"[Rapport Tâche]({Path(task_info['task_report_file_results_path']).name})"
            else:
                task_report_link = "N/A"

            f_global.write(f"| {strat} | {pair} | {context} | {metric_display} | {params_str} | {live_config_link} | {task_report_link} |\n")
            
    logger.info(f"{main_report_log_prefix} Rapport Markdown global généré : {global_report_path}")

    # 3. Sauvegarder un fichier de statut pour ce run de génération de rapports
    run_status_file = results_dir / "report_generation_run_status.json"
    status_content = {
        "orchestrator_run_id_processed": log_dir.name,
        "report_generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS" if all_task_summaries_info else "SUCCESS_NO_TASKS_FOUND",
        "total_wfo_tasks_found_in_log_dir": len(all_task_summaries_info), # Nombre de wfo_strategy_pair_summary.json trouvés
        "global_report_file": str(global_report_path.resolve()),
        "processed_tasks_details": all_task_summaries_info # Inclure les infos agrégées
    }
    save_json(run_status_file, status_content)
    logger.info(f"{main_report_log_prefix} Statut de la génération des rapports sauvegardé : {run_status_file}")

    logger.info(f"{main_report_log_prefix} Génération de tous les rapports terminée.")

