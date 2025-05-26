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
    from src.config.definitions import WfoSettings, StrategyParamsConfig # Pour le typage

# Tentative d'importation d'Optuna pour les visualisations
try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
        plot_pareto_front,
        plot_contour
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None # type: ignore
    plot_optimization_history = plot_param_importances = plot_slice = plot_pareto_front = plot_contour = None # type: ignore
    OPTUNA_AVAILABLE = False
    # Le logger n'est pas encore configuré ici, mais load_all_configs le fera.
    # Un message sera loggué par la fonction _generate_optuna_visualizations_for_fold.

# Utilisation de file_utils pour sauvegarder JSON
try:
    from src.utils.file_utils import ensure_dir_exists, save_json
except ImportError:
    # Fallback simple si file_utils n'est pas disponible
    _bootstrap_logger_mrg = logging.getLogger(__name__ + "_bootstrap_mrg")
    _bootstrap_logger_mrg.warning("src.utils.file_utils non trouvé. Utilisation de fallbacks pour ensure_dir_exists et save_json.")
    def ensure_dir_exists(path: Path) -> bool: # type: ignore
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception: return False
    def save_json(filepath: Path, data: Any, indent: int = 2, default_serializer=str) -> bool: # type: ignore
        try:
            # S'assurer que le répertoire parent existe pour le fallback aussi
            ensure_dir_exists(filepath.parent)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=default_serializer)
            return True
        except Exception: return False

logger = logging.getLogger(__name__)

def _format_metric(value: Any, precision: int = 4, is_pnl: bool = False, is_pct: bool = False) -> str:
    """Formate une métrique pour l'affichage dans le rapport Markdown."""
    if value is None: return "N/A"
    if isinstance(value, (float, np.floating, np.float64)): # type: ignore
        if np.isnan(value): return "NaN" # type: ignore
        if np.isinf(value): return "Infinity" if value > 0 else "-Infinity" # type: ignore
        # Pourcentage
        if is_pct:
            return f"{value:.2f}%"
        # PnL ou autres montants
        return f"{value:.{2 if is_pnl else precision}f}"
    if isinstance(value, (int, np.integer)): return str(value) # type: ignore
    return str(value)

def _generate_optuna_visualizations_for_fold(
    optuna_db_path: Path,
    output_viz_dir_for_fold: Path,
    study_name: str,
    app_config: 'AppConfig'
) -> None:
    """Génère et sauvegarde les visualisations Optuna pour une étude de fold donnée."""
    log_prefix_viz = f"[OptunaViz][{study_name}]"
    if not OPTUNA_AVAILABLE:
        logger.warning(f"{log_prefix_viz} Optuna non disponible, saut de la génération des visualisations.")
        return
    if not optuna_db_path.is_file():
        logger.warning(f"{log_prefix_viz} Fichier DB Optuna non trouvé ou n'est pas un fichier à {optuna_db_path}. Pas de visualisations.")
        return

    ensure_dir_exists(output_viz_dir_for_fold)
    storage_url = f"sqlite:///{optuna_db_path.resolve()}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e_load_study:
        logger.error(f"{log_prefix_viz} Échec du chargement de l'étude Optuna '{study_name}' depuis {storage_url}: {e_load_study}", exc_info=True)
        # Tenter de lister les études pour aider au débogage
        try:
            summaries = optuna.study.get_all_study_summaries(storage=storage_url)
            logger.info(f"{log_prefix_viz} Études disponibles dans {optuna_db_path}: {[s.study_name for s in summaries]}")
        except Exception as e_list:
            logger.error(f"{log_prefix_viz} Impossible de lister les études depuis {optuna_db_path}: {e_list}")
        return

    if not study.trials:
        logger.info(f"{log_prefix_viz} L'étude '{study_name}' ne contient aucun essai. Visualisations sautées.")
        return

    study_objective_names: List[str] = []
    if hasattr(study, 'objective_names') and study.objective_names:
        study_objective_names = list(study.objective_names)
    elif app_config.global_config.optuna_settings.objectives_names:
        study_objective_names = app_config.global_config.optuna_settings.objectives_names
    
    if not study_objective_names and study.directions:
        study_objective_names = [f"objectif_{i}" for i in range(len(study.directions))]
        if study.directions:
             logger.warning(f"{log_prefix_viz} Noms d'objectifs non trouvés, utilisation de noms génériques: {study_objective_names}")

    viz_functions_map = {
        "optimization_history": plot_optimization_history,
        "param_importances": plot_param_importances,
        "slice": plot_slice,
        "contour": plot_contour
    }
    if len(study.directions) > 1:
        viz_functions_map["pareto_front"] = plot_pareto_front

    for viz_name, viz_func in viz_functions_map.items():
        if viz_func is None: continue
        try:
            if len(study.directions) > 1 and viz_name in ["optimization_history", "slice", "param_importances", "contour"]:
                for i, objective_name_target in enumerate(study_objective_names):
                    fig = None
                    # Définir une fonction target pour Optuna qui sélectionne le i-ème objectif
                    # S'assurer que t.values existe et a assez d'éléments
                    target_lambda = lambda t, idx=i: t.values[idx] if t.values and len(t.values) > idx else None

                    if viz_name == "slice" or viz_name == "contour":
                        important_params_names = []
                        try:
                            importances_for_target = optuna.importance.get_param_importances(study, target=lambda t: target_lambda(t, i), target_name=objective_name_target)
                            if importances_for_target:
                                important_params_names = [p_name for p_name, _ in sorted(importances_for_target.items(), key=lambda item: item[1], reverse=True)[:2]]
                        except Exception: pass # Peut échouer si pas assez de trials
                        
                        if not important_params_names and study.best_trials:
                            important_params_names = list(study.best_trials[0].params.keys())[:2]

                        if len(important_params_names) >= (1 if viz_name == "slice" else 2):
                             if viz_name == "slice": fig = viz_func(study, params=important_params_names, target=lambda t: target_lambda(t, i), target_name=objective_name_target)
                             elif viz_name == "contour" and len(important_params_names) >=2: fig = viz_func(study, params=important_params_names[:2], target=lambda t: target_lambda(t, i), target_name=objective_name_target)
                        else: logger.info(f"{log_prefix_viz} Pas assez de paramètres pour {viz_name} pour '{objective_name_target}'. Saut.")
                    
                    elif viz_name == "param_importances": fig = viz_func(study, target=lambda t: target_lambda(t, i), target_name=objective_name_target)
                    elif viz_name == "optimization_history": fig = viz_func(study, target=lambda t: target_lambda(t, i), target_name=objective_name_target)
                    
                    if fig:
                        output_file = output_viz_dir_for_fold / f"{viz_name}_{objective_name_target.replace(' ', '_').replace('/', '_')}.html"
                        fig.write_html(str(output_file))
                        logger.info(f"{log_prefix_viz} Visualisation '{viz_name}' pour objectif '{objective_name_target}' sauvegardée : {output_file}")
            else: # Mono-objectif ou fonction gérant bien le multi-objectif sans target (ex: pareto_front)
                fig = None
                if viz_name == "slice" or viz_name == "contour":
                    important_params_names = []
                    try:
                        importances = optuna.importance.get_param_importances(study)
                        if importances: important_params_names = [p_name for p_name, _ in sorted(importances.items(), key=lambda item: item[1], reverse=True)[:2]]
                    except Exception: pass
                    if not important_params_names and study.best_trials: important_params_names = list(study.best_trials[0].params.keys())[:2]
                    
                    if len(important_params_names) >= (1 if viz_name == "slice" else 2):
                        if viz_name == "slice": fig = viz_func(study, params=important_params_names)
                        elif viz_name == "contour" and len(important_params_names) >=2: fig = viz_func(study, params=important_params_names[:2])
                    else: logger.info(f"{log_prefix_viz} Pas assez de paramètres pour {viz_name}. Saut.")
                elif viz_func: fig = viz_func(study)
                
                if fig:
                    output_file = output_viz_dir_for_fold / f"{viz_name}.html"
                    fig.write_html(str(output_file))
                    logger.info(f"{log_prefix_viz} Visualisation Optuna '{viz_name}' sauvegardée : {output_file}")
        except ValueError as ve:
            if "multi-objective optimization, please specify the `target`" in str(ve) or \
               "does not have any completed and feasible trials" in str(ve) or \
               "Cannot evaluate parameter importances" in str(ve) or \
               "cannot be plotted for studies without parameters" in str(ve):
                logger.warning(f"{log_prefix_viz} Saut de la visualisation '{viz_name}' : {ve}")
            else: logger.error(f"{log_prefix_viz} Erreur ValueError '{viz_name}': {ve}", exc_info=True)
        except Exception as e_viz_gen:
            logger.error(f"{log_prefix_viz} Échec génération/sauvegarde '{viz_name}': {e_viz_gen}", exc_info=True)

def _process_single_wfo_task_summary(
    task_summary_path: Path,
    task_results_output_dir: Path,
    task_log_dir: Path,
    app_config: 'AppConfig'
) -> Optional[Dict[str, Any]]:
    try:
        with open(task_summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except Exception as e:
        logger.error(f"Échec du chargement du résumé de la tâche WFO {task_summary_path}: {e}", exc_info=True)
        return None

    strategy_name = summary_data.get("strategy_name", "UnknownStrategy")
    pair_symbol = summary_data.get("pair_symbol", "UnknownPair")
    cli_context_label = summary_data.get("cli_context_label", "UnknownContext")
    wfo_task_run_id = summary_data.get("wfo_task_run_id")
    folds_data = summary_data.get("folds_data", [])

    log_prefix_task_proc = f"[ReportGenTask][{strategy_name}/{pair_symbol}/{cli_context_label}][Task:{wfo_task_run_id}]"
    logger.info(f"{log_prefix_task_proc} Traitement du résumé de la tâche WFO.")

    ensure_dir_exists(task_results_output_dir)

    best_params_for_live: Optional[Dict[str, Any]] = None
    best_oos_metrics_for_live: Optional[Dict[str, Any]] = None
    best_fold_id_for_live: Optional[int] = None
    
    metric_to_optimize_for_selection = app_config.global_config.wfo_settings.metric_to_optimize
    is_maximize_selection = app_config.global_config.wfo_settings.optimization_direction.lower() == "maximize"
    current_best_selection_metric_value = -float('inf') if is_maximize_selection else float('inf')

    valid_folds_with_results = [
        f_data for f_data in folds_data 
        if f_data.get("status", "").startswith("COMPLETED") and \
           f_data.get("representative_oos_metrics") and \
           isinstance(f_data.get("representative_oos_metrics"), dict) and \
           f_data.get("selected_is_params_for_oos")
    ]

    if not valid_folds_with_results:
        logger.warning(f"{log_prefix_task_proc} Aucun fold avec des résultats OOS valides trouvé pour la sélection live_config.")
    else:
        for fold_summary_item in valid_folds_with_results:
            oos_metrics = fold_summary_item["representative_oos_metrics"]
            metric_val_current_fold = oos_metrics.get(metric_to_optimize_for_selection)
            
            if metric_val_current_fold is not None and isinstance(metric_val_current_fold, (int, float)) and np.isfinite(metric_val_current_fold):
                if (is_maximize_selection and metric_val_current_fold > current_best_selection_metric_value) or \
                   (not is_maximize_selection and metric_val_current_fold < current_best_selection_metric_value):
                    current_best_selection_metric_value = metric_val_current_fold
                    best_params_for_live = fold_summary_item["selected_is_params_for_oos"]
                    best_oos_metrics_for_live = oos_metrics
                    best_fold_id_for_live = fold_summary_item.get("fold_id")
        
        if best_params_for_live:
            logger.info(f"{log_prefix_task_proc} Meilleurs paramètres pour live_config sélectionnés du Fold {best_fold_id_for_live} "
                        f"basé sur OOS '{metric_to_optimize_for_selection}': {_format_metric(current_best_selection_metric_value, is_pnl='PnL' in metric_to_optimize_for_selection)}")
            
            strategy_params_config_entry = app_config.strategies_config.strategies.get(strategy_name)
            if strategy_params_config_entry:
                live_config_content = {
                    "strategy_name_base": strategy_name,
                    "strategy_script_reference": strategy_params_config_entry.script_reference,
                    "strategy_class_name": strategy_params_config_entry.class_name,
                    "pair_symbol": pair_symbol,
                    "context_label_optimized_for": cli_context_label,
                    "best_params": best_params_for_live,
                    "selection_criteria_summary": {
                        "metric_optimized_in_wfo": metric_to_optimize_for_selection,
                        "metric_value_at_selection": current_best_selection_metric_value,
                        "source_fold_id_for_params": best_fold_id_for_live,
                        "oos_performance_at_selection": best_oos_metrics_for_live
                    },
                    "source_orchestrator_run_id": summary_data.get("orchestrator_run_id"),
                    "source_wfo_task_run_id": wfo_task_run_id,
                    "generation_timestamp_utc": datetime.now(timezone.utc).isoformat()
                }
                live_config_file = task_results_output_dir / "live_config.json"
                if save_json(live_config_file, live_config_content, indent=2, default_serializer=str):
                    logger.info(f"{log_prefix_task_proc} Fichier live_config.json généré : {live_config_file}")
                else:
                    logger.error(f"{log_prefix_task_proc} Échec de la sauvegarde de live_config.json.")
            else:
                logger.error(f"{log_prefix_task_proc} Configuration de la stratégie '{strategy_name}' non trouvée. Impossible de générer live_config.json.")
        else:
            logger.warning(f"{log_prefix_task_proc} Aucun paramètre n'a pu être sélectionné pour live_config.json basé sur la performance OOS.")

    task_report_path = task_results_output_dir / "performance_report_task.md"
    with open(task_report_path, "w", encoding="utf-8") as f_task_report:
        f_task_report.write(f"# Rapport WFO pour Tâche : {strategy_name} / {pair_symbol} / {cli_context_label}\n\n")
        f_task_report.write(f"* Run Orchestrateur ID: `{summary_data.get('orchestrator_run_id')}`\n")
        f_task_report.write(f"* Tâche WFO Run ID: `{wfo_task_run_id}`\n")
        f_task_report.write(f"* Fichier résumé source: `{task_summary_path.name}` (dans `{task_log_dir.name}`)\n\n")

        if best_params_for_live:
            f_task_report.write(f"## Paramètres Sélectionnés pour Déploiement (Live Config)\n")
            f_task_report.write(f"* Sélectionnés depuis le Fold ID: **{best_fold_id_for_live}**\n")
            f_task_report.write(f"* Métrique OOS optimisée ('{metric_to_optimize_for_selection}'): **{_format_metric(current_best_selection_metric_value, is_pnl='PnL' in metric_to_optimize_for_selection, is_pct='Pct' in metric_to_optimize_for_selection or 'Rate' in metric_to_optimize_for_selection)}**\n")
            f_task_report.write("```json\n" + json.dumps(best_params_for_live, indent=2, default=str) + "\n```\n")
            f_task_report.write(f"**Métriques OOS complètes pour cette sélection :**\n")
            f_task_report.write("```json\n" + json.dumps(best_oos_metrics_for_live, indent=2, default=str) + "\n```\n\n")
            f_task_report.write(f"[Voir live_config.json](./live_config.json)\n\n")
        else:
            f_task_report.write("## Aucun paramètre sélectionné pour déploiement basé sur OOS.\n\n")

        f_task_report.write("## Performance des Folds\n\n")
        f_task_report.write("| Fold ID | Statut | IS Début | IS Fin | OOS Début | OOS Fin | Métrique OOS Optimisée | Params IS (pour OOS) | Visualisations IS |\n")
        f_task_report.write("|:-------:|:------:|:--------:|:------:|:---------:|:-------:|:--------------------:|:--------------------:|:-----------------:|\n")

        for fold_summary_item in folds_data:
            f_id = fold_summary_item.get("fold_id", "N/A")
            f_status = fold_summary_item.get("status", "INCONNU")
            is_s = pd.to_datetime(fold_summary_item.get("is_period_start_utc")).strftime('%Y-%m-%d') if fold_summary_item.get("is_period_start_utc") else "N/A"
            is_e = pd.to_datetime(fold_summary_item.get("is_period_end_utc")).strftime('%Y-%m-%d') if fold_summary_item.get("is_period_end_utc") else "N/A"
            oos_s = pd.to_datetime(fold_summary_item.get("oos_period_start_utc")).strftime('%Y-%m-%d') if fold_summary_item.get("oos_period_start_utc") else "N/A"
            oos_e = pd.to_datetime(fold_summary_item.get("oos_period_end_utc")).strftime('%Y-%m-%d') if fold_summary_item.get("oos_period_end_utc") else "N/A"
            
            oos_metrics_fold = fold_summary_item.get("representative_oos_metrics", {})
            metric_val_fold = oos_metrics_fold.get(metric_to_optimize_for_selection) if isinstance(oos_metrics_fold, dict) else None
            metric_display = _format_metric(metric_val_fold, is_pnl='PnL' in metric_to_optimize_for_selection, is_pct='Pct' in metric_to_optimize_for_selection or 'Rate' in metric_to_optimize_for_selection)
            
            params_fold = fold_summary_item.get("selected_is_params_for_oos", {})
            params_display_parts = [f"{k}: {_format_metric(v,2)}" for k,v in params_fold.items()] if params_fold else ["N/A"]
            params_str = ", ".join(params_display_parts)
            if len(params_str) > 70: params_str = params_str[:67] + "..."

            # Construire le nom de l'objectif principal pour le lien de visualisation
            main_objective_name_for_link = app_config.global_config.optuna_settings.objectives_names[0].replace(' ', '_').replace('/', '_') if app_config.global_config.optuna_settings.objectives_names else 'objectif_0'
            viz_link = f"[Viz Fold {f_id}](./optuna_visualizations/fold_{f_id}/optimization_history_{main_objective_name_for_link}.html)"
            
            f_task_report.write(f"| {f_id} | {f_status} | {is_s} | {is_e} | {oos_s} | {oos_e} | {metric_display} | `{params_str}` | {viz_link} |\n")
        logger.info(f"{log_prefix_task_proc} Rapport Markdown spécifique à la tâche généré : {task_report_path}")

    if OPTUNA_AVAILABLE:
        optuna_viz_base_dir = task_results_output_dir / "optuna_visualizations"
        for fold_summary_item in folds_data:
            f_id_viz = fold_summary_item.get("fold_id")
            if f_id_viz is None: continue
            
            fold_log_dir_for_viz = task_log_dir / f"fold_{f_id_viz}"
            study_name_is_fold = f"{strategy_name}_{pair_symbol}_fold_{f_id_viz}_is_opt" # Nom de l'étude IS
            optuna_db_file_for_viz = fold_log_dir_for_viz / f"optuna_is_study_{strategy_name}_{pair_symbol}_fold_{f_id_viz}.db" # Nom du fichier DB
            
            if optuna_db_file_for_viz.is_file():
                output_viz_dir_for_this_fold = optuna_viz_base_dir / f"fold_{f_id_viz}"
                _generate_optuna_visualizations_for_fold(
                    optuna_db_path=optuna_db_file_for_viz,
                    output_viz_dir_for_fold=output_viz_dir_for_this_fold,
                    study_name=study_name_is_fold, # Passer le nom exact de l'étude
                    app_config=app_config
                )
            else:
                logger.warning(f"{log_prefix_task_proc}[Fold:{f_id_viz}] Fichier DB Optuna IS non trouvé à {optuna_db_file_for_viz}. Visualisations IS sautées.")
    
    return {
        "strategy_name": strategy_name, "pair_symbol": pair_symbol, "cli_context_label": cli_context_label,
        "wfo_task_run_id": wfo_task_run_id,
        "best_overall_oos_metric_value_for_task": current_best_selection_metric_value if best_params_for_live and current_best_selection_metric_value != -float('inf') and current_best_selection_metric_value != float('inf') else None,
        "best_params_for_task": best_params_for_live,
        "source_fold_id_for_best_params": best_fold_id_for_live,
        "task_summary_file_log_path": str(task_summary_path.resolve()),
        "task_report_file_results_path": str(task_report_path.resolve()),
        "live_config_file_results_path": str(task_results_output_dir / "live_config.json") if best_params_for_live else None
    }

def generate_all_reports(
    log_dir: Path,
    results_dir: Path,
    app_config: 'AppConfig'
) -> None:
    main_report_log_prefix = f"[MasterReportGen][OrchRun:{log_dir.name}]"
    logger.info(f"{main_report_log_prefix} Démarrage de la génération de tous les rapports.")
    logger.info(f"{main_report_log_prefix} Lecture des logs depuis : {log_dir}")
    logger.info(f"{main_report_log_prefix} Écriture des résultats dans : {results_dir}")

    ensure_dir_exists(results_dir)

    orchestrator_config_log_path = log_dir / "run_config_orchestrator.json"
    if orchestrator_config_log_path.is_file():
        try:
            shutil.copy2(orchestrator_config_log_path, results_dir / "run_config_orchestrator.json")
            logger.info(f"{main_report_log_prefix} Copie de {orchestrator_config_log_path.name} vers {results_dir}")
        except Exception as e_copy:
            logger.warning(f"{main_report_log_prefix} Échec de la copie de {orchestrator_config_log_path.name}: {e_copy}")

    all_task_summaries_info: List[Dict[str, Any]] = []
    for strat_dir in log_dir.iterdir():
        if not strat_dir.is_dir(): continue
        for pair_dir in strat_dir.iterdir():
            if not pair_dir.is_dir(): continue
            for context_cli_dir in pair_dir.iterdir():
                if not context_cli_dir.is_dir(): continue
                for task_run_id_dir in context_cli_dir.iterdir():
                    if not task_run_id_dir.is_dir(): continue
                    
                    summary_file = task_run_id_dir / "wfo_strategy_pair_summary.json"
                    if summary_file.is_file():
                        logger.info(f"{main_report_log_prefix} Traitement du fichier résumé : {summary_file}")
                        task_results_output_dir = results_dir / strat_dir.name / pair_dir.name / context_cli_dir.name
                        processed_task_info = _process_single_wfo_task_summary(
                            task_summary_path=summary_file,
                            task_results_output_dir=task_results_output_dir,
                            task_log_dir=task_run_id_dir, # Chemin vers .../TASK_RUN_ID/
                            app_config=app_config
                        )
                        if processed_task_info:
                            all_task_summaries_info.append(processed_task_info)
                    # else: # Log trop verbeux si beaucoup de sous-dossiers (ex: fold_N)
                    #     logger.debug(f"{main_report_log_prefix} Fichier wfo_strategy_pair_summary.json non trouvé dans {task_run_id_dir}")
    
    if not all_task_summaries_info:
        logger.warning(f"{main_report_log_prefix} Aucun résumé de tâche WFO trouvé ou traité avec succès dans {log_dir}. Le rapport global sera minimal.")
        global_report_path_empty = results_dir / "global_wfo_report.md"
        with open(global_report_path_empty, "w", encoding="utf-8") as f_empty:
            f_empty.write(f"# Rapport Global WFO - Run Orchestrateur: {log_dir.name}\n\nAucun résultat de tâche WFO traité avec succès.\n")
        # Sauvegarder un fichier de statut même si vide
        run_status_file_empty = results_dir / "report_generation_run_status.json"
        status_content_empty = {
            "orchestrator_run_id_processed": log_dir.name,
            "report_generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "SUCCESS_NO_TASKS_FOUND_OR_PROCESSED",
            "total_wfo_tasks_processed": 0,
            "global_report_file": str(global_report_path_empty.resolve()),
            "processed_tasks_details": []
        }
        save_json(run_status_file_empty, status_content_empty, indent=2, default_serializer=str)
        logger.info(f"{main_report_log_prefix} Rapport global minimal et statut générés.")
        return

    metric_to_sort_global = app_config.global_config.wfo_settings.metric_to_optimize
    is_maximize_global = app_config.global_config.wfo_settings.optimization_direction.lower() == "maximize"
    
    all_task_summaries_info.sort(
        key=lambda x: x.get("best_overall_oos_metric_value_for_task") if isinstance(x.get("best_overall_oos_metric_value_for_task"), (int, float)) else (-float('inf') if is_maximize_global else float('inf')),
        reverse=is_maximize_global
    )

    global_report_path = results_dir / "global_wfo_report.md"
    with open(global_report_path, "w", encoding="utf-8") as f_global:
        f_global.write(f"# Rapport Global d'Optimisation Walk-Forward\n\n")
        f_global.write(f"**Run Orchestrateur ID:** `{log_dir.name}`\n")
        f_global.write(f"**Date de Génération:** {datetime.now(timezone.utc).isoformat()}\n")
        f_global.write(f"**Nombre Total de Tâches WFO Traitées avec Succès (résumé trouvé):** {len(all_task_summaries_info)}\n")
        f_global.write(f"**Métrique d'Optimisation Principale (OOS) pour sélection live_config:** `{metric_to_sort_global}` (Direction: {'Maximiser' if is_maximize_global else 'Minimiser'})\n\n")
        f_global.write("## Résumé des Tâches WFO Traitées (Stratégie/Paire/Contexte)\n\n")
        f_global.write("| Stratégie | Paire | Contexte CLI | Meilleure Métrique OOS | Params Sélectionnés (du Fold ID) | Fichier Live Config | Rapport de Tâche |\n")
        f_global.write("|:----------|:------|:-------------|:-----------------------:|:---------------------------------|:--------------------|:----------------:|\n")

        for task_info in all_task_summaries_info:
            strat = task_info.get("strategy_name", "N/A")
            pair = task_info.get("pair_symbol", "N/A")
            context = task_info.get("cli_context_label", "N/A")
            metric_val = task_info.get("best_overall_oos_metric_value_for_task")
            metric_display = _format_metric(metric_val, is_pnl='PnL' in metric_to_sort_global, is_pct='Pct' in metric_to_sort_global or 'Rate' in metric_to_sort_global)
            params = task_info.get("best_params_for_task")
            source_fold = task_info.get("source_fold_id_for_best_params", "N/A")
            params_str = "N/A"
            if params and isinstance(params, dict):
                params_display_parts = [f"{k}: {_format_metric(v,2)}" for k,v in params.items()]
                params_str = f"`{', '.join(params_display_parts)}` (Fold: {source_fold})"
            if len(params_str) > 60: params_str = params_str[:57] + "...`"
            
            live_cfg_path_str = task_info.get("live_config_file_results_path")
            live_cfg_link = f"[{Path(live_cfg_path_str).name}](./{Path(live_cfg_path_str).relative_to(results_dir).as_posix()})" if live_cfg_path_str and Path(live_cfg_path_str).exists() else "Non généré"
            
            task_report_path_str = task_info.get("task_report_file_results_path")
            task_report_link = f"[{Path(task_report_path_str).name}](./{Path(task_report_path_str).relative_to(results_dir).as_posix()})" if task_report_path_str and Path(task_report_path_str).exists() else "N/A"

            f_global.write(f"| {strat} | {pair} | {context} | {metric_display} | {params_str} | {live_cfg_link} | {task_report_link} |\n")
            
    logger.info(f"{main_report_log_prefix} Rapport Markdown global généré : {global_report_path}")

    run_status_file = results_dir / "report_generation_run_status.json"
    status_content = {
        "orchestrator_run_id_processed": log_dir.name,
        "report_generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS" if all_task_summaries_info else "SUCCESS_NO_TASKS_FOUND_OR_PROCESSED",
        "total_wfo_tasks_processed": len(all_task_summaries_info),
        "global_report_file": str(global_report_path.resolve()),
        "processed_tasks_details": all_task_summaries_info # Inclure les infos agrégées pour chaque tâche
    }
    if save_json(run_status_file, status_content, indent=2, default_serializer=str):
        logger.info(f"{main_report_log_prefix} Statut de la génération des rapports sauvegardé : {run_status_file}")
    else:
        logger.error(f"{main_report_log_prefix} Échec de la sauvegarde du statut de la génération des rapports.")

    logger.info(f"{main_report_log_prefix} Génération de tous les rapports terminée.")

