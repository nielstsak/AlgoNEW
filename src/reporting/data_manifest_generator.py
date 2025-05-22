import pathlib
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple, Union
from datetime import datetime, timezone
import shutil # Pour copier run_config.json

# Optionnel, pour les visualisations Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
    logging.getLogger(__name__).info("Optuna library found.")
    try:
        import plotly # Optuna visualization often uses plotly
        PLOTLY_AVAILABLE = True
        logging.getLogger(__name__).info("Plotly library found.")
    except ImportError:
        plotly = None # type: ignore
        PLOTLY_AVAILABLE = False
        logging.getLogger(__name__).warning(
            "Plotly library not found. Some Optuna visualizations might not be available or may raise an error."
        )
except ImportError:
    optuna = None # type: ignore
    OPTUNA_AVAILABLE = False
    PLOTLY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Optuna library not found. Optuna visualizations will not be generated."
    )

logger = logging.getLogger(__name__)

def _format_metric(value: Any, precision: int = 4, is_pnl: bool = False) -> str:
    """Helper function to format metrics for display."""
    if isinstance(value, (float, np.floating, np.float64)):
        if np.isnan(value):
            return "NaN"
        elif np.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        else:
            return f"{value:.{2 if is_pnl else precision}f}" # PnL usually has 2 decimal places
    elif isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, str):
        return value
    elif value is None:
        return "N/A"
    else:
        return str(value)

def _generate_optuna_visualizations(
    optuna_db_path: pathlib.Path,
    output_viz_dir: pathlib.Path,
    study_name_prefix: str # e.g., strategy_pair_context
) -> None:
    """
    Generates and saves Optuna visualization plots for a given study.
    """
    logger.info(f"[{study_name_prefix}] Generating Optuna visualizations from DB: {optuna_db_path}")

    if not OPTUNA_AVAILABLE:
        logger.warning(f"[{study_name_prefix}] Optuna is not available. Skipping visualization generation.")
        return
    if not PLOTLY_AVAILABLE:
        logger.warning(f"[{study_name_prefix}] Plotly is not available. Some Optuna visualizations may fail or be unavailable.")
        # We can still attempt some visualizations that might not strictly require Plotly or have fallbacks.

    if not optuna_db_path.exists():
        logger.warning(f"[{study_name_prefix}] Optuna DB file not found at {optuna_db_path}. Cannot generate visualizations.")
        return

    output_viz_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{optuna_db_path.resolve()}"
    
    # Try to load all studies and find the one matching the prefix for the specific fold.
    # Optuna study names are often globally unique in a DB or have a prefix.
    # The study_name used during study creation in StudyManager was:
    # f"{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}_is_opt"
    # where self.study_output_dir.name is typically "fold_X".
    # So, study_name_prefix here should be strategy_pair_context, and we append _fold_X.
    # The actual study name in the DB would be like: "StrategyName_PairSymbol_fold_X_is_opt"
    
    # We need the exact study name as stored in the DB.
    # The study_name_prefix passed here is "strategy_pair_context_fold_X"
    # So the actual study name should be study_name_prefix + "_is_opt"
    # However, the prompt says study_name_for_viz = f"{strategy_name_from_file}_{pair_symbol_from_file}_{context_label_from_file}_fold_{fold_id}"
    # Let's assume study_name_for_viz IS the actual study name.

    actual_study_name = study_name_prefix # Assuming study_name_prefix is the full study name for this fold.

    try:
        study = optuna.load_study(study_name=actual_study_name, storage=storage_url)
        logger.info(f"[{actual_study_name}] Successfully loaded study for visualization.")
    except Exception as e:
        logger.error(f"[{actual_study_name}] Failed to load Optuna study from {storage_url}: {e}", exc_info=True)
        # Attempt to list studies if direct load fails, to help debugging
        try:
            all_studies = optuna.study.get_all_study_summaries(storage=storage_url)
            logger.info(f"[{actual_study_name}] Available studies in DB {optuna_db_path}: {[s.study_name for s in all_studies]}")
        except Exception as e_list:
            logger.error(f"[{actual_study_name}] Could not even list studies from DB {optuna_db_path}: {e_list}")
        return

    viz_functions = {
        "optimization_history": optuna.visualization.plot_optimization_history,
        "param_importances": optuna.visualization.plot_param_importances,
        "slice_plot": optuna.visualization.plot_slice,
    }
    if len(study.directions) > 1:
        viz_functions["pareto_front"] = optuna.visualization.plot_pareto_front

    for viz_name, viz_func in viz_functions.items():
        if not PLOTLY_AVAILABLE and viz_name in ["optimization_history", "pareto_front", "slice_plot"]: # These often rely heavily on Plotly
            logger.warning(f"[{actual_study_name}] Skipping {viz_name} plot as Plotly is not available.")
            continue
        try:
            if viz_name == "slice_plot":
                # plot_slice might need specific params, let's try with default or skip if too complex without specific params
                # For simplicity, we might skip slice plot or try with the first few important params if available.
                # Getting important params:
                try:
                    importances = optuna.importance.get_param_importances(study)
                    important_params = [p_name for p_name, _ in sorted(importances.items(), key=lambda item: item[1], reverse=True)[:2]] # Top 2
                    if important_params:
                        fig = viz_func(study, params=important_params)
                    else: # No important params or single param study
                        logger.info(f"[{actual_study_name}] Skipping slice plot as important parameters could not be determined or are too few.")
                        continue
                except Exception as e_slice_params:
                    logger.warning(f"[{actual_study_name}] Could not determine params for slice plot, skipping. Error: {e_slice_params}")
                    continue
            else:
                fig = viz_func(study)
            
            output_file = output_viz_dir / f"{viz_name}.html"
            fig.write_html(str(output_file))
            logger.info(f"[{actual_study_name}] Saved {viz_name} plot to {output_file}")
        except Exception as e_viz:
            logger.error(f"[{actual_study_name}] Failed to generate/save {viz_name} plot: {e_viz}", exc_info=True)


def _generate_markdown_report(
    wfo_summary_data: Dict[str, Any],
    base_fold_log_path: pathlib.Path, # This is context_label_dir
    report_file: pathlib.Path,
    live_config_params_selected: Optional[Dict[str, Any]],
    selection_fold_id: Optional[int],
    selection_oos_metric_value: Optional[float],
    selection_metric_name: str
    ):
    try:
        strategy_name = wfo_summary_data.get("strategy_name", "N/A")
        pair_symbol = wfo_summary_data.get("pair_symbol", "N/A")
        context_label = wfo_summary_data.get("context_label", "N/A")
        fold_results_from_wfo = wfo_summary_data.get("folds_data", [])

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# WFO Performance Report: {strategy_name} - {pair_symbol} ({context_label})\n\n")
            f.write("## WFO Configuration\n")
            f.write(f"- Strategy: {strategy_name}\n")
            f.write(f"- Pair: {pair_symbol}\n")
            f.write(f"- Context: {context_label}\n")
            f.write(f"- WFO Run Timestamp: {wfo_summary_data.get('wfo_run_timestamp', 'N/A')}\n\n")

            f.write("## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)\n")
            f.write("*Detailed OOS performance is shown per fold below.*\n\n")

            f.write("## Parameters Selected for Live Configuration\n")
            if live_config_params_selected:
                f.write(f"*Selected from Fold {selection_fold_id} based on best OOS '{selection_metric_name}': {_format_metric(selection_oos_metric_value, is_pnl='PnL' in selection_metric_name)}*\n")
                f.write("```json\n")
                f.write(json.dumps(live_config_params_selected, indent=2, default=str))
                f.write("\n```\n\n")
            else:
                f.write("*No parameters were selected for live configuration (e.g., no successful OOS trials found).*\n\n")

            f.write("## Fold-by-Fold OOS Results\n\n")
            if not fold_results_from_wfo:
                f.write("No fold results available in WFO summary.\n\n")
            else:
                f.write("| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary (Best '{metric}') | Optuna Visuals |\n".format(metric=selection_metric_name))
                f.write("| :--- | :-------- | :--------------------------- | :--------------------------------------- | :------------- |\n")
                for fold_wfo_data in fold_results_from_wfo:
                    fold_id = fold_wfo_data.get('fold_index', 'N/A')
                    is_status = fold_wfo_data.get('status', 'UNKNOWN')
                    selected_is_params_for_fold = fold_wfo_data.get('selected_params_for_fold', {})
                    
                    params_str_parts = []
                    if selected_is_params_for_fold:
                        for k, v in selected_is_params_for_fold.items():
                            param_name_short = k.split('_')[-1] if '_' in k and "frequence" in k else (k.split('_')[0][:4] if '_' in k else k[:4])
                            params_str_parts.append(f"{param_name_short}:{_format_metric(v,2 if isinstance(v,float) else 0)}")
                    params_display = ", ".join(params_str_parts) if params_str_parts else "N/A"

                    oos_summary_file_for_fold = base_fold_log_path / f"fold_{fold_id}" / "oos_validation_summary_TOP_N_TRIALS.json"
                    oos_trials_summary_str = "OOS summary file not found or N/A."
                    if oos_summary_file_for_fold.exists():
                        try:
                            with open(oos_summary_file_for_fold, 'r', encoding='utf-8') as oos_f:
                                oos_trials_data = json.load(oos_f)
                            if oos_trials_data:
                                best_oos_trial_for_fold = None
                                current_best_metric_val = -float('inf') if "maximize" in selection_metric_name.lower() or "pnl" in selection_metric_name.lower() else float('inf') # Crude direction check
                                
                                for trial_data in oos_trials_data:
                                    metric_val = trial_data.get("oos_metrics", {}).get(selection_metric_name)
                                    if metric_val is not None and isinstance(metric_val, (int, float)) and np.isfinite(metric_val):
                                        if ("maximize" in selection_metric_name.lower() or "pnl" in selection_metric_name.lower()) and metric_val > current_best_metric_val:
                                            current_best_metric_val = metric_val
                                            best_oos_trial_for_fold = trial_data
                                        elif ("minimize" in selection_metric_name.lower()) and metric_val < current_best_metric_val: # Assuming minimize for others like Max Drawdown
                                            current_best_metric_val = metric_val
                                            best_oos_trial_for_fold = trial_data
                                
                                if best_oos_trial_for_fold:
                                    best_is_params_str = ", ".join([f"{k[:4]}:{_format_metric(v,2 if isinstance(v,float) else 0)}" for k,v in best_oos_trial_for_fold.get("is_trial_params", {}).items()])
                                    oos_trials_summary_str = (f"{len(oos_trials_data)} OOS trials. Best '{selection_metric_name}': "
                                                              f"{_format_metric(current_best_metric_val, is_pnl='PnL' in selection_metric_name)}. "
                                                              f"IS Params: {best_is_params_str if best_is_params_str else 'N/A'}")
                                else:
                                    oos_trials_summary_str = f"{len(oos_trials_data)} OOS trials. No valid trial found for '{selection_metric_name}'."
                            else:
                                oos_trials_summary_str = "OOS trials run, but no data in summary file."
                        except Exception as e_oos_load:
                            oos_trials_summary_str = f"Error loading OOS summary: {str(e_oos_load)[:50]}"
                    elif is_status != "COMPLETED":
                         oos_trials_summary_str = f"(IS Status: {is_status})"
                    
                    viz_link = f"[Viz](./optuna_visualizations/fold_{fold_id}/optimization_history.html)" if (base_fold_log_path.parent.parent.parent / "results" / strategy_name / pair_symbol / context_label / "optuna_visualizations" / f"fold_{fold_id}" / "optimization_history.html").exists() else "N/A"

                    f.write(f"| {fold_id:<4} | {is_status:<9} | {params_display:<60} | {oos_trials_summary_str} | {viz_link} |\n")
                f.write("\n")
        logger.info(f"Generated Markdown report: {report_file}")
    except Exception as e:
        logger.error(f"Failed to generate Markdown report for {strategy_name}/{pair_symbol}: {e}", exc_info=True)
        raise # Re-raise to be caught by generate_all_reports

def _generate_live_config(
    wfo_summary_data: Dict[str, Any],
    base_fold_log_path: pathlib.Path, # This is context_label_dir
    live_config_output_file: pathlib.Path,
    selection_metric: str = "Total Net PnL USDC"
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[float], str]: # Added metric name to return

    strategy_name = wfo_summary_data.get("strategy_name")
    pair_symbol = wfo_summary_data.get("pair_symbol")
    context_label = wfo_summary_data.get("context_label")
    wfo_run_timestamp = wfo_summary_data.get("wfo_run_timestamp", "unknown_run")
    
    if not strategy_name or not pair_symbol or not context_label:
        logger.error("Strategy name, pair symbol, or context_label missing in WFO summary. Cannot generate live config.")
        return None, None, None, selection_metric

    fold_results_from_wfo = wfo_summary_data.get("folds_data", [])
    last_successful_fold_id: Optional[int] = None
    
    # Find the last fold that completed IS optimization successfully
    for fold_data in reversed(fold_results_from_wfo): # Iterate from last fold to first
        if fold_data.get("status") == "COMPLETED" and fold_data.get("selected_params_for_fold") is not None:
            last_successful_fold_id = fold_data.get("fold_index")
            break # Found the last one
            
    if last_successful_fold_id is None:
        logger.warning(f"No successful IS optimization fold found for {strategy_name}/{pair_symbol}/{context_label}. Cannot generate live config from OOS results.")
        return None, None, None, selection_metric

    logger.info(f"Attempting to generate live_config from OOS results of Fold {last_successful_fold_id} for {strategy_name}/{pair_symbol}/{context_label}.")
    
    oos_summary_file_path = base_fold_log_path / f"fold_{last_successful_fold_id}" / "oos_validation_summary_TOP_N_TRIALS.json"

    if not oos_summary_file_path.exists():
        logger.warning(f"OOS validation summary file not found for Fold {last_successful_fold_id}: {oos_summary_file_path}. Cannot generate live config from OOS.")
        return None, None, None, selection_metric

    try:
        with open(oos_summary_file_path, 'r', encoding='utf-8') as f:
            oos_trials_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OOS validation summary {oos_summary_file_path}: {e}", exc_info=True)
        return None, None, None, selection_metric

    if not oos_trials_data:
        logger.warning(f"OOS validation summary for Fold {last_successful_fold_id} is empty. Cannot select parameters.")
        return None, None, None, selection_metric

    best_oos_trial_for_live = None
    # Determine direction for selection_metric
    # Crude check, ideally this comes from OptunaSettings or is passed in
    is_maximize = "maximize" in selection_metric.lower() or "pnl" in selection_metric.lower() or "sharpe" in selection_metric.lower() or "profit" in selection_metric.lower()
    
    best_oos_metric_value = -float('inf') if is_maximize else float('inf')

    for oos_trial_run in oos_trials_data:
        current_metric_value = oos_trial_run.get("oos_metrics", {}).get(selection_metric)
        if current_metric_value is not None and isinstance(current_metric_value, (int, float)) and np.isfinite(current_metric_value):
            if (is_maximize and current_metric_value > best_oos_metric_value) or \
               (not is_maximize and current_metric_value < best_oos_metric_value):
                best_oos_metric_value = current_metric_value
                best_oos_trial_for_live = oos_trial_run
    
    if best_oos_trial_for_live and best_oos_trial_for_live.get("is_trial_params"):
        selected_params = best_oos_trial_for_live["is_trial_params"]
        
        live_strategy_id = f"{strategy_name}_{pair_symbol}_{context_label}_{wfo_run_timestamp}"

        live_config_content = {
            "strategy_id": live_strategy_id,
            "strategy_name_base": strategy_name,
            "pair_symbol": pair_symbol,
            "timeframe_context": context_label, 
            "parameters": selected_params,
            "source_wfo_run": wfo_run_timestamp,
            "source_fold_id": last_successful_fold_id,
            "source_is_trial_number": best_oos_trial_for_live.get("is_trial_number"),
            "selection_oos_metric_name": selection_metric, # Save the name of the metric used
            "selection_oos_metric_value": best_oos_metric_value
        }
        live_config_output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(live_config_output_file, "w", encoding="utf-8") as f:
            json.dump(live_config_content, f, indent=4, default=str)
        logger.info(f"Generated Live JSON config using best OOS trial from Fold {last_successful_fold_id}: {live_config_output_file}")
        return selected_params, last_successful_fold_id, best_oos_metric_value, selection_metric
    else:
        logger.warning(f"Could not find a suitable OOS trial to select parameters for live config from Fold {last_successful_fold_id} for {strategy_name}/{pair_symbol}/{context_label} based on metric '{selection_metric}'.")
        return None, None, None, selection_metric


def generate_all_reports(log_dir: pathlib.Path, results_dir: pathlib.Path):
    """
    Generates all reports for a specific WFO run.
    log_dir: Path to the root of a specific WFO run (e.g., logs/backtest_optimization/RUN_ID).
    results_dir: Path to where reports for this specific WFO run should be saved (e.g., results/RUN_ID).
    """
    logger.info(f"Starting report generation from WFO log directory: {log_dir}")
    logger.info(f"Saving reports to results directory: {results_dir}")

    results_dir.mkdir(parents=True, exist_ok=True)
    
    current_run_status: Dict[str, Any] = {
        "run_id": log_dir.name,
        "status": "SUCCESS", # Assume success initially
        "report_generation_start_utc": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "processed_combinations": []
    }

    # Copy run_config.json
    run_config_source_path = log_dir / "run_config.json"
    run_config_dest_path = results_dir / "run_config.json"
    if run_config_source_path.exists():
        try:
            shutil.copy2(run_config_source_path, run_config_dest_path)
            logger.info(f"Copied {run_config_source_path.name} to {run_config_dest_path}")
        except Exception as e_copy:
            err_msg = f"Failed to copy {run_config_source_path.name}: {e_copy}"
            logger.error(err_msg)
            current_run_status["errors"].append(err_msg)
    else:
        logger.warning(f"{run_config_source_path.name} not found in log directory. Cannot copy.")
        current_run_status["errors"].append(f"{run_config_source_path.name} not found in log directory.")

    # Load WFO settings from run_config.json if available, to get n_splits
    n_splits_configured = None
    if run_config_dest_path.exists(): # Check destination, as source might have failed to copy
        try:
            with open(run_config_dest_path, 'r') as f_rc:
                run_config_data = json.load(f_rc)
                n_splits_configured = run_config_data.get("global_config", {}).get("wfo_settings", {}).get("n_splits")
        except Exception as e_read_rc:
            logger.warning(f"Could not read n_splits from {run_config_dest_path}: {e_read_rc}")


    global_summary_entries: List[Dict[str, Any]] = []
    
    if not log_dir.is_dir():
        err_msg = f"Base log directory for run not found: {log_dir}"
        logger.error(err_msg)
        current_run_status["errors"].append(err_msg)
        current_run_status["status"] = "FAILURE"
    else:
        for strategy_dir in log_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
            
            for pair_symbol_dir in strategy_dir.iterdir():
                if not pair_symbol_dir.is_dir():
                    continue
                
                for context_label_dir in pair_symbol_dir.iterdir():
                    if not context_label_dir.is_dir():
                        continue

                    combination_id = f"{strategy_dir.name}/{pair_symbol_dir.name}/{context_label_dir.name}"
                    logger.info(f"Processing combination: {combination_id}")
                    
                    wfo_summary_file_path = context_label_dir / "wfo_strategy_pair_summary.json"
                    if not wfo_summary_file_path.is_file():
                        err_msg = f"WFO summary file not found for {combination_id}: {wfo_summary_file_path}"
                        logger.error(err_msg)
                        current_run_status["errors"].append(err_msg)
                        continue
                    
                    try:
                        with open(wfo_summary_file_path, 'r', encoding='utf-8') as f_wfo:
                            wfo_summary_data = json.load(f_wfo)

                        strategy_name_from_file = wfo_summary_data.get("strategy_name", strategy_dir.name)
                        pair_symbol_from_file = wfo_summary_data.get("pair_symbol", pair_symbol_dir.name)
                        context_label_from_file = wfo_summary_data.get("context_label", context_label_dir.name)

                        # Validate directory names against file content
                        if strategy_name_from_file != strategy_dir.name or \
                           pair_symbol_from_file != pair_symbol_dir.name or \
                           context_label_from_file != context_label_dir.name:
                            logger.warning(f"Mismatch between directory names and wfo_summary.json content for {combination_id}. Using content from JSON.")
                        
                        current_report_output_dir = results_dir / strategy_name_from_file / pair_symbol_from_file / context_label_from_file
                        current_report_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        live_config_output_path = current_report_output_dir / "live_config.json"
                        
                        # Determine selection metric (example: from Optuna settings if available in wfo_summary or run_config)
                        # For now, defaulting to "Total Net PnL USDC"
                        default_selection_metric = "Total Net PnL USDC"
                        # Attempt to get it from wfo_summary_data if it stores optuna settings used
                        optuna_settings_in_summary = wfo_summary_data.get("optuna_settings_used") # Hypothetical key
                        if isinstance(optuna_settings_in_summary, dict) and optuna_settings_in_summary.get("objectives_names"):
                             # Prioritize based on pareto_selection_strategy if possible, else first objective
                             sel_strat = optuna_settings_in_summary.get("pareto_selection_strategy")
                             obj_names = optuna_settings_in_summary.get("objectives_names")
                             if sel_strat == "PNL_MAX" and "Total Net PnL USDC" in obj_names:
                                 default_selection_metric = "Total Net PnL USDC"
                             elif obj_names: # Default to first objective
                                 default_selection_metric = obj_names[0]
                        
                        selected_live_params, sel_fold_id, sel_oos_metric_value, sel_metric_name_used = _generate_live_config(
                            wfo_summary_data=wfo_summary_data,
                            base_fold_log_path=context_label_dir,
                            live_config_output_file=live_config_output_path,
                            selection_metric=default_selection_metric # Pass the determined or default metric
                        )
                        
                        markdown_report_path = current_report_output_dir / "performance_report.md"
                        _generate_markdown_report(
                            wfo_summary_data=wfo_summary_data,
                            base_fold_log_path=context_label_dir,
                            report_file=markdown_report_path,
                            live_config_params_selected=selected_live_params,
                            selection_fold_id=sel_fold_id,
                            selection_oos_metric_value=sel_oos_metric_value,
                            selection_metric_name=sel_metric_name_used # Use the metric name returned by _generate_live_config
                        )

                        optuna_visualizations_dir = current_report_output_dir / "optuna_visualizations"
                        # optuna_visualizations_dir.mkdir(parents=True, exist_ok=True) # _generate_optuna_visualizations creates it

                        for fold_data in wfo_summary_data.get("folds_data", []):
                            fold_idx = fold_data.get("fold_index")
                            if fold_idx is None: continue

                            optuna_db_path = context_label_dir / f"fold_{fold_idx}" / "optuna_study.db"
                            # Construct study name as used in StudyManager
                            study_name_for_viz = f"{strategy_name_from_file}_{pair_symbol_from_file}_{context_label_from_file}_fold_{fold_idx}_is_opt"
                            
                            if optuna_db_path.exists():
                                _generate_optuna_visualizations(
                                    optuna_db_path=optuna_db_path,
                                    output_viz_dir=optuna_visualizations_dir / f"fold_{fold_idx}",
                                    study_name_prefix=study_name_for_viz # Pass the full study name
                                )
                            else:
                                logger.warning(f"Optuna DB not found for {combination_id}, Fold {fold_idx} at {optuna_db_path}. Skipping visualizations for this fold.")
                        
                        # Prepare entry for global summary CSV
                        folds_data = wfo_summary_data.get("folds_data", [])
                        num_folds_attempted = len(folds_data)
                        num_is_completed = sum(1 for fd in folds_data if fd.get("status") == "COMPLETED")
                        num_oos_validated = sum(1 for fd in folds_data if (context_label_dir / f"fold_{fd.get('fold_index')}" / "oos_validation_summary_TOP_N_TRIALS.json").exists())


                        global_entry = {
                            "RunTimestamp": log_dir.name,
                            "StrategyName": strategy_name_from_file,
                            "PairSymbol": pair_symbol_from_file,
                            "ContextLabel": context_label_from_file,
                            "TotalFoldsConfigured": n_splits_configured if n_splits_configured is not None else "N/A",
                            "FoldsAttempted": num_folds_attempted,
                            "FoldsISCompleted": num_is_completed,
                            "FoldsOOSValidated": num_oos_validated,
                            "LiveConfigGenerated": selected_live_params is not None,
                            "LiveConfigSelectionFoldID": sel_fold_id,
                            "LiveConfigSelectionMetricName": sel_metric_name_used,
                            "LiveConfigSelectionMetricOOSValue": _format_metric(sel_oos_metric_value, is_pnl='PnL' in sel_metric_name_used),
                            "LiveConfigSelectedParams": json.dumps(selected_live_params, default=str) if selected_live_params else "N/A"
                        }
                        global_summary_entries.append(global_entry)
                        current_run_status["processed_combinations"].append(combination_id)

                    except json.JSONDecodeError as e_json:
                        err_msg = f"Failed to decode JSON from {wfo_summary_file_path} for {combination_id}: {e_json}"
                        logger.error(err_msg, exc_info=True)
                        current_run_status["errors"].append(err_msg)
                    except Exception as e_combo:
                        err_msg = f"Error processing combination {combination_id}: {e_combo}"
                        logger.error(err_msg, exc_info=True)
                        current_run_status["errors"].append(err_msg)
    
    # Generate global CSV summary
    if global_summary_entries:
        try:
            global_df = pd.DataFrame(global_summary_entries)
            # Define column order for better readability
            cols_order = [
                "RunTimestamp", "StrategyName", "PairSymbol", "ContextLabel",
                "TotalFoldsConfigured", "FoldsAttempted", "FoldsISCompleted", "FoldsOOSValidated",
                "LiveConfigGenerated", "LiveConfigSelectionFoldID",
                "LiveConfigSelectionMetricName", "LiveConfigSelectionMetricOOSValue",
                "LiveConfigSelectedParams"
            ]
            # Ensure all expected columns are present, add others at the end
            final_cols = [c for c in cols_order if c in global_df.columns] + \
                         [c for c in global_df.columns if c not in cols_order]
            global_df = global_df[final_cols]

            global_csv_path = results_dir.parent / f"global_wfo_summary_for_run_{log_dir.name}.csv"
            global_df.to_csv(global_csv_path, index=False, float_format='%.4f')
            logger.info(f"Generated global WFO summary CSV: {global_csv_path}")
        except Exception as e_csv:
            err_msg = f"Failed to generate global WFO summary CSV: {e_csv}"
            logger.error(err_msg, exc_info=True)
            current_run_status["errors"].append(err_msg)
    else:
        logger.warning(f"No WFO results processed to generate a global summary for run {log_dir.name}.")

    # Finalize and save run_status.json
    if current_run_status["errors"]:
        current_run_status["status"] = "PARTIAL_FAILURE" if current_run_status["processed_combinations"] else "FAILURE"
    
    current_run_status["report_generation_end_utc"] = datetime.now(timezone.utc).isoformat()
    run_status_file_path = results_dir / "run_status.json"
    try:
        with open(run_status_file_path, 'w', encoding='utf-8') as f_status:
            json.dump(current_run_status, f_status, indent=4, default=str)
        logger.info(f"Run status file saved to: {run_status_file_path}")
    except Exception as e_status:
        logger.error(f"Failed to save run status file: {e_status}", exc_info=True)
        # Log this error to a fallback if the primary logging is also part of the issue
        # For now, assume logger is working.

    logger.info(f"Report generation process finished for run: {log_dir.name}. Status: {current_run_status['status']}")