# src/utils/metric_aggregator.py
"""
Agrégateur de métriques multi-niveaux pour consolider et analyser les résultats
de backtesting à travers les folds, les stratégies, et globalement.
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
# Pour TOPSIS, si implémenté en détail. Pour l'instant, une approche plus simple.
# from skcriteria import Data, mkdm # Example, if using scikit-criteria

# Pour le templating du résumé exécutif
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Jinja2 n'est pas installé. La génération de résumés exécutifs formatés sera limitée."
    )

# Pour l'export Excel avec styling (openpyxl est une dépendance de pandas pour xlsx)
try:
    import openpyxl # Utilisé par pandas pour to_excel avec .xlsx
    # from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    # from openpyxl.utils.dataframe import dataframe_to_rows
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "openpyxl n'est pas installé. L'export Excel stylisé sera limité ou non disponible."
    )


logger = logging.getLogger(__name__)

# --- Dataclasses pour structurer les résultats ---
@dataclass
class AggregatedMetrics:
    """Stocke les métriques agrégées (ex: pour un ensemble de folds)."""
    identifier: str # Ex: "StrategyA_BTCUSDT_Overall"
    count: int
    metrics_mean: Dict[str, Optional[float]] = field(default_factory=dict)
    metrics_median: Dict[str, Optional[float]] = field(default_factory=dict)
    metrics_std: Dict[str, Optional[float]] = field(default_factory=dict)
    metrics_mad: Dict[str, Optional[float]] = field(default_factory=dict) # Median Absolute Deviation
    # Intervalles de confiance bootstrap pour les moyennes
    metrics_mean_ci_lower: Dict[str, Optional[float]] = field(default_factory=dict)
    metrics_mean_ci_upper: Dict[str, Optional[float]] = field(default_factory=dict)
    # D'autres statistiques agrégées peuvent être ajoutées
    raw_fold_metrics_df: Optional[pd.DataFrame] = None # Optionnel: stocker le DF des métriques de fold brutes

@dataclass
class AnomalyDetails:
    """Détails sur une anomalie de métrique détectée."""
    metric_name: str
    value: float
    z_score: Optional[float] = None
    is_outlier_isolation_forest: Optional[bool] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict) # Contexte: fold_id, strategy_name, etc.

@dataclass
class ComparisonReport:
    """Rapport de comparaison entre stratégies."""
    ranked_strategies: List[Dict[str, Any]] = field(default_factory=list) # Liste de stratégies avec leur score/rang
    criteria_used: List[str] = field(default_factory=list)
    weights_used: Optional[Dict[str, float]] = None
    method_used: str = "SimpleScoring" # Ex: "TOPSIS", "SimpleWeightedSum"
    statistical_tests_summary: Optional[Dict[str, Any]] = None # Ex: résultats Kruskal-Wallis

@dataclass
class ExecutiveSummary:
    """Résumé exécutif des résultats globaux."""
    generation_timestamp_utc: str
    overall_best_strategy: Optional[Dict[str, Any]] = None # {name: str, primary_metric_value: float, ...}
    top_n_strategies: List[Dict[str, Any]] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    detected_anomalies_summary: List[str] = field(default_factory=list)
    global_stats_summary: Dict[str, Any] = field(default_factory=dict)
    report_markdown: Optional[str] = None # Le résumé formaté en Markdown


class MetricAggregator:
    """
    Agrège, compare, et analyse les métriques de performance de backtesting
    à différents niveaux hiérarchiques.
    """
    def __init__(self,
                 bootstrap_n_resamples: int = 1000,
                 bootstrap_ci_level: float = 0.95,
                 anomaly_z_threshold: float = 3.0):
        """
        Initialise l'agrégateur de métriques.

        Args:
            bootstrap_n_resamples (int): Nombre de rééchantillonnages pour le bootstrap.
            bootstrap_ci_level (float): Niveau de confiance pour les intervalles bootstrap.
            anomaly_z_threshold (float): Seuil Z-score pour la détection d'anomalies.
        """
        self.log_prefix = "[MetricAggregator]"
        self.bootstrap_n_resamples = bootstrap_n_resamples
        self.bootstrap_ci_level = bootstrap_ci_level
        self.anomaly_z_threshold = anomaly_z_threshold
        logger.info(f"{self.log_prefix} Initialisé. Bootstrap resamples: {bootstrap_n_resamples}, CI: {bootstrap_ci_level*100}%, Z-thresh: {anomaly_z_threshold}")

    def _bootstrap_ci(self, data: Union[List[float], np.ndarray], metric_func: Callable[[Any], float]) -> Tuple[Optional[float], Optional[float]]:
        """Calcule l'intervalle de confiance bootstrap pour une métrique."""
        if len(data) < 20: # Besoin d'assez de données pour un bootstrap significatif
            return None, None
        
        data_array = np.array(data)
        bootstrap_samples = np.random.choice(data_array,
                                             size=(self.bootstrap_n_resamples, len(data_array)),
                                             replace=True)
        
        bootstrap_metrics = np.apply_along_axis(metric_func, axis=1, arr=bootstrap_samples)
        bootstrap_metrics = bootstrap_metrics[~np.isnan(bootstrap_metrics)] # Retirer les NaNs
        
        if len(bootstrap_metrics) < self.bootstrap_n_resamples * 0.5: # Si trop de NaNs
            return None, None

        lower_percentile = (1.0 - self.bootstrap_ci_level) / 2.0 * 100
        upper_percentile = (100.0 - lower_percentile)
        
        ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile)
        return float(ci_lower), float(ci_upper)

    def aggregate_fold_metrics(
        self,
        fold_results_list: List[Dict[str, Any]], # Liste de dictionnaires, chaque dict = métriques d'un fold
        identifier: str, # Pour nommer cet ensemble agrégé
        weight_by_column: Optional[str] = 'oos_duration_days', # Colonne pour pondérer les moyennes
        metrics_to_aggregate: Optional[List[str]] = None # Si None, tente d'agréger toutes les colonnes numériques
    ) -> Optional[AggregatedMetrics]:
        """
        Agrège les métriques de plusieurs folds.
        Calcule des moyennes (pondérées si possible), médianes, std, MAD, et CIs bootstrap.
        """
        log_prefix_agg = f"{self.log_prefix}[AggFoldMetrics][{identifier}]"
        logger.info(f"{log_prefix_agg} Agrégation de {len(fold_results_list)} résultat(s) de fold.")

        if not fold_results_list:
            logger.warning(f"{log_prefix_agg} Liste de résultats de fold vide.")
            return None

        try:
            df_folds = pd.DataFrame(fold_results_list)
        except Exception as e_df_create:
            logger.error(f"{log_prefix_agg} Impossible de créer un DataFrame à partir de fold_results_list: {e_df_create}")
            return None
        
        if df_folds.empty:
            logger.warning(f"{log_prefix_agg} DataFrame des folds vide après création.")
            return None

        # Identifier les colonnes de métriques numériques à agréger
        if metrics_to_aggregate:
            numeric_metric_cols = [col for col in metrics_to_aggregate if col in df_folds.columns and pd.api.types.is_numeric_dtype(df_folds[col])]
        else:
            numeric_metric_cols = df_folds.select_dtypes(include=np.number).columns.tolist()

        if not numeric_metric_cols:
            logger.warning(f"{log_prefix_agg} Aucune colonne de métrique numérique trouvée ou spécifiée pour l'agrégation.")
            return AggregatedMetrics(identifier=identifier, count=len(df_folds), raw_fold_metrics_df=df_folds)

        logger.debug(f"{log_prefix_agg} Métriques numériques à agréger: {numeric_metric_cols}")
        
        agg_results = AggregatedMetrics(identifier=identifier, count=len(df_folds), raw_fold_metrics_df=df_folds.copy())
        
        weights: Optional[pd.Series] = None
        if weight_by_column and weight_by_column in df_folds.columns and pd.api.types.is_numeric_dtype(df_folds[weight_by_column]):
            weights_raw = df_folds[weight_by_column].astype(float)
            if weights_raw.sum() > 1e-9 and not weights_raw.isnull().all() and not (weights_raw < 0).any():
                weights = weights_raw / weights_raw.sum()
                logger.info(f"{log_prefix_agg} Utilisation de pondérations basées sur la colonne '{weight_by_column}'.")
            else:
                logger.warning(f"{log_prefix_agg} Colonne de pondération '{weight_by_column}' invalide (somme nulle/négative/NaNs). Pas de pondération.")
        elif weight_by_column:
            logger.warning(f"{log_prefix_agg} Colonne de pondération '{weight_by_column}' non trouvée ou non numérique. Pas de pondération.")


        for metric_col in numeric_metric_cols:
            metric_data = df_folds[metric_col].dropna().to_numpy()
            if len(metric_data) == 0:
                logger.debug(f"{log_prefix_agg} Métrique '{metric_col}' entièrement NaN. Agrégats seront None.")
                agg_results.metrics_mean[metric_col] = None
                agg_results.metrics_median[metric_col] = None
                # ... (autres à None)
                continue

            # Moyenne (pondérée si applicable)
            if weights is not None and len(weights) == len(df_folds[metric_col]):
                # Aligner les poids avec les données non-NaN pour la métrique actuelle
                valid_metric_indices = df_folds[metric_col].notna()
                weighted_avg = np.average(df_folds.loc[valid_metric_indices, metric_col], weights=weights[valid_metric_indices])
                agg_results.metrics_mean[metric_col] = float(weighted_avg) if pd.notna(weighted_avg) else None
            else:
                agg_results.metrics_mean[metric_col] = float(np.mean(metric_data)) if metric_data.size > 0 else None
            
            agg_results.metrics_median[metric_col] = float(np.median(metric_data)) if metric_data.size > 0 else None
            agg_results.metrics_std[metric_col] = float(np.std(metric_data, ddof=1)) if metric_data.size > 1 else None # ddof=1 pour std d'échantillon
            agg_results.metrics_mad[metric_col] = float(scipy_stats.median_abs_deviation(metric_data, scale='normal')) if metric_data.size > 0 else None

            # Intervalles de confiance Bootstrap pour la moyenne
            if metric_data.size >= 20: # Assez de données pour bootstrap
                ci_low, ci_high = self._bootstrap_ci(metric_data, np.mean)
                agg_results.metrics_mean_ci_lower[metric_col] = ci_low
                agg_results.metrics_mean_ci_upper[metric_col] = ci_high
            else:
                agg_results.metrics_mean_ci_lower[metric_col] = None
                agg_results.metrics_mean_ci_upper[metric_col] = None
        
        logger.info(f"{log_prefix_agg} Agrégation terminée. Moyenne de la première métrique ({numeric_metric_cols[0]}): {agg_results.metrics_mean.get(numeric_metric_cols[0])}")
        return agg_results

    def cross_strategy_comparison(
        self,
        strategy_results_map: Dict[str, AggregatedMetrics], # Clé: nom_stratégie, Valeur: AggregatedMetrics pour cette strat
        criteria_config: List[Dict[str, Any]], # Ex: [{"name": "Sharpe Ratio", "weight": 0.5, "is_beneficial": True}, ...]
        method: str = "TOPSIS_LIKE" # ou "WEIGHTED_SUM"
    ) -> Optional[ComparisonReport]:
        """
        Compare plusieurs stratégies basées sur leurs métriques agrégées.
        Implémente un classement multi-critères (simplifié, pas un TOPSIS complet sans lib).
        """
        log_prefix_comp = f"{self.log_prefix}[CrossStrategyComp]"
        logger.info(f"{log_prefix_comp} Comparaison de {len(strategy_results_map)} stratégie(s) avec méthode '{method}'.")

        if not strategy_results_map:
            logger.warning(f"{log_prefix_comp} Dictionnaire de résultats de stratégie vide.")
            return None
        if not criteria_config:
            logger.error(f"{log_prefix_comp} Configuration des critères manquante pour la comparaison.")
            return None

        report = ComparisonReport(method_used=method, criteria_used=[c['name'] for c in criteria_config])
        
        # Construire la matrice de décision: Stratégies (lignes) vs Critères (colonnes)
        decision_matrix_data: Dict[str, List[Optional[float]]] = {"strategy_name": []}
        for crit_cfg in criteria_config:
            decision_matrix_data[crit_cfg["name"]] = []

        for strat_name, agg_metrics_obj in strategy_results_map.items():
            decision_matrix_data["strategy_name"].append(strat_name)
            for crit_cfg in criteria_config:
                metric_name = crit_cfg["name"]
                # Utiliser la moyenne agrégée pour la comparaison
                value = agg_metrics_obj.metrics_mean.get(metric_name)
                decision_matrix_data[metric_name].append(value if value is not None and np.isfinite(value) else np.nan)
        
        df_decision = pd.DataFrame(decision_matrix_data).set_index("strategy_name")
        
        if df_decision.empty:
            logger.warning(f"{log_prefix_comp} Matrice de décision vide après extraction des métriques.")
            return report # Retourner un rapport vide mais initialisé

        # Gérer les NaNs (ex: par la moyenne de la colonne, ou en pénalisant)
        df_decision_filled = df_decision.apply(lambda x: x.fillna(x.mean() if x.notna().any() else 0), axis=0)
        
        # Normalisation (Vectorielle)
        # (x_ij - min_j(x_ij)) / (max_j(x_ij) - min_j(x_ij)) pour les critères bénéfiques
        # (max_j(x_ij) - x_ij) / (max_j(x_ij) - min_j(x_ij)) pour les critères de coût
        df_normalized = df_decision_filled.copy()
        for crit_cfg in criteria_config:
            col_name = crit_cfg["name"]
            col_data = df_decision_filled[col_name]
            min_val, max_val = col_data.min(), col_data.max()
            if (max_val - min_val) < 1e-9: # Si toutes les valeurs sont identiques
                df_normalized[col_name] = 0.5 if crit_cfg["is_beneficial"] else 0.5 # Score neutre
            elif crit_cfg["is_beneficial"]:
                df_normalized[col_name] = (col_data - min_val) / (max_val - min_val)
            else: # Critère de coût
                df_normalized[col_name] = (max_val - col_data) / (max_val - min_val)
        
        # Pondération
        weights = np.array([c.get("weight", 1.0 / len(criteria_config)) for c in criteria_config])
        if abs(weights.sum() - 1.0) > 1e-6 : # Normaliser les poids s'ils ne somment pas à 1
            logger.warning(f"{log_prefix_comp} La somme des poids des critères ({weights.sum()}) n'est pas 1. Normalisation.")
            weights = weights / weights.sum()
        report.weights_used = {c['name']: w for c, w in zip(criteria_config, weights)}

        df_weighted_normalized = df_normalized * weights
        
        # Calcul du score (Simple Somme Pondérée pour cet exemple "TOPSIS_LIKE")
        # Un vrai TOPSIS calculerait les distances à la solution idéale et anti-idéale.
        scores = df_weighted_normalized.sum(axis=1)
        
        ranked_strategies_list = []
        for strat_name, score_val in scores.sort_values(ascending=False).items():
            ranked_strategies_list.append({
                "strategy_name": strat_name,
                "score": float(score_val),
                "rank": len(ranked_strategies_list) + 1,
                "metrics_decision_matrix_raw": df_decision.loc[strat_name].to_dict(),
                "metrics_normalized_weighted": df_weighted_normalized.loc[strat_name].to_dict()
            })
        report.ranked_strategies = ranked_strategies_list
        
        # Tests statistiques (exemple Kruskal-Wallis si >2 stratégies, Mann-Whitney U pour paires)
        # Ces tests sont plus pertinents si on compare les *distributions* des métriques de fold
        # entre stratégies, pas seulement les moyennes agrégées.
        # Pour l'instant, on omet cette partie complexe.
        # report.statistical_tests_summary = {"status": "Non implémenté dans cette version."}

        logger.info(f"{log_prefix_comp} Comparaison de stratégies terminée. Meilleure stratégie (basée sur score): "
                    f"{report.ranked_strategies[0]['strategy_name'] if report.ranked_strategies else 'N/A'}")
        return report

    def detect_metric_anomalies(
        self,
        metrics_df: pd.DataFrame, # DataFrame où chaque ligne est une observation (ex: un fold), colonnes sont les métriques
        metrics_to_check: Optional[List[str]] = None,
        z_threshold: Optional[float] = None, # Si fourni, utilise z-score
        isolation_forest_contamination: Union[str, float] = 'auto' # Pour IsolationForest
    ) -> List[AnomalyDetails]:
        """Détecte les anomalies dans un ensemble de métriques."""
        log_prefix_anomaly = f"{self.log_prefix}[DetectAnomalies]"
        logger.info(f"{log_prefix_anomaly} Détection d'anomalies sur {len(metrics_df)} observations.")
        
        anomalies_detected: List[AnomalyDetails] = []
        if metrics_df.empty:
            return anomalies_detected

        cols_to_analyze = metrics_to_check if metrics_to_check else metrics_df.select_dtypes(include=np.number).columns.tolist()
        if not cols_to_analyze:
            logger.warning(f"{log_prefix_anomaly} Aucune colonne numérique à analyser pour les anomalies.")
            return anomalies_detected

        # Méthode 1: Z-score (si z_threshold est fourni)
        actual_z_threshold = z_threshold if z_threshold is not None else self.anomaly_z_threshold
        if actual_z_threshold > 0:
            for col in cols_to_analyze:
                if col not in metrics_df.columns: continue
                metric_series = metrics_df[col].dropna().astype(float)
                if len(metric_series) < 5: continue # Pas assez de points pour z-score significatif
                
                z_scores = np.abs(scipy_stats.zscore(metric_series, nan_policy='omit'))
                outliers_z = metric_series[z_scores > actual_z_threshold]
                for idx, val in outliers_z.items():
                    original_index = idx # Si metrics_df a un index significatif (ex: fold_id, strategy_name)
                    anomalies_detected.append(AnomalyDetails(
                        metric_name=col, value=float(val), z_score=float(z_scores.loc[idx]),
                        message=f"Z-score ({z_scores.loc[idx]:.2f}) > seuil ({actual_z_threshold})",
                        details={"original_index": original_index}
                    ))

        # Méthode 2: Isolation Forest
        # S'assurer que les données sont uniquement numériques et sans NaN/inf pour IsolationForest
        df_for_iso_forest = metrics_df[cols_to_analyze].copy()
        df_for_iso_forest.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Remplir les NaNs restants (ex: par la moyenne) ou supprimer les lignes/colonnes
        # Pour IsolationForest, il est souvent préférable de ne pas avoir de NaNs.
        # On peut imputer ou sélectionner des colonnes sans trop de NaNs.
        # Ici, on va imputer avec la moyenne pour la simplicité.
        for col_iso in df_for_iso_forest.columns:
            if df_for_iso_forest[col_iso].isnull().any():
                mean_val = df_for_iso_forest[col_iso].mean()
                df_for_iso_forest[col_iso].fillna(mean_val if pd.notna(mean_val) else 0, inplace=True)
        
        if not df_for_iso_forest.empty and len(df_for_iso_forest) >= 5: # Besoin d'assez de données
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df_for_iso_forest)
                
                model_iso = IsolationForest(contamination=isolation_forest_contamination, random_state=42)
                predictions_iso = model_iso.fit_predict(scaled_data) # -1 pour outlier, 1 pour inlier
                
                outlier_indices_iso = df_for_iso_forest.index[predictions_iso == -1]
                for original_idx_iso in outlier_indices_iso:
                    # Ajouter des détails si ce n'est pas déjà signalé par z-score pour la même métrique/index
                    # C'est une détection globale sur l'observation (ligne), pas par métrique.
                    # On peut lister les métriques qui ont contribué si on analyse les scores de décision.
                    # Pour l'instant, on signale l'observation.
                    is_already_reported_by_z = any(
                        ad.details.get("original_index") == original_idx_iso for ad in anomalies_detected
                    )
                    if not is_already_reported_by_z:
                         anomalies_detected.append(AnomalyDetails(
                            metric_name="ObservationGlobale", value=np.nan, # Pas une métrique spécifique
                            is_outlier_isolation_forest=True,
                            message=f"Détecté comme outlier par Isolation Forest.",
                            details={"original_index": original_idx_iso, "metrics_at_anomaly": metrics_df.loc[original_idx_iso].to_dict()}
                        ))
            except Exception as e_iso:
                logger.error(f"{log_prefix_anomaly} Erreur lors de l'exécution d'Isolation Forest: {e_iso}", exc_info=True)
        
        logger.info(f"{log_prefix_anomaly} Détection d'anomalies terminée. {len(anomalies_detected)} anomalie(s) potentielle(s) trouvée(s).")
        return anomalies_detected

    def generate_executive_summary(
        self,
        full_results_data: Dict[str, Any], # Structure complexe contenant tous les niveaux de résultats
        template_dir: Optional[Path] = None, # Répertoire pour les templates Jinja2
        template_name: str = "executive_summary_template.md"
    ) -> ExecutiveSummary:
        """
        Génère un résumé exécutif des résultats globaux.
        Utilise Jinja2 pour le formatage si disponible.
        """
        log_prefix_summary = f"{self.log_prefix}[ExecSummary]"
        logger.info(f"{log_prefix_summary} Génération du résumé exécutif...")

        summary_obj = ExecutiveSummary(generation_timestamp_utc=datetime.now(timezone.utc).isoformat())
        
        # Logique d'extraction des insights (simplifiée pour l'instant)
        # Exemple: trouver la meilleure stratégie basée sur une métrique clé
        # Supposons que full_results_data contient une clé 'comparison_report'
        comparison_report: Optional[ComparisonReport] = full_results_data.get('comparison_report')
        if comparison_report and comparison_report.ranked_strategies:
            summary_obj.overall_best_strategy = comparison_report.ranked_strategies[0]
            summary_obj.top_n_strategies = comparison_report.ranked_strategies[:min(3, len(comparison_report.ranked_strategies))]
            summary_obj.key_insights.append(
                f"Meilleure stratégie identifiée: {summary_obj.overall_best_strategy['strategy_name']} "
                f"(Score: {summary_obj.overall_best_strategy['score']:.3f})"
            )
        else:
            summary_obj.key_insights.append("Aucune comparaison de stratégie fournie ou aucune stratégie classée.")

        # Exemple: Résumer les anomalies
        anomalies_list: List[AnomalyDetails] = full_results_data.get('detected_anomalies', [])
        if anomalies_list:
            summary_obj.detected_anomalies_summary.append(f"{len(anomalies_list)} anomalie(s) de métrique détectée(s).")
            for anom in anomalies_list[:3]: # Lister les 3 premières
                summary_obj.detected_anomalies_summary.append(
                    f"  - Métrique: {anom.metric_name}, Valeur: {anom.value:.2f}, Raison: {anom.message} (Index: {anom.details.get('original_index')})"
                )
        else:
            summary_obj.detected_anomalies_summary.append("Aucune anomalie de métrique majeure détectée.")

        # Exemple: Stats globales (à extraire de full_results_data)
        # summary_obj.global_stats_summary = {"avg_sharpe_all_strategies": 0.5, ...}

        # Rendu Markdown avec Jinja2
        if JINJA2_AVAILABLE:
            actual_template_dir = template_dir if template_dir else Path(__file__).parent / "templates"
            if not actual_template_dir.is_dir():
                 logger.warning(f"{log_prefix_summary} Répertoire de templates Jinja2 non trouvé: {actual_template_dir}. "
                                "Le résumé Markdown ne sera pas formaté.")
                 summary_obj.report_markdown = "Erreur: Répertoire de templates Jinja2 non trouvé."
            else:
                env = Environment(
                    loader=FileSystemLoader(actual_template_dir),
                    autoescape=select_autoescape(['html', 'xml', 'md'])
                )
                try:
                    template = env.get_template(template_name)
                    summary_obj.report_markdown = template.render(summary=summary_obj)
                    logger.info(f"{log_prefix_summary} Résumé Markdown généré avec template {template_name}.")
                except Exception as e_jinja:
                    logger.error(f"{log_prefix_summary} Erreur lors du rendu du template Jinja2 '{template_name}': {e_jinja}", exc_info=True)
                    summary_obj.report_markdown = f"Erreur de rendu Jinja2: {e_jinja}"
        else:
            # Fallback si Jinja2 n'est pas disponible
            md_parts = [f"# Résumé Exécutif ({summary_obj.generation_timestamp_utc})\n"]
            if summary_obj.overall_best_strategy:
                md_parts.append(f"## Meilleure Stratégie\n- Nom: {summary_obj.overall_best_strategy['strategy_name']}\n- Score: {summary_obj.overall_best_strategy['score']:.3f}\n")
            md_parts.append("## Aperçu des Anomalies\n" + "\n".join([f"- {s}" for s in summary_obj.detected_anomalies_summary]))
            md_parts.append("\n## Perspectives Clés\n" + "\n".join([f"- {s}" for s in summary_obj.key_insights]))
            summary_obj.report_markdown = "\n".join(md_parts)
            logger.info(f"{log_prefix_summary} Résumé Markdown généré (fallback sans Jinja2).")
            
        return summary_obj

    def export_results(
        self,
        data_to_export: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        output_path_base: Path, # Ex: "results/run_id/strategy_report" (sans extension)
        formats: List[str] # Ex: ["json", "parquet", "excel"]
    ) -> Dict[str, Path]:
        """
        Exporte les données fournies (typiquement un DataFrame de métriques agrégées)
        vers les formats spécifiés.
        """
        log_prefix_export = f"{self.log_prefix}[ExportResults][{output_path_base.name}]"
        logger.info(f"{log_prefix_export} Exportation des résultats vers formats: {formats}")
        
        output_path_base.parent.mkdir(parents=True, exist_ok=True)
        exported_files: Dict[str, Path] = {}

        df_export: Optional[pd.DataFrame] = None
        if isinstance(data_to_export, pd.DataFrame):
            df_export = data_to_export
        elif isinstance(data_to_export, list) and all(isinstance(item, dict) for item in data_to_export):
            try:
                df_export = pd.DataFrame(data_to_export)
            except Exception as e_df_conv:
                logger.error(f"{log_prefix_export} Impossible de convertir la liste de dicts en DataFrame pour l'export: {e_df_conv}")
        elif isinstance(data_to_export, dict): # Si c'est un seul dict (ex: AggregatedMetrics)
            try:
                # Tenter de le normaliser en DataFrame si c'est une structure plate de métriques
                # Ou le sauvegarder directement en JSON
                if "json" in formats:
                    json_path = output_path_base.with_suffix(".json")
                    with open(json_path, 'w', encoding='utf-8') as f_json:
                        json.dump(data_to_export, f_json, indent=2, default=str) # default=str pour les types non sérialisables
                    exported_files["json"] = json_path
                    logger.info(f"{log_prefix_export} Dictionnaire exporté en JSON: {json_path}")
                
                # Pour Parquet/Excel, on a besoin d'un DataFrame. Si ce n'est pas convertible, on saute.
                # On pourrait essayer de créer un DF à partir du dict si c'est pertinent.
                # Ex: pd.DataFrame([data_to_export]) si c'est un ensemble de métriques.
                # Pour l'instant, on ne le fait que si data_to_export est déjà un DF ou une liste de dicts.
            except Exception as e_json_single_dict:
                 logger.error(f"{log_prefix_export} Erreur export dict en JSON: {e_json_single_dict}")


        if df_export is not None and not df_export.empty:
            if "json" in formats:
                json_path = output_path_base.with_suffix(".json")
                try:
                    df_export.to_json(json_path, orient="records", indent=2, date_format="iso", default_handler=str)
                    exported_files["json"] = json_path
                    logger.info(f"{log_prefix_export} Exporté en JSON: {json_path}")
                except Exception as e_json:
                    logger.error(f"{log_prefix_export} Erreur export JSON: {e_json}")
            
            if "parquet" in formats:
                parquet_path = output_path_base.with_suffix(".parquet")
                try:
                    df_export.to_parquet(parquet_path, index=False)
                    exported_files["parquet"] = parquet_path
                    logger.info(f"{log_prefix_export} Exporté en Parquet: {parquet_path}")
                except Exception as e_parquet:
                    logger.error(f"{log_prefix_export} Erreur export Parquet: {e_parquet}")

            if "excel" in formats:
                excel_path = output_path_base.with_suffix(".xlsx")
                try:
                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer: # type: ignore
                        df_export.to_excel(writer, sheet_name='Metrics', index=False)
                        # TODO: Ajouter du styling si OPENPYXL_AVAILABLE
                        if OPENPYXL_AVAILABLE:
                            workbook = writer.book
                            worksheet = writer.sheets['Metrics']
                            # Exemple de styling (ajuster les colonnes, etc.)
                            for column_cells in worksheet.columns:
                                length = max(len(str(cell.value)) for cell in column_cells)
                                worksheet.column_dimensions[column_cells[0].column_letter].width = length + 2
                    exported_files["excel"] = excel_path
                    logger.info(f"{log_prefix_export} Exporté en Excel: {excel_path}")
                except Exception as e_excel:
                    logger.error(f"{log_prefix_export} Erreur export Excel: {e_excel}. Assurez-vous que 'openpyxl' est installé.")
        
        elif df_export is None and "json" not in exported_files : # Si ce n'était pas un dict exportable en JSON et pas convertible en DF
             logger.warning(f"{log_prefix_export} Données non exportables en DataFrame pour Parquet/Excel, et non exportées en JSON.")


        return exported_files

