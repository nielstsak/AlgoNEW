# src/core/data_validator.py
"""
Module centralisé pour la validation et la préparation des données OHLCV et
des indicateurs.
"""
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytz # Pour la gestion des fuseaux horaires

# Tentative d'importation de l'interface IDataValidator
# Cela permet une vérification de type statique et runtime si disponible.
try:
    from src.core.interfaces import IDataValidator
except ImportError:
    # Définir un protocole factice si l'original n'est pas trouvable
    # pour que le reste du code puisse être analysé sans erreur d'import.
    class IDataValidator: # type: ignore
        def validate_ohlcv_data(self, df: pd.DataFrame, required_columns: List[str],
                                expected_frequency: Optional[str] = None,
                                symbol: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]: ...
        def ensure_datetime_index(self, df: pd.DataFrame, target_tz: str = "UTC") -> pd.DataFrame: ...
        def validate_indicators(self, df: pd.DataFrame, indicator_configs: List[Dict[str, Any]]) -> Dict[str, Any]: ...
    warnings.warn(
        "IDataValidator interface not found. Using a placeholder. "
        "Ensure src.core.interfaces is in PYTHONPATH for full type safety."
    )


logger = logging.getLogger(__name__)

@dataclass
class ValidationReport:
    """
    Rapport de validation pour les données.
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)

@dataclass
class AnomalyReport:
    """
    Rapport sur les anomalies détectées dans les données.
    """
    anomalies_found: bool = False
    gap_details: List[Dict[str, Any]] = field(default_factory=list)
    outlier_details: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def add_gap(self, start_gap: pd.Timestamp, end_gap: pd.Timestamp, duration: pd.Timedelta):
        self.gap_details.append({
            "start_gap_utc": start_gap.isoformat(),
            "end_gap_utc": end_gap.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "duration_human": str(duration)
        })
        self.anomalies_found = True

    def add_outlier(self, column: str, index: Union[pd.Timestamp, int], value: Any, reason: str):
        timestamp_iso = index.isoformat() if isinstance(index, pd.Timestamp) else str(index)
        self.outlier_details.append({
            "column": column,
            "timestamp_utc_or_index": timestamp_iso,
            "value": str(value), # Convertir en str pour la sérialisation
            "reason": reason
        })
        self.anomalies_found = True


class DataValidator(IDataValidator):
    """
    Classe pour la validation centralisée des données OHLCV et des indicateurs.
    Implémente le protocole IDataValidator.
    """

    def __init__(self, default_ohlcv_cols: Optional[List[str]] = None):
        """
        Initialise le DataValidator.

        Args:
            default_ohlcv_cols (Optional[List[str]]): Liste des colonnes OHLCV
                par défaut à vérifier (ex: ['open', 'high', 'low', 'close', 'volume']).
        """
        self.default_ohlcv_cols = default_ohlcv_cols if default_ohlcv_cols else \
                                  ['open', 'high', 'low', 'close', 'volume']
        self.log_prefix = "[DataValidator]"
        logger.info(f"{self.log_prefix} Initialisé.")

    def ensure_datetime_index(
        self,
        df: pd.DataFrame,
        target_tz_str: str = "UTC"
    ) -> pd.DataFrame:
        """
        S'assure que le DataFrame a un DatetimeIndex, qu'il est trié, unique,
        et localisé dans le fuseau horaire cible (par défaut UTC).

        Args:
            df (pd.DataFrame): DataFrame à traiter.
            target_tz_str (str): Fuseau horaire cible (ex: "UTC").

        Returns:
            pd.DataFrame: DataFrame avec un index DatetimeIndex standardisé.
                          Retourne une copie du DataFrame original si des erreurs
                          critiques empêchent la standardisation de l'index.
        """
        log_prefix_idx = f"{self.log_prefix}[EnsureDatetimeIndex]"
        df_copy = df.copy()

        if not isinstance(df_copy.index, pd.DatetimeIndex):
            if 'timestamp' in df_copy.columns:
                logger.debug(f"{log_prefix_idx} Index n'est pas DatetimeIndex. Conversion depuis colonne 'timestamp'.")
                try:
                    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
                    df_copy = df_copy.set_index('timestamp', drop=True)
                    if not isinstance(df_copy.index, pd.DatetimeIndex): # Double check
                         raise ValueError("La conversion de la colonne 'timestamp' en DatetimeIndex a échoué.")
                except Exception as e_conv_ts_col:
                    logger.error(f"{log_prefix_idx} Échec de la conversion de la colonne 'timestamp' en DatetimeIndex: {e_conv_ts_col}")
                    return df # Retourner l'original en cas d'échec critique
            else:
                logger.error(f"{log_prefix_idx} Index n'est pas DatetimeIndex et colonne 'timestamp' non trouvée.")
                return df

        # Gestion du fuseau horaire
        try:
            target_tz_info = pytz.timezone(target_tz_str)
            if df_copy.index.tz is None:
                logger.debug(f"{log_prefix_idx} Index naïf. Localisation en {target_tz_str}.")
                df_copy.index = df_copy.index.tz_localize(target_tz_info, ambiguous='infer', nonexistent='shift_forward')
            elif df_copy.index.tz.zone != target_tz_info.zone: # type: ignore
                logger.debug(f"{log_prefix_idx} Index en {df_copy.index.tz}. Conversion vers {target_tz_str}.")
                df_copy.index = df_copy.index.tz_convert(target_tz_info)
        except Exception as e_tz:
            logger.error(f"{log_prefix_idx} Erreur lors de la gestion du fuseau horaire de l'index: {e_tz}")
            return df # Retourner l'original

        # Tri de l'index
        if not df_copy.index.is_monotonic_increasing:
            logger.debug(f"{log_prefix_idx} Index non trié. Tri en cours.")
            df_copy = df_copy.sort_index()

        # Gestion des doublons dans l'index
        if df_copy.index.has_duplicates:
            num_duplicates = df_copy.index.duplicated().sum()
            logger.warning(f"{log_prefix_idx} {num_duplicates} timestamp(s) dupliqué(s) trouvé(s) dans l'index. Conservation de la dernière occurrence.")
            df_copy = df_copy[~df_copy.index.duplicated(keep='last')]

        logger.debug(f"{log_prefix_idx} Index DatetimeIndex standardisé avec succès.")
        return df_copy

    def validate_ohlcv_data(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        expected_frequency: Optional[str] = None, # Ex: "1min", "5T", "1H"
        symbol: Optional[str] = None
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """
        Valide la structure, les types, et la cohérence des données OHLCV.
        """
        report = ValidationReport()
        df_validated = df.copy()
        log_ctx_ohlcv = f"{self.log_prefix}[ValidateOHLCVData]"
        if symbol:
            log_ctx_ohlcv = f"{self.log_prefix}[{symbol}][ValidateOHLCVData]"

        logger.info(f"{log_ctx_ohlcv} Démarrage de la validation des données OHLCV. Shape initial: {df_validated.shape}")

        # 1. Standardiser l'index DatetimeIndex
        try:
            df_validated = self.ensure_datetime_index(df_validated)
            if not isinstance(df_validated.index, pd.DatetimeIndex) or df_validated.index.tz is None:
                 report.add_error("L'index n'a pas pu être standardisé en DatetimeIndex UTC.")
                 logger.error(f"{log_ctx_ohlcv} Échec critique de la standardisation de l'index.")
                 return df, report # Retourner l'original si l'index n'est pas bon
        except Exception as e_idx_std:
            report.add_error(f"Erreur lors de la standardisation de l'index DatetimeIndex: {e_idx_std}")
            logger.error(f"{log_ctx_ohlcv} Erreur critique lors de ensure_datetime_index: {e_idx_std}")
            return df, report

        # 2. Vérifier les colonnes requises
        cols_to_check = required_columns if required_columns else self.default_ohlcv_cols
        missing_cols = [col for col in cols_to_check if col not in df_validated.columns]
        if missing_cols:
            msg = f"Colonnes OHLCV requises manquantes : {', '.join(missing_cols)}."
            report.add_error(msg)
            logger.error(f"{log_ctx_ohlcv} {msg} Colonnes disponibles: {df_validated.columns.tolist()}")
            # Ne pas retourner ici, continuer la validation sur les colonnes présentes

        present_ohlcv_cols = [col for col in self.default_ohlcv_cols if col in df_validated.columns]

        # 3. Vérifier les types et les valeurs
        for col in present_ohlcv_cols:
            # Type
            if not pd.api.types.is_numeric_dtype(df_validated[col]):
                try:
                    df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
                    report.add_warning(f"Colonne '{col}' non numérique, tentative de conversion. NaNs introduits: {df_validated[col].isnull().sum()}.")
                except Exception as e_conv_num:
                    report.add_error(f"Colonne '{col}' n'est pas de type numérique et la conversion a échoué: {e_conv_num}.")
                    continue # Passer à la colonne suivante si la conversion échoue

            # NaNs
            nan_count = df_validated[col].isnull().sum()
            if nan_count > 0:
                report.add_warning(f"Colonne '{col}' contient {nan_count} valeur(s) NaN ({nan_count / len(df_validated):.2%}).")
                # Option: remplir les NaNs ici. Pour l'instant, on se contente de rapporter.
                # df_validated[col] = df_validated[col].ffill().bfill()

            # Valeurs négatives (pour prix et volume)
            if col in ['open', 'high', 'low', 'close', 'volume']:
                if (df_validated[col] < 0).any():
                    num_neg = (df_validated[col] < 0).sum()
                    report.add_error(f"Colonne '{col}' contient {num_neg} valeur(s) négative(s).")
            if col == 'volume' and (df_validated[col] == 0).any(): # Volume nul peut être valide mais à noter
                num_zero_vol = (df_validated[col] == 0).sum()
                report.add_warning(f"Colonne 'volume' contient {num_zero_vol} valeur(s) nulle(s).")

        # Cohérence H L O C
        if all(c in df_validated.columns for c in ['high', 'low']):
            if (df_validated['high'] < df_validated['low']).any():
                report.add_error("Incohérence détectée : 'high' < 'low' pour certaines lignes.")
        
        if all(c in df_validated.columns for c in ['high', 'open', 'close']):
            if (df_validated['high'] < df_validated['open']).any() or \
               (df_validated['high'] < df_validated['close']).any():
                report.add_warning("Avertissement : 'high' est inférieur à 'open' ou 'close' pour certaines lignes.")

        if all(c in df_validated.columns for c in ['low', 'open', 'close']):
            if (df_validated['low'] > df_validated['open']).any() or \
               (df_validated['low'] > df_validated['close']).any():
                report.add_warning("Avertissement : 'low' est supérieur à 'open' ou 'close' pour certaines lignes.")

        # 4. Vérification de fréquence et gaps (si expected_frequency est fourni)
        # Utilise une partie de la logique de detect_data_anomalies
        if expected_frequency and not df_validated.empty and isinstance(df_validated.index, pd.DatetimeIndex):
            try:
                expected_timedelta = pd.to_timedelta(expected_frequency)
                time_diffs = df_validated.index.to_series().diff()
                # Un gap est > 1.5 * expected_timedelta (tolérance pour petites variations)
                # et on ignore le premier diff qui est NaT.
                gaps = time_diffs[time_diffs > (expected_timedelta * 1.5)]
                if not gaps.empty:
                    report.add_warning(f"{len(gaps)} gap(s) temporel(s) détecté(s) par rapport à la fréquence attendue '{expected_frequency}'.")
                    # Ajouter plus de détails si nécessaire dans le rapport
                    report.summary_stats['temporal_gaps_detected'] = len(gaps)
                    report.summary_stats['largest_gap_duration'] = str(gaps.max())
            except ValueError:
                report.add_warning(f"La fréquence attendue '{expected_frequency}' n'est pas un format valide pour pd.to_timedelta.")
            except Exception as e_freq_check:
                report.add_warning(f"Erreur lors de la vérification de la fréquence des données : {e_freq_check}")

        report.summary_stats['initial_rows'] = len(df)
        report.summary_stats['rows_after_index_cleaning'] = len(df_validated)
        report.summary_stats['ohlcv_columns_validated'] = present_ohlcv_cols
        if present_ohlcv_cols:
             report.summary_stats['nan_counts_final'] = {
                 col: int(df_validated[col].isnull().sum()) for col in present_ohlcv_cols if col in df_validated
             }

        logger.info(f"{log_ctx_ohlcv} Validation OHLCV terminée. Est valide: {report.is_valid}. "
                    f"Erreurs: {len(report.errors)}, Avertissements: {len(report.warnings)}.")
        return df_validated, report

    def validate_indicators(
        self,
        df: pd.DataFrame, # DataFrame contenant déjà les indicateurs calculés
        indicator_configs: List[Dict[str, Any]] # Configurations des indicateurs attendus
    ) -> ValidationReport:
        """
        Valide la présence et la validité de base des colonnes d'indicateurs.
        """
        report = ValidationReport()
        log_ctx_indic = f"{self.log_prefix}[ValidateIndicators]"
        logger.info(f"{log_ctx_indic} Démarrage de la validation des indicateurs.")

        if df.empty:
            report.add_error("Le DataFrame fourni pour la validation des indicateurs est vide.")
            return report

        for config in indicator_configs:
            outputs_config = config.get('outputs')
            indic_name_for_log = config.get('indicator_name', 'Inconnu')

            output_column_names_to_check: List[str] = []
            if isinstance(outputs_config, str):
                output_column_names_to_check.append(outputs_config)
            elif isinstance(outputs_config, dict):
                output_column_names_to_check.extend(list(outputs_config.values()))
            else:
                report.add_error(f"Configuration 'outputs' invalide pour l'indicateur '{indic_name_for_log}': {outputs_config}")
                continue

            for output_col_name in output_column_names_to_check:
                if not isinstance(output_col_name, str) or not output_col_name:
                    report.add_error(f"Nom de colonne de sortie invalide ('{output_col_name}') dans la config de '{indic_name_for_log}'.")
                    continue

                if output_col_name not in df.columns:
                    report.add_error(f"Indicateur '{indic_name_for_log}': Colonne de sortie attendue '{output_col_name}' non trouvée dans le DataFrame.")
                    continue # Ne peut pas valider davantage si la colonne manque

                # Validation de base pour la colonne d'indicateur présente
                series_indic = df[output_col_name]
                if not pd.api.types.is_numeric_dtype(series_indic.dtype):
                    report.add_warning(f"Indicateur '{indic_name_for_log}' (colonne '{output_col_name}'): "
                                       f"Type de données non numérique ({series_indic.dtype}). Devrait être float/int.")

                nan_count_indic = series_indic.isnull().sum()
                if nan_count_indic > 0:
                    report.add_warning(f"Indicateur '{indic_name_for_log}' (colonne '{output_col_name}'): "
                                       f"Contient {nan_count_indic} valeur(s) NaN ({nan_count_indic / len(series_indic):.2%}).")
                
                # Vérifier les infinis
                inf_count = np.isinf(series_indic.replace([None], np.nan)).sum() # Remplacer None par NaN pour isinf
                if inf_count > 0:
                    report.add_warning(f"Indicateur '{indic_name_for_log}' (colonne '{output_col_name}'): "
                                       f"Contient {inf_count} valeur(s) infinie(s).")

        logger.info(f"{log_ctx_indic} Validation des indicateurs terminée. Est valide (basé sur erreurs): {report.is_valid}. "
                    f"Erreurs: {len(report.errors)}, Avertissements: {len(report.warnings)}.")
        return report

    def detect_data_anomalies(
        self,
        df: pd.DataFrame,
        anomaly_config: Optional[Dict[str, Any]] = None # Ex: {"gap_threshold_factor": 2.0, "price_min": 0.00000001}
    ) -> Tuple[pd.DataFrame, AnomalyReport]:
        """
        Détecte les anomalies communes dans les données (gaps temporels, valeurs aberrantes).
        """
        report = AnomalyReport()
        df_checked = df.copy() # Travailler sur une copie
        log_ctx_anomaly = f"{self.log_prefix}[DetectAnomalies]"
        logger.info(f"{log_ctx_anomaly} Démarrage de la détection d'anomalies. Shape initial: {df_checked.shape}")

        cfg = anomaly_config if anomaly_config else {}
        gap_threshold_factor = float(cfg.get("gap_threshold_factor", 2.5)) # Facteur pour détecter les gaps
        min_valid_price = float(cfg.get("price_min_threshold", 1e-9)) # Prix minimum valide
        max_valid_volume_change_factor = float(cfg.get("volume_max_spike_factor", 100.0)) # Ex: volume ne peut pas augmenter de 100x en 1 barre

        # 1. S'assurer d'un DatetimeIndex UTC trié et unique
        try:
            df_checked = self.ensure_datetime_index(df_checked)
            if not isinstance(df_checked.index, pd.DatetimeIndex) or df_checked.index.tz is None:
                 report.recommendations.append("L'index n'a pas pu être standardisé en DatetimeIndex UTC. Détection d'anomalies temporelles impossible.")
                 logger.error(f"{log_ctx_anomaly} Échec critique de la standardisation de l'index pour détection d'anomalies.")
                 report.anomalies_found = True # Marquer comme ayant des anomalies si l'index est mauvais
                 return df, report # Retourner l'original
        except Exception as e_idx_std_anomaly:
            report.recommendations.append(f"Erreur lors de la standardisation de l'index: {e_idx_std_anomaly}")
            logger.error(f"{log_ctx_anomaly} Erreur critique lors de ensure_datetime_index pour détection d'anomalies: {e_idx_std_anomaly}")
            report.anomalies_found = True
            return df, report

        # 2. Détection des Gaps Temporels
        if len(df_checked.index) > 1:
            time_diffs = df_checked.index.to_series().diff()
            median_freq = time_diffs.median() # Fréquence typique des données

            if pd.notna(median_freq) and median_freq > pd.Timedelta(0):
                gap_threshold_td = median_freq * gap_threshold_factor
                logger.debug(f"{log_ctx_anomaly} Fréquence médiane détectée: {median_freq}. Seuil de gap: {gap_threshold_td}.")
                
                potential_gaps = time_diffs[time_diffs > gap_threshold_td]
                for gap_end_ts, gap_duration in potential_gaps.items():
                    gap_start_ts_approx = gap_end_ts - gap_duration # Timestamp avant le gap
                    report.add_gap(gap_start_ts_approx, gap_end_ts, gap_duration)
                    logger.warning(f"{log_ctx_anomaly} Gap temporel détecté: de ~{gap_start_ts_approx} à {gap_end_ts}, durée {gap_duration}.")
            else:
                logger.warning(f"{log_ctx_anomaly} Fréquence médiane non calculable ou nulle. Détection de gap temporel sautée.")
                report.recommendations.append("Fréquence médiane non calculable, détection de gap sautée.")

        # 3. Détection des Valeurs Aberrantes (Prix et Volume)
        ohlcv_cols_present = [col for col in self.default_ohlcv_cols if col in df_checked.columns]
        price_cols = [col for col in ohlcv_cols_present if col in ['open', 'high', 'low', 'close']]
        volume_col = 'volume' if 'volume' in ohlcv_cols_present else None

        for p_col in price_cols:
            negative_prices = df_checked[df_checked[p_col] < min_valid_price]
            for idx, row_val in negative_prices[p_col].items():
                report.add_outlier(p_col, idx, row_val, f"Prix < seuil minimum ({min_valid_price})")
        
        if volume_col:
            negative_volumes = df_checked[df_checked[volume_col] < 0]
            for idx, row_val in negative_volumes[volume_col].items():
                report.add_outlier(volume_col, idx, row_val, "Volume négatif")
            
            # Détection de pics de volume (exemple simple)
            volume_diff_pct = df_checked[volume_col].pct_change().abs()
            volume_spikes = volume_diff_pct[volume_diff_pct > max_valid_volume_change_factor]
            for idx, spike_pct_change in volume_spikes.items():
                actual_volume_value = df_checked.loc[idx, volume_col]
                report.add_outlier(volume_col, idx, actual_volume_value,
                                    f"Pic de volume excessif (changement > {max_valid_volume_change_factor*100:.0f}%)")
        
        if report.anomalies_found:
            logger.warning(f"{log_ctx_anomaly} Détection d'anomalies terminée. {len(report.gap_details)} gap(s) et "
                           f"{len(report.outlier_details)} valeur(s) aberrante(s) trouvée(s).")
            if not report.recommendations:
                 report.recommendations.append("Examen manuel des gaps et outliers recommandé.")
        else:
            logger.info(f"{log_ctx_anomaly} Détection d'anomalies terminée. Aucune anomalie majeure trouvée selon les critères actuels.")

        return df_checked, report

