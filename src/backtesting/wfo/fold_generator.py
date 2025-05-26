# src/backtesting/wfo/fold_generator.py
"""
Ce module est responsable de la génération des découpages (folds)
In-Sample (IS) et Out-of-Sample (OOS) à partir d'un DataFrame de données
enrichies, en respectant la méthodologie Walk-Forward Optimization (WFO) configurée.
Il implémente une logique de fenêtres IS expansives avec une période OOS fixe.
"""
import logging
from typing import List, Tuple, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np # Pour np.linspace si besoin pour la segmentation
from datetime import timezone # Pour s'assurer que les timestamps sont UTC

if TYPE_CHECKING:
    from src.config.definitions import WfoSettings

logger = logging.getLogger(__name__)

class WfoFoldGenerator:
    """
    Génère les folds In-Sample (IS) et Out-of-Sample (OOS) pour le
    Walk-Forward Optimization, en utilisant une méthode de fenêtres IS expansives
    et une période OOS fixe.
    """

    def __init__(self, wfo_settings: 'WfoSettings'):
        """
        Initialise le générateur de folds.

        Args:
            wfo_settings (WfoSettings): Les paramètres de configuration pour le WFO.
        """
        self.wfo_settings = wfo_settings
        self.log_prefix = "[WfoFoldGenerator]"
        logger.info(f"{self.log_prefix} Initialisé avec les paramètres WFO : {wfo_settings}")

    def _validate_data_and_settings(self,
                                    df_enriched_data: pd.DataFrame,
                                    effective_wfo_start_date: pd.Timestamp,
                                    effective_wfo_end_date: pd.Timestamp) -> bool:
        """
        Valide les données d'entrée et les paramètres WFO pour la génération des folds.
        """
        if df_enriched_data.empty:
            logger.error(f"{self.log_prefix} Le DataFrame de données enrichies est vide.")
            return False
        if not isinstance(df_enriched_data.index, pd.DatetimeIndex):
            logger.error(f"{self.log_prefix} L'index de df_enriched_data doit être un DatetimeIndex.")
            return False
        if df_enriched_data.index.tz is None or df_enriched_data.index.tz.utcoffset(None) != timezone.utc.utcoffset(None): # type: ignore
            logger.error(f"{self.log_prefix} L'index de df_enriched_data doit être timezone-aware et en UTC. Actuel: {df_enriched_data.index.tz}")
            return False
        if not df_enriched_data.index.is_monotonic_increasing:
            logger.error(f"{self.log_prefix} L'index de df_enriched_data doit être trié de manière croissante.")
            return False # Un index non trié peut causer des problèmes de slicing

        if effective_wfo_start_date >= effective_wfo_end_date:
            logger.error(f"{self.log_prefix} La date de début WFO effective ({effective_wfo_start_date}) "
                         f"doit être antérieure à la date de fin WFO effective ({effective_wfo_end_date}).")
            return False

        min_total_duration_needed = pd.Timedelta(days=self.wfo_settings.oos_period_days + self.wfo_settings.min_is_period_days)
        actual_total_duration = effective_wfo_end_date - effective_wfo_start_date

        if actual_total_duration < min_total_duration_needed:
            logger.error(f"{self.log_prefix} Durée totale effective des données ({actual_total_duration}) "
                         f"est insuffisante pour une période OOS de {self.wfo_settings.oos_period_days} jours "
                         f"et une période IS minimale de {self.wfo_settings.min_is_period_days} jours. "
                         f"Nécessaire au moins : {min_total_duration_needed}.")
            return False

        if self.wfo_settings.n_splits <= 0:
            logger.error(f"{self.log_prefix} Le nombre de splits (n_splits={self.wfo_settings.n_splits}) doit être positif.")
            return False
        
        logger.debug(f"{self.log_prefix} Validation des entrées et des paramètres WFO réussie.")
        return True

    def generate_folds(self,
                       df_enriched_data: pd.DataFrame,
                       is_total_start_ts_config: Optional[pd.Timestamp] = None, # Date de début globale pour IS (config)
                       oos_total_end_ts_config: Optional[pd.Timestamp] = None   # Date de fin globale pour OOS (config)
                       ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Génère les folds IS et OOS selon la configuration (fenêtres IS expansives, OOS fixe).

        Args:
            df_enriched_data (pd.DataFrame): Le DataFrame complet des données enrichies,
                                             avec un DatetimeIndex UTC trié et unique.
            is_total_start_ts_config (Optional[pd.Timestamp]): Timestamp de début global
                pour la période In-Sample, tel que configuré. Si None, utilise le début des données.
            oos_total_end_ts_config (Optional[pd.Timestamp]): Timestamp de fin global
                pour la période Out-of-Sample, tel que configuré. Si None, utilise la fin des données.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
            Une liste de tuples, où chaque tuple représente un fold et contient :
            (df_is_fold, df_oos_fold, fold_id, is_start_ts, is_end_ts, oos_start_ts, oos_end_ts).
            Retourne une liste vide si aucun fold valide ne peut être généré.
        """
        logger.info(f"{self.log_prefix} Début de la génération des folds WFO de type '{self.wfo_settings.fold_type}'.")

        # 1. Déterminer les dates de début et de fin effectives pour le WFO global
        data_min_ts = df_enriched_data.index.min()
        data_max_ts = df_enriched_data.index.max()

        # S'assurer que les dates de configuration sont timezone-aware (UTC)
        def _ensure_utc(ts: Optional[pd.Timestamp], default_tz_ref: pd.Timestamp) -> Optional[pd.Timestamp]:
            if ts is None: return None
            if ts.tzinfo is None: return ts.tz_localize(default_tz_ref.tzinfo) # type: ignore
            if ts.tzinfo != default_tz_ref.tzinfo : return ts.tz_convert(default_tz_ref.tzinfo) # type: ignore
            return ts

        is_total_start_ts_config = _ensure_utc(is_total_start_ts_config, data_min_ts)
        oos_total_end_ts_config = _ensure_utc(oos_total_end_ts_config, data_max_ts)

        effective_wfo_start_date = is_total_start_ts_config if is_total_start_ts_config else data_min_ts
        effective_wfo_end_date = oos_total_end_ts_config if oos_total_end_ts_config else data_max_ts
        
        effective_wfo_start_date = max(effective_wfo_start_date, data_min_ts)
        effective_wfo_end_date = min(effective_wfo_end_date, data_max_ts)
        
        df_wfo_period_data = df_enriched_data.loc[effective_wfo_start_date:effective_wfo_end_date]
        if df_wfo_period_data.empty:
            logger.error(f"{self.log_prefix} Aucune donnée disponible dans la période WFO effective : "
                         f"[{effective_wfo_start_date} - {effective_wfo_end_date}].")
            return []
        
        # Utiliser les dates min/max réelles des données slicées pour la validation
        actual_wfo_data_start_ts = df_wfo_period_data.index.min()
        actual_wfo_data_end_ts = df_wfo_period_data.index.max()

        if not self._validate_data_and_settings(df_wfo_period_data, actual_wfo_data_start_ts, actual_wfo_data_end_ts):
            return []
        
        logger.info(f"{self.log_prefix} Période WFO effective pour la génération des folds : "
                    f"de {actual_wfo_data_start_ts} à {actual_wfo_data_end_ts}.")

        # 2. Logique de découpage spécifique au type de fold
        if self.wfo_settings.fold_type.lower() == "expanding":
            return self._generate_expanding_is_fixed_oos_folds(df_wfo_period_data)
        else:
            logger.error(f"{self.log_prefix} Type de fold '{self.wfo_settings.fold_type}' non supporté. "
                         "Seul 'expanding' (avec OOS fixe) est implémenté.")
            return []

    def _generate_expanding_is_fixed_oos_folds(self,
                                               df_wfo_data: pd.DataFrame
                                               ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Génère des folds avec une période OOS fixe à la fin et des périodes IS expansives.
        """
        folds_generated: List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        
        # 1. Définir la période OOS fixe (à la fin de df_wfo_data)
        oos_duration = pd.Timedelta(days=self.wfo_settings.oos_period_days)
        oos_fixed_end_ts = df_wfo_data.index.max()
        
        # Le début de OOS est calculé en reculant depuis la fin.
        # On cherche le premier timestamp dans les données qui est >= (oos_fixed_end_ts - oos_duration + 1 jour_min)
        oos_calculated_start_target = oos_fixed_end_ts - oos_duration + pd.Timedelta(microseconds=1) # Pour inclure le jour de fin et éviter chevauchement strict
        
        # Trouver le premier timestamp disponible dans les données à partir de oos_calculated_start_target
        df_potential_oos_starts = df_wfo_data.loc[df_wfo_data.index >= oos_calculated_start_target]
        if df_potential_oos_starts.empty:
            logger.error(f"{self.log_prefix} Impossible de déterminer un début OOS valide. "
                         f"Début OOS calculé ({oos_calculated_start_target}) est après la fin des données WFO disponibles ({oos_fixed_end_ts}) "
                         "ou la période OOS est trop longue pour les données disponibles.")
            return []
        actual_oos_fixed_start_ts = df_potential_oos_starts.index.min()
        
        df_oos_fixed_final = df_wfo_data.loc[actual_oos_fixed_start_ts : oos_fixed_end_ts]
        
        if df_oos_fixed_final.empty:
            logger.error(f"{self.log_prefix} La période OOS fixe (de {actual_oos_fixed_start_ts} à {oos_fixed_end_ts}) est vide. "
                         "Vérifiez oos_period_days et la plage de données.")
            return []
        
        actual_oos_fixed_end_ts = df_oos_fixed_final.index.max() # La fin réelle des données OOS
        logger.info(f"{self.log_prefix} Période OOS fixe définie : de {actual_oos_fixed_start_ts} à {actual_oos_fixed_end_ts}.")

        # 2. Définir la période IS totale disponible (avant la période OOS fixe)
        is_total_available_end_ts = actual_oos_fixed_start_ts - pd.Timedelta(microseconds=1)
        is_total_available_start_ts = df_wfo_data.index.min()

        df_is_total_available = df_wfo_data.loc[is_total_available_start_ts : is_total_available_end_ts]

        if df_is_total_available.empty:
            logger.error(f"{self.log_prefix} La période IS totale disponible (de {is_total_available_start_ts} à {is_total_available_end_ts}) est vide. "
                         "Cela peut se produire si la période OOS couvre toutes les données WFO disponibles ou plus.")
            return []

        actual_is_total_start_ts = df_is_total_available.index.min()
        actual_is_total_end_ts = df_is_total_available.index.max()
        duration_is_total_available = actual_is_total_end_ts - actual_is_total_start_ts
        
        logger.info(f"{self.log_prefix} Période IS totale disponible définie : de {actual_is_total_start_ts} à {actual_is_total_end_ts} (Durée: {duration_is_total_available}).")

        min_is_duration_needed_for_any_fold = pd.Timedelta(days=self.wfo_settings.min_is_period_days)
        if duration_is_total_available < min_is_duration_needed_for_any_fold:
            logger.error(f"{self.log_prefix} La durée IS totale disponible ({duration_is_total_available}) "
                         f"est inférieure à min_is_period_days ({self.wfo_settings.min_is_period_days} jours). Impossible de créer des folds IS valides.")
            return []

        # 3. Segmenter la période IS totale disponible pour créer les points de départ des fenêtres IS expansives
        n_splits = self.wfo_settings.n_splits
        
        # Les `n_splits` folds auront tous `actual_is_total_end_ts` comme date de fin IS.
        # Leurs dates de début IS vont varier.
        # Le fold 0 (le plus court IS) commencera le plus tard.
        # Le fold `n_splits - 1` (le plus long IS) commencera à `actual_is_total_start_ts`.
        
        # Déterminer les timestamps de début pour chaque fold IS.
        # On divise la "partie variable" de la période IS en `n_splits` segments (si n_splits > 1).
        # La partie variable est la durée totale IS disponible moins la durée IS minimale.
        
        duration_variable_part_is = duration_is_total_available - min_is_duration_needed_for_any_fold
        
        if duration_variable_part_is < pd.Timedelta(0): # Ne devrait pas arriver si la validation précédente est passée
            duration_variable_part_is = pd.Timedelta(0)
            logger.warning(f"{self.log_prefix} La partie variable de la durée IS est négative. Tous les folds IS commenceront à la même date (la plus ancienne possible).")

        is_fold_end_ts_common = actual_is_total_end_ts # Fin commune pour tous les IS

        for i in range(n_splits): # i de 0 à n_splits - 1
            # Pour le fold `i`, la fenêtre IS s'étend.
            # Fold 0: IS le plus court (mais >= min_is_period_days)
            # Fold n_splits-1: IS le plus long (couvre toute la période IS disponible)
            
            # Calculer le décalage de début pour ce fold.
            # Si n_splits = 1, start_offset_ratio = 0 (commence au début de IS total disponible)
            # Si n_splits > 1, le premier fold (i=0) commence plus tard.
            start_offset_ratio = (n_splits - 1 - i) / (n_splits -1) if n_splits > 1 else 0.0
            
            # Le début du fold IS est `actual_is_total_start_ts` + une fraction de la partie variable.
            # Le fold le plus long (i = n_splits - 1) a start_offset_ratio = 0.
            # Le fold le plus court (i = 0) a start_offset_ratio = 1 (si n_splits > 1).
            
            current_is_start_ts_calculated = actual_is_total_start_ts + (duration_variable_part_is * start_offset_ratio)
            
            # S'assurer que current_is_start_ts_calculated n'est pas après la fin de la période IS possible
            # (qui serait actual_is_total_end_ts - min_is_duration_needed_for_any_fold)
            max_possible_is_start_for_shortest_fold = actual_is_total_end_ts - min_is_duration_needed_for_any_fold + pd.Timedelta(microseconds=1)
            current_is_start_ts_calculated = min(current_is_start_ts_calculated, max_possible_is_start_for_shortest_fold)
            
            # Trouver le timestamp réel dans les données le plus proche (>=) de ce début calculé
            df_potential_is_starts_for_fold = df_is_total_available.loc[df_is_total_available.index >= current_is_start_ts_calculated]
            if df_potential_is_starts_for_fold.empty:
                # Cela peut arriver si current_is_start_ts_calculated est après la dernière date de df_is_total_available
                # ou si la granularité des données est faible. Prendre le début IS total comme fallback.
                logger.warning(f"{self.log_prefix} Fold {i}: Impossible de trouver un début IS >= {current_is_start_ts_calculated}. "
                               f"Utilisation de {actual_is_total_start_ts} comme début IS pour ce fold.")
                actual_is_fold_start_ts = actual_is_total_start_ts
            else:
                actual_is_fold_start_ts = df_potential_is_starts_for_fold.index.min()

            df_is_fold_current = df_is_total_available.loc[actual_is_fold_start_ts : is_fold_end_ts_common]

            if df_is_fold_current.empty:
                logger.warning(f"{self.log_prefix} Fold {i}: Période IS vide après slicing (de {actual_is_fold_start_ts} à {is_fold_end_ts_common}). Saut.")
                continue
            
            current_is_fold_duration_actual = df_is_fold_current.index.max() - df_is_fold_current.index.min()
            if current_is_fold_duration_actual < min_is_duration_needed_for_any_fold:
                logger.warning(f"{self.log_prefix} Fold {i}: Durée IS actuelle ({current_is_fold_duration_actual}) "
                               f"< min_is_period_days ({self.wfo_settings.min_is_period_days} jours). "
                               "Ce fold pourrait être trop court pour une optimisation robuste.")
                # Optionnel: sauter ce fold si jugé trop court. Pour l'instant, on l'inclut.
            
            folds_generated.append((
                df_is_fold_current.copy(),
                df_oos_fixed_final.copy(),
                i, # fold_id (0-indexed)
                df_is_fold_current.index.min(),
                df_is_fold_current.index.max(),
                actual_oos_fixed_start_ts,
                actual_oos_fixed_end_ts
            ))
            logger.info(f"{self.log_prefix} Fold {i} (expansif) généré. "
                        f"IS: [{df_is_fold_current.index.min()} - {df_is_fold_current.index.max()}], "
                        f"OOS: [{actual_oos_fixed_start_ts} - {actual_oos_fixed_end_ts}]")

        if not folds_generated:
            logger.error(f"{self.log_prefix} Aucun fold valide n'a pu être généré avec la méthode des fenêtres IS expansives et OOS fixe.")
        
        return folds_generated

