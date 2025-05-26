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

    def _validate_inputs(self,
                         df_enriched_data: pd.DataFrame,
                         effective_wfo_start_date: pd.Timestamp,
                         effective_wfo_end_date: pd.Timestamp) -> bool:
        """
        Valide les données d'entrée et les paramètres WFO.
        """
        if df_enriched_data.empty:
            logger.error(f"{self.log_prefix} Le DataFrame de données enrichies est vide.")
            return False
        if not isinstance(df_enriched_data.index, pd.DatetimeIndex):
            logger.error(f"{self.log_prefix} L'index de df_enriched_data doit être un DatetimeIndex.")
            return False
        if df_enriched_data.index.tz is None:
            logger.error(f"{self.log_prefix} L'index de df_enriched_data doit être timezone-aware (UTC attendu).")
            return False

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
        
        logger.debug(f"{self.log_prefix} Validation des entrées réussie.")
        return True

    def generate_folds(self,
                       df_enriched_data: pd.DataFrame,
                       is_total_start_ts_config: Optional[pd.Timestamp] = None,
                       oos_total_end_ts_config: Optional[pd.Timestamp] = None
                       ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Génère les folds IS et OOS selon la configuration.

        Args:
            df_enriched_data (pd.DataFrame): Le DataFrame complet des données enrichies,
                                             avec un DatetimeIndex (UTC attendu).
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

        # 1. Déterminer les dates de début et de fin effectives pour le WFO
        data_start_date = df_enriched_data.index.min()
        data_end_date = df_enriched_data.index.max()

        # Assurer la cohérence des timezones
        if data_start_date.tzinfo is None or data_end_date.tzinfo is None:
             logger.error(f"{self.log_prefix} Les données d'entrée doivent avoir un index DatetimeIndex avec timezone.")
             return [] # Ou lever une exception

        effective_wfo_start_date = is_total_start_ts_config if is_total_start_ts_config else data_start_date
        effective_wfo_end_date = oos_total_end_ts_config if oos_total_end_ts_config else data_end_date

        if effective_wfo_start_date.tzinfo is None: effective_wfo_start_date = effective_wfo_start_date.tz_localize(data_start_date.tzinfo)
        elif effective_wfo_start_date.tzinfo != data_start_date.tzinfo : effective_wfo_start_date = effective_wfo_start_date.tz_convert(data_start_date.tzinfo)
        
        if effective_wfo_end_date.tzinfo is None: effective_wfo_end_date = effective_wfo_end_date.tz_localize(data_end_date.tzinfo)
        elif effective_wfo_end_date.tzinfo != data_end_date.tzinfo : effective_wfo_end_date = effective_wfo_end_date.tz_convert(data_end_date.tzinfo)

        effective_wfo_start_date = max(effective_wfo_start_date, data_start_date)
        effective_wfo_end_date = min(effective_wfo_end_date, data_end_date)
        
        # Slice des données pour la période WFO effective
        df_wfo_period = df_enriched_data.loc[effective_wfo_start_date:effective_wfo_end_date]
        if df_wfo_period.empty:
            logger.error(f"{self.log_prefix} Aucune donnée disponible dans la période WFO effective : "
                         f"{effective_wfo_start_date} à {effective_wfo_end_date}.")
            return []
        
        # Utiliser les dates min/max réelles des données slicées pour la validation
        actual_wfo_data_start = df_wfo_period.index.min()
        actual_wfo_data_end = df_wfo_period.index.max()

        if not self._validate_inputs(df_wfo_period, actual_wfo_data_start, actual_wfo_data_end):
            return []
        
        logger.info(f"{self.log_prefix} Période WFO effective pour la génération des folds : "
                    f"de {actual_wfo_data_start} à {actual_wfo_data_end}.")

        # 2. Logique de découpage spécifique au type de fold
        if self.wfo_settings.fold_type == "expanding":
            return self._generate_expanding_folds_fixed_oos(df_wfo_period)
        else:
            logger.error(f"{self.log_prefix} Type de fold '{self.wfo_settings.fold_type}' non supporté.")
            return []

    def _generate_expanding_folds_fixed_oos(self,
                                            df_wfo_data: pd.DataFrame
                                            ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Génère des folds avec une période OOS fixe à la fin et des périodes IS expansives.
        """
        folds_generated: List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        
        # 1. Définir la période OOS fixe
        oos_duration = pd.Timedelta(days=self.wfo_settings.oos_period_days)
        
        # La fin de OOS est la fin des données WFO disponibles
        oos_end_date_fixed = df_wfo_data.index.max()
        # Le début de OOS est calculé en reculant depuis la fin
        oos_start_date_fixed_calc = oos_end_date_fixed - oos_duration + pd.Timedelta(microseconds=1) # Pour inclure le jour de fin et éviter chevauchement
        
        # Trouver le premier timestamp disponible dans les données >= oos_start_date_fixed_calc
        try:
            actual_oos_start_ts = df_wfo_data.loc[oos_start_date_fixed_calc:].index.min()
        except KeyError: # Si oos_start_date_fixed_calc est après la dernière date de df_wfo_data
             logger.error(f"{self.log_prefix} Impossible de déterminer un début OOS valide. "
                          f"Début OOS calculé ({oos_start_date_fixed_calc}) est potentiellement après la fin des données ({oos_end_date_fixed}).")
             return []


        df_oos_fixed = df_wfo_data.loc[actual_oos_start_ts : oos_end_date_fixed]
        
        if df_oos_fixed.empty:
            logger.error(f"{self.log_prefix} La période OOS fixe (de {actual_oos_start_ts} à {oos_end_date_fixed}) est vide. "
                         "Vérifiez oos_period_days et la plage de données.")
            return []
        
        actual_oos_end_ts = df_oos_fixed.index.max() # La fin réelle des données OOS
        logger.info(f"{self.log_prefix} Période OOS fixe définie : de {actual_oos_start_ts} à {actual_oos_end_ts}.")

        # 2. Définir la période IS totale
        # La fin de la période IS totale est juste avant le début de la période OOS fixe.
        is_total_end_date = actual_oos_start_ts - pd.Timedelta(microseconds=1)
        is_total_start_date = df_wfo_data.index.min() # Le début des données WFO disponibles

        df_is_total = df_wfo_data.loc[is_total_start_date : is_total_end_date]

        if df_is_total.empty:
            logger.error(f"{self.log_prefix} La période IS totale (de {is_total_start_date} à {is_total_end_date}) est vide. "
                         "Cela peut se produire si la période OOS couvre toutes les données disponibles ou plus.")
            return []

        actual_is_total_start_ts = df_is_total.index.min()
        actual_is_total_end_ts = df_is_total.index.max()
        duration_is_total = actual_is_total_end_ts - actual_is_total_start_ts
        
        logger.info(f"{self.log_prefix} Période IS totale définie : de {actual_is_total_start_ts} à {actual_is_total_end_ts} (Durée: {duration_is_total}).")

        min_is_duration_needed = pd.Timedelta(days=self.wfo_settings.min_is_period_days)
        if duration_is_total < min_is_duration_needed:
            logger.error(f"{self.log_prefix} La durée IS totale ({duration_is_total}) "
                         f"est inférieure à min_is_period_days ({self.wfo_settings.min_is_period_days} jours).")
            return []

        # 3. Segmenter la période IS totale et construire les folds IS expansifs
        n_splits = self.wfo_settings.n_splits
        
        # Les points d'ancrage temporels pour le début des périodes IS expansives.
        # Nous avons besoin de `n_splits` points de départ pour les périodes IS.
        # Le premier fold (le plus court) utilise le dernier segment.
        # Le dernier fold (le plus long) utilise tous les segments (commence à actual_is_total_start_ts).
        
        is_total_indices = df_is_total.index
        num_is_points = len(is_total_indices)

        if num_is_points < n_splits : # Pas assez de points pour avoir des segments distincts pour chaque début de fold
            logger.warning(f"{self.log_prefix} Moins de points de données ({num_is_points}) dans IS total que n_splits ({n_splits}). "
                           "Certains folds IS pourraient être identiques ou très courts.")
            # Ajuster n_splits ou gérer comme une erreur selon la robustesse souhaitée.
            # Pour l'instant, on continue, mais cela peut donner des résultats non optimaux.

        # Générer n_splits points de départ pour les périodes IS, répartis dans le temps.
        # Le dernier point de départ est actual_is_total_start_ts (pour le fold le plus long).
        # Le premier point de départ (pour le fold le plus court) sera plus proche de actual_is_total_end_ts.
        
        # Indices des points de départ des segments dans is_total_indices
        # Linspace génère n_splits+1 points pour définir n_splits segments.
        # Nous voulons n_splits points de *départ* pour nos fenêtres IS.
        # Le k-ième fold (0-indexed) commencera au k-ième point de départ (en partant du plus ancien).
        
        # Si n_splits = 1, le seul IS est [actual_is_total_start_ts, actual_is_total_end_ts]
        # Si n_splits = 3:
        #   Fold 0 (plus court): [T2, T_N]
        #   Fold 1: [T1, T_N]
        #   Fold 2 (plus long): [T0, T_N]
        # où T0=actual_is_total_start_ts, T_N=actual_is_total_end_ts
        # T1, T2 sont des points intermédiaires.
        
        # Les indices des points de début des `n_splits` fenêtres IS.
        # Ces indices sont dans `is_total_indices`.
        start_indices_for_is_folds = np.linspace(0, num_is_points - 1, n_splits, endpoint=False, dtype=int)
        # Correction: linspace pour les points de *début* des segments qui composent les folds.
        # Si on a N points et S splits, on peut avoir S points de début.
        # Le premier fold utilise le dernier (S-1)ème segment.
        # Le dernier fold utilise tous les segments, commençant au 0ème segment.

        # Pour `n_splits` folds, nous avons besoin de `n_splits` dates de début de période IS.
        # La fin de toutes les périodes IS est `actual_is_total_end_ts`.
        # Les dates de début sont réparties entre `actual_is_total_start_ts` et une date
        # telle que le fold le plus court respecte `min_is_period_days`.

        # Créons les `n_splits` dates de début.
        # La date de fin de toutes les périodes IS est `actual_is_total_end_ts`.
        is_fold_end_ts = actual_is_total_end_ts

        for i in range(n_splits):
            # Pour le fold `i` (0-indexed), le début de la période IS est déterminé.
            # Le fold 0 est le plus court, le fold `n_splits-1` est le plus long.
            # On divise la période IS totale en `n_splits` "blocs" de temps.
            # Le fold `i` commencera au début du `(n_splits - 1 - i)`-ième bloc.
            
            # Calculer le point de départ du segment le plus ancien pour ce fold
            # Exemple: 3 splits.
            # i=0 (Fold 0, plus court): commence au segment 2/3 de la durée IS totale.
            # i=1 (Fold 1): commence au segment 1/3 de la durée IS totale.
            # i=2 (Fold 2, plus long): commence au segment 0/3 (début IS totale).
            
            start_fraction = (n_splits - 1 - i) / n_splits
            current_is_start_offset_days = duration_is_total.total_seconds() * start_fraction / (24 * 60 * 60)
            
            # Le timestamp de début approximatif pour ce fold IS
            is_fold_start_ts_approx = actual_is_total_start_ts + pd.Timedelta(days=current_is_start_offset_days)
            
            # Trouver le timestamp réel dans les données le plus proche (>=) de ce début approximatif
            try:
                actual_is_fold_start_ts = df_is_total.loc[is_fold_start_ts_approx:].index.min()
            except KeyError: # Si approx est après la dernière date de df_is_total (ne devrait pas arriver)
                actual_is_fold_start_ts = actual_is_total_start_ts # Fallback

            df_is_fold = df_is_total.loc[actual_is_fold_start_ts : is_fold_end_ts]

            if df_is_fold.empty:
                logger.warning(f"{self.log_prefix} Fold {i}: Période IS vide (de {actual_is_fold_start_ts} à {is_fold_end_ts}). Saut.")
                continue
            
            current_is_fold_duration = df_is_fold.index.max() - df_is_fold.index.min()
            if current_is_fold_duration < min_is_duration_needed:
                logger.warning(f"{self.log_prefix} Fold {i}: Durée IS ({current_is_fold_duration}) "
                               f"< min_is_period_days ({self.wfo_settings.min_is_period_days} jours). "
                               "Ce fold pourrait être trop court pour une optimisation significative.")
                # On pourrait choisir de sauter ce fold si trop court, ou juste loguer un avertissement.
                # Pour l'instant, on continue.

            folds_generated.append((
                df_is_fold.copy(),
                df_oos_fixed.copy(),
                i, # fold_id
                df_is_fold.index.min(), # actual_is_fold_start_ts
                df_is_fold.index.max(), # is_fold_end_ts
                actual_oos_start_ts,
                actual_oos_end_ts
            ))
            logger.info(f"{self.log_prefix} Fold {i} (expansif) généré. "
                        f"IS: [{df_is_fold.index.min()} - {df_is_fold.index.max()}], "
                        f"OOS: [{actual_oos_start_ts} - {actual_oos_end_ts}]")

        if not folds_generated:
            logger.error(f"{self.log_prefix} Aucun fold valide n'a pu être généré avec la méthode des fenêtres expansives et OOS fixe.")
        
        return folds_generated

