# src/backtesting/wfo/fold_generator.py
"""
Ce module est responsable de la génération des découpages (folds)
In-Sample (IS) et Out-of-Sample (OOS) à partir d'un DataFrame de données
enrichies, en respectant la méthodologie Walk-Forward Optimization (WFO) configurée.
Il implémente plusieurs logiques de découpage, y compris expansif, adaptatif,
et combinatoire, ainsi que des outils d'analyse de stabilité des folds.
"""
import logging
from typing import List, Tuple, Optional, TYPE_CHECKING, Dict, Any, Callable
from dataclasses import dataclass, field
import itertools
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats # Pour les tests statistiques (KS, Jarque-Bera)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Tentative d'importation de ruptures, optionnel
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    rpt = None # type: ignore
    RUPTURES_AVAILABLE = False
    logging.getLogger(__name__).info("La bibliothèque 'ruptures' n'est pas installée. "
                                     "La fonction 'optimize_fold_boundaries' sera désactivée.")

if TYPE_CHECKING:
    from src.config.definitions import WfoSettings

logger = logging.getLogger(__name__)

@dataclass
class FoldAnalysisResult:
    """Rapport d'analyse de stabilité pour un fold."""
    fold_id: int
    is_period_str: str
    oos_period_str: str
    # Test de Kolmogorov-Smirnov pour la similarité des distributions (ex: sur les rendements)
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    ks_test_passed_alpha_0_05: Optional[bool] = None # True si p_value > 0.05
    # Test de Jarque-Bera pour la normalité
    is_jarque_bera_stat: Optional[float] = None
    is_jarque_bera_p_value: Optional[float] = None
    is_data_normal_alpha_0_05: Optional[bool] = None # True si p_value > 0.05
    oos_jarque_bera_stat: Optional[float] = None
    oos_jarque_bera_p_value: Optional[float] = None
    oos_data_normal_alpha_0_05: Optional[bool] = None
    warnings: List[str] = field(default_factory=list)


class WfoFoldGenerator:
    """
    Génère les folds In-Sample (IS) et Out-of-Sample (OOS) pour le
    Walk-Forward Optimization, avec support pour différentes stratégies de découpage
    et analyses de stabilité.
    """

    def __init__(self, wfo_settings: 'WfoSettings'):
        """
        Initialise le générateur de folds.

        Args:
            wfo_settings (WfoSettings): Les paramètres de configuration pour le WFO.
                S'attend à ce que WfoSettings contienne des champs comme:
                n_splits, oos_period_days, min_is_period_days, fold_type,
                overlap_ratio_is_oos, purging_period_days, embargo_period_days,
                adaptive_volatility_window, adaptive_target_volatility_quantile,
                combinatorial_n_combinations, combinatorial_is_duration_days,
                combinatorial_oos_duration_days, change_point_model, etc.
        """
        self.wfo_settings = wfo_settings
        self.log_prefix = "[WfoFoldGeneratorV2]"
        logger.info(f"{self.log_prefix} Initialisé avec les paramètres WFO : {wfo_settings}")
        if not hasattr(wfo_settings, 'fold_type'):
             logger.warning(f"{self.log_prefix} WfoSettings ne semble pas avoir 'fold_type'. "
                            "Utilisation de 'expanding' par défaut.")
        # Initialiser les attributs optionnels de WfoSettings avec des valeurs par défaut si manquants
        self.overlap_ratio = getattr(wfo_settings, 'overlap_ratio_is_oos', 0.0)
        self.purging_days = getattr(wfo_settings, 'purging_period_days', 0)
        self.embargo_days = getattr(wfo_settings, 'embargo_period_days', 0)

    def _validate_data_and_settings(self,
                                    df_enriched_data: pd.DataFrame,
                                    effective_wfo_start_date: pd.Timestamp,
                                    effective_wfo_end_date: pd.Timestamp) -> bool:
        """Valide les données d'entrée et les paramètres WFO pour la génération des folds."""
        # ... (validation existante) ...
        if df_enriched_data.empty:
            logger.error(f"{self.log_prefix} Le DataFrame de données enrichies est vide.")
            return False
        if not isinstance(df_enriched_data.index, pd.DatetimeIndex):
            logger.error(f"{self.log_prefix} L'index de df_enriched_data doit être un DatetimeIndex.")
            return False
        if df_enriched_data.index.tzinfo is None or df_enriched_data.index.tzinfo.utcoffset(None) != pd.Timestamp(0, tz='UTC').tzinfo.utcoffset(None): # type: ignore
            logger.error(f"{self.log_prefix} L'index de df_enriched_data doit être timezone-aware et en UTC. Actuel: {df_enriched_data.index.tzinfo}")
            return False
        if not df_enriched_data.index.is_monotonic_increasing:
            logger.error(f"{self.log_prefix} L'index de df_enriched_data doit être trié de manière croissante.")
            return False

        if effective_wfo_start_date >= effective_wfo_end_date:
            logger.error(f"{self.log_prefix} Date de début WFO ({effective_wfo_start_date}) >= date de fin WFO ({effective_wfo_end_date}).")
            return False
        
        min_total_duration_needed = pd.Timedelta(days=self.wfo_settings.oos_period_days + self.wfo_settings.min_is_period_days)
        actual_total_duration = effective_wfo_end_date - effective_wfo_start_date
        if actual_total_duration < min_total_duration_needed:
            logger.error(f"{self.log_prefix} Durée totale ({actual_total_duration}) insuffisante pour OOS ({self.wfo_settings.oos_period_days}j) et IS min ({self.wfo_settings.min_is_period_days}j).")
            return False
        
        logger.debug(f"{self.log_prefix} Validation des entrées et des paramètres WFO réussie.")
        return True

    def generate_folds(self,
                       df_enriched_data: pd.DataFrame,
                       is_total_start_ts_config: Optional[pd.Timestamp] = None,
                       oos_total_end_ts_config: Optional[pd.Timestamp] = None
                       ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Génère les folds IS et OOS selon la configuration."""
        fold_type_config = getattr(self.wfo_settings, 'fold_type', 'expanding').lower()
        logger.info(f"{self.log_prefix} Démarrage de la génération des folds WFO de type '{fold_type_config}'.")

        data_min_ts = df_enriched_data.index.min()
        data_max_ts = df_enriched_data.index.max()

        def _ensure_utc(ts: Optional[pd.Timestamp], default_tz_ref: pd.Timestamp) -> Optional[pd.Timestamp]:
            if ts is None: return None
            if ts.tzinfo is None: return ts.tz_localize(default_tz_ref.tzinfo) # type: ignore
            if ts.tzinfo != default_tz_ref.tzinfo : return ts.tz_convert(default_tz_ref.tzinfo) # type: ignore
            return ts

        effective_wfo_start_date = _ensure_utc(is_total_start_ts_config, data_min_ts) or data_min_ts
        effective_wfo_end_date = _ensure_utc(oos_total_end_ts_config, data_max_ts) or data_max_ts
        
        effective_wfo_start_date = max(effective_wfo_start_date, data_min_ts)
        effective_wfo_end_date = min(effective_wfo_end_date, data_max_ts)
        
        df_wfo_period_data = df_enriched_data.loc[effective_wfo_start_date:effective_wfo_end_date]
        if df_wfo_period_data.empty:
            logger.error(f"{self.log_prefix} Aucune donnée disponible dans la période WFO effective: [{effective_wfo_start_date} - {effective_wfo_end_date}].")
            return []
        
        actual_wfo_data_start_ts = df_wfo_period_data.index.min()
        actual_wfo_data_end_ts = df_wfo_period_data.index.max()

        if not self._validate_data_and_settings(df_wfo_period_data, actual_wfo_data_start_ts, actual_wfo_data_end_ts):
            return []
        
        logger.info(f"{self.log_prefix} Période WFO effective pour génération: {actual_wfo_data_start_ts} à {actual_wfo_data_end_ts}.")

        if fold_type_config == "expanding":
            return self._generate_expanding_is_fixed_oos_folds(df_wfo_period_data)
        elif fold_type_config == "adaptive":
            vol_window = getattr(self.wfo_settings, 'adaptive_volatility_window', 20)
            return self.generate_adaptive_folds(df_wfo_period_data, volatility_window=vol_window)
        elif fold_type_config == "combinatorial":
            n_combi = getattr(self.wfo_settings, 'combinatorial_n_combinations', 10)
            is_days = getattr(self.wfo_settings, 'combinatorial_is_duration_days', 90)
            oos_days = getattr(self.wfo_settings, 'combinatorial_oos_duration_days', 30)
            return self.generate_combinatorial_folds(df_wfo_period_data, n_combinations=n_combi, is_duration_days=is_days, oos_duration_days=oos_days)
        else:
            logger.error(f"{self.log_prefix} Type de fold '{fold_type_config}' non supporté.")
            return []

    def _generate_expanding_is_fixed_oos_folds(self, df_wfo_data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Génère des folds avec fenêtres IS expansives et OOS fixe, avec purging/embargo/overlap."""
        # ... (logique existante de _generate_expanding_is_fixed_oos_folds) ...
        # MODIFIÉE pour inclure purging, embargo, overlap
        folds_generated: List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        
        oos_duration_td = pd.Timedelta(days=self.wfo_settings.oos_period_days)
        min_is_duration_td = pd.Timedelta(days=self.wfo_settings.min_is_period_days)
        purging_td = pd.Timedelta(days=self.purging_days)
        embargo_td = pd.Timedelta(days=self.embargo_days)
        
        n_splits = self.wfo_settings.n_splits
        total_available_data_points = len(df_wfo_data)

        # La période OOS est toujours à la fin de la période de données pour ce fold.
        # La période IS s'étend jusqu'au début de la période d'embargo (si > 0) avant OOS.
        
        # Calculer la longueur totale de la "partie non IS minimale" pour chaque split
        # (OOS + Embargo + Purge)
        non_is_min_block_duration = oos_duration_td + embargo_td + purging_td
        
        # Durée totale disponible pour IS + OOS + gaps
        total_duration_available = df_wfo_data.index.max() - df_wfo_data.index.min()
        
        # Durée disponible pour la partie IS "variable" (au-delà du minimum IS et du bloc non-IS)
        available_for_is_expansion = total_duration_available - non_is_min_block_duration - min_is_duration_td
        
        if available_for_is_expansion < pd.Timedelta(0):
            logger.error(f"{self.log_prefix} Pas assez de données pour la configuration WFO avec purging/embargo. "
                         f"Besoin: {min_is_duration_td + non_is_min_block_duration}, Disponible: {total_duration_available}")
            return []

        # Taille de chaque segment d'expansion IS
        is_expansion_segment_duration = available_for_is_expansion / n_splits if n_splits > 0 else pd.Timedelta(0)

        current_is_start_ts = df_wfo_data.index.min()

        for i in range(n_splits):
            fold_id = i
            
            # Fin de la période IS pour ce fold
            # Pour le premier fold (i=0), IS est min_is_duration_td + 0 * segment
            # Pour le dernier fold (i=n_splits-1), IS est min_is_duration_td + (n_splits-1)*segment (presque toute la partie d'expansion)
            # Non, c'est l'inverse pour IS expansif: le premier IS est le plus court.
            # Le début IS avance pour les premiers folds, puis reste fixe pour les suivants qui s'étendent.
            # Avec IS expansif, la fin IS avance.
            
            # IS_end = Data_Start + Min_IS_Days + (i * OOS_step_size_if_rolling) - Purge
            # OOS_start = IS_end + Purge + Embargo
            # OOS_end = OOS_start + OOS_Days
            
            # Avec IS expansif et OOS fixe à la fin:
            # OOS_end_fold = df_wfo_data.index.max()
            # OOS_start_fold = OOS_end_fold - oos_duration_td + pd.Timedelta(microseconds=1) (pour être inclusif)
            # IS_end_fold = OOS_start_fold - embargo_td - pd.Timedelta(microseconds=1)
            # IS_start_fold = df_wfo_data.index.min() # Pour le premier fold, il s'étend
            # Pour les folds suivants, le IS_end avance.
            # C'est plus comme un "Anchored Walk Forward" ou "Rolling Origin Recalibration"
            # où l'origine de l'optimisation (IS_end) avance.

            # Si on veut des fenêtres IS expansives, la fin de IS avance à chaque fold.
            # La période OOS suit immédiatement (après purge/embargo).
            
            is_end_fold_target = df_wfo_data.index.min() + min_is_duration_td + (i * oos_duration_td * (1.0 - self.overlap_ratio)) - purging_td
            # S'assurer que is_end_fold_target est un index valide
            is_end_fold_ts = df_wfo_data.index[df_wfo_data.index.get_indexer([is_end_fold_target], method='bfill')[0]]


            oos_start_fold_target = is_end_fold_ts + purging_td + embargo_td + pd.Timedelta(microseconds=1)
            oos_start_fold_ts = df_wfo_data.index[df_wfo_data.index.get_indexer([oos_start_fold_target], method='bfill')[0]]
            
            oos_end_fold_target = oos_start_fold_ts + oos_duration_td - pd.Timedelta(microseconds=1)
            oos_end_fold_ts = df_wfo_data.index[df_wfo_data.index.get_indexer([oos_end_fold_target], method='ffill')[0]]

            is_start_fold_ts = df_wfo_data.index.min() # IS expansif

            if is_start_fold_ts >= is_end_fold_ts or oos_start_fold_ts >= oos_end_fold_ts or is_end_fold_ts >= oos_start_fold_ts:
                logger.warning(f"{self.log_prefix} Fold {fold_id}: Périodes invalides ou se chevauchant après calculs. Saut. "
                               f"IS: {is_start_fold_ts}-{is_end_fold_ts}, OOS: {oos_start_fold_ts}-{oos_end_fold_ts}")
                continue

            df_is_current = df_wfo_data.loc[is_start_fold_ts : is_end_fold_ts].copy()
            df_oos_current = df_wfo_data.loc[oos_start_fold_ts : oos_end_fold_ts].copy()

            if df_is_current.empty or df_oos_current.empty:
                logger.warning(f"{self.log_prefix} Fold {fold_id}: IS ou OOS vide après slicing. Saut.")
                continue
            
            # Vérifier la durée minimale IS
            if (df_is_current.index.max() - df_is_current.index.min()) < min_is_duration_td:
                 logger.warning(f"{self.log_prefix} Fold {fold_id}: Durée IS ({df_is_current.index.max() - df_is_current.index.min()}) "
                                f"< min_is_period_days ({min_is_duration_td}). Saut.")
                 continue


            folds_generated.append((
                df_is_current, df_oos_current, fold_id,
                df_is_current.index.min(), df_is_current.index.max(),
                df_oos_current.index.min(), df_oos_current.index.max()
            ))
            logger.info(f"{self.log_prefix} Fold {fold_id} (expansif) généré. "
                        f"IS: [{df_is_current.index.min()} - {df_is_current.index.max()}], "
                        f"OOS: [{df_oos_current.index.min()} - {df_oos_current.index.max()}]")

        return folds_generated


    def generate_adaptive_folds(
        self,
        df_enriched_data: pd.DataFrame,
        volatility_column: str = 'close', # Colonne pour calculer la volatilité
        volatility_window: int = 20,      # Fenêtre pour calculer la volatilité (ex: ATR ou std dev des rendements)
        target_volatility_quantile: float = 0.75, # Viser des périodes OOS dont la volatilité est autour de ce quantile
        min_fold_duration_days: Optional[int] = None, # Durée minimale pour un fold IS ou OOS
        max_fold_duration_days: Optional[int] = None  # Durée maximale
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Génère des folds en ajustant leur taille ou leur position en fonction de la volatilité du marché.
        Concept: Tenter de rendre les périodes OOS plus homogènes en termes de volatilité,
        ou s'assurer que les périodes IS capturent différents régimes de volatilité.
        Cette implémentation est un exemple conceptuel.
        """
        logger.info(f"{self.log_prefix} Génération de folds adaptatifs (basé sur volatilité de '{volatility_column}')...")
        if not RUPTURES_AVAILABLE: # Ruptures pourrait être utile ici aussi
            logger.warning(f"{self.log_prefix} La bibliothèque 'ruptures' n'est pas disponible. "
                           "Les folds adaptatifs basés sur la détection de changement de régime de volatilité seront limités.")

        if volatility_column not in df_enriched_data.columns:
            logger.error(f"{self.log_prefix} Colonne de volatilité '{volatility_column}' non trouvée. Impossible de générer des folds adaptatifs.")
            return []
        
        # 1. Calculer la volatilité (ex: rolling std dev des rendements journaliers)
        # S'assurer des rendements journaliers pour une mesure de volatilité standard
        if 'close' in df_enriched_data.columns:
            daily_returns = df_enriched_data['close'].resample('D').last().pct_change().dropna()
            rolling_volatility = daily_returns.rolling(window=volatility_window).std().dropna()
        else:
            logger.error(f"{self.log_prefix} Colonne 'close' nécessaire pour calculer les rendements pour la volatilité adaptative.")
            return []

        if rolling_volatility.empty:
            logger.warning(f"{self.log_prefix} Calcul de la volatilité roulante n'a produit aucune donnée.")
            return []

        # 2. Segmenter basé sur la volatilité (exemple: identifier des points de changement)
        # Ou, plus simplement, ajuster la longueur OOS pour essayer de capturer une "quantité" similaire de volatilité.
        # Cette partie est complexe et nécessite une stratégie de segmentation claire.
        # Pour cet exemple, nous allons simplifier et ne pas implémenter une logique adaptative complexe,
        # mais plutôt montrer la structure.
        logger.warning(f"{self.log_prefix} La logique de génération de folds adaptatifs est conceptuelle et non pleinement implémentée. "
                       "Retour des folds expansifs standards pour l'instant.")
        # Fallback sur la méthode standard si la logique adaptative n'est pas finie
        return self._generate_expanding_is_fixed_oos_folds(df_enriched_data)


    def generate_combinatorial_folds(
        self,
        df_enriched_data: pd.DataFrame,
        n_combinations: int = 10,
        is_duration_days: int = 90,
        oos_duration_days: int = 30,
        min_gap_days: int = 5,
        random_seed: Optional[int] = None
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        logger.info(f"{self.log_prefix} Génération de {n_combinations} folds combinatoires...")
        if not self._can_generate_combinatorial_folds(df_enriched_data, is_duration_days, oos_duration_days, min_gap_days):
            return []

        rng = np.random.default_rng(random_seed)
        folds_generated: List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        attempts = 0
        # Increase max_attempts to give more chances to find valid folds, especially if date range is tight or sparse.
        max_attempts = n_combinations * 20 

        while len(folds_generated) < n_combinations and attempts < max_attempts:
            attempts += 1
            fold_data = self._try_generate_single_combinatorial_fold(
                df_enriched_data, is_duration_days, oos_duration_days, min_gap_days, rng
            )
            if fold_data:
                df_is, df_oos, is_start_ts, is_end_ts, oos_start_ts, oos_end_ts = fold_data
                fold_id = len(folds_generated)
                folds_generated.append((df_is, df_oos, fold_id, is_start_ts, is_end_ts, oos_start_ts, oos_end_ts))
                logger.debug(f"{self.log_prefix} Fold combinatoire {fold_id} généré. IS: {is_start_ts}-{is_end_ts}, OOS: {oos_start_ts}-{oos_end_ts}")

        if len(folds_generated) < n_combinations:
            logger.warning(f"{self.log_prefix} Seulement {len(folds_generated)}/{n_combinations} folds combinatoires ont pu être générés après {attempts} tentatives.")
        return folds_generated

    def _can_generate_combinatorial_folds(self, df_data: pd.DataFrame, is_days: int, oos_days: int, gap_days: int) -> bool:
        """Vérifie si les données sont suffisantes pour générer des folds combinatoires."""
        if df_data.empty:
            logger.warning(f"{self.log_prefix} DataFrame vide, impossible de générer des folds combinatoires.")
            return False

        unique_days_available = len(df_data.index.normalize().unique())
        min_required_unique_days = is_days + oos_days + gap_days
        
        if unique_days_available < min_required_unique_days:
            logger.warning(f"{self.log_prefix} Pas assez de jours uniques ({unique_days_available}) pour générer des folds combinatoires "
                           f"avec IS={is_days}j, OOS={oos_days}j, Gap={gap_days}j (besoin: {min_required_unique_days}j).")
            return False
        
        total_data_span_days = (df_data.index.max() - df_data.index.min()).days
        if total_data_span_days < min_required_unique_days: # Should be caught by unique_days_available in most cases, but good as a sanity check.
            logger.warning(f"{self.log_prefix} Étendue totale des données ({total_data_span_days} jours) insuffisante pour générer "
                           f"des folds combinatoires (besoin: {min_required_unique_days} jours).")
            return False
        return True

    def _try_generate_single_combinatorial_fold(
        self, df_data: pd.DataFrame, is_duration_days: int, oos_duration_days: int, min_gap_days: int, rng: np.random.Generator
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Tente de générer un seul fold combinatoire valide."""
        date_index = df_data.index # Assumed to be sorted DatetimeIndex UTC
        min_date, max_date = date_index.min(), date_index.max()

        is_duration_td = pd.Timedelta(days=is_duration_days)
        oos_duration_td = pd.Timedelta(days=oos_duration_days)
        min_gap_td = pd.Timedelta(days=min_gap_days)

        # Determine valid range for IS start_ts
        # IS must end early enough to allow for gap + OOS duration
        max_allowable_is_end_for_selection = max_date - oos_duration_td - min_gap_td
        max_is_start_date = max_allowable_is_end_for_selection - is_duration_td
        
        if min_date > max_is_start_date:
            # This case should ideally be caught by _can_generate_combinatorial_folds
            logger.debug(f"{self.log_prefix} Plage de dates trop courte pour sélectionner un début IS aléatoire valide.")
            return None

        possible_is_starts = date_index[date_index <= max_is_start_date]
        if possible_is_starts.empty: 
            logger.debug(f"{self.log_prefix} Aucun début IS possible trouvé pour la plage max_is_start_date: {max_is_start_date}")
            return None
        
        is_start_ts = pd.Timestamp(rng.choice(possible_is_starts.to_numpy()), tz='UTC')
        is_end_ts_target = is_start_ts + is_duration_td
        
        # Find actual end date in index (<= target)
        is_end_ts_candidates = date_index[(date_index >= is_start_ts) & (date_index <= is_end_ts_target)]
        if is_end_ts_candidates.empty or (is_end_ts_candidates.max() - is_start_ts) < pd.Timedelta(days=is_duration_days * 0.8): # Ensure IS is reasonably long
            return None # Not enough data for IS period
        is_end_ts = is_end_ts_candidates.max()


        # Determine valid range for OOS start_ts
        min_oos_start_target = is_end_ts + min_gap_td
        max_allowable_oos_end_for_selection = max_date
        max_oos_start_date = max_allowable_oos_end_for_selection - oos_duration_td
        
        if min_oos_start_target > max_oos_start_date:
            return None # Not enough space for OOS after IS and gap

        possible_oos_starts = date_index[(date_index >= min_oos_start_target) & (date_index <= max_oos_start_date)]
        if possible_oos_starts.empty: return None

        oos_start_ts = pd.Timestamp(rng.choice(possible_oos_starts.to_numpy()), tz='UTC')
        oos_end_ts_target = oos_start_ts + oos_duration_td
        
        oos_end_ts_candidates = date_index[(date_index >= oos_start_ts) & (date_index <= oos_end_ts_target)]
        if oos_end_ts_candidates.empty or (oos_end_ts_candidates.max() - oos_start_ts) < pd.Timedelta(days=oos_duration_days * 0.8): # Ensure OOS is reasonably long
             return None # Not enough data for OOS period
        oos_end_ts = oos_end_ts_candidates.max()
        
        # Final check for validity (though logic above should mostly prevent these)
        if is_start_ts >= is_end_ts or oos_start_ts >= oos_end_ts or (is_end_ts + min_gap_td) > oos_start_ts :
            return None

        df_is = df_data.loc[is_start_ts:is_end_ts].copy()
        df_oos = df_data.loc[oos_start_ts:oos_end_ts].copy()

        # Ensure periods are not empty after slicing (can happen with sparse data)
        if df_is.empty or df_oos.empty:
            return None
            
        return df_is, df_oos, is_start_ts, is_end_ts, oos_start_ts, oos_end_ts

    def analyze_fold_stability(
        self,
        folds: List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
        target_column: str = 'close', # Colonne à utiliser pour l'analyse de distribution (ex: rendements dérivés de 'close')
        alpha_level: float = 0.05
    ) -> List[FoldAnalysisResult]:
        """
        Analyse la stabilité statistique des folds générés.
        Compare les distributions IS/OOS avec KS-test et teste la normalité avec Jarque-Bera.
        """
        logger.info(f"{self.log_prefix} Analyse de la stabilité pour {len(folds)} folds (colonne cible: {target_column}).")
        analysis_results: List[FoldAnalysisResult] = []

        for df_is, df_oos, fold_id, is_start, is_end, oos_start, oos_end in folds:
            result = FoldAnalysisResult(
                fold_id=fold_id,
                is_period_str=f"{is_start.date()} to {is_end.date()}",
                oos_period_str=f"{oos_start.date()} to {oos_end.date()}"
            )

            if target_column not in df_is.columns or target_column not in df_oos.columns:
                result.warnings.append(f"Colonne cible '{target_column}' non trouvée dans IS ou OOS.")
                analysis_results.append(result)
                continue
            
            # Utiliser les rendements journaliers pour les tests de distribution
            is_returns = df_is[target_column].pct_change().dropna()
            oos_returns = df_oos[target_column].pct_change().dropna()

            if len(is_returns) < 20 or len(oos_returns) < 20: # Seuil arbitraire pour la significativité des tests
                result.warnings.append("Pas assez de points de données de rendement (<20) dans IS ou OOS pour des tests statistiques robustes.")
            else:
                # Test de Kolmogorov-Smirnov (similarité des distributions)
                try:
                    ks_stat, ks_p_value = scipy_stats.ks_2samp(is_returns, oos_returns)
                    result.ks_statistic = ks_stat
                    result.ks_p_value = ks_p_value
                    result.ks_test_passed_alpha_0_05 = ks_p_value > alpha_level
                    if ks_p_value <= alpha_level:
                         result.warnings.append(f"KS-Test: Distributions IS/OOS significativement différentes (p={ks_p_value:.3f}).")
                except Exception as e_ks:
                    result.warnings.append(f"Erreur KS-Test: {e_ks}")

                # Test de Jarque-Bera (normalité)
                try:
                    jb_is_stat, jb_is_p = scipy_stats.jarque_bera(is_returns)
                    result.is_jarque_bera_stat = jb_is_stat
                    result.is_jarque_bera_p_value = jb_is_p
                    result.is_data_normal_alpha_0_05 = jb_is_p > alpha_level
                    if jb_is_p <= alpha_level:
                        result.warnings.append(f"Jarque-Bera IS: Non normal (p={jb_is_p:.3f}).")

                    jb_oos_stat, jb_oos_p = scipy_stats.jarque_bera(oos_returns)
                    result.oos_jarque_bera_stat = jb_oos_stat
                    result.oos_jarque_bera_p_value = jb_oos_p
                    result.oos_data_normal_alpha_0_05 = jb_oos_p > alpha_level
                    if jb_oos_p <= alpha_level:
                        result.warnings.append(f"Jarque-Bera OOS: Non normal (p={jb_oos_p:.3f}).")
                except Exception as e_jb:
                    result.warnings.append(f"Erreur Jarque-Bera Test: {e_jb}")
            
            analysis_results.append(result)
        return analysis_results

    def optimize_fold_boundaries(
        self,
        df_enriched_data: pd.DataFrame,
        series_to_segment: Optional[pd.Series] = None, # Série à utiliser pour la détection (ex: volatilité, prix)
        n_bkps_to_find: Optional[int] = None, # Nombre de points de rupture à trouver
        model: str = "l2", # Modèle de coût pour ruptures: "l1", "l2", "rbf", etc.
        penalty_value: Optional[float] = None # Pour Pelt, sinon None pour auto
    ) -> List[pd.Timestamp]:
        """
        Tente d'optimiser les frontières des folds en utilisant la détection de points de changement.
        Retourne une liste de timestamps suggérés comme frontières de folds.
        """
        logger.info(f"{self.log_prefix} Optimisation des frontières de folds (modèle: {model})...")
        if not self._check_ruptures_availability(): return []
        
        series_for_segmentation = self._prepare_series_for_segmentation(df_enriched_data, series_to_segment)
        if series_for_segmentation is None or series_for_segmentation.empty:
             # Erreur déjà loguée dans la méthode helper si series_for_segmentation est None
            if series_for_segmentation is not None and series_for_segmentation.empty : # Log specific empty case
                 logger.warning(f"{self.log_prefix} Série à segmenter est vide après préparation.")
            return []
        
        # Default n_bkps_to_find if not provided, based on wfo_settings.n_splits
        effective_n_bkps = n_bkps_to_find if n_bkps_to_find is not None else getattr(self.wfo_settings, 'n_splits', 1) -1
        if effective_n_bkps < 0: effective_n_bkps = 0 # Ensure non-negative

        min_required_points = effective_n_bkps * 2 + 1 # Need at least one point per segment +1 for start/end
        if len(series_for_segmentation) < min_required_points:
            logger.warning(f"{self.log_prefix} Série à segmenter trop courte ({len(series_for_segmentation)} points) pour {effective_n_bkps} points de rupture (besoin: {min_required_points}).")
            return []

        points_to_fit = series_for_segmentation.to_numpy().reshape(-1, 1)
        
        bkps_indices = self._run_change_point_algo(points_to_fit, model, penalty_value, effective_n_bkps)
        if bkps_indices is None: # Erreur ou algo non applicable
            return []

        return self._process_breakpoint_indices(bkps_indices, series_for_segmentation)

    def _check_ruptures_availability(self) -> bool:
        """Vérifie si la bibliothèque 'ruptures' est disponible."""
        if not RUPTURES_AVAILABLE:
            logger.error(f"{self.log_prefix} Bibliothèque 'ruptures' non disponible. Impossible d'optimiser les frontières.")
            return False
        return True

    def _prepare_series_for_segmentation(self, df_data: pd.DataFrame, series_input: Optional[pd.Series]) -> Optional[pd.Series]:
        """Prépare la série à utiliser pour la segmentation (volatilité par défaut ou fournie)."""
        if df_data.empty and series_input is None: # df_data can be empty if series_input is provided
            logger.warning(f"{self.log_prefix} DataFrame vide et aucune série fournie. Impossible de préparer la série pour segmentation.")
            return None

        if series_input is not None:
            if series_input.empty:
                logger.warning(f"{self.log_prefix} La série fournie pour segmentation est vide.")
                return None
            return series_input.dropna()

        # Calculer la volatilité par défaut si aucune série n'est fournie et df_data est disponible
        if 'close' in df_data.columns:
            volatility_window = getattr(self.wfo_settings, 'adaptive_volatility_window', 20)
            if volatility_window <= 0:
                logger.warning(f"{self.log_prefix} Fenêtre de volatilité adaptative ({volatility_window}) invalide. Doit être > 0.")
                return None
            series = df_data['close'].pct_change().rolling(window=volatility_window).std().dropna()
            if series.empty:
                logger.warning(f"{self.log_prefix} Calcul de la volatilité par défaut n'a produit aucune donnée.")
                return None
            return series
        else:
            logger.error(f"{self.log_prefix} Colonne 'close' non trouvée dans df_data pour calculer la série par défaut pour la segmentation.")
            return None

    def _run_change_point_algo(
        self, points: np.ndarray, model: str, penalty: Optional[float], n_bkps: int # n_bkps is now effective_n_bkps
    ) -> Optional[List[int]]:
        """Exécute l'algorithme de détection de points de changement (Pelt ou Binseg)."""
        algo_instance = None
        bkps_indices: Optional[List[int]] = None

        if penalty is not None and hasattr(rpt, "Pelt"):
            logger.debug(f"{self.log_prefix} Utilisation de Pelt avec pénalité {penalty}.")
            algo_instance = rpt.Pelt(model=model) # type: ignore
            try:
                algo_instance.fit(points)
                bkps_indices = algo_instance.predict(pen=penalty)
            except Exception as e: # Catch more specific exceptions if known for Pelt
                logger.error(f"{self.log_prefix} Erreur avec rpt.Pelt: {e}", exc_info=True)
                return None
        elif hasattr(rpt, "Binseg"): # n_bkps est maintenant garanti être effective_n_bkps (non-négatif)
            if n_bkps <= 0: # Binseg requires n_bkps > 0 for meaningful segmentation
                logger.info(f"{self.log_prefix} Nombre de points de rupture (n_bkps={n_bkps}) est <= 0. Aucun point de rupture ne sera détecté par Binseg.")
                return [] # Return empty list as no breakpoints are sought
            logger.debug(f"{self.log_prefix} Utilisation de Binseg avec n_bkps={n_bkps}.")
            algo_instance = rpt.Binseg(model=model) # type: ignore
            try:
                algo_instance.fit(points)
                bkps_indices = algo_instance.predict(n_bkps=n_bkps)
            except Exception as e: # Catch more specific exceptions if known for Binseg
                logger.error(f"{self.log_prefix} Erreur avec rpt.Binseg: {e}", exc_info=True)
                return None
        else:
            logger.error(f"{self.log_prefix} Configuration invalide pour détection de points de rupture: "
                           "Pelt non dispo ou pénalité non fournie, et Binseg non dispo.")
            return None
        
        if bkps_indices is None: 
            logger.warning(f"{self.log_prefix} Aucun indice de point de rupture retourné par l'algorithme (résultat None).")
            return [] 
            
        return bkps_indices

    def _process_breakpoint_indices(self, bkps_indices: List[int], series_segmented: pd.Series) -> List[pd.Timestamp]:
        """Convertit les indices de points de rupture en timestamps et finalise les frontières."""
        if not bkps_indices: # If bkps_indices is empty (e.g. n_bkps=0 for Binseg, or Pelt found none)
            logger.info(f"{self.log_prefix} Aucun point de rupture trouvé ou demandé. Frontières seront début/fin de série.")
            return sorted(list(set([series_segmented.index.min(), series_segmented.index.max()])))

        # Filter out indices that are out of bounds for series_segmented.index
        # Ruptures indices are typically 1-based for the point *after* the break, or can be 0-based.
        # Filtrer sûrement: les indices doivent être < len(series_segmented) pour series_segmented.index[idx]
        # Et > 0 pour series_segmented.index[idx-1]
        valid_timestamps = []
        for idx in bkps_indices:
            if 0 < idx < len(series_segmented):
                valid_timestamps.append(series_segmented.index[idx - 1]) # Point before the break
            elif idx == len(series_segmented) and len(series_segmented) > 0 : # Break after last point
                 # This means the last segment ends at the very last data point.
                 # We don't need to add series_segmented.index[-1] again if it's already the last point.
                 pass # No specific timestamp to add here other than series_segmented.index.max() later
            elif idx == 0: # Break before the first point, shouldn't happen with typical usage
                logger.debug(f"{self.log_prefix} Indice de point de rupture 0 ignoré.")


        # Add start and end of the series to ensure full coverage
        all_boundaries = sorted(list(set(
            [series_segmented.index.min()] + valid_timestamps + [series_segmented.index.max()]
        )))
        
        # Nombre de points de rupture réels identifiés (excluant début/fin de série sauf s'ils étaient des points de rupture)
        num_actual_bkps = len(all_boundaries) - 2 if len(all_boundaries) > 1 else 0 
        num_segments = len(all_boundaries) -1 if len(all_boundaries) > 0 else 0

        logger.info(f"{self.log_prefix} {num_actual_bkps} points de rupture uniques traités, "
                    f"résultant en {num_segments} segments.")
        return all_boundaries

    def plot_folds(
        self,
        df_full_data: pd.DataFrame, # DataFrame avec la série à plotter (ex: 'close') et un DatetimeIndex
        folds: List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
        plot_column: str = 'close',
        title: Optional[str] = None,
        output_path: Optional[Path] = None
    ) -> None:
        """Visualise les folds IS et OOS sur un graphique temporel."""
        if not folds:
            logger.info(f"{self.log_prefix} Aucun fold à visualiser.")
            return
        if plot_column not in df_full_data.columns:
            logger.error(f"{self.log_prefix} Colonne '{plot_column}' non trouvée dans df_full_data pour la visualisation.")
            return

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(df_full_data.index, df_full_data[plot_column], label=f'{plot_column} (Données Complètes)', color='grey', alpha=0.5)

        colors = plt.cm.get_cmap('viridis', len(folds) + 2) # +2 pour un peu plus de variation

        for i, (_, _, fold_id, is_start, is_end, oos_start, oos_end) in enumerate(folds):
            color_is = colors(i)
            color_oos = colors(i + 1) # Légèrement différent pour OOS

            # Plot IS period
            ax.axvspan(is_start, is_end, alpha=0.3, color=color_is, label=f'Fold {fold_id} IS' if i == 0 else None)
            # Plot OOS period
            ax.axvspan(oos_start, oos_end, alpha=0.4, color=color_oos, label=f'Fold {fold_id} OOS' if i == 0 else None)
            
            # Annotations
            ax.text(is_start + (is_end - is_start)/2, ax.get_ylim()[1]*0.95 - (i%3 * ax.get_ylim()[1]*0.03), f"F{fold_id} IS", 
                    horizontalalignment='center', color='black', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc=color_is, alpha=0.7))
            ax.text(oos_start + (oos_end - oos_start)/2, ax.get_ylim()[1]*0.90 - (i%3 * ax.get_ylim()[1]*0.03), f"F{fold_id} OOS", 
                    horizontalalignment='center', color='black', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc=color_oos, alpha=0.7))


        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.ylabel(plot_column)
        plt.xlabel("Date")
        
        final_title = title or f"Visualisation des Folds WFO pour {plot_column}"
        plt.title(final_title)
        
        # Créer une légende unique pour "IS" et "OOS"
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='grey', alpha=0.5, label=f'{plot_column} (Données Complètes)'),
            Patch(facecolor=colors(0), alpha=0.3, label='Période In-Sample (IS)'),
            Patch(facecolor=colors(1), alpha=0.4, label='Période Out-of-Sample (OOS)')
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        plt.tight_layout()
        if output_path:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=150)
                logger.info(f"{self.log_prefix} Visualisation des folds sauvegardée dans : {output_path}")
            except Exception as e_save_plot:
                logger.error(f"{self.log_prefix} Erreur lors de la sauvegarde de la visualisation des folds : {e_save_plot}")
        else:
            plt.show()
        plt.close(fig) # Fermer la figure pour libérer la mémoire

