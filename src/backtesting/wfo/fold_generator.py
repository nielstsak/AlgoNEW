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
        # Assurer des rendements journaliers pour une mesure de volatilité standard
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
        min_gap_days: int = 5, # Gap minimum entre la fin de IS et le début de OOS
        random_seed: Optional[int] = None
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Génère des folds combinatoires (Purged K-Fold CV by de Prado),
        où les périodes IS et OOS sont sélectionnées pour tester la robustesse
        sur différentes conditions de marché non nécessairement contiguës.
        """
        logger.info(f"{self.log_prefix} Génération de {n_combinations} folds combinatoires...")
        if df_enriched_data.empty or len(df_enriched_data) < (is_duration_days + oos_duration_days + min_gap_days):
            logger.warning(f"{self.log_prefix} Pas assez de données pour générer des folds combinatoires avec les durées spécifiées.")
            return []

        rng = np.random.default_rng(random_seed)
        folds_generated: List[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        
        is_duration_td = pd.Timedelta(days=is_duration_days)
        oos_duration_td = pd.Timedelta(days=oos_duration_days)
        min_gap_td = pd.Timedelta(days=min_gap_days)
        
        date_index = df_enriched_data.index
        min_date = date_index.min()
        max_date = date_index.max()

        attempts = 0
        max_attempts = n_combinations * 5 # Pour éviter une boucle infinie

        while len(folds_generated) < n_combinations and attempts < max_attempts:
            attempts += 1
            
            # Sélectionner un début IS aléatoire
            # Max start for IS = max_date - oos_duration - min_gap - is_duration
            max_is_start_date = max_date - oos_duration_td - min_gap_td - is_duration_td
            if min_date >= max_is_start_date: # Pas assez de place
                logger.warning(f"{self.log_prefix} Plage de dates trop courte pour sélectionner un début IS aléatoire valide.")
                break 
            
            # Choisir un index aléatoire parmi les dates possibles pour le début IS
            possible_is_start_indices = date_index[date_index <= max_is_start_date]
            if possible_is_start_indices.empty: continue
            
            is_start_ts = rng.choice(possible_is_start_indices.to_numpy())
            is_start_ts = pd.Timestamp(is_start_ts, tz='UTC') # S'assurer que c'est un Timestamp
            is_end_ts_target = is_start_ts + is_duration_td
            
            # Trouver la date réelle la plus proche dans l'index
            is_end_idx_loc = date_index.get_indexer([is_end_ts_target], method='ffill')[0]
            is_end_ts = date_index[is_end_idx_loc]

            # Sélectionner un début OOS aléatoire après IS + gap
            oos_start_min_target = is_end_ts + min_gap_td
            max_oos_start_date = max_date - oos_duration_td
            
            if oos_start_min_target >= max_oos_start_date: continue # Pas de place pour OOS

            possible_oos_start_indices = date_index[(date_index >= oos_start_min_target) & (date_index <= max_oos_start_date)]
            if possible_oos_start_indices.empty: continue
            
            oos_start_ts = rng.choice(possible_oos_start_indices.to_numpy())
            oos_start_ts = pd.Timestamp(oos_start_ts, tz='UTC')
            oos_end_ts_target = oos_start_ts + oos_duration_td
            
            oos_end_idx_loc = date_index.get_indexer([oos_end_ts_target], method='ffill')[0]
            oos_end_ts = date_index[oos_end_idx_loc]

            # Vérifier validité et non-chevauchement
            if is_start_ts >= is_end_ts or oos_start_ts >= oos_end_ts or is_end_ts >= oos_start_ts:
                continue

            df_is = df_enriched_data.loc[is_start_ts:is_end_ts].copy()
            df_oos = df_enriched_data.loc[oos_start_ts:oos_end_ts].copy()

            if df_is.empty or df_oos.empty:
                continue
            
            folds_generated.append((df_is, df_oos, len(folds_generated), is_start_ts, is_end_ts, oos_start_ts, oos_end_ts))
            logger.debug(f"{self.log_prefix} Fold combinatoire {len(folds_generated)} généré. IS: {is_start_ts}-{is_end_ts}, OOS: {oos_start_ts}-{oos_end_ts}")

        if len(folds_generated) < n_combinations:
            logger.warning(f"{self.log_prefix} Seulement {len(folds_generated)}/{n_combinations} folds combinatoires ont pu être générés.")
        return folds_generated


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
        if not RUPTURES_AVAILABLE:
            logger.error(f"{self.log_prefix} Bibliothèque 'ruptures' non disponible. Impossible d'optimiser les frontières.")
            return []
        if df_enriched_data.empty:
            logger.warning(f"{self.log_prefix} DataFrame vide. Impossible d'optimiser les frontières.")
            return []

        if series_to_segment is None:
            # Utiliser la volatilité du prix de clôture par défaut
            if 'close' in df_enriched_data.columns:
                series_to_segment = df_enriched_data['close'].pct_change().rolling(window=getattr(self.wfo_settings, 'adaptive_volatility_window', 20)).std().dropna()
            else:
                logger.error(f"{self.log_prefix} Colonne 'close' non trouvée pour calculer la série par défaut pour la segmentation.")
                return []
        
        if series_to_segment.empty or len(series_to_segment) < (n_bkps_to_find or self.wfo_settings.n_splits) * 2:
            logger.warning(f"{self.log_prefix} Série à segmenter vide ou trop courte ({len(series_to_segment)} points).")
            return []

        points = series_to_segment.to_numpy().reshape(-1, 1)
        
        algo = None
        if penalty_value is not None and hasattr(rpt, "Pelt"): # Pelt si une pénalité est donnée
            algo = rpt.Pelt(model=model).fit(points)
            try:
                bkps_indices = algo.predict(pen=penalty_value)
            except Exception as e_pelt:
                logger.error(f"{self.log_prefix} Erreur avec rpt.Pelt: {e_pelt}")
                return []
        elif hasattr(rpt, "Binseg"): # Binseg si n_bkps est donné ou déduit
            n_bkps = n_bkps_to_find if n_bkps_to_find is not None else self.wfo_settings.n_splits -1
            if n_bkps <=0:
                logger.warning(f"{self.log_prefix} Nombre de points de rupture (n_bkps={n_bkps}) invalide pour Binseg.")
                return []
            algo = rpt.Binseg(model=model).fit(points)
            try:
                bkps_indices = algo.predict(n_bkps=n_bkps)
            except Exception as e_binseg:
                logger.error(f"{self.log_prefix} Erreur avec rpt.Binseg: {e_binseg}")
                return []
        else:
            logger.error(f"{self.log_prefix} Configuration invalide pour la détection de points de rupture (ni pénalité pour Pelt, ni n_bkps pour Binseg).")
            return []

        # Les indices retournés par ruptures sont des positions *après* le point de rupture.
        # Ils sont relatifs à l'index de `series_to_segment`.
        # On les convertit en timestamps de `series_to_segment.index`.
        # On enlève le dernier qui est souvent la fin du signal.
        if bkps_indices and bkps_indices[-1] >= len(series_to_segment):
            bkps_indices = bkps_indices[:-1]
            
        breakpoint_timestamps = [series_to_segment.index[idx-1] for idx in bkps_indices if 0 < idx < len(series_to_segment)] # idx-1 pour prendre le point *avant* la rupture
        
        # Ajouter le début et la fin de la série globale si ce ne sont pas déjà des points de rupture
        all_boundaries = sorted(list(set([series_to_segment.index.min()] + breakpoint_timestamps + [series_to_segment.index.max()])))
        
        logger.info(f"{self.log_prefix} {len(breakpoint_timestamps)} points de rupture optimisés trouvés, résultant en {len(all_boundaries)-1} segments.")
        return all_boundaries # Retourne une liste de timestamps qui délimitent les segments


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
        
        final_title = title if title else f"Visualisation des Folds WFO pour {plot_column}"
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

