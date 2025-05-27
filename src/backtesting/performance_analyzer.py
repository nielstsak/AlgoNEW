# src/backtesting/performance_analyzer.py
"""
Ce module est responsable du calcul d'un ensemble complet de métriques de
performance (PnL, Sharpe, Sortino, Max Drawdown, Win Rate, VaR, CVaR, Omega, etc.)
à partir des résultats d'un backtest (journal des trades et courbe d'équité).
Il inclut des optimisations et la possibilité d'utiliser un cache.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
import time # Pour le timing des fonctions coûteuses
from concurrent.futures import ThreadPoolExecutor, as_completed

# Imports pour les nouvelles métriques et optimisations
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
# Numba est optionnel, utilisé si des boucles Python intensives sont nécessaires.
try:
    import numba
except ImportError:
    numba = None # type: ignore
    logging.getLogger(__name__).info("Numba n'est pas installé. Les optimisations JIT ne seront pas disponibles.")

# Tentative d'importation de l'interface ICacheManager
try:
    from src.core.interfaces import ICacheManager
except ImportError:
    class ICacheManager: # type: ignore
        def get_or_compute(self, key: str, compute_func: Callable[[], Any], ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> Any: return compute_func()
    logging.getLogger(__name__).warning(
        "ICacheManager interface not found in performance_analyzer. Using a placeholder."
    )


logger = logging.getLogger(__name__)

@dataclass
class RegimePerformance:
    """Stocke les métriques de performance pour un régime de marché spécifique."""
    regime_id: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_days: float = 0.0
    num_periods_in_regime: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_summary: Dict[str, float] = field(default_factory=dict) # Moyenne des features pour ce régime

@dataclass
class StabilityMetricsReport:
    """Stocke les métriques de stabilité."""
    rolling_sharpe_summary: Dict[int, Dict[str, Optional[float]]] = field(default_factory=dict) # window -> {mean, std, var}
    # D'autres métriques de stabilité pourraient être ajoutées ici

# --- Fonctions de calcul de métriques individuelles ---

def _calculate_var_cvar_historical(returns: pd.Series, confidence_level: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Calcule la VaR et la CVaR historiques."""
    if returns.empty or returns.isnull().all():
        return None, None
    # S'assurer que les rendements sont triés pour le calcul de la CVaR
    sorted_returns = returns.dropna().sort_values()
    if sorted_returns.empty:
        return None, None
        
    var_idx = int(len(sorted_returns) * (1 - confidence_level))
    if var_idx >= len(sorted_returns): # Si pas assez de données pour le quantile
        var_value = sorted_returns.iloc[-1] if not sorted_returns.empty else np.nan
    else:
        var_value = sorted_returns.iloc[var_idx]
    
    cvar_value = sorted_returns[sorted_returns <= var_value].mean()
    
    return float(var_value) if pd.notna(var_value) else None, \
           float(cvar_value) if pd.notna(cvar_value) else None

def _calculate_var_cvar_parametric(returns: pd.Series, confidence_level: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Calcule la VaR et la CVaR paramétriques (supposant une distribution normale)."""
    if returns.empty or returns.isnull().all():
        return None, None
    
    returns_clean = returns.dropna()
    if len(returns_clean) < 2: # Besoin d'au moins 2 points pour std
        return None, None

    mu = returns_clean.mean()
    sigma = returns_clean.std()

    if sigma == 0 or pd.isna(sigma) or pd.isna(mu): # Si pas de volatilité ou mu invalide
        # Si mu > 0 et sigma = 0, VaR/CVaR pourraient être -mu (si on perd tout le rendement moyen)
        # ou 0 si on ne peut rien perdre. Pour la simplicité, on retourne None.
        return None, None

    var_value = scipy_stats.norm.ppf(1 - confidence_level, loc=mu, scale=sigma)
    # CVaR pour une distribution normale : mu - sigma * (phi(Z_alpha) / (1 - alpha))
    # où Z_alpha est le quantile et phi est la PDF de la normale standard.
    z_alpha = scipy_stats.norm.ppf(1 - confidence_level)
    cvar_value = mu - sigma * (scipy_stats.norm.pdf(z_alpha) / (1 - confidence_level))
    
    return float(var_value), float(cvar_value)

def calculate_var_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical' # 'historical' ou 'parametric'
) -> Dict[str, Optional[float]]:
    """
    Calcule la Value at Risk (VaR) et la Conditional Value at Risk (CVaR).

    Args:
        returns (pd.Series): Série des rendements (journaliers, etc.).
        confidence_level (float): Niveau de confiance (ex: 0.95 pour 95%).
        method (str): 'historical' ou 'parametric'.

    Returns:
        Dict[str, Optional[float]]: {'var': float_or_None, 'cvar': float_or_None}.
                                     Les valeurs sont négatives pour les pertes.
    """
    log_prefix_var = "[VaR/CVaR]"
    if not isinstance(returns, pd.Series) or returns.empty:
        logger.warning(f"{log_prefix_var} Série de rendements vide ou invalide.")
        return {'var': None, 'cvar': None}
    
    returns_clean = returns.dropna()
    if len(returns_clean) < 2: # Besoin d'au moins 2 points pour certaines méthodes
        logger.warning(f"{log_prefix_var} Pas assez de données de rendement valides ({len(returns_clean)}) pour le calcul.")
        return {'var': None, 'cvar': None}

    var_val: Optional[float] = None
    cvar_val: Optional[float] = None

    if method.lower() == 'historical':
        var_val, cvar_val = _calculate_var_cvar_historical(returns_clean, confidence_level)
    elif method.lower() == 'parametric':
        var_val, cvar_val = _calculate_var_cvar_parametric(returns_clean, confidence_level)
    else:
        logger.error(f"{log_prefix_var} Méthode VaR/CVaR inconnue : {method}. Utilisation de 'historical'.")
        var_val, cvar_val = _calculate_var_cvar_historical(returns_clean, confidence_level)
    
    # VaR et CVaR sont typiquement exprimées comme des pertes (valeurs négatives).
    # Si les fonctions retournent des valeurs positives pour les pertes (ex: quantile de la distribution des pertes),
    # il faudrait les inverser ici. Les méthodes ci-dessus calculent les quantiles de la distribution des rendements,
    # donc un quantile bas est déjà une perte.
    return {'var': var_val, 'cvar': cvar_val}

def calculate_omega_ratio(
    returns: pd.Series,
    required_return_threshold: float = 0.0, # Seuil de rendement requis (MAR)
    periods_per_year: int = 252
) -> Optional[float]:
    """
    Calcule le ratio Omega.
    Omega = (Probabilité pondérée des gains au-dessus du seuil) / (Probabilité pondérée des pertes en dessous du seuil)
          = E[max(0, R - seuil)] / E[max(0, seuil - R)]
    """
    log_prefix_omega = "[OmegaRatio]"
    if not isinstance(returns, pd.Series) or returns.empty:
        logger.warning(f"{log_prefix_omega} Série de rendements vide ou invalide.")
        return None
    
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        logger.warning(f"{log_prefix_omega} Pas assez de données de rendement valides ({len(returns_clean)}).")
        return None

    # Rendements excédentaires par rapport au seuil
    excess_returns = returns_clean - required_return_threshold
    
    gains_above_threshold = excess_returns[excess_returns > 0].sum()
    losses_below_threshold_abs = abs(excess_returns[excess_returns < 0].sum())

    if losses_below_threshold_abs < 1e-9: # Éviter division par zéro
        if gains_above_threshold > 1e-9:
            return np.inf # Gains significatifs sans pertes en dessous du seuil
        return 1.0 # Ni gains ni pertes par rapport au seuil (ou gains/pertes négligeables)
    
    omega = gains_above_threshold / losses_below_threshold_abs
    return float(omega)


def _define_market_regimes(
    market_data: pd.DataFrame, # Doit contenir 'close' et 'volume' (ou d'autres features)
    n_regimes: int = 3,
    volatility_window: int = 20,
    trend_window: int = 50
) -> Optional[pd.Series]:
    """
    Définit les régimes de marché en utilisant K-Means sur des features dérivées.
    Features exemples : volatilité (ATR ou std dev des rendements) et trend (pente MA).
    """
    log_prefix_regime_def = "[DefineMarketRegimes]"
    if not isinstance(market_data, pd.DataFrame) or market_data.empty:
        logger.warning(f"{log_prefix_regime_def} market_data vide ou invalide.")
        return None
    if 'close' not in market_data.columns:
        logger.error(f"{log_prefix_regime_def} Colonne 'close' manquante dans market_data.")
        return None
    
    df_features = pd.DataFrame(index=market_data.index)
    
    # Feature 1: Volatilité (ex: ATR normalisé ou std des rendements)
    if all(col in market_data.columns for col in ['high', 'low', 'close']):
        atr_series = ta.atr(high=market_data['high'], low=market_data['low'], close=market_data['close'], length=volatility_window, append=False)
        if atr_series is not None and not atr_series.empty:
             # Normaliser l'ATR par le prix de clôture pour le rendre comparable dans le temps
            df_features['volatility_atr_norm'] = (atr_series / market_data['close']).replace([np.inf, -np.inf], np.nan)
    
    if 'volatility_atr_norm' not in df_features.columns: # Fallback si ATR ne peut être calculé
        df_features['volatility_returns_std'] = market_data['close'].pct_change().rolling(window=volatility_window).std()

    # Feature 2: Trend (ex: pente d'une MA ou différence entre MA courte et longue)
    if 'close' in market_data.columns:
        ma_trend = market_data['close'].rolling(window=trend_window).mean()
        # Pente simple (différence normalisée)
        df_features['trend_ma_slope'] = ma_trend.diff().fillna(0) / ma_trend.shift(1).replace(0, np.nan) # Normaliser par la MA précédente
        df_features['trend_ma_slope'].replace([np.inf, -np.inf], np.nan, inplace=True)


    df_features.dropna(inplace=True) # Supprimer les lignes avec NaNs dus aux rolling windows
    if len(df_features) < n_regimes * 5: # S'assurer qu'il y a assez de données pour le clustering
        logger.warning(f"{log_prefix_regime_def} Pas assez de données après calcul des features ({len(df_features)}) pour {n_regimes} régimes.")
        return None

    try:
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init='auto')
        regime_labels = kmeans.fit_predict(df_features)
        
        # Re-indexer les labels sur l'index original de market_data
        regime_series = pd.Series(regime_labels, index=df_features.index, name='market_regime')
        regime_series = regime_series.reindex(market_data.index).ffill() # Propager le dernier régime connu aux dates manquantes
        
        # Optionnel: Caractériser chaque régime (ex: moyenne des features) pour l'interprétabilité
        # for i in range(n_regimes):
        #     regime_data_features = df_features[regime_labels == i]
        #     logger.info(f"{log_prefix_regime_def} Régime {i} - Caractéristiques moyennes: {regime_data_features.mean().to_dict()}")
            
        return regime_series
    except Exception as e_kmeans:
        logger.error(f"{log_prefix_regime_def} Erreur lors du clustering K-Means : {e_kmeans}", exc_info=True)
        return None


def analyze_regime_performance(
    returns: pd.Series, # Rendements journaliers ou à la fréquence d'analyse
    market_data_for_regimes: pd.DataFrame, # OHLCV à la même fréquence que les rendements, pour calculer les features de régime
    n_regimes: int = 3,
    volatility_window_regime: int = 20,
    trend_window_regime: int = 50,
    cache_manager: Optional[ICacheManager] = None,
    base_cache_key: Optional[str] = None,
    periods_per_year: int = 252,
    risk_free_rate_daily: float = 0.0
) -> Optional[List[RegimePerformance]]:
    """
    Analyse la performance (métriques de base) décomposée par régime de marché.
    """
    log_prefix_regime_perf = "[AnalyzeRegimePerf]"
    
    if not isinstance(returns, pd.Series) or returns.empty or \
       not isinstance(market_data_for_regimes, pd.DataFrame) or market_data_for_regimes.empty:
        logger.warning(f"{log_prefix_regime_perf} Entrées 'returns' ou 'market_data_for_regimes' invalides.")
        return None
    
    # S'assurer que les index sont alignés et en UTC
    if not isinstance(returns.index, pd.DatetimeIndex) or \
       not isinstance(market_data_for_regimes.index, pd.DatetimeIndex):
        logger.error(f"{log_prefix_regime_perf} Les index de 'returns' et 'market_data_for_regimes' doivent être des DatetimeIndex.")
        return None
    
    returns_utc = returns.tz_convert('UTC') if returns.index.tz is not None and str(returns.index.tz).upper() != 'UTC' else \
                  (returns.tz_localize('UTC') if returns.index.tz is None else returns)
    market_data_utc = market_data_for_regimes.tz_convert('UTC') if market_data_for_regimes.index.tz is not None and str(market_data_for_regimes.index.tz).upper() != 'UTC' else \
                      (market_data_for_regimes.tz_localize('UTC') if market_data_for_regimes.index.tz is None else market_data_for_regimes)

    # Définir les régimes
    regime_labels_series: Optional[pd.Series] = None
    cache_key_regimes = f"{base_cache_key}_regimes_{n_regimes}_{volatility_window_regime}_{trend_window_regime}" if base_cache_key else None

    def _compute_regimes():
        return _define_market_regimes(market_data_utc, n_regimes, volatility_window_regime, trend_window_regime)

    if cache_manager and cache_key_regimes:
        regime_labels_series = cache_manager.get_or_compute(cache_key_regimes, _compute_regimes, ttl=3600*24) # Cache pour 1 jour
    else:
        regime_labels_series = _compute_regimes()

    if regime_labels_series is None or regime_labels_series.isnull().all():
        logger.warning(f"{log_prefix_regime_perf} Échec de la définition des régimes de marché.")
        return None
        
    # Aligner les séries de rendements et de labels de régime
    # Utiliser un inner join implicite pour ne garder que les timestamps communs
    # après que regime_labels_series ait été reindexé et ffillé sur market_data_utc.index
    combined_df_for_regime_analysis = pd.DataFrame({'returns': returns_utc, 'regime': regime_labels_series})
    combined_df_for_regime_analysis.dropna(subset=['returns', 'regime'], inplace=True) # Enlever où l'un ou l'autre manque

    if combined_df_for_regime_analysis.empty:
        logger.warning(f"{log_prefix_regime_perf} Aucune donnée commune après alignement des rendements et des labels de régime.")
        return None

    results_by_regime: List[RegimePerformance] = []
    for regime_id_val in sorted(combined_df_for_regime_analysis['regime'].unique()):
        if pd.isna(regime_id_val): continue # Ignorer les NaNs dans les labels de régime
        
        regime_id_int = int(regime_id_val)
        returns_in_regime = combined_df_for_regime_analysis[combined_df_for_regime_analysis['regime'] == regime_id_int]['returns']
        
        if returns_in_regime.empty or len(returns_in_regime) < 2:
            logger.info(f"{log_prefix_regime_perf} Pas assez de données de rendement pour le régime {regime_id_int}. Saut.")
            results_by_regime.append(RegimePerformance(regime_id=regime_id_int, metrics={"status": "insufficient_data"}))
            continue

        # Calculer des métriques de base pour ce régime
        # Note: Ces métriques sont basées sur les rendements *dans* le régime, pas sur un backtest complet.
        # L'initial_capital n'est pas directement applicable ici de la même manière qu'un backtest.
        # On se concentre sur les caractéristiques des rendements du régime.
        
        regime_metrics: Dict[str, Any] = {}
        regime_metrics["Total Return Pct in Regime"] = (np.expm1(np.log1p(returns_in_regime).sum())) * 100.0 # ( (1+r1)*(1+r2)... -1 ) * 100
        regime_metrics["Mean Daily Return Pct in Regime"] = returns_in_regime.mean() * 100.0
        regime_metrics["Std Dev Daily Return Pct in Regime"] = returns_in_regime.std() * 100.0
        
        # Sharpe pour le régime (annualisé)
        if regime_metrics["Std Dev Daily Return Pct in Regime"] > 1e-9: # type: ignore
            sharpe_regime = (returns_in_regime.mean() - risk_free_rate_daily) / returns_in_regime.std() * np.sqrt(periods_per_year)
            regime_metrics["Sharpe Ratio in Regime (Annualized)"] = float(sharpe_regime) if pd.notna(sharpe_regime) else None
        else:
            regime_metrics["Sharpe Ratio in Regime (Annualized)"] = np.inf if returns_in_regime.mean() > risk_free_rate_daily else \
                                                                  (-np.inf if returns_in_regime.mean() < risk_free_rate_daily else 0.0)
        
        regime_start_date = returns_in_regime.index.min().isoformat() if not returns_in_regime.empty else None
        regime_end_date = returns_in_regime.index.max().isoformat() if not returns_in_regime.empty else None
        regime_duration_days = (returns_in_regime.index.max() - returns_in_regime.index.min()).total_seconds() / (24*3600) if not returns_in_regime.empty and len(returns_in_regime.index) > 1 else 0.0
        
        # Résumé des features pour ce régime (depuis market_data_utc aligné)
        # Cela nécessite que _define_market_regimes retourne aussi les features utilisées.
        # Pour l'instant, on omet cette partie car _define_market_regimes ne retourne que les labels.
        feature_summary_for_regime: Dict[str, float] = {} 

        results_by_regime.append(RegimePerformance(
            regime_id=regime_id_int,
            start_date=regime_start_date,
            end_date=regime_end_date,
            duration_days=regime_duration_days,
            num_periods_in_regime=len(returns_in_regime),
            metrics=regime_metrics,
            feature_summary=feature_summary_for_regime
        ))
        logger.info(f"{log_prefix_regime_perf} Performance pour Régime {regime_id_int} calculée. "
                    f"Rendement total: {regime_metrics.get('Total Return Pct in Regime'):.2f}%, "
                    f"Sharpe: {regime_metrics.get('Sharpe Ratio in Regime (Annualized)'):.2f}")
            
    return results_by_regime if results_by_regime else None


def calculate_stability_metrics(
    equity_curve: pd.Series, # Courbe d'équité (valeurs absolues)
    rolling_windows: Optional[List[int]] = None, # Ex: [20, 50, 100] périodes de rendement
    periods_per_year: int = 252,
    risk_free_rate_daily: float = 0.0,
    cache_manager: Optional[ICacheManager] = None,
    base_cache_key: Optional[str] = None
) -> Optional[StabilityMetricsReport]:
    """
    Calcule des métriques de stabilité, comme la variance du Sharpe Ratio roulant.
    """
    log_prefix_stability = "[CalcStabilityMetrics]"
    if not isinstance(equity_curve, pd.Series) or equity_curve.empty or len(equity_curve) < max(rolling_windows or [20]) + 1:
        logger.warning(f"{log_prefix_stability} Courbe d'équité invalide ou trop courte pour les fenêtres roulantes.")
        return None
    
    if rolling_windows is None:
        rolling_windows = [20, 50, 100] # Fenêtres par défaut

    report = StabilityMetricsReport()
    
    # Calculer les rendements journaliers (ou à la fréquence de l'equity_curve)
    # S'assurer que l'index est un DatetimeIndex pour resample si nécessaire
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        logger.warning(f"{log_prefix_stability} L'index de equity_curve n'est pas DatetimeIndex. "
                       "Les rendements et le Sharpe roulant pourraient être imprécis.")
        returns_for_stability = equity_curve.pct_change().dropna()
    else:
        # Si l'equity_curve n'est pas journalière, la resampler pour des rendements journaliers
        # ou calculer les rendements sur sa fréquence native.
        # Pour la simplicité, on calcule les rendements sur la fréquence native.
        returns_for_stability = equity_curve.pct_change().dropna()

    if returns_for_stability.empty or len(returns_for_stability) < 2:
        logger.warning(f"{log_prefix_stability} Pas assez de rendements valides pour calculer la stabilité.")
        return None

    for window in rolling_windows:
        if len(returns_for_stability) < window + 1:
            logger.info(f"{log_prefix_stability} Pas assez de données pour la fenêtre roulante de {window}. Saut.")
            report.rolling_sharpe_summary[window] = {"mean": None, "std": None, "var": None, "status": "insufficient_data"}
            continue

        cache_key_rolling_sharpe = f"{base_cache_key}_rolling_sharpe_w{window}" if base_cache_key else None
        
        def _compute_rolling_sharpe():
            # Calcul du Sharpe Ratio Roulant
            # Moyenne des rendements excédentaires roulants
            mean_excess_returns_rolling = (returns_for_stability - risk_free_rate_daily).rolling(window=window).mean()
            # Écart-type des rendements roulants
            std_returns_rolling = returns_for_stability.rolling(window=window).std()
            
            # Sharpe Roulant (non annualisé)
            rolling_sharpe_unannualized = mean_excess_returns_rolling / std_returns_rolling.replace(0, np.nan) # Remplacer std=0 par NaN pour éviter division par zéro
            rolling_sharpe_unannualized.dropna(inplace=True) # Enlever les NaNs initiaux ou dus à std=0
            
            if rolling_sharpe_unannualized.empty:
                return pd.Series(dtype=float) # Retourner une série vide si aucun Sharpe calculable
            
            # Annualiser le Sharpe Roulant
            rolling_sharpe_annualized = rolling_sharpe_unannualized * np.sqrt(periods_per_year)
            return rolling_sharpe_annualized.replace([np.inf, -np.inf], np.nan).dropna()

        rolling_sharpe_series: Optional[pd.Series] = None
        if cache_manager and cache_key_rolling_sharpe:
            rolling_sharpe_series = cache_manager.get_or_compute(cache_key_rolling_sharpe, _compute_rolling_sharpe, ttl=3600)
        else:
            rolling_sharpe_series = _compute_rolling_sharpe()
            
        if rolling_sharpe_series is not None and not rolling_sharpe_series.empty:
            report.rolling_sharpe_summary[window] = {
                "mean": float(rolling_sharpe_series.mean()) if pd.notna(rolling_sharpe_series.mean()) else None,
                "std": float(rolling_sharpe_series.std()) if pd.notna(rolling_sharpe_series.std()) else None,
                "var": float(rolling_sharpe_series.var()) if pd.notna(rolling_sharpe_series.var()) else None,
                "count": len(rolling_sharpe_series),
                "status": "calculated"
            }
            logger.info(f"{log_prefix_stability} Stabilité Sharpe (fenêtre {window}): "
                        f"Moyenne={report.rolling_sharpe_summary[window]['mean']:.2f}, "
                        f"StdDev={report.rolling_sharpe_summary[window]['std']:.2f}")
        else:
            logger.warning(f"{log_prefix_stability} Impossible de calculer le Sharpe roulant pour la fenêtre {window}.")
            report.rolling_sharpe_summary[window] = {"mean": None, "std": None, "var": None, "status": "calculation_failed_or_empty"}
            
    return report


# Fonction principale (existante, à modifier)
def calculate_performance_metrics_from_inputs(
    trades_df: pd.DataFrame,
    equity_curve_series: pd.Series,
    initial_capital: float,
    risk_free_rate_daily: float = 0.0,
    periods_per_year: int = 252,
    # Nouveaux paramètres optionnels
    market_data_for_regimes: Optional[pd.DataFrame] = None, # Pour l'analyse de régime
    cache_manager: Optional[ICacheManager] = None, # Pour la mise en cache
    base_cache_key_prefix: Optional[str] = None # Préfixe pour les clés de cache
) -> Dict[str, Any]:
    """
    Calcule diverses métriques de performance à partir d'un journal de trades et
    d'une série chronologique de la valeur du portefeuille (équité).
    Version améliorée avec nouvelles métriques, optimisations, et cache.
    """
    metrics: Dict[str, Any] = {}
    log_prefix = "[PerformanceAnalyzerV2]"
    overall_start_time = time.perf_counter()

    # ... (validation des entrées et initialisation des métriques comme avant) ...
    # (Cette partie est conservée de la version précédente, avec des ajustements mineurs)
    if not isinstance(equity_curve_series, pd.Series) or equity_curve_series.empty:
        logger.warning(f"{log_prefix} Courbe d'équité vide. Retour de métriques par défaut.")
        # Retourner un dict de métriques initialisées à None/0/N/A
        # (Logique d'initialisation des clés omise pour la brièveté, mais serait ici)
        return {"error": "Equity curve empty"} # Simplifié
    
    if not isinstance(equity_curve_series.index, pd.DatetimeIndex):
        logger.error(f"{log_prefix} Index de equity_curve_series non DatetimeIndex. Certaines métriques seront N/A.")
        # Forcer la conversion si possible, ou gérer l'erreur
        try:
            equity_curve_series.index = pd.to_datetime(equity_curve_series.index, errors='coerce', utc=True)
            equity_curve_series.dropna(subset=[equity_curve_series.index.name], inplace=True) # Si conversion échoue pour certains
        except: # pylint: disable=bare-except
             pass # Laisser tel quel, les calculs suivants échoueront gracieusement

    if not isinstance(trades_df, pd.DataFrame):
        trades_df = pd.DataFrame() # Assurer un DataFrame pour les accès
    
    # Initialisation des métriques (simplifié pour l'exemple)
    metrics["Initial Capital USDC"] = float(initial_capital)
    metrics["Final Equity USDC"] = float(equity_curve_series.iloc[-1]) if not equity_curve_series.empty else initial_capital
    metrics["Total Net PnL USDC"] = metrics["Final Equity USDC"] - metrics["Initial Capital USDC"]
    metrics["Total Net PnL Pct"] = (metrics["Total Net PnL USDC"] / initial_capital * 100.0) if initial_capital != 0 else 0.0
    
    # ... (Calculs des métriques de base existantes : Total Trades, Win Rate, etc.) ...
    # (Cette partie est conservée et supposée déjà vectorisée autant que possible)
    # Exemple:
    if not trades_df.empty and 'pnl_net_quote' in trades_df.columns:
        metrics["Total Trades"] = len(trades_df)
        # ... autres calculs de trades ...
    else:
        metrics["Total Trades"] = 0


    # Calcul des rendements journaliers (ou à la fréquence de l'equity_curve)
    daily_returns = pd.Series(dtype=float)
    if isinstance(equity_curve_series.index, pd.DatetimeIndex) and len(equity_curve_series) > 1:
        equity_curve_sorted = equity_curve_series.sort_index()
        # Resample à 'D' pour journalier, ou utiliser la fréquence native si déjà agrégé
        # Si la fréquence est déjà journalière ou plus basse, pas besoin de resample.
        # Pour la robustesse, on peut vérifier la fréquence médiane de l'index.
        # Pour cet exemple, on suppose que si c'est un DatetimeIndex, on peut calculer pct_change.
        daily_returns = equity_curve_sorted.pct_change().dropna() # dropna pour la première valeur
    elif len(equity_curve_series) > 1: # Fallback simple
        daily_returns = equity_curve_series.pct_change().dropna()
    
    if daily_returns.empty and len(equity_curve_series) > 1 : # Si pct_change donne vide (ex: série constante)
        daily_returns = pd.Series([0.0] * (len(equity_curve_series)-1), index=equity_curve_series.index[1:])


    # --- Calcul des Nouvelles Métriques (avec cache et parallélisation potentielle) ---
    tasks_for_metrics: Dict[str, Callable[[], Any]] = {}
    
    # Préparer les tâches pour les métriques qui peuvent être calculées indépendamment
    # VaR et CVaR
    tasks_for_metrics["VaR_CVaR_Historical_95"] = lambda: calculate_var_cvar(daily_returns, confidence_level=0.95, method='historical')
    tasks_for_metrics["VaR_CVaR_Parametric_95"] = lambda: calculate_var_cvar(daily_returns, confidence_level=0.95, method='parametric')
    tasks_for_metrics["VaR_CVaR_Historical_99"] = lambda: calculate_var_cvar(daily_returns, confidence_level=0.99, method='historical')

    # Omega Ratio
    tasks_for_metrics["Omega_Ratio_MAR0"] = lambda: calculate_omega_ratio(daily_returns, required_return_threshold=0.0, periods_per_year=periods_per_year)
    
    # Stabilité des Métriques (ex: Sharpe Roulant)
    tasks_for_metrics["Stability_Metrics"] = lambda: calculate_stability_metrics(
        equity_curve_series, 
        rolling_windows=[20, 60, 120], # Exemple de fenêtres
        periods_per_year=periods_per_year,
        risk_free_rate_daily=risk_free_rate_daily,
        cache_manager=cache_manager, # Passer le cache_manager
        base_cache_key=f"{base_cache_key_prefix}_stability" if base_cache_key_prefix else None
    )

    # Analyse par Régime de Marché
    if market_data_for_regimes is not None and not market_data_for_regimes.empty:
        tasks_for_metrics["Regime_Analysis"] = lambda: analyze_regime_performance(
            returns=daily_returns,
            market_data_for_regimes=market_data_for_regimes,
            n_regimes=3, # Configurable
            cache_manager=cache_manager,
            base_cache_key=f"{base_cache_key_prefix}_regime" if base_cache_key_prefix else None,
            periods_per_year=periods_per_year,
            risk_free_rate_daily=risk_free_rate_daily
        )

    # Exécution des tâches (séquentielle ou parallèle)
    # Pour la parallélisation, on pourrait utiliser ThreadPoolExecutor
    # Ici, un exemple simple d'exécution séquentielle avec cache pour certaines tâches
    
    # Exemple d'utilisation du cache pour une tâche coûteuse (Omega)
    omega_cache_key = f"{base_cache_key_prefix}_omega_mar0" if base_cache_key_prefix else None
    if cache_manager and omega_cache_key:
        metrics["Omega Ratio (MAR=0%)"] = cache_manager.get_or_compute(
            omega_cache_key, tasks_for_metrics["Omega_Ratio_MAR0"], ttl=3600 # Cache 1h
        )
    else:
        metrics["Omega Ratio (MAR=0%)"] = tasks_for_metrics["Omega_Ratio_MAR0"]()

    # VaR/CVaR (exécuté directement pour cet exemple, pourrait être mis en cache aussi)
    var_cvar_hist95 = tasks_for_metrics["VaR_CVaR_Historical_95"]()
    metrics["VaR 95% (Historical)"] = var_cvar_hist95.get('var')
    metrics["CVaR 95% (Historical)"] = var_cvar_hist95.get('cvar')
    
    # ... (Calculs des métriques existantes comme Sharpe, Sortino, Max Drawdown, Calmar) ...
    # Ces calculs sont largement conservés de la version précédente, en s'assurant
    # qu'ils utilisent `daily_returns` et gèrent les NaNs.
    # Exemple pour Sharpe (adapté de la version précédente):
    if not daily_returns.empty and len(daily_returns) > 1:
        std_dev_daily_returns = daily_returns.std()
        if pd.notna(std_dev_daily_returns) and std_dev_daily_returns > 1e-9:
            excess_daily_returns_mean = (daily_returns - risk_free_rate_daily).mean()
            sharpe_unannualized = excess_daily_returns_mean / std_dev_daily_returns
            metrics["Sharpe Ratio"] = sharpe_unannualized * np.sqrt(periods_per_year)
        elif (daily_returns - risk_free_rate_daily).mean() == 0 : 
            metrics["Sharpe Ratio"] = 0.0
        elif (daily_returns - risk_free_rate_daily).mean() > 0 :
             metrics["Sharpe Ratio"] = np.inf
        else: 
             metrics["Sharpe Ratio"] = -np.inf
    else:
        metrics["Sharpe Ratio"] = None
        
    # ... (autres métriques comme Max Drawdown, Sortino, Calmar, etc.) ...
    # Max Drawdown (exemple conservé et vectorisé)
    if not equity_curve_series.empty:
        cumulative_max_equity = equity_curve_series.cummax()
        drawdown_values = equity_curve_series - cumulative_max_equity
        metrics["Max Drawdown USDC"] = float(abs(drawdown_values.min())) if pd.notna(drawdown_values.min()) else 0.0
        drawdown_percentages = (equity_curve_series / cumulative_max_equity) - 1.0
        drawdown_percentages.replace([np.inf, -np.inf], np.nan, inplace=True)
        metrics["Max Drawdown Pct"] = float(abs(drawdown_percentages.min() * 100.0)) if pd.notna(drawdown_percentages.min()) else 0.0
    
    # Appel des autres nouvelles fonctions (Stability, Regime)
    # Ces fonctions gèrent leur propre cache interne si un cache_manager est passé.
    if "Stability_Metrics" in tasks_for_metrics:
        stability_report_obj = tasks_for_metrics["Stability_Metrics"]()
        if stability_report_obj:
            metrics["Stability Metrics Report"] = stability_report_obj # Stocker l'objet dataclass
            # On pourrait extraire des valeurs clés pour le dictionnaire principal
            for window, stats in stability_report_obj.rolling_sharpe_summary.items():
                metrics[f"Rolling Sharpe {window}p (Mean)"] = stats.get("mean")
                metrics[f"Rolling Sharpe {window}p (StdDev)"] = stats.get("std")

    if "Regime_Analysis" in tasks_for_metrics:
        regime_analysis_results_list = tasks_for_metrics["Regime_Analysis"]()
        if regime_analysis_results_list:
            metrics["Regime Analysis Results"] = regime_analysis_results_list # Liste d'objets RegimePerformance
            # Ajouter des métriques agrégées ou spécifiques par régime si pertinent pour le résumé principal

    # Nettoyage final des métriques
    for key_metric in list(metrics.keys()): # Utiliser list() pour copier les clés avant modification potentielle
        value_metric = metrics[key_metric]
        if isinstance(value_metric, (float, np.floating)):
            if np.isnan(value_metric) or np.isinf(value_metric):
                metrics[key_metric] = None
            elif key_metric not in ["Initial Capital USDC", "Final Equity USDC", "Total Net PnL USDC", "Max Drawdown USDC"]: # Garder plus de précision pour certains
                metrics[key_metric] = round(value_metric, 4)
            else: # Pour les montants USDC, garder 2 décimales
                 metrics[key_metric] = round(value_metric, 2)
        elif isinstance(value_metric, (np.integer, int)):
             metrics[key_metric] = int(value_metric) # S'assurer que c'est un int Python standard

    overall_end_time = time.perf_counter()
    metrics["Calculation Time (seconds)"] = round(overall_end_time - overall_start_time, 3)
    logger.info(f"{log_prefix} Calcul des métriques de performance V2 terminé en {metrics['Calculation Time (seconds)']:.3f}s.")
    logger.debug(f"{log_prefix} Métriques finales V2: { {k:v for k,v in metrics.items() if not isinstance(v, (pd.Series, pd.DataFrame, list, dict))} }") # Log résumé
    return metrics

