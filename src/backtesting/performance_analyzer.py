# src/backtesting/performance_analyzer.py
"""
Ce module est responsable du calcul d'un ensemble complet de métriques de
performance (PnL, Sharpe, Sortino, Max Drawdown, Win Rate, etc.) à partir
des résultats d'un backtest (journal des trades et courbe d'équité).
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List # List n'est pas directement utilisé mais bon à avoir

logger = logging.getLogger(__name__)

def calculate_performance_metrics_from_inputs(
    trades_df: pd.DataFrame,
    equity_curve_series: pd.Series,
    initial_capital: float,
    risk_free_rate_daily: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    """
    Calcule diverses métriques de performance à partir d'un journal de trades et
    d'une série chronologique de la valeur du portefeuille (équité).

    Args:
        trades_df (pd.DataFrame): DataFrame des trades. Doit contenir au minimum
            `pnl_net_quote`. Idéalement aussi `entry_time`, `exit_time`,
            `entry_fee_quote`, `exit_fee_quote`.
        equity_curve_series (pd.Series): Série pandas de la valeur du portefeuille (équité)
            au fil du temps, indexée par pd.Timestamp (UTC attendu).
        initial_capital (float): Capital initial de la simulation.
        risk_free_rate_daily (float): Taux sans risque journalier (ex: 0.02 / 252).
        periods_per_year (int): Nombre de périodes de trading dans une année pour
            l'annualisation (ex: 252 pour jours de trading, 365 si calendaire).

    Returns:
        Dict[str, Any]: Un dictionnaire contenant les métriques de performance calculées.
                        Les valeurs NaN/inf sont converties en None ou 0.0.
    """
    metrics: Dict[str, Any] = {}
    log_prefix = "[PerformanceAnalyzer]"

    # Initialisation des métriques avec des valeurs par défaut/NaN
    metric_keys_numeric_nan = [
        "Total Net PnL Pct", "Win Rate Pct", "Loss Rate Pct",
        "Max Drawdown Pct", "Max Drawdown USDC", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
        "Profit Factor", "Average Win / Loss Ratio", "Annualized Return Pct",
        "Total Gross PnL USDC", "Total Commission USDC",
        "Average Winning Trade USDC", "Average Losing Trade USDC",
        "Largest Winning Trade USDC", "Largest Losing Trade USDC",
        "Peak Equity USDC", "Lowest Equity USDC"
    ]
    metric_keys_numeric_zero = [
        "Total Trades", "Number of Winning Trades", "Number of Losing Trades", "Number of Breakeven Trades"
    ]
    metric_keys_string_na = ["Start Date", "End Date", "Duration", "Average Trade Duration", "Median Trade Duration"]

    for key in metric_keys_numeric_nan: metrics[key] = None
    for key in metric_keys_numeric_zero: metrics[key] = 0
    for key in metric_keys_string_na: metrics[key] = "N/A"
    metrics["Duration Days"] = 0.0
    metrics["Initial Capital USDC"] = float(initial_capital)
    metrics["Final Equity USDC"] = float(initial_capital) # Sera mis à jour
    metrics["Total Net PnL USDC"] = 0.0


    # Validation des entrées principales
    if not isinstance(equity_curve_series, pd.Series) or equity_curve_series.empty:
        logger.warning(f"{log_prefix} La série de la courbe d'équité est vide ou n'est pas une Series. "
                       "La plupart des métriques ne pourront pas être calculées.")
        # Les valeurs par défaut initialisées ci-dessus seront retournées pour beaucoup de métriques.
        return metrics
    
    if not isinstance(equity_curve_series.index, pd.DatetimeIndex):
        logger.error(f"{log_prefix} L'index de equity_curve_series doit être un DatetimeIndex. "
                     "Métriques basées sur le temps et les rendements impossibles à calculer.")
        # On peut calculer certaines métriques basées sur les trades, mais les métriques de drawdown/rendement seront fausses.
        # Pour la robustesse, on pourrait s'arrêter ou continuer avec des avertissements.
        # Pour l'instant, on continue, mais les métriques temporelles seront "N/A" ou basées sur des suppositions.
    
    if not isinstance(trades_df, pd.DataFrame): # Peut être vide, mais doit être un DataFrame
        logger.warning(f"{log_prefix} trades_df n'est pas un DataFrame. "
                       "Les métriques basées sur les trades seront initialisées à 0/NaN.")
        trades_df = pd.DataFrame() # Assurer que c'est un DataFrame vide pour les accès ultérieurs

    # Période de la simulation
    if isinstance(equity_curve_series.index, pd.DatetimeIndex) and len(equity_curve_series.index) > 0:
        metrics["Start Date"] = equity_curve_series.index.min().isoformat()
        metrics["End Date"] = equity_curve_series.index.max().isoformat()
        if len(equity_curve_series.index) > 1:
            duration = equity_curve_series.index.max() - equity_curve_series.index.min()
            metrics["Duration Days"] = duration.total_seconds() / (24 * 60 * 60)
            metrics["Duration"] = str(duration)
    
    # PnL Global
    metrics["Final Equity USDC"] = float(equity_curve_series.iloc[-1])
    total_net_pnl = metrics["Final Equity USDC"] - metrics["Initial Capital USDC"]
    metrics["Total Net PnL USDC"] = float(total_net_pnl)
    metrics["Total Net PnL Pct"] = (total_net_pnl / initial_capital) * 100.0 if initial_capital != 0 else 0.0

    # Statistiques des Trades
    if not trades_df.empty and 'pnl_net_quote' in trades_df.columns:
        metrics["Total Trades"] = len(trades_df)
        
        # S'assurer que pnl_net_quote est numérique
        trades_df['pnl_net_quote'] = pd.to_numeric(trades_df['pnl_net_quote'], errors='coerce')
        # Supprimer les trades où le PnL n'a pas pu être converti (si errors='coerce')
        valid_pnl_trades = trades_df.dropna(subset=['pnl_net_quote'])

        winning_trades = valid_pnl_trades[valid_pnl_trades['pnl_net_quote'] > 1e-9] # Seuil pour éviter les erreurs de flottants
        losing_trades = valid_pnl_trades[valid_pnl_trades['pnl_net_quote'] < -1e-9]
        breakeven_trades = valid_pnl_trades[abs(valid_pnl_trades['pnl_net_quote']) <= 1e-9]

        metrics["Number of Winning Trades"] = len(winning_trades)
        metrics["Number of Losing Trades"] = len(losing_trades)
        metrics["Number of Breakeven Trades"] = len(breakeven_trades)

        if metrics["Total Trades"] > 0:
            metrics["Win Rate Pct"] = (metrics["Number of Winning Trades"] / metrics["Total Trades"]) * 100.0
            metrics["Loss Rate Pct"] = (metrics["Number of Losing Trades"] / metrics["Total Trades"]) * 100.0
        
        total_entry_fees = trades_df['entry_fee_quote'].sum() if 'entry_fee_quote' in trades_df.columns else 0.0
        total_exit_fees = trades_df['exit_fee_quote'].sum() if 'exit_fee_quote' in trades_df.columns else 0.0
        metrics["Total Commission USDC"] = float(total_entry_fees + total_exit_fees)
        metrics["Total Gross PnL USDC"] = metrics["Total Net PnL USDC"] + metrics["Total Commission USDC"]

        metrics["Average Winning Trade USDC"] = float(winning_trades['pnl_net_quote'].mean()) if not winning_trades.empty else 0.0
        metrics["Average Losing Trade USDC"] = float(losing_trades['pnl_net_quote'].mean()) if not losing_trades.empty else 0.0
        metrics["Largest Winning Trade USDC"] = float(winning_trades['pnl_net_quote'].max()) if not winning_trades.empty else 0.0
        metrics["Largest Losing Trade USDC"] = float(losing_trades['pnl_net_quote'].min()) if not losing_trades.empty else 0.0

        sum_gross_wins = winning_trades['pnl_net_quote'].sum() + \
                         (winning_trades['entry_fee_quote'].sum() if 'entry_fee_quote' in winning_trades else 0.0) + \
                         (winning_trades['exit_fee_quote'].sum() if 'exit_fee_quote' in winning_trades else 0.0)
        sum_gross_losses_abs = abs(losing_trades['pnl_net_quote'].sum() + \
                                (losing_trades['entry_fee_quote'].sum() if 'entry_fee_quote' in losing_trades else 0.0) + \
                                (losing_trades['exit_fee_quote'].sum() if 'exit_fee_quote' in losing_trades else 0.0))


        if sum_gross_losses_abs > 1e-9:
            metrics["Profit Factor"] = sum_gross_wins / sum_gross_losses_abs
        elif sum_gross_wins > 0: # Gains mais pas de pertes
            metrics["Profit Factor"] = np.inf
        else: # Pas de gains, ou pas de pertes (ou les deux)
            metrics["Profit Factor"] = 1.0 # Ou None/NaN si on préfère

        avg_gross_win = (winning_trades['pnl_net_quote'] + winning_trades.get('entry_fee_quote',0) + winning_trades.get('exit_fee_quote',0)).mean() if not winning_trades.empty else 0.0
        avg_gross_loss_abs = abs((losing_trades['pnl_net_quote'] + losing_trades.get('entry_fee_quote',0) + losing_trades.get('exit_fee_quote',0)).mean()) if not losing_trades.empty else 0.0

        if avg_gross_loss_abs > 1e-9:
            metrics["Average Win / Loss Ratio"] = avg_gross_win / avg_gross_loss_abs
        elif avg_gross_win > 0:
            metrics["Average Win / Loss Ratio"] = np.inf
        else:
            metrics["Average Win / Loss Ratio"] = 1.0

        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            try:
                entry_times = pd.to_datetime(trades_df['entry_time'], errors='coerce', utc=True)
                exit_times = pd.to_datetime(trades_df['exit_time'], errors='coerce', utc=True)
                valid_durations_mask = entry_times.notna() & exit_times.notna()
                if valid_durations_mask.any():
                    trade_durations = (exit_times[valid_durations_mask] - entry_times[valid_durations_mask])
                    metrics["Average Trade Duration"] = str(trade_durations.mean())
                    metrics["Median Trade Duration"] = str(trade_durations.median())
            except Exception as e_dur:
                logger.warning(f"{log_prefix} Erreur lors du calcul de la durée des trades: {e_dur}")
    
    # Max Drawdown
    if not equity_curve_series.empty:
        cumulative_max_equity = equity_curve_series.cummax()
        drawdown_values = equity_curve_series - cumulative_max_equity # Série de drawdowns en valeur
        metrics["Max Drawdown USDC"] = float(abs(drawdown_values.min())) if pd.notna(drawdown_values.min()) else 0.0
        
        # Pour le drawdown en pourcentage, il faut diviser par le pic *précédant* le creux
        # (equity_curve_series / cumulative_max_equity) - 1 donne le drawdown par rapport au pic courant
        drawdown_percentages = (equity_curve_series / cumulative_max_equity) - 1.0
        drawdown_percentages.replace([np.inf, -np.inf], np.nan, inplace=True) # Gérer division par zéro si equity était 0
        metrics["Max Drawdown Pct"] = float(abs(drawdown_percentages.min() * 100.0)) if pd.notna(drawdown_percentages.min()) else 0.0
    
    # Rendements Journaliers (basés sur l'index de equity_curve_series)
    daily_returns = pd.Series(dtype=float)
    if isinstance(equity_curve_series.index, pd.DatetimeIndex) and len(equity_curve_series) > 1:
        # S'assurer que la série est triée par temps pour resample et pct_change corrects
        equity_curve_sorted = equity_curve_series.sort_index()
        daily_equity = equity_curve_sorted.resample('D').last() # Prendre la dernière valeur d'équité de chaque jour
        daily_equity.ffill(inplace=True) # Remplir les jours non tradés avec la valeur précédente
        daily_returns = daily_equity.pct_change().fillna(0.0) # Remplir le premier NaN avec 0
    elif len(equity_curve_series) > 1 : # Si pas DatetimeIndex, tentative simple (moins précis)
        logger.warning(f"{log_prefix} equity_curve_series n'a pas de DatetimeIndex. Les rendements journaliers et les ratios basés sur le temps pourraient être imprécis.")
        daily_returns = equity_curve_series.pct_change().fillna(0.0)

    # Rendement Annualisé
    total_return_pct = metrics["Total Net PnL Pct"] / 100.0 # Convertir de % en décimal
    num_years = metrics["Duration Days"] / 365.25 if metrics["Duration Days"] > 0 else (1.0 / periods_per_year if periods_per_year > 0 else 1.0)

    if num_years > 1e-9: # Éviter division par zéro ou problèmes avec des durées très courtes
        # (1 + total_return_pct)^(1/num_years) - 1
        annualized_return = ((1.0 + total_return_pct) ** (1.0 / num_years)) - 1.0
    elif total_return_pct != 0 and periods_per_year > 0 and metrics["Duration Days"] > 0: # Extrapoler pour courtes périodes si rendement non nul
        # Si la durée est < 1 période, on peut annualiser en multipliant
        annualized_return = total_return_pct * (periods_per_year / (metrics["Duration Days"] * (periods_per_year / 365.25)))
    else:
        annualized_return = 0.0
    metrics["Annualized Return Pct"] = float(annualized_return * 100.0)

    # Calmar Ratio
    if metrics.get("Max Drawdown Pct") is not None and metrics["Max Drawdown Pct"] > 1e-9: # type: ignore
        metrics["Calmar Ratio"] = metrics["Annualized Return Pct"] / metrics["Max Drawdown Pct"] # type: ignore
    elif metrics["Annualized Return Pct"] != 0:
        metrics["Calmar Ratio"] = np.inf # Rendement positif avec drawdown nul ou négligeable
    else: # Rendement nul ou négatif avec drawdown nul
        metrics["Calmar Ratio"] = 0.0


    # Sharpe Ratio
    if not daily_returns.empty and len(daily_returns) > 1:
        std_dev_daily_returns = daily_returns.std()
        if std_dev_daily_returns > 1e-9:
            excess_daily_returns_mean = (daily_returns - risk_free_rate_daily).mean()
            sharpe_unannualized = excess_daily_returns_mean / std_dev_daily_returns
            metrics["Sharpe Ratio"] = sharpe_unannualized * np.sqrt(periods_per_year)
        elif (daily_returns - risk_free_rate_daily).mean() == 0 : # Pas de rendement excédentaire et pas de volatilité
            metrics["Sharpe Ratio"] = 0.0
        elif (daily_returns - risk_free_rate_daily).mean() > 0 : # Rendement excédentaire positif sans volatilité
             metrics["Sharpe Ratio"] = np.inf
        else: # Rendement excédentaire négatif sans volatilité
             metrics["Sharpe Ratio"] = -np.inf
    else:
        metrics["Sharpe Ratio"] = None # Ou 0.0 si on préfère

    # Sortino Ratio
    if not daily_returns.empty and len(daily_returns) > 1:
        target_daily_return = risk_free_rate_daily # Ou 0.0 si on compare à un rendement nul
        negative_excess_returns = daily_returns[daily_returns < target_daily_return] - target_daily_return
        downside_std_dev = np.sqrt((negative_excess_returns**2).mean()) if not negative_excess_returns.empty else 0.0

        if downside_std_dev > 1e-9:
            mean_excess_return_for_sortino = (daily_returns - target_daily_return).mean()
            sortino_unannualized = mean_excess_return_for_sortino / downside_std_dev
            metrics["Sortino Ratio"] = sortino_unannualized * np.sqrt(periods_per_year)
        elif (daily_returns - target_daily_return).mean() >= 0: # Pas de downside risk ou rendement >= cible
            metrics["Sortino Ratio"] = np.inf if (daily_returns - target_daily_return).mean() > 1e-9 else 0.0
        else: # Rendement < cible et pas de downside risk (ne devrait pas arriver si target_daily_return est bien défini)
            metrics["Sortino Ratio"] = -np.inf # Rendement négatif sans downside deviation mesurable
    else:
        metrics["Sortino Ratio"] = None

    # Peak et Lowest Equity
    if not equity_curve_series.empty:
        metrics["Peak Equity USDC"] = float(equity_curve_series.max())
        metrics["Lowest Equity USDC"] = float(equity_curve_series.min())
    else: # Devrait déjà être couvert par la validation initiale
        metrics["Peak Equity USDC"] = initial_capital
        metrics["Lowest Equity USDC"] = initial_capital


    # Nettoyage final des métriques numériques
    for key in metrics:
        if isinstance(metrics[key], (float, np.floating)):
            if np.isnan(metrics[key]) or np.isinf(metrics[key]):
                metrics[key] = None # Convertir en None pour JSON et usage général
            elif key not in ["Initial Capital USDC", "Final Equity USDC", "Total Net PnL USDC", "Total Gross PnL USDC", "Total Commission USDC", "Max Drawdown USDC", "Peak Equity USDC", "Lowest Equity USDC", "Average Winning Trade USDC", "Average Losing Trade USDC", "Largest Winning Trade USDC", "Largest Losing Trade USDC"]:
                # Arrondir la plupart des floats, mais garder plus de précision pour les montants en USDC
                 metrics[key] = round(metrics[key], 4) # type: ignore
            else: # Pour les montants USDC, garder 2 décimales
                 metrics[key] = round(metrics[key], 2) # type: ignore

    logger.info(f"{log_prefix} Calcul des métriques de performance terminé.")
    logger.debug(f"{log_prefix} Métriques finales : {metrics}")
    return metrics

