import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def calculate_performance_metrics(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series,
    initial_capital: float,
    risk_free_rate_daily: float = 0.0, # Taux journalier
    periods_per_year: int = 252 # Pour annualiser Sharpe. Ex: 252 pour jours de trading, 365 si calendaire, 52 si hebdo.
) -> Dict[str, Any]:
    """
    Calcule diverses métriques de performance à partir d'un journal de trades et d'une courbe d'équité.

    Args:
        trades_df (pd.DataFrame): DataFrame des trades. Doit contenir au moins 'pnl_net_usd'.
                                  Attend également 'entry_timestamp' et 'exit_timestamp' pour la durée des trades.
        equity_curve (pd.Series): Série pandas de la valeur du portefeuille au fil du temps, indexée par timestamp.
        initial_capital (float): Capital initial de la simulation.
        risk_free_rate_daily (float): Taux sans risque journalier pour le calcul du ratio de Sharpe.
        periods_per_year (int): Nombre de périodes de trading dans une année pour l'annualisation.

    Returns:
        Dict[str, Any]: Un dictionnaire contenant les métriques de performance calculées.
    """
    metrics: Dict[str, Any] = {}
    log_prefix = "[PerfCalc]"

    # Validation des entrées
    if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
        logger.warning(f"{log_prefix} Equity curve is empty or not a Series. Cannot calculate most metrics.")
        # Retourner des métriques de base ou des NaN
        metrics["Initial Capital USDC"] = initial_capital
        metrics["Final Equity USDC"] = initial_capital
        metrics["Total Net PnL USDC"] = 0.0
        metrics["Total Net PnL Pct"] = 0.0
        metrics["Total Trades"] = 0
        for key in ["Start Date", "End Date", "Duration Days", "Duration", "Win Rate Pct", 
                    "Max Drawdown Pct", "Max Drawdown USDC", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                    "Profit Factor", "Average Win / Loss Ratio"]:
            metrics[key] = np.nan if "Pct" in key or "Ratio" in key else ("N/A" if "Date" in key or "Duration" in key else 0)
        return metrics

    if not isinstance(trades_df, pd.DataFrame): # trades_df peut être vide s'il n'y a pas de trades
        logger.warning(f"{log_prefix} Trades DataFrame is not a DataFrame. Trade-based metrics will be 0 or NaN.")
        trades_df = pd.DataFrame() # Assurer un DataFrame vide pour éviter les erreurs d'attribut

    # Période de la simulation
    metrics["Start Date"] = equity_curve.index.min().isoformat()
    metrics["End Date"] = equity_curve.index.max().isoformat()
    duration = equity_curve.index.max() - equity_curve.index.min()
    metrics["Duration Days"] = duration.days
    metrics["Duration"] = str(duration)

    # Métriques de base PnL
    metrics["Initial Capital USDC"] = initial_capital
    metrics["Final Equity USDC"] = equity_curve.iloc[-1]
    total_net_pnl = metrics["Final Equity USDC"] - initial_capital
    metrics["Total Net PnL USDC"] = total_net_pnl
    metrics["Total Net PnL Pct"] = (total_net_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0
    
    # Métriques basées sur les trades
    if not trades_df.empty and 'pnl_net_usd' in trades_df.columns:
        metrics["Total Trades"] = len(trades_df)
        
        winning_trades = trades_df[trades_df['pnl_net_usd'] > 0]
        losing_trades = trades_df[trades_df['pnl_net_usd'] < 0]
        breakeven_trades = trades_df[trades_df['pnl_net_usd'] == 0]

        metrics["Number of Winning Trades"] = len(winning_trades)
        metrics["Number of Losing Trades"] = len(losing_trades)
        metrics["Number of Breakeven Trades"] = len(breakeven_trades)

        metrics["Win Rate Pct"] = (len(winning_trades) / metrics["Total Trades"]) * 100 if metrics["Total Trades"] > 0 else 0.0
        metrics["Loss Rate Pct"] = (len(losing_trades) / metrics["Total Trades"]) * 100 if metrics["Total Trades"] > 0 else 0.0

        metrics["Total Gross PnL USDC"] = trades_df['pnl_gross_usd'].sum() if 'pnl_gross_usd' in trades_df.columns else np.nan
        metrics["Total Commission USDC"] = trades_df['commission_usd'].sum() if 'commission_usd' in trades_df.columns else np.nan

        avg_win_usd = winning_trades['pnl_net_usd'].mean() if not winning_trades.empty else 0.0
        avg_loss_usd = losing_trades['pnl_net_usd'].mean() if not losing_trades.empty else 0.0 # Sera négatif
        metrics["Average Winning Trade USDC"] = avg_win_usd
        metrics["Average Losing Trade USDC"] = avg_loss_usd

        metrics["Largest Winning Trade USDC"] = winning_trades['pnl_net_usd'].max() if not winning_trades.empty else 0.0
        metrics["Largest Losing Trade USDC"] = losing_trades['pnl_net_usd'].min() if not losing_trades.empty else 0.0

        sum_losses = losing_trades['pnl_net_usd'].sum()
        if sum_losses != 0: # Eviter division par zéro
            metrics["Profit Factor"] = abs(winning_trades['pnl_net_usd'].sum() / sum_losses) if not losing_trades.empty else np.inf
        elif not winning_trades.empty: # Gains mais pas de pertes
             metrics["Profit Factor"] = np.inf
        else: # Ni gains ni pertes
             metrics["Profit Factor"] = 1.0 # Ou np.nan ou 0, selon la convention

        if avg_loss_usd != 0:
            metrics["Average Win / Loss Ratio"] = abs(avg_win_usd / avg_loss_usd) if avg_win_usd != 0 else 0.0
        elif avg_win_usd != 0: # Gains mais pas de pertes
             metrics["Average Win / Loss Ratio"] = np.inf
        else: # Ni gains ni pertes
             metrics["Average Win / Loss Ratio"] = 1.0


        if 'entry_timestamp' in trades_df.columns and 'exit_timestamp' in trades_df.columns:
            # S'assurer que les timestamps sont des datetime
            trades_df_copy = trades_df.copy() # Eviter SettingWithCopyWarning
            trades_df_copy.loc[:, 'entry_timestamp'] = pd.to_datetime(trades_df_copy['entry_timestamp'], errors='coerce')
            trades_df_copy.loc[:, 'exit_timestamp'] = pd.to_datetime(trades_df_copy['exit_timestamp'], errors='coerce')
            valid_durations = trades_df_copy.dropna(subset=['entry_timestamp', 'exit_timestamp'])
            if not valid_durations.empty:
                valid_durations.loc[:, 'duration'] = (valid_durations['exit_timestamp'] - valid_durations['entry_timestamp'])
                metrics["Average Trade Duration"] = str(valid_durations['duration'].mean())
                metrics["Median Trade Duration"] = str(valid_durations['duration'].median())
            else:
                metrics["Average Trade Duration"] = "N/A"
                metrics["Median Trade Duration"] = "N/A"
        else:
            metrics["Average Trade Duration"] = "N/A"
            metrics["Median Trade Duration"] = "N/A"
    else:
        logger.info(f"{log_prefix} Trades DataFrame vide ou 'pnl_net_usd' manquante. Métriques de trade par défaut.")
        metrics["Total Trades"] = 0
        metrics["Win Rate Pct"] = np.nan
        for key in ["Number of Winning Trades", "Number of Losing Trades", "Number of Breakeven Trades",
                    "Total Gross PnL USDC", "Total Commission USDC", "Average Winning Trade USDC",
                    "Average Losing Trade USDC", "Largest Winning Trade USDC", "Largest Losing Trade USDC",
                    "Profit Factor", "Average Win / Loss Ratio", "Average Trade Duration", "Median Trade Duration"]:
            metrics[key] = 0.0 if "Number" in key or "Total" in key else np.nan


    # Max Drawdown
    cumulative_max_equity = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max_equity) / cumulative_max_equity
    max_drawdown_pct = drawdown.min() * 100 # En pourcentage, sera négatif
    metrics["Max Drawdown Pct"] = abs(max_drawdown_pct) if pd.notna(max_drawdown_pct) else np.nan
    
    drawdown_value = equity_curve - cumulative_max_equity
    max_drawdown_usd = drawdown_value.min()
    metrics["Max Drawdown USDC"] = abs(max_drawdown_usd) if pd.notna(max_drawdown_usd) else np.nan

    # Rendements journaliers (ou par période de la courbe d'équité)
    # Assurer que l'index est DatetimeIndex et trié
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        logger.warning(f"{log_prefix} Equity curve index is not DatetimeIndex. Cannot calculate daily returns for Sharpe/Sortino.")
        daily_returns = pd.Series(dtype=float)
    else:
        equity_curve_sorted = equity_curve.sort_index()
        daily_returns = equity_curve_sorted.pct_change().fillna(0)

    # Rendement annualisé
    total_return_strategy = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1 if len(equity_curve) > 0 else 0.0
    num_years = metrics["Duration Days"] / 365.25 if metrics["Duration Days"] > 0 else (1/periods_per_year if periods_per_year > 0 else 1) # Fallback si durée 0
    if num_years > 0:
        annualized_return = ((1 + total_return_strategy) ** (1 / num_years)) - 1
    else: # Si la durée est très courte, l'annualisation peut être trompeuse.
        annualized_return = total_return_strategy * periods_per_year # Simple extrapolation
    metrics["Annualized Return Pct"] = annualized_return * 100

    # Calmar Ratio
    if pd.notna(max_drawdown_pct) and max_drawdown_pct != 0:
        metrics["Calmar Ratio"] = (annualized_return * 100) / abs(max_drawdown_pct) if abs(max_drawdown_pct) > 1e-9 else np.nan
    else:
        metrics["Calmar Ratio"] = np.nan

    # Sharpe Ratio
    if not daily_returns.empty and daily_returns.std() != 0:
        excess_returns = daily_returns - risk_free_rate_daily
        sharpe_ratio_unannualized = excess_returns.mean() / excess_returns.std()
        metrics["Sharpe Ratio"] = sharpe_ratio_unannualized * np.sqrt(periods_per_year) if pd.notna(sharpe_ratio_unannualized) else np.nan
    else:
        metrics["Sharpe Ratio"] = np.nan

    # Sortino Ratio
    if not daily_returns.empty:
        negative_returns = daily_returns[daily_returns < 0]
        downside_std_dev = negative_returns.std() # Ecart-type des rendements négatifs
        if pd.notna(downside_std_dev) and downside_std_dev != 0:
            excess_returns_mean = (daily_returns - risk_free_rate_daily).mean()
            sortino_ratio_unannualized = excess_returns_mean / downside_std_dev
            metrics["Sortino Ratio"] = sortino_ratio_unannualized * np.sqrt(periods_per_year) if pd.notna(sortino_ratio_unannualized) else np.nan
        else:
            metrics["Sortino Ratio"] = np.nan # Ou np.inf si pas de rendements négatifs mais rendement moyen positif
    else:
        metrics["Sortino Ratio"] = np.nan

    # Métriques supplémentaires
    metrics["Peak Equity USDC"] = cumulative_max_equity.max() if not equity_curve.empty else initial_capital
    metrics["Lowest Equity USDC"] = equity_curve.min() if not equity_curve.empty else initial_capital
    
    # Vérifier les valeurs NaN/Inf finales et les remplacer par None ou 0 pour la sérialisation JSON si nécessaire
    for k, v in metrics.items():
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                metrics[k] = None # Ou 0.0 selon la préférence pour l'affichage/stockage

    logger.info(f"{log_prefix} Performance metrics calculated: PnL={metrics.get('Total Net PnL USDC', 'N/A')}, Sharpe={metrics.get('Sharpe Ratio', 'N/A')}")
    return metrics

