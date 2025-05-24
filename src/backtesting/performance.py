import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def calculate_performance_metrics_from_inputs( # Renamed to avoid conflict with class method
    trades_df: pd.DataFrame,
    equity_curve_series: pd.Series, # Expects a Series now
    initial_capital: float,
    risk_free_rate_daily: float = 0.0, # Taux journalier
    periods_per_year: int = 252 # Pour annualiser Sharpe. Ex: 252 pour jours de trading, 365 si calendaire, 52 si hebdo.
) -> Dict[str, Any]:
    """
    Calcule diverses métriques de performance à partir d'un journal de trades et d'une courbe d'équité.

    Args:
        trades_df (pd.DataFrame): DataFrame des trades. Doit contenir au moins 'pnl_net_usd'.
                                  Attend également 'entry_timestamp' et 'exit_timestamp' pour la durée des trades.
        equity_curve_series (pd.Series): Série pandas de la valeur du portefeuille au fil du temps, indexée par timestamp.
        initial_capital (float): Capital initial de la simulation.
        risk_free_rate_daily (float): Taux sans risque journalier pour le calcul du ratio de Sharpe.
        periods_per_year (int): Nombre de périodes de trading dans une année pour l'annualisation.

    Returns:
        Dict[str, Any]: Un dictionnaire contenant les métriques de performance calculées.
    """
    metrics: Dict[str, Any] = {}
    log_prefix = "[PerfCalcFunc]" # Changed prefix for clarity

    # Validation des entrées
    if not isinstance(equity_curve_series, pd.Series) or equity_curve_series.empty:
        logger.warning(f"{log_prefix} Equity curve series is empty or not a Series. Cannot calculate most metrics.")
        metrics["Initial Capital USDC"] = initial_capital
        metrics["Final Equity USDC"] = initial_capital
        metrics["Total Net PnL USDC"] = 0.0
        metrics["Total Net PnL Pct"] = 0.0
        metrics["Total Trades"] = 0
        for key in ["Start Date", "End Date", "Duration Days", "Duration", "Win Rate Pct",
                    "Max Drawdown Pct", "Max Drawdown USDC", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                    "Profit Factor", "Average Win / Loss Ratio", "Annualized Return Pct",
                    "Peak Equity USDC", "Lowest Equity USDC",
                    "Number of Winning Trades", "Number of Losing Trades", "Number of Breakeven Trades",
                    "Average Winning Trade USDC", "Average Losing Trade USDC",
                    "Largest Winning Trade USDC", "Largest Losing Trade USDC",
                    "Average Trade Duration", "Median Trade Duration"]:
            metrics[key] = np.nan if "Pct" in key or "Ratio" in key or "USDC" in key and "Capital" not in key else ("N/A" if "Date" in key or "Duration" in key else 0)
        return metrics

    if not isinstance(trades_df, pd.DataFrame):
        logger.warning(f"{log_prefix} Trades DataFrame is not a DataFrame. Trade-based metrics will be 0 or NaN.")
        trades_df = pd.DataFrame()

    # Période de la simulation
    metrics["Start Date"] = equity_curve_series.index.min().isoformat() if isinstance(equity_curve_series.index, pd.DatetimeIndex) else "N/A"
    metrics["End Date"] = equity_curve_series.index.max().isoformat() if isinstance(equity_curve_series.index, pd.DatetimeIndex) else "N/A"
    
    duration_days_calc = np.nan
    if isinstance(equity_curve_series.index, pd.DatetimeIndex) and len(equity_curve_series.index) > 1:
        duration = equity_curve_series.index.max() - equity_curve_series.index.min()
        duration_days_calc = duration.total_seconds() / (24 * 60 * 60) # More precise than .days for sub-day durations
        metrics["Duration Days"] = duration_days_calc
        metrics["Duration"] = str(duration)
    else:
        metrics["Duration Days"] = 0.0
        metrics["Duration"] = "N/A"


    # Métriques de base PnL
    metrics["Initial Capital USDC"] = initial_capital
    metrics["Final Equity USDC"] = equity_curve_series.iloc[-1]
    total_net_pnl = metrics["Final Equity USDC"] - initial_capital
    metrics["Total Net PnL USDC"] = total_net_pnl
    metrics["Total Net PnL Pct"] = (total_net_pnl / initial_capital) * 100 if initial_capital != 0 else 0.0

    # Métriques basées sur les trades
    if not trades_df.empty and 'pnl_net_quote' in trades_df.columns: # Assuming pnl_net_quote is the primary PnL column
        metrics["Total Trades"] = len(trades_df)

        winning_trades = trades_df[trades_df['pnl_net_quote'] > 0]
        losing_trades = trades_df[trades_df['pnl_net_quote'] < 0]
        breakeven_trades = trades_df[trades_df['pnl_net_quote'] == 0]

        metrics["Number of Winning Trades"] = len(winning_trades)
        metrics["Number of Losing Trades"] = len(losing_trades)
        metrics["Number of Breakeven Trades"] = len(breakeven_trades)

        metrics["Win Rate Pct"] = (len(winning_trades) / metrics["Total Trades"]) * 100 if metrics["Total Trades"] > 0 else 0.0
        metrics["Loss Rate Pct"] = (len(losing_trades) / metrics["Total Trades"]) * 100 if metrics["Total Trades"] > 0 else 0.0
        
        # Assuming 'entry_fee_quote' and 'exit_fee_quote' exist
        total_fees = 0.0
        if 'entry_fee_quote' in trades_df.columns: total_fees += trades_df['entry_fee_quote'].sum()
        if 'exit_fee_quote' in trades_df.columns: total_fees += trades_df['exit_fee_quote'].sum()
        metrics["Total Commission USDC"] = total_fees # Assuming quote is USDC

        # Gross PnL can be derived if not directly present
        if 'pnl_net_quote' in trades_df.columns:
             metrics["Total Gross PnL USDC"] = trades_df['pnl_net_quote'].sum() + total_fees
        else:
             metrics["Total Gross PnL USDC"] = np.nan


        avg_win_usd = winning_trades['pnl_net_quote'].mean() if not winning_trades.empty else 0.0
        avg_loss_usd = losing_trades['pnl_net_quote'].mean() if not losing_trades.empty else 0.0
        metrics["Average Winning Trade USDC"] = avg_win_usd
        metrics["Average Losing Trade USDC"] = avg_loss_usd

        metrics["Largest Winning Trade USDC"] = winning_trades['pnl_net_quote'].max() if not winning_trades.empty else 0.0
        metrics["Largest Losing Trade USDC"] = losing_trades['pnl_net_quote'].min() if not losing_trades.empty else 0.0

        sum_winning_pnl = winning_trades['pnl_net_quote'].sum()
        sum_losing_pnl_abs = abs(losing_trades['pnl_net_quote'].sum())

        if sum_losing_pnl_abs > 1e-9: # Avoid division by zero or near-zero
            metrics["Profit Factor"] = sum_winning_pnl / sum_losing_pnl_abs
        elif sum_winning_pnl > 0: # Wins but no losses
            metrics["Profit Factor"] = np.inf
        else: # No wins and no losses, or only losses with zero sum (unlikely)
            metrics["Profit Factor"] = 1.0 # Or np.nan if preferred

        if abs(avg_loss_usd) > 1e-9:
            metrics["Average Win / Loss Ratio"] = abs(avg_win_usd / avg_loss_usd) if avg_win_usd != 0 else 0.0
        elif avg_win_usd != 0:
            metrics["Average Win / Loss Ratio"] = np.inf
        else:
            metrics["Average Win / Loss Ratio"] = 1.0

        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            trades_df_copy = trades_df.copy()
            trades_df_copy['entry_time'] = pd.to_datetime(trades_df_copy['entry_time'], errors='coerce', utc=True)
            trades_df_copy['exit_time'] = pd.to_datetime(trades_df_copy['exit_time'], errors='coerce', utc=True)
            valid_durations = trades_df_copy.dropna(subset=['entry_time', 'exit_time'])
            if not valid_durations.empty:
                valid_durations['duration'] = (valid_durations['exit_time'] - valid_durations['entry_time'])
                metrics["Average Trade Duration"] = str(valid_durations['duration'].mean())
                metrics["Median Trade Duration"] = str(valid_durations['duration'].median())
            else:
                metrics["Average Trade Duration"] = "N/A"; metrics["Median Trade Duration"] = "N/A"
        else:
            metrics["Average Trade Duration"] = "N/A"; metrics["Median Trade Duration"] = "N/A"
    else:
        logger.info(f"{log_prefix} Trades DataFrame vide ou 'pnl_net_quote' manquante. Métriques de trade par défaut.")
        metrics["Total Trades"] = 0; metrics["Win Rate Pct"] = 0.0
        for key in ["Number of Winning Trades", "Number of Losing Trades", "Number of Breakeven Trades",
                    "Total Gross PnL USDC", "Total Commission USDC", "Average Winning Trade USDC",
                    "Average Losing Trade USDC", "Largest Winning Trade USDC", "Largest Losing Trade USDC",
                    "Profit Factor", "Average Win / Loss Ratio", "Average Trade Duration", "Median Trade Duration"]:
            metrics[key] = 0.0 if "Number" in key or "Total" in key else np.nan

    # Max Drawdown
    if not equity_curve_series.empty:
        cumulative_max_equity = equity_curve_series.cummax()
        drawdown_val = (equity_curve_series - cumulative_max_equity)
        metrics["Max Drawdown USDC"] = abs(drawdown_val.min()) if pd.notna(drawdown_val.min()) else 0.0
        
        drawdown_pct = drawdown_val / cumulative_max_equity
        # Remplacer les infinis (si cumulative_max_equity était 0 à un point) par NaN avant min()
        drawdown_pct.replace([np.inf, -np.inf], np.nan, inplace=True)
        metrics["Max Drawdown Pct"] = abs(drawdown_pct.min() * 100) if pd.notna(drawdown_pct.min()) else 0.0
    else:
        metrics["Max Drawdown USDC"] = 0.0; metrics["Max Drawdown Pct"] = 0.0

    # Rendements journaliers
    daily_returns = pd.Series(dtype=float) # Initialize as empty
    if isinstance(equity_curve_series.index, pd.DatetimeIndex) and len(equity_curve_series) > 1:
        equity_curve_sorted = equity_curve_series.sort_index()
        # Resample to daily, then calculate pct_change. Use 'last' to get end-of-day equity.
        daily_equity = equity_curve_sorted.resample('D').last()
        daily_returns = daily_equity.pct_change().fillna(0)
    elif len(equity_curve_series) > 1 : # If not datetimeindex, but has data, try simple pct_change
        daily_returns = equity_curve_series.pct_change().fillna(0)


    # Rendement annualisé
    total_return_strategy = (equity_curve_series.iloc[-1] / initial_capital) - 1 if not equity_curve_series.empty else 0.0
    num_years = duration_days_calc / 365.25 if pd.notna(duration_days_calc) and duration_days_calc > 0 else (1/periods_per_year if periods_per_year > 0 else 1)
    
    if num_years > 1e-9: # Avoid division by zero or very small num_years issues
        annualized_return = ((1 + total_return_strategy) ** (1 / num_years)) - 1
    elif total_return_strategy != 0 and periods_per_year > 0 : # Extrapolate for very short periods if there was a return
        annualized_return = total_return_strategy * periods_per_year
    else: # No return or no valid period for annualization
        annualized_return = 0.0
    metrics["Annualized Return Pct"] = annualized_return * 100

    # Calmar Ratio
    if pd.notna(metrics.get("Max Drawdown Pct")) and metrics["Max Drawdown Pct"] > 1e-9 : # type: ignore
        metrics["Calmar Ratio"] = metrics["Annualized Return Pct"] / metrics["Max Drawdown Pct"] # type: ignore
    else:
        metrics["Calmar Ratio"] = np.nan if metrics["Annualized Return Pct"] != 0 else 0.0


    # Sharpe Ratio
    if not daily_returns.empty and daily_returns.std() > 1e-9:
        excess_returns = daily_returns - risk_free_rate_daily
        sharpe_ratio_unannualized = excess_returns.mean() / daily_returns.std() # Use std of daily_returns, not excess_returns for classic Sharpe
        metrics["Sharpe Ratio"] = sharpe_ratio_unannualized * np.sqrt(periods_per_year) if pd.notna(sharpe_ratio_unannualized) else np.nan
    else:
        metrics["Sharpe Ratio"] = 0.0 if not daily_returns.empty and daily_returns.mean() == risk_free_rate_daily else np.nan


    # Sortino Ratio
    if not daily_returns.empty:
        target_return_daily = risk_free_rate_daily # Ou 0 si on compare par rapport à ne rien perdre
        negative_excess_returns = (daily_returns - target_return_daily)[(daily_returns - target_return_daily) < 0]
        downside_std_dev = np.sqrt((negative_excess_returns**2).mean()) if not negative_excess_returns.empty else 0.0

        if downside_std_dev > 1e-9:
            excess_returns_mean_for_sortino = (daily_returns - target_return_daily).mean()
            sortino_ratio_unannualized = excess_returns_mean_for_sortino / downside_std_dev
            metrics["Sortino Ratio"] = sortino_ratio_unannualized * np.sqrt(periods_per_year) if pd.notna(sortino_ratio_unannualized) else np.nan
        elif (daily_returns - target_return_daily).mean() > 0 : # Positive returns, no downside deviation
            metrics["Sortino Ratio"] = np.inf
        else: # No returns or no downside deviation and no positive returns
            metrics["Sortino Ratio"] = 0.0 if (daily_returns - target_return_daily).mean() == 0 else np.nan
    else:
        metrics["Sortino Ratio"] = np.nan


    metrics["Peak Equity USDC"] = cumulative_max_equity.max() if not equity_curve_series.empty else initial_capital
    metrics["Lowest Equity USDC"] = equity_curve_series.min() if not equity_curve_series.empty else initial_capital

    for k, v in metrics.items():
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                metrics[k] = None
    logger.info(f"{log_prefix} Performance metrics calculated. PnL={metrics.get('Total Net PnL USDC', 'N/A')}, Sharpe={metrics.get('Sharpe Ratio', 'N/A')}")
    return metrics


class PerformanceCalculator:
    """
    Calculates and stores performance metrics for a backtest.
    """
    def __init__(self,
                 trades: List[Dict[str, Any]],
                 equity_curve: pd.DataFrame, # DataFrame with 'timestamp', 'equity'
                 daily_equity_values: List[float], # This seems redundant if equity_curve is provided
                 initial_capital: float,
                 risk_free_rate: float = 0.0, # Annualized risk-free rate
                 benchmark_returns: Optional[pd.Series] = None, # Not used yet
                 periods_per_year: int = 252):
        """
        Initializes the PerformanceCalculator.

        Args:
            trades (List[Dict[str, Any]]): List of trade dictionaries.
            equity_curve (pd.DataFrame): DataFrame with 'timestamp' and 'equity' columns.
            daily_equity_values (List[float]): List of daily equity values. (Consider deriving from equity_curve).
            initial_capital (float): Starting capital.
            risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%).
            benchmark_returns (Optional[pd.Series]): Series of benchmark returns (not currently used).
            periods_per_year (int): Number of trading periods in a year for annualization.
        """
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        if not isinstance(equity_curve, pd.DataFrame) or equity_curve.empty or \
           'timestamp' not in equity_curve.columns or 'equity' not in equity_curve.columns:
            logger.warning("PerformanceCalculator: Invalid or empty equity_curve DataFrame. Using initial capital as equity series.")
            self.equity_series = pd.Series([initial_capital], index=[pd.Timestamp.now(tz='UTC')]) # Fallback
        else:
            ec_df = equity_curve.copy()
            ec_df['timestamp'] = pd.to_datetime(ec_df['timestamp'], errors='coerce', utc=True)
            ec_df.dropna(subset=['timestamp', 'equity'], inplace=True)
            ec_df.sort_values(by='timestamp', inplace=True)
            self.equity_series = ec_df.set_index('timestamp')['equity']
            if self.equity_series.empty and not equity_curve.empty: # If all rows dropped
                 logger.warning("PerformanceCalculator: Equity series became empty after processing. Using initial capital.")
                 self.equity_series = pd.Series([initial_capital], index=[pd.Timestamp.now(tz='UTC')])


        self.initial_capital = initial_capital
        self.annual_risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.daily_risk_free_rate = (1 + self.annual_risk_free_rate)**(1/self.periods_per_year) - 1 if self.periods_per_year > 0 else 0.0
        
        self.metrics: Dict[str, Any] = {}
        logger.debug("PerformanceCalculator initialized.")

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculates all defined performance metrics.
        """
        self.metrics = calculate_performance_metrics_from_inputs(
            trades_df=self.trades_df,
            equity_curve_series=self.equity_series,
            initial_capital=self.initial_capital,
            risk_free_rate_daily=self.daily_risk_free_rate,
            periods_per_year=self.periods_per_year
        )
        return self.metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the calculated metrics."""
        if not self.metrics:
            self.calculate_all_metrics()
        return self.metrics

    def get_metric(self, metric_name: str) -> Optional[Any]:
        """Returns a specific metric by name."""
        if not self.metrics:
            self.calculate_all_metrics()
        return self.metrics.get(metric_name)

