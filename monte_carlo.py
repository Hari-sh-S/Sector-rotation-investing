"""
Monte Carlo Simulation Module
==============================
Trade reshuffling simulation to quantify real portfolio risk.
"""

import numpy as np
from typing import List, Dict, Tuple


class MonteCarloSimulator:
    """
    Monte Carlo simulator using trade reshuffling.
    
    This approach randomly reorders the sequence of trade PnLs to simulate
    alternative "what-if" scenarios, revealing the range of possible outcomes.
    """
    
    def __init__(self, trade_pnls: List[float], initial_capital: float, 
                 test_duration_years: float, n_simulations: int = 10000):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            trade_pnls: List of individual trade PnL values (profits and losses)
            initial_capital: Starting capital
            test_duration_years: Duration of the backtest in years (for CAGR calc)
            n_simulations: Number of Monte Carlo simulations to run
        """
        self.trade_pnls = np.array(trade_pnls)
        self.initial_capital = initial_capital
        self.test_duration_years = test_duration_years
        self.n_simulations = n_simulations
        
        # Results storage
        self.results = None
        self._historical_metrics = None
    
    def _compute_historical_metrics(self) -> Dict:
        """Compute metrics from the original (historical) trade sequence."""
        equity = self.initial_capital
        peak = self.initial_capital
        max_dd = 0.0
        current_losing_streak = 0
        max_losing_streak = 0
        
        for pnl in self.trade_pnls:
            equity += pnl
            
            # Update peak (only if equity is positive and above 50% of initial)
            if equity > 0 and equity >= self.initial_capital * 0.5:
                if equity > peak:
                    peak = equity
            
            # Calculate drawdown (cap at 100%)
            if peak > 0:
                dd = ((peak - equity) / peak) * 100
                dd = min(dd, 100.0)  # Cap at 100%
                max_dd = max(max_dd, dd)
            
            # Track losing streak
            if pnl < 0:
                current_losing_streak += 1
                max_losing_streak = max(max_losing_streak, current_losing_streak)
            else:
                current_losing_streak = 0
        
        # Final equity and CAGR
        final_equity = equity
        if self.test_duration_years > 0 and self.initial_capital > 0 and final_equity > 0:
            cagr = ((final_equity / self.initial_capital) ** (1 / self.test_duration_years) - 1) * 100
        else:
            cagr = 0.0
        
        return {
            'max_drawdown': max_dd,
            'max_losing_streak': max_losing_streak,
            'final_equity': final_equity,
            'cagr': cagr
        }
    
    def run_simulations(self, method: str = 'reshuffle') -> Dict:
        """
        Run Monte Carlo simulations.
        
        Args:
            method: 'reshuffle' (permutation) or 'resample' (bootstrap with replacement)
            
        Returns:
            Dictionary with simulation results and statistics
        """
        if len(self.trade_pnls) == 0:
            return self._empty_results()
        
        # Compute historical metrics first
        self._historical_metrics = self._compute_historical_metrics()
        
        # Storage for simulation results
        max_drawdowns = np.zeros(self.n_simulations)
        max_losing_streaks = np.zeros(self.n_simulations, dtype=int)
        final_equities = np.zeros(self.n_simulations)
        cagrs = np.zeros(self.n_simulations)
        ruin_count = 0
        
        # Store sample equity curves for charting (store first 100)
        n_sample_curves = min(100, self.n_simulations)
        sample_equity_curves = []
        
        # Also compute and store historical equity curve
        historical_curve = [self.initial_capital]
        equity_hist = self.initial_capital
        for pnl in self.trade_pnls:
            equity_hist += pnl
            historical_curve.append(equity_hist)
        
        # Run simulations
        for i in range(self.n_simulations):
            # Generate PnL sequence based on method
            if method == 'resample':
                # Bootstrap sampling with replacement
                sim_pnls = np.random.choice(self.trade_pnls, size=len(self.trade_pnls), replace=True)
            else:
                # Reshuffle (Permutation) without replacement
                sim_pnls = np.random.permutation(self.trade_pnls)
            
            # Simulate equity curve
            equity = self.initial_capital
            peak = self.initial_capital
            max_dd = 0.0
            current_losing_streak = 0
            max_losing_streak = 0
            ruin_hit = False
            
            # Track equity curve for sample simulations
            if i < n_sample_curves:
                curve = [self.initial_capital]
            
            for pnl in sim_pnls:
                equity += pnl
                
                # Store for sample curves
                if i < n_sample_curves:
                    curve.append(equity)
                
                # Update peak (only if equity is positive and above ruin threshold)
                if equity > 0 and equity >= self.initial_capital * 0.5:
                    if equity > peak:
                        peak = equity
                
                # Calculate drawdown (cap at 100%)
                if peak > 0:
                    dd = ((peak - equity) / peak) * 100
                    dd = min(dd, 100.0)  # Cap at 100%
                    max_dd = max(max_dd, dd)
                
                # Track losing streak
                if pnl < 0:
                    current_losing_streak += 1
                    max_losing_streak = max(max_losing_streak, current_losing_streak)
                else:
                    current_losing_streak = 0
                
                # Check ruin conditions
                # Ruin = equity < 50% of peak OR equity < starting capital
                if not ruin_hit:
                    if equity < 0.5 * peak or equity < self.initial_capital:
                        ruin_hit = True
            
            # Store sample equity curve
            if i < n_sample_curves:
                sample_equity_curves.append(curve)
            
            # Store results
            max_drawdowns[i] = max_dd
            max_losing_streaks[i] = max_losing_streak
            final_equities[i] = equity
            
            # Calculate CAGR for this simulation
            if self.test_duration_years > 0 and self.initial_capital > 0 and equity > 0:
                cagrs[i] = ((equity / self.initial_capital) ** (1 / self.test_duration_years) - 1) * 100
            else:
                cagrs[i] = -100.0  # Total loss
            
            if ruin_hit:
                ruin_count += 1
        
        # Compute statistics
        self.results = {
            # Max Drawdown Statistics
            'historical_max_dd': self._historical_metrics['max_drawdown'],
            'mc_max_dd_95': np.percentile(max_drawdowns, 95),
            'mc_max_dd_worst': np.max(max_drawdowns),
            'mc_max_dd_median': np.percentile(max_drawdowns, 50),
            
            # Losing Streak Statistics
            'historical_losing_streak': self._historical_metrics['max_losing_streak'],
            'mc_losing_streak_95': int(np.percentile(max_losing_streaks, 95)),
            'mc_losing_streak_worst': int(np.max(max_losing_streaks)),
            
            # Ruin Probability
            'ruin_probability': (ruin_count / self.n_simulations) * 100,
            'ruin_count': ruin_count,
            
            # CAGR Distribution
            'historical_cagr': self._historical_metrics['cagr'],
            'mc_cagr_median': np.percentile(cagrs, 50),
            'mc_cagr_5th': np.percentile(cagrs, 5),
            'mc_cagr_95th': np.percentile(cagrs, 95),
            
            # Additional info
            'n_simulations': self.n_simulations,
            'n_trades': len(self.trade_pnls),
            'initial_capital': self.initial_capital,
            
            # Raw distributions for potential charting
            'max_dd_distribution': max_drawdowns,
            'cagr_distribution': cagrs,
            'losing_streak_distribution': max_losing_streaks,
            
            # Equity curves for visualization
            'sample_equity_curves': sample_equity_curves,
            'historical_equity_curve': historical_curve,
            
            # Metadata
            'method': method,
            'method_name': "Reshuffle (Permutation)" if method == 'reshuffle' else "Resample (Bootstrap)"
        }
        
        return self.results
    
    def _empty_results(self) -> Dict:
        """Return empty results when no trades available."""
        return {
            'historical_max_dd': 0,
            'mc_max_dd_95': 0,
            'mc_max_dd_worst': 0,
            'mc_max_dd_median': 0,
            'historical_losing_streak': 0,
            'mc_losing_streak_95': 0,
            'mc_losing_streak_worst': 0,
            'ruin_probability': 0,
            'ruin_count': 0,
            'historical_cagr': 0,
            'mc_cagr_median': 0,
            'mc_cagr_5th': 0,
            'mc_cagr_95th': 0,
            'n_simulations': 0,
            'n_trades': 0,
            'initial_capital': self.initial_capital,
            'max_dd_distribution': np.array([]),
            'cagr_distribution': np.array([]),
            'losing_streak_distribution': np.array([]),
            'sample_equity_curves': [],
            'historical_equity_curve': []
        }
    
    def get_results(self) -> Dict:
        """Get simulation results. Runs simulation if not already done."""
        if self.results is None:
            self.run_simulations()
        return self.results
    
    def get_interpretation(self) -> Dict[str, str]:
        """
        Generate practical risk interpretations for each metric.
        
        Returns:
            Dictionary mapping metric names to interpretation strings
        """
        if self.results is None:
            self.run_simulations()
        
        r = self.results
        
        interpretations = {}
        
        # Max Drawdown interpretation
        if r['mc_max_dd_95'] > r['historical_max_dd']:
            dd_diff = r['mc_max_dd_95'] - r['historical_max_dd']
            interpretations['max_drawdown'] = (
                f"Historical DD was {r['historical_max_dd']:.1f}%, but 95% of simulations "
                f"show DD up to {r['mc_max_dd_95']:.1f}% (+{dd_diff:.1f}%). "
                f"Worst case: {r['mc_max_dd_worst']:.1f}%. "
                f"You may have been lucky in the historical sequence."
            )
        else:
            interpretations['max_drawdown'] = (
                f"Historical DD of {r['historical_max_dd']:.1f}% was near the 95th percentile "
                f"({r['mc_max_dd_95']:.1f}%). The trade sequence didn't mask significant DD risk."
            )
        
        # Losing Streak interpretation
        if r['mc_losing_streak_95'] > r['historical_losing_streak']:
            interpretations['losing_streak'] = (
                f"Historical worst streak was {r['historical_losing_streak']} losses. "
                f"In 5% of simulations, you could face {r['mc_losing_streak_95']}+ consecutive losses. "
                f"Worst case: {r['mc_losing_streak_worst']} losses in a row."
            )
        else:
            interpretations['losing_streak'] = (
                f"Historical streak of {r['historical_losing_streak']} losses was already "
                f"near the worst case ({r['mc_losing_streak_worst']}). "
                f"The sequence didn't hide streak risk."
            )
        
        # Ruin Probability interpretation
        if r['ruin_probability'] == 0:
            interpretations['ruin'] = (
                f"No simulations hit ruin conditions (equity < 50% peak or < starting capital). "
                f"The strategy appears robust to trade sequence randomization."
            )
        elif r['ruin_probability'] < 5:
            interpretations['ruin'] = (
                f"{r['ruin_probability']:.2f}% of simulations hit ruin. "
                f"Low but non-zero risk of significant capital loss under unlucky sequences."
            )
        else:
            interpretations['ruin'] = (
                f"{r['ruin_probability']:.2f}% of simulations hit ruin ({r['ruin_count']:,} of {r['n_simulations']:,}). "
                f"Significant risk of capital loss if trade sequence differs from historical."
            )
        
        # CAGR interpretation
        cagr_range = r['mc_cagr_95th'] - r['mc_cagr_5th']
        interpretations['cagr'] = (
            f"CAGR ranges from {r['mc_cagr_5th']:.1f}% (bad luck) to {r['mc_cagr_95th']:.1f}% (good luck). "
            f"Median: {r['mc_cagr_median']:.1f}%. Historical: {r['historical_cagr']:.1f}%. "
            f"Spread of {cagr_range:.1f}% shows sequence-dependent performance."
        )
        
        # Add context about method
        method_note = ""
        if r.get('method') == 'resample':
            method_note = " (Resampling allows worst-case scenarios to repeat, showing potentially more severe risks)"
        elif r.get('method') == 'reshuffle':
            method_note = " (Reshuffling rearranges existing trades, preserving the exact PnL set)"
            
        interpretations['method_note'] = method_note
        
        return interpretations


def extract_trade_pnls(trades_df, buy_trades_df=None) -> List[float]:
    """
    Extract individual trade PnLs from trades dataframe.
    
    Args:
        trades_df: DataFrame with trades (Action, Ticker, Date, Price, Shares, Value)
        buy_trades_df: Optional separate buy trades df (if not included in trades_df)
    
    Returns:
        List of trade PnL values
    """
    import pandas as pd
    
    if trades_df is None or trades_df.empty:
        return []
    
    pnls = []
    
    # Get BUY and SELL trades
    buy_trades = trades_df[trades_df['Action'] == 'BUY'].copy()
    sell_trades = trades_df[trades_df['Action'] == 'SELL'].copy()
    
    if sell_trades.empty:
        return []
    
    # Match each SELL with its corresponding BUY
    for _, sell in sell_trades.iterrows():
        ticker = sell['Ticker']
        sell_date = sell['Date']
        sell_price = float(sell['Price'])
        shares = int(sell['Shares'])
        
        # Find the most recent BUY for this ticker before this SELL
        prev_buys = buy_trades[
            (buy_trades['Ticker'] == ticker) & 
            (buy_trades['Date'] < sell_date)
        ]
        
        if not prev_buys.empty:
            buy = prev_buys.iloc[-1]
            buy_price = float(buy['Price'])
            
            # Calculate PnL
            pnl = (sell_price - buy_price) * shares
            pnls.append(pnl)
    
    return pnls


def extract_monthly_returns(trades_df, initial_capital: float = 100000) -> List[float]:
    """
    Extract monthly portfolio returns from trades (GPT's exact logic).
    
    This calculates returns based on trade PnLs grouped by exit month,
    which is the correct approach for rebalancing strategies.
    
    Args:
        trades_df: DataFrame with trade data (must have 'Exit Date', 
                   'Exit Price', 'Buy Price', 'Shares' columns)
        initial_capital: Starting capital
    
    Returns:
        List of monthly returns (e.g., [0.023, -0.041, 0.068])
    """
    import pandas as pd
    
    if trades_df is None or trades_df.empty:
        return []
    
    # Make a copy to avoid modifying original
    df = trades_df.copy()
    
    # Ensure date columns are datetime
    if 'Exit Date' in df.columns:
        df['Exit Date'] = pd.to_datetime(df['Exit Date'])
    else:
        return []
    
    # Calculate trade PnL (GPT's exact formula)
    if 'Exit Price' in df.columns and 'Buy Price' in df.columns and 'Shares' in df.columns:
        df['pnl'] = (df['Exit Price'] - df['Buy Price']) * df['Shares']
    else:
        return []
    
    # Group by exit month
    df['month'] = df['Exit Date'].dt.to_period('M').dt.to_timestamp()
    monthly_pnl = df.groupby('month')['pnl'].sum().sort_index()
    
    # Calculate monthly returns (GPT's exact logic)
    equity = initial_capital
    monthly_returns = []
    
    for pnl in monthly_pnl:
        ret = pnl / equity
        monthly_returns.append(ret)
        equity *= (1 + ret)
    
    return monthly_returns


class PortfolioMonteCarloSimulator:
    """
    Monte Carlo simulator using monthly portfolio returns.
    
    This approach is correct for portfolio systems with:
    - Multiple concurrent positions
    - Inverse volatility / Risk parity weighting
    - Monthly rebalancing
    
    It preserves the diversification and vol-targeting embedded in the returns.
    """
    
    def __init__(self, monthly_returns: List[float], initial_capital: float, 
                 n_simulations: int = 10000):
        """
        Initialize Portfolio Monte Carlo simulator.
        
        Args:
            monthly_returns: List of monthly portfolio returns (e.g., [0.023, -0.041])
            initial_capital: Starting capital
            n_simulations: Number of Monte Carlo simulations to run
        """
        self.monthly_returns = np.array(monthly_returns)
        self.initial_capital = initial_capital
        self.n_simulations = n_simulations
        
        # Results storage
        self.results = None
        self._historical_metrics = None
    
    def _compute_historical_metrics(self) -> Dict:
        """Compute metrics from the original (historical) return sequence."""
        equity = self.initial_capital
        peak = self.initial_capital
        max_dd = 0.0
        current_losing_streak = 0
        max_losing_streak = 0
        
        for ret in self.monthly_returns:
            equity = equity * (1 + ret)
            
            # Update peak
            if equity > peak:
                peak = equity
            
            # Calculate drawdown (cap at 100%)
            if peak > 0:
                dd = ((peak - equity) / peak) * 100
                dd = min(dd, 100.0)
                max_dd = max(max_dd, dd)
            
            # Track losing streak (negative months)
            if ret < 0:
                current_losing_streak += 1
                max_losing_streak = max(max_losing_streak, current_losing_streak)
            else:
                current_losing_streak = 0
        
        # Final equity and CAGR
        final_equity = equity
        years = len(self.monthly_returns) / 12.0
        if years > 0 and self.initial_capital > 0 and final_equity > 0:
            cagr = ((final_equity / self.initial_capital) ** (1 / years) - 1) * 100
        else:
            cagr = 0.0
        
        return {
            'max_drawdown': max_dd,
            'max_losing_streak': max_losing_streak,
            'final_equity': final_equity,
            'cagr': cagr
        }
    
    def run_simulations(self, method: str = 'reshuffle') -> Dict:
        """
        Run Monte Carlo simulations on monthly returns.
        
        Args:
            method: 'reshuffle' (permutation) or 'resample' (bootstrap with replacement)
            
        Returns:
            Dictionary with simulation results and statistics
        """
        if len(self.monthly_returns) == 0:
            return self._empty_results()
        
        # Compute historical metrics first
        self._historical_metrics = self._compute_historical_metrics()
        
        # Storage for simulation results
        max_drawdowns = np.zeros(self.n_simulations)
        max_losing_streaks = np.zeros(self.n_simulations, dtype=int)
        final_equities = np.zeros(self.n_simulations)
        cagrs = np.zeros(self.n_simulations)
        ruin_count = 0
        
        # Store sample equity curves for charting (1000 paths like GPT)
        n_sample_curves = min(1000, self.n_simulations)
        sample_equity_curves = []
        
        # Also compute and store historical equity curve
        historical_curve = [self.initial_capital]
        equity_hist = self.initial_capital
        for ret in self.monthly_returns:
            equity_hist = equity_hist * (1 + ret)
            historical_curve.append(equity_hist)
        
        years = len(self.monthly_returns) / 12.0
        
        # Run simulations
        for i in range(self.n_simulations):
            # Generate return sequence based on method
            if method == 'resample':
                sim_returns = np.random.choice(self.monthly_returns, size=len(self.monthly_returns), replace=True)
            else:
                sim_returns = np.random.permutation(self.monthly_returns)
            
            # Simulate equity curve
            equity = self.initial_capital
            peak = self.initial_capital
            max_dd = 0.0
            current_losing_streak = 0
            max_losing_streak = 0
            ruin_hit = False
            
            # Track equity curve for sample simulations (200 paths like GPT)
            if i < n_sample_curves:
                curve = [self.initial_capital]
            
            for ret in sim_returns:
                equity = equity * (1 + ret)
                
                # Store for sample curves
                if i < n_sample_curves:
                    curve.append(equity)
                
                # Update peak (GPT's simple approach)
                peak = max(peak, equity)
                
                # Calculate drawdown (GPT's simple approach - no artificial cap)
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                
                # Track losing streak
                if ret < 0:
                    current_losing_streak += 1
                    max_losing_streak = max(max_losing_streak, current_losing_streak)
                else:
                    current_losing_streak = 0
                
                # Ruin check (50% drawdown from peak)
                if not ruin_hit and dd >= 0.5:
                    ruin_hit = True
            
            # Store sample equity curve
            if i < n_sample_curves:
                sample_equity_curves.append(curve)
            
            # Store results (DD as percentage)
            max_drawdowns[i] = max_dd * 100  # Convert to percentage
            max_losing_streaks[i] = max_losing_streak
            final_equities[i] = equity
            
            # Calculate CAGR for this simulation
            if years > 0 and self.initial_capital > 0 and equity > 0:
                cagrs[i] = ((equity / self.initial_capital) ** (1 / years) - 1) * 100
            else:
                cagrs[i] = -100.0
            
            if ruin_hit:
                ruin_count += 1
        
        # Compute statistics
        self.results = {
            # Max Drawdown Statistics
            'historical_max_dd': self._historical_metrics['max_drawdown'],
            'mc_max_dd_95': np.percentile(max_drawdowns, 95),
            'mc_max_dd_worst': np.max(max_drawdowns),
            'mc_max_dd_median': np.percentile(max_drawdowns, 50),
            
            # Losing Streak Statistics (months)
            'historical_losing_streak': self._historical_metrics['max_losing_streak'],
            'mc_losing_streak_95': int(np.percentile(max_losing_streaks, 95)),
            'mc_losing_streak_worst': int(np.max(max_losing_streaks)),
            
            # Ruin Probability
            'ruin_probability': (ruin_count / self.n_simulations) * 100,
            'ruin_count': ruin_count,
            
            # CAGR Distribution
            'historical_cagr': self._historical_metrics['cagr'],
            'mc_cagr_median': np.percentile(cagrs, 50),
            'mc_cagr_5th': np.percentile(cagrs, 5),
            'mc_cagr_95th': np.percentile(cagrs, 95),
            
            # Additional info
            'n_simulations': self.n_simulations,
            'n_periods': len(self.monthly_returns),
            'initial_capital': self.initial_capital,
            
            # Raw distributions
            'max_dd_distribution': max_drawdowns,
            'cagr_distribution': cagrs,
            'losing_streak_distribution': max_losing_streaks,
            
            # Equity curves for visualization
            'sample_equity_curves': sample_equity_curves,
            'historical_equity_curve': historical_curve,
            
            # Metadata
            'method': method,
            'method_name': "Reshuffle (Permutation)" if method == 'reshuffle' else "Resample (Bootstrap)",
            'level': 'portfolio'  # Flag to indicate this is portfolio-level MC
        }
        
        return self.results
    
    def _empty_results(self) -> Dict:
        """Return empty results when no returns available."""
        return {
            'historical_max_dd': 0,
            'mc_max_dd_95': 0,
            'mc_max_dd_worst': 0,
            'mc_max_dd_median': 0,
            'historical_losing_streak': 0,
            'mc_losing_streak_95': 0,
            'mc_losing_streak_worst': 0,
            'ruin_probability': 0,
            'ruin_count': 0,
            'historical_cagr': 0,
            'mc_cagr_median': 0,
            'mc_cagr_5th': 0,
            'mc_cagr_95th': 0,
            'n_simulations': 0,
            'n_periods': 0,
            'initial_capital': self.initial_capital,
            'max_dd_distribution': np.array([]),
            'cagr_distribution': np.array([]),
            'losing_streak_distribution': np.array([]),
            'sample_equity_curves': [],
            'historical_equity_curve': [],
            'level': 'portfolio'
        }
    
    def get_results(self) -> Dict:
        """Get simulation results. Runs simulation if not already done."""
        if self.results is None:
            self.run_simulations()
        return self.results
    
    def get_interpretation(self) -> Dict[str, str]:
        """Generate practical risk interpretations for each metric."""
        if self.results is None:
            self.run_simulations()
        
        r = self.results
        
        interpretations = {}
        
        # Max Drawdown interpretation
        if r['mc_max_dd_95'] > r['historical_max_dd']:
            dd_diff = r['mc_max_dd_95'] - r['historical_max_dd']
            interpretations['max_drawdown'] = (
                f"Historical DD was {r['historical_max_dd']:.1f}%, but 95% of simulations "
                f"show DD up to {r['mc_max_dd_95']:.1f}% (+{dd_diff:.1f}%). "
                f"Worst case: {r['mc_max_dd_worst']:.1f}%."
            )
        else:
            interpretations['max_drawdown'] = (
                f"Historical DD of {r['historical_max_dd']:.1f}% was near the 95th percentile "
                f"({r['mc_max_dd_95']:.1f}%). The month sequence didn't mask significant DD risk."
            )
        
        # Losing Streak interpretation
        if r['mc_losing_streak_95'] > r['historical_losing_streak']:
            interpretations['losing_streak'] = (
                f"Historical worst streak was {r['historical_losing_streak']} losing months. "
                f"In 5% of simulations, you could face {r['mc_losing_streak_95']}+ consecutive red months. "
                f"Worst case: {r['mc_losing_streak_worst']} months in a row."
            )
        else:
            interpretations['losing_streak'] = (
                f"Historical streak of {r['historical_losing_streak']} losing months was already "
                f"near the worst case ({r['mc_losing_streak_worst']}). "
                f"The sequence didn't hide streak risk."
            )
        
        # Ruin Probability interpretation
        if r['ruin_probability'] == 0:
            interpretations['ruin'] = (
                f"No simulations hit ruin conditions (equity < 50% peak or < starting capital). "
                f"The strategy appears robust to return sequence randomization."
            )
        elif r['ruin_probability'] < 5:
            interpretations['ruin'] = (
                f"{r['ruin_probability']:.2f}% of simulations hit ruin. "
                f"Low but non-zero risk of significant capital loss under unlucky sequences."
            )
        else:
            interpretations['ruin'] = (
                f"{r['ruin_probability']:.2f}% of simulations hit ruin ({r['ruin_count']:,} of {r['n_simulations']:,}). "
                f"Moderate risk of capital loss if return sequence differs from historical."
            )
        
        # CAGR interpretation
        cagr_range = r['mc_cagr_95th'] - r['mc_cagr_5th']
        interpretations['cagr'] = (
            f"CAGR ranges from {r['mc_cagr_5th']:.1f}% (bad luck) to {r['mc_cagr_95th']:.1f}% (good luck). "
            f"Median: {r['mc_cagr_median']:.1f}%. Historical: {r['historical_cagr']:.1f}%. "
            f"Spread of {cagr_range:.1f}% shows sequence-dependent performance."
        )
        
        # Method note
        method_note = ""
        if r.get('method') == 'resample':
            method_note = " (Resampling allows bad months to repeat, showing potentially more severe risks)"
        elif r.get('method') == 'reshuffle':
            method_note = " (Reshuffling rearranges existing months, preserving the exact return set)"
            
        interpretations['method_note'] = method_note
        
        return interpretations
