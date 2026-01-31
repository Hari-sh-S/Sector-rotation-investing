"""
Multi-Asset Rotation Portfolio Engine
Supports two modes:
1. Asset Class Rotation - Rotate between 6 asset classes (ETF or Stock mode)
2. Sector Rotation - Rotate between sectoral/thematic indices
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import timedelta
from indicators import IndicatorLibrary
from scoring import ScoreParser
from indices_universe import (
    ASSET_CLASS_INDICES, ASSET_CLASS_ORDER, SECTORAL_INDICES, THEMATIC_INDICES,
    get_yahoo_ticker, get_etf, get_stock_count, is_equity_asset, get_nse_name
)


class DataCache:
    """Efficient Parquet-based cache for stock data."""
    
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, ticker):
        """Generate cache file path."""
        safe_ticker = ticker.replace("^", "_").replace("=", "_")
        return self.cache_dir / f"{safe_ticker}.parquet"
    
    def get(self, ticker):
        """Retrieve cached data if available."""
        path = self._get_cache_path(ticker)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                return df
            except Exception as e:
                print(f"Cache read error for {ticker}: {e}")
        return None
    
    def set(self, ticker, data):
        """Store data in cache as Parquet."""
        if data is not None and not data.empty:
            path = self._get_cache_path(ticker)
            try:
                data.to_parquet(path)
            except Exception as e:
                print(f"Cache write error for {ticker}: {e}")
    
    def exists(self, ticker):
        """Check if ticker data exists in cache."""
        return self._get_cache_path(ticker).exists()


class RotationEngine:
    """
    Multi-Asset Rotation Backtesting Engine
    
    Modes:
    - 'asset_class': Rotate between 6 asset classes
    - 'sector': Rotate between sectoral or thematic indices
    """
    
    def __init__(self, mode, start_date, end_date, initial_capital=200000, use_cache=True):
        """
        Initialize rotation engine.
        
        Args:
            mode: 'asset_class' or 'sector'
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital (default 200000)
            use_cache: Whether to use local cache
        """
        self.mode = mode
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.cache = DataCache() if use_cache else None
        
        # Data storage
        self.index_data = {}  # Index price data for scoring
        self.stock_data = {}  # Individual stock data
        self.etf_data = {}    # ETF price data
        
        # Results
        self.trades = []
        self.portfolio_history = []
        self.metrics = {}
        
        # Index regime filter data
        self.regime_index_data = None
        
        self.parser = ScoreParser()
    
    @staticmethod
    def _get_scalar(value):
        """Safely extract scalar from potential Series or DataFrame."""
        if hasattr(value, 'iloc'):
            return float(value.iloc[0])
        return float(value) if value is not None else 0.0
    
    def fetch_index_data(self, indices, progress_callback=None):
        """
        Fetch historical data for a list of indices.
        
        Args:
            indices: List of index names
            progress_callback: Optional callback(current, total, ticker)
        """
        print(f"Fetching data for {len(indices)} indices...")
        extended_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=400)
        
        for i, index_name in enumerate(indices):
            if progress_callback:
                progress_callback(i + 1, len(indices), index_name)
            
            ticker = get_yahoo_ticker(index_name)
            if ticker is None:
                print(f"No Yahoo ticker mapping for {index_name}, skipping...")
                continue
            
            # Try cache first
            if self.cache:
                cached = self.cache.get(f"INDEX_{ticker}")
                if cached is not None:
                    try:
                        if 'Date' in cached.columns:
                            cached['Date'] = pd.to_datetime(cached['Date'])
                            cached.set_index('Date', inplace=True)
                        if not isinstance(cached.index, pd.DatetimeIndex):
                            cached.index = pd.to_datetime(cached.index)
                        
                        mask = (cached.index >= extended_start) & (cached.index <= pd.Timestamp(self.end_date))
                        df_filtered = cached[mask].copy()
                        
                        if not df_filtered.empty and len(df_filtered) >= 50:
                            self.index_data[index_name] = df_filtered
                            continue
                    except Exception as e:
                        print(f"Cache error for {index_name}: {e}")
            
            # Download from Yahoo Finance
            try:
                df = yf.download(ticker, start=extended_start, end=self.end_date, progress=False)
                if df.empty:
                    print(f"No data for {index_name} ({ticker})")
                    continue
                
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Calculate indicators
                df = IndicatorLibrary.add_momentum_volatility_metrics(df)
                df = IndicatorLibrary.add_regime_filters(df)
                
                self.index_data[index_name] = df
                
                # Cache the data
                if self.cache:
                    self.cache.set(f"INDEX_{ticker}", df)
                    
            except Exception as e:
                print(f"Error fetching {index_name}: {e}")
        
        print(f"Successfully loaded {len(self.index_data)} indices")
        return len(self.index_data) > 0
    
    def fetch_etf_data(self, etfs, progress_callback=None):
        """
        Fetch historical data for ETFs.
        
        Args:
            etfs: List of ETF tickers
            progress_callback: Optional callback
        """
        print(f"Fetching data for {len(etfs)} ETFs...")
        extended_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=100)
        
        for i, etf in enumerate(etfs):
            if progress_callback:
                progress_callback(i + 1, len(etfs), etf)
            
            # Add .NS suffix for NSE ETFs
            ticker = f"{etf}.NS"
            
            # Try cache first
            if self.cache:
                cached = self.cache.get(f"ETF_{etf}")
                if cached is not None:
                    try:
                        if 'Date' in cached.columns:
                            cached['Date'] = pd.to_datetime(cached['Date'])
                            cached.set_index('Date', inplace=True)
                        if not isinstance(cached.index, pd.DatetimeIndex):
                            cached.index = pd.to_datetime(cached.index)
                        
                        mask = (cached.index >= extended_start) & (cached.index <= pd.Timestamp(self.end_date))
                        df_filtered = cached[mask].copy()
                        
                        if not df_filtered.empty and len(df_filtered) >= 30:
                            self.etf_data[etf] = df_filtered
                            continue
                    except Exception as e:
                        print(f"Cache error for {etf}: {e}")
            
            # Download from Yahoo Finance
            try:
                df = yf.download(ticker, start=extended_start, end=self.end_date, progress=False)
                if df.empty:
                    print(f"No data for {etf} ({ticker})")
                    continue
                
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                self.etf_data[etf] = df
                
                # Cache the data
                if self.cache:
                    self.cache.set(f"ETF_{etf}", df)
                    
            except Exception as e:
                print(f"Error fetching {etf}: {e}")
        
        print(f"Successfully loaded {len(self.etf_data)} ETFs")
        return len(self.etf_data) > 0
    
    def fetch_stock_data(self, stocks, progress_callback=None):
        """
        Fetch historical data for individual stocks.
        
        Args:
            stocks: List of stock symbols
            progress_callback: Optional callback
        """
        print(f"Fetching data for {len(stocks)} stocks...")
        extended_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=400)
        
        for i, stock in enumerate(stocks):
            if progress_callback:
                progress_callback(i + 1, len(stocks), stock)
            
            # Add .NS suffix for NSE stocks
            ticker = f"{stock}.NS"
            
            # Try cache first
            if self.cache:
                cached = self.cache.get(stock)
                if cached is not None:
                    try:
                        if 'Date' in cached.columns:
                            cached['Date'] = pd.to_datetime(cached['Date'])
                            cached.set_index('Date', inplace=True)
                        if not isinstance(cached.index, pd.DatetimeIndex):
                            cached.index = pd.to_datetime(cached.index)
                        
                        # Remove duplicate indices
                        if cached.index.duplicated().any():
                            cached = cached[~cached.index.duplicated(keep='last')]
                        
                        mask = (cached.index >= extended_start) & (cached.index <= pd.Timestamp(self.end_date))
                        df_filtered = cached[mask].copy()
                        
                        if not df_filtered.empty and len(df_filtered) >= 100:
                            self.stock_data[stock] = df_filtered
                            continue
                    except Exception as e:
                        print(f"Cache error for {stock}: {e}")
            
            # Download from Yahoo Finance
            try:
                df = yf.download(ticker, start=extended_start, end=self.end_date, progress=False)
                if df.empty:
                    print(f"No data for {stock} ({ticker})")
                    continue
                
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Calculate indicators
                df = IndicatorLibrary.add_momentum_volatility_metrics(df)
                
                self.stock_data[stock] = df
                
                # Cache the data
                if self.cache:
                    self.cache.set(stock, df)
                    
            except Exception as e:
                print(f"Error fetching {stock}: {e}")
        
        print(f"Successfully loaded {len(self.stock_data)} stocks")
        return len(self.stock_data) > 0
    
    def calculate_indicators(self, index_formula, stock_formula=None, regime_config=None):
        """
        Calculate indicators needed for scoring formulas.
        
        Args:
            index_formula: Scoring formula for indices
            stock_formula: Scoring formula for stocks (if Stock mode)
            regime_config: Regime filter configuration
        """
        # Extract required periods from formulas
        index_periods = self.parser.extract_required_periods(index_formula)
        stock_periods = self.parser.extract_required_periods(stock_formula) if stock_formula else set()
        all_periods = index_periods | stock_periods
        
        # Calculate indicators for indices
        for index_name, df in self.index_data.items():
            try:
                # Flatten columns if needed
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Add momentum metrics if needed
                if '6 Month Performance' not in df.columns:
                    df = IndicatorLibrary.add_momentum_volatility_metrics(df, all_periods)
                
                # Add regime filters
                if 'EMA_200' not in df.columns:
                    df = IndicatorLibrary.add_regime_filters(df)
                
                self.index_data[index_name] = df
            except Exception as e:
                print(f"Error calculating indicators for {index_name}: {e}")
        
        # Calculate indicators for stocks
        for stock, df in self.stock_data.items():
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                if '6 Month Performance' not in df.columns:
                    df = IndicatorLibrary.add_momentum_volatility_metrics(df, stock_periods)
                
                self.stock_data[stock] = df
            except Exception as e:
                print(f"Error calculating indicators for {stock}: {e}")
    
    def score_indices(self, date, formula, indices_list):
        """
        Score a list of indices on a given date using the formula.
        
        Args:
            date: Date to score on
            formula: Scoring formula
            indices_list: List of index names to score
            
        Returns:
            List of (index_name, score) tuples sorted by score descending
        """
        scores = []
        
        for index_name in indices_list:
            if index_name not in self.index_data:
                continue
            
            df = self.index_data[index_name]
            
            # Get data for the date (or nearest prior date)
            if date not in df.index:
                nearest = df.index.asof(date)
                if pd.isna(nearest):
                    continue
                row = df.loc[nearest]
            else:
                row = df.loc[date]
            
            # Calculate score using formula
            try:
                score = self.parser.parse_and_calculate(formula, row)
                if score > -999999:  # Valid score
                    scores.append((index_name, score))
            except Exception as e:
                print(f"Error scoring {index_name}: {e}")
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def score_stocks(self, date, formula, stocks_list):
        """
        Score a list of stocks on a given date using the formula.
        
        Returns:
            List of (stock, score) tuples sorted by score descending
        """
        scores = []
        
        for stock in stocks_list:
            if stock not in self.stock_data:
                continue
            
            df = self.stock_data[stock]
            
            if date not in df.index:
                nearest = df.index.asof(date)
                if pd.isna(nearest):
                    continue
                row = df.loc[nearest]
            else:
                row = df.loc[date]
            
            try:
                score = self.parser.parse_and_calculate(formula, row)
                if score > -999999:
                    scores.append((stock, score))
            except Exception as e:
                print(f"Error scoring {stock}: {e}")
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _get_rebalance_dates(self, all_dates, rebal_config):
        """Generate rebalance dates based on config."""
        freq = rebal_config['frequency']
        
        if freq == 'Weekly':
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
            target_day = day_map[rebal_config.get('day', 'Monday')]
            rebalance_dates = [d for d in all_dates if d.weekday() == target_day]
        
        elif freq == 'Every 2 Weeks':
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
            target_day = day_map[rebal_config.get('day', 'Monday')]
            matching_dates = [d for d in all_dates if d.weekday() == target_day]
            rebalance_dates = matching_dates[::2]
        
        else:  # Monthly and above
            target_date = rebal_config.get('date', 1)
            alt_option = rebal_config.get('alt_day', 'Next Day')
            
            freq_to_skip = {
                'Monthly': 1,
                'Bi-Monthly': 2,
                'Quarterly': 3,
                'Half-Yearly': 6,
                'Annually': 12
            }
            skip_months = freq_to_skip.get(freq, 1)
            
            rebalance_dates = []
            month_groups = {}
            for date in all_dates:
                key = (date.year, date.month)
                if key not in month_groups:
                    month_groups[key] = []
                month_groups[key].append(date)
            
            sorted_months = sorted(month_groups.keys())
            selected_months = sorted_months[::skip_months]
            
            for (year, month) in selected_months:
                month_dates_sorted = sorted(month_groups[(year, month)])
                rebalance_date = None
                
                for d in month_dates_sorted:
                    if d.day == target_date:
                        rebalance_date = d
                        break
                
                if rebalance_date is None:
                    if alt_option == 'Previous Day':
                        for d in reversed(month_dates_sorted):
                            if d.day < target_date:
                                rebalance_date = d
                                break
                        if rebalance_date is None and month_dates_sorted:
                            rebalance_date = month_dates_sorted[0]
                    else:  # Next Day
                        for d in month_dates_sorted:
                            if d.day > target_date:
                                rebalance_date = d
                                break
                        if rebalance_date is None and month_dates_sorted:
                            rebalance_date = month_dates_sorted[-1]
                
                if rebalance_date:
                    rebalance_dates.append(rebalance_date)
        
        print(f"Generated {len(rebalance_dates)} rebalance dates ({freq})")
        return sorted(rebalance_dates)
    
    def _check_regime_filter(self, date, regime_config, target_indices):
        """
        Check if regime filter is triggered for any of the target indices.
        
        Args:
            date: Current date
            regime_config: Regime filter configuration
            target_indices: List of index names to check
            
        Returns:
            (triggered: bool, action: str)
        """
        if not regime_config or not regime_config.get('enabled'):
            return False, 'none'
        
        regime_type = regime_config.get('type', 'EMA_1D')
        action = regime_config.get('action', 'Go Cash')
        
        # Check each target index
        for index_name in target_indices:
            if index_name not in self.index_data:
                continue
            
            df = self.index_data[index_name]
            
            if date not in df.index:
                nearest = df.index.asof(date)
                if pd.isna(nearest):
                    continue
                row = df.loc[nearest]
            else:
                row = df.loc[date]
            
            # Helper to extract scalar
            def get_scalar(val):
                if hasattr(val, 'iloc'):
                    return float(val.iloc[0])
                return float(val) if val is not None else 0.0
            
            triggered = False
            
            if regime_type in ['SMA_1D', 'SMA_1W', 'SMA_1M']:
                direction_col = f'{regime_type}_Direction'
                direction = row.get(direction_col, 'BUY')
                if hasattr(direction, 'iloc'):
                    direction = direction.iloc[0]
                triggered = direction == 'SELL'
            
            elif regime_type in ['EMA_1D', 'EMA_1W', 'EMA_1M']:
                direction_col = f'{regime_type}_Direction'
                direction = row.get(direction_col, 'BUY')
                if hasattr(direction, 'iloc'):
                    direction = direction.iloc[0]
                triggered = direction == 'SELL'
            
            elif regime_type == 'MACD':
                macd_val = get_scalar(row.get('MACD', 0))
                signal_val = get_scalar(row.get('MACD_Signal', 0))
                triggered = macd_val < signal_val
            
            elif regime_type in ['SUPERTREND_1D', 'SUPERTREND_1W', 'SUPERTREND_1M']:
                if regime_type == 'SUPERTREND_1D':
                    direction_col = 'Supertrend_Direction'
                elif regime_type == 'SUPERTREND_1W':
                    direction_col = 'Supertrend_1W_Direction'
                else:
                    direction_col = 'Supertrend_1M_Direction'
                
                direction = row.get(direction_col, 'BUY')
                if hasattr(direction, 'iloc'):
                    direction = direction.iloc[0]
                triggered = direction == 'SELL'
            
            if triggered:
                print(f"REGIME TRIGGERED on {index_name} at {date}: {regime_type}")
                return True, action
        
        return False, 'none'
    
    def run_asset_class_backtest(self, index_formula, stock_formula=None, investment_type='ETF',
                                  rebal_config=None, regime_config=None, position_sizing='equal_weight'):
        """
        Run Asset Class Rotation backtest.
        
        Args:
            index_formula: Formula to score asset class indices
            stock_formula: Formula to score stocks (if investment_type='Stock')
            investment_type: 'ETF' or 'Stock'
            rebal_config: Rebalancing configuration
            regime_config: Regime filter configuration
            position_sizing: Position sizing method
        """
        if rebal_config is None:
            rebal_config = {'frequency': 'Monthly', 'date': 1, 'alt_day': 'Next Day'}
        
        # Validate formula
        is_valid, msg = self.parser.validate_formula(index_formula)
        if not is_valid:
            print(f"Invalid index formula: {msg}")
            return
        
        if investment_type == 'Stock' and stock_formula:
            is_valid, msg = self.parser.validate_formula(stock_formula)
            if not is_valid:
                print(f"Invalid stock formula: {msg}")
                return
        
        # Calculate indicators
        self.calculate_indicators(index_formula, stock_formula, regime_config)
        
        # Get common date range from index data
        all_dates = sorted(list(set().union(*[df.index for df in self.index_data.values()])))
        all_dates = [d for d in all_dates if d >= pd.Timestamp(self.start_date) and d <= pd.Timestamp(self.end_date)]
        
        if not all_dates:
            print("No valid dates in range")
            return
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(all_dates, rebal_config)
        
        # Initialize portfolio
        cash = self.initial_capital
        holdings = {}  # {ticker: {'shares': n, 'buy_price': p, 'asset_class': ac}}
        last_known_prices = {}
        
        regime_active = False
        
        for date in all_dates:
            is_rebalance = date in rebalance_dates
            
            # Calculate current equity
            holdings_value = 0.0
            for ticker, info in holdings.items():
                # Get price from appropriate data source
                if ticker in self.etf_data:
                    df = self.etf_data[ticker]
                elif ticker in self.stock_data:
                    df = self.stock_data[ticker]
                else:
                    continue
                
                if date in df.index:
                    price = self._get_scalar(df.loc[date, 'Close'])
                    last_known_prices[ticker] = price
                elif ticker in last_known_prices:
                    price = last_known_prices[ticker]
                else:
                    continue
                
                holdings_value += info['shares'] * price
            
            current_equity = cash + holdings_value
            
            # Check regime filter on selected asset classes
            if regime_config and regime_config.get('enabled'):
                # Get current held asset classes
                held_assets = list(set(info['asset_class'] for info in holdings.values()))
                triggered, action = self._check_regime_filter(date, regime_config, held_assets)
                
                if triggered and not regime_active:
                    # Exit all positions
                    print(f"[{date}] Regime filter triggered - exiting all positions")
                    for ticker, info in holdings.items():
                        if ticker in self.etf_data:
                            df = self.etf_data[ticker]
                        elif ticker in self.stock_data:
                            df = self.stock_data[ticker]
                        else:
                            continue
                        
                        if date in df.index:
                            sell_price = self._get_scalar(df.loc[date, 'Close'])
                        else:
                            sell_price = last_known_prices.get(ticker, info['buy_price'])
                        
                        trade_value = info['shares'] * sell_price
                        cash += trade_value
                        
                        self.trades.append({
                            'date': date,
                            'type': 'SELL',
                            'ticker': ticker,
                            'shares': info['shares'],
                            'price': sell_price,
                            'value': trade_value,
                            'reason': 'Regime Filter'
                        })
                    
                    holdings = {}
                    regime_active = True
                
                elif not triggered and regime_active:
                    print(f"[{date}] Regime filter recovered")
                    regime_active = False
            
            # Rebalance on rebalance dates (if not in regime)
            if is_rebalance and not regime_active:
                print(f"\n[{date}] REBALANCING...")
                
                # Score asset class indices
                indices_list = list(ASSET_CLASS_INDICES.keys())
                scores = self.score_indices(date, index_formula, indices_list)
                
                if len(scores) < 2:
                    print("Not enough indices with valid scores")
                    continue
                
                # Select top 2 asset classes
                top_2 = [s[0] for s in scores[:2]]
                print(f"Top 2 asset classes: {top_2} (scores: {[s[1] for s in scores[:2]]})")
                
                # Calculate target portfolio
                target_holdings = {}
                capital_per_class = current_equity / 2  # Equal split between 2 asset classes
                
                for asset_class in top_2:
                    asset_info = ASSET_CLASS_INDICES[asset_class]
                    
                    if investment_type == 'ETF' or not is_equity_asset(asset_class):
                        # Invest in ETF
                        etf = asset_info['etf']
                        if etf in self.etf_data and date in self.etf_data[etf].index:
                            price = self._get_scalar(self.etf_data[etf].loc[date, 'Close'])
                            shares = int(capital_per_class / price)
                            if shares > 0:
                                target_holdings[etf] = {
                                    'shares': shares,
                                    'price': price,
                                    'asset_class': asset_class
                                }
                    else:
                        # Invest in stocks within asset class
                        stock_count = asset_info['stock_count']
                        # Get stocks from universe (would need NSE fetcher integration)
                        # For now, use stocks that we have data for
                        available_stocks = [s for s in self.stock_data.keys()]
                        stock_scores = self.score_stocks(date, stock_formula, available_stocks)
                        
                        top_stocks = [s[0] for s in stock_scores[:stock_count]]
                        capital_per_stock = capital_per_class / len(top_stocks) if top_stocks else 0
                        
                        for stock in top_stocks:
                            if stock in self.stock_data and date in self.stock_data[stock].index:
                                price = self._get_scalar(self.stock_data[stock].loc[date, 'Close'])
                                shares = int(capital_per_stock / price)
                                if shares > 0:
                                    target_holdings[stock] = {
                                        'shares': shares,
                                        'price': price,
                                        'asset_class': asset_class
                                    }
                
                # Execute trades: Sell what we don't want, buy what we need
                # First, sell positions not in target
                for ticker in list(holdings.keys()):
                    if ticker not in target_holdings:
                        info = holdings[ticker]
                        if ticker in self.etf_data:
                            df = self.etf_data[ticker]
                        elif ticker in self.stock_data:
                            df = self.stock_data[ticker]
                        else:
                            continue
                        
                        if date in df.index:
                            sell_price = self._get_scalar(df.loc[date, 'Close'])
                        else:
                            sell_price = last_known_prices.get(ticker, info['buy_price'])
                        
                        trade_value = info['shares'] * sell_price
                        cash += trade_value
                        
                        self.trades.append({
                            'date': date,
                            'type': 'SELL',
                            'ticker': ticker,
                            'shares': info['shares'],
                            'price': sell_price,
                            'value': trade_value,
                            'reason': 'Rebalance - Exit'
                        })
                        
                        del holdings[ticker]
                
                # Then, buy new positions
                for ticker, target in target_holdings.items():
                    if ticker not in holdings:
                        buy_cost = target['shares'] * target['price']
                        if buy_cost <= cash:
                            cash -= buy_cost
                            holdings[ticker] = {
                                'shares': target['shares'],
                                'buy_price': target['price'],
                                'asset_class': target['asset_class']
                            }
                            
                            self.trades.append({
                                'date': date,
                                'type': 'BUY',
                                'ticker': ticker,
                                'shares': target['shares'],
                                'price': target['price'],
                                'value': buy_cost,
                                'reason': 'Rebalance - Entry'
                            })
            
            # Record portfolio value
            holdings_value = 0.0
            for ticker, info in holdings.items():
                if ticker in self.etf_data:
                    df = self.etf_data[ticker]
                elif ticker in self.stock_data:
                    df = self.stock_data[ticker]
                else:
                    continue
                
                if date in df.index:
                    price = self._get_scalar(df.loc[date, 'Close'])
                    last_known_prices[ticker] = price
                elif ticker in last_known_prices:
                    price = last_known_prices[ticker]
                else:
                    continue
                
                holdings_value += info['shares'] * price
            
            self.portfolio_history.append({
                'date': date,
                'cash': cash,
                'holdings_value': holdings_value,
                'equity': cash + holdings_value,
                'regime_active': regime_active
            })
        
        # Calculate final metrics
        self._calculate_metrics()
        
        print(f"\n=== BACKTEST COMPLETE ===")
        print(f"Final equity: ₹{self.portfolio_history[-1]['equity']:,.0f}")
        print(f"Total trades: {len(self.trades)}")
    
    def run_sector_rotation_backtest(self, index_type, index_formula, stock_formula,
                                      num_stocks=10, exit_rank=15, rebal_config=None,
                                      regime_config=None, position_sizing='equal_weight'):
        """
        Run Sector Rotation backtest.
        
        Args:
            index_type: 'sectoral' or 'thematic'
            index_formula: Formula to score indices
            stock_formula: Formula to score stocks
            num_stocks: Number of top stocks to buy (default 10)
            exit_rank: Exit rank threshold (default 15)
            rebal_config: Rebalancing configuration
            regime_config: Regime filter configuration
            position_sizing: Position sizing method
        """
        if rebal_config is None:
            rebal_config = {'frequency': 'Monthly', 'date': 1, 'alt_day': 'Next Day'}
        
        # Validate formulas
        is_valid, msg = self.parser.validate_formula(index_formula)
        if not is_valid:
            print(f"Invalid index formula: {msg}")
            return
        
        is_valid, msg = self.parser.validate_formula(stock_formula)
        if not is_valid:
            print(f"Invalid stock formula: {msg}")
            return
        
        # Get indices list based on type
        if index_type == 'sectoral':
            indices_list = SECTORAL_INDICES
        else:
            indices_list = THEMATIC_INDICES
        
        # Calculate indicators
        self.calculate_indicators(index_formula, stock_formula, regime_config)
        
        # Get common date range
        all_dates = sorted(list(set().union(*[df.index for df in self.index_data.values()])))
        all_dates = [d for d in all_dates if d >= pd.Timestamp(self.start_date) and d <= pd.Timestamp(self.end_date)]
        
        if not all_dates:
            print("No valid dates in range")
            return
        
        rebalance_dates = self._get_rebalance_dates(all_dates, rebal_config)
        
        # Initialize portfolio
        cash = self.initial_capital
        holdings = {}  # {ticker: {'shares': n, 'buy_price': p, 'rank': r}}
        last_known_prices = {}
        
        regime_active = False
        current_universe = []  # Merged universe from top 5 indices
        
        for date in all_dates:
            is_rebalance = date in rebalance_dates
            
            # Calculate current equity
            holdings_value = 0.0
            for ticker, info in holdings.items():
                if ticker not in self.stock_data:
                    continue
                
                df = self.stock_data[ticker]
                if date in df.index:
                    price = self._get_scalar(df.loc[date, 'Close'])
                    last_known_prices[ticker] = price
                elif ticker in last_known_prices:
                    price = last_known_prices[ticker]
                else:
                    continue
                
                holdings_value += info['shares'] * price
            
            current_equity = cash + holdings_value
            
            # Check regime filter
            if regime_config and regime_config.get('enabled'):
                # Check on current top indices
                triggered, action = self._check_regime_filter(date, regime_config, current_universe[:5])
                
                if triggered and not regime_active:
                    print(f"[{date}] Regime filter triggered - exiting all positions")
                    for ticker, info in holdings.items():
                        if ticker not in self.stock_data:
                            continue
                        
                        df = self.stock_data[ticker]
                        if date in df.index:
                            sell_price = self._get_scalar(df.loc[date, 'Close'])
                        else:
                            sell_price = last_known_prices.get(ticker, info['buy_price'])
                        
                        trade_value = info['shares'] * sell_price
                        cash += trade_value
                        
                        self.trades.append({
                            'date': date,
                            'type': 'SELL',
                            'ticker': ticker,
                            'shares': info['shares'],
                            'price': sell_price,
                            'value': trade_value,
                            'reason': 'Regime Filter'
                        })
                    
                    holdings = {}
                    regime_active = True
                
                elif not triggered and regime_active:
                    print(f"[{date}] Regime filter recovered")
                    regime_active = False
            
            # Rebalance
            if is_rebalance and not regime_active:
                print(f"\n[{date}] REBALANCING...")
                
                # Score indices
                scores = self.score_indices(date, index_formula, indices_list)
                
                if len(scores) < 5:
                    print(f"Not enough indices with valid scores ({len(scores)})")
                    continue
                
                # Select top 5 indices
                top_5_indices = [s[0] for s in scores[:5]]
                print(f"Top 5 indices: {top_5_indices}")
                
                # Merge unique stocks from top 5 indices
                # (In production, would fetch from NSE API)
                # For now, use all available stocks as universe
                current_universe = list(self.stock_data.keys())
                
                # Score stocks
                stock_scores = self.score_stocks(date, stock_formula, current_universe)
                
                # Get top N stocks
                top_stocks = [(s[0], i+1, s[1]) for i, s in enumerate(stock_scores[:num_stocks])]
                
                # Check exit rank for current holdings
                stock_rank_map = {s[0]: i+1 for i, s in enumerate(stock_scores)}
                
                # Sell stocks below exit rank
                for ticker in list(holdings.keys()):
                    rank = stock_rank_map.get(ticker, 9999)
                    if rank > exit_rank:
                        info = holdings[ticker]
                        df = self.stock_data[ticker]
                        
                        if date in df.index:
                            sell_price = self._get_scalar(df.loc[date, 'Close'])
                        else:
                            sell_price = last_known_prices.get(ticker, info['buy_price'])
                        
                        trade_value = info['shares'] * sell_price
                        cash += trade_value
                        
                        self.trades.append({
                            'date': date,
                            'type': 'SELL',
                            'ticker': ticker,
                            'shares': info['shares'],
                            'price': sell_price,
                            'value': trade_value,
                            'reason': f'Exit Rank ({rank} > {exit_rank})'
                        })
                        
                        del holdings[ticker]
                
                # Calculate capital for new positions
                current_holdings_count = len(holdings)
                slots_available = num_stocks - current_holdings_count
                
                if slots_available > 0:
                    # Recalculate equity after sells
                    holdings_value = sum(
                        holdings[t]['shares'] * last_known_prices.get(t, holdings[t]['buy_price'])
                        for t in holdings
                    )
                    available_capital = cash
                    capital_per_stock = available_capital / slots_available if slots_available > 0 else 0
                    
                    # Buy new stocks
                    for stock, rank, score in top_stocks:
                        if stock not in holdings and len(holdings) < num_stocks:
                            if stock not in self.stock_data:
                                continue
                            
                            df = self.stock_data[stock]
                            if date not in df.index:
                                continue
                            
                            price = self._get_scalar(df.loc[date, 'Close'])
                            shares = int(capital_per_stock / price)
                            
                            if shares > 0:
                                buy_cost = shares * price
                                if buy_cost <= cash:
                                    cash -= buy_cost
                                    holdings[stock] = {
                                        'shares': shares,
                                        'buy_price': price,
                                        'rank': rank
                                    }
                                    
                                    self.trades.append({
                                        'date': date,
                                        'type': 'BUY',
                                        'ticker': stock,
                                        'shares': shares,
                                        'price': price,
                                        'value': buy_cost,
                                        'reason': f'Rebalance (Rank {rank})'
                                    })
            
            # Record portfolio value
            holdings_value = 0.0
            for ticker, info in holdings.items():
                if ticker not in self.stock_data:
                    continue
                
                df = self.stock_data[ticker]
                if date in df.index:
                    price = self._get_scalar(df.loc[date, 'Close'])
                    last_known_prices[ticker] = price
                elif ticker in last_known_prices:
                    price = last_known_prices[ticker]
                else:
                    continue
                
                holdings_value += info['shares'] * price
            
            self.portfolio_history.append({
                'date': date,
                'cash': cash,
                'holdings_value': holdings_value,
                'equity': cash + holdings_value,
                'regime_active': regime_active
            })
        
        self._calculate_metrics()
        
        print(f"\n=== BACKTEST COMPLETE ===")
        print(f"Final equity: ₹{self.portfolio_history[-1]['equity']:,.0f}")
        print(f"Total trades: {len(self.trades)}")
    
    def _calculate_metrics(self):
        """Calculate performance metrics from portfolio history."""
        if not self.portfolio_history:
            return
        
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Basic metrics
        initial = df['equity'].iloc[0]
        final = df['equity'].iloc[-1]
        
        # CAGR
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Returns
        df['returns'] = df['equity'].pct_change()
        
        # Volatility (annualized)
        volatility = df['returns'].std() * np.sqrt(252) * 100
        
        # Sharpe Ratio (assuming 6% risk-free)
        rf = 0.06 / 252
        excess_returns = df['returns'] - rf
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Max Drawdown
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['peak'] - df['equity']) / df['peak']
        max_dd = df['drawdown'].max() * 100
        
        # Win Rate
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty and 'type' in trades_df.columns:
            sells = trades_df[trades_df['type'] == 'SELL']
            # Simple win rate based on sell price vs theoretical entry
            # (would need proper PnL tracking)
            win_rate = 0
        else:
            win_rate = 0
        
        # Monthly returns
        monthly = df['equity'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        
        self.metrics = {
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'total_return': (final / initial - 1) * 100,
            'initial_capital': initial,
            'final_capital': final,
            'total_trades': len(self.trades),
            'years': years
        }
        
        # Store for analysis
        self.equity_df = df
        self.monthly_returns = monthly_returns
    
    def get_results(self):
        """Return backtest results."""
        return {
            'metrics': self.metrics,
            'trades': self.trades,
            'portfolio_history': self.portfolio_history,
            'equity_df': getattr(self, 'equity_df', None),
            'monthly_returns': getattr(self, 'monthly_returns', None)
        }
