import re
import pandas as pd
import numpy as np

class ScoreParser:
    def __init__(self):
        # Metric types that support dynamic month values (1-24 months)
        self.metric_types = [
            'Performance', 'Volatility', 'Downside Volatility',
            'Max Drawdown', 'Sharpe', 'Sortino', 'Calmar',
            'Positive Days', 'Negative Days',
            'Distance From High', 'Distance From Low'
        ]
        
        # Pattern to match dynamic metrics like "15 Month Performance", "10 Week Performance" or "1 Year Performance"
        # Supports: "N Month MetricType" (1-24), "N Week MetricType" (1-52), or "1 Year MetricType"
        self.metric_pattern = re.compile(
            r'(\d{1,2})\s+Month\s+(Performance|Volatility|Downside Volatility|Max Drawdown|Sharpe|Sortino|Calmar|Positive Days|Negative Days|Distance From High|Distance From Low)|'
            r'(\d{1,2})\s+Week\s+(Performance|Volatility|Downside Volatility|Max Drawdown|Sharpe|Sortino|Calmar|Positive Days|Negative Days|Distance From High|Distance From Low)|'
            r'1\s+Year\s+(Performance|Volatility|Downside Volatility|Max Drawdown|Sharpe|Sortino|Calmar|Positive Days|Negative Days|Distance From High|Distance From Low)',
            re.IGNORECASE
        )
        
        # Common examples for UI display (not exhaustive)
        self.example_periods = [1, 3, 6, 9, 12]
        
        # Build example metrics for UI (subset of all possible)
        self.allowed_metrics = []
        for period in self.example_periods:
            period_name = '1 Year' if period == 12 else f'{period} Month'
            for metric_type in self.metric_types:
                self.allowed_metrics.append(f'{period_name} {metric_type}')
        
        # Create metric groups for UI display
        self.metric_groups = {
            'Performance': [f'{p if p != 12 else "1 Year"} {"Month" if p != 12 else ""} Performance'.replace("  ", " ").strip() 
                          for p in self.example_periods],
            'Volatility': [f'{p if p != 12 else "1 Year"} {"Month" if p != 12 else ""} Volatility'.replace("  ", " ").strip()
                         for p in self.example_periods],
            'Downside Volatility': [f'{p if p != 12 else "1 Year"} {"Month" if p != 12 else ""} Downside Volatility'.replace("  ", " ").strip()
                                   for p in self.example_periods],
            'Max Drawdown': [f'{p if p != 12 else "1 Year"} {"Month" if p != 12 else ""} Max Drawdown'.replace("  ", " ").strip()
                           for p in self.example_periods],
            'Sharpe Ratio': [f'{p if p != 12 else "1 Year"} {"Month" if p != 12 else ""} Sharpe'.replace("  ", " ").strip()
                           for p in self.example_periods],
            'Sortino Ratio': [f'{p if p != 12 else "1 Year"} {"Month" if p != 12 else ""} Sortino'.replace("  ", " ").strip()
                            for p in self.example_periods],
            'Calmar Ratio': [f'{p if p != 12 else "1 Year"} {"Month" if p != 12 else ""} Calmar'.replace("  ", " ").strip()
                           for p in self.example_periods],
        }
        
        # Fix the metric groups formatting
        self.metric_groups = {
            'Performance': ['1 Month Performance', '3 Month Performance', '6 Month Performance', 
                          '9 Month Performance', '1 Year Performance'],
            'Volatility': ['1 Month Volatility', '3 Month Volatility', '6 Month Volatility',
                         '9 Month Volatility', '1 Year Volatility'],
            'Downside Volatility': ['1 Month Downside Volatility', '3 Month Downside Volatility', 
                                   '6 Month Downside Volatility', '9 Month Downside Volatility', 
                                   '1 Year Downside Volatility'],
            'Max Drawdown': ['1 Month Max Drawdown', '3 Month Max Drawdown', '6 Month Max Drawdown',
                           '9 Month Max Drawdown', '1 Year Max Drawdown'],
            'Sharpe Ratio': ['1 Month Sharpe', '3 Month Sharpe', '6 Month Sharpe',
                           '9 Month Sharpe', '1 Year Sharpe'],
            'Sortino Ratio': ['1 Month Sortino', '3 Month Sortino', '6 Month Sortino',
                            '9 Month Sortino', '1 Year Sortino'],
            'Calmar Ratio': ['1 Month Calmar', '3 Month Calmar', '6 Month Calmar',
                           '9 Month Calmar', '1 Year Calmar'],
            '% Positive Days': ['1 Month Positive Days', '3 Month Positive Days', '6 Month Positive Days',
                               '9 Month Positive Days', '1 Year Positive Days'],
            '% Negative Days': ['1 Month Negative Days', '3 Month Negative Days', '6 Month Negative Days',
                               '9 Month Negative Days', '1 Year Negative Days'],
            '% Distance From High': ['1 Month Distance From High', '3 Month Distance From High', '6 Month Distance From High',
                                    '9 Month Distance From High', '1 Year Distance From High'],
            '% Distance From Low': ['1 Month Distance From Low', '3 Month Distance From Low', '6 Month Distance From Low',
                                   '9 Month Distance From Low', '1 Year Distance From Low'],
        }
    
    def extract_required_periods(self, formula):
        """Extract all period requirements.
        Returns a set of (value, unit, metric_type) tuples."""
        required = set()
        
        # Find all matches in the formula
        for match in self.metric_pattern.finditer(formula):
            if match.group(1):  # N Month pattern
                months = int(match.group(1))
                metric_type = match.group(2)
                if 1 <= months <= 24:
                    required.add((months, 'Month', metric_type))
            elif match.group(3):  # N Week pattern
                weeks = int(match.group(3))
                metric_type = match.group(4)
                if 1 <= weeks <= 52:
                    required.add((weeks, 'Week', metric_type))
            elif match.group(5):  # 1 Year pattern
                metric_type = match.group(5)
                required.add((12, 'Month', metric_type))
        
        return required

    def validate_formula(self, formula):
        """
        Validates the formula before execution.
        Returns (is_valid, error_message)
        """
        if not formula or formula.strip() == "":
            return False, "Formula cannot be empty"
        
        # Check for balanced parentheses
        if formula.count('(') != formula.count(')'):
            return False, "Unbalanced parentheses"
        
        # Preprocess
        processed = self._preprocess_formula(formula)
        
        # Replace all valid metric patterns with (1) for testing
        test_formula = self.metric_pattern.sub('(1)', processed)
        
        # Check if only valid characters remain
        test_clean = re.sub(r'[\d\.\+\-\*\/\(\)\s]', '', test_formula)
        if test_clean:
            return False, f"Unknown: {test_clean[:15]}"
        
        # Validate month/week ranges
        for match in self.metric_pattern.finditer(formula):
            if match.group(1):  # N Month pattern
                months = int(match.group(1))
                if months < 1 or months > 24:
                    return False, f"Invalid period: {months} months (use 1-24)"
            elif match.group(3):  # N Week pattern
                weeks = int(match.group(3))
                if weeks < 1 or weeks > 52:
                    return False, f"Invalid period: {weeks} weeks (use 1-52)"
        
        # Try to evaluate with dummy values to check syntax
        try:
            result = eval(test_formula)
            if result is None:
                return False, "Returns None"
            return True, "Valid"
        except ZeroDivisionError:
            return True, "Valid"
        except SyntaxError as e:
            return False, f"Syntax error"
        except Exception as e:
            return False, f"Error: {str(e)[:25]}"
    
    def _preprocess_formula(self, formula):
        """Internal method to preprocess formula."""
        processed = formula
        
        # Replace percentages (e.g., 80% -> 0.80)
        processed = re.sub(r'(\d+(\.\d+)?)%', r'(\1/100)', processed)
        
        # Replace operators (only standalone x, not x within words like 'Max')
        # Use word boundary to match 'x' only when it's between numbers/spaces
        processed = re.sub(r'(?<=[\d\s\)])x(?=[\d\s\(])', '*', processed)
        processed = processed.replace('×', '*')
        processed = processed.replace('÷', '/')
        
        return processed

    def parse_and_calculate(self, formula, row):
        """
        Parses and evaluates formula for a single row.
        Returns the score or a very negative number on error.
        """
        processed_formula = self._preprocess_formula(formula)
        
        # Function to replace metric with value from row
        def replace_metric(match):
            if match.group(1):  # N Month pattern
                months = int(match.group(1))
                metric_type = match.group(2)
                metric_name = f'{months} Month {metric_type}'
            elif match.group(3):  # N Week pattern
                weeks = int(match.group(3))
                metric_type = match.group(4)
                metric_name = f'{weeks} Week {metric_type}'
            else:  # 1 Year pattern
                metric_type = match.group(5)
                metric_name = f'1 Year {metric_type}'
            
            val = row.get(metric_name, 0)
            
            # Handle Series case
            if isinstance(val, pd.Series):
                val = val.iloc[0] if len(val) > 0 else 0
            
            # Handle NaN and inf
            if pd.isna(val):
                val = 0
            elif np.isinf(val):
                val = 0
            
            return str(val)
        
        # Replace all metrics using regex
        processed_formula = self.metric_pattern.sub(replace_metric, processed_formula)
        
        # Final safety check
        if not re.match(r'^[\d\.\+\-\*\/\(\)\s]+$', processed_formula):
            return -999999
            
        try:
            score = eval(processed_formula)
            if np.isnan(score) or np.isinf(score):
                return -999999
            return score
        except Exception as e:
            return -999999

    def calculate_scores(self, df, formula):
        """
        Calculate scores for entire dataframe using row-by-row processing.
        Returns a Series of scores.
        """
        # Use reliable row-by-row calculation
        return df.apply(lambda row: self.parse_and_calculate(formula, row), axis=1)
    
    def get_example_formulas(self):
        """Returns a dictionary of example formulas with descriptions."""
        return {
            # Basic formulas
            "Simple Momentum": "6 Month Performance",
            "Risk-Adjusted Momentum": "(6 Month Performance / 6 Month Volatility)",
            "Weighted Performance": "(70% * 6 Month Performance) + (20% * 3 Month Performance) + (10% * 1 Month Performance)",
            "Sharpe-Based": "6 Month Sharpe",
            "Multi-Factor": "((80% * 9 Month Performance) + (20% * 6 Month Performance)) / 1 Month Volatility",
            "Calmar Focus": "6 Month Calmar",
            "Low Volatility": "1 / 3 Month Volatility",
            "Momentum + Quality": "(6 Month Performance * 6 Month Sharpe)",
            # Advanced formulas
            "Trend Consistency": "(6 Month Performance / 1 Month Volatility) * 6 Month Sharpe",
            "Acceleration Momentum": "(3 Month Performance - 6 Month Performance) / 1 Month Volatility",
            "Drawdown-Aware Momentum": "6 Month Performance / 6 Month Max Drawdown",
            "Defensive Momentum": "6 Month Performance / 3 Month Volatility",
            "Smooth Returns": "6 Month Sharpe * 6 Month Sortino",
            "Multi-Horizon Momentum": "(1 Month Performance + 3 Month Performance + 6 Month Performance) / 3",
            "Crash-Resistant Momentum": "6 Month Performance / 6 Month Downside Volatility",
            "Momentum Persistence": "6 Month Performance / (1 + 6 Month Volatility)",
            "Quality-Adjusted Trend": "(6 Month Performance * 6 Month Sharpe) / 6 Month Max Drawdown",
            "Regime-Adaptive Momentum": "(9 Month Performance * 3 Month Sharpe) / 1 Month Volatility",
            # Positive/Negative Days formulas
            "Consistency Focus": "6 Month Positive Days",
            "Win Rate Momentum": "6 Month Performance * 6 Month Positive Days",
            "Downside Avoidance": "6 Month Performance / 6 Month Negative Days",
            # Distance from High/Low formulas
            "Near 52W High": "1 / (1 + 1 Year Distance From High)",
            "Breakout Momentum": "6 Month Performance / (1 + 6 Month Distance From High)",
            "Value + Momentum": "6 Month Performance * 6 Month Distance From Low",
        }
