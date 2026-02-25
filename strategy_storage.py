"""
Strategy Storage - Save and load backtest configurations locally as JSON.
"""

import json
import os
from datetime import datetime
from pathlib import Path


STRATEGY_FILE = "saved_strategies.json"


def load_strategies():
    """Load all saved strategies from JSON file."""
    if not os.path.exists(STRATEGY_FILE):
        return {}
    try:
        with open(STRATEGY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_strategy(name: str, config: dict) -> bool:
    """Save a strategy configuration under the given name."""
    try:
        strategies = load_strategies()
        strategies[name] = {
            "config": config,
            "saved_at": datetime.now().isoformat(),
        }
        with open(STRATEGY_FILE, "w") as f:
            json.dump(strategies, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving strategy: {e}")
        return False


def delete_strategy(name: str) -> bool:
    """Delete a strategy by name."""
    try:
        strategies = load_strategies()
        if name in strategies:
            del strategies[name]
            with open(STRATEGY_FILE, "w") as f:
                json.dump(strategies, f, indent=2, default=str)
            return True
        return False
    except Exception as e:
        print(f"Error deleting strategy: {e}")
        return False


def get_strategy_names():
    """Get list of saved strategy names."""
    return list(load_strategies().keys())


def get_strategy(name: str):
    """Get a specific strategy config by name."""
    strategies = load_strategies()
    return strategies.get(name, {}).get("config", None)
