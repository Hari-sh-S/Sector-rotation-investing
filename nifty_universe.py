# NIFTY Universe Data - 19 Broad Market Indices Only
# Last Updated: 2026-01-31
# Version: 1.1 - Added SMA/EMA timeframe variants support
# Source: https://www.niftyindices.com/reports/index-factsheet

# All 19 Broad Market Indices using exact NSE API names
INDEX_NAMES = [
    "NIFTY 50",
    "NIFTY NEXT 50",
    "NIFTY 100",
    "NIFTY 200",
    "NIFTY 500",
    "NIFTY MIDCAP 150",
    "NIFTY MIDCAP 50",
    "NIFTY MID SELECT",           # NSE API name for Midcap Select
    "NIFTY MIDCAP 100",
    "NIFTY SMLCAP 250",           # NSE API name for Smallcap 250
    "NIFTY SMLCAP 50",            # NSE API name for Smallcap 50
    "NIFTY SMLCAP 100",           # NSE API name for Smallcap 100
    "NIFTY LARGEMID250",          # NSE API name for LargeMidcap 250
    "NIFTY MIDSML 400",           # NSE API name for MidSmallcap 400
    "NIFTY500 MULTICAP",          # NSE API name for Multicap 50:25:25
    "NIFTY MICROCAP250",          # NSE API name for Microcap 250
    "NIFTY TOTAL MKT",            # NSE API name for Total Market
    "NIFTY500 LMS EQL",           # NSE API name for LMS Equal-Cap Weighted
    "NIFTY FPI 150",              # NSE API name for India FPI 150
]

# Display names for UI (maps to INDEX_NAMES)
DISPLAY_NAMES = {
    "NIFTY 50": "Nifty 50",
    "NIFTY NEXT 50": "Nifty Next 50",
    "NIFTY 100": "Nifty 100",
    "NIFTY 200": "Nifty 200",
    "NIFTY 500": "Nifty 500",
    "NIFTY MIDCAP 150": "Nifty Midcap 150",
    "NIFTY MIDCAP 50": "Nifty Midcap 50",
    "NIFTY MID SELECT": "Nifty Midcap Select",
    "NIFTY MIDCAP 100": "Nifty Midcap 100",
    "NIFTY SMLCAP 250": "Nifty Smallcap 250",
    "NIFTY SMLCAP 50": "Nifty Smallcap 50",
    "NIFTY SMLCAP 100": "Nifty Smallcap 100",
    "NIFTY LARGEMID250": "Nifty LargeMidcap 250",
    "NIFTY MIDSML 400": "Nifty MidSmallcap 400",
    "NIFTY500 MULTICAP": "Nifty 500 Multicap 50:25:25",
    "NIFTY MICROCAP250": "Nifty Microcap 250",
    "NIFTY TOTAL MKT": "Nifty Total Market",
    "NIFTY500 LMS EQL": "Nifty500 LargeMidSmall Equal-Cap Weighted",
    "NIFTY FPI 150": "Nifty India FPI 150",
}

# Fallback stock lists for key indexes (used when NSE cache is unavailable)
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
    "WIPRO", "ULTRACEMCO", "TITAN", "BAJFINANCE", "SUNPHARMA",
    "NESTLEIND", "TECHM", "ONGC", "NTPC", "POWERGRID",
    "M&M", "TATAMOTORS", "TATASTEEL", "ADANIPORTS", "COALINDIA",
    "BAJAJFINSV", "INDUSINDBK", "HINDALCO", "JSWSTEEL", "GRASIM",
    "DIVISLAB", "DRREDDY", "CIPLA", "APOLLOHOSP", "EICHERMOT",
    "HEROMOTOCO", "BRITANNIA", "TATACONSUM", "BAJAJ-AUTO", "SHRIRAMFIN",
    "ADANIENT", "SBILIFE", "HDFCLIFE", "BPCL", "IOC"
]

NIFTY_NEXT_50 = [
    "ADANIGREEN", "ADANIPOWER", "ADANITRANS", "AMBUJACEM", "ATGL",
    "BERGEPAINT", "BEL", "BOSCHLTD", "CANBK", "CHOLAFIN",
    "COLPAL", "DABUR", "DLF", "DMART", "GAIL",
    "GODREJCP", "HAL", "HAVELLS", "ICICIPRULI", "INDIGO",
    "JINDALSTEL", "LTIM", "MARICO", "MOTHERSON", "MPHASIS",
    "NAUKRI", "NMDC", "PAYTM", "PIDILITIND", "PNB",
    "RECLTD", "SAIL", "SBICARD", "SIEMENS", "TATAPOWER",
    "TATATECH", "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL",
    "VEDL", "VOLTAS", "ZOMATO", "ZYDUSLIFE", "ICICIGI",
    "PGHH", "PETRONET", "PFC", "INDHOTEL", "BAJAJHLDNG"
]

NIFTY_100 = list(dict.fromkeys(NIFTY_50 + NIFTY_NEXT_50))

# Universe dictionary - 19 Broad Market Indices
UNIVERSES = {name: [] for name in INDEX_NAMES}

# Populate fallback lists
UNIVERSES["NIFTY 50"] = NIFTY_50
UNIVERSES["NIFTY NEXT 50"] = NIFTY_NEXT_50
UNIVERSES["NIFTY 100"] = NIFTY_100
UNIVERSES["NIFTY 200"] = NIFTY_100
UNIVERSES["NIFTY 500"] = NIFTY_100
UNIVERSES["NIFTY MIDCAP 150"] = NIFTY_NEXT_50
UNIVERSES["NIFTY MIDCAP 50"] = NIFTY_NEXT_50[:50]
UNIVERSES["NIFTY MID SELECT"] = NIFTY_NEXT_50[:25]
UNIVERSES["NIFTY MIDCAP 100"] = NIFTY_NEXT_50
UNIVERSES["NIFTY SMLCAP 250"] = NIFTY_100
UNIVERSES["NIFTY SMLCAP 50"] = NIFTY_NEXT_50[:50]
UNIVERSES["NIFTY SMLCAP 100"] = NIFTY_NEXT_50
UNIVERSES["NIFTY LARGEMID250"] = NIFTY_100
UNIVERSES["NIFTY MIDSML 400"] = NIFTY_100
UNIVERSES["NIFTY500 MULTICAP"] = NIFTY_100
UNIVERSES["NIFTY MICROCAP250"] = NIFTY_NEXT_50
UNIVERSES["NIFTY TOTAL MKT"] = NIFTY_100
UNIVERSES["NIFTY500 LMS EQL"] = NIFTY_100
UNIVERSES["NIFTY FPI 150"] = NIFTY_100


def _load_nse_cache():
    """Load NSE cache if available."""
    try:
        from pathlib import Path
        import json
        cache_file = Path("nse_universe_cache.json")
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('universes', {})
    except:
        pass
    return {}


def get_universe(name, as_of_date=None):
    """
    Get stock list for a given universe name.
    
    Args:
        name: Universe name (e.g., 'NIFTY 500')
        as_of_date: Optional date for point-in-time lookup (survivorship-bias free)
                   If None, returns current constituents
    
    Returns:
        List of stock symbols
    """
    # If as_of_date provided, try historical constituents first
    if as_of_date is not None:
        try:
            from historical_constituents import get_index_universe, get_universe_with_fallback
            import pandas as pd
            
            # Convert universe name to index name format (e.g., "NIFTY 500" -> "NIFTY500")
            index_name = name.replace(' ', '').upper()
            
            # Convert date if needed
            if not isinstance(as_of_date, pd.Timestamp):
                as_of_date = pd.Timestamp(as_of_date)
            
            # Try historical, fallback to current
            current_universe = _get_current_universe(name)
            historical, used_historical = get_universe_with_fallback(
                index_name, as_of_date, current_universe
            )
            return historical
        except ImportError:
            pass  # historical_constituents not available
        except Exception as e:
            print(f"Warning: Historical constituents lookup failed: {e}")
    
    # Default: return current constituents
    return _get_current_universe(name)


def _get_current_universe(name):
    """Get current (today's) universe - internal helper."""
    # First try NSE cache
    cached = _load_nse_cache()
    if name in cached and cached[name]:
        return cached[name]
    
    # Fallback to hardcoded
    return UNIVERSES.get(name, [])


def get_all_universe_names():
    """Get all available universe names (19 Broad Market Indices)."""
    return INDEX_NAMES.copy()


def get_display_name(index_name):
    """Get user-friendly display name for an index."""
    return DISPLAY_NAMES.get(index_name, index_name)


def get_broad_market_universes():
    """Get all broad market indices (all 19)."""
    return INDEX_NAMES.copy()

