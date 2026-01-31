"""
NSE Index Constituents Fetcher
Downloads live index constituents from NSE India using nsetools library
"""

import json
import time
from pathlib import Path

try:
    from nsetools import Nse
    NSE_AVAILABLE = True
except ImportError:
    NSE_AVAILABLE = False
    print("nsetools not installed. Run: pip install nsetools")

CACHE_FILE = Path("nse_universe_cache.json")


def get_nse_client():
    """Get NSE client instance."""
    if not NSE_AVAILABLE:
        return None
    return Nse()


def fetch_all_indices(progress_callback=None):
    """Fetch constituents only for our specified 55 indexes."""
    nse = get_nse_client()
    if not nse:
        print("NSE client not available")
        return {}
    
    # Only fetch our specified indexes
    try:
        from nifty_universe import INDEX_NAMES
        target_indices = INDEX_NAMES
    except:
        # Fallback to all NSE indices
        target_indices = nse.get_index_list()
    
    print(f"Fetching {len(target_indices)} indexes...")
    
    results = {}
    total = len(target_indices)
    success_count = 0
    fail_count = 0
    
    for i, index_name in enumerate(target_indices):
        if progress_callback:
            progress_callback(i / total, f"Fetching {index_name}...")
        
        print(f"Fetching {index_name}...", end=" ")
        
        try:
            stocks = nse.get_stocks_in_index(index_name)
            if stocks:
                results[index_name] = stocks
                print(f"OK - {len(stocks)} stocks")
                success_count += 1
            else:
                print("Empty")
                fail_count += 1
        except Exception as e:
            print(f"Error: {e}")
            fail_count += 1
        
        time.sleep(0.3)  # Rate limiting
    
    print(f"\nSummary: {success_count} succeeded, {fail_count} failed")
    return results


def save_to_cache(data):
    """Save fetched data to cache file."""
    cache_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'universes': data
    }
    
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"Saved {len(data)} universes to {CACHE_FILE}")


def load_from_cache():
    """Load data from cache file."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('universes', {}), data.get('timestamp', 'Unknown')
        except Exception as e:
            print(f"Error loading cache: {e}")
    return {}, None


def get_universe(name):
    """Get a universe by name, from cache or hardcoded fallback."""
    cached, timestamp = load_from_cache()
    if name in cached:
        return cached[name]
    
    # Fallback to hardcoded
    try:
        from nifty_universe import UNIVERSES
        return UNIVERSES.get(name, [])
    except:
        return []


def get_all_universe_names():
    """Get all available universe names - returns our 60 specified indexes."""
    # Import from nifty_universe to get the specified list
    try:
        from nifty_universe import INDEX_NAMES
        return INDEX_NAMES.copy()
    except:
        pass
    
    # Fallback to cache
    cached, _ = load_from_cache()
    if cached:
        return sorted(cached.keys())
    
    return []


def refresh_universes(progress_callback=None):
    """Refresh all universe data from NSE."""
    print("Refreshing universe data from NSE India...")
    data = fetch_all_indices(progress_callback)
    
    if data:
        save_to_cache(data)
        return True, f"Updated {len(data)} universes"
    else:
        return False, "Failed to fetch data from NSE"


if __name__ == "__main__":
    print("NSE Universe Fetcher (using nsetools)")
    print("=" * 50)
    
    if not NSE_AVAILABLE:
        print("Please install nsetools: pip install nsetools")
        exit(1)
    
    success, message = refresh_universes()
    print(f"\nResult: {message}")
    
    # Show summary
    cached, timestamp = load_from_cache()
    if cached:
        print(f"\nCached at: {timestamp}")
        print(f"Total universes: {len(cached)}")
        
        # Show stock counts
        print("\nUniverse stock counts:")
        for name, stocks in sorted(cached.items()):
            print(f"  {name}: {len(stocks)} stocks")
