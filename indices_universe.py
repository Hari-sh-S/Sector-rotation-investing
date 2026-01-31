# Indices and ETF Definitions for Multi-Asset Rotation System
# Contains asset class mappings, sectoral indices, and thematic indices

# ==================== ASSET CLASS ROTATION ====================
# 6 Asset Classes with corresponding ETFs and stock universe info
ASSET_CLASS_INDICES = {
    "NIFTY 100": {
        "etf": "NIF100BEES",
        "type": "largecap",
        "stock_count": 10,
        "yahoo_ticker": "^CNX100",
        "description": "Large Cap Index"
    },
    "NIFTY MIDCAP 150": {
        "etf": "MID150BEES",
        "type": "midcap", 
        "stock_count": 15,
        "yahoo_ticker": "^NSEMDCP150",
        "description": "Mid Cap Index"
    },
    "NIFTY SMLCAP 250": {
        "etf": "GROWWSC250",
        "type": "smallcap",
        "stock_count": 20,
        "yahoo_ticker": "^CNXSC",
        "description": "Small Cap Index"
    },
    "GOLD": {
        "etf": "GOLDBEES",
        "type": "commodity",
        "stock_count": 0,
        "yahoo_ticker": "GC=F",
        "description": "Gold Commodity"
    },
    "SILVER": {
        "etf": "SILVERBEES",
        "type": "commodity",
        "stock_count": 0,
        "yahoo_ticker": "SI=F",
        "description": "Silver Commodity"
    },
    "GILT 5Y": {
        "etf": "GILT5YBEES",
        "type": "bond",
        "stock_count": 0,
        "yahoo_ticker": None,  # Use ETF directly
        "description": "5 Year G-Sec Bond"
    },
}

# Asset class display order
ASSET_CLASS_ORDER = [
    "NIFTY 100",
    "NIFTY MIDCAP 150", 
    "NIFTY SMLCAP 250",
    "GOLD",
    "SILVER",
    "GILT 5Y"
]

# ==================== SECTOR ROTATION ====================
# 19 Sectoral Indices
SECTORAL_INDICES = [
    "NIFTY AUTO",
    "NIFTY FINANCIAL SERVICES 25/50",
    "NIFTY FMCG",
    "NIFTY IT",
    "NIFTY MEDIA",
    "NIFTY METAL",
    "NIFTY PHARMA",
    "NIFTY PSU BANK",
    "NIFTY PRIVATE BANK",
    "NIFTY REALTY",
    "NIFTY HEALTHCARE INDEX",
    "NIFTY CONSUMER DURABLES",
    "NIFTY OIL & GAS",
    "NIFTY MIDSMALL HEALTHCARE",
    "NIFTY FINANCIAL SERVICES EX-BANK",
    "NIFTY MIDSMALL FINANCIAL SERVICES",
    "NIFTY MIDSMALL IT & TELECOM",
    "NIFTY CHEMICALS",
    "NIFTY500 HEALTHCARE"
]

# 25 Thematic Indices
THEMATIC_INDICES = [
    "NIFTY COMMODITIES",
    "NIFTY INDIA CONSUMPTION",
    "NIFTY CPSE",
    "NIFTY ENERGY",
    "NIFTY INFRASTRUCTURE",
    "NIFTY MNC",
    "NIFTY PSE",
    "NIFTY SERVICES SECTOR",
    "NIFTY INDIA DIGITAL",
    "NIFTY INDIA MANUFACTURING",
    "NIFTY INDIA DEFENCE",
    "NIFTY INDIA TOURISM",
    "NIFTY CAPITAL MARKETS",
    "NIFTY EV & NEW AGE AUTOMOTIVE",
    "NIFTY INDIA NEW AGE CONSUMPTION",
    "NIFTY INDIA SELECT 5 CORPORATE GROUPS (MAATR)",
    "NIFTY MOBILITY",
    "NIFTY CORE HOUSING",
    "NIFTY HOUSING",
    "NIFTY NON-CYCLICAL CONSUMER",
    "NIFTY RURAL",
    "NIFTY TRANSPORTATION & LOGISTICS",
    "NIFTY INDIA INTERNET",
    "NIFTY WAVES",
    "NIFTY INDIA INFRASTRUCTURE & LOGISTICS"
]

# Yahoo Finance ticker mapping for indices
INDEX_YAHOO_TICKERS = {
    # Broad Market
    "NIFTY 50": "^NSEI",
    "NIFTY 100": "^CNX100",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX",
    "NIFTY MIDCAP 150": "^NSEMDCP150",
    "NIFTY MIDCAP 50": "^NSEMDCP50",
    "NIFTY MIDCAP 100": "^CNXMC",
    "NIFTY SMLCAP 250": "^CNXSC",
    "NIFTY SMLCAP 50": "^NIFTYSMCP50",
    "NIFTY SMLCAP 100": "^CNXSC",
    # Sectoral
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY FINANCIAL SERVICES": "^CNXFINANCE",
    "NIFTY FINANCIAL SERVICES 25/50": "^CNXFINANCE",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY IT": "^CNXIT",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY PRIVATE BANK": "^NIFTYPVTBANK",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY HEALTHCARE INDEX": "^CNXPHARMA",
    "NIFTY CONSUMER DURABLES": "^CNXCONSUM",
    "NIFTY OIL & GAS": "^CNXENERGY",
    # Thematic
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY INFRASTRUCTURE": "^CNXINFRA",
    "NIFTY PSE": "^CNXPSE",
    "NIFTY MNC": "^CNXMNC",
    "NIFTY COMMODITIES": "^CNXCOMMODITIES",
    "NIFTY INDIA CONSUMPTION": "^CNXCONSUM",
    "NIFTY CPSE": "^CNXPSE",
    # Commodities
    "GOLD": "GC=F",
    "SILVER": "SI=F",
}

# NSE API names for fetching constituents
INDEX_NSE_NAMES = {
    "NIFTY 100": "NIFTY 100",
    "NIFTY MIDCAP 150": "NIFTY MIDCAP 150",
    "NIFTY SMLCAP 250": "NIFTY SMLCAP 250",
    "NIFTY AUTO": "NIFTY AUTO",
    "NIFTY FINANCIAL SERVICES 25/50": "NIFTY FINANCIAL SERVICES 25/50",
    "NIFTY FMCG": "NIFTY FMCG",
    "NIFTY IT": "NIFTY IT",
    "NIFTY MEDIA": "NIFTY MEDIA",
    "NIFTY METAL": "NIFTY METAL",
    "NIFTY PHARMA": "NIFTY PHARMA",
    "NIFTY PSU BANK": "NIFTY PSU BANK",
    "NIFTY PRIVATE BANK": "NIFTY PRIVATE BANK",
    "NIFTY REALTY": "NIFTY REALTY",
    "NIFTY HEALTHCARE INDEX": "NIFTY HEALTHCARE INDEX",
    "NIFTY CONSUMER DURABLES": "NIFTY CONSUMER DURABLES",
    "NIFTY OIL & GAS": "NIFTY OIL & GAS",
    "NIFTY MIDSMALL HEALTHCARE": "NIFTY MIDSMALL HEALTHCARE",
    "NIFTY FINANCIAL SERVICES EX-BANK": "NIFTY FIN SERVICE EX-BANK",
    "NIFTY MIDSMALL FINANCIAL SERVICES": "NIFTY MIDSMALL FIN SERVICE",
    "NIFTY MIDSMALL IT & TELECOM": "NIFTY MIDSMALL IT & TELECOM",
    "NIFTY CHEMICALS": "NIFTY CHEMICALS",
    "NIFTY500 HEALTHCARE": "NIFTY500 HEALTHCARE",
    # Thematic
    "NIFTY COMMODITIES": "NIFTY COMMODITIES",
    "NIFTY INDIA CONSUMPTION": "NIFTY INDIA CONSUMPTION",
    "NIFTY CPSE": "NIFTY CPSE",
    "NIFTY ENERGY": "NIFTY ENERGY",
    "NIFTY INFRASTRUCTURE": "NIFTY INFRASTRUCTURE",
    "NIFTY MNC": "NIFTY MNC",
    "NIFTY PSE": "NIFTY PSE",
    "NIFTY SERVICES SECTOR": "NIFTY SERVICES SECTOR",
    "NIFTY INDIA DIGITAL": "NIFTY INDIA DIGITAL",
    "NIFTY INDIA MANUFACTURING": "NIFTY INDIA MANUFACTURING",
    "NIFTY INDIA DEFENCE": "NIFTY INDIA DEFENCE",
    "NIFTY INDIA TOURISM": "NIFTY INDIA TOURISM",
    "NIFTY CAPITAL MARKETS": "NIFTY CAPITAL MARKETS",
    "NIFTY EV & NEW AGE AUTOMOTIVE": "NIFTY EV & NEW AGE AUTO",
    "NIFTY INDIA NEW AGE CONSUMPTION": "NIFTY IND NEW AGE CONSUMPTION",
    "NIFTY INDIA SELECT 5 CORPORATE GROUPS (MAATR)": "NIFTY MAATR",
    "NIFTY MOBILITY": "NIFTY MOBILITY",
    "NIFTY CORE HOUSING": "NIFTY CORE HOUSING",
    "NIFTY HOUSING": "NIFTY HOUSING",
    "NIFTY NON-CYCLICAL CONSUMER": "NIFTY NON-CYCLICAL CONSUMER",
    "NIFTY RURAL": "NIFTY RURAL",
    "NIFTY TRANSPORTATION & LOGISTICS": "NIFTY TRANS & LOGISTICS",
    "NIFTY INDIA INTERNET": "NIFTY INDIA INTERNET",
    "NIFTY WAVES": "NIFTY WAVES",
    "NIFTY INDIA INFRASTRUCTURE & LOGISTICS": "NIFTY IND INFRA & LOGISTICS",
}


def get_asset_class_list():
    """Return list of asset classes for Asset Class Rotation mode."""
    return ASSET_CLASS_ORDER.copy()


def get_asset_class_info(asset_class):
    """Get ETF and other info for an asset class."""
    return ASSET_CLASS_INDICES.get(asset_class, {})


def get_sectoral_indices():
    """Return list of sectoral indices."""
    return SECTORAL_INDICES.copy()


def get_thematic_indices():
    """Return list of thematic indices."""
    return THEMATIC_INDICES.copy()


def get_yahoo_ticker(index_name):
    """Get Yahoo Finance ticker for an index."""
    # First check asset class mapping
    if index_name in ASSET_CLASS_INDICES:
        return ASSET_CLASS_INDICES[index_name].get("yahoo_ticker")
    # Then check general mapping
    return INDEX_YAHOO_TICKERS.get(index_name)


def get_nse_name(index_name):
    """Get NSE API name for fetching constituents."""
    return INDEX_NSE_NAMES.get(index_name, index_name)


def is_equity_asset(asset_class):
    """Check if asset class invests in stocks (not commodity/bond ETF)."""
    info = ASSET_CLASS_INDICES.get(asset_class, {})
    return info.get("type") in ["largecap", "midcap", "smallcap"]


def get_stock_count(asset_class):
    """Get number of stocks to buy for an asset class."""
    info = ASSET_CLASS_INDICES.get(asset_class, {})
    return info.get("stock_count", 0)


def get_etf(asset_class):
    """Get ETF ticker for an asset class."""
    info = ASSET_CLASS_INDICES.get(asset_class, {})
    return info.get("etf")
