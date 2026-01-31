"""
Multi-Asset Rotation Backtest System
=====================================
Two modes:
1. Asset Class Rotation - Rotate between 6 asset classes (ETF or Stock mode)
2. Sector Rotation - Rotate between sectoral/thematic indices

Features:
- Dual scoring boxes (Index & Stock)
- Regime filters (applied to index only)
- Position sizing options
- Configurable rebalancing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

from portfolio_engine import RotationEngine
from scoring import ScoreParser
from indices_universe import (
    ASSET_CLASS_ORDER, ASSET_CLASS_INDICES, SECTORAL_INDICES, THEMATIC_INDICES,
    get_etf, get_stock_count
)

# Page config
st.set_page_config(
    page_title="Multi-Asset Rotation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for dark theme styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: #1e1e1e;
        border-radius: 6px;
        padding: 0 20px;
        border: 1px solid #333;
        color: #fff !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #28a745;
        color: white !important;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border: 1px solid #2a2a4a;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00ff88;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# Header
st.markdown("### 🔄 Multi-Asset Rotation Backtest")

# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode Selection
    st.subheader("🎯 Rotation Mode")
    rotation_mode = st.radio(
        "Select Mode",
        ["Asset Class Rotation", "Sector Rotation"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Capital
    st.subheader("💰 Capital")
    initial_capital = st.number_input(
        "Starting Capital (₹)",
        min_value=10000,
        max_value=100000000,
        value=200000,
        step=10000,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Mode-specific options
    if rotation_mode == "Asset Class Rotation":
        st.subheader("📊 Asset Class Options")
        
        # Investment Type
        investment_type = st.radio(
            "Investment Type",
            ["ETF", "Stock"],
            horizontal=True,
            help="ETF: Invest in ETFs directly | Stock: Invest in individual stocks within asset class"
        )
        
        # Show asset class list
        with st.expander("📈 Asset Classes", expanded=False):
            st.markdown("""
            | Asset Class | ETF | Stocks |
            |-------------|-----|--------|
            | NIFTY 100 (Large Cap) | NIF100BEES | 10 |
            | NIFTY MIDCAP 150 | MID150BEES | 15 |
            | NIFTY SMLCAP 250 | GROWWSC250 | 20 |
            | GOLD | GOLDBEES | - |
            | SILVER | SILVERBEES | - |
            | GILT 5Y | GILT5YBEES | - |
            """)
    
    else:  # Sector Rotation
        st.subheader("📊 Sector Options")
        
        index_type = st.radio(
            "Index Type",
            ["Sectoral", "Thematic"],
            horizontal=True
        )
        
        # Stock parameters
        col1, col2 = st.columns(2)
        with col1:
            num_stocks = st.number_input("Stocks to Buy", 5, 30, 10)
        with col2:
            exit_rank = st.number_input("Exit Rank", 10, 50, 15)
        
        # Show index list
        indices_list = SECTORAL_INDICES if index_type == "Sectoral" else THEMATIC_INDICES
        with st.expander(f"📈 {index_type} Indices ({len(indices_list)})", expanded=False):
            for idx in indices_list:
                st.text(f"• {idx}")
        
        investment_type = "Stock"  # Sector rotation always uses stock scoring
    
    st.markdown("---")
    
    # Index Scoring
    st.subheader("📐 Index Scoring")
    parser = ScoreParser()
    examples = parser.get_example_formulas()
    
    index_template = st.selectbox(
        "Template",
        ["Custom"] + list(examples.keys()),
        key="index_template"
    )
    
    default_index_formula = examples.get(index_template, "6 Month Performance")
    index_formula = st.text_area(
        "Index Formula",
        default_index_formula,
        height=80,
        key="index_formula",
        help="Formula to score and rank asset classes/indices"
    )
    
    # Validate index formula
    valid, msg = parser.validate_formula(index_formula)
    if valid:
        st.success("✅ Index formula valid")
    else:
        st.error(f"❌ {msg}")
    
    st.markdown("---")
    
    # Stock Scoring (only if Stock mode or Sector Rotation)
    if (rotation_mode == "Asset Class Rotation" and investment_type == "Stock") or rotation_mode == "Sector Rotation":
        st.subheader("📈 Stock Scoring")
        
        stock_template = st.selectbox(
            "Template",
            ["Custom"] + list(examples.keys()),
            key="stock_template"
        )
        
        default_stock_formula = examples.get(stock_template, "6 Month Performance / 3 Month Volatility")
        stock_formula = st.text_area(
            "Stock Formula",
            default_stock_formula,
            height=80,
            key="stock_formula",
            help="Formula to score and rank individual stocks"
        )
        
        valid_stock, msg_stock = parser.validate_formula(stock_formula)
        if valid_stock:
            st.success("✅ Stock formula valid")
        else:
            st.error(f"❌ {msg_stock}")
        
        st.markdown("---")
    else:
        stock_formula = None
    
    # Time Period
    st.subheader("📅 Time Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime.date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End", datetime.date.today())
    
    st.markdown("---")
    
    # Rebalancing
    st.subheader("🔄 Rebalancing")
    rebal_freq = st.selectbox(
        "Frequency",
        ["Weekly", "Every 2 Weeks", "Monthly", "Bi-Monthly", "Quarterly", "Half-Yearly", "Annually"],
        index=2
    )
    
    if rebal_freq in ["Weekly", "Every 2 Weeks"]:
        rebal_day = st.selectbox(
            "Day",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        )
        rebal_date = None
    else:
        rebal_date = st.number_input("Date (1-28)", 1, 28, 1)
        rebal_day = None
    
    alt_day = st.selectbox(
        "If Holiday",
        ["Previous Day", "Next Day"],
        index=1
    )
    
    st.markdown("---")
    
    # Regime Filter
    st.subheader("🛡️ Regime Filter")
    use_regime = st.checkbox("Enable Regime Filter", value=False)
    
    regime_config = None
    if use_regime:
        regime_type = st.selectbox(
            "Type",
            ["EMA_1D", "EMA_1W", "EMA_1M", "SMA_1D", "SMA_1W", "SMA_1M",
             "MACD", "SUPERTREND_1D", "SUPERTREND_1W", "SUPERTREND_1M"]
        )
        
        if regime_type.startswith("EMA"):
            regime_value = st.selectbox("EMA Period", [34, 68, 100, 150, 200], index=1)
        elif regime_type.startswith("SMA"):
            regime_value = st.selectbox("SMA Period", [20, 50, 100, 150, 200], index=1)
        elif regime_type == "MACD":
            regime_value = st.selectbox("MACD Settings", ["35-70-12", "50-100-15"])
        else:  # SuperTrend
            regime_value = st.selectbox("SuperTrend (Period-Mult)", ["7-2", "7-3", "10-2", "10-3"], index=1)
        
        regime_action = st.selectbox("Action", ["Go Cash", "Half Portfolio"])
        
        regime_config = {
            'enabled': True,
            'type': regime_type,
            'value': regime_value,
            'action': regime_action
        }
        
        st.info("ℹ️ Regime filter applies to INDEX only, not individual stocks")
    
    st.markdown("---")
    
    # Position Sizing
    st.subheader("📊 Position Sizing")
    position_sizing = st.selectbox(
        "Method",
        ["Equal Weight", "Inverse Volatility", "Score-Weighted"],
        help="Within each asset class or for final stocks"
    )
    
    st.markdown("---")
    
    # Run Button
    run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)


# ==================== MAIN CONTENT ====================
if run_backtest:
    # Validate formulas
    if not valid:
        st.error("Please fix the index scoring formula before running")
    elif (rotation_mode == "Asset Class Rotation" and investment_type == "Stock") or rotation_mode == "Sector Rotation":
        if not valid_stock:
            st.error("Please fix the stock scoring formula before running")
    else:
        # Create rebalancing config
        rebal_config = {
            'frequency': rebal_freq,
            'date': rebal_date,
            'day': rebal_day,
            'alt_day': alt_day
        }
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, ticker):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Loading {ticker}... ({current}/{total})")
        
        try:
            with st.spinner("Initializing backtest engine..."):
                engine = RotationEngine(
                    mode='asset_class' if rotation_mode == "Asset Class Rotation" else 'sector',
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital
                )
            
            # Fetch data based on mode
            if rotation_mode == "Asset Class Rotation":
                status_text.text("Fetching asset class index data...")
                indices = list(ASSET_CLASS_INDICES.keys())
                engine.fetch_index_data(indices, update_progress)
                
                if investment_type == "ETF":
                    status_text.text("Fetching ETF data...")
                    etfs = [info['etf'] for info in ASSET_CLASS_INDICES.values()]
                    engine.fetch_etf_data(etfs, update_progress)
                else:
                    # Fetch stock data for equity asset classes
                    status_text.text("Fetching stock data...")
                    # For demo, we'll use a sample universe
                    # In production, would fetch from NSE
                    from nifty_universe import get_universe
                    stocks = get_universe("NIFTY 100")[:50]
                    engine.fetch_stock_data(stocks, update_progress)
                
                # Run backtest
                status_text.text("Running backtest...")
                engine.run_asset_class_backtest(
                    index_formula=index_formula,
                    stock_formula=stock_formula,
                    investment_type=investment_type,
                    rebal_config=rebal_config,
                    regime_config=regime_config,
                    position_sizing=position_sizing.lower().replace(' ', '_')
                )
            
            else:  # Sector Rotation
                status_text.text("Fetching sector index data...")
                indices = SECTORAL_INDICES if index_type == "Sectoral" else THEMATIC_INDICES
                engine.fetch_index_data(indices, update_progress)
                
                status_text.text("Fetching stock data...")
                # In production, would merge from top 5 indices
                from nifty_universe import get_universe
                stocks = get_universe("NIFTY 500")[:100]
                engine.fetch_stock_data(stocks, update_progress)
                
                # Run backtest
                status_text.text("Running backtest...")
                engine.run_sector_rotation_backtest(
                    index_type=index_type.lower(),
                    index_formula=index_formula,
                    stock_formula=stock_formula,
                    num_stocks=num_stocks,
                    exit_rank=exit_rank,
                    rebal_config=rebal_config,
                    regime_config=regime_config,
                    position_sizing=position_sizing.lower().replace(' ', '_')
                )
            
            # Store results
            st.session_state.backtest_results = engine.get_results()
            
            progress_bar.progress(1.0)
            status_text.text("✅ Backtest complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Backtest error: {e}")
            import traceback
            st.code(traceback.format_exc())

# Display results
if st.session_state.backtest_results:
    results = st.session_state.backtest_results
    metrics = results['metrics']
    trades = results['trades']
    equity_df = results['equity_df']
    
    st.markdown("---")
    st.subheader("📊 Backtest Results")
    
    # Metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("CAGR", f"{metrics.get('cagr', 0):.1f}%")
    with col2:
        st.metric("Sharpe", f"{metrics.get('sharpe', 0):.2f}")
    with col3:
        st.metric("Max DD", f"{metrics.get('max_drawdown', 0):.1f}%")
    with col4:
        st.metric("Volatility", f"{metrics.get('volatility', 0):.1f}%")
    with col5:
        st.metric("Total Return", f"{metrics.get('total_return', 0):.1f}%")
    with col6:
        st.metric("Trades", metrics.get('total_trades', 0))
    
    # Equity Curve
    st.subheader("📈 Equity Curve")
    if equity_df is not None and not equity_df.empty:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Portfolio Value", "Drawdown")
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                mode='lines',
                name='Portfolio',
                line=dict(color='#00ff88', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=-equity_df['drawdown'] * 100,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='#ff4444', width=1)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            template='plotly_dark',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        fig.update_yaxes(title_text="₹", row=1, col=1)
        fig.update_yaxes(title_text="%", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Returns Heatmap
    monthly_returns = results.get('monthly_returns')
    if monthly_returns is not None and len(monthly_returns) > 0:
        st.subheader("📅 Monthly Returns")
        
        # Create heatmap data
        monthly_df = monthly_returns.reset_index()
        monthly_df.columns = ['date', 'return']
        monthly_df['year'] = monthly_df['date'].dt.year
        monthly_df['month'] = monthly_df['date'].dt.month
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot = pivot * 100  # Convert to percentage
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f'{v:.1f}%' if not pd.isna(v) else '' for v in row] for row in pivot.values],
            texttemplate='%{text}',
            textfont=dict(size=10),
            hovertemplate='%{y} %{x}: %{z:.1f}%<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            height=300,
            template='plotly_dark',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Trade History
    st.subheader("📝 Trade History")
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
        trades_df['value'] = trades_df['value'].apply(lambda x: f"₹{x:,.0f}")
        trades_df['price'] = trades_df['price'].apply(lambda x: f"₹{x:,.2f}")
        
        # Show last 50 trades
        st.dataframe(
            trades_df[['date', 'type', 'ticker', 'shares', 'price', 'value', 'reason']].tail(50),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No trades executed")

else:
    # Welcome screen
    st.markdown("""
    ### Welcome to Multi-Asset Rotation Backtest! 🎯
    
    Configure your strategy using the sidebar and click **Run Backtest** to start.
    
    **Available Modes:**
    
    | Mode | Description |
    |------|-------------|
    | 🏢 **Asset Class Rotation** | Rotate between 6 asset classes (Large/Mid/Small Cap, Gold, Silver, Bonds) |
    | 📊 **Sector Rotation** | Rotate between 19 Sectoral or 25 Thematic indices |
    
    **Key Features:**
    - ✅ Dual scoring formulas (Index + Stock)
    - ✅ ETF or Stock investment modes
    - ✅ Regime filters (EMA, SMA, MACD, SuperTrend)
    - ✅ Configurable rebalancing (Weekly to Annually)
    - ✅ Position sizing options
    """)
    
    # Show sample asset classes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Asset Classes (Mode 1)")
        for ac in ASSET_CLASS_ORDER:
            info = ASSET_CLASS_INDICES[ac]
            st.text(f"• {ac} → {info['etf']}")
    
    with col2:
        st.markdown("#### 📊 Sample Sectoral Indices (Mode 2)")
        for idx in SECTORAL_INDICES[:8]:
            st.text(f"• {idx}")
        st.text("... and more")
