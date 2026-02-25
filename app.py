"""
Multi-Asset Rotation Backtest System
Ported from investing-scanner.streamlit.app
Features: 2-col layout, Monte Carlo, advanced metrics, benchmark comparison,
          backtest logs, strategy save/load, Zerodha charges, regime analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import io

from portfolio_engine import RotationEngine
from scoring import ScoreParser
from indices_universe import (
    ASSET_CLASS_ORDER, ASSET_CLASS_INDICES, SECTORAL_INDICES, THEMATIC_INDICES,
    get_etf, get_stock_count
)
from strategy_storage import (
    save_strategy, load_strategies, delete_strategy, get_strategy, get_strategy_names
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Asset Rotation",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🔄"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 0.7rem; padding-bottom: 0rem; }
.stTabs [data-baseweb="tab-list"] { gap: 3px; }
.stTabs [data-baseweb="tab"] {
    height: 38px; background-color: #1a1a2e; border-radius: 6px;
    padding: 0 16px; border: 1px solid #2a2a4a;
    color: #ccc !important; font-weight: 500; font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background-color: #00ff88 !important; color: #0e1117 !important;
    font-weight: 700;
}
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a; border-radius: 10px;
    padding: 12px 16px;
}
.config-panel {
    background: #111827; border: 1px solid #1f2d40;
    border-radius: 12px; padding: 16px; height: 100%;
}
.section-header {
    font-size: 13px; font-weight: 600; color: #00ff88;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin: 12px 0 6px 0; border-bottom: 1px solid #1f2d40; padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ('backtest_results', None),
    ('backtest_logs', []),
    ('current_backtest', None),
    ('current_backtest_active', False),
    ('benchmark_selection', 'NIFTY 50'),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── SCORING TEMPLATES ────────────────────────────────────────────────────────
INDEX_TEMPLATES = {
    "6-Month Momentum": "6 Month Performance",
    "12-Month Momentum": "12 Month Performance",
    "3M + 6M + 12M": "(3 Month Performance + 6 Month Performance + 12 Month Performance) / 3",
    "Risk-Adjusted (3M/Vol)": "3 Month Performance / Annualized Volatility",
    "Dual Momentum": "0.5 * 6 Month Performance + 0.5 * 12 Month Performance",
    "Multi-Period": "0.25*1 Month Performance + 0.25*3 Month Performance + 0.25*6 Month Performance + 0.25*12 Month Performance",
}
STOCK_TEMPLATES = {
    "6-Month Momentum": "6 Month Performance",
    "Risk-Adjusted": "6 Month Performance / Annualized Volatility",
    "Dual Momentum": "0.5 * 6 Month Performance + 0.5 * 12 Month Performance",
    "Multi-Period": "0.25*1 Month Performance + 0.25*3 Month Performance + 0.5*6 Month Performance",
}
REGIME_TYPES = {
    "EMA (Daily)": "EMA_1D", "EMA (Weekly)": "EMA_1W", "EMA (Monthly)": "EMA_1M",
    "SMA (Daily)": "SMA_1D", "SMA (Weekly)": "SMA_1W", "SMA (Monthly)": "SMA_1M",
    "MACD": "MACD",
    "SuperTrend (Daily)": "SUPERTREND_1D", "SuperTrend (Weekly)": "SUPERTREND_1W",
    "SuperTrend (Monthly)": "SUPERTREND_1M",
}
YAHOO_BENCHMARK_MAP = {
    "NIFTY 50": "^NSEI", "NIFTY NEXT 50": "^NSMIDCP", "NIFTY 100": "^CNX100",
    "NIFTY 200": "^CNX200", "NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT",
    "NIFTY MIDCAP 50": "^NIFTYMIDCAP50", "NIFTY MIDCAP 100": "^CNXMDCP",
    "NIFTY SMLCAP 100": "^CNXSC", "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA", "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL", "NIFTY REALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY", "NIFTY INFRA": "^CNXINFRA",
    "NIFTY PSU BANK": "^CNXPSUBANK", "NIFTY PRIVATE BANK": "^NIFTYPVTBANK",
}

# ─── Helper ───────────────────────────────────────────────────────────────────
parser = ScoreParser()

def color_metric(val, positive_good=True):
    if isinstance(val, str):
        return val
    return "🟢" if (val > 0) == positive_good else "🔴"

def make_excel_download(engine, metrics, result_name):
    """Create in-memory Excel workbook with backtest data."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = [(k, v) for k, v in metrics.items() if not isinstance(v, (dict, list))]
        pd.DataFrame(summary_data, columns=['Metric', 'Value']).to_excel(writer, sheet_name='Summary', index=False)
        # Equity curve
        if hasattr(engine, 'portfolio_df') and engine.portfolio_df is not None:
            engine.portfolio_df[['Portfolio Value']].to_excel(writer, sheet_name='Equity Curve')
        # Trades
        if hasattr(engine, 'trades_df') and not engine.trades_df.empty:
            engine.trades_df.to_excel(writer, sheet_name='Trades', index=False)
        # Monthly returns
        if hasattr(engine, 'monthly_returns') and engine.monthly_returns is not None:
            engine.monthly_returns.to_frame('Monthly Return').to_excel(writer, sheet_name='Monthly Returns')
    buf.seek(0)
    return buf

# ─── MAIN TABS ────────────────────────────────────────────────────────────────
st.markdown("## 🔄 Multi-Asset Rotation Backtest")
main_tabs = st.tabs(["📊 Backtest", "📁 Backtest Logs", "💾 Strategies"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with main_tabs[0]:
    # ── 2-column layout: Config (left) | Results (right) ──────────────────────
    cfg_col, res_col = st.columns([1, 2], gap="medium")

    # ─── LEFT: CONFIGURATION ──────────────────────────────────────────────────
    with cfg_col:
        st.markdown('<div class="config-panel">', unsafe_allow_html=True)

        # ── Mode ──────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🎯 Rotation Mode</div>', unsafe_allow_html=True)
        rotation_mode = st.radio("Mode", ["Asset Class Rotation", "Sector Rotation"],
                                 horizontal=True, label_visibility="collapsed")

        # ── Date & Capital ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📅 Period & Capital</div>', unsafe_allow_html=True)
        dc1, dc2 = st.columns(2)
        start_date = dc1.date_input("Start", datetime.date(2018, 1, 1), key="start_date")
        end_date = dc2.date_input("End", datetime.date.today(), key="end_date")
        initial_capital = st.number_input("Capital (₹)", min_value=10000, value=200000,
                                          step=10000, format="%d")
        reinvest_profits = st.checkbox("Reinvest Profits", value=True,
                                       help="Compound returns by reinvesting gains")

        # ── Mode-specific config ───────────────────────────────────────────────
        if rotation_mode == "Asset Class Rotation":
            st.markdown('<div class="section-header">📦 Asset Class Config</div>', unsafe_allow_html=True)
            investment_type = st.radio("Investment Type", ["ETF", "Stock"],
                                       horizontal=True, label_visibility="collapsed")
        else:
            st.markdown('<div class="section-header">📦 Sector Config</div>', unsafe_allow_html=True)
            index_type = st.radio("Index Type", ["Sectoral", "Thematic"],
                                   horizontal=True, label_visibility="collapsed")
            sc1, sc2 = st.columns(2)
            num_stocks = sc1.number_input("# Stocks", min_value=3, max_value=30, value=10)
            exit_rank = sc2.number_input("Exit Rank", min_value=5, max_value=50, value=15)

        # ── Index Scoring ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📈 Index Scoring Formula</div>', unsafe_allow_html=True)
        idx_tmpl = st.selectbox("Template", ["Custom"] + list(INDEX_TEMPLATES.keys()),
                                 key="idx_template", label_visibility="collapsed")
        default_idx_formula = INDEX_TEMPLATES.get(idx_tmpl, "6 Month Performance") if idx_tmpl != "Custom" else "6 Month Performance"
        index_formula = st.text_area("Index Formula", value=default_idx_formula,
                                     height=68, key="idx_formula")
        idx_ok, idx_msg = parser.validate_formula(index_formula)
        if not idx_ok:
            st.error(f"⚠️ {idx_msg}")
        else:
            st.success("✅ Valid formula")

        # ── Stock Scoring (shown when relevant) ───────────────────────────────
        show_stock_formula = (rotation_mode == "Sector Rotation") or \
                             (rotation_mode == "Asset Class Rotation" and 'investment_type' in dir() and investment_type == "Stock")
        if show_stock_formula or rotation_mode == "Sector Rotation":
            st.markdown('<div class="section-header">📊 Stock Scoring Formula</div>', unsafe_allow_html=True)
            stk_tmpl = st.selectbox("Stock Template", ["Custom"] + list(STOCK_TEMPLATES.keys()),
                                     key="stk_template", label_visibility="collapsed")
            default_stk = STOCK_TEMPLATES.get(stk_tmpl, "6 Month Performance") if stk_tmpl != "Custom" else "6 Month Performance"
            stock_formula = st.text_area("Stock Formula", value=default_stk,
                                         height=68, key="stk_formula")
            stk_ok, stk_msg = parser.validate_formula(stock_formula)
            if not stk_ok:
                st.error(f"⚠️ {stk_msg}")
            else:
                st.success("✅ Valid formula")
        else:
            stock_formula = None

        # ── Rebalancing ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🔁 Rebalancing</div>', unsafe_allow_html=True)
        freq = st.selectbox("Frequency",
                             ["Weekly", "Every 2 Weeks", "Monthly", "Bi-Monthly",
                              "Quarterly", "Half-Yearly", "Annually"],
                             index=2, label_visibility="collapsed")
        if freq in ["Weekly", "Every 2 Weeks"]:
            rebal_day = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday"])
            rebal_config = {'frequency': freq, 'day': rebal_day}
        else:
            rb1, rb2 = st.columns(2)
            rebal_date = rb1.number_input("Date", 1, 28, 1)
            alt_day = rb2.selectbox("If Holiday", ["Next Day", "Previous Day"])
            rebal_config = {'frequency': freq, 'date': rebal_date, 'alt_day': alt_day}

        # ── Position Sizing ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">⚖️ Position Sizing</div>', unsafe_allow_html=True)
        position_sizing = st.selectbox(
            "Method",
            ["equal_weight", "inverse_volatility", "score_weighted",
             "inverse_drawdown", "risk_parity"],
            format_func=lambda x: {
                "equal_weight": "Equal Weight",
                "inverse_volatility": "Inverse Volatility",
                "score_weighted": "Score Weighted",
                "inverse_drawdown": "Inverse Max Drawdown",
                "risk_parity": "Risk Parity",
            }.get(x, x),
            label_visibility="collapsed"
        )
        use_position_cap = st.checkbox("Max Position Cap", value=False)
        if use_position_cap:
            max_position_pct = st.slider("Cap %", 10, 100, 50, 5)
        else:
            max_position_pct = 100

        # ── Regime Filter ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🛡️ Regime Filter</div>', unsafe_allow_html=True)
        use_regime = st.checkbox("Enable Regime Filter", value=False)
        regime_config = {'enabled': False}
        if use_regime:
            regime_label = st.selectbox("Filter Type", list(REGIME_TYPES.keys()),
                                         label_visibility="collapsed")
            regime_type = REGIME_TYPES[regime_label]
            regime_action = st.selectbox("Action When Triggered",
                                          ["Go Cash", "Hold", "Reduce Exposure 50%"])
            regime_config = {
                'enabled': True,
                'type': regime_type,
                'action': regime_action,
            }

        # ── Run Button ─────────────────────────────────────────────────────────
        st.markdown("---")
        bt_name = st.text_input("Backtest Name (optional)", placeholder="My Strategy v1")
        run_button = st.button("🚀 Run Backtest", use_container_width=True, type="primary")

        st.markdown('</div>', unsafe_allow_html=True)

    # ─── RIGHT: RESULTS ────────────────────────────────────────────────────────
    with res_col:
        if run_button:
            # Validate
            if not idx_ok:
                st.error("Fix the index formula before running.")
                st.stop()

            with st.spinner("⏳ Fetching data and running backtest…"):
                progress_bar = st.progress(0, text="Initializing…")
                t0 = time.time()

                engine = RotationEngine(
                    mode=rotation_mode.lower().replace(" ", "_").replace("_rotation", "").strip(),
                    start_date=str(start_date),
                    end_date=str(end_date),
                    initial_capital=initial_capital,
                    use_cache=True
                )

                # Determine indices to load
                if rotation_mode == "Asset Class Rotation":
                    all_indices = list(ASSET_CLASS_INDICES.keys())
                else:
                    all_indices = SECTORAL_INDICES if index_type == "Sectoral" else THEMATIC_INDICES

                # Fetch index data
                total_tickers = len(all_indices)
                def prog_cb(cur, tot, name):
                    pct = int(cur / tot * 50)
                    elapsed = time.time() - t0
                    progress_bar.progress(pct, text=f"📥 Fetching {name}… ({cur}/{tot}) | {elapsed:.0f}s")

                ok = engine.fetch_index_data(all_indices, progress_callback=prog_cb)
                if not ok:
                    st.error("Failed to fetch index data.")
                    st.stop()

                # Fetch ETF/stock data for asset class mode
                if rotation_mode == "Asset Class Rotation":
                    if investment_type == "ETF":
                        etfs = [ASSET_CLASS_INDICES[ac]['etf'] for ac in all_indices
                                if 'etf' in ASSET_CLASS_INDICES[ac]]
                        progress_bar.progress(55, text="📥 Fetching ETFs…")
                        engine.fetch_etf_data(etfs)
                    else:
                        progress_bar.progress(55, text="📥 Fetching stocks…")
                        # Stock universe would be loaded here

                progress_bar.progress(70, text="🧮 Running backtest…")

                # Run backtest
                if rotation_mode == "Asset Class Rotation":
                    engine.run_asset_class_backtest(
                        index_formula=index_formula,
                        stock_formula=stock_formula,
                        investment_type=investment_type,
                        rebal_config=rebal_config,
                        regime_config=regime_config,
                        position_sizing=position_sizing
                    )
                else:
                    engine.run_sector_rotation_backtest(
                        index_type=index_type.lower(),
                        index_formula=index_formula,
                        stock_formula=stock_formula,
                        num_stocks=int(num_stocks),
                        exit_rank=int(exit_rank),
                        rebal_config=rebal_config,
                        regime_config=regime_config,
                        position_sizing=position_sizing
                    )

                progress_bar.progress(100, text="✅ Done!")
                time.sleep(0.3)
                progress_bar.empty()

            metrics = engine.get_metrics()
            if not metrics:
                st.error("Backtest produced no results. Check data availability.")
                st.stop()

            # Store results
            log_name = bt_name.strip() or f"{rotation_mode} — {start_date} to {end_date}"
            log_entry = {
                'name': log_name,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                'config': {
                    'rotation_mode': rotation_mode,
                    'start_date': str(start_date),
                    'end_date': str(end_date),
                    'initial_capital': initial_capital,
                    'formula': index_formula,
                    'rebal_config': rebal_config,
                    'regime_config': regime_config,
                    'position_sizing': position_sizing,
                    'universe_name': rotation_mode,
                },
                'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
            }
            st.session_state.backtest_logs.append(log_entry)
            st.session_state.current_backtest = {'engine': engine, 'start_date': start_date, 'end_date': end_date, 'metrics': metrics}
            st.session_state.current_backtest_active = True
            st.session_state.backtest_results = engine

        # ── Display results if available ───────────────────────────────────────
        if st.session_state.current_backtest_active and st.session_state.current_backtest:
            stored = st.session_state.current_backtest
            engine = stored['engine']
            metrics = stored['metrics']

            # ── Quick metrics strip ────────────────────────────────────────────
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Final Value", f"₹{metrics.get('Final Value', 0):,.0f}")
            m2.metric("Total Return", f"{metrics.get('Return %', 0):.1f}%")
            m3.metric("CAGR", f"{metrics.get('CAGR %', 0):.2f}%")
            m4.metric("Sharpe", f"{metrics.get('Sharpe Ratio', 0):.2f}")
            m5.metric("Max DD", f"{metrics.get('Max Drawdown %', 0):.1f}%")
            m6.metric("Win Rate", f"{metrics.get('Win Rate %', 0):.1f}%")

            # ── Download button ────────────────────────────────────────────────
            dl_col1, dl_col2, _ = st.columns([1, 1, 4])
            excel_buf = make_excel_download(engine, metrics, "backtest")
            dl_col1.download_button("📥 Excel", excel_buf,
                                    file_name=f"backtest_{datetime.date.today()}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True)
            if hasattr(engine, 'trades_df') and not engine.trades_df.empty:
                csv_data = engine.trades_df.to_csv(index=False).encode()
                dl_col2.download_button("📥 Trades CSV", csv_data,
                                        file_name="trades.csv", mime="text/csv",
                                        use_container_width=True)

            # ── Result tabs ────────────────────────────────────────────────────
            tab_names = ["📋 Performance", "📈 Charts", "📅 Monthly", "📝 Trades", "🎲 Monte Carlo", "📊 Benchmark"]
            result_tabs = st.tabs(tab_names)

            # ═══ Tab 1: Performance ════════════════════════════════════════════
            with result_tabs[0]:
                st.markdown("### 📊 Performance Summary")
                pa, pb = st.columns(2)
                with pa:
                    st.markdown("**Returns**")
                    perf_data = {
                        "Initial Capital": f"₹{metrics.get('Initial Capital', 0):,.0f}",
                        "Final Value": f"₹{metrics.get('Final Value', 0):,.0f}",
                        "Total Return": f"{metrics.get('Return %', 0):.2f}%",
                        "CAGR": f"{metrics.get('CAGR %', 0):.2f}%",
                    }
                    st.table(pd.DataFrame(list(perf_data.items()), columns=["Metric", "Value"]).set_index("Metric"))

                with pb:
                    st.markdown("**Risk**")
                    risk_data = {
                        "Sharpe Ratio": f"{metrics.get('Sharpe Ratio', 0):.2f}",
                        "Sortino Ratio": f"{metrics.get('Sortino Ratio', 0):.2f}",
                        "Max Drawdown": f"{metrics.get('Max Drawdown %', 0):.2f}%",
                        "CVaR 5%": f"{metrics.get('CVaR 5%', 0):.2f}%",
                        "Volatility": f"{metrics.get('Volatility %', 0):.2f}%",
                        "Days to Recover": f"{metrics.get('Days to Recover from DD', 0)}",
                    }
                    st.table(pd.DataFrame(list(risk_data.items()), columns=["Metric", "Value"]).set_index("Metric"))

                st.markdown("---")
                st.markdown("**Trade Statistics**")
                ta, tb, tc = st.columns(3)
                with ta:
                    st.metric("Total Trades", metrics.get('Total Trades', 0))
                    st.metric("Win Rate", f"{metrics.get('Win Rate %', 0):.1f}%")
                    st.metric("Expectancy", f"₹{metrics.get('Expectancy', 0):,.0f}")
                with tb:
                    st.metric("Avg Win", f"₹{metrics.get('Avg Win', 0):,.0f}")
                    st.metric("Avg Loss", f"₹{metrics.get('Avg Loss', 0):,.0f}")
                    st.metric("Turnover", f"{metrics.get('Turnover', 0):.2f}x/yr")
                with tc:
                    st.metric("Max Consec. Wins", metrics.get('Max Consecutive Wins', 0))
                    st.metric("Max Consec. Losses", metrics.get('Max Consecutive Losses', 0))
                    st.metric("Total Charges", f"₹{metrics.get('Total Charges', 0):,.0f}")

                st.markdown("---")
                st.markdown("**Risk Metrics (MAE)**")
                ma_a, ma_b, ma_c = st.columns(3)
                ma_a.metric("MAE Median", f"{metrics.get('MAE Median %', 0):.2f}%")
                ma_b.metric("MAE 95th %ile", f"{metrics.get('MAE 95% %', 0):.2f}%")
                ma_c.metric("MAE Max", f"{metrics.get('MAE Max %', 0):.2f}%")

            # ═══ Tab 2: Charts ════════════════════════════════════════════════
            with result_tabs[1]:
                if hasattr(engine, 'portfolio_df') and engine.portfolio_df is not None:
                    pdf = engine.portfolio_df

                    # Equity curve
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=pdf.index, y=pdf['Portfolio Value'],
                        name='Portfolio', fill='tozeroy',
                        line=dict(color='#00ff88', width=2),
                        fillcolor='rgba(0,255,136,0.08)'
                    ))
                    fig_eq.update_layout(
                        title="Equity Curve", xaxis_title="Date",
                        yaxis_title="Portfolio Value (₹)",
                        height=400, template='plotly_dark',
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

                    # Drawdown chart
                    if 'Drawdown_Pct' in pdf.columns:
                        fig_dd = go.Figure()
                        fig_dd.add_trace(go.Scatter(
                            x=pdf.index, y=-pdf['Drawdown_Pct'],
                            fill='tozeroy', name='Drawdown %',
                            line=dict(color='#dc3545', width=1),
                            fillcolor='rgba(220,53,69,0.2)'
                        ))
                        fig_dd.update_layout(
                            title="Drawdown from Peak",
                            height=300, template='plotly_dark',
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        st.plotly_chart(fig_dd, use_container_width=True)

            # ═══ Tab 3: Monthly Breakup ═══════════════════════════════════════
            with result_tabs[2]:
                if hasattr(engine, 'monthly_returns') and engine.monthly_returns is not None:
                    mr = engine.monthly_returns
                    # Build heatmap
                    heat_data = {}
                    for date, val in mr.items():
                        y, m = date.year, date.strftime('%b')
                        if y not in heat_data:
                            heat_data[y] = {}
                        heat_data[y][m] = round(val * 100, 2)

                    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                    years_list = sorted(heat_data.keys())
                    z = [[heat_data.get(y, {}).get(m, None) for m in months] for y in years_list]

                    fig_heat = go.Figure(go.Heatmap(
                        z=z, x=months, y=[str(y) for y in years_list],
                        colorscale=[[0, '#dc3545'], [0.5, '#1a1a2e'], [1, '#28a745']],
                        text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z],
                        texttemplate="%{text}",
                        zmid=0,
                        showscale=True,
                    ))
                    fig_heat.update_layout(
                        title="Monthly Returns Heatmap (%)",
                        height=max(300, 30 * len(years_list) + 100),
                        template='plotly_dark',
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                    # Yearly summary
                    st.markdown("**Yearly Returns**")
                    yearly = {}
                    for date, val in mr.items():
                        y = date.year
                        yearly[y] = yearly.get(y, 1) * (1 + val)
                    yearly_df = pd.DataFrame([
                        {'Year': y, 'Return %': round((v - 1) * 100, 2)}
                        for y, v in sorted(yearly.items())
                    ])
                    st.dataframe(yearly_df, use_container_width=True, hide_index=True)

            # ═══ Tab 4: Trades ════════════════════════════════════════════════
            with result_tabs[3]:
                if hasattr(engine, 'trades_df') and not engine.trades_df.empty:
                    tdf = engine.trades_df.copy()
                    st.markdown(f"**{len(tdf)} total trade records** ({len(tdf[tdf['type']=='BUY'])} buys, {len(tdf[tdf['type']=='SELL'])} sells)")
                    # Format
                    if 'date' in tdf.columns:
                        tdf['date'] = pd.to_datetime(tdf['date']).dt.strftime('%Y-%m-%d')
                    if 'price' in tdf.columns:
                        tdf['price'] = tdf['price'].apply(lambda x: f"₹{x:,.2f}")
                    if 'value' in tdf.columns:
                        tdf['value'] = tdf['value'].apply(lambda x: f"₹{x:,.0f}")
                    st.dataframe(tdf, use_container_width=True, hide_index=True)
                else:
                    st.info("No trades recorded.")

            # ═══ Tab 5: Monte Carlo ══════════════════════════════════════════
            with result_tabs[4]:
                st.markdown("### 🎲 Monte Carlo Analysis")
                if hasattr(engine, 'monthly_returns') and engine.monthly_returns is not None:
                    try:
                        from monte_carlo import run_monte_carlo_portfolio
                        mc_col1, mc_col2 = st.columns(2)
                        n_sims = mc_col1.selectbox("Simulations", [1000, 5000, 10000], index=1)
                        mc_years = mc_col2.number_input("Projection Years", 1, 20, 5)

                        if st.button("▶ Run Monte Carlo"):
                            with st.spinner("Running simulations…"):
                                # Use monthly returns as base
                                returns = engine.monthly_returns.values
                                mc_result = run_monte_carlo_portfolio(
                                    returns=returns,
                                    n_simulations=int(n_sims),
                                    n_periods=int(mc_years * 12),
                                    initial_capital=metrics.get('Initial Capital', 200000)
                                )
                            if mc_result is not None:
                                st.success("Simulation complete!")
                                # Plot fan chart
                                if hasattr(mc_result, 'sim_df'):
                                    sim_df = mc_result.sim_df
                                    fig_mc = go.Figure()
                                    fig_mc.add_trace(go.Scatter(
                                        x=sim_df.index, y=sim_df.quantile(0.95, axis=1),
                                        name='95th %ile', line=dict(color='#28a745', dash='dash')
                                    ))
                                    fig_mc.add_trace(go.Scatter(
                                        x=sim_df.index, y=sim_df.quantile(0.50, axis=1),
                                        name='Median', line=dict(color='#00ff88', width=2)
                                    ))
                                    fig_mc.add_trace(go.Scatter(
                                        x=sim_df.index, y=sim_df.quantile(0.05, axis=1),
                                        name='5th %ile', line=dict(color='#dc3545', dash='dash')
                                    ))
                                    fig_mc.update_layout(
                                        title=f"Monte Carlo: {n_sims} simulations over {mc_years} years",
                                        height=450, template='plotly_dark'
                                    )
                                    st.plotly_chart(fig_mc, use_container_width=True)
                    except ImportError:
                        st.warning("Monte Carlo module not fully configured. Run after monte_carlo.py is set up.")
                    except Exception as e:
                        st.error(f"Monte Carlo error: {e}")
                else:
                    st.info("Run a backtest first to enable Monte Carlo analysis.")

            # ═══ Tab 6: Benchmark ════════════════════════════════════════════
            with result_tabs[5]:
                st.markdown("### 📊 Benchmark Comparison")
                bench_options = list(YAHOO_BENCHMARK_MAP.keys())
                selected_bench = st.selectbox(
                    "Select Benchmark",
                    bench_options,
                    index=bench_options.index(st.session_state.benchmark_selection)
                        if st.session_state.benchmark_selection in bench_options else 0
                )
                st.session_state.benchmark_selection = selected_bench

                try:
                    import yfinance as yf
                    bench_ticker = YAHOO_BENCHMARK_MAP[selected_bench]
                    bench_data = yf.download(bench_ticker,
                                             start=stored['start_date'],
                                             end=stored['end_date'],
                                             progress=False)
                    if not bench_data.empty and hasattr(engine, 'portfolio_df'):
                        pv = engine.portfolio_df['Portfolio Value']
                        port_norm = (pv / pv.iloc[0] - 1) * 100

                        bc = bench_data['Close']
                        if isinstance(bc, pd.DataFrame):
                            bc = bc.iloc[:, 0]
                        bench_norm = (bc / bc.iloc[0] - 1) * 100

                        fig_cmp = go.Figure()
                        fig_cmp.add_trace(go.Scatter(x=port_norm.index, y=port_norm,
                                                     name="Portfolio", line=dict(color="#00ff88", width=2)))
                        fig_cmp.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm,
                                                     name=selected_bench, line=dict(color="#17a2b8", width=2)))
                        fig_cmp.update_layout(
                            title=f"Cumulative Returns: Portfolio vs {selected_bench}",
                            xaxis_title="Date", yaxis_title="Return (%)",
                            height=400, template='plotly_dark'
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True)

                        # Drawdown comparison
                        port_dd = (pv / pv.cummax() - 1) * 100
                        bench_dd = (bc / bc.cummax() - 1) * 100
                        fig_dd2 = go.Figure()
                        fig_dd2.add_trace(go.Scatter(x=port_dd.index, y=port_dd,
                                                     name="Portfolio DD", fill='tozeroy',
                                                     line=dict(color="#00ff88", width=1)))
                        fig_dd2.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd,
                                                     name=f"{selected_bench} DD", fill='tozeroy',
                                                     line=dict(color="#17a2b8", width=1, dash='dot')))
                        fig_dd2.update_layout(title="Drawdown Comparison", height=300, template='plotly_dark')
                        st.plotly_chart(fig_dd2, use_container_width=True)

                        # Summary metrics
                        b1, b2, b3, b4 = st.columns(4)
                        alpha = port_norm.iloc[-1] - bench_norm.iloc[-1]
                        b1.metric("Portfolio Return", f"{port_norm.iloc[-1]:.1f}%")
                        b2.metric(f"{selected_bench} Return", f"{bench_norm.iloc[-1]:.1f}%")
                        b3.metric("Portfolio Max DD", f"{port_dd.min():.1f}%")
                        b4.metric(f"{selected_bench} Max DD", f"{bench_dd.min():.1f}%")
                        if alpha > 0:
                            st.success(f"🎯 **Alpha Generated: +{alpha:.1f}%** vs {selected_bench}")
                        else:
                            st.warning(f"📉 **Alpha: {alpha:.1f}%** vs {selected_bench}")
                    else:
                        st.warning(f"Could not fetch benchmark data for {selected_bench}.")
                except Exception as e:
                    st.error(f"Benchmark error: {e}")

        else:
            st.info("👈 Configure and run a backtest to see results here.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BACKTEST LOGS
# ══════════════════════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.subheader("📁 Backtest History")

    if not st.session_state.backtest_logs:
        st.info("No backtest logs yet. Run a backtest to see results here.")
    else:
        st.markdown(f"**{len(st.session_state.backtest_logs)} backtest(s) recorded this session**")

        if st.button("🗑️ Clear All Logs", key="clear_logs"):
            st.session_state.backtest_logs = []
            st.rerun()

        for idx, log in enumerate(reversed(st.session_state.backtest_logs)):
            with st.expander(f"📊 {log['name']} — {log['timestamp']}", expanded=idx == 0):
                cfg = log['config']
                st.markdown(f"**Mode:** {cfg.get('rotation_mode', '—')} | **Period:** {cfg.get('start_date')} → {cfg.get('end_date')} | **Capital:** ₹{cfg.get('initial_capital', 0):,}")
                st.markdown(f"**Formula:** `{cfg.get('formula', '—')}`")
                st.markdown(f"**Rebalancing:** {cfg.get('rebal_config', {}).get('frequency', '—')} | **Sizing:** {cfg.get('position_sizing', '—')}")

                m = log['metrics']
                log_cols = st.columns(5)
                log_cols[0].metric("Final Value", f"₹{m.get('Final Value', 0):,.0f}")
                log_cols[1].metric("CAGR", f"{m.get('CAGR %', 0):.2f}%")
                log_cols[2].metric("Sharpe", f"{m.get('Sharpe Ratio', 0):.2f}")
                log_cols[3].metric("Max DD", f"{m.get('Max Drawdown %', 0):.2f}%")
                log_cols[4].metric("Win Rate", f"{m.get('Win Rate %', 0):.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: STRATEGIES (Save / Load)
# ══════════════════════════════════════════════════════════════════════════════
with main_tabs[2]:
    st.subheader("💾 Strategy Save / Load")
    s_col1, s_col2 = st.columns(2)

    with s_col1:
        st.markdown("#### 💾 Save Current Config")
        strat_name = st.text_input("Strategy Name", placeholder="My 6M Momentum Strategy")
        if st.button("Save Strategy", use_container_width=True, disabled=not strat_name):
            if st.session_state.current_backtest and strat_name:
                stored_cfg = st.session_state.current_backtest.get('config', {})
                # Build a compact config to save
                save_cfg = {
                    'rotation_mode': rotation_mode if 'rotation_mode' in dir() else 'Asset Class Rotation',
                    'formula': index_formula if 'index_formula' in dir() else '',
                    'rebal_config': rebal_config if 'rebal_config' in dir() else {},
                    'regime_config': regime_config if 'regime_config' in dir() else {},
                    'position_sizing': position_sizing if 'position_sizing' in dir() else 'equal_weight',
                    'initial_capital': initial_capital if 'initial_capital' in dir() else 200000,
                }
                if save_strategy(strat_name, save_cfg):
                    st.success(f"✅ Strategy '{strat_name}' saved!")
                else:
                    st.error("Failed to save strategy.")
            else:
                st.warning("Run a backtest first, then save.")

    with s_col2:
        st.markdown("#### 📂 Saved Strategies")
        strategy_names = get_strategy_names()
        if not strategy_names:
            st.info("No saved strategies yet.")
        else:
            for name in strategy_names:
                cfg_data = get_strategy(name)
                with st.expander(f"📌 {name}"):
                    if cfg_data:
                        st.json(cfg_data)
                    del_btn = st.button(f"🗑️ Delete '{name}'", key=f"del_{name}")
                    if del_btn:
                        delete_strategy(name)
                        st.rerun()
