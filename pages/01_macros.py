"""
pages/01_macros.py
===================
Global Macros Intelligence page.

Sections
--------
01 · Global Financial Markets     yfinance — VIX, SPX, DXY, UST10Y, Brent, Gold, Copper, MSCI EM
02 · War & Armed Conflict          placeholder
03 · Sanctions                     Finnhub news, keyword-filtered
04 · Epidemics                     WHO DON scraper (Playwright)
05 · Tariffs                       Finnhub news, keyword-filtered
06 · National Economic Indicators  Finnhub economic calendar
07 · Central Bank Decisions        placeholder
08 · Global Credit & Contagion     placeholder

All live data is cached to  data/macros_cache/*.feather  with per-section TTLs.
"""

import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Macros · Commodities Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# SHARED THEME CSS  (mirrors dashboard.py)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    height: 88px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
    box-sizing: border-box;
}
.metric-label {
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value {
    font-size: 22px;
    font-weight: 600;
    color: #e6edf3;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-delta-pos { color: #3fb950; font-size: 12px; }
.metric-delta-neg { color: #f85149; font-size: 12px; }

.section-header {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-bottom: 12px;
    margin-top: 4px;
}
.main { background: #0d1117; }
.block-container { padding: 1.5rem 2rem; }
hr { border-color: #21262d; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
from utils.macros import (
    render_global_markets,
    render_epidemics,
    render_sanctions,
    render_war,
    render_tariffs,
    render_econ_calendar,
    render_central_banks,
    render_credit_contagion,
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 Global Macros")
    st.markdown("Cross-asset macro risk signals for commodity market context.")
    st.markdown("---")

    market_period = st.selectbox(
        "Market history window",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
        key="macro_mkt_period",
    )

    st.markdown("---")

    sections = {
        "01 · Global Financial Markets":        True,
        "02 · War":                              True,
        "03 · Sanctions":                        True,
        "04 · Epidemics":                        True,
        "05 · Tariffs":                          True,
        "06 · National Economic Indicators":     True,
        "07 · Central Bank Decisions":           True,
        "08 · Credit & Default Contagion":       True,
    }

    st.markdown("**Show / Hide Sections**")
    vis = {k: st.checkbox(k, value=v, key=f"macro_vis_{i}") for i, (k, v) in enumerate(sections.items())}

    st.markdown("---")
    st.caption(f"Data as of: {__import__('datetime').datetime.today().strftime('%d %b %Y %H:%M')}")
    st.caption("© Commodities Intelligence | SD Guthrie")

# ─────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("# 🌐 Global Macros")
st.markdown(
    '<span style="color:#8b949e;font-family:\'IBM Plex Mono\',monospace;font-size:13px;">'
    'Cross-asset risk signals · Geopolitics · Economic calendar · Epidemics</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 01 — GLOBAL FINANCIAL MARKETS  (full-width)
# ─────────────────────────────────────────────────────────────
if vis.get("01 · Global Financial Markets"):
    render_global_markets(period=market_period)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────
# ROW A — WAR | SANCTIONS  (side by side)
# ─────────────────────────────────────────────────────────────
show_war      = vis.get("02 · War")
show_sanctions = vis.get("03 · Sanctions")

if show_war or show_sanctions:
    col_war, col_sanc = st.columns(2)
    if show_war:
        with col_war:
            render_war()
    if show_sanctions:
        with col_sanc:
            render_sanctions()
    st.markdown("---")

# ─────────────────────────────────────────────────────────────
# ROW B — EPIDEMICS | TARIFFS  (side by side)
# ─────────────────────────────────────────────────────────────
show_epi     = vis.get("04 · Epidemics")
show_tariffs = vis.get("05 · Tariffs")

if show_epi or show_tariffs:
    col_epi, col_tar = st.columns(2)
    if show_epi:
        with col_epi:
            render_epidemics()
    if show_tariffs:
        with col_tar:
            render_tariffs()
    st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 06 — NATIONAL ECONOMIC INDICATORS  (full-width)
# ─────────────────────────────────────────────────────────────
if vis.get("06 · National Economic Indicators"):
    render_econ_calendar()
    st.markdown("---")

# ─────────────────────────────────────────────────────────────
# ROW C — CENTRAL BANKS | CREDIT CONTAGION  (side by side)
# ─────────────────────────────────────────────────────────────
show_cb  = vis.get("07 · Central Bank Decisions")
show_crd = vis.get("08 · Credit & Default Contagion")

if show_cb or show_crd:
    col_cb, col_crd = st.columns(2)
    if show_cb:
        with col_cb:
            render_central_banks()
    if show_crd:
        with col_crd:
            render_credit_contagion()
    st.markdown("---")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<p style="color:#8b949e;font-family:\'IBM Plex Mono\',monospace;font-size:10px;text-align:center;">'
    'INTERNAL USE ONLY · SD Guthrie International · Commodities Intelligence Dashboard · '
    'Market data via Yahoo Finance (delayed) · News via Finnhub · Epidemics via WHO DON'
    '</p>',
    unsafe_allow_html=True,
)
