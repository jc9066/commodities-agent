"""
Commodities Intelligence Dashboard
====================================
Streamlit dashboard covering price, FX, weather, ENSO, supply-demand,
crop calendar, volatility/risk flows, and policy tracker.

Run with:
    pip install streamlit plotly pandas numpy pillow requests
    streamlit run commodities_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
from plotly.subplots import make_subplots # type: ignore
from datetime import datetime, timedelta
import random
from utils.enso import load_oni, get_enso_status, make_oni_chart
from utils.psd_aep import render_aep_tab
from utils.policy import load_policy, render_policy_tracker
from pathlib import Path
from PIL import Image
from utils.weather import (
    make_cached_weather_fetcher,
    PARAM_LABEL,
    PRIMARY_PARAM,
)
from datetime import datetime
from utils.fx_live_monitor import fetch_fx_history, get_fx_snapshot
from utils.atlas_market_data import (
    render_atlas_market_section,
    get_atlas_range_price_metric,
)
from utils.psd import (
    COMMODITY_CODES, ATTRIBUTE_LABELS,
    load_world_sd, load_country_sd,
    make_sd_balance_chart, make_top_countries_chart, make_export_stacked_chart,
)
from utils.psd_cache import ensure_psd_cache
ensure_psd_cache()
from utils.weather import render_weather_section
from datetime import date, timedelta
 
fetch_weather = make_cached_weather_fetcher() 


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Commodity Fundamental Data Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME / CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

    /* Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        height: 95px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
        box-sizing: border-box;
    }
    .metric-label {
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b949e;
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1.3;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #e6edf3;
        font-family: 'IBM Plex Mono', monospace;
        margin: 4px 0;
    }
    .metric-delta-pos { color: #3fb950; font-size: 13px; }
    .metric-delta-neg { color: #f85149; font-size: 13px; }

    /* ENSO badge */
    .enso-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
    }
    .enso-elnino  { background: #f85149; color: #fff; }
    .enso-lanina  { background: #1f6feb; color: #fff; }
    .enso-neutral { background: #8b949e; color: #fff; }

    /* Section headers */
    .section-header {
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8b949e;
        font-family: 'IBM Plex Mono', monospace;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }

    /* Tables */
    .stDataFrame { border: 1px solid #21262d !important; border-radius: 6px; }

    /* Main background */
    .main { background: #0d1117; }
    .block-container { padding: 1.5rem 2rem; }

    /* Divider */
    hr { border-color: #21262d; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COMMODITY UNIVERSE
# ─────────────────────────────────────────────────────────────
COMMODITY_MAP = {
    "Oilseeds": {
        "Crude Palm Oil": {
            "exchanges": ["BMD (Malaysia)"],
            "twin": {"Palm Olein (DCE)", "Bean Oil (CME)", "Bean Oil (DCE)", "Rapeseed Oil (ZCE)",
                     "Brent Crude (CME/ICE)", "WTI Crude (CME/ICE)"},
            "currency": {"BMD (Malaysia)": "MYR/MT"},
            "fx_pair": {"BMD (Malaysia)": "USD/MYR", },
            "top5_producer": ["Indonesia", "Malaysia", "Thailand", "Columbia", "Nigeria"],
        },
        "Palm Olein": {
            "exchanges": ["DCE (China)"],
            "twin": {"Crude Palm Oil (BMD)", "Bean Oil (CME)", "Bean Oil (DCE)", "Rapeseed Oil (ZCE)",
                      "Brent Crude (CME/ICE)", "WTI Crude (CME/ICE)"},
            "currency": {"DCE (China)": "CNY/MT"},
            "fx_pair": {"DCE (China)": "USD/CNY"},
            "top5_producer": ["Indonesia", "Malaysia", "Thailand", "Columbia", "Nigeria"],
        },
        "Soybeans": {
            "exchanges": ["CME (US)", "DCE (China)"],
            "twin": {"Bean Oil (CME)", "Bean Oil (DCE)", "Soymeal (CME)", "Soymeal (DCE)",
                     "Rapeseed (LIFFE/Euronext)"},
            "currency": {"CME (US)": "USc/bu", "DCE (China)": "CNY/MT"},
            "fx_pair": {"CME (US)": "USD/USD", "DCE (China)": "USD/CNY"},
            "top5_producer": ["Brazil", "United States", "Argentina", "China", "Paraguay"],
        },
        "Bean Oil": {
            "exchanges": ["CME (US)", "DCE (China)"],
            "twin": {"Soybeans (CME)", "Soybeans (DCE)", "Rapeseed Oil (ZCE)",
                     "Palm Olein (DCE)", "Crude Palm Oil (BMD)"},
            "currency": {"CME (US)": "USc/bu", "DCE (China)": "CNY/MT"},
            "fx_pair": {"CME (US)": "USD/USD", "DCE (China)": "USD/CNY"},
            "top5_producer": ["China", "United States", "Brazil", "Argentina", "European Union"],
        },
        "Soymeal": {
            "exchanges": ["CME (US)", "DCE (China)"],
            "twin": {"Soybeans (CME)", "Soybeans (DCE)", "Corn (CME)", "Corn (DCE)"},
            "currency": {"CME (US)": "USc/bu", "DCE (China)": "CNY/MT"},
            "fx_pair": {"CME (US)": "USD/USD", "DCE (China)": "USD/CNY"},
            "top5_producer": ["China", "United States", "Brazil", "Argentina", "European Union"],
        },
        "Canola/Rapeseed": {
            "exchanges": ["ICE (Canada)", "LIFFE/Euronext (EU)"],
            "twin": {"Rapeseed Oil (ZCE)", "Rapeseed (LIFFE)", "Canola (ICE)", "Soybeans (CME)", 
                     "Soybeans(DCE)"},
            "currency": {"ICE (Canada)": "CAD/T", "LIFFE/Euronext (EU)": "EUR/MT"},
            "fx_pair": {"ICE (Canada)": "USD/CAD", "LIFFE/Euronext (EU)": "USD/EUR"},
            "top5_producer": ["Canada", "European Union", "China", "India", "Australia"],
        },
        "Rapeseed Oil": {
            "exchanges": ["ZCE (China)"],
            "twin": {"Rapeseed (LIFFE/Euronext)", "Canola (ICE)", "Bean Oil (CME)", 
                     "Bean Oil (DCE)", "Palm Olein (DCE)", "Crude Palm Oil (BMD)"},
            "currency": {"ZCE (China)": "CNY/MT"},
            "fx_pair": {"ZCE (China)": "USD/CNY"},
            "top5_producer": ["European Union", "China", "Canada", "India", "Russia"],
        },
    },
    "Grains": {
        "Corn": {
            "exchanges": ["CME (US)", "DCE (China)"],
            "twin": {"Wheat (CME)", "Soybeans (CME)", "Soybeans (DCE)"},
            "currency": {"CME (US)": "USc/bu", "DCE (China)": "CNY/MT"},
            "fx_pair": {"CME (US)": "USD/USD", "DCE (China)": "USD/CNY"},
            "top5_producer": ["United States", "China", "Brazil", "Argentina", "European Union"],
        },
        "Wheat": {
            "exchanges": ["CME (US)", "MIAX (US)", "LIFFE/Euronext (EU)"],
            "twin": {"Wheat (SRW (CME)", "Wheat HRW (CME)", 
                     "Minneapolis Wheat, HRS (MIAX)", "Milling Wheat (LIFFE/Euronext)"},
            "currency": {"CME (US)": "USc/bu", "MIAX (US)": "USc/bu", "LIFFE/Euronext (EU)": "EUR/MT"},
            "fx_pair": {"CME (US)": "USD/USD", "MIAX (US)": "USD/USD", "LIFFE/Euronext (EU)": "USD/EUR"},
            "top5_producer": ["European Union", "China", "India", "Russia", "United States"],
        },
    },
    "Softs": {
        "Sugar": {
            "exchanges": ["ICE (US)", "ZCE (China)"],
            "twin": {"White Sugar (ICE)", "White Sugar (ZCE)", "Raw Sugar (ICE)"},
            "currency": {"ICE (US)": "USD/MT", "ZCE (China)": "CNY/MT", "ICE (US)": "USc/lb"},
            "fx_pair": {"ICE (US)": "USD/USD", "ZCE (China)": "USD/CNY"},
            "top5_producer": ["Brazil", "India", "European Union", "China", "Thailand"],
        },
        "Coffee": {
            "exchanges": ["ICE (Europe)", "ICE (US)"],
            "twin": {"Robusta Coffee (ICE)", "Arabica Coffee (ICE)"},
            "currency": {"ICE (Europe)": "USD/MT", "ICE (US)": "USD/MT"},
            "fx_pair": {"ICE (Europe)": "USD/USD", "ICE (US)": "USD/BRL"},
            "top5_producer": ["Brazil", "Vietnam", "Columbia", "Indonesia", "Ethiopia"],
        },
        "Cocoa": {
            "exchanges": ["ICE (US)", "ICE (Europe)"],
            "twin": {"Cocoa (ICE)", "LDN Cocoa (ICE)"},
            "currency": {"ICE (US)": "USD/MT", "ICE (Europe)": "GBP/MT"},
            "fx_pair": {"ICE (US)": "USD/USD", "ICE (Europe)": "USD/GBP"},
            "top5_producer": ["Ivory Coast", "Ghana", "Ecuador", "Indonesia", "Nigeria"],
        },
        "Cotton": {
            "exchanges": ["ICE (US)", "ZCE (China)"],
            "twin": {"Cotton No. 2 (ICE)", "Cotton (ZCE)"},
            "currency": {"ICE (US)": "USD/lb", "ZCE (China)": "CNY/MT"},
            "fx_pair": {"ICE (US)": "USD/USD", "ZCE (China)": "USD/CNY"},
            "top5_producer": ["China", "India", "Brazil", "United States", "Pakistan"],
        },
    },
    "Livestock": {
        "Lean Hogs": {
            "exchanges": ["CME (US)"],
            "twin": {"Live Cattle (CME)", "Corn (CME)", "Corn (DCE)", "Soymeal (CME)"
                     "Soymeal (DCE)"},
            "currency": {"CME (US)": "USc/lb"},
            "fx_pair": {"CME (US)": "USD/USD"},
            "top5_producer": ["China", "European Union", "United States", "Brazil", "Russia"],
        },
        "Live Cattle": {
            "exchanges": ["CME (US)"],
            "twin": {"Feeder Cattle (CME)", "Corn (CME)", "Corn (DCE)"},
            "currency": {"CME (US)": "USc/lb"},
            "fx_pair": {"CME (US)": "USD/USD"},
            "top5_producer": ["Brazil", "United States", "China", "European Union", "India"],
        },
        "Feeder Cattle": {
            "exchanges": ["CME (US)"],
            "twin": {"Live Cattle (CME)", "Corn (CME)", "Corn (DCE)", "Soymeal (CME)"
                     "Soymeal (DCE)"},
            "currency": {"CME (US)": "USc/lb"},
            "fx_pair": {"CME (US)": "USD/USD"},
            "top5_producer": ["Brazil", "United States", "China", "European Union", "India"],
        },
    },
    "Seasonal Energy": {
        "GasOil": {
            "exchanges": ["ICE (Europe)"],
            "twin": {"Heating Oil (CME)", "Brent Crude (CME/ICE)", "WTI Crude (CME/ICE)"},
            "currency": {"ICE (Europe)": "USD/MT"},
            "fx_pair": {"ICE (Europe)": "USD/USD"},
            "top5_producer": ["United States", "China", "India", "Russia", "Saudi Arabia"],
        },
        "Henry Hub Natural Gas": {
            "exchanges": ["CME (US)"],
            "twin": {"Brent Crude (CME/ICE)", "WTI Crude (CME/ICE)"},
            "currency": {"CME (US)": "USD/MMBtu"},
            "fx_pair": {"CME (US)": "USD/USD"},
            "top5_producer": ["United States", "Russia", "Iran", "China", "Canada"],
        },
        "Heating Oil": {
            "exchanges": ["CME (US)"],
            "twin": {"Gasoil (ICE)", "Brent Crude (CME/ICE)", "WTI Crude (CME/ICE)"},
            "currency": {"CME (US)": "USD/gal"},
            "fx_pair": {"CME (US)": "USD/USD"},
            "top5_producer": ["United States", "China", "India", "Russia", "Saudi Arabia"],
        },
        "RBOB Gasoline": {
            "exchanges": ["CME (US)"],
            "twin": {"Brent Crude (CME/ICE)", "WTI Crude (CME/ICE)"},
            "currency": {"CME (US)": "USD/gal"},
            "fx_pair": {"CME (US)": "USD/USD"},
            "top5_producer": ["United States", "China", "India", "Russia", "Saudi Arabia"],
        },
        "Fuel Oil 380 CST": {
            "exchanges": ["SHFE (China)"],
            "twin": {"Brent Crude (CME/ICE)", "WTI Crude (CME/ICE)"},
            "currency": {"SHFE (China)": "CNY/MT"},
            "fx_pair": {"SHFE (China)": "USD/CNY"},
            "top5_producer": ["Russia", "United States", "China", "Saudi Arabia", "India"],
        },
        "LSFO": {
            "exchanges": ["SHFE (China)"],
            "twin": {"Brent Crude (CME/ICE)", "WTI Crude (CME/ICE)"},
            "currency": {"SHFE (China)": "CNY/MT"},
            "fx_pair": {"SHFE (China)": "USD/CNY"},
            "top5_producer": ["United States", "China", "Russia", "Saudi Arabia", "South Korea"],
        },
    },
    "Non-Seasonal Energy": {
        "WTI Crude": {
            "exchanges": ["CME (US)", "ICE (Europe)"],
            "twin": "Brent Crude (CME/ICE)",
            "currency": {"CME (US)": "USD/bbl", "ICE (Europe)": "USD/bbl"},
            "fx_pair": {"CME (US)": "USD/USD", "ICE (Europe)": "USD/USD"},
            "top5_producer": ["United States", "Russia", "Saudi Arabia", "Canada", "Iraq"],
        },
        "Brent Crude": {
            "exchanges": ["CME (US)", "ICE (Europe)"],
            "twin": "WTI Crude (CME/ICE)",
            "currency": {"CME (US)": "USD/bbl", "ICE (Europe)": "USD/bbl"},
            "fx_pair": {"CME (US)": "USD/USD", "ICE (Europe)": "USD/USD"},
            "top5_producer": ["United States", "Russia", "Saudi Arabia", "Canada", "China"],
        },
        "MS Crude Oil": {
            "exchanges": ["SHFE (China)"],
            "twin": {"WTI Crude (CME/ICE)", "Brent Crude (CME/ICE)"},
            "currency": {"SHFE (China)": "CNY/bbl"},
            "fx_pair": {"SHFE (China)": "USD/CNY"},
            "top5_producer": ["United States", "Russia", "Saudi Arabia", "Canada", "Iraq"],
        },
    },
}

SUPPLY_DEMAND_ATTRS = [
    "Production", "Beginning stocks", "Domestic consumption",
    "Ending stocks", "Exports", "Imports"
]


# ─────────────────────────────────────────────────────────────
# DUMMY DATA GENERATORS
# ─────────────────────────────────────────────────────────────



def make_iv_term_structure(atm_vol=0.25, seed=3):
    rng = np.random.default_rng(seed)
    tenors = ["1M", "2M", "3M", "6M", "9M", "12M"]
    # Typical backwardation in ag vols
    vols = atm_vol - np.array([0, 0.01, 0.018, 0.03, 0.04, 0.048]) + rng.normal(0, 0.005, 6)
    return tenors, np.clip(vols, 0.10, 0.60)


def make_cot_data(seed=11):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2024-01-01", periods=20, freq="W-TUE")
    comm_long  = rng.integers(30000, 60000, 20)
    comm_short = rng.integers(20000, 50000, 20)
    spec_long  = rng.integers(40000, 90000, 20)
    spec_short = rng.integers(30000, 80000, 20)
    return pd.DataFrame({
        "Comm Long": comm_long, "Comm Short": comm_short,
        "Spec Long": spec_long, "Spec Short": spec_short,
    }, index=weeks)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Commodities\nIntelligence Dashboard")
    st.markdown("---")

    category = st.selectbox("**Asset Class**", list(COMMODITY_MAP.keys()))
    commodity = st.selectbox("**Commodity**", list(COMMODITY_MAP[category].keys()))

    cfg = COMMODITY_MAP[category][commodity]
    exchange = st.selectbox("**Exchange / Market**", cfg["exchanges"])
    twin_raw = cfg.get("twin", "")
    all_twins = sorted(twin_raw) if isinstance(twin_raw, set) else ([twin_raw] if twin_raw else [])

    st.markdown("---")
    st.caption("Data as of: May 30, 2025 (Dummy)")
    st.caption("© Commodities Intelligence | SD Guthrie")


# ─────────────────────────────────────────────────────────────
# SHARED MARKET PRICE RANGE FILTER
# ─────────────────────────────────────────────────────────────
RANGE_OPTIONS = ["1M", "3M", "6M", "YTD", "1Y", "2Y", "Custom"]

def resolve_market_range(option, custom_range):
    today = date.today()

    if option == "1M":
        return today - timedelta(days=31), today
    if option == "3M":
        return today - timedelta(days=92), today
    if option == "6M":
        return today - timedelta(days=183), today
    if option == "YTD":
        return date(today.year, 1, 1), today
    if option == "1Y":
        return today - timedelta(days=365), today
    if option == "2Y":
        return today - timedelta(days=730), today

    if isinstance(custom_range, tuple) and len(custom_range) == 2:
        return custom_range[0], custom_range[1]

    return today - timedelta(days=365), today


range_key = f"atlas_range_{commodity}_{exchange}"
custom_key = f"atlas_custom_dates_{commodity}_{exchange}"

# Default must match your render_atlas_market_section default.
if range_key not in st.session_state:
    st.session_state[range_key] = "6M"

if custom_key not in st.session_state:
    st.session_state[custom_key] = (
        date.today() - timedelta(days=365),
        date.today(),
    )

market_range_option = st.session_state[range_key]
market_custom_range = st.session_state[custom_key]

market_start_date, market_end_date = resolve_market_range(
    market_range_option,
    market_custom_range,
)

# Shorthand helpers
currency   = cfg["currency"][exchange]
regions        = cfg.get("growing_regions", [])
top5_producers = cfg.get("top5_producer", [])
currency = cfg["currency"].get(exchange, "")
fx_pair  = cfg.get("fx_pair", {}).get(exchange, "USD/USD")

def _parse_twin_meta(twin_label: str, commodity_map: dict) -> tuple[str, str]:
    """
    Given a twin_label like "Bean Oil (CME)" or "Soybeans (DCE)",
    look up its currency and fx_pair from COMMODITY_MAP.
    Returns (currency_string, fx_pair_string).
    Falls back to ("USD/MT", "USD/USD") if not found.
    """
    for cat in commodity_map.values():
        for comm_name, comm_cfg in cat.items():
            if comm_name.lower() in twin_label.lower():
                # find matching exchange key
                for exch_key, ccy in comm_cfg.get("currency", {}).items():
                    # match exchange abbreviation that appears in twin_label
                    abbr = exch_key.split("(")[-1].rstrip(")").strip()  # e.g. "CME"
                    if abbr.upper() in twin_label.upper():
                        fx = comm_cfg.get("fx_pair", {}).get(exch_key, "USD/USD")
                        return ccy, fx
                # fallback: take first exchange entry
                first_exch = list(comm_cfg.get("currency", {}).keys())[0]
                return (
                    comm_cfg["currency"][first_exch],
                    comm_cfg.get("fx_pair", {}).get(first_exch, "USD/USD"),
                )
    return "USD/MT", "USD/USD"
 

policy_df = load_policy() 
# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"# {commodity}")
st.markdown(
    f'<span style="color:#8b949e;font-family:\'IBM Plex Mono\',monospace;font-size:13px;">'
    f'{category} · {exchange} · {currency}</span>',
    unsafe_allow_html=True
)
# st.markdown("---")

# ─────────────────────────────────────────────────────────────
# ROW 1 — TOP METRICS
# ─────────────────────────────────────────────────────────────
_atlas_metric = get_atlas_range_price_metric(
    commodity=commodity,
    exchange_label=exchange,
    start_date=market_start_date,
    end_date=market_end_date,
)

latest_px = _atlas_metric.get("latest_px", np.nan)
chg_pct = _atlas_metric.get("chg_pct", np.nan)
# Live FX rate from Yahoo Finance via fx.py (cached, 1-hour TTL)
# lookback_days=10 guarantees >=5 business days even across weekends/holidays
try:
    _fx_history = fetch_fx_history((fx_pair,), lookback_days=10)
    _fx_snap    = get_fx_snapshot(_fx_history.get(fx_pair, pd.Series(dtype=float)))
    fx_val      = _fx_snap["rate"] if not np.isnan(_fx_snap["rate"]) else 1.0
    fx_chg      = _fx_snap["chg_1d"] if not np.isnan(_fx_snap["chg_1d"]) else 0.0
except Exception:
    fx_val = 1.0
    fx_chg = 0.0
 


# ------- ENSO DATA ---------------------------------------------------------------
oni_df = load_oni()
enso_status, enso_oni, enso_class = get_enso_status(oni_df)

def metric_card(container, label, value, delta=None, delta_pos=True):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-pos" if delta_pos else "metric-delta-neg"
        arrow = "▲" if delta_pos else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    container.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# ── ROW 2 — KEY METRICS | 02 FX IMPACT | 03 SUPPLY & DEMAND ────────────────
_metric_cols = st.columns([1, 1, 1, 1, 2.2])

metric_card(
    _metric_cols[0],
    f"{commodity} ({exchange})",
    "N/A" if pd.isna(latest_px) else f"{latest_px:,.1f}",
    None if pd.isna(chg_pct) else f"{chg_pct:+.2f}% over {market_range_option}",
    delta_pos=False if pd.isna(chg_pct) else chg_pct >= 0,
)

metric_card(
    _metric_cols[1],
    fx_pair,
    f"{fx_val:.4f}",
    f"{fx_chg:+.2f}%",
    delta_pos=fx_chg >= 0,
)

with _metric_cols[2]:
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">ENSO Status</div>
        <div class="metric-value" style="font-size:16px;margin:4px 0">
            <span class="enso-badge {enso_class}">{enso_status}</span>
        </div>
        <div style="color:#8b949e;font-family:'IBM Plex Mono',monospace;font-size:11px;">
            ONI: {enso_oni}
        </div>
    </div>''', unsafe_allow_html=True)

metric_card(
    _metric_cols[3],
    "30-Day Realized Vol",
    "18.4%",
    "ATM IV: 22.1%",
    delta_pos=True,
)

st.markdown("---")

_col_fx, _col_sd, _col_policy = st.columns([1.55, 1.55, 1.45])

with _col_fx:
    render_atlas_market_section(
        commodity=commodity,
        exchange=exchange,
        currency=currency,
        fx_pair=fx_pair,
        twin_labels=all_twins,
        lookback_days=365,
    )

with _col_sd:
    st.markdown('<div class="section-header">03 · Supply & Demand Analysis</div>', unsafe_allow_html=True)

    # ── Resolve USDA commodity code for selected commodity ────────
    _psd_supported = commodity in COMMODITY_CODES

    tab_sd1, tab_sd2, tab_sd3, tab_sd4 = st.tabs(["📊 S&D Balance", "🏆 Top Countries", "📦 Export Breakdown", "Actual / Est. / Proj."])

    # ── Tab 1: World S&D Balance ──────────────────────────────────
    with tab_sd1:
        if not _psd_supported:
            st.info(f"USDA FAS PSD data not available for **{commodity}**.")
        else:
            ctrl_a, ctrl_b = st.columns([3, 1])
            with ctrl_a:
                selected_attrs = st.multiselect(
                    "Attributes",
                    list(ATTRIBUTE_LABELS.values()),
                    default=["Production", "Exports", "Domestic Consumption",
                             "Ending Stocks", "Imports"],
                    key="sd_attrs",
                )
            with ctrl_b:
                chart_type = st.radio(
                    "Chart Type", ["Line", "Bar", "Stacked Area"],
                    key="sd_chart_type",
                )

            if not selected_attrs:
                st.info("Select at least one attribute.")
            else:
                with st.spinner("Loading USDA FAS world data…"):
                    _world_sd = load_world_sd()
                if _world_sd.empty:
                    st.error("Failed to load USDA FAS world data.")
                else:
                    _fig_sd = make_sd_balance_chart(
                        _world_sd, commodity, selected_attrs, chart_type
                    )
                    if _fig_sd is None:
                        st.info(f"No world S&D data for {commodity}.")
                    else:
                        st.plotly_chart(_fig_sd, width='stretch')
    # ── Tab 2: Top Countries by Attribute ─────────────────────────
    with tab_sd2:
        if not _psd_supported:
            st.info(f"USDA FAS PSD data not available for **{commodity}**.")
        else:
            ctrl_c, ctrl_d = st.columns([3, 1])
            with ctrl_c:
                tc_attr = st.selectbox(
                    "Attribute",
                    list(ATTRIBUTE_LABELS.values()),
                    index=list(ATTRIBUTE_LABELS.values()).index("Exports"),
                    key="tc_attr",
                )
            with ctrl_d:
                tc_topn = st.slider("Top N", 5, 20, 10, key="tc_topn")

            with st.spinner("Loading USDA FAS country data…"):
                _country_sd = load_country_sd()
            if _country_sd.empty:
                st.error("Failed to load USDA FAS country data.")
            else:
                _fig_tc = make_top_countries_chart(
                    _country_sd, commodity, tc_attr, tc_topn
                )
                if _fig_tc is None:
                    st.info(f"No country data for {commodity} / {tc_attr}.")
                else:
                    st.plotly_chart(_fig_tc, width='stretch')

    # ── Tab 3: Export Breakdown + Global Consumption ──────────────
    with tab_sd3:
        if not _psd_supported:
            st.info(f"USDA FAS PSD data not available for **{commodity}**.")
        else:
            eb_topn = st.slider(
                "Top exporter groups", 3, 8, 4,
                key="eb_topn",
                help="Number of individually named exporters; remainder grouped as Rest of World",
            )

            with st.spinner("Loading USDA FAS country data…"):
                _country_sd = load_country_sd()

            if _country_sd.empty:
                st.error("Failed to load USDA FAS country data.")
            else:
                _fig_eb = make_export_stacked_chart(
                    _country_sd, commodity, top_n=eb_topn
                )
                if _fig_eb is None:
                    st.info(f"No export data for {commodity}.")
                else:
                    st.plotly_chart(_fig_eb, width='stretch')

    # ── Tab 4: Actual / Estimated / Projected (USDA PSD) ─────────
    with tab_sd4:
        render_aep_tab(commodity)

with _col_policy:
    st.markdown('<div class="section-header">05 · Policy Tracker</div>', unsafe_allow_html=True)
    render_policy_tracker(policy_df, commodity)

st.markdown("---")

_today  = datetime.today()
_mstart = _today.replace(day=1).strftime("%Y-%m-%d")
_mend   = _today.strftime("%Y-%m-%d")


def render_multi_param_mini_charts(weather_result, selected_group, commodity):
    """
    Renders small sparkline cards for each parameter in the selected group.
    Call this right after the main weather chart inside col_rain.
    """
    grp_data = next(
        (g for g in weather_result["groups"] if g["label"] == selected_group), None
    )
    if not grp_data:
        return
 
    params = [p for p in grp_data["data"] if not grp_data["data"][p].empty]
    if len(params) <= 1:
        return   # nothing extra to show
 
    zone_col = grp_data["zone_col"]
    cols = st.columns(len(params))
    colors_p = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]
 
    for col, param, color in zip(cols, params, colors_p):
        df = grp_data["data"][param]
        # Average across all zones
        agg = df.groupby("date")[
            [c for c in df.columns if c not in (zone_col, "date", "station_count")][0]
        ].mean().reset_index()
 
        fig = go.Figure(go.Scatter(
            x=agg["date"], y=agg.iloc[:, 1],
            mode="lines", line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=color.replace("#", "rgba(").rstrip(")") + ",0.08)",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            height=120, margin=dict(l=0, r=0, t=18, b=0),
            showlegend=False,
            title=dict(text=PARAM_LABEL.get(param, param), font=dict(size=10), x=0),
            font=dict(family="IBM Plex Mono", size=9),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(gridcolor="#21262d", tickfont=dict(size=8))
        col.plotly_chart(fig, width="stretch")

# ── ROW 3 — 03 WEATHER | 08 CROP CALENDAR | 04 ENSO ONI INDEX ───────────────
_col_weather, _col_right, _col_risk = st.columns([1.55, 1.55, 1.45])


with _col_weather:
    render_weather_section(
        commodity,
        dark=True,
        cache_path="data/weather_cache/weather_daily.feather",
        force_refresh=False,
        refresh_recent_days=2,
    )

with _col_right:
    st.markdown('<div class="section-header">08 · Crop Calendar</div>', unsafe_allow_html=True)

    # Each commodity maps to a list of (label, image_path) tuples
    CROP_CALENDAR_IMAGES = {
        "Crude Palm Oil": [
            ("Malaysia",   "crop_calendars/my_palmoil.png"),
            ("Indonesia",  "crop_calendars/indo_palmoil.png"),
        ],
        "Palm Olein": [
            ("Malaysia",   "crop_calendars/my_palmoil.png"),
            ("Indonesia",  "crop_calendars/indo_palmoil.png"),
        ],
        "Soybeans": [
            ("Brazil",         "crop_calendars/brazil_soybean.png"),
            ("United States",  "crop_calendars/us_soybean.png"),
            ("China",          "crop_calendars/china_soybean.png"),
        ],
        "Bean Oil": [
            ("Brazil",         "crop_calendars/brazil_soybean.png"),
            ("United States",  "crop_calendars/us_soybean.png"),
            ("China",          "crop_calendars/china_soybean.png"),
        ],
        "Soymeal": [
           ("Brazil",         "crop_calendars/brazil_soybean.png"),
            ("United States",  "crop_calendars/us_soybean.png"),
            ("China",          "crop_calendars/china_soybean.png"),
        ],
        "Canola/Rapeseed": [
            ("Canada",          "crop_calendars/canada_rapeseedcanola.png"),
            ("China",           "crop_calendars/china_rapeseed.png"),
        ],
        "Corn": [
            ("United States",  "crop_calendars/us_corn.png"),
            ("China",          "crop_calendars/china_corn.png"),
        ],
        "Wheat": [
            ("United States",  "crop_calendars/us_wheat.png"),
            ("European Union", "crop_calendars/eu_crop_spring.png"),
            ("European Union", "crop_calendars/eu_winter.png"),
            ("Canada",         "crop_calendars/canada_wheat.png"),
        ],
        "Cotton": [
            ("United States",  "crop_calendars/us_cotton.png"),
            ("China",          "crop_calendars/china_cotton.png"),
        ],
    }

    calendar_entries = CROP_CALENDAR_IMAGES.get(commodity, [])

    # Filter to only images that actually exist on disk
    available = [(label, p) for label, p in calendar_entries if Path(p).exists()]
    missing   = [(label, p) for label, p in calendar_entries if not Path(p).exists()]

    if available:
        # Show a selectbox if more than 2 images — avoids cramped layout
        if len(available) > 2:
            selected_labels = st.multiselect(
                "Select regions to display",
                options=[label for label, _ in available],
                default=[available[0][0], available[1][0]],
                key="crop_cal_select",
            )
            to_display = [(label, p) for label, p in available if label in selected_labels]
        else:
            to_display = available

        if to_display:
            cols = st.columns(len(to_display))
            for col, (label, img_path) in zip(cols, to_display):
                with col:
                    st.markdown(
                        f'<div style="color:#8b949e;font-family:IBM Plex Mono,monospace;'
                        f'font-size:11px;text-transform:uppercase;letter-spacing:0.08em;'
                        f'margin-bottom:6px;">{label}</div>',
                        unsafe_allow_html=True
                    )
                    st.image(Image.open(img_path), width="stretch")
        else:
            st.info("Select at least one region to display.")

    else:
        st.info(
            f"No crop calendar images found for **{commodity}**. "
            f"Add images to the `crop_calendars/` folder."
        )

    # Show which images are still missing
    if missing:
        with st.expander(f"⚠️ {len(missing)} image(s) not yet added"):
            for label, p in missing:
                st.caption(f"• {label} → `{p}`")
    # ─────────────────────────────────────────────────────────────

with _col_right:
    st.markdown('<div class="section-header">04 · ENSO ONI Index</div>',
                unsafe_allow_html=True)

    start_year = st.slider(
        "Start year", 
        min_value=1950, 
        max_value=2020, 
        value=1990, 
        step=5,
        key="oni_start_year",
    )

    fig_oni = make_oni_chart(oni_df, start_year=start_year)
    st.plotly_chart(fig_oni, width="stretch")
    st.caption("Source: NOAA CPC · 3-month running mean SST Niño 3.4")

with _col_risk:
    st.markdown('<div class="section-header">09 · Risk Flows & Implied Volatility</div>',
                unsafe_allow_html=True)

    col_iv, col_cot = st.columns(2)

    with col_iv:
        st.markdown("##### IV Term Structure")
        tenors, iv_vals = make_iv_term_structure(atm_vol=0.22 + np.random.uniform(-0.05, 0.05))
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(
            x=tenors, y=iv_vals * 100,
            mode="lines+markers",
            name="ATM IV",
            line=dict(color="#58a6ff", width=2.5),
            marker=dict(size=8, symbol="circle"),
        ))
        fig_iv.add_trace(go.Scatter(
            x=tenors, y=(iv_vals + 0.04) * 100,
            mode="lines", name="25Δ Put",
            line=dict(color="#f78166", width=1.5, dash="dot"),
        ))
        fig_iv.add_trace(go.Scatter(
            x=tenors, y=(iv_vals - 0.02) * 100,
            mode="lines", name="25Δ Call",
            line=dict(color="#3fb950", width=1.5, dash="dot"),
        ))
        fig_iv.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.08, x=0),
            font=dict(family="IBM Plex Mono", size=11),
            yaxis_title="Implied Vol (%)",
            xaxis_title="Tenor",
        )
        fig_iv.update_yaxes(gridcolor="#21262d")
        st.plotly_chart(fig_iv, width="stretch")

    with col_cot:
        st.markdown("##### COT — Managed Money Net Positioning")
        cot_df = make_cot_data()
        net_spec = cot_df["Spec Long"] - cot_df["Spec Short"]
        net_comm = cot_df["Comm Long"] - cot_df["Comm Short"]

        fig_cot = go.Figure()
        fig_cot.add_trace(go.Bar(
            x=cot_df.index, y=net_spec,
            name="Managed Money Net",
            marker_color=["#3fb950" if v >= 0 else "#f85149" for v in net_spec],
        ))
        fig_cot.add_trace(go.Scatter(
            x=cot_df.index, y=net_comm,
            name="Commercials Net",
            mode="lines",
            line=dict(color="#ffa657", width=1.5, dash="dash"),
        ))
        fig_cot.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.08, x=0),
            font=dict(family="IBM Plex Mono", size=11),
            yaxis_title="Contracts (net)",
            barmode="relative",
        )
        fig_cot.update_yaxes(gridcolor="#21262d")
        st.plotly_chart(fig_cot, width="stretch")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="color:#8b949e;font-family:\'IBM Plex Mono\',monospace;font-size:10px;text-align:center;">'
    'INTERNAL USE ONLY · SD Guthrie International · Commodities Intelligence Dashboard · '
    'All data shown is dummy/illustrative. Replace with live Bloomberg / Refinitiv / MPOB / USDA feeds.'
    '</p>',
    unsafe_allow_html=True
)