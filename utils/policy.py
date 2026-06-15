"""
utils/policy.py
===============
Fetches trade policy interventions from the Global Trade Alert (GTA) API,
cleans them, maps to dashboard commodities, and renders the Policy Tracker
section in dashboard.py.

Replaces the hardcoded POLICY_DATA dict and render_policy_table() function.

Dashboard wiring
----------------
    from utils.policy import load_policy, render_policy_tracker

    policy_df = load_policy()          # cached 6h, auto-refreshes

    # In Section 10:
    render_policy_tracker(policy_df, commodity)
"""

import json
import ast
import time
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
CACHE_DIR    = Path(__file__).parent.parent / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FEATHER_PATH = CACHE_DIR / "policyshifts.feather"

# ─────────────────────────────────────────────────────────────
# GTA API CONFIG
# ─────────────────────────────────────────────────────────────
GTA_API_KEY = os.getenv("GTA_API_KEY")
GTA_URL     = "https://api.globaltradealert.org/api/v2/gta/data/"
GTA_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"APIKey {GTA_API_KEY}",
}

# ─────────────────────────────────────────────────────────────
# PRODUCT IDs covered by your notebook (full list)
# ─────────────────────────────────────────────────────────────
ALL_PRODUCT_IDS = [
    # Maize/Corn (missing from original)
    100510,100590,
    # Wheat (missing from original)
    100110,100190,100191,100199,
    # Soybean meal/oilcake (missing from original)
    230400,
    271600,271111,271112,271113,271114,271119,271121,271129,271012,271019,
    271020,271091,271099,270900,270710,270720,270730,270740,270750,270791,
    270799,90111,90112,90121,90122,90190,90210,90220,90230,90240,90300,
    90411,90412,90421,90422,90510,90520,90611,90619,90620,90710,90720,
    90811,90812,90821,90822,90831,90832,90921,90922,90931,90932,90961,
    90962,91011,91012,91020,91030,91091,91099,80211,80212,80221,80222,
    80231,80232,80241,80242,80251,80252,80261,80262,80270,80280,80291,
    80292,80299,10121,10129,10130,10190,999999,520100,520210,520291,520299,
    520300,520411,520419,520420,520511,520512,520513,520514,520515,520521,
    520522,520523,520524,520526,520527,520528,520531,520532,520533,520534,
    520535,520541,520542,520543,520544,520546,520547,520548,520611,520612,
    520613,520614,520615,520621,520622,520623,520624,520625,520631,520632,
    520633,520634,520635,520641,520642,520643,520644,520645,520710,520790,
    520811,520812,520813,520819,520821,520822,520823,520829,520831,520832,
    520833,520839,520841,520842,520843,520849,520851,520852,520859,520911,
    520912,520919,520921,520922,520929,520931,520932,520939,520941,520942,
    520943,520949,520951,520952,520959,521011,521019,521021,521029,521031,
    521032,521039,521041,521049,521051,521059,521111,521112,521119,521120,
    521131,521132,521139,521141,521142,521143,521149,521151,521152,521159,
    521211,521212,521213,521214,521215,521221,521222,521223,521224,521225,
    180100,180200,180310,180320,180400,180500,180610,180620,180631,180632,
    180690,150110,150120,150190,150210,150290,150300,150410,150420,150430,
    150500,150600,150710,150790,150810,150890,150920,150930,150940,150990,
    151010,151090,151110,151190,151211,151219,151221,151229,151311,151319,
    151321,151329,151411,151419,151491,151499,151511,151519,151521,151529,
    151530,151550,151560,151590,151610,151620,151630,151710,151790,151800,
    152000,152110,152190,152200,170112,170113,170114,170191,170199,170211,
    170219,170220,170230,170240,170250,170260,170290,170310,170390,170410,
    170490,120110,120190,120230,120241,120242,120300,120400,120510,120590,
    120600,120710,120721,120729,120730,120740,120750,120760,120770,120791,
    120799,120810,120890,120910,120921,120922,120923,120924,120925,120929,
    120930,120991,120999,121010,121020,121120,121130,121140,121150,121160,
    121190,121221,121229,121291,121292,121293,121294,121299,121300,121410,
    121490,110100,110220,110290,110311,110313,110319,110320,110412,110419,
    110422,110423,110429,110430,110510,110520,110610,110620,110630,110710,
    110720,110811,110812,110813,110814,110819,110820,110900,20110,20120,
    20130,20210,20220,20230,20311,20312,20319,20321,20322,20329,20410,
    20421,20422,20423,20430,20441,20442,20443,20450,20500,20610,20621,
    20622,20629,20630,20641,20649,20680,20690,20711,20712,20713,20714,
    20724,20725,20726,20727,20741,20742,20743,20744,20745,20751,20752,
    20753,20754,20755,20760,20810,20830,20840,20850,20860,20890,20910,
    20990,21011,21012,21019,21020,21091,21092,21093,21099,10221,10229,
    10231,10239,10290,10310,10391,10392,10410,10420,10511,10512,10513,
    10514,10515,10594,10599,10611,10612,10613,10614,10619,10620,10631,
    10632,10633,10639,10641,10649,10690,
]

# ─────────────────────────────────────────────────────────────
# COMMODITY → HS PRODUCT ID MAPPING
# Maps each COMMODITY_MAP key to the relevant HS product_ids
# so filtering is commodity-specific
# ─────────────────────────────────────────────────────────────
COMMODITY_PRODUCT_MAP = {
    "Crude Palm Oil": [
        151110, 151190,                          # Palm oil & fractions
        151311, 151319, 151321, 151329,          # Coconut/palm kernel oils
    ],
    "Palm Olein":     [151110, 151190, 151311, 151319, 151321, 151329],
    "Soybeans":       [230400, 120110, 120190, 151211, 151219, 151221, 151229],          # Soybeans
    "Bean Oil":       [230400, 120110, 120190, 151211, 151219, 151221, 151229],  # Soybean oil
    "Soymeal":        [230400, 120110, 120190, 151211, 151219, 151221, 151229],                  # Soybean meal (oilcake)
    "Canola/Rapeseed":[120510, 120590, 151411, 151419, 151491, 151499],          # Rape/colza seeds
    "Rapeseed Oil":   [120510, 120590, 151411, 151419, 151491, 151499],  # Rape/mustard oil
    "Corn":           [100510, 100590,           # Maize
                       110220, 110812],
    "Wheat":          [100110, 100190, 100191, 100199,   # Wheat
                       110100, 110311, 110319],
    "Sugar":          [170112, 170113, 170114, 170191, 170199,
                       170211, 170219, 170220, 170230, 170240,
                       170250, 170260, 170290, 170310, 170390,
                       170410, 170490],
    "Coffee":         [90111, 90112, 90121, 90122, 90190,
                       90210, 90220, 90230, 90240, 90300],
    "Cocoa":          [180100, 180200, 180310, 180320,
                       180400, 180500, 180610, 180620,
                       180631, 180632, 180690],
    "Cotton":         [520100, 520210, 520291, 520299, 520300],
    "Lean Hogs":      [10121, 10129, 10130, 10190,
                       20311, 20312, 20319, 20321, 20322, 20329],
    "Live Cattle":    [10221, 10229, 10231, 10239, 10290,
                       20110, 20120, 20130, 20210, 20220, 20230],
    "Feeder Cattle":  [10221, 10229, 10231, 10239, 10290],
    "GasOil":         [271012, 271019, 271020],
    "Heating Oil":    [271012, 271019, 271020],
    "RBOB Gasoline":  [271012, 271019],
    "Henry Hub Natural Gas": [271111, 271112, 271113, 271114,
                              271119, 271121, 271129],
    "Fuel Oil 380 CST": [271091, 271099, 271019],
    "LSFO":           [271091, 271099, 271019],
    "WTI Crude":      [270900],
    "Brent Crude":    [270900],
    "MS Crude Oil":   [270900],
}

# GTA colour → dashboard Impact label
GTA_EVAL_MAP = {
    "Red":   "Bearish",    # harmful / liberalisation-reducing
    "Amber": "Neutral",    # uncertain
    "Green": "Bullish",    # trade-liberalising
}

IMPACT_COLORS = {
    "Bullish": "#3fb950",
    "Bearish": "#f85149",
    "Neutral": "#ffa657",
}

# ─────────────────────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────────────────────

def extract_names(value) -> str | None:
    """Flatten GTA nested list/dict columns to comma-separated names."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except Exception:
            try:
                value = ast.literal_eval(value)
            except Exception:
                return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        names = [item.get("name") for item in value
                 if isinstance(item, dict) and item.get("name")]
        return ", ".join(names) if names else None
    if isinstance(value, dict):
        return value.get("name")
    return value


def _make_feather_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns with nested lists/dicts to JSON strings for Feather."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    return df


# ─────────────────────────────────────────────────────────────
# 2. FETCH  (mirrors your notebook fetch_all_gta_data)
# ─────────────────────────────────────────────────────────────

def _fetch_gta(
    latest_action_start: str = "2023-01-01",
    limit: int = 1000,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    today = datetime.today().strftime("%Y-%m-%d")
    all_records = []
    offset = 0

    while True:
        payload = {
            "limit": limit,
            "offset": offset,
            "sorting": "-latest_action_date",
            "show_full_names": True,
            "show_keys": [],
            "text_format": "markdown",
            "request_data": {
                "affected_products": ALL_PRODUCT_IDS,
                "keep_affected": True,
                "keep_implementer": True,
                "keep_intervention_types": True,
                "keep_affected_sectors": True,
                "keep_affected_products": True,
                "keep_state_act_id": True,
                "keep_intervention_id": True,
                "keep_thread": True,
                "keep_rationale": True,
                "keep_in_force_on_date": True,
                "latest_action_period": [latest_action_start, today],
                "in_force_on_date": today,
            },
        }
        resp = requests.post(GTA_URL, headers=GTA_HEADERS, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"GTA API error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        records = (data if isinstance(data, list)
                   else next((data[k] for k in ["data","results","items","records"]
                              if k in data and isinstance(data[k], list)), []))
        if not records:
            break
        all_records.extend(records)
        if len(records) < limit:
            break
        offset += limit
        time.sleep(sleep_seconds)

    return pd.json_normalize(all_records) if all_records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 3. CLEAN  (mirrors your notebook cell 5)
# ─────────────────────────────────────────────────────────────

def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    cols = [
        "state_act_title", "state_act_url", "gta_evaluation",
        "implementing_jurisdictions", "affected_jurisdictions",
        "affected_products", "intervention_type",
        "date_announced", "last_updated",
    ]
    available = [c for c in cols if c in raw.columns]
    df = raw[available].copy()

    for col in ["implementing_jurisdictions", "affected_jurisdictions", "affected_products"]:
        if col in df.columns:
            df[col] = df[col].apply(extract_names)

    for col in ["date_announced", "last_updated"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.rename(columns={
        "state_act_title":           "Policy Title",
        "state_act_url":             "Policy URL",
        "gta_evaluation":            "GTA Evaluation",
        "implementing_jurisdictions":"Implementing Country",
        "affected_jurisdictions":    "Affected Countries",
        "affected_products":         "Affected Products",
        "intervention_type":         "Intervention Type",
        "date_announced":            "Date Announced",
        "last_updated":              "Last Updated",
    })

    # Map GTA evaluation to Impact
    df["Impact"] = df["GTA Evaluation"].map(GTA_EVAL_MAP).fillna("Neutral")

    # Format date
    df["Date"] = df["Date Announced"].dt.strftime("%b %Y").fillna("—")

    return df


# ─────────────────────────────────────────────────────────────
# 4. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

FEATHER_TTL_HOURS = 6  # refresh feather if older than this


def _feather_is_stale() -> bool:
    """True if feather file doesn't exist or is older than FEATHER_TTL_HOURS."""
    if not FEATHER_PATH.exists():
        return True
    age_seconds = time.time() - FEATHER_PATH.stat().st_mtime
    return age_seconds > FEATHER_TTL_HOURS * 3600


def _refresh_feather() -> None:
    """Fetch from GTA API and overwrite the feather file. No-op on failure."""
    try:
        raw = _fetch_gta()
        if not raw.empty:
            _make_feather_safe(raw).to_feather(FEATHER_PATH)
    except Exception as e:
        st.warning(f"⚠️ GTA API refresh failed — using existing cache: {e}")


@st.cache_data(show_spinner="Loading policy data…")
def _read_feather(mtime: float) -> pd.DataFrame:
    """
    Reads and cleans the feather file. `mtime` is passed purely to bust
    Streamlit's cache when the file changes on disk — it is not used directly.
    """
    raw = pd.read_feather(FEATHER_PATH)
    return _clean(raw)


def load_policy() -> pd.DataFrame:
    """
    Loads GTA policy data with persistent disk-based caching.

    Strategy:
      - On every Streamlit run, always reads from the feather file (instant).
      - If the feather file is missing or older than FEATHER_TTL_HOURS, it
        fetches fresh data from the GTA API and writes a new feather file
        before reading it. This only happens at most once per TTL window,
        even across server restarts.

    Returns cleaned DataFrame with columns:
        Date, Policy Title, Implementing Country, Affected Countries,
        Affected Products, Intervention Type, Impact, GTA Evaluation,
        Policy URL, Date Announced, Last Updated
    """
    if _feather_is_stale():
        with st.spinner("Refreshing policy data from GTA API…"):
            _refresh_feather()

    if FEATHER_PATH.exists():
        try:
            mtime = FEATHER_PATH.stat().st_mtime
            return _read_feather(mtime)
        except Exception as e:
            st.warning(f"⚠️ Could not read policy cache: {e}")
            return pd.DataFrame()

    st.warning("⚠️ No policy data available — GTA API fetch failed and no local cache found.")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 5. COMMODITY FILTER
# ─────────────────────────────────────────────────────────────

def filter_policy(df: pd.DataFrame, commodity_map_key: str) -> pd.DataFrame:
    """
    Filters the full policy DataFrame to rows relevant to a specific commodity
    by matching HS product IDs in the Affected Products column.

    Falls back to keyword matching on Policy Title if no product IDs match.
    """
    if df.empty:
        return df

    product_ids = COMMODITY_PRODUCT_MAP.get(commodity_map_key, [])

    # ── Method 1: match product_id strings in Affected Products column ──
    if product_ids:
        id_pattern = "|".join(r"\b" + str(p) + r"\b" for p in product_ids)
        mask = df["Affected Products"].fillna("").str.contains(id_pattern, regex=True)
        filtered = df[mask].copy()
        if not filtered.empty:
            return filtered.sort_values("Date Announced", ascending=False)

    # ── Method 2: keyword fallback ────────────────────────────────────
    KEYWORD_MAP = {
        "Crude Palm Oil":    ["palm oil", "palm", "cpo", "rbd"],
        "Palm Olein":        ["palm olein", "palm oil", "palm"],
        "Soybeans":          ["soybean", "soya bean", "soy bean"],
        "Bean Oil":          ["soybean oil", "soya oil"],
        "Soymeal":           ["soybean meal", "soy meal", "soya meal"],
        "Canola/Rapeseed":   ["canola", "rapeseed", "colza"],
        "Rapeseed Oil":      ["rapeseed oil", "canola oil"],
        "Corn":              ["corn", "maize"],
        "Wheat":             ["wheat"],
        "Sugar":             ["sugar", "sucrose"],
        "Coffee":            ["coffee"],
        "Cocoa":             ["cocoa", "cacao", "chocolate"],
        "Cotton":            ["cotton"],
        "Lean Hogs":         ["pork", "swine", "hog", "pig"],
        "Live Cattle":       ["beef", "cattle", "bovine"],
        "Feeder Cattle":     ["cattle", "bovine", "feeder"],
        "GasOil":            ["gas oil", "gasoil", "diesel", "fuel oil"],
        "Heating Oil":       ["heating oil", "fuel oil", "diesel"],
        "RBOB Gasoline":     ["gasoline", "petrol", "rbob"],
        "Henry Hub Natural Gas": ["natural gas", "lng", "lpg"],
        "Fuel Oil 380 CST":  ["fuel oil", "bunker"],
        "LSFO":              ["fuel oil", "lsfo", "low sulphur"],
        "WTI Crude":         ["crude oil", "petroleum", "crude"],
        "Brent Crude":       ["crude oil", "brent", "petroleum"],
        "MS Crude Oil":      ["crude oil", "petroleum"],
    }
    keywords = KEYWORD_MAP.get(commodity_map_key, [commodity_map_key.lower()])
    pattern  = "|".join(re.escape(k) for k in keywords)
    mask     = df["Policy Title"].fillna("").str.lower().str.contains(pattern)
    filtered = df[mask].copy()
    return filtered.sort_values("Date Announced", ascending=False)


# ─────────────────────────────────────────────────────────────
# 6. RENDER
# ─────────────────────────────────────────────────────────────

def render_policy_tracker(policy_df: pd.DataFrame, commodity_map_key: str) -> None:
    """
    Drop-in replacement for the Policy Tracker section in dashboard.py.

    Usage:
        from utils.policy import load_policy, render_policy_tracker
        policy_df = load_policy()
        render_policy_tracker(policy_df, commodity)
    """
    filtered = filter_policy(policy_df, commodity_map_key)

    if filtered.empty:
        st.info(
            f"No policy interventions found for **{commodity_map_key}** "
            f"in the current GTA dataset (2026–present)."
        )
        return

    # ── Controls ──────────────────────────────────────────────
    ctrl1, ctrl2, date_col = st.columns([2, 2, 1.5])
    with ctrl1:
        impact_filter = st.multiselect(
            "Impact",
            ["Bullish", "Neutral", "Bearish"],
            default=["Bullish", "Neutral", "Bearish"],
            key="policy_impact_filter",
        )
    with ctrl2:
        max_rows = st.slider("Max rows", 3, 50, 20, key="policy_max_rows")
    with ctrl2:
        country_options = sorted(
            set(", ".join(filtered["Implementing Country"].dropna()).split(", "))
        )
        selected_countries = st.multiselect(
            "Implementing Country",
            country_options,
            default=[],
            key="policy_country_filter",
            placeholder="All countries",
        )
    
    with date_col:
        available_years = sorted(
            filtered["Date Announced"].dropna().dt.year.unique().astype(int),
            reverse=True,
        )
        selected_years = st.multiselect(
            "Year",
            options=available_years,
            default=available_years[:2] if len(available_years) >= 2 else available_years,
            key="policy_year_filter",
            placeholder="All years",
        )
    with date_col:
        MONTH_NAMES = {
            1:"January", 2:"February", 3:"March", 4:"April",
            5:"May", 6:"June", 7:"July", 8:"August",
            9:"September", 10:"October", 11:"November", 12:"December",
        }
        available_months = sorted(
            filtered["Date Announced"].dropna().dt.month.unique().astype(int)
        )
        selected_months = st.multiselect(
            "Month",
            options=available_months,
            format_func=lambda m: MONTH_NAMES.get(m, str(m)),
            default=[],
            key="policy_month_filter",
            placeholder="All months",
        )

 
    # ── Apply all filters ─────────────────────────────────────
    display = filtered[filtered["Impact"].isin(impact_filter)].copy()
 
    if selected_years:
        display = display[display["Date Announced"].dt.year.isin(selected_years)]
 
    if selected_months:
        display = display[display["Date Announced"].dt.month.isin(selected_months)]
 
    if selected_countries:
        mask = display["Implementing Country"].fillna("").apply(
            lambda x: any(c in x for c in selected_countries)
        )
        display = display[mask]
 
    display = display.head(max_rows)
 
    if display.empty:
        st.info("No rows match the selected filters.")
        return
    
    
 
    # ── Build HTML table ──────────────────────────────────────
    TABLE_COLS = [
        ("Date",                 "left",  "60px"),
        ("Implementing Country", "left",  "10px"),
        ("Policy Title",         "left",  "auto"),
        ("Intervention Type",    "left",  "10px"),
        ("Affected Countries",   "left",  "250px"),
        ("Impact",               "center","80px"),
    ]
 
    header = "".join(
        f'<th style="background:#f6f8fa;color:#57606a;text-align:{align};'
        f'padding:7px 10px;border-bottom:2px solid #d0d7de;'
        f'font-family:IBM Plex Mono,monospace;font-size:10px;'
        f'letter-spacing:0.08em;text-transform:uppercase;'
        f'white-space:nowrap;min-width:{width};">{col}</th>'
        for col, align, width in TABLE_COLS
        if col in display.columns or col == "Impact"
    )
 
    rows_html = ""
    for _, row in display.iterrows():
        impact  = str(row.get("Impact", "Neutral"))
        color   = IMPACT_COLORS.get(impact, "#8b949e")
        url     = row.get("Policy URL", "")
        title   = str(row.get("Policy Title", "—"))
        title_cell = (
            f'<a href="{url}" target="_blank" '
            f'style="color:#0969da;text-decoration:none;">{title}</a>'
            if url and str(url).startswith("http") else title
        )
 
        cells = ""
        for col, align, _ in TABLE_COLS:
            if col == "Policy Title":
                val_html = title_cell
            elif col == "Impact":
                val_html = (
                    f'<span style="color:{color};font-weight:600;">'
                    f'{"▲" if impact=="Bullish" else "▼" if impact=="Bearish" else "●"} '
                    f'{impact}</span>'
                )
            else:
                val = str(row.get(col, "—")) if col in row.index else "—"
                val_html = val if val else "—"
 
            cells += (
                f'<td style="color:#24292f;text-align:{align};'
                f'padding:7px 10px;border-bottom:1px solid #d0d7de;'
                f'font-family:IBM Plex Sans,sans-serif;font-size:12px;'
                f'vertical-align:top;">{val_html}</td>'
            )
        rows_html += f"<tr>{cells}</tr>"
 
    st.markdown(
        f'<div style="overflow-x:auto;overflow-y:auto;max-height:350px;border:1px solid #d0d7de;'
        f'border-radius:8px;margin-top:4px;">'
        f'<table style="width:100%;border-collapse:collapse;background:#ffffff;">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )
 
    total = len(filtered)
    shown = len(display)
    st.caption(
        f"Showing {shown} of {total} interventions · "
        f"Source: Global Trade Alert (GTA) · "
        f"🟢 Green = trade-liberalising · 🟡 Amber = uncertain · 🔴 Red = harmful"
    )