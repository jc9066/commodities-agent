"""
utils/psd.py
────────────────────────────────────────────────────────────────
USDA FAS PSD data helpers.

Data is read from feather files managed by utils/psd_cache.py.
Ensure dashboard.py calls ensure_psd_cache() at startup.
"""

from __future__ import annotations

import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from utils.psd_cache import (
    CACHE_TTL_HOURS, COMMODITY_CODES, N_YEARS,
    _BASE_URL, _HEADERS, _is_fresh, _read_meta, _save, read_feather,
)

# ── Attribute config ───────────────────────────────────────────
ATTRIBUTE_LABELS: dict[int, str] = {
     57: "Imports",
     88: "Exports",
     28: "Production",
    125: "Domestic Consumption",
    176: "Ending Stocks",
     20: "Beginning Stocks",
     86: "Total Supply",
    178: "Total Distribution",
}

_ATTR_LABEL_TO_ID: dict[str, int] = {v: k for k, v in ATTRIBUTE_LABELS.items()}
_ATTR_ID_SET = set(ATTRIBUTE_LABELS.keys())


# ── Reference loaders (feather-first, live fallback) ──────────
@st.cache_data(ttl=86400, show_spinner=False)
def _load_country_df() -> pd.DataFrame:
    if _is_fresh("lookups"):
        df = read_feather("psd_lookup_countries")
        if df is not None:
            return df
    r = requests.get(f"{_BASE_URL}/api/psd/countries", headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=86400, show_spinner=False)
def _load_commodity_df() -> pd.DataFrame:
    if _is_fresh("lookups"):
        df = read_feather("psd_lookup_commodities")
        if df is not None:
            return df
    r = requests.get(f"{_BASE_URL}/api/psd/commodities", headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=86400, show_spinner=False)
def _load_unit_df() -> pd.DataFrame:
    if _is_fresh("lookups"):
        df = read_feather("psd_lookup_units")
        if df is not None:
            return df
    r = requests.get(f"{_BASE_URL}/api/psd/unitsOfMeasure", headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())


# ── Shared post-processing ─────────────────────────────────────
def _enrich(df: pd.DataFrame, comod_df: pd.DataFrame,
            unit_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(comod_df[["commodityCode", "commodityName"]],
                  on="commodityCode", how="left")
    df = df.merge(unit_df[["unitId", "unitDescription"]],
                  on="unitId", how="left")
    df["commodityCode"] = df["commodityCode"].astype(str)
    df["attributeId"]   = pd.to_numeric(df["attributeId"], errors="coerce").astype("Int64")
    df["marketYear"]    = pd.to_numeric(df["marketYear"],  errors="coerce").astype("Int64")
    df["value"]         = pd.to_numeric(df["value"],       errors="coerce")
    df["attributeLabel"] = df["attributeId"].apply(
        lambda x: ATTRIBUTE_LABELS.get(int(x)) if pd.notna(x) else None
    )
    _code_to_name = {v: k for k, v in COMMODITY_CODES.items()}
    df["commodityDisplay"] = df["commodityCode"].map(_code_to_name)
    df["marketYearLabel"] = df["marketYear"].apply(
        lambda x: f"{int(x)}/{int(x)+1}" if pd.notna(x) else None
    )
    return df


def _filter_attrs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["attributeId"].apply(
        lambda x: pd.notna(x) and int(x) in _ATTR_ID_SET
    )].copy()


# ── World S&D ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_world_sd(n_years: int = 5) -> pd.DataFrame:
    comod_df = _load_commodity_df()
    unit_df  = _load_unit_df()

    if _is_fresh("world_sd"):
        cached = read_feather("psd_world_sd")
        if cached is not None:
            return _filter_attrs(_enrich(cached, comod_df, unit_df))

    # background thread hasn't finished yet — live fetch
    prod_yr   = datetime.date.today().year - 1
    mkt_years = range(prod_yr - (n_years - 1), prod_yr + 1)
    codes     = list(dict.fromkeys(COMMODITY_CODES.values()))
    records: list[pd.DataFrame] = []
    for code in codes:
        for yr in mkt_years:
            try:
                r = requests.get(
                    f"{_BASE_URL}/api/psd/commodity/{code}/world/year/{yr}",
                    headers=_HEADERS, timeout=30)
                r.raise_for_status()
                data = r.json()
                if data:
                    records.append(pd.DataFrame(data))
            except Exception:
                pass
    if not records:
        return pd.DataFrame()
    raw = pd.concat(records, ignore_index=True)
    _save(raw, "psd_world_sd", "world_sd")
    return _filter_attrs(_enrich(raw, comod_df, unit_df))


# ── Country S&D ───────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_country_sd(n_years: int = 5) -> pd.DataFrame:
    comod_df   = _load_commodity_df()
    unit_df    = _load_unit_df()
    country_df = _load_country_df()

    if _is_fresh("country_sd"):
        cached = read_feather("psd_country_sd")
        if cached is not None:
            df = _enrich(cached, comod_df, unit_df)
            return df.merge(country_df[["countryCode", "countryName"]],
                            on="countryCode", how="left")

    prod_yr   = datetime.date.today().year - 1
    mkt_years = range(prod_yr - (n_years - 1), prod_yr + 1)
    codes     = list(dict.fromkeys(COMMODITY_CODES.values()))
    records: list[pd.DataFrame] = []
    for code in codes:
        for yr in mkt_years:
            try:
                r = requests.get(
                    f"{_BASE_URL}/api/psd/commodity/{code}/country/all/year/{yr}",
                    headers=_HEADERS, timeout=30)
                r.raise_for_status()
                data = r.json()
                if data:
                    records.append(pd.DataFrame(data))
            except Exception:
                pass
    if not records:
        return pd.DataFrame()
    raw = pd.concat(records, ignore_index=True)
    _save(raw, "psd_country_sd", "country_sd")
    df = _enrich(raw, comod_df, unit_df)
    return df.merge(country_df[["countryCode", "countryName"]],
                    on="countryCode", how="left")


# ── Chart theme ────────────────────────────────────────────────
_LIGHT = dict(
    template      = "plotly_white",
    paper_bgcolor = "#ffffff",
    plot_bgcolor  = "#f6f8fa",
    font          = dict(family="IBM Plex Mono", size=11),
)
_GRID = dict(gridcolor="#d0d7de")


# ── Tab 1: World S&D Balance ───────────────────────────────────
def make_sd_balance_chart(
    world_sd_df: pd.DataFrame,
    commodity: str,
    selected_attrs: list[str],
    chart_type: str = "Line",
    height: int = 340,
) -> go.Figure | None:
    code = COMMODITY_CODES.get(commodity)
    if code is None:
        return None
    plot_df = world_sd_df[
        (world_sd_df["commodityCode"] == code) &
        (world_sd_df["attributeLabel"].isin(selected_attrs))
    ].copy()
    if plot_df.empty:
        return None
    plot_df = plot_df.sort_values(["marketYear", "attributeLabel"])
    year_order = (
        plot_df[["marketYear", "marketYearLabel"]].drop_duplicates()
        .sort_values("marketYear")["marketYearLabel"].tolist()
    )
    unit_desc = plot_df["unitDescription"].dropna().iloc[0].strip()
    shared_kw = dict(
        data_frame      = plot_df,
        x               = "marketYearLabel",
        y               = "value",
        color           = "attributeLabel",
        category_orders = {"marketYearLabel": year_order},
        labels          = {"marketYearLabel": "Market Year", "value": unit_desc, "attributeLabel": ""},
    )
    if chart_type == "Line":
        fig = px.line(**shared_kw, markers=True)
    elif chart_type == "Bar":
        fig = px.bar(**shared_kw, barmode="group")
    else:
        fig = px.area(**shared_kw)
    fig.update_layout(
        **_LIGHT, height=height, yaxis_title=unit_desc,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


# ── Tab 2: Top Countries ───────────────────────────────────────
def make_top_countries_chart(
    country_sd_df: pd.DataFrame,
    commodity: str,
    attribute: str = "Exports",
    top_n: int = 10,
    height: int = 380,
) -> go.Figure | None:
    code    = COMMODITY_CODES.get(commodity)
    attr_id = _ATTR_LABEL_TO_ID.get(attribute)
    if code is None or attr_id is None:
        return None
    mkt_yr = datetime.date.today().year - 1
    mask = (
        (country_sd_df["commodityCode"].astype(str) == code) &
        (country_sd_df["attributeId"].apply(lambda x: pd.notna(x) and int(x) == attr_id)) &
        (country_sd_df["marketYear"].apply(lambda x: pd.notna(x) and int(x) == mkt_yr)) &
        (country_sd_df["value"] > 0) &
        (~country_sd_df["countryCode"].astype(str).isin(["00", "WLD"]))
    )
    sel = country_sd_df.loc[mask].copy()
    if sel.empty:
        return None
    unit_desc    = sel["unitDescription"].dropna().iloc[0].strip()
    mkt_yr_label = f"{mkt_yr}/{mkt_yr+1}"
    top_df       = sel.sort_values("value", ascending=False).head(top_n).copy()
    fig = px.bar(
            top_df, x="value", y="countryName", orientation="h",
            labels={"value": f"{attribute} {unit_desc}", "countryName": ""},
        )
        # Color bars by rank: darkest = highest value (already sorted descending)
    n = len(top_df)
    bar_colors = [
        f"rgba(30, 90, 160, {1.0 - i * 0.06})"   # dark navy → lighter blue
        for i in range(n)
    ]
    fig.update_traces(marker_color=bar_colors, marker_line_width=0)
    fig.update_layout(
        **_LIGHT, height=height, coloraxis_showscale=False,
        yaxis=dict(autorange="reversed", **_GRID),
        xaxis=dict(tickformat=",.0f", **_GRID),
        title=dict(text=f"Top {top_n} · {commodity} {attribute}  ·  {mkt_yr_label}",
                   font=dict(size=12), x=0),
        margin=dict(l=0, r=10, t=40, b=0),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


# ── Tab 3: Export Breakdown + Global Consumption ──────────────
def make_export_stacked_chart(
    country_sd_df: pd.DataFrame,
    commodity: str,
    top_n: int = 4,
    export_attr_id: int = 88,
    consumption_attr_id: int = 125,
    height: int = 420,
) -> go.Figure | None:
    code = COMMODITY_CODES.get(commodity)
    if code is None:
        return None
    df = country_sd_df[
        (country_sd_df["commodityCode"].astype(str) == code) &
        (~country_sd_df["countryCode"].astype(str).isin(["00", "WLD"]))
    ].copy()
    if df.empty:
        return None
    exp_df = df[df["attributeId"].apply(lambda x: pd.notna(x) and int(x) == export_attr_id)].copy()
    if exp_df.empty:
        return None
    top_countries = (
        exp_df.groupby("countryName", as_index=False)["value"].sum()
        .sort_values("value", ascending=False).head(top_n)["countryName"].tolist()
    )
    exp_df["exportGroup"] = exp_df["countryName"].where(
        exp_df["countryName"].isin(top_countries), "Rest of World"
    )
    exp_plot = (
        exp_df.groupby(["marketYear", "marketYearLabel", "exportGroup"], as_index=False)["value"].sum()
    )
    cons_df = df[df["attributeId"].apply(lambda x: pd.notna(x) and int(x) == consumption_attr_id)].copy()
    cons_agg = (
        cons_df.groupby(["marketYear", "marketYearLabel"], as_index=False)["value"].sum()
        .rename(columns={"value": "globalConsumption"})
    )
    year_order = (
        df[["marketYear", "marketYearLabel"]].drop_duplicates()
        .sort_values("marketYear")["marketYearLabel"].tolist()
    )
    unit_desc = df["unitDescription"].dropna().iloc[0].strip()
    fig = go.Figure()
    pal = px.colors.qualitative.Set2
    for i, grp in enumerate(top_countries + ["Rest of World"]):
        grp_df = exp_plot[exp_plot["exportGroup"] == grp]
        if grp_df.empty:
            continue
        fig.add_trace(go.Bar(x=grp_df["marketYearLabel"], y=grp_df["value"],
                             name=grp, marker_color=pal[i % len(pal)]))
    if not cons_agg.empty:
        fig.add_trace(go.Scatter(x=cons_agg["marketYearLabel"], y=cons_agg["globalConsumption"],
                                 mode="lines+markers", name="Global Consumption",
                                 line=dict(color="#f78166", width=2), marker=dict(size=6)))
    fig.update_layout(
        **_LIGHT, barmode="stack", height=height,
        xaxis=dict(categoryorder="array", categoryarray=year_order, title="Market Year", **_GRID),
        yaxis=dict(title=unit_desc, **_GRID),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=60),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        title=dict(text=f"{commodity} – Export Breakdown + Global Consumption",
                   font=dict(size=12), x=0),
    )
    return fig