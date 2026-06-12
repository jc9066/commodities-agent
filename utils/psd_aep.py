"""
utils/psd_aep.py
================
USDA FAS PSD — Actual / Estimate / Projection horizontal bar chart.

Reads from data/psd_aep_raw.feather (managed by utils/psd_cache.py).
Falls back to live API only if the background refresh hasn't finished yet.

Public API
----------
render_aep_tab(commodity: str) -> None
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from utils.psd_cache import (
    COMMODITY_CODES, _BASE_URL, _HEADERS,
    _is_fresh, _save, read_feather,
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
_LABEL_TO_ID: dict[str, int] = {v: k for k, v in ATTRIBUTE_LABELS.items()}


# ── Lookup tables ──────────────────────────────────────────────
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_lookup_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if _is_fresh("lookups"):
        c = read_feather("psd_lookup_countries")
        m = read_feather("psd_lookup_commodities")
        u = read_feather("psd_lookup_units")
        if c is not None and m is not None and u is not None:
            return c, m, u
    def _get(ep: str) -> pd.DataFrame:
        r = requests.get(_BASE_URL + ep, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    return _get("/api/psd/countries"), _get("/api/psd/commodities"), _get("/api/psd/unitsOfMeasure")


# ── Helpers ────────────────────────────────────────────────────
def _market_year_label(year: int) -> str:
    return f"{year}/{str(year + 1)[-2:]}"


def _clean_psd_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["marketYear", "calendarYear", "attributeId", "unitId"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "month" in df.columns:
        df["month"] = df["month"].astype(str).str.zfill(2)
    if {"calendarYear", "month"}.issubset(df.columns):
        df["snapshot_date"] = pd.to_datetime(
            df["calendarYear"].astype(str) + "-" + df["month"].astype(str) + "-01",
            errors="coerce",
        )
    return df


def _add_lookup_columns(df, country_df, comod_df, unit_df):
    df = df.copy()
    if "commodityName" not in df.columns:
        df = df.merge(comod_df[["commodityCode", "commodityName"]].drop_duplicates(),
                      on="commodityCode", how="left")
    if "unitDescription" not in df.columns:
        df = df.merge(unit_df[["unitId", "unitDescription"]].drop_duplicates(),
                      on="unitId", how="left")
    if "countryName" not in df.columns:
        df = df.merge(country_df[["countryCode", "countryName"]].drop_duplicates(),
                      on="countryCode", how="left")
    return df


def _get_display_unit_and_scale(unit_description: str) -> tuple[str, float]:
    if pd.isna(unit_description):
        return "value", 1
    unit = str(unit_description).upper().replace(".", "").strip()
    if "1000 MT" in unit:        return "million metric tons", 1000
    if "1000 HA" in unit:        return "million hectares", 1000
    if "1000 60 KG BAGS" in unit: return "million 60kg bags", 1000
    if "1000 480 LB BALES" in unit: return "million 480 lb bales", 1000
    if "1000 BALES" in unit:     return "million bales", 1000
    return unit_description, 1


# ── Data fetch (feather-first) ─────────────────────────────────
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _fetch_psd_years(commodity_code: str, years: tuple[int, ...]) -> pd.DataFrame:
    if _is_fresh("aep_raw"):
        cached = read_feather("psd_aep_raw")
        if cached is not None and "_commodity_code" in cached.columns:
            subset = cached[
                (cached["_commodity_code"] == commodity_code) &
                (pd.to_numeric(cached.get("marketYear", pd.Series(dtype=float)),
                               errors="coerce").isin(years))
            ].drop(columns=["_commodity_code"], errors="ignore")
            if not subset.empty:
                return subset.reset_index(drop=True)

    # background thread hasn't finished — live fetch
    frames = []
    for year in years:
        r = requests.get(
            f"{_BASE_URL}/api/psd/commodity/{commodity_code}/country/all/year/{year}",
            headers=_HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if data:
            frames.append(pd.DataFrame(data))
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result["_commodity_code"] = commodity_code
    _save(result, "psd_aep_raw", "aep_raw")
    return result.drop(columns=["_commodity_code"], errors="ignore")


# ── Snapshot selection ─────────────────────────────────────────
def _choose_snapshot_month(df, projection_year, as_of_date=None):
    if as_of_date is None:
        as_of_date = dt.date.today()
    target  = pd.Timestamp(as_of_date.year, as_of_date.month, 1)
    proj_df = df[df["marketYear"] == projection_year].dropna(subset=["snapshot_date"])
    if proj_df.empty:
        raise ValueError(f"No projection-year data for market year {projection_year}.")
    if not proj_df[proj_df["snapshot_date"] == target].empty:
        return target
    prev = proj_df[proj_df["snapshot_date"] <= target]
    return prev["snapshot_date"].max() if not prev.empty else proj_df["snapshot_date"].max()


# ── Prepare ────────────────────────────────────────────────────
def _prepare_aep_data(commodity_name, attribute_id, top_n=6, as_of_date=None):
    if as_of_date is None:
        as_of_date = dt.date.today()
    cur = as_of_date.year
    actual_year, estimate_year, projection_year = cur - 2, cur - 1, cur
    years = (actual_year, estimate_year, projection_year)

    commodity_code = COMMODITY_CODES[commodity_name]
    country_df, comod_df, unit_df = _load_lookup_tables()

    raw_df = _fetch_psd_years(commodity_code, years)
    if raw_df.empty:
        raise ValueError(f"No USDA PSD data for {commodity_name}.")

    df = _add_lookup_columns(raw_df, country_df, comod_df, unit_df)
    df = _clean_psd_types(df)
    df = df[df["attributeId"] == attribute_id].copy()
    if df.empty:
        raise ValueError(f"No data for {commodity_name} — {ATTRIBUTE_LABELS.get(attribute_id)}.")

    snapshot_date = _choose_snapshot_month(df, projection_year, as_of_date)
    snapshot_df   = df[(df["snapshot_date"] == snapshot_date) & (df["marketYear"].isin(years))].copy()
    if snapshot_df.empty:
        raise ValueError("No data for selected snapshot month.")
    snapshot_df = snapshot_df.dropna(subset=["countryName"])

    top_country_codes = (
        snapshot_df[
            (snapshot_df["marketYear"] == projection_year) &
            (snapshot_df["value"].notna()) & (snapshot_df["value"] > 0)
        ]
        .sort_values("value", ascending=False)
        .drop_duplicates("countryCode")
        .head(top_n)["countryCode"].tolist()
    )
    if not top_country_codes:
        raise ValueError("No positive projection values for top country selection.")

    plot_base = snapshot_df[snapshot_df["countryCode"].isin(top_country_codes)].copy()
    wide = plot_base.pivot_table(
        index=["countryCode", "countryName"], columns="marketYear",
        values="value", aggfunc="first",
    ).reset_index()
    for year in years:
        if year not in wide.columns:
            wide[year] = np.nan

    plot_df = wide.melt(
        id_vars=["countryCode", "countryName"], value_vars=list(years),
        var_name="marketYear", value_name="value",
    )
    plot_df["country_rank"] = plot_df["countryCode"].map(
        {code: i for i, code in enumerate(top_country_codes)}
    )

    snapshot_month_name = snapshot_date.strftime("%B")
    status_map = {actual_year: "Actual", estimate_year: "Estimate",
                  projection_year: f"{snapshot_month_name} Projection"}
    short_status_map  = {actual_year: "Actual", estimate_year: "Estimate", projection_year: "Projection"}
    series_order_map  = {actual_year: 0, estimate_year: 1, projection_year: 2}

    plot_df["status"]       = plot_df["marketYear"].map(short_status_map)
    plot_df["status_label"] = plot_df["marketYear"].apply(
        lambda y: f"{_market_year_label(int(y))} {status_map[int(y)]}"
    )
    plot_df["series_order"] = plot_df["marketYear"].map(series_order_map)

    mode_val  = snapshot_df["unitDescription"].dropna().mode() if "unitDescription" in snapshot_df else []
    raw_unit  = mode_val.iloc[0] if len(mode_val) > 0 else ""
    display_unit, scale = _get_display_unit_and_scale(raw_unit)
    plot_df["display_value"] = plot_df["value"] / scale
    plot_df = plot_df.sort_values(["country_rank", "series_order"], ascending=[True, True])

    meta = dict(
        commodity_name=commodity_name, attribute_id=attribute_id,
        attribute_name=ATTRIBUTE_LABELS.get(attribute_id, str(attribute_id)),
        snapshot_date=snapshot_date, snapshot_month_name=snapshot_month_name,
        snapshot_year=snapshot_date.year, raw_unit=raw_unit,
        display_unit=display_unit, scale=scale,
        actual_year=actual_year, estimate_year=estimate_year,
        projection_year=projection_year, top_n=top_n,
    )
    return plot_df, meta


# ── Chart ──────────────────────────────────────────────────────
def _make_aep_chart(plot_df, meta):
    color_map = {"Actual": "#BFC5CC", "Estimate": "#8F99A6", "Projection": "#1F77B4"}
    plot_df = plot_df.copy()
    plot_df["bar_color"]  = plot_df["status"].map(color_map)
    plot_df["text_label"] = plot_df["display_value"].apply(
        lambda x: "" if pd.isna(x) else f"{x:,.2f}"
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["display_value"],
        y=[plot_df["countryName"], plot_df["status_label"]],
        orientation="h", marker_color=plot_df["bar_color"],
        text=plot_df["text_label"], textposition="outside",
        customdata=np.stack([plot_df["countryName"], plot_df["status_label"],
                             plot_df["value"], plot_df["display_value"]], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
            "Raw: %{customdata[2]:,.2f}<br>"
            f"Displayed: %{{customdata[3]:,.2f}} {meta['display_unit']}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"{meta['commodity_name']} — {meta['attribute_name']} ({meta['display_unit']})"
              f"<br><sup>Top {meta['top_n']} countries · "
              f"{meta['snapshot_month_name']} {meta['snapshot_year']} USDA PSD · "
              f"Raw unit: {meta['raw_unit']}</sup>",
        height=max(500, meta["top_n"] * 95),
        margin=dict(l=180, r=80, t=90, b=50),
        plot_bgcolor="#f6f8fa", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Mono", size=11, color="#24292f"),
        showlegend=False, bargap=0.18,
        xaxis_title=meta["display_unit"], yaxis_title=None,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#21262d", zeroline=False, rangemode="tozero")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


# ── Public render ──────────────────────────────────────────────
def render_aep_tab(commodity: str) -> None:
    if commodity not in COMMODITY_CODES:
        st.info(f"USDA FAS PSD data not configured for **{commodity}**.")
        return

    ctrl_a, ctrl_b = st.columns([3, 1])
    with ctrl_a:
        attr_label = st.selectbox(
            "Attribute", list(_LABEL_TO_ID.keys()),
            index=list(_LABEL_TO_ID.keys()).index("Production"),
            key="aep_attr",
        )
    with ctrl_b:
        top_n = st.slider("Top N countries", 3, 12, 6, key="aep_topn")

    with st.spinner("Loading USDA FAS PSD data…"):
        try:
            plot_df, meta = _prepare_aep_data(
                commodity_name=commodity,
                attribute_id=_LABEL_TO_ID[attr_label],
                top_n=top_n,
            )
        except Exception as e:
            st.error(f"Failed to load AEP data: {e}")
            return

    st.plotly_chart(_make_aep_chart(plot_df, meta), width="stretch")
    st.caption(
        f"Source: USDA FAS PSD · Snapshot: {meta['snapshot_month_name']} {meta['snapshot_year']} · "
        f"Raw unit: {meta['raw_unit']}"
    )