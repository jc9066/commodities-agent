"""
utils/fx.py
===========
FX data layer for the Commodities Intelligence Dashboard.

Pulls FX rates from Yahoo Finance (no API key needed) and exposes:

  1. fetch_fx_history()   — cached OHLCV history for one or more pairs
  2. get_fx_snapshot()    — latest rate + 1-day / 1-week / 1-month changes
  3. get_fx_attribution() — daily price-change decomposition into
                            "local move" vs "FX translation" components
  4. make_fx_monitor_chart()   — compact sparkline grid (Panel C)
  5. make_fx_overlay_chart()   — commodity price + FX rate dual-axis (Panel D)
  6. make_fx_attribution_chart() — waterfall bar chart (Panel B)
  7. render_fx_section()  — drop-in Streamlit renderer for dashboard.py

Yahoo Finance tickers
─────────────────────
Convention: "USD/MYR"  →  yf ticker "MYR=X"   (USD quoted against MYR)
            "USD/CNY"  →  yf ticker "CNY=X"
            "EUR/USD"  →  yf ticker "EURUSD=X"
            "USD/EUR"  →  yf ticker "EUR=X"
            "USD/CAD"  →  yf ticker "CAD=X"
            "USD/BRL"  →  yf ticker "BRL=X"
            "GBP/USD"  →  yf ticker "GBPUSD=X"  (GBP is base)
            "USD/GBP"  →  yf ticker "GBP=X" 
            "USD/USD"  →  synthetic, always 1.0

The module handles both "USD/XXX" and "XXX/USD" orientations so downstream
callers never need to think about it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional
import streamlit as st
import yfinance as yf 

# ─────────────────────────────────────────────────────────────────────────────
# TICKER MAP
# Maps your fx_pair strings → Yahoo Finance ticker strings
# ─────────────────────────────────────────────────────────────────────────────

# USD is the base; "XXX=X" gives units of XXX per 1 USD
_YF_TICKER: dict[str, str] = {
    "USD/MYR":  "MYR=X",
    "USD/CNY":  "CNY=X",
    "USD/CAD":  "CAD=X",
    "USD/BRL":  "BRL=X",
    "USD/GHS":  "GHS=X",
    "USD/EUR":  "EUR=X",  
    "USD/GBP":  "GBP=X",  
    "USD/USD":  None,          # synthetic; always 1.0
    # "CNY/USD":  "CNY=X",       # same underlying, inverted
}

# # Pairs that need to be inverted after download (because Yahoo returns USD/X not X/USD)
# _INVERT: set[str] = {"EUR/USD", "USD/GBP"}

# All unique pairs used across the COMMODITY_MAP
ALL_PAIRS: list[str] = [
    "USD/MYR", "USD/CNY", "USD/EUR", "USD/CAD",
    "USD/BRL", "USD/GBP", "USD/USD",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL: download with yfinance
# ─────────────────────────────────────────────────────────────────────────────

def _download_pair(pair: str, start: str, end: str) -> pd.Series:
    """
    Download daily close prices for one FX pair.
    Returns a pd.Series indexed by date, in units implied by `pair`.
    e.g.  "USD/MYR" → MYR per 1 USD
          "EUR/USD" → USD per 1 EUR
    """

    if pair == "USD/USD":
        dates = pd.bdate_range(start=start, end=end)
        return pd.Series(1.0, index=dates, name="USD/USD")

    ticker = _YF_TICKER.get(pair)
    if ticker is None:
        raise ValueError(f"Unknown FX pair: {pair!r}")

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned from Yahoo Finance for {ticker}")

    close = raw["Close"].squeeze()
    close.name = pair

    # if pair in _INVERT:
    #     close = 1.0 / close

    return close.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# CACHED HISTORY FETCHER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)   # refresh every hour
def fetch_fx_history(
    pairs: tuple[str, ...],
    lookback_days: int = 365,
) -> dict[str, pd.Series]:
    """
    Fetch and cache daily close history for multiple FX pairs.

    Parameters
    ----------
    pairs           : tuple of fx_pair strings, e.g. ("USD/MYR", "USD/CNY")
    lookback_days   : calendar days of history to retrieve

    Returns
    -------
    dict  {pair → pd.Series of close prices}

    Falls back to a deterministic synthetic series on any download error
    so the dashboard never hard-crashes due to a Yahoo outage.
    """
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    result: dict[str, pd.Series] = {}
    for pair in pairs:
        try:
            result[pair] = _download_pair(pair, start, end)
        except Exception as exc:
            st.warning(f"FX fetch failed for {pair} ({exc}). Using synthetic fallback.")
            result[pair] = _make_synthetic(pair, lookback_days)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT — latest rate + change metrics
# ─────────────────────────────────────────────────────────────────────────────

def get_fx_snapshot(series: pd.Series) -> dict:
    """
    Compute the current rate and period changes from a daily close series.

    Returns
    -------
    dict with keys:
        rate        float   latest close
        chg_1d      float   1-day % change
        chg_1w      float   5-trading-day % change
        chg_1m      float   21-trading-day % change
        chg_ytd     float   year-to-date % change
    """
    s = series.dropna()
    if s.empty:
        return {"rate": np.nan, "chg_1d": np.nan, "chg_1w": np.nan,
                "chg_1m": np.nan, "chg_ytd": np.nan}

    rate = float(s.iloc[-1])

    def _pct(n: int) -> float:
        if len(s) <= n:
            return np.nan
        return (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100

    # YTD: first trading day of the current year
    year_start = s[s.index >= f"{s.index[-1].year}-01-01"]
    ytd = (s.iloc[-1] / year_start.iloc[0] - 1) * 100 if not year_start.empty else np.nan

    return {
        "rate":    rate,
        "chg_1d":  _pct(1),
        "chg_1w":  _pct(5),
        "chg_1m":  _pct(21),
        "chg_ytd": ytd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LIVE FX MONITOR — Yahoo live snapshot + intraday ticks
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value) -> float:
    """Return a clean float or np.nan for missing / non-numeric Yahoo values."""
    try:
        if value is None:
            return np.nan
        if isinstance(value, pd.Series):
            value = value.dropna().iloc[-1] if not value.dropna().empty else np.nan
        elif isinstance(value, (list, tuple, np.ndarray)):
            value = value[-1] if len(value) else np.nan
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _first_valid_float(*values) -> float:
    """Return the first usable float from several possible Yahoo fields."""
    for value in values:
        x = _safe_float(value)
        if not np.isnan(x):
            return x
    return np.nan


def _pct_from_live(live_rate: float, reference_rate: float) -> float:
    """Percentage move from reference_rate to live_rate."""
    if np.isnan(live_rate) or np.isnan(reference_rate) or reference_rate == 0:
        return np.nan
    return (live_rate / reference_rate - 1.0) * 100.0


def _closed_daily_closes(hist_close: pd.Series) -> pd.Series:
    """
    Keep only completed daily candles where possible.
    This avoids using today's partial daily candle as the comparison base.
    """
    s = hist_close.dropna().copy()
    if s.empty:
        return s

    s.index = pd.to_datetime(s.index)
    today = pd.Timestamp.today(tz=s.index.tz).date() if getattr(s.index, "tz", None) else pd.Timestamp.today().date()
    closed = s[s.index.date < today]
    return closed if not closed.empty else s


def _nth_close_from_end(closed_close: pd.Series, n: int) -> float:
    """
    n=1 gives the latest completed close.
    n=5 gives roughly one trading week ago.
    n=21 gives roughly one trading month ago.
    """
    s = closed_close.dropna()
    if len(s) < n:
        return np.nan
    return _safe_float(s.iloc[-n])


@st.cache_data(ttl=300, show_spinner=False)  # live monitor refreshes every 5 minutes
def fetch_fx_live_monitor(
    pairs: tuple[str, ...],
    intraday_period: str = "1d",
    intraday_interval: str = "5m",
    history_period: str = "1y",
) -> pd.DataFrame:
    """
    Build the FX Monitor table from Yahoo live snapshot fields.

    Rate is taken from Yahoo's live snapshot first:
        currentPrice → regularMarketPrice → fast_info.last_price → intraday last close

    1D / 1W / 1M / YTD are calculated using the live rate as numerator.
    Historical daily closes are used only as reference points.
    Live Trend uses intraday ticks, not the old 30-day daily close sparkline.
    """
    rows: list[dict] = []

    for pair in pairs:
        if pair == "USD/USD":
            rows.append({
                "Pair": pair,
                "Rate": 1.0,
                "1D %": 0.0,
                "1W %": 0.0,
                "1M %": 0.0,
                "YTD %": 0.0,
                "Live Trend": [1.0] * 30,
            })
            continue

        ticker_symbol = _YF_TICKER.get(pair)
        if ticker_symbol is None:
            rows.append({
                "Pair": pair,
                "Rate": np.nan,
                "1D %": np.nan,
                "1W %": np.nan,
                "1M %": np.nan,
                "YTD %": np.nan,
                "Live Trend": [],
            })
            continue

        try:
            ticker = yf.Ticker(ticker_symbol)

            # Method 1: Yahoo live snapshot statistics
            try:
                live_info = ticker.info or {}
            except Exception:
                live_info = {}

            try:
                fast_info = dict(ticker.fast_info or {})
            except Exception:
                fast_info = {}

            # Method 2: latest intraday ticks for live fallback + sparkline
            try:
                intraday = ticker.history(
                    period=intraday_period,
                    interval=intraday_interval,
                    auto_adjust=True,
                )
            except Exception:
                intraday = pd.DataFrame()

            # Longer daily history is only for 1W / 1M / YTD references.
            try:
                daily = ticker.history(
                    period=history_period,
                    interval="1d",
                    auto_adjust=True,
                )
            except Exception:
                daily = pd.DataFrame()

            intraday_close = intraday["Close"].dropna() if "Close" in intraday else pd.Series(dtype=float)
            daily_close = daily["Close"].dropna() if "Close" in daily else pd.Series(dtype=float)
            closed_close = _closed_daily_closes(daily_close)

            bid = _safe_float(live_info.get("bid"))
            ask = _safe_float(live_info.get("ask"))
            bid_ask_mid = (bid + ask) / 2 if not np.isnan(bid) and not np.isnan(ask) else np.nan

            live_rate = _first_valid_float(
                live_info.get("currentPrice"),
                live_info.get("regularMarketPrice"),
                fast_info.get("last_price"),
                bid_ask_mid,
                intraday_close,
                daily_close,
            )

            prev_close = _first_valid_float(
                live_info.get("previousClose"),
                live_info.get("regularMarketPreviousClose"),
                fast_info.get("previous_close"),
                _nth_close_from_end(closed_close, 1),
            )

            week_ref = _nth_close_from_end(closed_close, 5)
            month_ref = _nth_close_from_end(closed_close, 21)

            year_start = closed_close[closed_close.index >= f"{datetime.today().year}-01-01"] if not closed_close.empty else pd.Series(dtype=float)
            ytd_ref = _safe_float(year_start.iloc[0]) if not year_start.empty else np.nan

            # Prefer live intraday sparkline. Fall back to daily 30 observations.
            trend = intraday_close.tail(80).tolist() if not intraday_close.empty else daily_close.tail(30).tolist()

            rows.append({
                "Pair": pair,
                "Rate": live_rate,
                "1D %": _pct_from_live(live_rate, prev_close),
                "1W %": _pct_from_live(live_rate, week_ref),
                "1M %": _pct_from_live(live_rate, month_ref),
                "YTD %": _pct_from_live(live_rate, ytd_ref),
                "Live Trend": trend,
            })

        except Exception:
            # Do not hard-crash the dashboard if Yahoo temporarily blocks / fails.
            fallback = _make_synthetic(pair, 365)
            snap = get_fx_snapshot(fallback)
            rows.append({
                "Pair": pair,
                "Rate": snap["rate"],
                "1D %": snap["chg_1d"],
                "1W %": snap["chg_1w"],
                "1M %": snap["chg_1m"],
                "YTD %": snap["chg_ytd"],
                "Live Trend": fallback.tail(30).tolist(),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# FX ATTRIBUTION — decompose commodity price moves
# ─────────────────────────────────────────────────────────────────────────────

def get_fx_attribution(
    local_prices: pd.Series,
    fx_series: pd.Series,
    local_ccy: str = "MYR",
    target_ccy: str = "USD",
    unit_conv: float = 1.0,
) -> pd.DataFrame:
    """
    Decompose daily commodity price changes (in local currency) into:
      • Local component  — move due to local price change, holding FX constant
      • FX component     — move due to FX change, holding local price constant
      • Total USD move   — actual change in USD terms

    The identity is:
        ΔP_usd ≈  ΔP_local / FX  +  P_local * Δ(1/FX)
                = local_component + fx_component  + interaction term

    We keep the interaction in the "local" bucket for simplicity (it's tiny
    for daily returns).

    Parameters
    ----------
    local_prices : commodity prices in local currency (e.g. MYR/MT)
    fx_series    : FX rate series — units of LOCAL per 1 USD
                   (so for MYR: fx = 4.47 means 4.47 MYR per USD)
    local_ccy    : label for the local currency
    target_ccy   : label for the target currency (usually USD)
    unit_conv    : multiply local price by this before dividing by FX,
                   e.g. 0.001 if local is in cents/bu and you want per MT

    Returns
    -------
    pd.DataFrame with columns:
        date, local_px, fx_rate, usd_px,
        local_component, fx_component, total_usd_chg
    All "component" columns are in target_ccy per unit.
    """
    # Align on common dates
    df = pd.DataFrame({
        "local_px": local_prices,
        "fx_rate":  fx_series,
    }).dropna()

    df["usd_px"] = df["local_px"] * unit_conv / df["fx_rate"]

    # Previous-day values
    df["local_px_prev"] = df["local_px"].shift(1)
    df["fx_prev"]       = df["fx_rate"].shift(1)
    df["usd_px_prev"]   = df["usd_px"].shift(1)

    # Local component: what would USD price have been if only local moved
    df["local_component"] = (df["local_px"] - df["local_px_prev"]) * unit_conv / df["fx_prev"]

    # FX component: what would USD price have been if only FX moved
    df["fx_component"] = df["local_px_prev"] * unit_conv * (
        1.0 / df["fx_rate"] - 1.0 / df["fx_prev"]
    )

    df["total_usd_chg"] = df["usd_px"] - df["usd_px_prev"]

    return df.dropna().reset_index().rename(columns={"index": "date"})


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

_LIGHT = dict(
    template      = "plotly_white",
    paper_bgcolor = "#ffffff",
    plot_bgcolor  = "#f6f8fa",
    font          = dict(family="IBM Plex Mono", size=11),
)
_GRID = dict(gridcolor="#d0d7de")
_PAL  = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffa657", "#79c0ff"]


def make_fx_monitor_chart(
    history: dict[str, pd.Series],
    active_pair: str,
) -> go.Figure:
    """
    Panel C — compact sparkline grid for all FX pairs in `history`.
    The active pair (currently selected commodity's pair) is highlighted.

    Returns a Plotly figure.
    """
    pairs = list(history.keys())
    n     = len(pairs)
    cols  = min(n, 3)
    rows  = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=pairs,
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    for idx, pair in enumerate(pairs):
        row = idx // cols + 1
        col = idx % cols  + 1
        s   = history[pair].dropna()
        if s.empty:
            continue

        color    = "#58a6ff" if pair == active_pair else "#8b949e"
        width    = 2.0      if pair == active_pair else 1.2
        snap     = get_fx_snapshot(s)
        chg_str  = f"{snap['chg_1d']:+.2f}%" if not np.isnan(snap['chg_1d']) else ""

        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines",
            name=pair,
            line=dict(color=color, width=width),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)",
            hovertemplate=f"{pair}<br>%{{x|%Y-%m-%d}}<br>%{{y:.4f}}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

        # Annotate current rate in top-right of each sparkline
        fig.add_annotation(
            xref=f"x{idx+1 if idx>0 else ''} domain",
            yref=f"y{idx+1 if idx>0 else ''} domain",
            x=1.0, y=1.05,
            text=f"<b>{snap['rate']:.4f}</b>  {chg_str}",
            showarrow=False,
            font=dict(size=9, color=color, family="IBM Plex Mono"),
            xanchor="right",
            row=row, col=col,
        )

    fig.update_layout(
        **_LIGHT,
        height=100 + rows * 120,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)

    return fig


def make_fx_overlay_chart(
    commodity_prices: pd.Series,
    fx_series: pd.Series,
    commodity_label: str,
    fx_pair: str,
    commodity_ccy: str,
) -> go.Figure:
    """
    Panel D — commodity price (left axis) with FX rate overlaid (right axis).
    Highlights periods where FX and price move in the same / opposite direction.

    Returns a Plotly figure.
    """
    # Align
    df = pd.DataFrame({"price": commodity_prices, "fx": fx_series}).dropna()

    # Correlation annotation
    if len(df) > 20:
        corr = df["price"].pct_change().corr(df["fx"].pct_change())
        corr_label = f"ρ(ΔPrice, ΔFX) = {corr:.2f}"
    else:
        corr_label = ""

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df.index, y=df["price"],
        name=commodity_label,
        mode="lines",
        line=dict(color="#58a6ff", width=1.8),
        hovertemplate=f"{commodity_label}: %{{y:,.1f}} {commodity_ccy}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["fx"],
        name=fx_pair,
        mode="lines",
        line=dict(color="#ffa657", width=1.4, dash="dot"),
        hovertemplate=f"{fx_pair}: %{{y:.4f}}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        **_LIGHT,
        height=320,
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        title=dict(
            text=f"Price vs FX  ·  {corr_label}",
            font=dict(size=11, family="IBM Plex Mono"),
            x=0,
        ),
    )
    fig.update_yaxes(title_text=f"{commodity_ccy}", **_GRID, secondary_y=False)
    fig.update_yaxes(title_text=fx_pair, **_GRID, secondary_y=True)

    return fig


def make_fx_attribution_chart(
    attr_df: pd.DataFrame,
    window: int = 60,
    commodity_ccy: str = "USD",
) -> go.Figure:
    """
    Panel B — stacked bar chart decomposing daily commodity price changes
    (in USD) into local-price and FX components.

    attr_df   : output of get_fx_attribution()
    window    : number of most-recent trading days to display
    """
    df = attr_df.tail(window).copy()

    fig = go.Figure()

    # FX component bars
    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["fx_component"],
        name="FX Component",
        marker_color="#ffa657",
        opacity=0.85,
        hovertemplate="FX: %{y:+.2f}<extra></extra>",
    ))

    # Local component bars
    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["local_component"],
        name="Local Price Component",
        marker_color="#58a6ff",
        opacity=0.85,
        hovertemplate="Local: %{y:+.2f}<extra></extra>",
    ))

    # Total USD change as line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["total_usd_chg"],
        name="Total USD Δ",
        mode="lines",
        line=dict(color="#e6edf3", width=1.5),
        hovertemplate="Total: %{y:+.2f}<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line_color="#21262d", line_width=1)

    fig.update_layout(
        **_LIGHT,
        barmode="relative",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.08, x=0),
        yaxis_title=f"USD/MT change",
        title=dict(
            text="Daily P&L Attribution: Local Price vs FX Translation",
            font=dict(size=11, family="IBM Plex Mono"),
            x=0,
        ),
    )
    fig.update_yaxes(**_GRID)
    fig.update_xaxes(**_GRID)

    return fig


def make_fx_normalized_chart(
    price_series_map: dict[str, pd.Series],   # {label: series_in_usd_per_mt}
    highlight_label: str | None = None,
) -> go.Figure:
    """
    Panel A — normalized USD/MT price comparison across exchanges / twin markets.

    price_series_map : dict of {display_label: pd.Series in USD/MT}
    highlight_label  : which series to draw thicker / brighter
    """
    fig = go.Figure()

    for i, (label, s) in enumerate(price_series_map.items()):
        s = s.dropna()
        if s.empty:
            continue
        is_hl  = (label == highlight_label)
        color  = _PAL[i % len(_PAL)]
        width  = 2.5 if is_hl else 1.2
        dash   = "solid" if is_hl else "dot"
        opacity = 1.0 if is_hl else 0.7

        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=label,
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
            hovertemplate=f"{label}: %{{y:,.1f}} USD/MT<extra></extra>",
        ))

    fig.update_layout(
        **_LIGHT,
        height=340,
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        yaxis_title="USD / MT",
        title=dict(
            text="Cross-Exchange Normalized Prices (USD/MT)",
            font=dict(size=11, family="IBM Plex Mono"),
            x=0,
        ),
    )
    fig.update_yaxes(**_GRID)
    fig.update_xaxes(**_GRID)
    return fig


def make_market_prices_chart(
    commodity_prices: pd.Series,
    twin_prices: dict[str, pd.Series],
    commodity_label: str,
    commodity_ccy: str,
) -> go.Figure:
    """
    01 Market Prices — raw selected market price with twin-market lines.

    The main commodity uses the left axis. Twin markets use an overlaid right
    axis because their units can differ across exchanges.
    """
    fig = go.Figure()

    main = commodity_prices.dropna()
    fig.add_trace(go.Scatter(
        x=main.index,
        y=main.values,
        name=commodity_label,
        mode="lines",
        line=dict(color="#58a6ff", width=2),
        yaxis="y1",
        hovertemplate=f"{commodity_label}: %{{y:,.1f}} {commodity_ccy}<extra></extra>",
    ))

    for i, (label, series) in enumerate(twin_prices.items()):
        s = series.dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index,
            y=s.values,
            name=label,
            mode="lines",
            line=dict(color=_PAL[(i + 1) % len(_PAL)], width=1.2, dash="dot"),
            yaxis="y2",
            visible=True if i < 2 else "legendonly",
            hovertemplate=f"{label}: %{{y:,.1f}}<extra></extra>",
        ))

    fig.update_layout(
        **_LIGHT,
        height=320,
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        yaxis=dict(title=commodity_ccy, gridcolor="#21262d"),
        yaxis2=dict(
            title="Twin markets",
            overlaying="y",
            side="right",
            showgrid=False,
            showticklabels=False,
        ),
        title=dict(
            text="Market Price vs Twin Markets",
            font=dict(size=11, family="IBM Plex Mono"),
            x=0,
        ),
    )
    fig.update_xaxes(**_GRID)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UNIT CONVERSION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# All conversion factors → USD/MT
# multiply raw price by this factor to get USD/MT
UNIT_CONV: dict[str, float] = {
    "MYR/MT":   None,   # needs FX: price / fx_rate
    "CNY/MT":   None,   # needs FX
    "USD/MT":   1.0,
    "USc/bu":   None,   # needs commodity-specific bu→MT factor; see _BUSHEL_MT
    "USc/lb":   22.0462,   # cents/lb → USD/MT  (1 MT = 2204.62 lb; /100 for cents)
    "USD/bbl":  None,   # energy, barrel basis — leave in bbl
    "CAD/T":    None,   # needs FX
    "EUR/MT":   None,   # needs FX
    "GBP/MT":   None,   # needs FX
    "USD/gal":  None,   # energy — leave in gal
    "USD/MMBtu": None,  # leave as is
    "CNY/bbl":  None,
}

# MT equivalent for common grain/oilseed bushel sizes
_BUSHEL_MT: dict[str, float] = {
    "Soybeans":  0.027216,   # 1 bu = 60 lb → /1000 * 2204.62 * 100 (for cents)
    "Corn":      0.025401,   # 1 bu = 56 lb
    "Wheat":     0.027216,   # 1 bu = 60 lb (SRW/HRW)
    "Bean Oil":  0.027216,
    "Soymeal":   0.027216,
}


def to_usd_per_mt(
    price: float | pd.Series,
    currency: str,
    commodity: str,
    fx_rate: float | pd.Series = 1.0,
) -> float | pd.Series:
    """
    Convert a price in `currency` units to USD/MT.

    Parameters
    ----------
    price       : raw price(s)
    currency    : one of the UNIT_CONV keys, e.g. "MYR/MT", "USc/bu"
    commodity   : used to look up bushel-to-MT factor
    fx_rate     : for non-USD currencies: units of local ccy per 1 USD
                  (so divide by fx_rate to get USD)

    Returns
    -------
    price(s) converted to USD/MT
    """
    if currency in ("USD/MT", "USD/bbl", "USD/gal", "USD/MMBtu"):
        return price

    if currency in ("MYR/MT", "CNY/MT", "CAD/T", "EUR/MT", "GBP/MT", "CNY/bbl"):
        return price / fx_rate

    if currency == "USc/lb":
        return price * 22.0462         # cents/lb → USD/MT

    if currency == "USc/bu":
        factor = _BUSHEL_MT.get(commodity, 0.027216)
        # price is in USc/bu → USD/bu = price/100 → USD/MT = (price/100)/factor
        return price / 100.0 / factor

    if currency == "USD/lbs":          # some Cotton quotes
        return price * 2204.62

    # fallback — return as-is with a warning
    return price


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC FALLBACK (deterministic, keyed on pair string)
# ─────────────────────────────────────────────────────────────────────────────

_REFERENCE_RATES: dict[str, float] = {
    "USD/MYR": 4.47,
    "USD/CNY": 7.25,
    "USD/EUR": 1.085,
    "USD/CAD": 1.36,
    "USD/BRL": 5.10,
    "USD/GBP": 0.79,
    "USD/USD": 1.00,
}


def _make_synthetic(pair: str, lookback_days: int = 365) -> pd.Series:
    """
    Deterministic GBM path for a given FX pair.
    Used as fallback when Yahoo Finance is unreachable.
    """
    base  = _REFERENCE_RATES.get(pair, 1.0)
    seed  = abs(hash(pair)) % (2**31)
    rng   = np.random.default_rng(seed)
    n     = lookback_days
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    rets  = rng.normal(0, 0.004, n)
    prices = base * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=dates, name=pair)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT RENDERER — drop into dashboard.py
# ─────────────────────────────────────────────────────────────────────────────

def render_fx_section(
    commodity: str,
    exchange: str,
    currency: str,
    fx_pair: str,
    commodity_prices: pd.Series,          # daily closes in local currency
    twin_prices: dict[str, pd.Series],    # {label: series in local ccy} — for Panel A
    twin_currencies: dict[str, str],      # {label: currency string}
    twin_fx_pairs: dict[str, str],        # {label: fx_pair string}
    lookback_days: int = 365,
) -> None:
    """
    Render the full FX Impact section into the active Streamlit column/container.

    Drop-in usage in dashboard.py:
    ─────────────────────────────
        from utils.fx import render_fx_section

        render_fx_section(
            commodity        = commodity,
            exchange         = exchange,
            currency         = currency,
            fx_pair          = fx_pair,
            commodity_prices = pd.Series(prices, index=dates_d),
            twin_prices      = {twin_label: pd.Series(twin_px, index=dates_d)},
            twin_currencies  = {twin_label: twin_currency},
            twin_fx_pairs    = {twin_label: twin_fx_pair},
        )
    """
    import streamlit as st

    st.markdown(
        '<div class="section-header">02 · FX Impact Analysis</div>',
        unsafe_allow_html=True,
    )

    # ── Determine all pairs we need ───────────────────────────────────────────
    # Keep a stable order: active pair first, then core monitor pairs, then twin pairs.
    monitor_pairs = tuple(
        pair for pair in dict.fromkeys([
            fx_pair,
            "USD/MYR", "USD/CNY", "USD/EUR", "USD/BRL", "USD/GBP", "USD/CAD",
            *twin_fx_pairs.values(),
        ])
        if pair != "USD/USD"
    )

    # ── Fetch FX history (cached) ─────────────────────────────────────────────
    with st.spinner("Loading FX data…"):
        history = fetch_fx_history(monitor_pairs, lookback_days=lookback_days)

    active_fx = history.get(fx_pair, _make_synthetic(fx_pair, lookback_days))

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_market, tab_norm, tab_attr, tab_overlay, tab_monitor = st.tabs([
        "01 · Market Prices",
        "📊 Normalized Prices",
        "🔀 Attribution",
        "📈 Price vs FX Overlay",
        "💱 FX Monitor",
    ])

    # ── 01 Market Prices ──────────────────────────────────────────────────────
    with tab_market:
        st.caption(
            "Selected commodity price with twin-market references. "
            "Twin-market lines are shown on a hidden secondary axis because units differ."
        )
        fig_market = make_market_prices_chart(
            commodity_prices = commodity_prices,
            twin_prices      = twin_prices,
            commodity_label  = f"{commodity} ({exchange})",
            commodity_ccy    = currency,
        )
        st.plotly_chart(fig_market, width="stretch")
        st.caption("Source: Bloomberg/dummy prices · Twin markets shown as trend references")

    # ── Panel A: Normalized price comparison ──────────────────────────────────
    with tab_norm:
        st.caption(
            "All prices converted to USD/MT using spot FX at date of observation. "
            "Allows apples-to-apples comparison across exchanges and currencies."
        )

        # Convert main commodity
        usd_main = to_usd_per_mt(commodity_prices, currency, commodity, active_fx)
        label_main = f"{commodity} ({exchange})"

        price_map: dict[str, pd.Series] = {label_main: usd_main}

        for lbl, ts in twin_prices.items():
            t_fx_pair = twin_fx_pairs.get(lbl, "USD/USD")
            t_ccy     = twin_currencies.get(lbl, "USD/MT")
            t_fx      = history.get(t_fx_pair, _make_synthetic(t_fx_pair, lookback_days))
            usd_twin  = to_usd_per_mt(ts, t_ccy, lbl.split("(")[0].strip(), t_fx)
            price_map[lbl] = usd_twin

        fig_norm = make_fx_normalized_chart(price_map, highlight_label=label_main)
        st.plotly_chart(fig_norm, width="stretch")
        st.caption("Source: Yahoo Finance (FX) · Bloomberg/dummy (prices) · Converted at daily close")

    # ── Panel B: FX Attribution ───────────────────────────────────────────────
    with tab_attr:
        st.caption(
            "Decomposes each day's USD price change into: "
            "**(1) local price movement** (fundamentals) vs "
            "**(2) FX translation** (currency drag/boost). "
            "Residual interaction term folded into local component."
        )

        # unit_conv: for non-MT or non-USD bases, adjust here
        # For MYR/MT the division by FX gives USD/MT directly
        unit_conv = 1.0

        if currency == "USc/bu":
            comm_stripped = commodity.split("(")[0].strip()
            factor = _BUSHEL_MT.get(comm_stripped, 0.027216)
            unit_conv = 1.0 / 100.0 / factor
        elif currency == "USc/lb":
            unit_conv = 22.0462 / 100.0

        attr_df = get_fx_attribution(
            local_prices = commodity_prices,
            fx_series    = active_fx,
            local_ccy    = currency,
            target_ccy   = "USD",
            unit_conv    = unit_conv if currency not in ("USD/MT",) else 1.0,
        )

        window = st.slider(
            "Rolling window (trading days)", 20, 252, 60, step=5, key="fx_attr_window"
        )
        fig_attr = make_fx_attribution_chart(attr_df, window=window, commodity_ccy="USD")
        st.plotly_chart(fig_attr, width="stretch")

        # Summary stats
        recent = attr_df.tail(window)
        if not recent.empty:
            fx_share = (
                recent["fx_component"].abs().sum()
                / (recent["total_usd_chg"].abs().sum() + 1e-9)
                * 100
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("FX share of total move", f"{fx_share:.1f}%",
                      help="% of gross USD price movement attributable to FX translation")
            c2.metric("Avg daily FX drag/boost",
                      f"{recent['fx_component'].mean():+.2f} USD/MT")
            c3.metric("Avg daily local move",
                      f"{recent['local_component'].mean():+.2f} USD/MT")

        st.caption("Source: Yahoo Finance (FX) · Bloomberg/dummy (prices)")

    # ── Panel D: Overlay chart ────────────────────────────────────────────────
    with tab_overlay:
        st.caption(
            "Commodity price (left axis) vs FX rate (right axis). "
            "Correlation ρ measures co-movement of daily % changes."
        )
        fig_ov = make_fx_overlay_chart(
            commodity_prices = commodity_prices,
            fx_series        = active_fx,
            commodity_label  = f"{commodity} ({exchange})",
            fx_pair          = fx_pair,
            commodity_ccy    = currency,
        )
        st.plotly_chart(fig_ov, width="stretch")
        st.caption(f"Source: Yahoo Finance ({fx_pair}) · Bloomberg/dummy (prices)")

# ── Panel C: FX Monitor ───────────────────────────────────────────────────
    with tab_monitor:
        st.caption(
            "Live Yahoo snapshot for the rate; historical closes are used only "
            "as reference points for 1D / 1W / 1M / YTD changes."
        )

        with st.spinner("Loading live FX snapshot…"):
            snap_df = fetch_fx_live_monitor(monitor_pairs)

        snap_df["Active"] = snap_df["Pair"].eq(fx_pair).map({True: "●", False: ""})
        snap_df = snap_df[[
            "Pair", "Rate", "1D %", "1W %", "1M %", "YTD %", "Active", "Live Trend"
        ]]

        st.dataframe(
            snap_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Pair":   st.column_config.TextColumn("Pair"),
                "Rate":   st.column_config.NumberColumn("Rate", format="%.4f"),
                "1D %":   st.column_config.NumberColumn("1D %",  format="%+.2f%%"),
                "1W %":   st.column_config.NumberColumn("1W %",  format="%+.2f%%"),
                "1M %":   st.column_config.NumberColumn("1M %",  format="%+.2f%%"),
                "YTD %":  st.column_config.NumberColumn("YTD %", format="%+.2f%%"),
                "Active": st.column_config.TextColumn(width="small"),
                "Live Trend": st.column_config.LineChartColumn(
                    "Live Trend",
                    y_min=None,
                    y_max=None,
                    width="medium",
                ),
            },
        )

        st.caption(
            "Source: Yahoo Finance · Rate from ticker.info / fast_info / intraday ticks · "
            "Live Trend uses 1D 5-minute ticks"
        )

    # # ── Panel C: FX Monitor ───────────────────────────────────────────────────
    # with tab_monitor:
    #     st.caption("Key FX rates across your commodity universe. Active pair is highlighted.")

    #     # Snapshot table
    #     rows = []
    #     for pair, s in history.items():
    #         snap = get_fx_snapshot(s)
    #         rows.append({
    #             "Pair":    pair,
    #             "Rate":    f"{snap['rate']:.4f}" if not np.isnan(snap["rate"]) else "—",
    #             "1D %":    f"{snap['chg_1d']:+.2f}%" if not np.isnan(snap["chg_1d"]) else "—",
    #             "1W %":    f"{snap['chg_1w']:+.2f}%" if not np.isnan(snap["chg_1w"]) else "—",
    #             "1M %":    f"{snap['chg_1m']:+.2f}%" if not np.isnan(snap["chg_1m"]) else "—",
    #             "YTD %":   f"{snap['chg_ytd']:+.2f}%" if not np.isnan(snap["chg_ytd"]) else "—",
    #             "Active":  "●" if pair == fx_pair else "",
    #         })

    #     snap_df = pd.DataFrame(rows)
    #     st.dataframe(
    #         snap_df,
    #         hide_index=True,
    #         width="stretch",
    #         column_config={
    #             "Active": st.column_config.TextColumn(width="small"),
    #         }
    #     )

    #     st.markdown("---")
    #     fig_mon = make_fx_monitor_chart(history, active_pair=fx_pair)
    #     st.plotly_chart(fig_mon, width="stretch")
    #     st.caption("Source: Yahoo Finance · Daily close · 12-month history")
