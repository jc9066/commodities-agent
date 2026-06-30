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
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

_LIGHT = dict(
    template      = "plotly_white",
    paper_bgcolor = "#ffffff",
    plot_bgcolor  = "#f6f8fa",
    font          = dict(family="IBM Plex Mono", size=11),
)
_GRID = dict(gridcolor="#d0d7de")


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

