"""
utils/atlas_market_data.py
==========================

Atlas-backed market price layer for the Streamlit commodities dashboard.

What this replaces:
- The dummy `make_price_series(...)` numbers used in dashboard.py.
- The old Market Prices tab that showed selected market + twin-market lines.

What this provides:
- Selected futures price line + volume bar in one Plotly chart.
- Chart filtering by quick range or custom dates.
- Twin-market normalized price index chart in the Normalized Prices tab.
- Clear "unavailable currently" messages for exchanges/products that Atlas does
  not currently provide.

Drop this file into your project as:
    utils/atlas_market_data.py

Then see the integration snippet at the bottom of this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import streamlit as st
from dotenv import load_dotenv
from atlas_client import AtlasClient

# Reuse your existing FX helpers where possible. The fallback import lets this
# module run both from `utils/` and from the project root during quick testing.
try:  # preferred when this file is saved in utils/
    from utils.fx_live_monitor import (  # type: ignore
        fetch_fx_history,
        get_fx_snapshot,
        to_usd_per_mt,
        # get_fx_attribution,
        # make_fx_attribution_chart,
        make_fx_overlay_chart,
        fetch_fx_live_monitor,
    )
except Exception:  # fallback when this file is beside fx_live_monitor.py
    from fx_live_monitor import (  # type: ignore
        fetch_fx_history,
        get_fx_snapshot,
        to_usd_per_mt,
        # get_fx_attribution,
        # make_fx_attribution_chart,
        make_fx_overlay_chart,
        fetch_fx_live_monitor,
    )

from utils.atlas_cache import load_cached_history


# ─────────────────────────────────────────────────────────────────────────────
# Atlas connection
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_env_names() -> None:
    """
    atlas-client expects ATLAS_DB_USER / ATLAS_DB_PASSWORD or DB_USER / DB_PASSWORD.
    Some local .env files use DB_ATLAS_USER / DB_ATLAS_USER_PASSWORD instead.
    This maps the local names into the names atlas-client reads without exposing
    any credentials in the dashboard.
    """
    if not os.getenv("ATLAS_DB_USER") and os.getenv("DB_ATLAS_USER"):
        os.environ["ATLAS_DB_USER"] = os.environ["DB_ATLAS_USER"]
    if not os.getenv("ATLAS_DB_PASSWORD") and os.getenv("DB_ATLAS_USER_PASSWORD"):
        os.environ["ATLAS_DB_PASSWORD"] = os.environ["DB_ATLAS_USER_PASSWORD"]


@st.cache_resource(show_spinner=False)
def get_atlas_client() -> AtlasClient:
    load_dotenv()
    _normalise_env_names()
    return AtlasClient.from_env()


def to_df(data) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.reset_index() if data.index.name else data.copy()
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Product / availability mapping
# Source: product_specs.xlsx + the exchange availability table you shared.
# ─────────────────────────────────────────────────────────────────────────────

def _norm(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().replace("/", " ").split())

@dataclass(frozen=True)
class AtlasProductSpec:
    commodity: str
    exchange_label: str
    exchange_code: str       # Atlas MIC / exchange code, e.g. XKLS
    product_code: str        # Atlas product/root code, e.g. FCPO
    product_desc: str
    currency_unit: str       # dashboard unit, e.g. MYR/MT
    provider_exchange: str   # Barchart/MIC display, e.g. MDEX
    provider_symbol: str
    data_available: bool

    @property
    def key(self) -> tuple[str, str]:
        return (_norm(self.commodity), _norm(self.exchange_label))


# Exchange availability from your screenshot.
EXCHANGE_DATA_AVAILABLE: dict[str, bool] = {
    "XDCE": True,   # DCE
    "IFUS": False,  # ICEUS
    "IFCA": True,   # WCE
    "IFEU": False,  # ICE
    "XZCE": True,   # CZCE
    "XCBT": False,  # CBOT
    "XCME": False,  # CME
    "XNYM": False,  # NYMEX
    "XKLS": True,   # MDEX
    "XSGE": True,   # SHFE
    "MIHI": True,   # MIAX
    "XPAR": True,   # MATIF
}


def _available(exchange_code: str) -> bool:
    return bool(EXCHANGE_DATA_AVAILABLE.get(exchange_code, False))


PRODUCT_SPECS: list[AtlasProductSpec] = [
    # ── Oilseeds ─────────────────────────────────────────────────────────────
    AtlasProductSpec("Crude Palm Oil", "BMD (Malaysia)", "XKLS", "FCPO", "Crude Palm Oil", "MYR/MT", "MDEX", "KO", _available("XKLS")),
    AtlasProductSpec("Palm Olein", "DCE (China)", "XDCE", "P", "RBD Palm Olein", "CNY/MT", "DCE", "YH", _available("XDCE")),
    AtlasProductSpec("Soybeans", "DCE (China)", "XDCE", "A", "No.1 Soybean", "CNY/MT", "DCE", "XT", _available("XDCE")),
    AtlasProductSpec("Soybeans No.2", "DCE (China)", "XDCE", "B", "No.2 Soybean", "CNY/MT", "DCE", "XS", _available("XDCE")),
    AtlasProductSpec("Bean Oil", "DCE (China)", "XDCE", "Y", "Soybean Oil", "CNY/MT", "DCE", "XX", _available("XDCE")),
    AtlasProductSpec("Soymeal", "DCE (China)", "XDCE", "M", "Soybean Meal", "CNY/MT", "DCE", "XU", _available("XDCE")),
    AtlasProductSpec("Canola/Rapeseed", "ICE (Canada)", "IFCA", "RS", "Canola", "CAD/T", "WCE", "RS", _available("IFCA")),
    AtlasProductSpec("Canola/Rapeseed", "LIFFE/Euronext (EU)", "XPAR", "ECO", "Rapeseed", "EUR/MT", "MATIF", "XR", _available("XPAR")),
    AtlasProductSpec("Rapeseed Oil", "ZCE (China)", "XZCE", "OI", "Rapeseed Oil", "CNY/MT", "CZCE", "YF", _available("XZCE")),
    # Unavailable but mapped so the dashboard can display an explicit message.
    AtlasProductSpec("Soybeans", "CME (US)", "XCBT", "ZS", "Soybean", "USc/bu", "CBOT", "ZS", _available("XCBT")),
    AtlasProductSpec("Bean Oil", "CME (US)", "XCBT", "ZL", "Soybean Oil", "USc/lb", "CBOT", "ZL", _available("XCBT")),
    AtlasProductSpec("Soymeal", "CME (US)", "XCBT", "ZM", "Soybean Meal", "USD/MT", "CBOT", "ZM", _available("XCBT")),

    # ── Grains ───────────────────────────────────────────────────────────────
    AtlasProductSpec("Corn", "DCE (China)", "XDCE", "C", "Yellow Corn", "CNY/MT", "DCE", "XV", _available("XDCE")),
    AtlasProductSpec("Corn", "CME (US)", "XCBT", "ZC", "Corn", "USc/bu", "CBOT", "ZC", _available("XCBT")),
    AtlasProductSpec("Wheat", "MIAX (US)", "MIHI", "MWE", "Hard Red Spring Wheat", "USc/bu", "MIAX", "MW", _available("MIHI")),
    AtlasProductSpec("Wheat", "LIFFE/Euronext (EU)", "XPAR", "EBM", "Milling Wheat No.2", "EUR/MT", "MATIF", "ML", _available("XPAR")),
    AtlasProductSpec("Wheat", "CME (US)", "XCBT", "ZW", "Chicago SRW Wheat", "USc/bu", "CBOT", "ZW", _available("XCBT")),

    # ── Softs ────────────────────────────────────────────────────────────────
    AtlasProductSpec("Sugar", "ZCE (China)", "XZCE", "SR", "White Sugar", "CNY/MT", "CZCE", "WO", _available("XZCE")),
    AtlasProductSpec("Sugar", "ICE (US)", "IFUS", "SB", "Sugar No. 11", "USc/lb", "ICEUS", "SB", _available("IFUS")),
    AtlasProductSpec("Coffee", "ICE (US)", "IFUS", "KC", "Coffee C", "USc/lb", "ICEUS", "KC", _available("IFUS")),
    AtlasProductSpec("Coffee", "ICE (Europe)", "IFEU", "RC", "Robusta Coffee", "USD/MT", "ICE", "RM", _available("IFEU")),
    AtlasProductSpec("Cocoa", "ICE (US)", "IFUS", "CC", "Cocoa", "USD/MT", "ICEUS", "CC", _available("IFUS")),
    AtlasProductSpec("Cocoa", "ICE (Europe)", "IFEU", "C", "London Cocoa/Cocoa #7", "GBP/MT", "ICE", "CA", _available("IFEU")),
    AtlasProductSpec("Cotton", "ZCE (China)", "XZCE", "CF", "Cotton", "CNY/MT", "CZCE", "WQ", _available("XZCE")),
    AtlasProductSpec("Cotton", "ICE (US)", "IFUS", "CT", "Cotton No. 2", "USc/lb", "ICEUS", "CT", _available("IFUS")),

    # ── Livestock / unavailable ──────────────────────────────────────────────
    AtlasProductSpec("Lean Hogs", "CME (US)", "XCME", "HE", "Lean Hog", "USc/lb", "CME", "HE", _available("XCME")),
    AtlasProductSpec("Live Cattle", "CME (US)", "XCME", "LE", "Live Cattle", "USc/lb", "CME", "LE", _available("XCME")),
    AtlasProductSpec("Feeder Cattle", "CME (US)", "XCME", "GF", "Feeder Cattle", "USc/lb", "CME", "GF", _available("XCME")),

    # ── Energy ───────────────────────────────────────────────────────────────
    AtlasProductSpec("Fuel Oil 380 CST", "SHFE (China)", "XSGE", "FU", "Fuel Oil", "CNY/MT", "SHFE", "VQ", _available("XSGE")),
    AtlasProductSpec("LSFO", "SHFE (China)", "XSGE", "LU", "Low Sulfur Fuel Oil", "CNY/MT", "SHFE", "VA", _available("XSGE")),
    AtlasProductSpec("MS Crude Oil", "SHFE (China)", "XSGE", "SC", "Medium Sour Crude Oil", "CNY/bbl", "SHFE", "UW", _available("XSGE")),
    AtlasProductSpec("Brent Crude", "ICE (Europe)", "IFEU", "BRN", "Brent Crude", "USD/bbl", "ICE", "CB", _available("IFEU")),
    AtlasProductSpec("WTI Crude", "ICE (Europe)", "IFEU", "WBS", "WTI Crude", "USD/bbl", "ICE", "WI", _available("IFEU")),
    AtlasProductSpec("WTI Crude", "CME (US)", "XNYM", "CL", "Light Sweet Crude Oil", "USD/bbl", "NYMEX", "CL", _available("XNYM")),
    AtlasProductSpec("Henry Hub Natural Gas", "CME (US)", "XNYM", "NG", "Henry Hub Natural Gas", "USD/MMBtu", "NYMEX", "NG", _available("XNYM")),
    AtlasProductSpec("Heating Oil", "CME (US)", "XNYM", "HO", "NY Harbor ULSD", "USD/gal", "NYMEX", "HO", _available("XNYM")),
    AtlasProductSpec("RBOB Gasoline", "CME (US)", "XNYM", "RB", "RBOB Gasoline", "USD/gal", "NYMEX", "RB", _available("XNYM")),
    AtlasProductSpec("GasOil", "ICE (Europe)", "IFEU", "G", "Low Sulphur Gasoil", "USD/MT", "ICE", "LF", _available("IFEU")),
]

SPEC_BY_KEY: dict[tuple[str, str], AtlasProductSpec] = {spec.key: spec for spec in PRODUCT_SPECS}


# Twin labels used in your dashboard.py are not always exact commodity/exchange
# names, so resolve them explicitly.
TWIN_LABEL_TO_SPEC_KEY: dict[str, tuple[str, str]] = {
    "Crude Palm Oil (BMD)": ("Crude Palm Oil", "BMD (Malaysia)"),
    "Palm Olein (DCE)": ("Palm Olein", "DCE (China)"),
    "Bean Oil (CME)": ("Bean Oil", "CME (US)"),
    "Bean Oil (DCE)": ("Bean Oil", "DCE (China)"),
    "Soybeans (CME)": ("Soybeans", "CME (US)"),
    "Soybeans (DCE)": ("Soybeans", "DCE (China)"),
    "Soybeans(DCE)": ("Soybeans", "DCE (China)"),
    "Soymeal (CME)": ("Soymeal", "CME (US)"),
    "Soymeal (DCE)": ("Soymeal", "DCE (China)"),
    "Corn (CME)": ("Corn", "CME (US)"),
    "Corn (DCE)": ("Corn", "DCE (China)"),
    "Rapeseed Oil (ZCE)": ("Rapeseed Oil", "ZCE (China)"),
    "Rapeseed (LIFFE/Euronext)": ("Canola/Rapeseed", "LIFFE/Euronext (EU)"),
    "Rapeseed (LIFFE)": ("Canola/Rapeseed", "LIFFE/Euronext (EU)"),
    "Canola (ICE)": ("Canola/Rapeseed", "ICE (Canada)"),
    "Wheat (CME)": ("Wheat", "CME (US)"),
    "Wheat (SRW (CME)": ("Wheat", "CME (US)"),
    "Wheat HRW (CME)": ("Wheat", "CME (US)"),
    "Minneapolis Wheat, HRS (MIAX)": ("Wheat", "MIAX (US)"),
    "Milling Wheat (LIFFE/Euronext)": ("Wheat", "LIFFE/Euronext (EU)"),
    "White Sugar (ZCE)": ("Sugar", "ZCE (China)"),
    "White Sugar (ICE)": ("Sugar", "ICE (US)"),
    "Raw Sugar (ICE)": ("Sugar", "ICE (US)"),
    "Cotton (ZCE)": ("Cotton", "ZCE (China)"),
    "Cotton No. 2 (ICE)": ("Cotton", "ICE (US)"),
    "Robusta Coffee (ICE)": ("Coffee", "ICE (Europe)"),
    "Arabica Coffee (ICE)": ("Coffee", "ICE (US)"),
    "Cocoa (ICE)": ("Cocoa", "ICE (US)"),
    "LDN Cocoa (ICE)": ("Cocoa", "ICE (Europe)"),
    "Gasoil (ICE)": ("GasOil", "ICE (Europe)"),
    "Heating Oil (CME)": ("Heating Oil", "CME (US)"),
    "Brent Crude (CME/ICE)": ("Brent Crude", "ICE (Europe)"),
    "WTI Crude (CME/ICE)": ("WTI Crude", "CME (US)"),
    "Live Cattle (CME)": ("Live Cattle", "CME (US)"),
    "Feeder Cattle (CME)": ("Feeder Cattle", "CME (US)"),
}

FX_PAIR_BY_UNIT: dict[str, str] = {
    "MYR/MT": "USD/MYR",
    "CNY/MT": "USD/CNY",
    "CNY/bbl": "USD/CNY",
    "CAD/T": "USD/CAD",
    "EUR/MT": "USD/EUR",
    "GBP/MT": "USD/GBP",
    "USD/MT": "USD/USD",
    "USD/bbl": "USD/USD",
    "USD/gal": "USD/USD",
    "USD/MMBtu": "USD/USD",
    "USc/bu": "USD/USD",
    "USc/lb": "USD/USD",
    "USD/lb": "USD/USD",
}





def resolve_product_spec(commodity: str, exchange_label: str) -> AtlasProductSpec | None:
    return SPEC_BY_KEY.get((_norm(commodity), _norm(exchange_label)))


def resolve_twin_spec(twin_label: str) -> AtlasProductSpec | None:
    key = TWIN_LABEL_TO_SPEC_KEY.get(twin_label)
    if key:
        return SPEC_BY_KEY.get((_norm(key[0]), _norm(key[1])))

    # Best-effort fallback if a new twin label looks like "Commodity (Exchange)".
    lower = twin_label.lower()
    for spec in PRODUCT_SPECS:
        exch_token = spec.exchange_label.split("(")[-1].replace(")", "").strip().lower()
        if spec.commodity.lower() in lower and exch_token and exch_token in lower:
            return spec
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Atlas fetchers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_atlas_futures_contracts(exchange_code: str, product_code: str) -> pd.DataFrame:
    """Return active futures contracts for an Atlas exchange/product."""
    client = get_atlas_client()
    df = to_df(client.futures(exchange_code=exchange_code, active_only=True))
    if df.empty:
        return df

    product_code_upper = product_code.upper()

    # Prefer explicit product/root columns where Atlas exposes them.
    filter_cols = [
        "product_code", "exchange_symbol", "root_symbol", "root", "symbol_root",
        "product", "exchange_product_code",
    ]
    for col in filter_cols:
        if col in df.columns:
            mask = df[col].astype(str).str.upper().eq(product_code_upper)
            if mask.any():
                return df.loc[mask].copy()

    # Fallback: canonical symbols normally start with the product/root code
    # e.g. FCPOK26.XKLS.MYR.FUT.
    if "canonical_symbol" in df.columns:
        mask = df["canonical_symbol"].astype(str).str.upper().str.startswith(product_code_upper)
        if mask.any():
            return df.loc[mask].copy()

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_atlas_futures_contracts_all(exchange_code: str, product_code: str):
    try:
        client = get_atlas_client()
        raw = client.futures(exchange_code=exchange_code, active_only=False)

        df = pd.DataFrame(raw)

        if df.empty:
            return pd.DataFrame()

        if "product_code" in df.columns:
            df = df[df["product_code"].astype(str) == str(product_code)]

        return df

    except Exception as e:
        st.warning(f"Atlas futures contracts could not be loaded: {e}")
        return pd.DataFrame()

def _contract_sort_key(df: pd.DataFrame, as_of: date | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "expiry_date" in out.columns:
        out["_expiry_dt"] = pd.to_datetime(out["expiry_date"], errors="coerce")
        if as_of is not None:
            asof_ts = pd.Timestamp(as_of)
            future = out[out["_expiry_dt"].isna() | (out["_expiry_dt"] >= asof_ts)]
            if not future.empty:
                out = future
        out = out.sort_values(["_expiry_dt", "canonical_symbol"], na_position="last")
    elif "canonical_symbol" in out.columns:
        out = out.sort_values("canonical_symbol")
    return out


def contract_label(row: pd.Series) -> str:
    symbol = str(row.get("canonical_symbol", ""))
    expiry = row.get("expiry_date")
    if pd.notna(expiry):
        try:
            expiry_s = pd.to_datetime(expiry).strftime("%Y-%m-%d")
            return f"{symbol}  ·  exp {expiry_s}"
        except Exception:
            pass
    return symbol


def _date_str(value: date | datetime | str) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _end_of_day_str(value: date | datetime | str) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d") + " 23:59:59"


def _normalise_time_index(df: pd.DataFrame, preferred_cols: Iterable[str] = ("time", "snapshot_time", "date")) -> pd.DataFrame:
    """Return a dataframe with a clean daily `date` column."""
    if df.empty:
        return df

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()

    time_col = None
    for col in preferred_cols:
        if col in out.columns:
            time_col = col
            break

    if time_col is None:
        # Last-resort: first datetime-like column.
        for col in out.columns:
            s = pd.to_datetime(out[col], errors="coerce")
            if s.notna().sum() >= max(1, len(out) // 2):
                time_col = col
                break

    if time_col is None:
        return pd.DataFrame()

    dt_index = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    out = out.loc[dt_index.notna()].copy()
    out["date"] = dt_index.loc[dt_index.notna()].dt.tz_convert(None).dt.normalize()
    return out


@st.cache_data(ttl=900, show_spinner=False)
def load_atlas_futures_history(canonical_symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily bars + settlements for a futures contract.
    Price uses settlement first, then falls back to close.
    Volume comes from bars.
    """
    client = get_atlas_client()
    start_s = _date_str(start)
    end_s = _end_of_day_str(end)

    bars = to_df(client.bars(canonical_symbol, timeframe="1d", start=start_s, end=end_s))
    settlements = to_df(client.settlements(canonical_symbol, start=start_s, end=end_s))

    bars_n = _normalise_time_index(bars)
    set_n = _normalise_time_index(settlements)

    pieces: list[pd.DataFrame] = []

    if not bars_n.empty:
        keep_cols = ["date"] + [c for c in ["open", "high", "low", "close", "volume", "vwap"] if c in bars_n.columns]
        b = bars_n[keep_cols].copy()
        for c in b.columns:
            if c != "date":
                b[c] = pd.to_numeric(b[c], errors="coerce")
        b = b.sort_values("date").groupby("date", as_index=False).last()
        pieces.append(b)

    if not set_n.empty:
        settlement_col = None
        for c in ["settlement", "price", "mark_price", "close"]:
            if c in set_n.columns:
                settlement_col = c
                break
        if settlement_col:
            s = set_n[["date", settlement_col]].copy()
            s[settlement_col] = pd.to_numeric(s[settlement_col], errors="coerce")
            s = s.rename(columns={settlement_col: "settlement"})
            s = s.sort_values("date").groupby("date", as_index=False).last()
            pieces.append(s)

    if not pieces:
        return pd.DataFrame(columns=["date", "price", "volume", "price_source"])

    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on="date", how="outer")

    if "settlement" in out.columns and "close" in out.columns:
        out["price"] = out["settlement"].combine_first(out["close"])
        out["price_source"] = np.where(out["settlement"].notna(), "settlement", "close")
    elif "settlement" in out.columns:
        out["price"] = out["settlement"]
        out["price_source"] = "settlement"
    elif "close" in out.columns:
        out["price"] = out["close"]
        out["price_source"] = "close"
    else:
        out["price"] = np.nan
        out["price_source"] = "unavailable"

    if "volume" not in out.columns:
        out["volume"] = np.nan

    return out.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers for replacing dummy metric numbers in dashboard.py
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def get_latest_atlas_price_metric(commodity: str, exchange_label: str, lookback_days: int = 45) -> dict:
    """
    Return latest price and 1-observation % change for the top metric card.
    This is intentionally small and cached so the metric card can load quickly.
    """
    spec = resolve_product_spec(commodity, exchange_label)
    if spec is None:
        return {"available": False, "message": f"Atlas mapping unavailable currently for {commodity} / {exchange_label}."}
    if not spec.data_available:
        return {"available": False, "message": f"{spec.exchange_code} / {spec.provider_exchange} data is unavailable currently."}

    try:
        contracts = load_atlas_futures_contracts(spec.exchange_code, spec.product_code)
        contracts = _contract_sort_key(contracts, date.today())
        if contracts.empty or "canonical_symbol" not in contracts.columns:
            return {"available": False, "message": f"No active Atlas futures contract found for {spec.product_desc}."}

        symbol = str(contracts.iloc[0]["canonical_symbol"])
        end = date.today()
        start = end - timedelta(days=lookback_days)
        hist = load_atlas_futures_history(symbol, _date_str(start), _date_str(end))
        price = hist["price"].dropna() if "price" in hist.columns else pd.Series(dtype=float)
        if price.empty:
            return {"available": False, "message": f"No Atlas price history found for {symbol}."}

        latest = float(price.iloc[-1])
        prev = float(price.iloc[-2]) if len(price) >= 2 else np.nan
        chg_pct = (latest / prev - 1.0) * 100.0 if prev and not np.isnan(prev) else np.nan
        latest_date = hist.loc[hist["price"].notna(), "date"].iloc[-1]

        return {
            "available": True,
            "latest_px": latest,
            "chg_pct": chg_pct,
            "symbol": symbol,
            "date": pd.to_datetime(latest_date).strftime("%Y-%m-%d"),
            "currency_unit": spec.currency_unit,
            "message": "",
        }
    except Exception as exc:
        return {"available": False, "message": f"Atlas market price unavailable currently: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

_LIGHT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f6f8fa",
    font=dict(family="IBM Plex Mono", size=11),
)
_GRID = dict(gridcolor="#d0d7de")


def make_futures_price_volume_chart(
    df: pd.DataFrame,
    *,
    title: str,
    price_unit: str,
) -> go.Figure:
    """Selected futures price line + volume bar in one chart."""
    plot_df = df.copy()
    plot_df = plot_df[plot_df["price"].notna() | plot_df["volume"].notna()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if "volume" in plot_df.columns and plot_df["volume"].notna().any():
        fig.add_trace(
            go.Bar(
                x=plot_df["date"],
                y=plot_df["volume"],
                name="Volume",
                opacity=0.35,
                hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

    if "price" in plot_df.columns and plot_df["price"].notna().any():
        source = plot_df.get("price_source", pd.Series([""] * len(plot_df)))
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["price"],
                name="Futures price",
                mode="lines",
                customdata=source,
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    f"Price: %{{y:,.2f}} {price_unit}<br>"
                    "Source: %{customdata}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    fig.update_layout(
        **_LIGHT,
        height=340,
        margin=dict(l=0, r=0, t=35, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        title=dict(text=title, font=dict(size=11, family="IBM Plex Mono"), x=0),
    )
    fig.update_yaxes(title_text=price_unit, **_GRID, secondary_y=False)
    fig.update_yaxes(title_text="Volume", showgrid=False, secondary_y=True)
    fig.update_xaxes(**_GRID)
    return fig


def _normalise_to_100(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(float)
    if s.empty:
        return s
    first = s.iloc[0]
    if first == 0 or np.isnan(first):
        return pd.Series(dtype=float)
    return s / first * 100.0


def make_normalized_twin_price_chart(price_index_map: dict[str, pd.Series], highlight_label: str | None = None) -> go.Figure:
    """Twin-market normalized price index chart; first valid observation = 100."""
    fig = go.Figure()

    for label, s in price_index_map.items():
        idx = _normalise_to_100(s)
        if idx.empty:
            continue
        is_main = label == highlight_label
        fig.add_trace(
            go.Scatter(
                x=idx.index,
                y=idx.values,
                name=label,
                mode="lines",
                line=dict(width=2.4 if is_main else 1.3, dash="solid" if is_main else "dot"),
                opacity=1.0 if is_main else 0.75,
                hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>Index: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.add_hline(y=100, line_width=1, line_dash="dash")
    fig.update_layout(
        **_LIGHT,
        height=340,
        margin=dict(l=0, r=0, t=35, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        yaxis_title="Normalized price index (start = 100)",
        title=dict(text="Twin-market normalized prices", font=dict(size=11, family="IBM Plex Mono"), x=0),
    )
    fig.update_yaxes(**_GRID)
    fig.update_xaxes(**_GRID)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Date/range filters
# ─────────────────────────────────────────────────────────────────────────────

RANGE_OPTIONS = ["1M", "3M", "6M", "YTD", "1Y", "2Y", "Custom"]


def _resolve_range(option: str, custom_range) -> tuple[date, date]:
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
    if isinstance(custom_range, list) and len(custom_range) == 2:
        return custom_range[0], custom_range[1]
    return today - timedelta(days=365), today


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _show_unavailable(message: str) -> None:
    st.info(f"{message} Unavailable currently.")


def _status_row(label: str, spec: AtlasProductSpec | None, status: str, symbol: str | None = None) -> dict:
    if spec is None:
        return {
            "Market": label,
            "Atlas Exchange": "-",
            "Provider Exchange": "-",
            "Product": "-",
            "Symbol": symbol or "-",
            "Status": status,
        }
    return {
        "Market": label,
        "Atlas Exchange": spec.exchange_code,
        "Provider Exchange": spec.provider_exchange,
        "Product": spec.product_desc,
        "Symbol": symbol or "-",
        "Status": status,
    }


def _load_front_contract_history(
    spec: AtlasProductSpec,
    start_date: date,
    end_date: date,
) -> tuple[str | None, pd.DataFrame, str]:
    if not spec.data_available:
        return None, pd.DataFrame(), f"{spec.exchange_code} / {spec.provider_exchange} data unavailable currently"
    contracts = load_atlas_futures_contracts(spec.exchange_code, spec.product_code)
    contracts = _contract_sort_key(contracts, end_date)
    if contracts.empty or "canonical_symbol" not in contracts.columns:
        return None, pd.DataFrame(), f"No active Atlas contract found for {spec.product_desc}"
    symbol = str(contracts.iloc[0]["canonical_symbol"])
    hist = load_atlas_futures_history(symbol, _date_str(start_date), _date_str(end_date))
    if hist.empty or not hist["price"].notna().any():
        return symbol, hist, f"No Atlas price history found for {symbol}"
    return symbol, hist, "OK"

def get_atlas_range_price_metric(
    commodity: str,
    exchange_label: str,
    start_date: date,
    end_date: date,
) -> dict:
    """
    Metric card value based on selected chart range.

    latest_px = last valid price in selected range
    chg_pct   = % change from first valid price in selected range to latest price
    """
    spec = resolve_product_spec(commodity, exchange_label)

    if spec is None:
        return {
            "available": False,
            "latest_px": np.nan,
            "chg_pct": np.nan,
            "message": f"Atlas mapping unavailable currently for {commodity} / {exchange_label}.",
        }

    if not spec.data_available:
        return {
            "available": False,
            "latest_px": np.nan,
            "chg_pct": np.nan,
            "message": f"{spec.exchange_code} / {spec.provider_exchange} data unavailable currently.",
        }

    try:
        # If you already added the continuous front-month function, use it.
        if "_load_continuous_front_month_history" in globals():
            symbol, hist, status = _load_continuous_front_month_history(
                spec, start_date, end_date
            )
        else:
            symbol, hist, status = _load_front_contract_history(
                spec, start_date, end_date
            )

        if status != "OK" or hist.empty or "price" not in hist.columns:
            return {
                "available": False,
                "latest_px": np.nan,
                "chg_pct": np.nan,
                "message": status,
            }

        px = hist[["date", "price"]].dropna().sort_values("date")

        if len(px) < 2:
            return {
                "available": False,
                "latest_px": np.nan,
                "chg_pct": np.nan,
                "message": "Not enough Atlas price history for selected range.",
            }

        first_px = float(px["price"].iloc[0])
        latest_px = float(px["price"].iloc[-1])

        chg_pct = (
            (latest_px / first_px - 1.0) * 100.0
            if first_px != 0 and not np.isnan(first_px)
            else np.nan
        )

        return {
            "available": True,
            "latest_px": latest_px,
            "first_px": first_px,
            "chg_pct": chg_pct,
            "symbol": symbol,
            "start_date": pd.to_datetime(px["date"].iloc[0]).strftime("%Y-%m-%d"),
            "end_date": pd.to_datetime(px["date"].iloc[-1]).strftime("%Y-%m-%d"),
            "message": "",
        }

    except Exception as exc:
        return {
            "available": False,
            "latest_px": np.nan,
            "chg_pct": np.nan,
            "message": f"Atlas range metric unavailable currently: {exc}",
        }

def _load_continuous_front_month_history(
    spec: AtlasProductSpec,
    start_date: date,
    end_date: date,
    max_contracts: int = 36,
) -> tuple[str | None, pd.DataFrame, str]:
    """
    Build a synthetic continuous front-month futures series.

    For each trading date, this chooses the priced contract with the nearest
    expiry date that has not expired on that date. This removes the manual
    futures-contract dropdown and gives one front-month chart.
    """
    if not spec.data_available:
        return None, pd.DataFrame(), f"{spec.exchange_code} / {spec.provider_exchange} data unavailable currently"

    contracts = load_atlas_futures_contracts_all(
        spec.exchange_code,
        spec.product_code
    )

    if contracts.empty:
        return None, pd.DataFrame(), "Atlas connection failed or no futures contracts found."

    # Fallback if the client/source only returns active contracts.
    if contracts.empty:
        contracts = load_atlas_futures_contracts(spec.exchange_code, spec.product_code)

    if contracts.empty or "canonical_symbol" not in contracts.columns:
        return None, pd.DataFrame(), f"No Atlas futures contract found for {spec.product_desc}"

    contracts = contracts.copy()
    if "expiry_date" in contracts.columns:
        contracts["_expiry_dt"] = pd.to_datetime(contracts["expiry_date"], errors="coerce")
    else:
        contracts["_expiry_dt"] = pd.NaT

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Keep contracts that could have been the front month somewhere in the range.
    if contracts["_expiry_dt"].notna().any():
        contracts = contracts[
            contracts["_expiry_dt"].isna()
            | (contracts["_expiry_dt"] >= start_ts)
        ].copy()
        contracts = contracts.sort_values(["_expiry_dt", "canonical_symbol"], na_position="last")
    else:
        contracts = contracts.sort_values("canonical_symbol")

    # Avoid too many DB calls. 36 monthly contracts covers about 3 years.
    contracts = contracts.head(max_contracts)

    histories: list[pd.DataFrame] = []
    for _, row in contracts.iterrows():
        symbol = str(row.get("canonical_symbol", ""))
        if not symbol:
            continue

        expiry_dt = pd.to_datetime(row.get("_expiry_dt"), errors="coerce")
        expiry_date = expiry_dt.date() if pd.notna(expiry_dt) else None

        try:
            hist = _load_history_with_feather_cache(spec, symbol, start_date, end_date)
        except Exception:
            continue

        if hist.empty or "price" not in hist.columns or not hist["price"].notna().any():
            continue

        h = hist.copy()
        h["canonical_symbol"] = symbol
        h["contract_expiry"] = pd.Timestamp(expiry_date) if expiry_date else pd.NaT
        histories.append(h)

    if not histories:
        return "Continuous front month", pd.DataFrame(), f"No Atlas price history found for {spec.product_desc}"

    all_hist = pd.concat(histories, ignore_index=True)
    all_hist["date"] = pd.to_datetime(all_hist["date"]).dt.normalize()
    all_hist = all_hist[(all_hist["date"] >= start_ts) & (all_hist["date"] <= end_ts)]
    all_hist = all_hist[all_hist["price"].notna()].copy()

    if all_hist.empty:
        return "Continuous front month", pd.DataFrame(), f"No Atlas price history found for {spec.product_desc}"

    # A front-month contract must not have expired yet on that price date.
    if all_hist["contract_expiry"].notna().any():
        valid = all_hist["contract_expiry"].isna() | (all_hist["contract_expiry"] >= all_hist["date"])
        all_hist = all_hist[valid].copy()

    if all_hist.empty:
        return "Continuous front month", pd.DataFrame(), f"No unexpired front-month prices found for {spec.product_desc}"

    # For each date, keep the contract with the nearest valid expiry.
    all_hist = all_hist.sort_values(["date", "contract_expiry", "canonical_symbol"], na_position="last")
    front = all_hist.groupby("date", as_index=False).first()

    # Make hover/source show which contract supplied each point.
    if "price_source" in front.columns:
        front["price_source"] = front["price_source"].astype(str) + " · " + front["canonical_symbol"].astype(str)
    else:
        front["price_source"] = front["canonical_symbol"].astype(str)

    # Keep a small roll history for display under the chart.
    roll_cols = ["date", "canonical_symbol", "contract_expiry"]
    roll_history = front[roll_cols].copy()
    roll_history = roll_history[roll_history["canonical_symbol"].ne(roll_history["canonical_symbol"].shift())]
    roll_history["date"] = pd.to_datetime(roll_history["date"]).dt.strftime("%Y-%m-%d")
    roll_history["contract_expiry"] = pd.to_datetime(roll_history["contract_expiry"], errors="coerce").dt.strftime("%Y-%m-%d")
    front.attrs["roll_history"] = roll_history

    return "Continuous front month", front.sort_values("date").reset_index(drop=True), "OK"


def _series_for_normalized_chart(hist: pd.DataFrame, spec: AtlasProductSpec, label: str, fx_history: dict[str, pd.Series]) -> pd.Series:
    if hist.empty or "price" not in hist.columns:
        return pd.Series(dtype=float)

    local = hist[["date", "price"]].dropna().copy()
    local = local.set_index(pd.to_datetime(local["date"]))["price"].astype(float)

    fx_pair = FX_PAIR_BY_UNIT.get(spec.currency_unit, "USD/USD")
    if fx_pair == "USD/USD":
        fx = pd.Series(1.0, index=local.index, name=fx_pair)
    else:
        fx_raw = fx_history.get(fx_pair, pd.Series(dtype=float)).dropna()
        if fx_raw.empty:
            return pd.Series(dtype=float)
        fx_raw.index = pd.to_datetime(fx_raw.index).tz_localize(None).normalize()
        fx = fx_raw.reindex(local.index, method="ffill")

    converted = to_usd_per_mt(local, spec.currency_unit, spec.commodity, fx)
    converted.name = label
    return converted.dropna()


def _get_contract_options(spec: AtlasProductSpec, as_of: date) -> pd.DataFrame:
    if not spec.data_available:
        return pd.DataFrame()
    try:
        contracts = load_atlas_futures_contracts(spec.exchange_code, spec.product_code)
        return _contract_sort_key(contracts, as_of)
    except Exception:
        return pd.DataFrame()

def _load_history_with_feather_cache(
    spec: "AtlasProductSpec",
    canonical_symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Drop-in wrapper around load_atlas_futures_history that persists to feather.
    Use this instead of calling load_atlas_futures_history directly.
    """
    return load_cached_history(
        exchange_code=spec.exchange_code,
        product_code=spec.product_code,
        canonical_symbol=canonical_symbol,
        start=start_date,
        end=end_date,
    )

def render_atlas_market_section(
    *,
    commodity: str,
    exchange: str,
    currency: str | None = None,
    fx_pair: str | None = None,
    twin_labels: list[str] | tuple[str, ...] | set[str] | None = None,
    lookback_days: int = 365,
    section_title: str = "02 · FX Impact Analysis",
) -> None:
    """
    Drop-in renderer for the dashboard's FX/Market section.

    Market Prices tab:
        continuous front-month futures price line + volume bar in one chart.
    Normalized Prices tab:
        continuous front-month futures + available twin markets, converted and indexed to 100.
    Unavailable data:
        shown explicitly as "unavailable currently".
    """
    st.markdown(f'<div class="section-header">{section_title}</div>', unsafe_allow_html=True)

    spec = resolve_product_spec(commodity, exchange)
    if spec is None:
        _show_unavailable(f"Atlas mapping for {commodity} / {exchange} is not configured.")
        return

    active_currency = currency or spec.currency_unit
    active_fx_pair = fx_pair or FX_PAIR_BY_UNIT.get(spec.currency_unit, "USD/USD")
    twins = list(twin_labels or [])

    # Chart filters. These controls are outside the tabs so both Market Prices
    # and Normalized Prices use the same date window.
    f1, f2 = st.columns([1, 2])
    with f1:
        range_option = st.selectbox("Chart Range", RANGE_OPTIONS, index=2, key=f"atlas_range_{commodity}_{exchange}")
    with f2:
        custom_range = st.date_input(
            "Custom Dates",
            value=(date.today() - timedelta(days=lookback_days), date.today()),
            disabled=range_option != "Custom",
            key=f"atlas_custom_dates_{commodity}_{exchange}",
        )
    start_date, end_date = _resolve_range(range_option, custom_range)

    if start_date > end_date:
        st.warning("Start date must be before end date.")
        return

    # Continuous front-month mode: no manual futures-contract dropdown.
    selected_symbol = "Continuous front month"

    tab_market, tab_norm, tab_overlay, tab_monitor = st.tabs([
        "01 · Market Prices",
        "📊 Normalized Prices",
        # "🔀 Attribution",
        "📈 Price vs FX Overlay",
        "💱 FX Monitor",
    ])

    main_hist = pd.DataFrame()
    main_status = ""

    if not spec.data_available:
        main_status = f"{spec.exchange_code} / {spec.provider_exchange} data unavailable currently"
    else:
        try:
            selected_symbol, main_hist, main_status = _load_continuous_front_month_history(
                spec, start_date, end_date
            )
            selected_symbol = selected_symbol or "Continuous front month"
        except Exception as exc:
            main_status = f"Atlas continuous front-month price unavailable currently: {exc}"

    with tab_market:
        st.caption("Continuous front-month futures price with volume. Twin markets are not shown in this tab.")
        if main_status != "OK":
            _show_unavailable(main_status)
        else:
            fig = make_futures_price_volume_chart(
                main_hist,
                title=f"{commodity} continuous front-month futures price and volume",
                price_unit=active_currency,
            )
            st.plotly_chart(fig, width="stretch")
            last_source = main_hist["price_source"].dropna().iloc[-1] if "price_source" in main_hist and main_hist["price_source"].notna().any() else "price"
            st.caption(f"Source: Atlas · Price uses {last_source}; volume uses daily bars · Range: {start_date} to {end_date}")

            roll_history = main_hist.attrs.get("roll_history")
            if isinstance(roll_history, pd.DataFrame) and not roll_history.empty:
                with st.expander("View front-month roll history", expanded=False):
                    st.dataframe(roll_history, hide_index=True, width="stretch")

    with tab_norm:
        st.caption(
            "Continuous front-month futures and available twin markets converted where possible, "
            "then indexed to 100 at the first valid observation for the selected range."
        )

        # Build FX history only for currency conversions we really need.
        specs_for_fx: list[AtlasProductSpec] = [spec]
        for label in twins:
            twin_spec = resolve_twin_spec(label)
            if twin_spec is not None:
                specs_for_fx.append(twin_spec)
        fx_pairs = tuple(
            pair for pair in dict.fromkeys(FX_PAIR_BY_UNIT.get(s.currency_unit, "USD/USD") for s in specs_for_fx)
            if pair != "USD/USD"
        )
        fx_history = fetch_fx_history(fx_pairs, lookback_days=max(lookback_days, (end_date - start_date).days + 10)) if fx_pairs else {}

        price_map: dict[str, pd.Series] = {}
        status_rows: list[dict] = []

        main_label = f"{commodity} ({exchange})"
        if main_status == "OK":
            price_map[main_label] = _series_for_normalized_chart(main_hist, spec, main_label, fx_history)
            status_rows.append(_status_row(main_label, spec, "OK", selected_symbol))
        else:
            status_rows.append(_status_row(main_label, spec, main_status, selected_symbol))

        for label in twins:
            twin_spec = resolve_twin_spec(label)
            if twin_spec is None:
                status_rows.append(_status_row(label, None, "Atlas mapping unavailable currently"))
                continue
            symbol, hist, status = _load_continuous_front_month_history(twin_spec, start_date, end_date)
            status_rows.append(_status_row(label, twin_spec, status, symbol))
            if status == "OK":
                price_map[label] = _series_for_normalized_chart(hist, twin_spec, label, fx_history)

        if not any(not s.dropna().empty for s in price_map.values()):
            _show_unavailable("No selected or twin market price history is available from Atlas for this range.")
        else:
            fig_norm = make_normalized_twin_price_chart(price_map, highlight_label=main_label)
            st.plotly_chart(fig_norm, width="stretch")

        # Always show data status so unavailable exchanges are visible.
        status_df = pd.DataFrame(status_rows)
        if not status_df.empty:
            with st.expander("Atlas data availability / status", expanded=False):
                st.dataframe(status_df, hide_index=True, width="stretch")

    # with tab_attr:
    #     if main_status != "OK":
    #         _show_unavailable(main_status)
    #     else:
    #         st.caption("Daily price move split into local price move and FX translation using the selected futures series.")
    #         try:
    #             pair = active_fx_pair
    #             if pair == "USD/USD":
    #                 active_fx = pd.Series(1.0, index=pd.to_datetime(main_hist["date"]), name=pair)
    #             else:
    #                 hist = fetch_fx_history((pair,), lookback_days=max(lookback_days, (end_date - start_date).days + 10))
    #                 active_fx = hist.get(pair, pd.Series(dtype=float))
    #             local_prices = main_hist.set_index(pd.to_datetime(main_hist["date"]))["price"].dropna()
    #             active_fx.index = pd.to_datetime(active_fx.index).tz_localize(None).normalize()
    #             active_fx = active_fx.reindex(local_prices.index, method="ffill")

    #             unit_conv = 1.0
    #             if active_currency == "USc/bu":
    #                 # to_usd_per_mt handles the full conversion, but get_fx_attribution
    #                 # expects a simple multiplier from local price into USD before FX.
    #                 # Use a commodity-specific approximation consistent with fx_live_monitor.py.
    #                 bushel_mt = {"Corn": 0.025401, "Wheat": 0.027216, "Soybeans": 0.027216}.get(commodity, 0.027216)
    #                 unit_conv = 1.0 / 100.0 / bushel_mt
    #             elif active_currency == "USc/lb":
    #                 unit_conv = 22.0462

    #             attr_df = get_fx_attribution(
    #                 local_prices=local_prices,
    #                 fx_series=active_fx,
    #                 local_ccy=active_currency,
    #                 target_ccy="USD",
    #                 unit_conv=unit_conv,
    #             )
    #             if attr_df.empty:
    #                 _show_unavailable("FX attribution cannot be calculated for the selected range.")
    #             else:
    #                 window = st.slider("Rolling window (trading days)", 20, 252, min(60, max(20, len(attr_df))), step=5, key=f"atlas_attr_{commodity}_{exchange}")
    #                 st.plotly_chart(make_fx_attribution_chart(attr_df, window=window, commodity_ccy="USD"), width="stretch")
    #         except Exception as exc:
    #             _show_unavailable(f"FX attribution unavailable currently: {exc}")

    with tab_overlay:
        if main_status != "OK":
            _show_unavailable(main_status)
        else:
            st.caption("Selected futures price on the left axis and FX rate on the right axis.")
            try:
                pair = active_fx_pair
                if pair == "USD/USD":
                    active_fx = pd.Series(1.0, index=pd.to_datetime(main_hist["date"]), name=pair)
                else:
                    hist = fetch_fx_history((pair,), lookback_days=max(lookback_days, (end_date - start_date).days + 10))
                    active_fx = hist.get(pair, pd.Series(dtype=float))
                active_fx.index = pd.to_datetime(active_fx.index).tz_localize(None).normalize()
                local_prices = main_hist.set_index(pd.to_datetime(main_hist["date"]))["price"].dropna()
                st.plotly_chart(
                    make_fx_overlay_chart(
                        commodity_prices=local_prices,
                        fx_series=active_fx,
                        commodity_label=f"{commodity} ({exchange})",
                        fx_pair=pair,
                        commodity_ccy=active_currency,
                    ),
                    width="stretch",
                )
            except Exception as exc:
                _show_unavailable(f"Price vs FX overlay unavailable currently: {exc}")

    with tab_monitor:
        st.caption("Live FX snapshot; unavailable Atlas futures markets do not affect this FX table.")
        monitor_pairs = tuple(
            pair for pair in dict.fromkeys([
                active_fx_pair,
                "USD/MYR", "USD/CNY", "USD/EUR", "USD/BRL", "USD/GBP", "USD/CAD",
            ])
            if pair != "USD/USD"
        )
        try:
            snap_df = fetch_fx_live_monitor(monitor_pairs)
            snap_df["Active"] = snap_df["Pair"].eq(active_fx_pair).map({True: "●", False: ""})
            snap_df = snap_df[["Pair", "Rate", "1D %", "1W %", "1M %", "YTD %", "Active", "Live Trend"]]
            st.dataframe(
                snap_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Rate": st.column_config.NumberColumn("Rate", format="%.4f"),
                    "1D %": st.column_config.NumberColumn("1D %", format="%+.2f%%"),
                    "1W %": st.column_config.NumberColumn("1W %", format="%+.2f%%"),
                    "1M %": st.column_config.NumberColumn("1M %", format="%+.2f%%"),
                    "YTD %": st.column_config.NumberColumn("YTD %", format="%+.2f%%"),
                    "Live Trend": st.column_config.LineChartColumn("Live Trend", width="medium"),
                },
            )
        except Exception as exc:
            _show_unavailable(f"FX monitor unavailable currently: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard.py integration snippet
# ─────────────────────────────────────────────────────────────────────────────
"""
1) Save this file as:
       utils/atlas_market_data.py

2) In dashboard.py, replace this import:
       from utils.fx_live_monitor import render_fx_section, fetch_fx_history, get_fx_snapshot

   with:
       from utils.fx_live_monitor import fetch_fx_history, get_fx_snapshot
       from utils.atlas_market_data import render_atlas_market_section, get_latest_atlas_price_metric

3) Replace the dummy price block:
       dates_d = make_dates(252)
       prices  = make_price_series(base_px, vol=0.012)
       latest_px = prices[-1]
       prev_px   = prices[-2]
       chg_pct   = (latest_px - prev_px) / prev_px * 100

   with:
       _atlas_metric = get_latest_atlas_price_metric(commodity, exchange)
       latest_px = _atlas_metric.get("latest_px", np.nan)
       chg_pct = _atlas_metric.get("chg_pct", np.nan)

4) Replace your first metric card call with a safe display:
       metric_card(
           _metric_cols[0],
           f"{commodity} ({exchange})",
           "N/A" if np.isnan(latest_px) else f"{latest_px:,.1f}",
           None if np.isnan(chg_pct) else f"{chg_pct:+.2f}%",
           delta_pos=(False if np.isnan(chg_pct) else chg_pct >= 0),
       )

5) Replace the old render_fx_section(...) block with:
       render_atlas_market_section(
           commodity=commodity,
           exchange=exchange,
           currency=currency,
           fx_pair=fx_pair,
           twin_labels=all_twins,
           lookback_days=365,
       )
"""