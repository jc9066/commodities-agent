"""
utils/atlas_cache.py
====================
Feather-backed cache for Atlas futures history.
Same pattern as utils/psd_cache.py / weather_daily.feather.

Cache layout:  data/atlas_cache/<exchange_code>_<product_code>.feather
"""

from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import streamlit as st

CACHE_DIR = Path("data/atlas_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# How many recent trailing days to always re-fetch from Atlas
# (handles late settlements arriving after market close)
REFRESH_RECENT_DAYS = 3


def _cache_path(exchange_code: str, product_code: str, canonical_symbol: str = "") -> Path:
    if canonical_symbol:
        # sanitise the symbol — dots and slashes break filenames
        safe = canonical_symbol.replace(".", "_").replace("/", "_")
        return CACHE_DIR / f"{safe}.feather"
    return CACHE_DIR / f"{exchange_code}_{product_code}.feather"


def load_cached_history(
    exchange_code: str,
    product_code: str,
    canonical_symbol: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    cache_file = _cache_path(exchange_code, product_code, canonical_symbol)
    cutoff = end - timedelta(days=REFRESH_RECENT_DAYS)

    cached: pd.DataFrame = pd.DataFrame()

    if cache_file.exists():
        try:
            cached = pd.read_feather(cache_file)
            cached["date"] = pd.to_datetime(cached["date"])
        except Exception:
            cached = pd.DataFrame()

    # Determine what date range still needs to be pulled from Atlas
    if not cached.empty:
        cached_max = cached["date"].max().date()
        fetch_from = max(cutoff, cached_max - timedelta(days=1))
    else:
        fetch_from = start  # cold start — fetch everything

    # Lazy import to avoid circular dependency
    from utils.atlas_market_data import load_atlas_futures_history

    fresh = load_atlas_futures_history(
        canonical_symbol=canonical_symbol,
        start=str(fetch_from),
        end=str(end),
    )

    if fresh.empty and cached.empty:
        return pd.DataFrame()

    # Merge: drop overlapping rows from cache, append fresh tail
    if not fresh.empty and not cached.empty:
        fresh["date"] = pd.to_datetime(fresh["date"])
        cached = cached[cached["date"] < fresh["date"].min()]
        merged = pd.concat([cached, fresh], ignore_index=True)
    elif not fresh.empty:
        merged = fresh.copy()
        merged["date"] = pd.to_datetime(merged["date"])
    else:
        merged = cached.copy()

    merged = merged.sort_values("date").drop_duplicates("date")

    # Trim to requested range
    merged = merged[
        (merged["date"] >= pd.Timestamp(start)) &
        (merged["date"] <= pd.Timestamp(end))
    ]

    # Persist updated cache
    try:
        merged.reset_index(drop=True).to_feather(cache_file)
    except Exception as e:
        st.warning(f"Atlas cache write failed for {exchange_code}/{product_code}: {e}")

    return merged