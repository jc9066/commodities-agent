"""
utils/psd_cache.py
──────────────────────────────────────────────────────────────
Auto-refreshing feather cache for USDA FAS PSD data.

Called once at dashboard startup via:

    from utils.psd_cache import ensure_psd_cache
    ensure_psd_cache()          # call near the top of dashboard.py

What it does
------------
- Checks which feather files are missing or older than CACHE_TTL_HOURS.
- Fetches only those tables from the USDA API (in a background thread
  so Streamlit doesn't block on the first load).
- Subsequent runs read straight from disk — no API calls.

Feather files written to data/ (relative to project root)
----------------------------------------------------------
  psd_world_sd.feather
  psd_country_sd.feather
  psd_aep_raw.feather
  psd_lookup_countries.feather
  psd_lookup_commodities.feather
  psd_lookup_units.feather
  psd_cache_meta.json
"""

from __future__ import annotations

import datetime
import json
import sys
import threading
from pathlib import Path

import pandas as pd
import requests

# ── paths ─────────────────────────────────────────────────────
_ROOT      = Path(__file__).parent.parent
_DATA_DIR  = _ROOT / "data"
_META_FILE = _DATA_DIR / "psd_cache_meta.json"

CACHE_TTL_HOURS = 6

# ── API ───────────────────────────────────────────────────────
_BASE_URL = "https://api.fas.usda.gov"
_API_KEY  = "5D9acNtwcN86Sznb3nDQrcBAvgLTxl9QNunu3uLz"
_HEADERS  = {"accept": "application/json", "X-API-Key": _API_KEY}

COMMODITY_CODES: dict[str, str] = {
    "Crude Palm Oil":  "4243000",
    "Palm Olein":      "4243000",
    "Bean Oil":        "4232000",
    "Soybeans":        "2222000",
    "Canola/Rapeseed": "2226000",
    "Rapeseed Oil":    "4239100",
    "Wheat":           "0410000",
    "Cotton":          "2631000",
    "Soymeal":         "0813100",
    "Corn":            "0440000",
    "Coffee":          "0711100",
}

N_YEARS = 5  # rolling window for world/country S&D


# ── meta helpers ──────────────────────────────────────────────
def _read_meta() -> dict:
    if not _META_FILE.exists():
        return {}
    try:
        return json.loads(_META_FILE.read_text())
    except Exception:
        return {}


def _write_meta(meta: dict) -> None:
    _META_FILE.write_text(json.dumps(meta, indent=2))


def _is_fresh(key: str, meta: dict | None = None) -> bool:
    if meta is None:
        meta = _read_meta()
    ts = meta.get(key)
    if not ts:
        return False
    try:
        age = datetime.datetime.utcnow() - datetime.datetime.fromisoformat(ts)
        return age < datetime.timedelta(hours=CACHE_TTL_HOURS)
    except Exception:
        return False


def _feather_exists(name: str) -> bool:
    return (_DATA_DIR / f"{name}.feather").exists()


# ── feather I/O ───────────────────────────────────────────────
def _save(df: pd.DataFrame, name: str, meta_key: str) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    df_out = df.copy()
    for col in df_out.select_dtypes("Int64").columns:
        df_out[col] = df_out[col].astype("float64")
    df_out.reset_index(drop=True).to_feather(_DATA_DIR / f"{name}.feather")
    meta = _read_meta()
    meta[meta_key] = datetime.datetime.utcnow().isoformat()
    _write_meta(meta)
    print(f"  [psd_cache] wrote {name}.feather ({len(df_out):,} rows)", flush=True)


def read_feather(name: str) -> pd.DataFrame | None:
    """Public: read a feather file, return None if missing or unreadable."""
    path = _DATA_DIR / f"{name}.feather"
    if not path.exists():
        return None
    try:
        return pd.read_feather(path)
    except Exception:
        return None


# ── API fetchers ──────────────────────────────────────────────
def _get(endpoint: str) -> list | dict:
    r = requests.get(_BASE_URL + endpoint, headers=_HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def _fetch_lookups() -> None:
    print("  [psd_cache] fetching lookup tables…", flush=True)
    country_df = pd.DataFrame(_get("/api/psd/countries"))
    comod_df   = pd.DataFrame(_get("/api/psd/commodities"))
    unit_df    = pd.DataFrame(_get("/api/psd/unitsOfMeasure"))
    _save(country_df, "psd_lookup_countries",   "lookups")
    _save(comod_df,   "psd_lookup_commodities", "lookups")
    _save(unit_df,    "psd_lookup_units",        "lookups")


def _fetch_world_sd() -> None:
    print("  [psd_cache] fetching world S&D…", flush=True)
    today     = datetime.date.today()
    prod_yr   = today.year - 1
    mkt_years = range(prod_yr - (N_YEARS - 1), prod_yr + 1)
    codes     = list(dict.fromkeys(COMMODITY_CODES.values()))
    records: list[pd.DataFrame] = []
    for code in codes:
        for yr in mkt_years:
            try:
                data = _get(f"/api/psd/commodity/{code}/world/year/{yr}")
                if data:
                    records.append(pd.DataFrame(data))
            except Exception as exc:
                print(f"    skip world {code}/{yr}: {exc}", file=sys.stderr, flush=True)
    if records:
        _save(pd.concat(records, ignore_index=True), "psd_world_sd", "world_sd")


def _fetch_country_sd() -> None:
    print("  [psd_cache] fetching country S&D…", flush=True)
    today     = datetime.date.today()
    prod_yr   = today.year - 1
    mkt_years = range(prod_yr - (N_YEARS - 1), prod_yr + 1)
    codes     = list(dict.fromkeys(COMMODITY_CODES.values()))
    records: list[pd.DataFrame] = []
    for code in codes:
        for yr in mkt_years:
            try:
                data = _get(f"/api/psd/commodity/{code}/country/all/year/{yr}")
                if data:
                    records.append(pd.DataFrame(data))
            except Exception as exc:
                print(f"    skip country {code}/{yr}: {exc}", file=sys.stderr, flush=True)
    if records:
        _save(pd.concat(records, ignore_index=True), "psd_country_sd", "country_sd")


def _fetch_aep_raw() -> None:
    print("  [psd_cache] fetching AEP raw…", flush=True)
    today = datetime.date.today()
    cur   = today.year
    years = (cur - 2, cur - 1, cur)
    codes = list(dict.fromkeys(COMMODITY_CODES.values()))
    records: list[pd.DataFrame] = []
    for code in codes:
        for yr in years:
            try:
                data = _get(f"/api/psd/commodity/{code}/country/all/year/{yr}")
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                if data:
                    frame = pd.DataFrame(data)
                    frame["_commodity_code"] = code
                    records.append(frame)
            except Exception as exc:
                print(f"    skip aep {code}/{yr}: {exc}", file=sys.stderr, flush=True)
    if records:
        _save(pd.concat(records, ignore_index=True), "psd_aep_raw", "aep_raw")


# ── main entry point ──────────────────────────────────────────
_refresh_started = False
_refresh_lock    = threading.Lock()


def ensure_psd_cache() -> None:
    """
    Call once near the top of dashboard.py.

    Checks which tables are missing or stale and kicks off a background
    thread to refresh only those.  Returns immediately — the dashboard
    renders while data is being fetched in the background.

    On the next Streamlit rerun (user interaction or st.rerun()) the
    fresh feather files will be picked up by psd.py / psd_aep.py.
    """
    global _refresh_started
    with _refresh_lock:
        if _refresh_started:
            return
        _refresh_started = True

    meta = _read_meta()

    # decide what needs refreshing
    tasks: list[tuple[str, callable]] = []

    lookup_files = ["psd_lookup_countries", "psd_lookup_commodities", "psd_lookup_units"]
    if not _is_fresh("lookups", meta) or not all(_feather_exists(f) for f in lookup_files):
        tasks.append(("lookups", _fetch_lookups))

    if not _is_fresh("world_sd", meta) or not _feather_exists("psd_world_sd"):
        tasks.append(("world_sd", _fetch_world_sd))

    if not _is_fresh("country_sd", meta) or not _feather_exists("psd_country_sd"):
        tasks.append(("country_sd", _fetch_country_sd))

    if not _is_fresh("aep_raw", meta) or not _feather_exists("psd_aep_raw"):
        tasks.append(("aep_raw", _fetch_aep_raw))

    if not tasks:
        print("  [psd_cache] all tables fresh — no fetch needed", flush=True)
        return

    labels = [t[0] for t in tasks]
    print(f"  [psd_cache] stale/missing: {labels} — refreshing in background…", flush=True)

    def _run():
        for label, fn in tasks:
            try:
                fn()
            except Exception as exc:
                print(f"  [psd_cache] ERROR refreshing {label}: {exc}",
                      file=sys.stderr, flush=True)

    t = threading.Thread(target=_run, daemon=True, name="psd-cache-refresh")
    t.start()
