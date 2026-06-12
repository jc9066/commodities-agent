"""
Weather data utilities for the commodities dashboard.

This module replaces the duplicated notebook weather cells with one reusable
pipeline that can be imported by both Jupyter notebooks and Streamlit.

What it does
------------
1. Fetches WeatherDesk GWI station weather by crop / region / parameter.
2. Standardises growing-zone names and filters exact regions only.
3. Saves daily pulled data to a feather cache, so future runs fetch only
   missing/new dates.
4. Builds Plotly charts for commodity-specific weather risk views.
5. Keeps the old dashboard.py helper names available:
   make_cached_weather_fetcher, get_available_groups, get_available_zones,
   get_available_parameters, PARAM_LABEL, PRIMARY_PARAM, to_rainfall_wide.

Recommended Streamlit use
-------------------------
from utils.weather import render_weather_section

with _col_weather:
    render_weather_section(commodity, dark=True)

Recommended notebook use
------------------------
from weather import make_weather_figures

figs = make_weather_figures("Soybeans", force_refresh=False)
for name, fig in figs.items():
    fig.show()
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests


# =============================================================================
# API CONFIG
# =============================================================================

BASE_URL = "https://api.weatherdesk.xweather.com"
TABLE_ENDPOINT = "/d9293fc8-b396-4c3e-9166-2dee2012a6f2/services/gwi/v1/getdata"

REGION_MAP: dict[str, int] = {
    "Southeast Asia": 84,
    "Indonesia": 30,
    "Vietnam/Thailand": 22,
    "East Asia": 95,
    "United States": 35,
    "China": 85,
    "India": 15,
    "Brazil/Argentina": 86,
    "Brazil": 4,
    "Argentina": 13,
    "Europe": 1,
    "Canada": 89,
    "Australia": 2,
    "Colombia/Venezuela": 23,
    "West Africa": 25,
    "Kazakhstan/Central Russia": 21,
}

CROP_MAP: dict[str, int] = {
    "None": 0,
    "Rapeseed": 1,
    "Cocoa": 3,
    "Coffee": 4,
    "Corn": 5,
    "Cotton": 6,
    "Soybeans": 13,
    "Wheat": 17,
    "Palm Oil": 8,
    "Wheat - Winter": 20,
    "Wheat - Spring": 19,
}

PARAM_LABEL: dict[str, str] = {
    "PRCP": "Rainfall (mm)",
    "PRCP_NORM": "Normal rainfall (mm)",
    "PRCP_DEP": "Rainfall vs normal (%)",
    "SWL1": "Soil moisture — topsoil",
    "TMAX": "Max temperature (°C)",
    "TMIN": "Min temperature (°C)",
    "TAVG": "Avg temperature (°C)",
    "GDD": "Growing degree days",
}

PARAM_VALUE_COL: dict[str, str] = {
    "PRCP": "rainfall_mm",
    "PRCP_NORM": "normal_rainfall_mm",
    "PRCP_DEP": "rainfall_departure_pct",
    "SWL1": "soil_moisture_topsoil",
    "TMAX": "max_temp_c",
    "TMIN": "min_temp_c",
    "TAVG": "avg_temp_c",
    "GDD": "gdd",
}

PRIMARY_PARAM: dict[str, str] = {
    "Crude Palm Oil": "PRCP",
    "Palm Olein": "PRCP",
    "Soybeans": "PRCP",
    "Bean Oil": "PRCP",
    "Soymeal": "PRCP",
    "Canola/Rapeseed": "PRCP",
    "Rapeseed Oil": "PRCP",
    "Coffee": "PRCP",
    "Cocoa": "PRCP",
    "Corn": "PRCP",
    "Cotton": "TMAX",
    "Wheat": "PRCP",
}

DEFAULT_CACHE_DIR = Path("data") / "weather_cache"
DEFAULT_CACHE_PATH = DEFAULT_CACHE_DIR / "weather_daily.feather"


# =============================================================================
# DATE HELPERS
# =============================================================================

def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def as_date(value: str | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return _today()
    return pd.to_datetime(value).normalize()


def clamp_date_range(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    *,
    allow_future: bool = False,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return a clean inclusive date range in chronological order."""
    start = as_date(start_date)
    end = as_date(end_date)

    if not allow_future:
        end = min(end, _today())

    if start > end:
        raise ValueError(f"start_date {start.date()} is after end_date {end.date()}.")

    return start, end


def season_year_start(as_of: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = as_date(as_of)
    return pd.Timestamp(year=end.year, month=1, day=1), end


def current_month_range(as_of: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = as_date(as_of)
    return end.replace(day=1), end


def winter_wheat_range(as_of: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Winter wheat season: Sep 1 to current date."""
    end = as_date(as_of)
    start_year = end.year if end.month >= 9 else end.year - 1
    return pd.Timestamp(year=start_year, month=9, day=1), end


def spring_wheat_range(as_of: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Spring wheat season: Mar 1 to current date. If before Mar, use prior Mar."""
    end = as_date(as_of)
    start_year = end.year if end.month >= 3 else end.year - 1
    return pd.Timestamp(year=start_year, month=3, day=1), end


def make_date_chunks(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    max_days: int = 15,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start, end = clamp_date_range(start_date, end_date, allow_future=False)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start = start

    while current_start <= end:
        current_end = min(current_start + timedelta(days=max_days - 1), end)
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    return chunks


def compress_dates_to_ranges(dates: Iterable[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    unique_dates = sorted(pd.to_datetime(list(dates)).normalize().unique())
    if not unique_dates:
        return []

    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = prev = pd.Timestamp(unique_dates[0])

    for raw_date in unique_dates[1:]:
        current = pd.Timestamp(raw_date)
        if current == prev + timedelta(days=1):
            prev = current
        else:
            ranges.append((start, prev))
            start = prev = current

    ranges.append((start, prev))
    return ranges


# =============================================================================
# REGION CONFIG
# =============================================================================

@dataclass(frozen=True)
class AreaConfig:
    api_regions: list[str]
    crop_candidates: list[str]
    zone_col_candidates: list[str]
    zones: list[str]


@dataclass(frozen=True)
class ProfileConfig:
    title: str
    areas: dict[str, AreaConfig]
    standard_map: list[tuple[str, str]]
    default_range: Callable[[pd.Timestamp | None], tuple[pd.Timestamp, pd.Timestamp]]
    chart_profile: str
    parameters: list[str]


def _area(
    api_regions: list[str],
    crop_candidates: list[str],
    zone_col_candidates: list[str],
    zones: list[str],
) -> AreaConfig:
    return AreaConfig(
        api_regions=api_regions,
        crop_candidates=crop_candidates,
        zone_col_candidates=zone_col_candidates,
        zones=zones,
    )


PROFILE_CONFIGS: dict[str, ProfileConfig] = {
    "Crude Palm Oil": ProfileConfig(
        title="Crude Palm Oil",
        default_range=season_year_start,
        chart_profile="cpo",
        parameters=["PRCP", "PRCP_NORM"],
        standard_map=[
            (r"(?i)^johor$", "Johor"),
            (r"(?i)^kedah$", "Kedah"),
            (r"(?i)^melaka$", "Melaka"),
            (r"(?i)^pahang$", "Pahang"),
            (r"(?i)^perak$", "Perak"),
            (r"(?i)^sabah$", "Sabah"),
            (r"(?i)^sarawak$", "Sarawak"),
            (r"(?i)^selangor$", "Selangor"),
            (r"(?i)^aceh$", "Aceh"),
            (r"(?i)^jambi$", "Jambi"),
            (r"(?i)^kepulauan riau$", "Kepulauan Riau"),
            (r"(?i)^riau$", "Riau"),
            (r"(?i)^sumatera barat$", "Sumatera Barat"),
            (r"(?i)^sumatera selatan$", "Sumatera Selatan"),
            (r"(?i)^sumatera utara$", "Sumatera Utara"),
            (r"(?i)^kalimantan barat$", "Kalimantan Barat"),
            (r"(?i)^kalimantan tengah$", "Kalimantan Tengah"),
            (r"(?i)^kalimantan timur$", "Kalimantan Timur"),
        ],
        areas={
            "Malaysia": _area(
                ["Southeast Asia"],
                ["Palm Oil"],
                ["admin", "subregion"],
                ["Johor", "Kedah", "Melaka", "Pahang", "Perak", "Sabah", "Sarawak", "Selangor"],
            ),
            "Indonesia": _area(
                ["Southeast Asia"],
                ["Palm Oil"],
                ["admin", "subregion"],
                [
                    "Aceh", "Jambi", "Kepulauan Riau", "Riau", "Sumatera Barat",
                    "Sumatera Selatan", "Sumatera Utara", "Kalimantan Barat",
                    "Kalimantan Tengah", "Kalimantan Timur",
                ],
            ),
        },
    ),
    "Soybeans": ProfileConfig(
        title="Soybeans",
        default_range=season_year_start,
        chart_profile="soybeans",
        parameters=["PRCP", "PRCP_NORM"],
        standard_map=[
            (r"(?i)^iowa$", "Iowa"),
            (r"(?i)^illinois$", "Illinois"),
            (r"(?i)^indiana$", "Indiana"),
            (r"(?i)^minnesota$", "Minnesota"),
            (r"(?i)^nebraska$", "Nebraska"),
            (r"(?i)^mato grosso$", "Mato Grosso"),
            (r"(?i)^paran[aá]$", "Parana"),
            (r"(?i)^rio grande do sul$", "Rio Grande do Sul"),
            (r"(?i)^goi[aá]s$", "Goias"),
            (r"(?i)^c[oó]rdoba$", "Cordoba"),
            (r"(?i)^buenos aires$", "Buenos Aires"),
            (r"(?i)^santa fe$", "Santa Fe"),
        ],
        areas={
            "US": _area(
                ["United States"],
                ["Soybeans"],
                ["subregion", "admin"],
                ["Iowa", "Illinois", "Indiana", "Minnesota", "Nebraska"],
            ),
            "Brazil": _area(
                ["Brazil"],
                ["Soybeans"],
                ["admin", "subregion"],
                ["Mato Grosso", "Parana", "Rio Grande do Sul", "Goias"],
            ),
            "Argentina": _area(
                ["Argentina"],
                ["Soybeans"],
                ["admin", "subregion"],
                ["Cordoba", "Buenos Aires", "Santa Fe"],
            ),
        },
    ),
    "Canola/Rapeseed": ProfileConfig(
        title="Rapeseed / Canola",
        default_range=season_year_start,
        chart_profile="rapeseed",
        parameters=["PRCP", "PRCP_NORM"],
        standard_map=[
            (r"(?i)^alberta$", "Alberta"),
            (r"(?i)^saskatchewan$", "Saskatchewan"),
            (r"(?i)^manitoba$", "Manitoba"),
            (r"(?i)^france$", "France"),
            (r"(?i)^germany$", "Germany"),
            (r"(?i)^poland$", "Poland"),
            (r"(?i)^romania$", "Romania"),
        ],
        areas={
            "Canada": _area(
                ["Canada"],
                ["Rapeseed"],
                ["subregion", "admin"],
                ["Alberta", "Saskatchewan", "Manitoba"],
            ),
            "Europe": _area(
                ["Europe"],
                ["Rapeseed"],
                ["admin", "subregion"],
                ["France", "Germany", "Poland", "Romania"],
            ),
        },
    ),
    "Coffee": ProfileConfig(
        title="Coffee",
        default_range=season_year_start,
        chart_profile="coffee",
        parameters=["PRCP", "TMAX", "TMIN"],
        standard_map=[
            (r"(?i)^minas gerais$", "Minas Gerais"),
            (r"(?i)^s[aã]o paulo$", "Sao Paulo"),
            (r"(?i)^esp[ií]rito santo$", "Espirito Santo"),
            (r"(?i)^bahia$", "Bahia"),
            (r"(?i)^rond[oô]nia$", "Rondonia"),
            (r"(?i)^dak lak$", "Dak Lak"),
            (r"(?i)^gia lai$", "Gia Lai"),
            (r"(?i)^nam dong$", "Nam Dong"),
            (r"(?i)^lam dong$", "Lam Dong"),
            (r"(?i)^huila$", "Huila"),
            (r"(?i)^tolima$", "Tolima"),
            (r"(?i)^antioquia$", "Antioquia"),
        ],
        areas={
            "Brazil": _area(
                ["Brazil"],
                ["Coffee"],
                ["admin", "subregion"],
                ["Minas Gerais", "Sao Paulo", "Espirito Santo", "Bahia", "Rondonia"],
            ),
            "Vietnam": _area(
                ["Vietnam/Thailand"],
                ["Coffee"],
                ["admin", "subregion"],
                # Nam Dong was requested; Lam Dong is included because many APIs store the coffee area that way.
                ["Dak Lak", "Gia Lai", "Nam Dong", "Lam Dong"],
            ),
            "Colombia": _area(
                ["Colombia/Venezuela"],
                ["Coffee"],
                ["admin", "subregion"],
                ["Huila", "Tolima", "Antioquia"],
            ),
        },
    ),
    "Cocoa": ProfileConfig(
        title="Cocoa",
        default_range=season_year_start,
        chart_profile="cocoa",
        parameters=["PRCP", "PRCP_NORM"],
        standard_map=[
            (r"(?i)^san pedro$", "San Pedro"),
            (r"(?i)^gagnoa$", "Gagnoa"),
            (r"(?i)^western$", "Western"),
            (r"(?i)^ashanti$", "Ashanti"),
            (r"(?i)^brong ahafo$", "Brong Ahafo"),
            (r"(?i)^ogun$", "Ogun"),
        ],
        areas={
            "Cote d'Ivoire": _area(
                ["West Africa"],
                ["Cocoa"],
                ["admin", "subregion", "name"],
                ["San Pedro", "Gagnoa"],
            ),
            "Ghana": _area(
                ["West Africa"],
                ["Cocoa"],
                ["admin", "subregion"],
                ["Western", "Ashanti", "Brong Ahafo"],
            ),
            "Nigeria": _area(
                ["West Africa"],
                ["Cocoa"],
                ["admin", "subregion"],
                ["Ogun"],
            ),
        },
    ),
    "Wheat - Winter": ProfileConfig(
        title="Winter Wheat",
        default_range=winter_wheat_range,
        chart_profile="wheat_winter",
        parameters=["PRCP", "TMIN"],
        standard_map=[
            (r"(?i)^kansas$", "Kansas"),
            (r"(?i)^oklahoma$", "Oklahoma"),
            (r"(?i)^texas$", "Texas"),
            (r"(?i)^france$", "France"),
            (r"(?i)^germany$", "Germany"),
            (r"(?i)^poland$", "Poland"),
            (r"(?i)^romania$", "Romania"),
        ],
        areas={
            "US": _area(
                ["United States"],
                ["Wheat - Winter"],
                ["subregion", "admin"],
                ["Kansas", "Oklahoma", "Texas"],
            ),
            "EU": _area(
                ["Europe"],
                ["Wheat - Winter"],
                ["admin", "subregion"],
                ["France", "Germany", "Poland", "Romania"],
            ),
        },
    ),
    "Wheat - Spring": ProfileConfig(
        title="Spring Wheat",
        default_range=spring_wheat_range,
        chart_profile="wheat_spring",
        parameters=["PRCP", "PRCP_NORM", "TMAX"],
        standard_map=[
            (r"(?i)^alberta$", "Alberta"),
            (r"(?i)^manitoba$", "Manitoba"),
            (r"(?i)^saskatchewan$", "Saskatchewan"),
            (r"(?i)^north dakota$", "North Dakota"),
            (r"(?i)^montana$", "Montana"),
            (r"(?i)^minnesota$", "Minnesota"),
            (r"(?i)^russia$", "Russia"),
            (r"(?i)^russian federation$", "Russia"),
            (r"(?i)^kazakhstan$", "Kazakhstan"),
        ],
        areas={
            "Canada": _area(
                ["Canada"],
                ["Wheat - Spring"],
                ["subregion", "admin"],
                ["Alberta", "Manitoba", "Saskatchewan"],
            ),
            "US": _area(
                ["United States"],
                ["Wheat - Spring"],
                ["subregion", "admin"],
                ["North Dakota", "Montana", "Minnesota"],
            ),
            "Kazakhstan/Central Russia": _area(
                ["Kazakhstan/Central Russia"],
                ["Wheat - Spring"],
                ["admin", "subregion"],
                ["Russia", "Kazakhstan"],
            ),
        },
    ),
    "Corn": ProfileConfig(
        title="Corn",
        default_range=season_year_start,
        chart_profile="generic_rain_temp",
        parameters=["PRCP", "PRCP_NORM", "TAVG", "TMAX"],
        standard_map=[
            (r"(?i)^iowa$", "Iowa"),
            (r"(?i)^illinois$", "Illinois"),
            (r"(?i)^nebraska$", "Nebraska"),
            (r"(?i)^minnesota$", "Minnesota"),
            (r"(?i)^indiana$", "Indiana"),
            (r"(?i)^henan$", "Henan"),
            (r"(?i)^heilongjiang$", "Heilongjiang"),
            (r"(?i)^jilin$", "Jilin"),
        ],
        areas={
            "US": _area(
                ["United States"],
                ["Corn"],
                ["subregion", "admin"],
                ["Iowa", "Illinois", "Nebraska", "Minnesota", "Indiana"],
            ),
            "China": _area(
                ["China"],
                ["Corn"],
                ["admin", "subregion"],
                ["Henan", "Heilongjiang", "Jilin"],
            ),
        },
    ),
    "Cotton": ProfileConfig(
        title="Cotton",
        default_range=season_year_start,
        chart_profile="generic_rain_temp",
        parameters=["PRCP", "PRCP_NORM", "TMAX"],
        standard_map=[
            (r"(?i)^texas$", "Texas"),
            (r"(?i)^georgia$", "Georgia"),
            (r"(?i)^mississippi$", "Mississippi"),
            (r"(?i)^arkansas$", "Arkansas"),
            (r"(?i)^xinjiang uygur$", "Xinjiang Uygur"),
        ],
        areas={
            "US": _area(
                ["United States"],
                ["Cotton"],
                ["subregion", "admin"],
                ["Texas", "Georgia", "Mississippi", "Arkansas"],
            ),
            "China": _area(
                ["China"],
                ["Cotton"],
                ["admin", "subregion"],
                ["Xinjiang Uygur"],
            ),
        },
    ),
}

PROFILE_ALIASES: dict[str, str] = {
    "Palm Olein": "Crude Palm Oil",
    "Bean Oil": "Soybeans",
    "Soymeal": "Soybeans",
    "Rapeseed Oil": "Canola/Rapeseed",
    # Dashboard has one Wheat commodity; render both winter and spring charts.
    "Wheat": "Wheat",
}


def resolve_profile_key(commodity: str) -> str:
    if commodity in PROFILE_CONFIGS:
        return commodity
    return PROFILE_ALIASES.get(commodity, commodity)


def standardize_region_name(value: Any, profile_key: str) -> Any:
    if pd.isna(value):
        return value

    profile = PROFILE_CONFIGS[profile_key]
    cleaned = re.sub(r"\s+", " ", str(value).strip())

    for pattern, replacement in profile.standard_map:
        if re.match(pattern, cleaned):
            return replacement

    return cleaned


def resolve_code(value: str | int | None, code_map: dict[str, int]) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)

    cleaned = str(value).strip().lower()
    for key, code in code_map.items():
        if cleaned == key.lower():
            return code

    raise ValueError(f"{value!r} not found. Available options: {list(code_map)}")


# =============================================================================
# API RESPONSE CLEANING
# =============================================================================

def is_date_column(col: Any) -> bool:
    try:
        pd.to_datetime(str(col), format="%Y-%m-%d")
        return True
    except Exception:
        return False


def flatten_gwi_response(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    if "data" not in df.columns:
        raise ValueError(f"API response has no 'data' column. Columns: {list(df.columns)}")

    first_valid = df["data"].dropna().iloc[0] if not df["data"].dropna().empty else None
    if isinstance(first_valid, str):
        df["data"] = df["data"].apply(ast.literal_eval)

    df = df.explode("data").reset_index(drop=True)
    station_df = pd.json_normalize(df["data"])

    tidy = pd.concat(
        [df.drop(columns=["data"]).reset_index(drop=True), station_df.reset_index(drop=True)],
        axis=1,
    )
    return tidy


def convert_wide_to_long(tidy_df: pd.DataFrame, value_name: str = "weather_value") -> pd.DataFrame:
    date_cols = [col for col in tidy_df.columns if is_date_column(col)]
    if not date_cols:
        raise ValueError("No real date columns found in API response.")

    id_cols = [col for col in tidy_df.columns if col not in date_cols]

    # WeatherDesk sometimes returns metadata columns named `value` or `normal`.
    # pandas.melt() fails if value_name is already present in id_vars, so rename
    # any colliding metadata column before melting. This matters during refresh,
    # because refresh uses the raw API response instead of the feather cache.
    df_to_melt = tidy_df.copy()
    if value_name in id_cols:
        safe_existing_col = f"_orig_{value_name}"
        suffix = 1
        while safe_existing_col in df_to_melt.columns:
            safe_existing_col = f"_orig_{value_name}_{suffix}"
            suffix += 1
        df_to_melt = df_to_melt.rename(columns={value_name: safe_existing_col})
        id_cols = [safe_existing_col if col == value_name else col for col in id_cols]

    long_df = df_to_melt.melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="date",
        value_name=value_name,
    )

    long_df["date"] = pd.to_datetime(long_df["date"]).dt.normalize()
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    long_df = long_df.dropna(subset=[value_name]).copy()
    return long_df


def fetch_station_daily_by_area(
    *,
    profile_key: str,
    area_label: str,
    area_cfg: AreaConfig,
    parameter: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    base_url: str = BASE_URL,
    table_endpoint: str = TABLE_ENDPOINT,
    model: int = 0,
    metric: int = 1,
    grouping: str = "D",
    agg_method: str = "mean",
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch one parameter for one area config; tries crop / region / zone-column options."""
    errors: list[dict[str, str]] = []

    start, end = clamp_date_range(start_date, end_date)
    max_days = 15 if grouping == "D" else 185

    for crop_name in area_cfg.crop_candidates:
        for api_region in area_cfg.api_regions:
            for zone_col in area_cfg.zone_col_candidates:
                try:
                    crop_code = resolve_code(crop_name, CROP_MAP)
                    region_code = resolve_code(api_region, REGION_MAP)

                    all_raw: list[pd.DataFrame] = []
                    for chunk_start, chunk_end in make_date_chunks(start, end, max_days=max_days):
                        params = {
                            "type": "station",
                            "grouping": grouping,
                            "metric": metric,
                            "region": region_code,
                            "parameter": parameter,
                            "model": model,
                            "crop": crop_code,
                            "start": chunk_start.strftime("%Y-%m-%d"),
                            "end": chunk_end.strftime("%Y-%m-%d"),
                        }

                        response = requests.get(base_url + table_endpoint, params=params, timeout=timeout)
                        response.raise_for_status()
                        raw_df = pd.DataFrame(response.json())

                        if not raw_df.empty:
                            all_raw.append(raw_df)

                    if not all_raw:
                        raise ValueError("Empty API response.")

                    tidy = flatten_gwi_response(pd.concat(all_raw, ignore_index=True))
                    long_df = convert_wide_to_long(tidy, value_name="_weather_value")

                    if zone_col not in long_df.columns:
                        raise ValueError(
                            f"zone_col={zone_col!r} not found. Available columns: {list(long_df.columns)}"
                        )

                    long_df["_region_std"] = long_df[zone_col].apply(
                        lambda value: standardize_region_name(value, profile_key)
                    )

                    # Exact match only. This prevents e.g. Mato Grosso from pulling Mato Grosso do Sul.
                    long_df = long_df[long_df["_region_std"].isin(area_cfg.zones)].copy()

                    if long_df.empty:
                        raise ValueError("No rows left after exact region filtering.")

                    station_count_col = "name" if "name" in long_df.columns else zone_col
                    daily = (
                        long_df.groupby(["_region_std", "date"], as_index=False)
                        .agg(
                            value=("_weather_value", agg_method),
                            station_count=(station_count_col, "nunique"),
                        )
                        .rename(columns={"_region_std": "region"})
                    )

                    daily["profile"] = profile_key
                    daily["area"] = area_label
                    daily["parameter"] = parameter
                    daily["crop_used"] = crop_name
                    daily["api_region_used"] = api_region
                    daily["zone_col_used"] = zone_col
                    daily["updated_at"] = pd.Timestamp.utcnow().tz_localize(None)

                    daily = daily[
                        [
                            "profile", "area", "region", "parameter", "date", "value",
                            "station_count", "crop_used", "api_region_used", "zone_col_used",
                            "updated_at",
                        ]
                    ].sort_values(["area", "region", "parameter", "date"])

                    return daily

                except Exception as exc:
                    errors.append(
                        {
                            "crop": crop_name,
                            "api_region": api_region,
                            "zone_col": zone_col,
                            "error": str(exc),
                        }
                    )

    error_df = pd.DataFrame(errors)
    raise ValueError(
        f"Could not fetch {parameter} for {profile_key} / {area_label}. "
        f"Tried combinations:\n{error_df.tail(12).to_string(index=False)}"
    )


# =============================================================================
# FEATHER CACHE
# =============================================================================

def load_weather_cache(cache_path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    path = Path(cache_path)
    fallback = path.with_suffix(".pkl")

    if path.exists():
        return pd.read_feather(path)

    if fallback.exists():
        return pd.read_pickle(fallback)

    return pd.DataFrame(
        columns=[
            "profile", "area", "region", "parameter", "date", "value",
            "station_count", "crop_used", "api_region_used", "zone_col_used", "updated_at",
        ]
    )


def save_weather_cache(df: pd.DataFrame, cache_path: str | Path = DEFAULT_CACHE_PATH) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out["updated_at"] = pd.to_datetime(out["updated_at"], errors="coerce")

    try:
        out.reset_index(drop=True).to_feather(path)
        return path
    except Exception as exc:
        fallback = path.with_suffix(".pkl")
        warnings.warn(
            f"Could not save feather cache because {exc!r}. Saved pickle fallback: {fallback}",
            RuntimeWarning,
        )
        out.reset_index(drop=True).to_pickle(fallback)
        return fallback


def _drop_rows(
    cache: pd.DataFrame,
    *,
    profile_key: str,
    area: str,
    parameter: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if cache.empty:
        return cache

    dates = pd.to_datetime(cache["date"]).dt.normalize()
    mask = (
        cache["profile"].eq(profile_key)
        & cache["area"].eq(area)
        & cache["parameter"].eq(parameter)
        & dates.between(start, end)
    )
    return cache.loc[~mask].copy()


def _existing_area_dates(
    cache: pd.DataFrame,
    *,
    profile_key: str,
    area: str,
    parameter: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> set[pd.Timestamp]:
    if cache.empty:
        return set()

    tmp = cache[
        cache["profile"].eq(profile_key)
        & cache["area"].eq(area)
        & cache["parameter"].eq(parameter)
    ].copy()

    if tmp.empty:
        return set()

    tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
    tmp = tmp[tmp["date"].between(start, end)]
    return set(tmp["date"].unique())


def update_weather_cache(
    commodity: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    *,
    parameters: list[str] | None = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    force_refresh: bool = False,
    refresh_recent_days: int = 1,
) -> pd.DataFrame:
    """
    Update cache for a commodity profile and return the full cache.

    The function only fetches missing dates plus the last N recent days, so
    repeated runs are much faster than pulling the full period again.
    """
    profile_key = resolve_profile_key(commodity)
    if profile_key == "Wheat":
        # Update both wheat profiles.
        cache = load_weather_cache(cache_path)
        for wheat_profile in ["Wheat - Winter", "Wheat - Spring"]:
            cache = update_weather_cache(
                wheat_profile,
                start_date=start_date,
                end_date=end_date,
                parameters=parameters,
                cache_path=cache_path,
                force_refresh=force_refresh,
                refresh_recent_days=refresh_recent_days,
            )
        return load_weather_cache(cache_path)

    if profile_key not in PROFILE_CONFIGS:
        raise ValueError(f"No weather profile configured for {commodity!r}.")

    cfg = PROFILE_CONFIGS[profile_key]

    if start_date is None or end_date is None:
        default_start, default_end = cfg.default_range(None)
        start = as_date(start_date) if start_date is not None else default_start
        end = as_date(end_date) if end_date is not None else default_end
    else:
        start, end = clamp_date_range(start_date, end_date)

    start, end = clamp_date_range(start, end)
    params = parameters or cfg.parameters

    cache = load_weather_cache(cache_path)
    if not cache.empty:
        cache["date"] = pd.to_datetime(cache["date"]).dt.normalize()

    wanted_dates = set(pd.date_range(start, end, freq="D"))

    # Always refresh the most recent days because today's feed can be revised.
    if refresh_recent_days > 0:
        refresh_start = max(start, end - timedelta(days=refresh_recent_days - 1))
        recent_dates = set(pd.date_range(refresh_start, end, freq="D"))
    else:
        recent_dates = set()

    fetched_frames: list[pd.DataFrame] = []

    for area_label, area_cfg in cfg.areas.items():
        for parameter in params:
            if force_refresh:
                missing_dates = wanted_dates
            else:
                existing = _existing_area_dates(
                    cache,
                    profile_key=profile_key,
                    area=area_label,
                    parameter=parameter,
                    start=start,
                    end=end,
                )
                missing_dates = (wanted_dates - existing) | recent_dates

            for fetch_start, fetch_end in compress_dates_to_ranges(missing_dates):
                fresh = fetch_station_daily_by_area(
                    profile_key=profile_key,
                    area_label=area_label,
                    area_cfg=area_cfg,
                    parameter=parameter,
                    start_date=fetch_start,
                    end_date=fetch_end,
                )

                cache = _drop_rows(
                    cache,
                    profile_key=profile_key,
                    area=area_label,
                    parameter=parameter,
                    start=fetch_start,
                    end=fetch_end,
                )
                fetched_frames.append(fresh)

    if fetched_frames:
        cache = pd.concat([cache, *fetched_frames], ignore_index=True)

    if not cache.empty:
        cache["date"] = pd.to_datetime(cache["date"]).dt.normalize()
        cache = (
            cache.drop_duplicates(
                subset=["profile", "area", "region", "parameter", "date"],
                keep="last",
            )
            .sort_values(["profile", "area", "region", "parameter", "date"])
            .reset_index(drop=True)
        )

    save_weather_cache(cache, cache_path)
    return cache


def load_weather_daily(
    commodity: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    *,
    parameters: list[str] | None = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    force_refresh: bool = False,
    refresh_recent_days: int = 1,
) -> pd.DataFrame:
    """Load daily weather for one commodity from cache, updating missing dates first."""
    profile_key = resolve_profile_key(commodity)

    if profile_key == "Wheat":
        frames = []
        for wheat_profile in ["Wheat - Winter", "Wheat - Spring"]:
            frames.append(
                load_weather_daily(
                    wheat_profile,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=parameters,
                    cache_path=cache_path,
                    force_refresh=force_refresh,
                    refresh_recent_days=refresh_recent_days,
                )
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if profile_key not in PROFILE_CONFIGS:
        return pd.DataFrame()

    cfg = PROFILE_CONFIGS[profile_key]

    if start_date is None or end_date is None:
        default_start, default_end = cfg.default_range(None)
        start = as_date(start_date) if start_date is not None else default_start
        end = as_date(end_date) if end_date is not None else default_end
    else:
        start, end = clamp_date_range(start_date, end_date)

    params = parameters or cfg.parameters

    update_weather_cache(
        profile_key,
        start,
        end,
        parameters=params,
        cache_path=cache_path,
        force_refresh=force_refresh,
        refresh_recent_days=refresh_recent_days,
    )

    cache = load_weather_cache(cache_path)
    if cache.empty:
        return cache

    cache["date"] = pd.to_datetime(cache["date"]).dt.normalize()
    out = cache[
        cache["profile"].eq(profile_key)
        & cache["parameter"].isin(params)
        & cache["date"].between(start, end)
    ].copy()

    return out.sort_values(["profile", "area", "region", "parameter", "date"]).reset_index(drop=True)


def to_daily_wide(daily_long: pd.DataFrame) -> pd.DataFrame:
    if daily_long.empty:
        return daily_long.copy()

    wide = (
        daily_long.pivot_table(
            index=["profile", "area", "region", "date"],
            columns="parameter",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    rename_map = {param: PARAM_VALUE_COL.get(param, param.lower()) for param in wide.columns}
    wide = wide.rename(columns=rename_map)
    wide["date"] = pd.to_datetime(wide["date"]).dt.normalize()
    return wide.sort_values(["profile", "area", "region", "date"]).reset_index(drop=True)


def to_rainfall_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible helper used by older dashboard code."""
    return to_daily_wide(df)


# =============================================================================
# METRIC CALCULATIONS
# =============================================================================

def monthly_rainfall_summary(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame()

    df = wide.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    agg_map = {"rainfall_mm": ("rainfall_mm", "sum")}
    if "normal_rainfall_mm" in df.columns:
        agg_map["normal_rainfall_mm"] = ("normal_rainfall_mm", "sum")

    monthly = (
        df.groupby(["profile", "area", "region", "month"], as_index=False)
        .agg(**agg_map)
        .sort_values(["area", "region", "month"])
    )

    if "normal_rainfall_mm" in monthly.columns:
        monthly["rainfall_anomaly_mm"] = monthly["rainfall_mm"] - monthly["normal_rainfall_mm"]
        monthly["rainfall_pct_of_normal"] = np.where(
            monthly["normal_rainfall_mm"] > 0,
            monthly["rainfall_mm"] / monthly["normal_rainfall_mm"] * 100,
            np.nan,
        )
    else:
        monthly["rainfall_anomaly_mm"] = np.nan
        monthly["rainfall_pct_of_normal"] = np.nan

    monthly["month_label"] = monthly["month"].dt.strftime("%b %Y")
    monthly["region_label"] = monthly["area"] + " - " + monthly["region"]
    return monthly


def add_monthly_cumulative_rainfall(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["area", "region", "month", "date"])
    df["monthly_cumulative_rainfall_mm"] = (
        df.groupby(["area", "region", "month"])["rainfall_mm"].cumsum()
    )

    if "normal_rainfall_mm" in df.columns:
        df["monthly_cumulative_normal_mm"] = (
            df.groupby(["area", "region", "month"])["normal_rainfall_mm"].cumsum()
        )
        df["cumulative_pct_of_normal"] = np.where(
            df["monthly_cumulative_normal_mm"] > 0,
            df["monthly_cumulative_rainfall_mm"] / df["monthly_cumulative_normal_mm"] * 100,
            np.nan,
        )

    return df


def max_consecutive_true(values: Iterable[bool]) -> int:
    max_streak = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def monthly_consecutive_dry_days(
    wide: pd.DataFrame,
    *,
    dry_day_threshold_mm: float = 1.0,
) -> pd.DataFrame:
    df = wide.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["is_dry_day"] = df["rainfall_mm"].fillna(0) < dry_day_threshold_mm
    df = df.sort_values(["area", "region", "date"])

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(["profile", "area", "region", "month"], sort=False):
        profile, area, region, month = keys
        rows.append(
            {
                "profile": profile,
                "area": area,
                "region": region,
                "month": month,
                "max_consecutive_dry_days": max_consecutive_true(group["is_dry_day"]),
                "dry_days_count": int(group["is_dry_day"].sum()),
                "total_days": int(group["is_dry_day"].count()),
                "monthly_rainfall_mm": float(group["rainfall_mm"].sum()),
            }
        )

    monthly = pd.DataFrame(rows)
    if not monthly.empty:
        monthly["month_label"] = pd.to_datetime(monthly["month"]).dt.strftime("%b %Y")
    return monthly


def monthly_temperature_summary(
    wide: pd.DataFrame,
    *,
    heat_watch_c: float = 30.0,
    extreme_heat_c: float = 35.0,
    frost_threshold_c: float = 2.0,
    freeze_threshold_c: float = 0.0,
    hard_freeze_c: float = -10.0,
    winterkill_watch_c: float = -15.0,
) -> pd.DataFrame:
    df = wide.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    agg_spec: dict[str, tuple[str, str | Callable[[pd.Series], Any]]] = {}
    if "max_temp_c" in df.columns:
        agg_spec.update(
            {
                "hottest_day_c": ("max_temp_c", "max"),
                "avg_tmax_c": ("max_temp_c", "mean"),
                "heat_watch_days": ("max_temp_c", lambda s: int((s >= heat_watch_c).sum())),
                "extreme_heat_days": ("max_temp_c", lambda s: int((s >= extreme_heat_c).sum())),
            }
        )

    if "min_temp_c" in df.columns:
        agg_spec.update(
            {
                "coldest_tmin_c": ("min_temp_c", "min"),
                "avg_tmin_c": ("min_temp_c", "mean"),
                "frost_days": ("min_temp_c", lambda s: int((s <= frost_threshold_c).sum())),
                "freeze_days": ("min_temp_c", lambda s: int((s <= freeze_threshold_c).sum())),
                "hard_freeze_days": ("min_temp_c", lambda s: int((s <= hard_freeze_c).sum())),
                "winterkill_watch_days": ("min_temp_c", lambda s: int((s <= winterkill_watch_c).sum())),
            }
        )

    if not agg_spec:
        return pd.DataFrame()

    monthly = (
        df.groupby(["profile", "area", "region", "month"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["area", "region", "month"])
    )
    monthly["month_label"] = monthly["month"].dt.strftime("%b %Y")
    return monthly


# =============================================================================
# PLOTLY CHARTS
# =============================================================================

def _template(dark: bool = False) -> str:
    return "plotly_dark" if dark else "plotly_white"


def _style_fig(fig: go.Figure, *, dark: bool = False, height: int = 520) -> go.Figure:
    fig.update_layout(
        height=height,
        template=_template(dark),
        hovermode="x unified",
        legend_title_text="Region",
        margin=dict(l=20, r=20, t=80, b=40),
    )
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def make_cumulative_rainfall_chart(
    wide: pd.DataFrame,
    *,
    title: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    latest_month_only: bool = True,
    dark: bool = False,
) -> go.Figure:
    df = add_monthly_cumulative_rainfall(wide)

    hover_data = {
        "area": True,
        "region": True,
        "rainfall_mm": ":.1f",
        "monthly_cumulative_rainfall_mm": ":.1f",
        "date": "|%d %b %Y",
    }
    if "monthly_cumulative_normal_mm" in df.columns:
        hover_data["monthly_cumulative_normal_mm"] = ":.1f"
    if "cumulative_pct_of_normal" in df.columns:
        hover_data["cumulative_pct_of_normal"] = ":.0f"

    fig = px.line(
        df,
        x="date",
        y="monthly_cumulative_rainfall_mm",
        color="region",
        facet_col="area",
        facet_col_spacing=0.06,
        markers=True,
        title=f"{title}<br><sup>{start_date.date()} to {end_date.date()}</sup>",
        labels={
            "date": "Date",
            "monthly_cumulative_rainfall_mm": "Monthly cumulative rainfall (mm)",
            "area": "Area",
            "region": "Region",
        },
        hover_data=hover_data,
    )
    return _style_fig(fig, dark=dark)


def make_season_cumulative_rainfall_chart(
    wide: pd.DataFrame,
    *,
    title: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    dark: bool = False,
) -> go.Figure:
    df = wide.sort_values(["area", "region", "date"]).copy()
    df["season_cumulative_rainfall_mm"] = (
        df.groupby(["area", "region"])["rainfall_mm"].cumsum()
    )

    fig = px.line(
        df,
        x="date",
        y="season_cumulative_rainfall_mm",
        color="region",
        facet_col="area",
        facet_col_spacing=0.07,
        title=f"{title}<br><sup>{start_date.date()} to {end_date.date()}</sup>",
        labels={
            "date": "Date",
            "season_cumulative_rainfall_mm": "Season cumulative rainfall (mm)",
            "area": "Area",
            "region": "Region",
        },
        hover_data={"rainfall_mm": ":.1f", "date": "|%d %b %Y"},
    )
    return _style_fig(fig, dark=dark)


def make_rainfall_anomaly_chart(
    wide: pd.DataFrame,
    *,
    title: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    dark: bool = False,
) -> go.Figure:
    monthly = monthly_rainfall_summary(wide)

    fig = px.bar(
        monthly,
        x="month",
        y="rainfall_anomaly_mm",
        color="region",
        facet_col="area",
        facet_col_spacing=0.07,
        barmode="group",
        title=f"{title}<br><sup>{start_date.date()} to {end_date.date()}</sup>",
        labels={
            "month": "Month",
            "rainfall_anomaly_mm": "Rainfall anomaly (mm)",
            "area": "Area",
            "region": "Region",
        },
        hover_data={
            "month_label": True,
            "rainfall_mm": ":.1f",
            "normal_rainfall_mm": ":.1f",
            "rainfall_pct_of_normal": ":.1f",
            "month": False,
        },
    )
    fig = _style_fig(fig, dark=dark)
    fig.update_yaxes(zeroline=True, zerolinewidth=1)
    return fig


def make_excessive_rainfall_heatmap(
    wide: pd.DataFrame,
    *,
    title: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    region_order: list[str] | None = None,
    dark: bool = False,
    zmin: float = 50,
    zmax: float = 200,
) -> go.Figure:
    monthly = monthly_rainfall_summary(wide)
    monthly["risk_label"] = np.select(
        [
            monthly["rainfall_pct_of_normal"] >= 150,
            monthly["rainfall_pct_of_normal"] >= 120,
            monthly["rainfall_pct_of_normal"] < 80,
        ],
        ["Excessive", "Wet", "Dry"],
        default="Normal",
    )

    if region_order:
        monthly["region_label"] = pd.Categorical(
            monthly["region_label"],
            categories=region_order,
            ordered=True,
        )

    monthly = monthly.sort_values(["region_label", "month"])

    month_col_order = (
        monthly[["month", "month_label"]]
        .drop_duplicates()
        .sort_values("month")["month_label"]
        .tolist()
    )

    z_df = monthly.pivot_table(
        index="region_label",
        columns="month_label",
        values="rainfall_pct_of_normal",
        aggfunc="mean",
        observed=False,
    ).reindex(columns=month_col_order)
    risk_df = monthly.pivot_table(
        index="region_label",
        columns="month_label",
        values="risk_label",
        aggfunc="first",
        observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)
    rain_df = monthly.pivot_table(
        index="region_label",
        columns="month_label",
        values="rainfall_mm",
        aggfunc="sum",
        observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)
    normal_df = monthly.pivot_table(
        index="region_label",
        columns="month_label",
        values="normal_rainfall_mm",
        aggfunc="sum",
        observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)
    anomaly_df = monthly.pivot_table(
        index="region_label",
        columns="month_label",
        values="rainfall_anomaly_mm",
        aggfunc="sum",
        observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)

    z_text = z_df.round(0).astype("Int64").astype(str).replace("<NA>", "")
    text_grid = risk_df.fillna("").astype(str) + "<br>" + z_text + "%"
    text_grid = text_grid.replace("<br>%", "")

    customdata = np.stack(
        [
            risk_df.fillna("").values,
            rain_df.round(1).values,
            normal_df.round(1).values,
            anomaly_df.round(1).values,
        ],
        axis=-1,
    )

    colorscale = [
        [0.00, "#b2182b"],
        [0.35, "#f7f7f7"],
        [0.60, "#92c5de"],
        [1.00, "#2166ac"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_df.values,
            x=z_df.columns,
            y=z_df.index.astype(str),
            text=text_grid.values,
            texttemplate="%{text}",
            customdata=customdata,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            colorbar=dict(title="% of normal", ticksuffix="%"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Month: %{x}<br>"
                "Risk: %{customdata[0]}<br>"
                "Rainfall: %{customdata[1]} mm<br>"
                "Normal: %{customdata[2]} mm<br>"
                "Anomaly: %{customdata[3]} mm<br>"
                "% of normal: %{z:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{title}<br><sup>{start_date.date()} to {end_date.date()} | Excessive ≥ 150% of normal</sup>",
        xaxis_title="Month",
        yaxis_title="Production region",
        template=_template(dark),
        height=560,
        margin=dict(l=170, r=70, t=90, b=60),
    )
    return fig


def make_coffee_temperature_chart(
    wide: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    dark: bool = False,
    heat_threshold_c: float = 34.0,
    frost_threshold_c: float = 2.0,
) -> go.Figure:
    monthly = monthly_temperature_summary(
        wide,
        heat_watch_c=heat_threshold_c,
        extreme_heat_c=heat_threshold_c + 3,
        frost_threshold_c=frost_threshold_c,
    )

    plot_df = monthly.melt(
        id_vars=[
            "profile", "area", "region", "month", "month_label",
            "avg_tmax_c", "avg_tmin_c", "heat_watch_days", "frost_days",
        ],
        value_vars=["hottest_day_c", "coldest_tmin_c"],
        var_name="metric",
        value_name="temp_c",
    )
    plot_df["metric"] = plot_df["metric"].replace(
        {
            "hottest_day_c": "Hottest day",
            "coldest_tmin_c": "Coldest night",
        }
    )

    fig = px.line(
        plot_df,
        x="month",
        y="temp_c",
        color="region",
        line_dash="metric",
        facet_col="area",
        facet_col_spacing=0.06,
        markers=True,
        title=f"Coffee Temperature Risk: Heat / Frost<br><sup>{start_date.date()} to {end_date.date()}</sup>",
        labels={
            "month": "Month",
            "temp_c": "Temperature (°C)",
            "area": "Area",
            "region": "Region",
            "metric": "Metric",
        },
        hover_data={
            "month_label": True,
            "avg_tmax_c": ":.1f",
            "avg_tmin_c": ":.1f",
            "heat_watch_days": True,
            "frost_days": True,
            "month": False,
        },
    )
    fig = _style_fig(fig, dark=dark)

    fig.add_hline(
        y=heat_threshold_c,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Heat watch ({heat_threshold_c:g}°C)",
        annotation_position="top left",
    )
    fig.add_hline(
        y=frost_threshold_c,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Frost watch ({frost_threshold_c:g}°C)",
        annotation_position="bottom left",
    )
    return fig


def make_consecutive_dry_days_chart(
    wide: pd.DataFrame,
    *,
    title: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    dark: bool = False,
    dry_day_threshold_mm: float = 1.0,
) -> go.Figure:
    monthly = monthly_consecutive_dry_days(wide, dry_day_threshold_mm=dry_day_threshold_mm)

    fig = px.line(
        monthly,
        x="month",
        y="max_consecutive_dry_days",
        color="region",
        facet_col="area",
        facet_col_spacing=0.07,
        markers=True,
        title=(
            f"{title}<br><sup>{start_date.date()} to {end_date.date()} | "
            f"Dry day: rainfall < {dry_day_threshold_mm:g} mm</sup>"
        ),
        labels={
            "month": "Month",
            "max_consecutive_dry_days": "Max consecutive dry days",
            "area": "Area",
            "region": "Region",
        },
        hover_data={
            "month_label": True,
            "dry_days_count": True,
            "total_days": True,
            "monthly_rainfall_mm": ":.1f",
            "month": False,
        },
    )
    return _style_fig(fig, dark=dark)


def make_winter_min_temp_chart(
    wide: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    dark: bool = False,
    freeze_threshold_c: float = 0.0,
    hard_freeze_c: float = -10.0,
    winterkill_watch_c: float = -15.0,
) -> go.Figure:
    monthly = monthly_temperature_summary(
        wide,
        freeze_threshold_c=freeze_threshold_c,
        hard_freeze_c=hard_freeze_c,
        winterkill_watch_c=winterkill_watch_c,
    )

    fig = px.line(
        monthly,
        x="month",
        y="coldest_tmin_c",
        color="region",
        facet_col="area",
        facet_col_spacing=0.07,
        markers=True,
        title=f"Winter Wheat Minimum Temperature Risk<br><sup>{start_date.date()} to {end_date.date()}</sup>",
        labels={
            "month": "Month",
            "coldest_tmin_c": "Coldest min temperature (°C)",
            "area": "Area",
            "region": "Region",
        },
        hover_data={
            "month_label": True,
            "avg_tmin_c": ":.1f",
            "freeze_days": True,
            "hard_freeze_days": True,
            "winterkill_watch_days": True,
            "month": False,
        },
    )
    fig = _style_fig(fig, dark=dark)

    fig.add_hline(
        y=freeze_threshold_c,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Freeze ({freeze_threshold_c:g}°C)",
        annotation_position="top left",
    )
    fig.add_hline(
        y=hard_freeze_c,
        line_dash="dash",
        line_color="purple",
        annotation_text=f"Hard freeze ({hard_freeze_c:g}°C)",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=winterkill_watch_c,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Winterkill watch ({winterkill_watch_c:g}°C)",
        annotation_position="bottom left",
    )
    return fig


def make_spring_max_temp_chart(
    wide: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    dark: bool = False,
    heat_watch_c: float = 30.0,
    extreme_heat_c: float = 35.0,
) -> go.Figure:
    monthly = monthly_temperature_summary(
        wide,
        heat_watch_c=heat_watch_c,
        extreme_heat_c=extreme_heat_c,
    )

    fig = px.line(
        monthly,
        x="month",
        y="hottest_day_c",
        color="region",
        facet_col="area",
        facet_col_spacing=0.07,
        markers=True,
        title=f"Spring Wheat Maximum Temperature Risk<br><sup>{start_date.date()} to {end_date.date()}</sup>",
        labels={
            "month": "Month",
            "hottest_day_c": "Hottest max temperature (°C)",
            "area": "Area",
            "region": "Region",
        },
        hover_data={
            "month_label": True,
            "avg_tmax_c": ":.1f",
            "heat_watch_days": True,
            "extreme_heat_days": True,
            "month": False,
        },
    )
    fig = _style_fig(fig, dark=dark)

    fig.add_hline(
        y=heat_watch_c,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Heat watch ({heat_watch_c:g}°C)",
        annotation_position="top left",
    )
    fig.add_hline(
        y=extreme_heat_c,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Extreme heat ({extreme_heat_c:g}°C)",
        annotation_position="top left",
    )
    return fig


# =============================================================================
# GENERIC RAIN + TEMP CHART FUNCTIONS  (Corn, Cotton)
# =============================================================================

def make_drought_index_heatmap(
    wide: pd.DataFrame,
    *,
    title: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    region_order: list[str] | None = None,
    dark: bool = False,
    zmin: float = 0,
    zmax: float = 80,
) -> go.Figure:
    """Drought index heatmap: 100 - (rainfall / normal * 100), clipped 0-100."""
    df = wide.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby(["profile", "area", "region", "month"], as_index=False)
        .agg(
            rainfall_mm=("rainfall_mm", "sum"),
            normal_rainfall_mm=("normal_rainfall_mm", "sum"),
        )
    )

    monthly["rainfall_pct_of_normal"] = np.where(
        monthly["normal_rainfall_mm"] > 0,
        monthly["rainfall_mm"] / monthly["normal_rainfall_mm"] * 100,
        np.nan,
    )
    monthly["drought_index"] = (100 - monthly["rainfall_pct_of_normal"]).clip(lower=0, upper=100)
    monthly["drought_label"] = np.select(
        [
            monthly["rainfall_pct_of_normal"] <= 50,
            monthly["rainfall_pct_of_normal"] <= 80,
            monthly["rainfall_pct_of_normal"] < 100,
        ],
        ["Severe drought", "Drought", "Below normal"],
        default="Normal / wet",
    )
    monthly["month_label"] = monthly["month"].dt.strftime("%b %Y")
    monthly["region_label"] = monthly["area"] + " - " + monthly["region"]

    if region_order:
        monthly["region_label"] = pd.Categorical(
            monthly["region_label"], categories=region_order, ordered=True
        )
    monthly = monthly.sort_values(["region_label", "month"])

    # Derive chronologically-ordered column list from the sorted month timestamps
    month_col_order = (
        monthly[["month", "month_label"]]
        .drop_duplicates()
        .sort_values("month")["month_label"]
        .tolist()
    )

    z_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="drought_index", aggfunc="mean", observed=False,
    ).reindex(columns=month_col_order)
    rain_pct_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="rainfall_pct_of_normal", aggfunc="mean", observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)
    label_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="drought_label", aggfunc="first", observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)
    rain_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="rainfall_mm", aggfunc="sum", observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)
    normal_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="normal_rainfall_mm", aggfunc="sum", observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)

    z_text = z_df.round(0).astype("Int64").astype(str).replace("<NA>", "")
    text_grid = label_df.fillna("").astype(str) + "<br>" + z_text
    text_grid = text_grid.replace("<br>", "")

    customdata = np.stack(
        [
            label_df.fillna("").values,
            rain_pct_df.round(1).values,
            rain_df.round(1).values,
            normal_df.round(1).values,
        ],
        axis=-1,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_df.values,
            x=z_df.columns,
            y=z_df.index.astype(str),
            text=text_grid.values,
            texttemplate="%{text}",
            customdata=customdata,
            zmin=zmin,
            zmax=zmax,
            colorscale=[
                [0.00, "#f7f7f7"],
                [0.35, "#fdd49e"],
                [0.65, "#fc8d59"],
                [1.00, "#b30000"],
            ],
            colorbar=dict(title="Drought index"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Month: %{x}<br>"
                "Risk: %{customdata[0]}<br>"
                "Rainfall vs normal: %{customdata[1]}%<br>"
                "Rainfall: %{customdata[2]} mm<br>"
                "Normal rainfall: %{customdata[3]} mm<br>"
                "Drought index: %{z:.1f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=(
            f"{title}<br><sup>{start_date.date()} to {end_date.date()} | "
            "Higher value = rainfall further below normal</sup>"
        ),
        xaxis_title="Month",
        yaxis_title="Production region",
        template=_template(dark),
        height=430,
        margin=dict(l=170, r=80, t=90, b=70),
    )
    return fig


def make_temperature_stress_heatmap(
    wide: pd.DataFrame,
    *,
    title: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    region_order: list[str] | None = None,
    dark: bool = False,
    ideal_low_c: float = 25.0,
    ideal_high_c: float = 32.8,
    severe_heat_c: float = 35.0,
    zmin: float = 0,
    zmax: float = 25,
) -> go.Figure:
    """Temperature stress index heatmap using TAVG + TMAX (corn-style)."""
    df = wide.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    tavg_col = "avg_temp_c" if "avg_temp_c" in df.columns else None
    tmax_col = "max_temp_c" if "max_temp_c" in df.columns else None

    if tavg_col is None:
        raise ValueError("make_temperature_stress_heatmap requires avg_temp_c (TAVG) in wide data.")

    df["ideal_temp_day"] = (
        (df[tavg_col] >= ideal_low_c) & (df[tavg_col] <= ideal_high_c)
    ).astype(int)
    df["cool_stress_day"] = (df[tavg_col] < ideal_low_c).astype(int)
    df["heat_stress_day"] = (df[tavg_col] > ideal_high_c).astype(int)
    df["severe_heat_day"] = (
        (df[tmax_col] >= severe_heat_c).astype(int) if tmax_col else pd.Series(0, index=df.index)
    )

    agg_spec: dict[str, Any] = {
        "mean_tavg_c": (tavg_col, "mean"),
        "ideal_temp_days": ("ideal_temp_day", "sum"),
        "cool_stress_days": ("cool_stress_day", "sum"),
        "heat_stress_days": ("heat_stress_day", "sum"),
        "severe_heat_days": ("severe_heat_day", "sum"),
    }
    if tmax_col:
        agg_spec["mean_tmax_c"] = (tmax_col, "mean")
        agg_spec["max_tmax_c"] = (tmax_col, "max")

    monthly = df.groupby(["profile", "area", "region", "month"], as_index=False).agg(**agg_spec)
    monthly["temperature_stress_index"] = (
        monthly["cool_stress_days"]
        + monthly["heat_stress_days"]
        + 2 * monthly["severe_heat_days"]
    )
    monthly["temperature_label"] = np.select(
        [
            monthly["temperature_stress_index"] >= 18,
            monthly["temperature_stress_index"] >= 10,
            monthly["temperature_stress_index"] >= 4,
        ],
        ["Extreme temp stress", "High temp stress", "Moderate temp stress"],
        default="Ideal / low stress",
    )
    monthly["month_label"] = monthly["month"].dt.strftime("%b %Y")
    monthly["region_label"] = monthly["area"] + " - " + monthly["region"]

    if region_order:
        monthly["region_label"] = pd.Categorical(
            monthly["region_label"], categories=region_order, ordered=True
        )
    monthly = monthly.sort_values(["region_label", "month"])

    month_col_order = (
        monthly[["month", "month_label"]]
        .drop_duplicates()
        .sort_values("month")["month_label"]
        .tolist()
    )

    z_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="temperature_stress_index", aggfunc="mean", observed=False,
    ).reindex(columns=month_col_order)

    def _pivot(col: str) -> pd.DataFrame:
        return monthly.pivot_table(
            index="region_label", columns="month_label",
            values=col, aggfunc="mean" if col.startswith("mean") else ("sum" if col.endswith("days") else "max"),
            observed=False,
        ).reindex(index=z_df.index, columns=month_col_order)

    label_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="temperature_label", aggfunc="first", observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)

    z_text = z_df.round(0).astype("Int64").astype(str).replace("<NA>", "")
    text_grid = label_df.fillna("").astype(str) + "<br>" + z_text
    text_grid = text_grid.replace("<br>", "")

    mean_tavg_df = _pivot("mean_tavg_c")
    max_tmax_df = (
        _pivot("max_tmax_c") if "max_tmax_c" in monthly.columns
        else pd.DataFrame(np.nan, index=z_df.index, columns=z_df.columns)
    )
    ideal_df = _pivot("ideal_temp_days")
    cool_df = _pivot("cool_stress_days")
    heat_df = _pivot("heat_stress_days")
    severe_df = _pivot("severe_heat_days")

    customdata = np.stack(
        [
            label_df.fillna("").values,
            mean_tavg_df.round(1).values,
            max_tmax_df.round(1).values,
            ideal_df.round(0).values,
            cool_df.round(0).values,
            heat_df.round(0).values,
            severe_df.round(0).values,
        ],
        axis=-1,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_df.values,
            x=z_df.columns,
            y=z_df.index.astype(str),
            text=text_grid.values,
            texttemplate="%{text}",
            customdata=customdata,
            zmin=zmin,
            zmax=zmax,
            colorscale=[
                [0.00, "#f7f7f7"],
                [0.35, "#fee08b"],
                [0.65, "#f46d43"],
                [1.00, "#a50026"],
            ],
            colorbar=dict(title="Temp stress index"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Month: %{x}<br>"
                "Risk: %{customdata[0]}<br>"
                "Mean Tavg: %{customdata[1]}°C<br>"
                "Max Tmax: %{customdata[2]}°C<br>"
                "Ideal temp days: %{customdata[3]}<br>"
                "Cool stress days: %{customdata[4]}<br>"
                "Heat stress days: %{customdata[5]}<br>"
                "Severe heat days: %{customdata[6]}<br>"
                "Temperature stress index: %{z:.1f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=(
            f"{title}<br><sup>{start_date.date()} to {end_date.date()} | "
            f"Ideal range = {ideal_low_c}°C to {ideal_high_c}°C, severe heat = Tmax ≥ {severe_heat_c}°C</sup>"
        ),
        xaxis_title="Month",
        yaxis_title="Production region",
        template=_template(dark),
        height=500,
        margin=dict(l=170, r=80, t=90, b=70),
    )
    return fig


def make_rainfall_bias_heatmap(
    wide: pd.DataFrame,
    *,
    title: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    region_order: list[str] | None = None,
    dark: bool = False,
) -> go.Figure:
    """Rainfall bias / moisture risk heatmap (bullish dry ↔ excess moisture)."""
    df = wide.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby(["profile", "area", "region", "month"], as_index=False)
        .agg(
            rainfall_mm=("rainfall_mm", "sum"),
            normal_rainfall_mm=("normal_rainfall_mm", "sum"),
        )
    )
    monthly["rainfall_pct_of_normal"] = np.where(
        monthly["normal_rainfall_mm"] > 0,
        monthly["rainfall_mm"] / monthly["normal_rainfall_mm"] * 100,
        np.nan,
    )
    monthly["rainfall_bias_score"] = np.select(
        [
            monthly["rainfall_pct_of_normal"] < 80,
            monthly["rainfall_pct_of_normal"] <= 120,
            monthly["rainfall_pct_of_normal"] <= 150,
            monthly["rainfall_pct_of_normal"] > 150,
        ],
        [2, 0, -1, -2],
        default=np.nan,
    )
    monthly["rainfall_label"] = np.select(
        [
            monthly["rainfall_pct_of_normal"] < 80,
            monthly["rainfall_pct_of_normal"] <= 120,
            monthly["rainfall_pct_of_normal"] <= 150,
            monthly["rainfall_pct_of_normal"] > 150,
        ],
        ["Bullish dry", "Normal", "Wet volatility", "Excess moisture"],
        default="No data",
    )
    monthly["month_label"] = monthly["month"].dt.strftime("%b %Y")
    monthly["region_label"] = monthly["area"] + " - " + monthly["region"]

    if region_order:
        monthly["region_label"] = pd.Categorical(
            monthly["region_label"], categories=region_order, ordered=True
        )
    monthly = monthly.sort_values(["region_label", "month"])

    month_col_order = (
        monthly[["month", "month_label"]]
        .drop_duplicates()
        .sort_values("month")["month_label"]
        .tolist()
    )

    z_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="rainfall_bias_score", aggfunc="mean", observed=False,
    ).reindex(columns=month_col_order)
    label_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="rainfall_label", aggfunc="first", observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)
    pct_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="rainfall_pct_of_normal", aggfunc="mean", observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)
    rain_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="rainfall_mm", aggfunc="sum", observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)
    normal_df = monthly.pivot_table(
        index="region_label", columns="month_label",
        values="normal_rainfall_mm", aggfunc="sum", observed=False,
    ).reindex(index=z_df.index, columns=month_col_order)

    pct_text = pct_df.round(0).astype("Int64").astype(str).replace("<NA>", "")
    text_grid = label_df.fillna("").astype(str) + "<br>" + pct_text + "%"
    text_grid = text_grid.replace("<br>%", "")

    customdata = np.stack(
        [
            label_df.fillna("").values,
            pct_df.round(1).values,
            rain_df.round(1).values,
            normal_df.round(1).values,
        ],
        axis=-1,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_df.values,
            x=z_df.columns,
            y=z_df.index.astype(str),
            text=text_grid.values,
            texttemplate="%{text}",
            customdata=customdata,
            zmin=-2,
            zmax=2,
            colorscale=[
                [0.00, "#2166ac"],
                [0.25, "#92c5de"],
                [0.50, "#f7f7f7"],
                [0.75, "#fdae61"],
                [1.00, "#b2182b"],
            ],
            colorbar=dict(
                title="Bias score",
                tickvals=[-2, -1, 0, 2],
                ticktext=["Excess moisture", "Wet volatility", "Normal", "Bullish dry"],
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Month: %{x}<br>"
                "Signal: %{customdata[0]}<br>"
                "Rainfall vs normal: %{customdata[1]}%<br>"
                "Rainfall: %{customdata[2]} mm<br>"
                "Normal rainfall: %{customdata[3]} mm<br>"
                "Bias score: %{z:.1f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=(
            f"{title}<br><sup>{start_date.date()} to {end_date.date()} | "
            "Dry = bullish bias, excessive moisture = volatility / downside risk</sup>"
        ),
        xaxis_title="Month",
        yaxis_title="Production region",
        template=_template(dark),
        height=500,
        margin=dict(l=170, r=80, t=90, b=70),
    )
    return fig


# =============================================================================
# HIGH-LEVEL DASHBOARD FIGURE BUILDER
# =============================================================================

def _load_wide_for_profile(
    profile_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_path: str | Path,
    force_refresh: bool,
    refresh_recent_days: int,
) -> pd.DataFrame:
    daily_long = load_weather_daily(
        profile_key,
        start,
        end,
        parameters=PROFILE_CONFIGS[profile_key].parameters,
        cache_path=cache_path,
        force_refresh=force_refresh,
        refresh_recent_days=refresh_recent_days,
    )
    return to_daily_wide(daily_long)


def make_weather_figures(
    commodity: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    force_refresh: bool = False,
    refresh_recent_days: int = 1,
    selected_areas: list[str] | None = None,
    dark: bool = False,
) -> dict[str, go.Figure]:
    """
    Return a dictionary of Plotly figures for the selected dashboard commodity.

    For Wheat, both winter and spring wheat views are returned.
    """
    profile_key = resolve_profile_key(commodity)

    if profile_key == "Wheat":
        figures: dict[str, go.Figure] = {}

        winter_cfg = PROFILE_CONFIGS["Wheat - Winter"]
        winter_start, winter_end = winter_cfg.default_range(None)
        if start_date is not None:
            winter_start = as_date(start_date)
        if end_date is not None:
            winter_end = as_date(end_date)
        winter_wide = _load_wide_for_profile(
            "Wheat - Winter",
            winter_start,
            winter_end,
            cache_path=cache_path,
            force_refresh=force_refresh,
            refresh_recent_days=refresh_recent_days,
        )
        if selected_areas:
            winter_wide = winter_wide[winter_wide["area"].isin(selected_areas)].copy()

        if not winter_wide.empty:
            figures["Winter wheat rainfall"] = make_season_cumulative_rainfall_chart(
                winter_wide,
                title="Winter Wheat Season-to-Date Cumulative Rainfall",
                start_date=winter_start,
                end_date=winter_end,
                dark=dark,
            )
            figures["Winter wheat min temp"] = make_winter_min_temp_chart(
                winter_wide,
                start_date=winter_start,
                end_date=winter_end,
                dark=dark,
            )

        spring_cfg = PROFILE_CONFIGS["Wheat - Spring"]
        spring_start, spring_end = spring_cfg.default_range(None)
        if start_date is not None:
            spring_start = as_date(start_date)
        if end_date is not None:
            spring_end = as_date(end_date)
        spring_wide = _load_wide_for_profile(
            "Wheat - Spring",
            spring_start,
            spring_end,
            cache_path=cache_path,
            force_refresh=force_refresh,
            refresh_recent_days=refresh_recent_days,
        )
        if selected_areas:
            spring_wide = spring_wide[spring_wide["area"].isin(selected_areas)].copy()

        if not spring_wide.empty:
            figures["Spring wheat rainfall anomaly"] = make_rainfall_anomaly_chart(
                spring_wide,
                title="Spring Wheat Rainfall Anomaly / Drought Risk",
                start_date=spring_start,
                end_date=spring_end,
                dark=dark,
            )
            figures["Spring wheat max temp"] = make_spring_max_temp_chart(
                spring_wide,
                start_date=spring_start,
                end_date=spring_end,
                dark=dark,
            )

        return figures

    if profile_key not in PROFILE_CONFIGS:
        return {}

    cfg = PROFILE_CONFIGS[profile_key]
    default_start, default_end = cfg.default_range(None)
    start = as_date(start_date) if start_date is not None else default_start
    end = as_date(end_date) if end_date is not None else default_end
    start, end = clamp_date_range(start, end)

    wide = _load_wide_for_profile(
        profile_key,
        start,
        end,
        cache_path=cache_path,
        force_refresh=force_refresh,
        refresh_recent_days=refresh_recent_days,
    )

    if selected_areas:
        wide = wide[wide["area"].isin(selected_areas)].copy()

    if wide.empty:
        return {}

    figures: dict[str, go.Figure] = {}

    if cfg.chart_profile == "cpo":
        figures["Cumulative rainfall"] = make_cumulative_rainfall_chart(
            wide,
            title="CPO Key Regions: Monthly Cumulative Rainfall",
            start_date=start,
            end_date=end,
            latest_month_only=True,
            dark=dark,
        )
        figures["Excessive rainfall"] = make_excessive_rainfall_heatmap(
            wide,
            title="CPO Excessive Rainfall Risk",
            start_date=start,
            end_date=end,
            dark=dark,
        )

    elif cfg.chart_profile == "soybeans":
        figures["Cumulative rainfall"] = make_cumulative_rainfall_chart(
            wide,
            title="Soybean Key Regions: Monthly Cumulative Rainfall",
            start_date=start,
            end_date=end,
            latest_month_only=True,
            dark=dark,
        )

    elif cfg.chart_profile == "rapeseed":
        region_order = [
            "Canada - Alberta",
            "Canada - Saskatchewan",
            "Canada - Manitoba",
            "Europe - France",
            "Europe - Germany",
            "Europe - Poland",
            "Europe - Romania",
        ]
        if selected_areas:
            region_order = [row for row in region_order if row.split(" - ", 1)[0] in selected_areas]

        figures["Excessive rainfall"] = make_excessive_rainfall_heatmap(
            wide,
            title="Rapeseed / Canola Excessive Rainfall Risk",
            start_date=start,
            end_date=end,
            region_order=region_order,
            dark=dark,
        )

    elif cfg.chart_profile == "coffee":
        figures["Cumulative rainfall"] = make_cumulative_rainfall_chart(
            wide,
            title="Coffee Key Regions: Monthly Cumulative Rainfall",
            start_date=start,
            end_date=end,
            latest_month_only=True,
            dark=dark,
        )
        figures["Heat / frost"] = make_coffee_temperature_chart(
            wide,
            start_date=start,
            end_date=end,
            dark=dark,
        )

    elif cfg.chart_profile == "cocoa":
        figures["Rainfall anomaly"] = make_rainfall_anomaly_chart(
            wide,
            title="Cocoa Key Regions: Monthly Rainfall Anomaly",
            start_date=start,
            end_date=end,
            dark=dark,
        )
        figures["Consecutive dry days"] = make_consecutive_dry_days_chart(
            wide,
            title="Cocoa Maximum Consecutive Dry Days",
            start_date=start,
            end_date=end,
            dark=dark,
        )

    elif cfg.chart_profile == "wheat_winter":
        figures["Season rainfall"] = make_season_cumulative_rainfall_chart(
            wide,
            title="Winter Wheat Season-to-Date Cumulative Rainfall",
            start_date=start,
            end_date=end,
            dark=dark,
        )
        figures["Min temperature"] = make_winter_min_temp_chart(
            wide,
            start_date=start,
            end_date=end,
            dark=dark,
        )

    elif cfg.chart_profile == "wheat_spring":
        figures["Rainfall anomaly"] = make_rainfall_anomaly_chart(
            wide,
            title="Spring Wheat Rainfall Anomaly / Drought Risk",
            start_date=start,
            end_date=end,
            dark=dark,
        )
        figures["Max temperature"] = make_spring_max_temp_chart(
            wide,
            start_date=start,
            end_date=end,
            dark=dark,
        )

    elif cfg.chart_profile == "generic_rain_temp":
        # Build region order from profile config so rows are stable
        region_order = [
            f"{area} - {zone}"
            for area in cfg.areas
            for zone in cfg.areas[area].zones
        ]
        if selected_areas:
            region_order = [r for r in region_order if r.split(" - ", 1)[0] in selected_areas]

        # Drought index (requires PRCP + PRCP_NORM)
        if "rainfall_mm" in wide.columns and "normal_rainfall_mm" in wide.columns:
            figures["Drought index"] = make_drought_index_heatmap(
                wide,
                title=f"{cfg.title} Futures Weather Risk: Drought Index by Region",
                start_date=start,
                end_date=end,
                region_order=region_order,
                dark=dark,
            )
            figures["Rainfall bias"] = make_rainfall_bias_heatmap(
                wide,
                title=f"{cfg.title} Futures Weather Bias: Rainfall Stress and Moisture Risk",
                start_date=start,
                end_date=end,
                region_order=region_order,
                dark=dark,
            )

        # Temperature stress index (requires TAVG; TMAX adds severe heat days)
        if "avg_temp_c" in wide.columns:
            figures["Temperature stress"] = make_temperature_stress_heatmap(
                wide,
                title=f"{cfg.title} Futures Weather Risk: Temperature Stress Index by Region",
                start_date=start,
                end_date=end,
                region_order=region_order,
                dark=dark,
            )

    else:
        figures["Cumulative rainfall"] = make_cumulative_rainfall_chart(
            wide,
            title=f"{cfg.title}: Monthly Cumulative Rainfall",
            start_date=start,
            end_date=end,
            latest_month_only=True,
            dark=dark,
        )
        if "normal_rainfall_mm" in wide.columns:
            figures["Rainfall anomaly"] = make_rainfall_anomaly_chart(
                wide,
                title=f"{cfg.title}: Monthly Rainfall Anomaly",
                start_date=start,
                end_date=end,
                dark=dark,
            )

    return figures


# =============================================================================
# SOIL MAP — WEATHERDESK GWI IMAGE API
# =============================================================================

_SOIL_IMAGE_ENDPOINT = "/d9293fc8-b396-4c3e-9166-2dee2012a6f2/services/gwi/v1/getimage"

_SOIL_PARAMETER_MAP: dict[str, str] = {
    "Soil Moisture - Subsoil (SWL2)": "SWL2",
    "Soil Moisture - Topsoil (SWL1)": "SWL1",
}

_SOIL_SUBPARAM_LABELS: dict[str, str] = {
    "CURR": "Current",
    "ANOM": "Anomaly vs Normal",
    "PCTL": "Percentile",
}

# Commodity → (default_region, default_crop)
_SOIL_DEFAULTS: dict[str, tuple[str, str]] = {
    "Crude Palm Oil":  ("Southeast Asia", "Palm Oil"),
    "Palm Olein":      ("Southeast Asia", "Palm Oil"),
    "Soybeans":        ("Brazil/Argentina", "Soybeans"),
    "Bean Oil":        ("Brazil/Argentina", "Soybeans"),
    "Soymeal":         ("Brazil/Argentina", "Soybeans"),
    "Canola/Rapeseed": ("Canada", "Rapeseed"),
    "Rapeseed Oil":    ("Europe", "Rapeseed"),
    "Corn":            ("United States", "Corn"),
    "Wheat":           ("United States", "Wheat"),
    "Cotton":          ("United States", "Cotton"),
    "Coffee":          ("Brazil/Argentina", "Coffee"),
    "Cocoa":           ("West Africa", "Cocoa"),
}


def _fetch_soil_image_url(
    region: str,
    crop: str,
    parameter_code: str,
    subparameter: str,
    target_date: str,
) -> str | None:
    region_code = REGION_MAP.get(region)
    crop_code   = CROP_MAP.get(crop, 0)
    if region_code is None:
        return None
    params: dict = dict(
        type="soil",
        date=target_date,
        region=region_code,
        parameter=parameter_code,
        subparameter=subparameter,
        metric=1,
        label=1,
        colorscale=1,
    )
    if crop_code:
        params["crop"] = crop_code
    try:
        r = requests.get(BASE_URL + _SOIL_IMAGE_ENDPOINT, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()["outputs"]["resource"]
    except Exception:
        pass
    return None


def _render_soil_map_tab(commodity: str) -> None:
    """Render the soil moisture map controls + image inside a Streamlit tab."""
    import streamlit as st
    from datetime import date as _date

    default_region, default_crop = _SOIL_DEFAULTS.get(commodity, ("Southeast Asia", "None"))

    # Crop and period are derived automatically — no extra filters needed
    soil_crop = default_crop
    soil_subparam = "CURR"

    c1, c2, c3= st.columns([2, 2, 2])
    with c1:
        region_keys = list(REGION_MAP.keys())
        soil_region = st.selectbox(
            "Region",
            region_keys,
            index=region_keys.index(default_region) if default_region in region_keys else 0,
            key=f"soil_region_{commodity}",
        )
    with c2:
        param_keys = list(_SOIL_PARAMETER_MAP.keys())
        soil_param_label = st.selectbox(
            "Parameter",
            param_keys,
            key=f"soil_param_{commodity}",
        )
        soil_param_code = _SOIL_PARAMETER_MAP[soil_param_label]

    with c3:
        soil_date = st.date_input(
            "Date",
            value=_date.today(),
            key=f"soil_date_{commodity}",
        ).strftime("%Y-%m-%d")

    with st.spinner("Fetching soil moisture map…"):
        img_url = _fetch_soil_image_url(
            soil_region, soil_crop, soil_param_code, soil_subparam, soil_date
        )

    if img_url:
        st.image(img_url, width="stretch")
        st.caption(
            f"Source: WeatherDesk GWI · {soil_region} · {soil_param_label} · "
            f"{_SOIL_SUBPARAM_LABELS[soil_subparam]} · {soil_date}"
        )
    else:
        st.warning(
            "Could not fetch soil moisture map. "
            "Check API credentials or try a different region / parameter."
        )


# =============================================================================
# STREAMLIT RENDERING
# =============================================================================

def render_weather_section(
    commodity: str,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    force_refresh: bool = False,
    refresh_recent_days: int = 1,
    dark: bool = True,
) -> None:
    """Render the weather section directly in Streamlit."""
    import streamlit as st

    st.markdown(
        '<div class="section-header">03 · Weather — Futures Risk View</div>',
        unsafe_allow_html=True,
    )

    resolved_profile = resolve_profile_key(commodity)

    # Dashboard has one generic "Wheat" commodity, but the weather module has
    # separate winter/spring risk views. Pick one view first so the chart does
    # not show every wheat geography at once.
    if resolved_profile == "Wheat":
        profile_for_chart = st.selectbox(
            "Wheat weather view",
            ["Wheat - Winter", "Wheat - Spring"],
            format_func=lambda x: "Winter wheat" if x == "Wheat - Winter" else "Spring wheat",
            key=f"wx_profile_{commodity}",
        )
    else:
        profile_for_chart = resolved_profile

    if profile_for_chart not in PROFILE_CONFIGS:
        st.info(f"No weather chart configured for **{commodity}**.")
        return

    profile_cfg = PROFILE_CONFIGS[profile_for_chart]
    default_start, default_end = profile_cfg.default_range(None)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        user_start = st.date_input(
            "Weather start date",
            value=as_date(start_date).date() if start_date else default_start.date(),
            key=f"wx_start_{commodity}_{profile_for_chart}",
        )
    with col_b:
        user_end = st.date_input(
            "Weather end date",
            value=as_date(end_date).date() if end_date else default_end.date(),
            key=f"wx_end_{commodity}_{profile_for_chart}",
        )

    area_options = list(profile_cfg.areas.keys())
    default_areas = area_options[:1]  # keep the first chart clean by default

    selected_areas = st.multiselect(
        "Country / region group",
        options=area_options,
        default=default_areas,
        key=f"wx_areas_{commodity}_{profile_for_chart}",
        help="Select one or more producing countries/regions to display. Keeping one selected makes the legend much easier to read.",
    )

    ctrl_a, ctrl_b = st.columns([1, 1])
    with ctrl_a:
        refresh = st.checkbox(
            "Force full refresh",
            value=force_refresh,
            key=f"wx_refresh_{commodity}_{profile_for_chart}",
        )
    with ctrl_b:
        recent_days = st.slider(
            "Refresh recent days",
            min_value=0,
            max_value=7,
            value=refresh_recent_days,
            key=f"wx_recent_{commodity}_{profile_for_chart}",
            help="Re-pulls recent days because today's feed may be revised.",
        )

    if not selected_areas:
        st.info("Select at least one country / region group.")
        return

    try:
        with st.spinner("Loading weather data…"):
            figures = make_weather_figures(
                profile_for_chart,
                user_start,
                user_end,
                cache_path=cache_path,
                force_refresh=refresh,
                refresh_recent_days=recent_days,
                selected_areas=selected_areas,
                dark=dark,
            )

        if not figures:
            st.info(f"No weather data returned for **{commodity}** / {', '.join(selected_areas)}.")
            return

        tab_labels = list(figures.keys()) + ["🌱 Soil Moisture"]
        all_tabs = st.tabs(tab_labels)

        for tab, (_, fig) in zip(all_tabs, figures.items()):
            with tab:
                st.plotly_chart(fig, width="stretch")

        with all_tabs[-1]:
            _render_soil_map_tab(commodity)

        st.caption(
            f"Weather cache: `{Path(cache_path)}` · Showing: {', '.join(selected_areas)}"
        )

    except Exception as exc:
        st.error(f"Weather data failed: {exc}")

# =============================================================================
# BACKWARD-COMPATIBLE HELPERS FOR EXISTING DASHBOARD.PY
# =============================================================================

def get_available_groups(commodity: str) -> list[str]:
    profile_key = resolve_profile_key(commodity)
    if profile_key == "Wheat":
        return ["Winter wheat", "Spring wheat"]
    if profile_key not in PROFILE_CONFIGS:
        return []
    return list(PROFILE_CONFIGS[profile_key].areas.keys())


def get_available_zones(commodity: str, group: str) -> list[str]:
    profile_key = resolve_profile_key(commodity)

    if profile_key == "Wheat":
        profile_key = "Wheat - Winter" if group == "Winter wheat" else "Wheat - Spring"

    if profile_key not in PROFILE_CONFIGS:
        return []

    cfg = PROFILE_CONFIGS[profile_key]
    if group not in cfg.areas:
        return []
    return cfg.areas[group].zones


def get_available_parameters(commodity: str, group: str | None = None) -> list[str]:
    profile_key = resolve_profile_key(commodity)

    if profile_key == "Wheat":
        if group == "Spring wheat":
            return PROFILE_CONFIGS["Wheat - Spring"].parameters
        return PROFILE_CONFIGS["Wheat - Winter"].parameters

    if profile_key not in PROFILE_CONFIGS:
        return []
    return PROFILE_CONFIGS[profile_key].parameters


def make_cached_weather_fetcher(
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    force_refresh: bool = False,
    refresh_recent_days: int = 2,
) -> Callable[[str, str | pd.Timestamp, str | pd.Timestamp], dict[str, Any]]:
    """
    Backward-compatible fetcher for the current dashboard.py weather block.

    It returns:
    {
        "groups": [
            {
                "label": "US",
                "zone_col": "region",
                "data": {"PRCP": dataframe, ...}
            }
        ]
    }
    """

    def _fetch(commodity: str, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> dict[str, Any]:
        profile_key = resolve_profile_key(commodity)

        if profile_key == "Wheat":
            # Old dashboard expects groups. Use both wheat profiles as groups.
            result_groups: list[dict[str, Any]] = []
            for label, wheat_profile in [("Winter wheat", "Wheat - Winter"), ("Spring wheat", "Wheat - Spring")]:
                params = PROFILE_CONFIGS[wheat_profile].parameters
                daily = load_weather_daily(
                    wheat_profile,
                    start_date,
                    end_date,
                    parameters=params,
                    cache_path=cache_path,
                    force_refresh=force_refresh,
                    refresh_recent_days=refresh_recent_days,
                )
                result_groups.append(
                    {
                        "label": label,
                        "zone_col": "region",
                        "data": _make_old_param_frames(daily),
                    }
                )
            return {"groups": result_groups}

        if profile_key not in PROFILE_CONFIGS:
            return {"groups": []}

        params = PROFILE_CONFIGS[profile_key].parameters
        daily = load_weather_daily(
            profile_key,
            start_date,
            end_date,
            parameters=params,
            cache_path=cache_path,
            force_refresh=force_refresh,
            refresh_recent_days=refresh_recent_days,
        )

        groups: list[dict[str, Any]] = []
        for area in PROFILE_CONFIGS[profile_key].areas:
            area_daily = daily[daily["area"].eq(area)].copy()
            groups.append(
                {
                    "label": area,
                    "zone_col": "region",
                    "data": _make_old_param_frames(area_daily),
                }
            )

        return {"groups": groups}

    return _fetch


def _make_old_param_frames(daily_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}

    if daily_long.empty:
        return frames

    for parameter, param_df in daily_long.groupby("parameter"):
        value_col = PARAM_VALUE_COL.get(parameter, parameter.lower())
        df = (
            param_df[["region", "date", "value", "station_count"]]
            .rename(columns={"value": value_col})
            .sort_values(["region", "date"])
            .copy()
        )
        if parameter == "PRCP":
            df[f"cumulative_{value_col}"] = df.groupby("region")[value_col].cumsum()
        frames[parameter] = df

    return frames