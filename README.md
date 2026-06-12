# 🌾 Commodities Intelligence Dashboard

A multi-page Streamlit application for agricultural and energy commodity market intelligence, built for SD Guthrie International Trading. Covers FX impact, supply & demand, weather, climate signals, trade policy, and global macro risk — all in a single internal research tool.

---

## Screenshots

> Add screenshots to `/docs/screenshots/` and reference them here.

---

## Architecture

```
dashboard.py                  ← Main page: commodity fundamentals
pages/
  01_macros.py                ← Global Macros intelligence page
utils/
  fx_live_monitor.py          ← FX impact analysis & live monitor
  psd.py                      ← USDA FAS PSD world & country S&D charts
  psd_aep.py                  ← USDA PSD Actual / Estimate / Projection chart
  psd_cache.py                ← Auto-refreshing feather cache (background thread)
  enso.py                     ← NOAA ONI climate indicator
  weather.py                  ← WeatherDesk GWI station weather by crop region
  policy.py                   ← Global Trade Alert (GTA) trade policy tracker
  macros.py                   ← Cross-asset macro data fetchers & renderers
data/
  *.feather                   ← Local cache files (auto-managed)
  macros_cache/               ← Macro module cache
  weather_cache/              ← Weather module cache
crop_calendars/               ← Static crop calendar images (PNG)
```

---

## Modules

### `dashboard.py` — Main Page

The root page. Filters by asset class, commodity, and exchange via a sidebar. Renders:

- **Key metrics row** — live commodity price (dummy), FX rate (live via Yahoo Finance), ENSO status badge, 30-day realized vol
- **Section 02 · FX Impact** — attribution waterfall, sparkline monitor, dual-axis overlay
- **Section 03 · Supply & Demand** — four-tab USDA FAS PSD panel (S&D balance, top countries, export breakdown, AEP chart)
- **Section 04 · ENSO ONI Index** — NOAA ONI filled area chart
- **Section 05 · Policy Tracker** — GTA trade interventions filtered by commodity
- **Section 08 · Crop Calendar** — static per-region crop calendar images
- **Section 09 · Risk Flows & IV** — dummy IV term structure + COT managed money net positioning

**Commodity universe:** Oilseeds (CPO, Palm Olein, Soybeans, Bean Oil, Soymeal, Canola/Rapeseed, Rapeseed Oil), Grains (Corn, Wheat), Softs (Sugar, Coffee, Cocoa, Cotton), Livestock (Lean Hogs, Live/Feeder Cattle), Seasonal Energy (GasOil, Natural Gas, Heating Oil, RBOB, Fuel Oil, LSFO), Non-Seasonal Energy (WTI, Brent, MS Crude).

---

### `pages/01_macros.py` — Global Macros

Standalone Streamlit page. Sidebar toggles visibility of eight sections:

| Section | Data source |
|---|---|
| 01 · Global Financial Markets | Yahoo Finance — VIX, SPX, DXY, UST10Y, Brent, Gold, Copper, MSCI EM |
| 02 · War & Armed Conflict | Placeholder |
| 03 · Sanctions | Finnhub news, keyword-filtered |
| 04 · Epidemics | WHO DON scraper (Playwright/BeautifulSoup) |
| 05 · Tariffs | Finnhub news, keyword-filtered |
| 06 · National Economic Indicators | Finnhub economic calendar |
| 07 · Central Bank Decisions | Placeholder |
| 08 · Credit & Default Contagion | Placeholder |

All live data is cached to `data/macros_cache/*.feather` with per-section TTLs.

---

### `utils/fx_live_monitor.py` — FX Impact Analysis

Pulls intraday FX prices from Yahoo Finance's chart API (not EOD closes). Exposes:

- `fetch_fx_history()` — cached OHLCV history for one or more pairs (1-hour TTL)
- `get_fx_snapshot()` — latest rate + 1d / 1w / 1m changes
- `get_fx_attribution()` — daily price-change decomposition into local move vs. FX translation
- `make_fx_monitor_chart()` — compact sparkline grid
- `make_fx_overlay_chart()` — commodity price + FX rate, dual-axis
- `make_fx_attribution_chart()` — waterfall bar chart
- `render_fx_section()` — drop-in Streamlit renderer

**Supported pairs:** USD/MYR, USD/CNY, USD/CAD, USD/BRL, USD/EUR, USD/GBP, USD/GHS, and their inverses.

---

### `utils/psd.py` + `utils/psd_aep.py` — USDA FAS PSD Supply & Demand

Reads from feather files managed by `psd_cache.py`. Falls back to live USDA FAS API if the cache isn't warm yet.

`psd.py` provides:
- `load_world_sd()` — world-level S&D for configured commodities
- `load_country_sd()` — country-level S&D
- `make_sd_balance_chart()` — line / bar / stacked area
- `make_top_countries_chart()` — ranked horizontal bar by attribute
- `make_export_stacked_chart()` — stacked bar export breakdown + global consumption line

`psd_aep.py` provides:
- `render_aep_tab()` — horizontal grouped bar chart showing Actual / Estimate / Projection for the top N countries, using the most recent available USDA snapshot month

**Attributes:** Production, Imports, Exports, Domestic Consumption, Beginning/Ending Stocks, Total Supply, Total Distribution.

---

### `utils/psd_cache.py` — Auto-Refreshing Feather Cache

Called once at startup via `ensure_psd_cache()`. Checks which feather files are missing or older than `CACHE_TTL_HOURS` (default 6h) and kicks off a background thread to refresh only stale tables. The dashboard renders immediately; fresh data is picked up on the next Streamlit rerun.

Files managed:

```
data/psd_world_sd.feather
data/psd_country_sd.feather
data/psd_aep_raw.feather
data/psd_lookup_countries.feather
data/psd_lookup_commodities.feather
data/psd_lookup_units.feather
data/psd_cache_meta.json
```

---

### `utils/enso.py` — ENSO / ONI Climate Indicator

- `load_oni()` — fetches NOAA CPC ONI ascii file (24h TTL), parses seasons → monthly dates, classifies El Niño / La Niña / Neutral
- `get_enso_status()` — returns display string, ONI value, and CSS class for the metric badge
- `make_oni_chart()` — NOAA-style filled area chart (red above zero, blue below), with threshold annotation lines

---

### `utils/weather.py` — Crop Weather by Region

Fetches WeatherDesk GWI station weather via API. Maintains a local feather cache at `data/weather_cache/weather_daily.feather`, fetching only missing or recent dates on subsequent runs.

Six commodities are pre-configured with growing region and parameter profiles. The main entry point is:

```python
render_weather_section(commodity, dark=True, cache_path="...", force_refresh=False)
```

---

### `utils/policy.py` — Trade Policy Tracker

Pulls trade intervention records from the Global Trade Alert (GTA) API, maps HS product codes to dashboard commodities, and renders a filtered policy table per selected commodity. Cache TTL: 6 hours.

---

### `utils/macros.py` — Global Macro Renderers

Data fetchers and chart builders for `pages/01_macros.py`. Sections use feather caches with per-section TTLs. News sections (sanctions, tariffs) use Finnhub with keyword filtering. Market data uses `yfinance`.

---

## Setup

### Requirements

```
streamlit
plotly
pandas
numpy
requests
yfinance
pillow
pyarrow          # feather I/O
playwright       # WHO DON scraper (optional)
beautifulsoup4
```

Install:

```bash
pip install streamlit plotly pandas numpy requests yfinance pillow pyarrow beautifulsoup4
playwright install chromium   # only needed for epidemic scraper
```

### API Keys

Configure the following in the relevant utils files (or via environment variables — refactoring recommended):

| Key | Module | Used for |
|---|---|---|
| `USDA FAS API key` | `psd_cache.py` | USDA FAS PSD supply/demand data |
| `Finnhub API key` | `macros.py` | News, economic calendar |
| `FRED API key` | `macros.py` | Macro indicators |
| `WeatherDesk GWI token` | `weather.py` | Crop weather by region |
| `GTA API` | `policy.py` | Trade policy interventions |

### Data Directory

Create the required directories before first run:

```bash
mkdir -p data/macros_cache data/weather_cache crop_calendars
```

Feather caches are written automatically on first run.

### Run

```bash
streamlit run dashboard.py
```

---

## Data & Cache Strategy

All external data is cached locally as feather files. Each module manages its own TTL:

| Module | TTL |
|---|---|
| USDA PSD (world/country S&D, AEP) | 6 hours |
| USDA PSD lookups | 24 hours |
| FX rates (Yahoo Finance) | 1 hour (intraday) |
| ENSO/ONI | 24 hours |
| Macro markets | 1 hour |
| Macro news (sanctions, tariffs) | 6 hours |
| Economic calendar | 6 hours |
| Trade policy (GTA) | 6 hours |

The USDA PSD cache uses a background thread (`threading.Thread`) so the dashboard is never blocked waiting for API calls on startup.

---

## Commodity Universe (USDA PSD)

```python
COMMODITY_CODES = {
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
```

---

## Notes

- Price data in `dashboard.py` is currently simulated (dummy random walks). Replace with live Bloomberg / Refinitiv / Atlas feeds for production use.
- IV term structure and COT data are also dummy. Connect to MDEX options or CFTC COT feeds to make these live.
- The `crop_calendars/` directory must be populated manually with PNG images named per the mapping in `dashboard.py`.
- `pages/01_macros.py` sections 02 (War), 07 (Central Banks), and 08 (Credit Contagion) are placeholders pending data source decisions.

---

## Internal Use Only

INTERNAL USE · SD Guthrie International Trading Pte Ltd · Commodities Intelligence
