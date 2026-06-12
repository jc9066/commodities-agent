"""
utils/macros.py
================
Data fetchers and renderers for the Global Macros page.

Sections
--------
1. Global Financial Markets  (yfinance — VIX, SPX, DXY, UST10Y, Brent, Gold)
2. Epidemics                 (WHO DON scraper via Playwright)
3. Sanctions                 (Finnhub news, keyword-filtered)
4. War                       (placeholder — no live data yet)
5. Tariffs                   (Finnhub news, keyword-filtered)
6. National Economic Indicators  (Finnhub economic calendar, filtered)
7. Central Bank Decisions    (placeholder — no live data yet)
8. Global Credit & Default Contagion  (placeholder — no live data yet)

Caching
-------
All live data is persisted to Feather files under  data/macros_cache/.
TTLs are enforced by comparing file mtime to a max-age constant.

Usage (inside Streamlit page)
------------------------------
    from utils.macros import render_macro_section
    render_macro_section("Global Financial Markets")
"""

from __future__ import annotations

import re
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, date
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
CACHE_DIR = Path("data/macros_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FINNHUB_KEY = "d8lnv49r01qnkjl7pp7gd8lnv49r01qnkjl7pp80"
FRED_KEY    = "65089965a39d120644a7310569c55c8b"

# Feather file paths
FP_MARKETS   = CACHE_DIR / "global_markets.feather"
FP_EPIDEMICS = CACHE_DIR / "epidemics.feather"
FP_SANCTIONS = CACHE_DIR / "sanctions.feather"
FP_TARIFFS   = CACHE_DIR / "tariffs.feather"
FP_ECON_CAL  = CACHE_DIR / "econ_calendar.feather"

# Cache max ages (seconds)
TTL_MARKETS  = 3600        # 1 hour
TTL_NEWS     = 6 * 3600    # 6 hours
TTL_ECON_CAL = 6 * 3600    # 6 hours

# ──────────────────────────────────────────────────────────────
# COLOUR PALETTE  (matches dashboard dark theme)
# ──────────────────────────────────────────────────────────────
C_BLUE   = "#58a6ff"
C_GREEN  = "#3fb950"
C_RED    = "#f85149"
C_ORANGE = "#ffa657"
C_PURPLE = "#d2a8ff"
C_GREY   = "#8b949e"
BG_MAIN  = "#0d1117"
BG_CARD  = "#161b22"
BORDER   = "#21262d"

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=BG_MAIN,
    plot_bgcolor=BG_CARD,
    font=dict(family="IBM Plex Mono", size=11),
)

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _is_stale(fp: Path, ttl: int) -> bool:
    if not fp.exists():
        return True
    age = time.time() - fp.stat().st_mtime
    return age > ttl


def _save_feather(df: pd.DataFrame, fp: Path) -> None:
    # Feather requires string column names and specific dtypes
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    # Convert object cols with mixed types to string for feather safety
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
    df.reset_index(drop=True).to_feather(fp)


def _load_feather(fp: Path) -> pd.DataFrame:
    try:
        return pd.read_feather(fp)
    except Exception:
        return pd.DataFrame()


def _section_header(label: str) -> None:
    st.markdown(
        f'<div class="section-header">{label}</div>',
        unsafe_allow_html=True,
    )


def _placeholder_card(title: str, message: str = "No live data source connected yet.") -> None:
    st.markdown(
        f"""
        <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;
                    padding:20px 24px;margin:8px 0;">
            <div style="color:{C_GREY};font-family:'IBM Plex Mono',monospace;
                        font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
                        margin-bottom:8px;">{title}</div>
            <div style="color:{C_GREY};font-size:13px;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# 1.  GLOBAL FINANCIAL MARKETS
# ──────────────────────────────────────────────────────────────

MARKET_TICKERS = {
    "VIX":        ("^VIX",      "CBOE VIX",            C_RED),
    "S&P 500":    ("^GSPC",     "S&P 500",              C_GREEN),
    "DXY":        ("DX-Y.NYB",  "US Dollar Index",      C_BLUE),
    "UST 10Y":    ("^TNX",      "US 10Y Yield (%)",     C_PURPLE),
    "Brent":      ("BZ=F",      "Brent Crude (USD/bbl)",C_ORANGE),
    "Gold":       ("GC=F",      "Gold (USD/oz)",        "#f0d060"),
    "Copper":     ("HG=F",      "Copper (USc/lb)",      "#e88c34"),
    "MSCI EM":    ("EEM",       "iShares MSCI EM ETF",  C_GREY),
}


@st.cache_data(ttl=TTL_MARKETS, show_spinner=False)
def load_market_data(period: str = "1y") -> pd.DataFrame:
    """
    Fetch daily close for all MARKET_TICKERS via yfinance.
    Falls back to feather cache if yfinance fails.
    Returns wide DataFrame indexed by date, columns = display names.
    """
    try:
        import yfinance as yf
        symbols = [v[0] for v in MARKET_TICKERS.values()]
        raw = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]
        # rename symbols → display names
        rev = {v[0]: k for k, v in MARKET_TICKERS.items()}
        raw = raw.rename(columns=rev)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.dropna(how="all")
        _save_feather(raw.reset_index().rename(columns={"index": "date"}), FP_MARKETS)
        return raw
    except Exception as e:
        logger.warning(f"yfinance failed: {e}")
        cached = _load_feather(FP_MARKETS)
        if not cached.empty:
            cached["date"] = pd.to_datetime(cached["date"])
            return cached.set_index("date")
        return pd.DataFrame()


def _hex_to_rgba(hex_color: str, alpha: float = 0.09) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _spark_fig(series: pd.Series, color: str, label: str) -> go.Figure:
    s = series.dropna()
    chg = (s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) >= 2 else 0
    chg_color = C_GREEN if chg >= 0 else C_RED
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values,
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(color),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        height=90,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, gridcolor=BORDER),
    )
    return fig, s.iloc[-1], chg, chg_color


def render_global_markets(period: str = "1y") -> None:
    _section_header("01 · Global Financial Markets")

    with st.spinner("Loading market data…"):
        mdf = load_market_data(period)

    if mdf.empty:
        st.error("Market data unavailable.")
        return

    # ── snapshot metric row ──────────────────────────────────
    cols = st.columns(len(MARKET_TICKERS))
    for col, (display_name, (ticker, desc, color)) in zip(cols, MARKET_TICKERS.items()):
        if display_name not in mdf.columns:
            continue
        s = mdf[display_name].dropna()
        if s.empty:
            continue
        last = s.iloc[-1]
        prev = s.iloc[-2] if len(s) >= 2 else s.iloc[-1]
        chg  = (last / prev - 1) * 100
        arrow = "▲" if chg >= 0 else "▼"
        delta_cls = "metric-delta-pos" if chg >= 0 else "metric-delta-neg"
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{display_name}</div>
            <div class="metric-value" style="font-size:20px;">{last:,.2f}</div>
            <div class="{delta_cls}">{arrow} {chg:+.2f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

    # ── sparkline grid: 4 per row ────────────────────────────
    items = [(k, v) for k, v in MARKET_TICKERS.items() if k in mdf.columns]
    for row_start in range(0, len(items), 4):
        row_items = items[row_start:row_start + 4]
        spark_cols = st.columns(4)
        for scol, (display_name, (ticker, desc, color)) in zip(spark_cols, row_items):
            s = mdf[display_name].dropna()
            if s.empty:
                continue
            fig, last, chg, chg_color = _spark_fig(s, color, display_name)
            scol.markdown(
                f'<div style="color:{C_GREY};font-family:IBM Plex Mono,monospace;'
                f'font-size:10px;text-transform:uppercase;letter-spacing:0.08em;">'
                f'{desc}</div>',
                unsafe_allow_html=True,
            )
            scol.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── VIX regime + SPX together ────────────────────────────
    if "VIX" in mdf.columns and "S&P 500" in mdf.columns:
        st.markdown("---")
        fig2 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.4, 0.6],
            vertical_spacing=0.04,
        )
        vix = mdf["VIX"].dropna()
        spx = mdf["S&P 500"].dropna()

        fig2.add_trace(go.Scatter(
            x=vix.index, y=vix.values,
            mode="lines", line=dict(color=C_RED, width=1.5),
            name="VIX", fill="tozeroy", fillcolor=_hex_to_rgba(C_RED),
        ), row=1, col=1)

        # VIX regime bands
        for level, label, col_ in [(20, "Low Stress", C_GREEN), (30, "Elevated", C_ORANGE), (40, "Crisis", C_RED)]:
            fig2.add_hline(y=level, line_dash="dot", line_color=col_, line_width=0.8,
                           annotation_text=label,
                           annotation_font=dict(size=9, color=col_),
                           row=1, col=1)

        fig2.add_trace(go.Scatter(
            x=spx.index, y=spx.values,
            mode="lines", line=dict(color=C_GREEN, width=1.8),
            name="S&P 500",
        ), row=2, col=1)

        fig2.update_layout(
            **PLOTLY_BASE,
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(orientation="h", y=1.04, x=0),
        )
        fig2.update_yaxes(gridcolor=BORDER)
        st.plotly_chart(fig2, width="stretch")
    st.caption("Source: Yahoo Finance · delayed ~15 min")


# ──────────────────────────────────────────────────────────────
# 2.  EPIDEMICS
# ──────────────────────────────────────────────────────────────

WHO_BASE = "https://www.who.int"
WHO_LIST = "https://www.who.int/emergencies/disease-outbreak-news"
DATE_PAT = re.compile(r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(ttl=TTL_NEWS, show_spinner=False)
def load_epidemics(year: int | None = None) -> pd.DataFrame:
    """Scrape WHO DON page. Falls back to feather cache."""
    if year is None:
        year = datetime.today().year

    # check feather freshness
    if not _is_stale(FP_EPIDEMICS, TTL_NEWS):
        df = _load_feather(FP_EPIDEMICS)
        if not df.empty:
            df["PublicationDate"] = pd.to_datetime(df["PublicationDate"], errors="coerce")
            return df

    try:
        from playwright.sync_api import sync_playwright
        from bs4 import BeautifulSoup

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(WHO_LIST, wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_timeout(8_000)
            for _ in range(8):
                page.mouse.wheel(0, 4000)
                page.wait_for_timeout(1000)
            html = page.content()
            browser.close()

        soup = BeautifulSoup(html, "html.parser")
        rows, seen = [], set()
        for link in soup.select('a[href*="/emergencies/disease-outbreak-news/item/"]'):
            href = link.get("href", "")
            url  = urljoin(WHO_BASE, href)
            if url in seen:
                continue
            seen.add(url)
            text = _clean(link.get_text(" ", strip=True))
            dm   = DATE_PAT.search(text)
            pub  = dm.group(1) if dm else None
            title = text.split("|", 1)[-1].strip() if "|" in text else text
            rows.append({"PublicationDate": pub, "Title": title, "URL": url})

        df = pd.DataFrame(rows)
        df["PublicationDate"] = pd.to_datetime(df["PublicationDate"], format="%d %B %Y", errors="coerce")
        df = (df[df["PublicationDate"].dt.year == year]
                .drop_duplicates("URL")
                .sort_values("PublicationDate", ascending=False)
                .reset_index(drop=True))
        _save_feather(df, FP_EPIDEMICS)
        return df

    except Exception as e:
        logger.warning(f"WHO scrape failed: {e}")
        cached = _load_feather(FP_EPIDEMICS)
        if not cached.empty:
            cached["PublicationDate"] = pd.to_datetime(cached["PublicationDate"], errors="coerce")
            return cached
        return pd.DataFrame(columns=["PublicationDate", "Title", "URL"])


def render_epidemics() -> None:
    _section_header("04 · Epidemics · WHO Disease Outbreak News")

    year_sel = st.selectbox("Year", list(range(datetime.today().year, 2018, -1)),
                            key="epi_year", label_visibility="collapsed")
    with st.spinner("Loading WHO DON data…"):
        df = load_epidemics(year_sel)

    if df.empty:
        st.info("No WHO outbreak data loaded. Playwright may not be installed, or check cache.")
        return

    st.caption(f"{len(df)} outbreaks · {year_sel}")

    # HTML table with links
    rows_html = ""
    for _, row in df.iterrows():
        date_str = row["PublicationDate"].strftime("%d %b %Y") if pd.notna(row["PublicationDate"]) else "—"
        title    = str(row["Title"])[:120]
        url      = str(row["URL"])
        rows_html += (
            f'<tr>'
            f'<td style="white-space:nowrap;color:{C_GREY};padding:5px 10px;">{date_str}</td>'
            f'<td style="padding:5px 10px;">'
            f'<a href="{url}" target="_blank" style="color:{C_BLUE};text-decoration:none;">{title}</a>'
            f'</td>'
            f'</tr>'
        )

    st.markdown(f"""
    <div style="max-height:420px;overflow-y:auto;border:1px solid {BORDER};border-radius:6px;">
    <table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:12px;">
        <thead>
            <tr style="background:{BG_CARD};position:sticky;top:0;">
                <th style="text-align:left;padding:6px 10px;color:{C_GREY};border-bottom:1px solid {BORDER};">Date</th>
                <th style="text-align:left;padding:6px 10px;color:{C_GREY};border-bottom:1px solid {BORDER};">Outbreak Report</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Source: WHO Disease Outbreak News · who.int/emergencies/disease-outbreak-news")


# ──────────────────────────────────────────────────────────────
# 3.  SANCTIONS  &  5. TARIFFS  (Finnhub news, keyword-filtered)
# ──────────────────────────────────────────────────────────────

SECTION_NEWS_CONFIG = {
    "sanctions": {
        "fp":       FP_SANCTIONS,
        "keywords": ["sanction", "sanctioned", "embargo", "OFAC", "export control", "blacklist",
                     "asset freeze", "designated", "SDN list", "trade restriction"],
        "label":    "03 · Sanctions",
        "header_num": "03",
        "color":    C_ORANGE,
    },
    "tariffs": {
        "fp":       FP_TARIFFS,
        "keywords": ["tariff", "trade war", "import duty", "customs duty", "Section 301",
                     "Section 232", "countervailing duty", "anti-dumping", "trade barrier",
                     "trade dispute", "WTO", "retaliatory", "levy"],
        "label":    "05 · Tariffs",
        "header_num": "05",
        "color":    C_PURPLE,
    },
    "war": {
        "fp":       CACHE_DIR / "war.feather",
        "keywords": ["war", "warfare", "invasion", "airstrike", "missile", "military operation",
                     "ceasefire", "offensive", "troops", "combat", "conflict", "shelling",
                     "drone attack", "armed forces", "bombardment", "front line"],
        "label":    "02 · War & Armed Conflict",
        "header_num": "02",
        "color":    C_RED,
    },
}


FINNHUB_NEWS_MAX_PAGES = 25   # ≈ 500–750 articles per refresh

@st.cache_data(ttl=TTL_NEWS, show_spinner=False)
def _fetch_finnhub_news() -> pd.DataFrame:
    """
    Fetch general news from Finnhub using paginated min_id walk.
    Collects up to FINNHUB_NEWS_MAX_PAGES batches (~20-30 items each),
    giving ~500-750 articles per refresh — far more coverage than a
    single call.  Shared by sanctions, tariffs, and war sections.
    """
    try:
        import finnhub
        client = finnhub.Client(api_key=FINNHUB_KEY)
        all_news: list = []
        min_id = 0
        for _ in range(FINNHUB_NEWS_MAX_PAGES):
            batch = client.general_news("general", min_id=min_id)
            if not batch:
                break
            all_news.extend(batch)
            min_id = min(item["id"] for item in batch)  # walk backwards

        df = pd.DataFrame(all_news)
        if df.empty:
            return df
        df = df[["id", "datetime", "headline", "summary", "source", "url"]].copy()
        df = df.drop_duplicates("id").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
        df["headline"] = df["headline"].str.replace(r"\s+", " ", regex=True).str.strip()
        df["summary"]  = df["summary"].str.replace(r"\s+", " ", regex=True).str.strip()
        df = df.drop_duplicates("headline").sort_values("datetime", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Finnhub news fetch failed: {e}")
        return pd.DataFrame()


def _filter_news(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    pat = "|".join(re.escape(k) for k in keywords)
    mask = (
        df["headline"].str.lower().str.contains(pat, na=False) |
        df["summary"].str.lower().str.contains(pat, na=False)
    )
    return df[mask].reset_index(drop=True)


def _render_news_table(df: pd.DataFrame, color: str, fp: Path) -> None:
    if df.empty:
        st.info("No matching news items found.")
        return

    _save_feather(df, fp)
    st.caption(f"{len(df)} items")

    rows_html = ""
    for _, row in df.head(60).iterrows():
        dt_str  = row["datetime"].strftime("%d %b %Y %H:%M") if pd.notna(row["datetime"]) else "—"
        headline = str(row["headline"])[:140]
        source   = str(row.get("source", ""))
        url      = str(row.get("url", "#"))
        rows_html += (
            f'<tr>'
            f'<td style="white-space:nowrap;color:{C_GREY};padding:5px 8px;font-size:11px;">{dt_str}</td>'
            f'<td style="padding:5px 8px;">'
            f'<a href="{url}" target="_blank" style="color:{color};text-decoration:none;">{headline}</a>'
            f'</td>'
            f'<td style="color:{C_GREY};padding:5px 8px;font-size:10px;white-space:nowrap;">{source}</td>'
            f'</tr>'
        )

    st.markdown(f"""
    <div style="max-height:380px;overflow-y:auto;border:1px solid {BORDER};border-radius:6px;">
    <table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:12px;">
        <thead>
            <tr style="background:{BG_CARD};position:sticky;top:0;">
                <th style="text-align:left;padding:6px 8px;color:{C_GREY};border-bottom:1px solid {BORDER};">Time</th>
                <th style="text-align:left;padding:6px 8px;color:{C_GREY};border-bottom:1px solid {BORDER};">Headline</th>
                <th style="text-align:left;padding:6px 8px;color:{C_GREY};border-bottom:1px solid {BORDER};">Source</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Source: Finnhub General News (keyword-filtered)")


def render_sanctions() -> None:
    cfg = SECTION_NEWS_CONFIG["sanctions"]
    _section_header(cfg["label"])
    with st.spinner("Fetching news…"):
        raw = _fetch_finnhub_news()
    df  = _filter_news(raw, cfg["keywords"])
    _render_news_table(df, cfg["color"], cfg["fp"])


def render_tariffs() -> None:
    cfg = SECTION_NEWS_CONFIG["tariffs"]
    _section_header(cfg["label"])
    with st.spinner("Fetching news…"):
        raw = _fetch_finnhub_news()
    df  = _filter_news(raw, cfg["keywords"])
    _render_news_table(df, cfg["color"], cfg["fp"])


# ──────────────────────────────────────────────────────────────
# 4.  WAR  (placeholder)
# ──────────────────────────────────────────────────────────────

def render_war() -> None:
    cfg = SECTION_NEWS_CONFIG["war"]
    _section_header(cfg["label"])
    with st.spinner("Fetching news…"):
        raw = _fetch_finnhub_news()
    df  = _filter_news(raw, cfg["keywords"])
    _render_news_table(df, cfg["color"], cfg["fp"])


# ──────────────────────────────────────────────────────────────
# 6.  NATIONAL ECONOMIC INDICATORS  (ForexFactory calendar)
# ──────────────────────────────────────────────────────────────

FF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FF_COUNTRIES = ["USD", "EUR", "GBP", "SGD", "CAD", "BRL", "CNY", "INR", "IDR", "MYR", "All", "Tent."]

FF_KEYWORDS = [
    "cpi", "gdp", "interest rate", "industrial production", "fomc", "auction",
    "confidence", "ppi", "fed", "pmi", "trade", "imports", "exports",
    "jobless claims", "unemployment", "meetings", "crude", "retail", "jolts",
    "employment", "non-farm", "trump", "federal", "ism", "fed's", "inventories", "oil", "OPEC", "M2", "consumer", "inflation"
]

IMPACT_ORDER = {"★★★": 0, "★★☆": 1, "★☆☆": 2, "—": 3}


def _scrape_forexfactory(week: str = "This Week") -> pd.DataFrame:
    """Scrape ForexFactory calendar for a given week. Returns cleaned DataFrame."""
    base_url = "https://www.forexfactory.com/calendar"
    if week == "This Week":        url = base_url
    elif week == "Next Week":      url = f"{base_url}?week=next"
    elif week == "Last Week":      url = f"{base_url}?week=last"
    else:                     url = f"{base_url}?week={week}"

    r = requests.get(url, headers=FF_HEADERS, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("tr.calendar__row--grey, tr.calendar__row")

    data = []
    current_date = ""
    for row in rows:
        date_cell = row.select_one(".calendar__date")
        if date_cell and date_cell.get_text(strip=True):
            raw_date = date_cell.get_text(strip=True)
            try:
                current_date = datetime.strptime(
                    raw_date + f" {datetime.today().year}", "%a%b %d %Y"
                ).strftime("%Y-%m-%d")
            except Exception:
                current_date = raw_date

        if not row.select_one(".calendar__event"):
            continue

        def safe(selector):
            el = row.select_one(selector)
            return el.get_text(strip=True) if el else ""

        impact_el = row.select_one(".calendar__impact span")
        impact_class = " ".join(impact_el.get("class", [])) if impact_el else ""
        if "red" in impact_class:    impact = "high"
        elif "ora" in impact_class:  impact = "medium"
        elif "yel" in impact_class:  impact = "low"
        else:                        impact = ""

        data.append({
            "date":     current_date,
            "time":     safe(".calendar__time"),
            "country":  safe(".calendar__currency"),
            "event":    safe(".calendar__event-title"),
            "impact":   impact,
            "actual":   safe(".calendar__actual"),
            "forecast": safe(".calendar__forecast"),
            "previous": safe(".calendar__previous"),
        })

    df = pd.DataFrame(data)
    if df.empty:
        return df

    df = df[df["event"] != ""].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["impact"] = df["impact"].map({"low": "★☆☆", "medium": "★★☆", "high": "★★★"}).fillna("—")

    def _parse_val(val):
        if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
            return "—"
        s = str(val).strip()
        try:
            suffix = next((u for u in ["%", "B", "M", "K"] if u in s), "")
            cleaned = s.replace("%", "").replace("B", "").replace("M", "").replace("K", "").replace(",", "").strip()
            return f"{round(float(cleaned), 2)}{suffix}"
        except ValueError:
            return s

    for col in ["actual", "forecast", "previous"]:
        df[col] = df[col].apply(_parse_val)

    df = df.sort_values(["date", "time"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=TTL_ECON_CAL, show_spinner=False)
def load_econ_calendar(week: str = "This Week") -> pd.DataFrame:
    """
    Load ForexFactory economic calendar for the requested week.
    Falls back to feather cache on network failure.
    Applies keyword + country filters matching the notebook logic.
    """
    try:
        df = _scrape_forexfactory(week)
        if df.empty:
            raise ValueError("empty scrape result")

        # apply default filters
        pattern = "|".join(FF_KEYWORDS)
        df = df[df["event"].str.lower().str.contains(pattern, na=False)]
        df = df[df["country"].isin(FF_COUNTRIES)]
        df = df.reset_index(drop=True)

        _save_feather(df.assign(date=df["date"].astype(str)), FP_ECON_CAL)
        return df

    except Exception as e:
        logger.warning(f"ForexFactory scrape failed: {e}")
        cached = _load_feather(FP_ECON_CAL)
        if not cached.empty:
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.date
            return cached
        return pd.DataFrame()


def render_econ_calendar() -> None:
    _section_header("06 · National Economic Indicators · Economic Calendar")

    # ── Week selector ────────────────────────────────────────
    wc1, wc2 = st.columns([2, 4])
    with wc1:
        week_sel = st.selectbox(
            "Week",
            ["This Week", "Last Week", "Next Week"],
            key="macro_ff_week",
            label_visibility="collapsed",
        )

    with st.spinner("Loading ForexFactory calendar…"):
        df = load_econ_calendar(week_sel)

    if df.empty:
        st.error("Calendar unavailable — ForexFactory may be blocking the request or no events matched.")
        return

    # ── Additional filters ───────────────────────────────────
    with wc2:
        sel_impact = st.multiselect(
            "Impact", ["★★★", "★★☆", "★☆☆"],
            default=["★★★", "★★☆"],
            key="macro_econ_impact",
        )

    fc1, fc2 = st.columns([3, 3])
    with fc1:
        all_countries = sorted(df["country"].dropna().unique().tolist())
        sel_countries = st.multiselect(
            "Countries", all_countries,
            default=all_countries,
            key="macro_econ_countries",
        )
    with fc2:
        kw_filter = st.text_input(
            "Additional keyword filter", value="",
            key="macro_econ_kw", placeholder="e.g. cpi, gdp",
        )

    # apply UI filters on top of already-filtered df
    fdf = df.copy()
    if sel_countries:
        fdf = fdf[fdf["country"].isin(sel_countries)]
    if sel_impact:
        fdf = fdf[fdf["impact"].isin(sel_impact)]
    if kw_filter.strip():
        pat = "|".join(re.escape(k.strip()) for k in kw_filter.split(",") if k.strip())
        fdf = fdf[fdf["event"].str.lower().str.contains(pat, case=False, na=False)]

    st.caption(f"{len(fdf)} events · {week_sel} week · source: ForexFactory")

    if fdf.empty:
        st.info("No events match the current filter.")
        return

    # ── Render grouped by date ───────────────────────────────
    IMPACT_COLOR = {"★★★": C_RED, "★★☆": C_ORANGE, "★☆☆": C_GREY, "—": C_GREY}

    rows_html = ""
    prev_date = None
    for _, row in fdf.iterrows():
        d = row["date"]
        if d != prev_date:
            try:
                date_label = pd.Timestamp(d).strftime("%A, %d %b %Y")
            except Exception:
                date_label = str(d)
            rows_html += (
                f'<tr><td colspan="7" style="background:{BORDER};color:{C_GREY};'
                f'font-family:IBM Plex Mono,monospace;font-size:10px;letter-spacing:0.1em;'
                f'text-transform:uppercase;padding:4px 8px;">{date_label}</td></tr>'
            )
            prev_date = d

        imp_col = IMPACT_COLOR.get(str(row.get("impact", "—")), C_GREY)
        rows_html += (
            f'<tr>'
            f'<td style="color:{C_GREY};padding:4px 8px;font-size:11px;white-space:nowrap;">{row.get("time","—")}</td>'
            f'<td style="color:{C_BLUE};padding:4px 8px;font-size:11px;font-weight:600;">{row.get("country","—")}</td>'
            f'<td style="padding:4px 8px;font-size:12px;">{row.get("event","—")}</td>'
            f'<td style="color:{imp_col};padding:4px 8px;font-size:11px;text-align:center;">{row.get("impact","—")}</td>'
            f'<td style="color:{C_GREEN};padding:4px 8px;font-size:11px;text-align:right;">{row.get("actual","—")}</td>'
            f'<td style="color:{C_GREY};padding:4px 8px;font-size:11px;text-align:right;">{row.get("forecast","—")}</td>'
            f'<td style="color:{C_GREY};padding:4px 8px;font-size:11px;text-align:right;">{row.get("previous","—")}</td>'
            f'</tr>'
        )

    st.markdown(f"""
    <div style="max-height:520px;overflow-y:auto;border:1px solid {BORDER};border-radius:6px;">
    <table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;">
        <thead>
            <tr style="background:{BG_CARD};position:sticky;top:0;z-index:1;">
                <th style="text-align:left;padding:6px 8px;color:{C_GREY};border-bottom:1px solid {BORDER};font-size:11px;">Time</th>
                <th style="text-align:left;padding:6px 8px;color:{C_GREY};border-bottom:1px solid {BORDER};font-size:11px;">Country</th>
                <th style="text-align:left;padding:6px 8px;color:{C_GREY};border-bottom:1px solid {BORDER};font-size:11px;">Event</th>
                <th style="text-align:center;padding:6px 8px;color:{C_GREY};border-bottom:1px solid {BORDER};font-size:11px;">Impact</th>
                <th style="text-align:right;padding:6px 8px;color:{C_GREY};border-bottom:1px solid {BORDER};font-size:11px;">Actual</th>
                <th style="text-align:right;padding:6px 8px;color:{C_GREY};border-bottom:1px solid {BORDER};font-size:11px;">Forecast</th>
                <th style="text-align:right;padding:6px 8px;color:{C_GREY};border-bottom:1px solid {BORDER};font-size:11px;">Previous</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Source: ForexFactory · keyword + country filtered")


# ──────────────────────────────────────────────────────────────
# 7.  CENTRAL BANK DECISIONS  (placeholder)
# ──────────────────────────────────────────────────────────────

def render_central_banks() -> None:
    _section_header("07 · Central Bank Decisions")
    _placeholder_card(
        "No live data source connected",
        "Planned: Fed FOMC minutes (FRED), ECB decisions, BNM/BI/RBI rate decisions. "
        "Add loader to  utils/macros.py  →  load_cb_decisions().",
    )


# ──────────────────────────────────────────────────────────────
# 8.  GLOBAL CREDIT & DEFAULT CONTAGION  (placeholder)
# ──────────────────────────────────────────────────────────────

def render_credit_contagion() -> None:
    _section_header("08 · Global Credit & Default Contagion")
    _placeholder_card(
        "No live data source connected",
        "Planned: sovereign CDS spreads (FRED / Bloomberg), EM hard-currency bond spreads (EMBI), "
        "BIS cross-border banking stats. Add loader to  utils/macros.py  →  load_credit_data().",
    )