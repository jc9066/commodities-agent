"""
utils/markets.py
================
Global Financial Markets section — extracted from utils/macros.py.

Layout
------
· Left panel  (2/3 width): 6 tickers in a 2-per-row sparkline grid
· Right panel (1/3 width): VIX + S&P 500 dual-panel chart with price/% labels
· VIX and S&P 500 are NOT shown as individual sparklines — they only appear in the combo chart

Theme: light mode — white background, dark text, muted grid.

Usage
-----
    from utils.markets import render_global_markets
    render_global_markets(period="1y")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# CACHE
# ──────────────────────────────────────────────────────────────
CACHE_DIR   = Path("data/macros_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FP_MARKETS  = CACHE_DIR / "global_markets.feather"
TTL_MARKETS = 3_600  # 1 hour

# ──────────────────────────────────────────────────────────────
# LIGHT-MODE PALETTE
# ──────────────────────────────────────────────────────────────
C_UP      = "#16a34a"
C_DOWN    = "#dc2626"
C_VIX     = "#dc2626"
C_SPX     = "#16a34a"
C_DXY     = "#2563eb"
C_YIELD   = "#7c3aed"
C_BRENT   = "#ea580c"
C_GOLD    = "#ca8a04"
C_COPPER  = "#b45309"
C_EM      = "#6b7280"

BG        = "#ffffff"
BG_CARD   = "#f8fafc"
BORDER    = "#e2e8f0"
TEXT_MAIN = "#0f172a"
TEXT_MUTE = "#64748b"
GRID_COL  = "#f1f5f9"

PLOTLY_LIGHT = dict(
    template="plotly_white",
    paper_bgcolor=BG,
    plot_bgcolor=BG_CARD,
    font=dict(family="IBM Plex Mono, monospace", size=11, color=TEXT_MAIN),
)

# ──────────────────────────────────────────────────────────────
# TICKER REGISTRY
# VIX and S&P 500 excluded from sparkline grid — combo chart only
# ──────────────────────────────────────────────────────────────
MARKET_TICKERS: dict[str, tuple[str, str, str]] = {
    "VIX":      ("^VIX",     "CBOE VIX",             C_VIX),
    "S&P 500":  ("^GSPC",    "S&P 500",               C_SPX),
    "DXY":      ("DX-Y.NYB", "US Dollar Index",       C_DXY),
    "UST 10Y":  ("^TNX",     "US 10Y Yield (%)",      C_YIELD),
    "Brent":    ("BZ=F",     "Brent Crude (USD/bbl)", C_BRENT),
    "Gold":     ("GC=F",     "Gold (USD/oz)",         C_GOLD),
    "Copper":   ("HG=F",     "Copper (USc/lb)",       C_COPPER),
    "MSCI EM":  ("EEM",      "iShares MSCI EM ETF",   C_EM),
}

# Tickers shown only in the combo chart — excluded from the sparkline grid
COMBO_ONLY = {"VIX", "S&P 500"}

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.10) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _save_feather(df: pd.DataFrame, fp: Path) -> None:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
    df.reset_index(drop=True).to_feather(fp)


def _load_feather(fp: Path) -> pd.DataFrame:
    try:
        return pd.read_feather(fp)
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# DATA LOADER
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=TTL_MARKETS, show_spinner=False)
def load_market_data(period: str = "1y") -> pd.DataFrame:
    try:
        import yfinance as yf
        symbols = [v[0] for v in MARKET_TICKERS.values()]
        raw = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]
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


# ──────────────────────────────────────────────────────────────
# SPARKLINE  (compact, axes visible)
# ──────────────────────────────────────────────────────────────

def _spark_fig(series: pd.Series, color: str) -> go.Figure:
    s = series.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index,
        y=s.values,
        mode="lines",
        line=dict(color=color, width=1.6),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(color, 0.07),
        hovertemplate="%{x|%b %d '%y}<br>%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LIGHT,
        height=90,
        margin=dict(l=38, r=6, t=2, b=22),
        showlegend=False,
        xaxis=dict(
            showticklabels=True,
            tickformat="%b '%y",
            tickfont=dict(size=8, color=TEXT_MUTE),
            tickangle=0,
            nticks=4,
            showgrid=False,
            zeroline=False,
            linecolor=BORDER,
            linewidth=1,
        ),
        yaxis=dict(
            showticklabels=True,
            tickfont=dict(size=8, color=TEXT_MUTE),
            nticks=3,
            gridcolor=GRID_COL,
            gridwidth=1,
            zeroline=False,
            linecolor=BORDER,
            linewidth=1,
            tickformat=",.0f",
        ),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# VIX + SPX COMBO CHART  — with inline price/% annotations
# ──────────────────────────────────────────────────────────────

def _vix_spx_fig(
    vix: pd.Series,
    spx: pd.Series,
    vix_last: float,
    vix_chg: float,
    spx_last: float,
    spx_chg: float,
) -> go.Figure:
    """
    Dual-panel chart (VIX top, SPX bottom) with price + % change
    rendered as annotation labels in the chart — matching the reference
    image style where price and badge sit above the trace.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.40, 0.60],
        vertical_spacing=0.08,
    )

    # ── VIX ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=vix.index, y=vix.values,
        mode="lines",
        line=dict(color=C_VIX, width=1.5),
        name="VIX",
        fill="tozeroy",
        fillcolor=_hex_to_rgba(C_VIX, 0.07),
        hovertemplate="%{x|%b %d '%y}<br>VIX: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # VIX regime bands
    for level, label, col_ in [
        (20, "Low Stress", C_UP),
        (30, "Elevated",   C_BRENT),
        (40, "Crisis",     C_DOWN),
    ]:
        fig.add_hline(
            y=level, line_dash="dot", line_color=col_, line_width=0.8,
            annotation_text=label, annotation_position="right",
            annotation_font=dict(size=8, color=col_),
            row=1, col=1,
        )

    # ── SPX ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=spx.index, y=spx.values,
        mode="lines",
        line=dict(color=C_SPX, width=1.8),
        name="S&P 500",
        hovertemplate="%{x|%b %d '%y}<br>SPX: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    # ── Axes ─────────────────────────────────────────────────
    axis_common = dict(
        showgrid=True, gridcolor=GRID_COL, gridwidth=1,
        zeroline=False, linecolor=BORDER,
        tickfont=dict(size=8, color=TEXT_MUTE),
    )
    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)
    fig.update_xaxes(tickformat="%b '%y", nticks=5, row=2, col=1)

    # ── Price + % badge annotations (top-left of each panel) ─
    vix_chg_sym  = "▲" if vix_chg >= 0 else "▼"
    spx_chg_sym  = "▲" if spx_chg >= 0 else "▼"
    vix_chg_col  = C_UP if vix_chg >= 0 else C_DOWN
    spx_chg_col  = C_UP if spx_chg >= 0 else C_DOWN

    fig.add_annotation(
        text=(
            f"<b style='font-size:14px'>{vix_last:.2f}</b>"
            f"  <span style='color:{vix_chg_col}'>{vix_chg_sym} {vix_chg:+.2f}%</span>"
        ),
        xref="paper", yref="paper",
        x=0.01, y=0.995,
        showarrow=False,
        align="left",
        font=dict(family="IBM Plex Mono, monospace", size=11, color=TEXT_MAIN),
        bgcolor=BG, borderpad=3,
    )
    fig.add_annotation(
        text=(
            f"<b style='font-size:14px'>{spx_last:,.2f}</b>"
            f"  <span style='color:{spx_chg_col}'>{spx_chg_sym} {spx_chg:+.2f}%</span>"
        ),
        xref="paper", yref="paper",
        x=0.01, y=0.415,
        showarrow=False,
        align="left",
        font=dict(family="IBM Plex Mono, monospace", size=11, color=TEXT_MAIN),
        bgcolor=BG, borderpad=3,
    )

    fig.update_layout(
        **PLOTLY_LIGHT,
        height=320,
        margin=dict(l=46, r=72, t=28, b=28),
        showlegend=True,
        legend=dict(
            orientation="h", x=0, y=1.06,
            font=dict(size=9, color=TEXT_MUTE),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# SPARKLINE CARD  (label + chart + price/badge below)
# ──────────────────────────────────────────────────────────────

def _render_spark_card(col, display_name: str, desc: str, color: str, s: pd.Series) -> None:
    last = s.iloc[-1]
    prev = s.iloc[-2] if len(s) >= 2 else last
    chg  = (last / prev - 1) * 100
    chg_color = C_UP if chg >= 0 else C_DOWN
    badge_bg  = "#dcfce7" if chg >= 0 else "#fee2e2"
    arrow     = "▲" if chg >= 0 else "▼"

    with col:
        # ticker badge + description
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:1px;">'
            f'<span style="background:{color};color:#fff;font-family:IBM Plex Mono,monospace;'
            f'font-size:9px;font-weight:600;padding:2px 6px;border-radius:4px;">'
            f'{display_name}</span>'
            f'<span style="color:{TEXT_MUTE};font-size:10px;font-family:IBM Plex Mono,monospace;">'
            f'{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # sparkline
        st.plotly_chart(
            _spark_fig(s, color),
            width="stretch",
            config={"displayModeBar": False},
        )
        # price + % badge
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:8px;margin-top:-8px;margin-bottom:6px;">'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:17px;'
            f'font-weight:700;color:{TEXT_MAIN};">{last:,.2f}</span>'
            f'<span style="background:{badge_bg};color:{chg_color};'
            f'font-family:IBM Plex Mono,monospace;font-size:10px;font-weight:600;'
            f'padding:2px 6px;border-radius:4px;">{arrow} {chg:+.2f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────
# MAIN RENDERER
# ──────────────────────────────────────────────────────────────

def render_global_markets(period: str = "1y") -> None:
    """
    Render the Global Financial Markets section.

    Layout
    ------
    Left col (2/3)  : 6 ticker sparklines in a 2-per-row grid
    Right col (1/3) : VIX + S&P 500 dual-panel combo chart
                      with price/% labels overlaid per panel
    """
    st.markdown(
        '<div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{TEXT_MUTE};font-family:IBM Plex Mono,monospace;'
        f'border-bottom:1px solid {BORDER};padding-bottom:6px;margin-bottom:14px;">'
        "01 · Global Financial Markets</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading market data…"):
        mdf = load_market_data(period)

    if mdf.empty:
        st.error("Market data unavailable.")
        return

    # Split tickers: sparkline grid vs combo chart
    spark_items = [
        (k, v) for k, v in MARKET_TICKERS.items()
        if k in mdf.columns and k not in COMBO_ONLY
    ]

    # ── outer split: left sparklines | right combo chart ─────
    left_col, right_col = st.columns([2, 2], gap="medium")

    # ── LEFT: 3-per-row sparkline grid ───────────────────────
    with left_col:
        for row_start in range(0, len(spark_items), 3):
            trio = spark_items[row_start : row_start + 3]
            c1, c2, c3 = st.columns(3, gap="small")
            cols_trio = [c1, c2, c3]
            for i, (display_name, (ticker, desc, color)) in enumerate(trio):
                s = mdf[display_name].dropna()
                if s.empty:
                    continue
                _render_spark_card(cols_trio[i], display_name, desc, color, s)

    # ── RIGHT: VIX + SPX combo chart ─────────────────────────
    with right_col:
        if "VIX" in mdf.columns and "S&P 500" in mdf.columns:
            vix = mdf["VIX"].dropna()
            spx = mdf["S&P 500"].dropna()

            def _last_chg(s):
                last = s.iloc[-1]
                prev = s.iloc[-2] if len(s) >= 2 else last
                return last, (last / prev - 1) * 100

            vix_last, vix_chg = _last_chg(vix)
            spx_last, spx_chg = _last_chg(spx)

            fig = _vix_spx_fig(vix, spx, vix_last, vix_chg, spx_last, spx_chg)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.caption(
        f'<span style="color:{TEXT_MUTE};font-size:10px;font-family:IBM Plex Mono,monospace;">'
        f'Source: Yahoo Finance · delayed ~15 min</span>',
        unsafe_allow_html=True,
    )