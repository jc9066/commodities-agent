import pandas as pd
import streamlit as st
import plotly.graph_objects as go # type: ignore

@st.cache_data(ttl=60*60*24)
def load_oni():
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    oni = pd.read_csv(url, sep=r"\s+")
    oni = oni.rename(columns={
        "SEAS": "season", "YR": "year",
        "TOTAL": "sst",   "ANOM": "oni"
    })
    oni["year"] = oni["year"].astype(int)
    oni["oni"]  = pd.to_numeric(oni["oni"], errors="coerce")

    season_to_month = {
        "DJF":1,"JFM":2,"FMA":3,"MAM":4,"AMJ":5,"MJJ":6,
        "JJA":7,"JAS":8,"ASO":9,"SON":10,"OND":11,"NDJ":12
    }
    oni["month"] = oni["season"].map(season_to_month)
    oni["date"]  = pd.to_datetime(
        oni["year"].astype(str) + "-" + oni["month"].astype(str) + "-01"
    )
    oni = oni.sort_values("date").reset_index(drop=True)

    def classify_enso(x):
        if x >= 0.5:    return "El Niño"
        elif x <= -0.5: return "La Niña"
        else:           return "Neutral"

    oni["enso_signal"] = oni["oni"].apply(classify_enso)
    return oni


def get_enso_status(oni_df):
    """Returns (enso_status, enso_oni, enso_class) for the metric card."""
    latest      = oni_df.iloc[-1]
    oni_val     = latest["oni"]
    signal      = latest["enso_signal"]
    status      = f"{signal} Watch" if abs(oni_val) < 1.0 else signal
    css_class   = (
        "enso-elnino" if signal == "El Niño"  else
        "enso-lanina" if signal == "La Niña"  else
        "enso-neutral"
    )
    return status, f"{oni_val:+.2f}", css_class

def make_oni_chart(oni_df, start_year=1990):
    """
    NOAA-style ONI filled area chart.
    Red fill = El Niño (ONI > 0), Blue fill = La Niña (ONI < 0).
    start_year: controls how far back the chart goes.
    """
    df = oni_df[oni_df["date"].dt.year >= start_year].copy()

    fig = go.Figure()

    # ── Red fill (El Niño — above zero) ──────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["oni"].clip(lower=0),
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor="rgba(248, 81, 73, 0.45)",
        name="El Niño",
        hoverinfo="skip",
    ))

    # ── Blue fill (La Niña — below zero) ─────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["oni"].clip(upper=0),
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor="rgba(31, 111, 235, 0.35)",
        name="La Niña",
        hoverinfo="skip",
    ))

    # ── Black ONI line on top ─────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["oni"],
        mode="lines",
        line=dict(color="#c9d1d9", width=1.2),
        name="ONI",
        hovertemplate="%{x|%b %Y}<br>ONI: %{y:.2f}°C<extra></extra>",
    ))

    # ── Threshold lines ───────────────────────────────────────
    thresholds = [
        ( 0.5,  "#f85149", "Weak"),
        ( 1.0,  "#f85149", "Moderate"),
        ( 1.5,  "#f85149", "Strong"),
        ( 2.0,  "#f85149", "Very Strong"),
        (-0.5,  "#1f6feb", "Weak"),
        (-1.0,  "#1f6feb", "Moderate"),
        (-1.5,  "#1f6feb", "Strong"),
    ]
    for level, color, label in thresholds:
        fig.add_hline(
            y=level,
            line=dict(color=color, dash="dot", width=0.8),
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(color=color, size=9, family="IBM Plex Mono"),
        )

    # ── Zero baseline ─────────────────────────────────────────
    fig.add_hline(y=0, line=dict(color="#8b949e", width=1))

    fig.update_layout(
        title=dict(
            text=f"Oceanic Niño Index (ONI) · {start_year}–present",
            font=dict(family="IBM Plex Mono", size=13, color="#24292f"),
            x=0,
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f6f8fa",
        height=300,
        margin=dict(l=10, r=80, t=40, b=10),
        showlegend=True,
        legend=dict(
            orientation="h", x=0, y=1.12,
            font=dict(family="IBM Plex Mono", size=10),
        ),
        hovermode="x unified",
        font=dict(family="IBM Plex Mono", size=10),
        yaxis=dict(
            title="ONI Anomaly (°C)",
            gridcolor="#d0d7de",
            zeroline=False,
        ),
        xaxis=dict(gridcolor="#d0d7de"),
    )
    return fig