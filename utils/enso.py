import pandas as pd
import streamlit as st
import plotly.graph_objects as go  # type: ignore
import requests
from io import StringIO


RONI_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/"

SEASON_TO_MONTH = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12,
}
SEASON_COLS = list(SEASON_TO_MONTH.keys())


def _normalise_roni_columns(columns):
    """Convert CPC HTML table column names to: year, DJF, JFM, ..."""
    clean_cols = []

    for col in columns:
        # pd.read_html can return a MultiIndex depending on the HTML structure.
        if isinstance(col, tuple):
            col = " ".join(str(part) for part in col if str(part).lower() != "nan")

        token = str(col).strip().split()[0].upper()

        if token == "YEAR":
            clean_cols.append("year")
        elif token in SEASON_COLS:
            clean_cols.append(token)
        else:
            clean_cols.append(str(col).strip())

    return clean_cols


@st.cache_data(ttl=60 * 60 * 24)
def load_oni():
    """
    Load NOAA CPC Relative Oceanic Nino Index (RONI) data.

    Kept as load_oni() so dashboard.py does not need import changes.
    The returned dataframe includes both `roni` and `oni` columns:
    - `roni` = the true RONI value from the CPC RONI page
    - `oni`  = compatibility alias used by the existing dashboard/chart code
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
    }

    response = requests.get(RONI_URL, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    roni_wide = None
    for table in tables:
        t = table.copy()
        t.columns = _normalise_roni_columns(t.columns)

        # Fallback for cases where the first row is used as data instead of header.
        if "year" not in t.columns and not t.empty:
            first_row = [str(x).strip().split()[0].upper() for x in t.iloc[0].tolist()]
            if "YEAR" in first_row and any(season in first_row for season in SEASON_COLS):
                t.columns = _normalise_roni_columns(first_row)
                t = t.iloc[1:].reset_index(drop=True)

        season_count = sum(col in t.columns for col in SEASON_COLS)
        if "year" in t.columns and season_count >= 6:
            roni_wide = t[["year"] + [col for col in SEASON_COLS if col in t.columns]].copy()
            break

    if roni_wide is None:
        raise ValueError("Could not find the RONI data table on the CPC page.")

    roni_wide["year"] = pd.to_numeric(roni_wide["year"], errors="coerce")
    roni_wide = roni_wide.dropna(subset=["year"]).copy()
    roni_wide["year"] = roni_wide["year"].astype(int)

    # Make sure every season exists, even if the latest year only has partial data.
    for season in SEASON_COLS:
        if season not in roni_wide.columns:
            roni_wide[season] = pd.NA

        roni_wide[season] = pd.to_numeric(
            roni_wide[season]
            .astype(str)
            .str.replace("\u2212", "-", regex=False)  # handle Unicode minus if present
            .str.strip(),
            errors="coerce",
        )

    roni = roni_wide.melt(
        id_vars="year",
        value_vars=SEASON_COLS,
        var_name="season",
        value_name="roni",
    )

    roni = roni.dropna(subset=["roni"]).copy()
    roni["month"] = roni["season"].map(SEASON_TO_MONTH)
    roni["date"] = pd.to_datetime(
        roni["year"].astype(str) + "-" + roni["month"].astype(str) + "-01",
        errors="coerce",
    )
    roni = roni.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    def classify_enso(x):
        if x >= 0.5:
            return "El Niño"
        elif x <= -0.5:
            return "La Niña"
        else:
            return "Neutral"

    roni["enso_signal"] = roni["roni"].apply(classify_enso)

    # Compatibility with the existing dashboard code that expects a column named `oni`.
    roni["oni"] = roni["roni"]
    roni["index_name"] = "RONI"
    roni["source_url"] = RONI_URL

    return roni[[
        "date",
        "year",
        "month",
        "season",
        "roni",
        "oni",
        "enso_signal",
        "index_name",
        "source_url",
    ]]


def get_enso_status(oni_df):
    """Returns (enso_status, enso_oni, enso_class) for the metric card."""
    if oni_df is None or oni_df.empty:
        return "No Data", "N/A", "enso-neutral"

    value_col = "roni" if "roni" in oni_df.columns else "oni"
    df = oni_df.dropna(subset=[value_col]).copy()

    if df.empty:
        return "No Data", "N/A", "enso-neutral"

    latest = df.iloc[-1]
    index_val = latest[value_col]
    signal = latest["enso_signal"]

    if signal == "Neutral":
        status = "Neutral"
    elif abs(index_val) < 1.0:
        status = f"{signal} Watch"
    else:
        status = signal

    css_class = (
        "enso-elnino" if signal == "El Niño" else
        "enso-lanina" if signal == "La Niña" else
        "enso-neutral"
    )
    return status, f"{index_val:+.2f}", css_class


def make_oni_chart(oni_df, start_year=1990):
    """
    NOAA-style RONI filled area chart.
    Red fill = El Nino/RONI > 0, Blue fill = La Nina/RONI < 0.

    Function name is kept as make_oni_chart() so dashboard.py does not need import changes.
    """
    fig = go.Figure()

    if oni_df is None or oni_df.empty:
        fig.update_layout(
            title=dict(
                text="Relative Oceanic Niño Index (RONI)",
                font=dict(family="IBM Plex Mono", size=13, color="#24292f"),
                x=0,
            ),
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f6f8fa",
            height=300,
            margin=dict(l=10, r=80, t=40, b=10),
            annotations=[dict(
                text="No RONI data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(family="IBM Plex Mono", size=12),
            )],
        )
        return fig

    value_col = "roni" if "roni" in oni_df.columns else "oni"
    df = oni_df[oni_df["date"].dt.year >= start_year].copy()
    df = df.dropna(subset=[value_col])

    # ── Red fill (El Niño — above zero) ──────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df[value_col].clip(lower=0),
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
        y=df[value_col].clip(upper=0),
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor="rgba(31, 111, 235, 0.35)",
        name="La Niña",
        hoverinfo="skip",
    ))

    # ── RONI line on top ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df[value_col],
        mode="lines",
        line=dict(color="#c9d1d9", width=1.2),
        name="RONI",
        hovertemplate="%{x|%b %Y}<br>RONI: %{y:.2f}°C<extra></extra>",
    ))

    # ── Threshold lines ───────────────────────────────────────
    thresholds = [
        (0.5, "#f85149", "Weak"),
        (1.0, "#f85149", "Moderate"),
        (1.5, "#f85149", "Strong"),
        (2.0, "#f85149", "Very Strong"),
        (-0.5, "#1f6feb", "Weak"),
        (-1.0, "#1f6feb", "Moderate"),
        (-1.5, "#1f6feb", "Strong"),
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
            text=f"Relative Oceanic Niño Index (RONI) · {start_year}–present",
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
            orientation="h",
            x=0,
            y=1.12,
            font=dict(family="IBM Plex Mono", size=10),
        ),
        hovermode="x unified",
        font=dict(family="IBM Plex Mono", size=10),
        yaxis=dict(
            title="RONI Anomaly (°C)",
            gridcolor="#d0d7de",
            zeroline=False,
        ),
        xaxis=dict(gridcolor="#d0d7de"),
    )
    return fig
