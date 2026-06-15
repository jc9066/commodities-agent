import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from atlas_client import AtlasClient

load_dotenv()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Atlas Data Report", layout="wide")
st.title("Atlas Data Report")


@st.cache_resource
def get_client():
    return AtlasClient.from_env()


@st.cache_data(ttl=3600)
def load_symbols():
    client = AtlasClient.from_env()
    data = client.futures(exchange_code="XKLS", active_only=True)
    df = to_df(data)

    if "canonical_symbol" in df.columns:
        return sorted(df["canonical_symbol"].dropna().unique())

    return ["FCPOK26.XKLS.MYR.FUT"]


def to_df(data):
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.reset_index()
    return pd.DataFrame(data)


try:
    client = get_client()
    st.success("Atlas connected")
except Exception as e:
    st.error(f"Atlas connection failed: {e}")
    st.stop()


with st.sidebar:
    st.header("Filters")

    date_range = st.date_input(
        "Date Range",
        value=(
            datetime.date.today() - datetime.timedelta(days=30),
            datetime.date.today(),
        ),
    )

    if len(date_range) != 2:
        st.warning("Please select start date and end date.")
        st.stop()

    start_date, end_date = date_range
    as_of = end_date

    symbol = st.selectbox(
        "FCPO Contract",
        load_symbols(),
    )

st.write("Start:", start_date)
st.write("End:", end_date)
st.write("As of:", as_of)
st.write("Symbol:", symbol)

st.divider()

if st.button("Pull Atlas Data"):
    tabs = st.tabs([
        "Products",
        "Futures",
        "Options",
        "Bars",
        "Settlements",
        "Open Interest",
        "Vol Surface",
        "Yield Curve",
    ])

    with tabs[0]:
        try:
            df = to_df(client.exchange_products(product_code="OCPO", active_only=False))
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Products CSV", df.to_csv(index=False), "exchange_products.csv")
        except Exception as e:
            st.error(e)

    with tabs[1]:
        try:
            df = to_df(client.futures(exchange_code="XKLS", active_only=True))
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Futures CSV", df.to_csv(index=False), "futures.csv")
        except Exception as e:
            st.error(e)

    with tabs[2]:
        try:
            df = to_df(client.options(exchange_code="XKLS", active_only=True))
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Options CSV", df.to_csv(index=False), "options.csv")
        except Exception as e:
            st.error(e)

    with tabs[3]:
        try:
            df = to_df(client.bars(
                symbol,
                timeframe="1d",
                start=str(start_date),
                end=str(end_date),
            ))
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Bars CSV", df.to_csv(index=False), "bars.csv")
        except Exception as e:
            st.error(e)

    with tabs[4]:
        try:
            df = to_df(client.settlements(
                symbol,
                start=str(start_date),
                end=str(end_date),
            ))
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Settlements CSV", df.to_csv(index=False), "settlements.csv")
        except Exception as e:
            st.error(e)

    with tabs[5]:
        try:
            df = to_df(client.open_interest(
                symbol,
                start=str(start_date),
                end=str(end_date),
            ))
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Open Interest CSV", df.to_csv(index=False), "open_interest.csv")
        except Exception as e:
            st.error(e)

    with tabs[6]:
        try:
            df = to_df(client.volatility_surfaces(
                start=str(as_of),
                end=str(as_of) + " 23:59:59",
            ))

            if not df.empty and "canonical_symbol" in df.columns:
                df = df[df["canonical_symbol"].astype(str).str.startswith("FCPO")]

            st.dataframe(df, use_container_width=True)
            st.download_button("Download Vol Surface CSV", df.to_csv(index=False), "vol_surface.csv")
        except Exception as e:
            st.error(e)

    with tabs[7]:
        try:
            df = to_df(client.rates(
                curve_name="MYR_YC",
                start=str(as_of),
                end=str(as_of) + " 23:59:59",
            ))
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Yield Curve CSV", df.to_csv(index=False), "yield_curve.csv")
        except Exception as e:
            st.error(e)