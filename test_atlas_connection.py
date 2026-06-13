from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from atlas_client import AtlasClient

load_dotenv()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"✅ Saved {len(df)} rows -> {path}")

with AtlasClient.from_env() as client:
    print("✅ Atlas connected")

    # 1. Exchange products
    products = client.exchange_products(product_code="OCPO", active_only=False)
    save_csv(products, "exchange_products_ocpo.csv")

    # 2. FCPO futures list
    futures = client.futures(exchange_code="XKLS", active_only=True)
    save_csv(futures, "active_fcpo_futures.csv")

    # 3. OCPO options list
    options = client.options(exchange_code="XKLS", active_only=True)
    save_csv(options, "active_ocpo_options.csv")

    # 4. Futures bars
    bars = client.bars(
        "FCPOK26.XKLS.MYR.FUT",
        timeframe="1d",
        start="2026-03-02",
        end="2026-04-10",
    ).reset_index()
    save_csv(bars, "fcpo_bars.csv")

    # 5. Settlements
    settlements = client.settlements(
        "FCPOK26.XKLS.MYR.FUT",
        start="2026-03-02",
        end="2026-04-10",
    ).reset_index()
    save_csv(settlements, "fcpo_settlements.csv")

    # 6. Open interest
    oi = client.open_interest(
        "FCPOK26.XKLS.MYR.FUT",
        start="2026-03-02",
        end="2026-04-10",
    ).reset_index()
    save_csv(oi, "fcpo_open_interest.csv")

    # 7. Volatility surface
    vol_surface = client.volatility_surfaces(
        start="2026-03-06",
        end="2026-03-06 23:59:59",
    ).reset_index()
    vol_surface = vol_surface[
        vol_surface["canonical_symbol"].str.startswith("FCPO", na=False)
    ]
    save_csv(vol_surface, "vol_surface.csv")

    # 8. Vol curve
    vol_curve = vol_surface[
        vol_surface["canonical_symbol"].eq("FCPOK26.XKLS.MYR.FUT")
    ]
    save_csv(vol_curve, "vol_curve_fcpo_k26.csv")

    # 9. Yield curve
    yield_curve = client.rates(
        curve_name="MYR_YC",
        start="2026-03-06",
        end="2026-03-06 23:59:59",
    ).reset_index()
    save_csv(yield_curve, "yield_curve_myr.csv")

print("✅ All CSV data pulled.")
