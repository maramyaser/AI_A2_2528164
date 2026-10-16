
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


# Project settings
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

SEED = 8164
VERIFICATION_CODE = "AI-A2-8164"

DATA_PATH = DATA_DIR / "AI_A2_2528164.csv"
FINGERPRINT_PATH = ARTIFACTS_DIR / "data_fingerprint.txt"


def generate_data():
    """Generate the personalized synthetic potato farm dataset."""
    rng = np.random.default_rng(SEED)

    # The assignment specifies 240 + (seed modulo 41) base rows.
    base_rows = 240 + (SEED % 41)
    missing_count = round(0.05 * base_rows)

    # Generate farm features using the required distributions.
    rainfall = rng.uniform(600, 1400, base_rows)
    temperature = rng.uniform(12, 26, base_rows)
    soil_moisture = rng.uniform(20, 80, base_rows)
    fertilizer = rng.uniform(0, 250, base_rows)
    field_size = rng.uniform(0.2, 4.0, base_rows)

    season = rng.choice(["A", "B"], size=base_rows)
    support = rng.integers(0, 2, size=base_rows)

    # Season contribution required by the assignment.
    season_effect = np.where(season == "A", 0.50, -0.20)

    # Generate potato yield using the specified formula.
    noise = rng.normal(0, 0.80, base_rows)

    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil_moisture
        + 0.02 * fertilizer
        + 1.30 * field_size
        - 0.25 * np.abs(temperature - 18)
        + 0.70 * support
        + season_effect
        + noise
    )

    yield_tons = np.maximum(yield_tons, 0.50)

    df = pd.DataFrame({
        "rainfall_mm": rainfall,
        "avg_temp_c": temperature,
        "soil_moisture_pct": soil_moisture,
        "fertilizer_kg": fertilizer,
        "field_size_ha": field_size,
        "season": season,
        "cooperative_support": support,
        "yield_tons": yield_tons,
    })

    # High risk means yield per hectare is below its median.
    yield_per_hectare = df["yield_tons"] / df["field_size_ha"]
    median_yield_per_hectare = yield_per_hectare.median()

    df["high_risk"] = (
        yield_per_hectare < median_yield_per_hectare
    ).astype(int)

    # Introduce the required missing values at unique positions.
    soil_positions = rng.choice(
        base_rows, size=missing_count, replace=False
    )
    fertilizer_positions = rng.choice(
        base_rows, size=missing_count, replace=False
    )

    df.loc[soil_positions, "soil_moisture_pct"] = np.nan
    df.loc[fertilizer_positions, "fertilizer_kg"] = np.nan

    # Append three exact duplicate rows.
    df = pd.concat(
        [df, df.iloc[:3].copy()],
        ignore_index=True,
    )

    # Save reproducible CSV using four decimal places.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(DATA_PATH, index=False, float_format="%.4f")

    # Fingerprint the exact bytes of the saved CSV.
    csv_bytes = DATA_PATH.read_bytes()
    fingerprint = hashlib.sha256(csv_bytes).hexdigest()

    FINGERPRINT_PATH.write_text(
        f"Verification code: {VERIFICATION_CODE}\n"
        f"Seed: {SEED}\n"
        f"Base rows: {base_rows}\n"
        f"Final rows: {len(df)}\n"
        f"SHA-256: {fingerprint}\n",
        encoding="utf-8",
    )

    print("=" * 55)
    print("Musanze Harvest Intelligence Service")
    print(f"Verification code: {VERIFICATION_CODE}")
    print(f"Seed: {SEED}")
    print(f"Base rows: {base_rows}")
    print(f"Missing values introduced per column: {missing_count}")
    print("Duplicate rows appended: 3")
    print(f"Final dataset shape: {df.shape}")
    print(f"CSV saved to: {DATA_PATH}")
    print(f"Fingerprint saved to: {FINGERPRINT_PATH}")
    print(f"SHA-256: {fingerprint}")
    print("=" * 55)


if __name__ == "__main__":
    generate_data()
