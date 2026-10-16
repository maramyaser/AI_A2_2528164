
import pandas as pd

REQUIRED_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
    "yield_tons",
    "high_risk",
]

TARGET_COLUMNS = ["yield_tons", "high_risk"]


def validate_schema(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(REQUIRED_COLUMNS)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected columns: {sorted(extra)}")


def validate_ranges(df: pd.DataFrame) -> None:
    ranges = {
        "rainfall_mm": (600, 1400),
        "avg_temp_c": (12, 26),
        "soil_moisture_pct": (20, 80),
        "fertilizer_kg": (0, 250),
        "field_size_ha": (0.2, 4.0),
        "cooperative_support": (0, 1),
        "yield_tons": (0.5, float("inf")),
        "high_risk": (0, 1),
    }

    for column, (minimum, maximum) in ranges.items():
        values = df[column].dropna()
        if not values.between(minimum, maximum).all():
            raise ValueError(f"Out-of-range values in {column}")

    if not df["season"].dropna().isin(["A", "B"]).all():
        raise ValueError("Season must be A or B.")


def validate_no_target_leakage(
    feature_columns: list[str],
) -> None:
    leaked = set(feature_columns).intersection(TARGET_COLUMNS)
    if leaked:
        raise ValueError(f"Target leakage detected: {sorted(leaked)}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    validate_schema(df)
    validate_ranges(df)

    duplicate_count = int(df.duplicated().sum())
    cleaned = df.drop_duplicates().copy()
    cleaned = cleaned.dropna(subset=TARGET_COLUMNS).copy()

    print(f"Original rows: {len(df)}")
    print(f"Duplicate rows detected: {duplicate_count}")
    print(f"Rows after duplicate removal: {len(cleaned)}")
    print(f"Missing values by column:\n{cleaned.isna().sum()}")

    return cleaned


def validate_csv(csv_path) -> pd.DataFrame:
    return clean_dataset(pd.read_csv(csv_path))
