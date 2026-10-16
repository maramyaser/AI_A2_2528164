
"""Train/test splitting and reusable preprocessing pipelines."""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.validation import (
    validate_no_target_leakage,
    validate_schema,
)


SEED = 8164
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]

CATEGORICAL_FEATURES = ["season"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build transformations for numeric and categorical features."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def split_data(df: pd.DataFrame) -> dict:
    """
    Split the cleaned data into training and testing sets.

    The split is stratified by high_risk so that both sets retain
    a representative distribution of the classification target.
    """

    validate_schema(df)
    validate_no_target_leakage(FEATURE_COLUMNS)

    if df[["yield_tons", "high_risk"]].isna().any().any():
        raise ValueError(
            "Missing target values must be removed before splitting."
        )

    X = df[FEATURE_COLUMNS].copy()
    y_regression = df["yield_tons"].copy()
    y_classification = df["high_risk"].copy()

    (
        X_train,
        X_test,
        y_reg_train,
        y_reg_test,
        y_cls_train,
        y_cls_test,
    ) = train_test_split(
        X,
        y_regression,
        y_classification,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_classification,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "y_cls_train": y_cls_train,
        "y_cls_test": y_cls_test,
    }
