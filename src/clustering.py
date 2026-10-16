
"""Cluster farms and compare K-Means models using silhouette scores."""
import joblib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "AI_A2_2528164.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

SEED = 8164

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]
CATEGORICAL_FEATURES = ["season"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    """Fit K-Means models for K=2..6 and save evaluation and profiles."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run data/generate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(FEATURES) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    # Remove exact duplicate rows and reset indexes.
    df = df.drop_duplicates().reset_index(drop=True)

    # Clustering uses farm conditions only, not yield or risk targets.
    X = df[FEATURES].copy()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    # Fit imputation, encoding and scaling on this clustering input.
    # There is no prediction target in unsupervised clustering.
    X_transformed = preprocessor.fit_transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_transformed = np.asarray(X_transformed)

    if len(df) < 7:
        raise ValueError("At least 7 rows are required to test K=2..6.")

    results = []
    fitted_models = {}

    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=10,
        )
        labels = model.fit_predict(X_transformed)
        score = silhouette_score(X_transformed, labels)

        results.append(
            {
                "k": k,
                "silhouette_score": float(score),
                "inertia": float(model.inertia_),
            }
        )
        fitted_models[k] = (model, labels)

    best_result = max(results, key=lambda item: item["silhouette_score"])
    selected_k = best_result["k"]
    selected_model, selected_labels = fitted_models[selected_k]
    joblib.dump(
    {
        "preprocessor": preprocessor,
        "model": selected_model,
        "features": FEATURES,
        "selected_k": selected_k,
        "seed": SEED,
    },
    ROOT / "models" / "clustering_model.joblib",
)

    clustered_df = df.copy()
    clustered_df["cluster"] = selected_labels

    # Plain-language profiles based on the original farm measurements.
    profiles = []
    for cluster_id in sorted(clustered_df["cluster"].unique()):
        group = clustered_df[clustered_df["cluster"] == cluster_id]

        numeric_means = {
            column: (
                float(group[column].mean())
                if group[column].notna().any()
                else None
            )
            for column in NUMERIC_FEATURES
        }

        season_counts = {
            str(key): int(value)
            for key, value in group["season"]
            .fillna("Unknown")
            .value_counts()
            .items()
        }

        profiles.append(
            {
                "cluster": int(cluster_id),
                "farm_count": int(len(group)),
                "numeric_feature_means": numeric_means,
                "season_counts": season_counts,
                "interpretation": (
                    "Compare this cluster's average rainfall, temperature, "
                    "soil moisture, fertilizer, field size and support with "
                    "the other clusters to describe its farm conditions."
                ),
            }
        )

    report = {
        "student_id": "ug2528164@ines.ac.rw",
        "seed": SEED,
        "verification_code": "AI-A2-8164",
        "features": FEATURES,
        "rows_after_duplicate_removal": int(len(df)),
        "k_comparison": results,
        "selected_k": int(selected_k),
        "selection_reason": (
            "Selected the K value with the highest silhouette score "
            "among K=2, 3, 4, 5 and 6."
        ),
        "profiles": profiles,
        "limitations": [
            "The dataset is synthetic and may not reflect real farms.",
            "Silhouette score measures separation in the encoded and scaled "
            "feature space, not agricultural value.",
            "Cluster labels are arbitrary identifiers, not risk ratings.",
        ],
    }

    report_path = ARTIFACTS_DIR / "clustering_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # PCA projects the transformed features into two dimensions for plotting.
    pca = PCA(n_components=2, random_state=SEED)
    coordinates = pca.fit_transform(X_transformed)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=selected_labels,
        cmap="viridis",
        alpha=0.8,
    )
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title(f"Farm Clusters (K={selected_k})")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()

    plot_path = PLOTS_DIR / "clusters.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Verification code: AI-A2-8164")
    print(f"Rows clustered: {len(df)}")
    print("\nSilhouette scores:")

    for result in results:
        print(
            f"K={result['k']}: "
            f"silhouette={result['silhouette_score']:.4f}, "
            f"inertia={result['inertia']:.4f}"
        )

    print(f"\nSelected K: {selected_k}")
    print("\nCluster profiles:")
    for profile in profiles:
        print(
            f"Cluster {profile['cluster']}: "
            f"{profile['farm_count']} farms; "
            f"mean features={profile['numeric_feature_means']}; "
            f"seasons={profile['season_counts']}"
        )

    print(f"\nReport saved to: {report_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
