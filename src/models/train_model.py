from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ARTIFACTS_DIR, FEATURE_COLUMNS_PATH, MODEL_PATH, PROCESSED_DATA_DIR  # noqa: E402
from src.models.evaluate_model import evaluate_classifier, format_metrics_table  # noqa: E402


SEASON_SLUG = "2023_24"
FEATURES_FILE = PROCESSED_DATA_DIR / f"features_{SEASON_SLUG}.csv"
TARGET_COLUMN = "home_win"
FEATURE_COLUMNS = [
    "home_last_5_win_pct",
    "home_last_10_point_diff",
    "home_avg_points_scored",
    "home_avg_points_allowed",
    "away_last_5_win_pct",
    "away_last_10_point_diff",
    "away_avg_points_scored",
    "away_avg_points_allowed",
    "home_rest_days",
    "away_rest_days",
    "home_back_to_back",
    "away_back_to_back",
]
RANDOM_STATE = 42
TRAIN_FRACTION = 0.80


def load_feature_dataset(input_path: Path = FEATURES_FILE) -> pd.DataFrame:
    """Load and validate the feature dataset."""
    games = pd.read_csv(input_path, dtype={"game_id": str})

    required_columns = {"game_id", "game_date", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing_columns = required_columns.difference(games.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Feature dataset is missing required columns: {missing}")

    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games[TARGET_COLUMN] = pd.to_numeric(games[TARGET_COLUMN], errors="coerce")

    for column in FEATURE_COLUMNS:
        games[column] = pd.to_numeric(games[column], errors="coerce")

    if games.loc[:, ["game_date", TARGET_COLUMN, *FEATURE_COLUMNS]].isna().any().any():
        missing_counts = games.loc[:, ["game_date", TARGET_COLUMN, *FEATURE_COLUMNS]].isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        raise ValueError(
            "Feature dataset contains missing model values:\n"
            f"{missing_counts.to_string()}"
        )

    invalid_target = ~games[TARGET_COLUMN].isin([0, 1])
    if invalid_target.any():
        raise ValueError(f"Found {invalid_target.sum()} rows with invalid {TARGET_COLUMN}.")

    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def split_train_test(
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(games) * TRAIN_FRACTION)
    if split_index <= 0 or split_index >= len(games):
        raise ValueError("Train/test split produced an empty train or test set.")

    train = games.iloc[:split_index]
    test = games.iloc[split_index:]

    x_train = train.loc[:, FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN].astype(int)
    x_test = test.loc[:, FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN].astype(int)

    return x_train, x_test, y_train, y_test


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[str, object, dict[str, dict[str, float]]]:
    models = build_models()
    metrics_by_model: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        metrics_by_model[model_name] = evaluate_classifier(model, x_test, y_test)

    best_model_name = min(
        metrics_by_model,
        key=lambda model_name: metrics_by_model[model_name]["log_loss"],
    )

    return best_model_name, models[best_model_name], metrics_by_model


def save_artifacts(
    model: object,
    model_path: Path = MODEL_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    with feature_columns_path.open("w", encoding="utf-8") as file:
        json.dump(FEATURE_COLUMNS, file, indent=2)


def main() -> None:
    print(f"Reading feature dataset from {FEATURES_FILE}")
    games = load_feature_dataset()

    x_train, x_test, y_train, y_test = split_train_test(games)
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")

    best_model_name, best_model, metrics_by_model = train_and_evaluate_models(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("\nModel metrics:")
    print(format_metrics_table(metrics_by_model))
    print(f"\nSelected best model: {best_model_name}")

    save_artifacts(best_model)
    print(f"Saved model artifact to {MODEL_PATH}")
    print(f"Saved feature columns to {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
