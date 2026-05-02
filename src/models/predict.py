from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEATURE_COLUMNS_PATH, MODEL_PATH  # noqa: E402


def load_model(model_path: Path = MODEL_PATH):
    """Load the trained model artifact."""
    return joblib.load(model_path)


def load_feature_columns(feature_columns_path: Path = FEATURE_COLUMNS_PATH) -> list[str]:
    """Load the ordered feature columns used during training."""
    with feature_columns_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def predict_home_win_probability(
    row: pd.DataFrame | pd.Series,
    model_path: Path = MODEL_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
) -> float:
    """Predict P(home_team wins) for one game row."""
    model = load_model(model_path)
    feature_columns = load_feature_columns(feature_columns_path)

    if isinstance(row, pd.Series):
        features = row.to_frame().T
    else:
        features = row.copy()

    if len(features) != 1:
        raise ValueError("predict_home_win_probability expects exactly one row.")

    missing_columns = set(feature_columns).difference(features.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input row is missing required feature columns: {missing}")

    positive_class_index = list(model.classes_).index(1)
    probability = model.predict_proba(features.loc[:, feature_columns])[
        :,
        positive_class_index,
    ][0]
    return float(probability)
