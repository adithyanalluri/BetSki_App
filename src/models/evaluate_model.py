from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def evaluate_classifier(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a binary classifier with probability-aware metrics."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
    }


def format_metrics_table(metrics_by_model: dict[str, dict[str, float]]) -> str:
    """Return a readable metrics table for terminal output."""
    metrics = pd.DataFrame.from_dict(metrics_by_model, orient="index")
    metrics = metrics.loc[:, ["accuracy", "roc_auc", "log_loss"]]
    return metrics.round(4).to_string()
