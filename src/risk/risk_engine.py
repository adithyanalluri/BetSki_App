from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


LOW_RISK_MAX = 3
MODERATE_RISK_MAX = 6
MIN_RISK_SCORE = 1
MAX_RISK_SCORE = 10
REQUIRED_CONTEXT_COLUMNS = [
    "home_win_probability",
    "home_last_5_win_pct",
    "away_last_5_win_pct",
    "home_last_10_point_diff",
    "away_last_10_point_diff",
    "home_rest_days",
    "away_rest_days",
    "home_back_to_back",
    "away_back_to_back",
]


class RiskEngine:
    """Convert model probabilities and matchup context into a 1-10 risk score.

    Future versions can add odds-implied probability gaps, injury impact,
    travel load, market volatility, ensemble disagreement, and calibrated
    uncertainty once those inputs exist in the pipeline.
    """

    def calculate_risk(
        self,
        risk_input: Mapping[str, Any] | pd.Series,
    ) -> dict[str, Any]:
        context = self._normalize_context(risk_input)
        home_win_probability = context["home_win_probability"]
        self._validate_probability(home_win_probability)

        base_score = self._base_score_from_probability(home_win_probability)
        form_adjustment = self._form_adjustment(context)
        point_diff_adjustment = self._point_diff_adjustment(context)
        rest_adjustment = self._rest_adjustment(context)

        raw_score = (
            base_score
            + form_adjustment
            + point_diff_adjustment
            + rest_adjustment
        )
        risk_score = self._clamp_score(raw_score)

        return {
            "risk_score": risk_score,
            "risk_level": self._risk_level(risk_score),
        }

    def _normalize_context(
        self,
        risk_input: Mapping[str, Any] | pd.Series,
    ) -> dict[str, float]:
        if isinstance(risk_input, pd.Series):
            context = risk_input.to_dict()
        else:
            context = dict(risk_input)

        missing_columns = set(REQUIRED_CONTEXT_COLUMNS).difference(context)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Risk context is missing required fields: {missing}")

        normalized_context: dict[str, float] = {}
        for column in REQUIRED_CONTEXT_COLUMNS:
            value = context[column]
            if pd.isna(value):
                raise ValueError(f"Risk context contains missing value for {column}.")
            normalized_context[column] = float(value)

        return normalized_context

    def _base_score_from_probability(self, probability: float) -> int:
        confidence = abs(probability - 0.50)

        if confidence < 0.05:
            return 9
        if confidence < 0.10:
            return 8
        if confidence < 0.15:
            return 7
        if confidence < 0.20:
            return 6
        if confidence < 0.25:
            return 5
        if confidence < 0.30:
            return 4
        if confidence < 0.35:
            return 3
        return 2

    def _form_adjustment(self, context: dict[str, float]) -> int:
        win_pct_gap = abs(
            context["home_last_5_win_pct"] - context["away_last_5_win_pct"]
        )

        if win_pct_gap <= 0.20:
            return 1
        if win_pct_gap >= 0.50:
            return -1
        return 0

    def _point_diff_adjustment(self, context: dict[str, float]) -> int:
        point_diff_gap = abs(
            context["home_last_10_point_diff"]
            - context["away_last_10_point_diff"]
        )

        if point_diff_gap <= 3:
            return 1
        if point_diff_gap >= 10:
            return -1
        return 0

    def _rest_adjustment(self, context: dict[str, float]) -> int:
        home_back_to_back = context["home_back_to_back"] == 1
        away_back_to_back = context["away_back_to_back"] == 1

        if home_back_to_back != away_back_to_back:
            return 1
        return 0

    def _clamp_score(self, score: int) -> int:
        return max(MIN_RISK_SCORE, min(MAX_RISK_SCORE, int(round(score))))

    def _risk_level(self, risk_score: int) -> str:
        if risk_score <= LOW_RISK_MAX:
            return "Low"
        if risk_score <= MODERATE_RISK_MAX:
            return "Moderate"
        return "High"

    def _validate_probability(self, probability: float) -> None:
        if not 0 <= probability <= 1:
            raise ValueError(
                "home_win_probability must be between 0 and 1. "
                f"Received {probability}."
            )


if __name__ == "__main__":
    sample_input = {
        "home_win_probability": 0.62,
        "home_last_5_win_pct": 0.60,
        "away_last_5_win_pct": 0.40,
        "home_last_10_point_diff": 4.5,
        "away_last_10_point_diff": -1.5,
        "home_rest_days": 2,
        "away_rest_days": 1,
        "home_back_to_back": 0,
        "away_back_to_back": 1,
    }
    engine = RiskEngine()
    print(engine.calculate_risk(sample_input))
