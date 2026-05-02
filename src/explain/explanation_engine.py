from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


MIN_REASONS = 3
MAX_REASONS = 5
HIGH_CONFIDENCE_MIN = 0.65
MODERATE_CONFIDENCE_MIN = 0.55
MEANINGFUL_WIN_PCT_GAP = 0.20
MEANINGFUL_POINT_DIFF_GAP = 3.0
MEANINGFUL_REST_GAP = 1.0
REQUIRED_EXPLANATION_COLUMNS = [
    "home_team",
    "away_team",
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


class ExplanationEngine:
    """Generate concise, human-readable reasons for a prediction and risk score."""

    def generate_explanation(
        self,
        explanation_input: Mapping[str, Any] | pd.Series,
    ) -> dict[str, list[str]]:
        context = self._normalize_context(explanation_input)
        favored_side = self._favored_side(context["home_win_probability"])
        underdog_side = "away" if favored_side == "home" else "home"

        reasons = [
            self._confidence_reason(context, favored_side),
            self._form_reason(context, favored_side, underdog_side),
            self._point_diff_reason(context, favored_side, underdog_side),
            self._rest_reason(context, favored_side, underdog_side),
            self._back_to_back_reason(context, favored_side, underdog_side),
        ]
        filtered_reasons = [reason for reason in reasons if reason]

        if len(filtered_reasons) < MIN_REASONS:
            alignment_reason = self._alignment_reason(
                context,
                favored_side,
                underdog_side,
            )
            if alignment_reason:
                filtered_reasons.append(alignment_reason)

        if len(filtered_reasons) < MIN_REASONS:
            filtered_reasons.append(
                self._fallback_risk_reason(context, favored_side)
            )

        return {"reasons": filtered_reasons[:MAX_REASONS]}

    def _normalize_context(
        self,
        explanation_input: Mapping[str, Any] | pd.Series,
    ) -> dict[str, Any]:
        if isinstance(explanation_input, pd.Series):
            context = explanation_input.to_dict()
        else:
            context = dict(explanation_input)

        missing_columns = set(REQUIRED_EXPLANATION_COLUMNS).difference(context)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Explanation context is missing required fields: {missing}"
            )

        normalized_context: dict[str, Any] = {
            "home_team": str(context["home_team"]),
            "away_team": str(context["away_team"]),
        }
        for column in REQUIRED_EXPLANATION_COLUMNS:
            if column in {"home_team", "away_team"}:
                continue

            value = context[column]
            if pd.isna(value):
                raise ValueError(
                    f"Explanation context contains missing value for {column}."
                )
            normalized_context[column] = float(value)

        self._validate_probability(normalized_context["home_win_probability"])
        return normalized_context

    def _confidence_reason(
        self,
        context: dict[str, Any],
        favored_side: str,
    ) -> str:
        favored_team = self._team(context, favored_side)
        favored_probability = self._win_probability(context, favored_side)

        if favored_probability >= HIGH_CONFIDENCE_MIN:
            return f"Model confidence strongly favors {favored_team}."
        if favored_probability >= MODERATE_CONFIDENCE_MIN:
            return f"Model confidence favors {favored_team}, but the edge is moderate."
        return "Model confidence is close, increasing risk."

    def _form_reason(
        self,
        context: dict[str, Any],
        favored_side: str,
        underdog_side: str,
    ) -> str | None:
        favored_form = context[f"{favored_side}_last_5_win_pct"]
        underdog_form = context[f"{underdog_side}_last_5_win_pct"]
        form_gap = favored_form - underdog_form

        if abs(form_gap) < MEANINGFUL_WIN_PCT_GAP:
            return None

        favored_team = self._team(context, favored_side)
        underdog_team = self._team(context, underdog_side)

        if form_gap > 0:
            return f"{favored_team} has a stronger recent win rate than {underdog_team}."
        return f"{underdog_team} has better recent form, which raises risk."

    def _point_diff_reason(
        self,
        context: dict[str, Any],
        favored_side: str,
        underdog_side: str,
    ) -> str | None:
        favored_point_diff = context[f"{favored_side}_last_10_point_diff"]
        underdog_point_diff = context[f"{underdog_side}_last_10_point_diff"]
        point_diff_gap = favored_point_diff - underdog_point_diff

        if abs(point_diff_gap) < MEANINGFUL_POINT_DIFF_GAP:
            return None

        favored_team = self._team(context, favored_side)
        underdog_team = self._team(context, underdog_side)

        if point_diff_gap > 0:
            return f"{favored_team} has a better recent point differential."
        return f"{underdog_team} has a better recent point differential, which raises risk."

    def _rest_reason(
        self,
        context: dict[str, Any],
        favored_side: str,
        underdog_side: str,
    ) -> str | None:
        favored_rest = context[f"{favored_side}_rest_days"]
        underdog_rest = context[f"{underdog_side}_rest_days"]
        rest_gap = favored_rest - underdog_rest

        if abs(rest_gap) < MEANINGFUL_REST_GAP:
            return None

        favored_team = self._team(context, favored_side)
        underdog_team = self._team(context, underdog_side)

        if rest_gap > 0:
            return f"{underdog_team} is playing on shorter rest."
        return f"{favored_team} is playing on shorter rest, which raises risk."

    def _back_to_back_reason(
        self,
        context: dict[str, Any],
        favored_side: str,
        underdog_side: str,
    ) -> str | None:
        favored_team = self._team(context, favored_side)
        underdog_team = self._team(context, underdog_side)
        favored_back_to_back = context[f"{favored_side}_back_to_back"] == 1
        underdog_back_to_back = context[f"{underdog_side}_back_to_back"] == 1

        if favored_back_to_back == underdog_back_to_back:
            return None
        if underdog_back_to_back:
            return f"{underdog_team} is on a back-to-back."
        return f"{favored_team} is on a back-to-back, which raises risk."

    def _alignment_reason(
        self,
        context: dict[str, Any],
        favored_side: str,
        underdog_side: str,
    ) -> str | None:
        favored_team = self._team(context, favored_side)
        underdog_team = self._team(context, underdog_side)
        favored_form = context[f"{favored_side}_last_5_win_pct"]
        underdog_form = context[f"{underdog_side}_last_5_win_pct"]
        favored_point_diff = context[f"{favored_side}_last_10_point_diff"]
        underdog_point_diff = context[f"{underdog_side}_last_10_point_diff"]

        form_is_close = abs(favored_form - underdog_form) < MEANINGFUL_WIN_PCT_GAP
        point_diff_is_close = (
            abs(favored_point_diff - underdog_point_diff)
            < MEANINGFUL_POINT_DIFF_GAP
        )

        if form_is_close and point_diff_is_close:
            return (
                f"{favored_team} and {underdog_team} are close in recent form "
                "and point differential, increasing risk."
            )
        return None

    def _fallback_risk_reason(
        self,
        context: dict[str, Any],
        favored_side: str,
    ) -> str:
        favored_probability = self._win_probability(context, favored_side)

        if favored_probability < HIGH_CONFIDENCE_MIN:
            return "Model confidence is not high enough for a low-risk pick."
        return "Model confidence is high enough to reduce risk."

    def _favored_side(self, home_win_probability: float) -> str:
        if home_win_probability >= 0.5:
            return "home"
        return "away"

    def _team(self, context: dict[str, Any], side: str) -> str:
        return str(context[f"{side}_team"])

    def _win_probability(self, context: dict[str, Any], side: str) -> float:
        home_win_probability = context["home_win_probability"]

        if side == "home":
            return home_win_probability
        return 1 - home_win_probability

    def _validate_probability(self, probability: float) -> None:
        if not 0 <= probability <= 1:
            raise ValueError(
                "home_win_probability must be between 0 and 1. "
                f"Received {probability}."
            )


if __name__ == "__main__":
    sample_input = {
        "home_team": "BOS",
        "away_team": "NYK",
        "home_win_probability": 0.62,
        "home_last_5_win_pct": 0.80,
        "away_last_5_win_pct": 0.40,
        "home_last_10_point_diff": 7.5,
        "away_last_10_point_diff": 1.5,
        "home_rest_days": 2,
        "away_rest_days": 1,
        "home_back_to_back": 0,
        "away_back_to_back": 1,
    }
    engine = ExplanationEngine()
    print(engine.generate_explanation(sample_input))
