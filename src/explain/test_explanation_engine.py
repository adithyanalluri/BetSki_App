from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain.explanation_engine import ExplanationEngine  # noqa: E402


def _sample_context() -> dict[str, float | int | str]:
    return {
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


def _validate_expected_reasons(reasons: list[str]) -> None:
    if not 3 <= len(reasons) <= 5:
        raise ValueError(f"Expected 3-5 reasons, received {len(reasons)}.")

    expected_fragments = [
        "Model confidence favors BOS",
        "BOS has a stronger recent win rate than NYK.",
        "BOS has a better recent point differential.",
        "NYK is playing on shorter rest.",
    ]

    for fragment in expected_fragments:
        if not any(fragment in reason for reason in reasons):
            raise ValueError(f"Missing expected explanation fragment: {fragment}")


def _validate_conflicting_signal_reason(engine: ExplanationEngine) -> None:
    context = _sample_context()
    context["home_last_5_win_pct"] = 0.40
    context["away_last_5_win_pct"] = 0.80

    reasons = engine.generate_explanation(context)["reasons"]

    expected_reason = "NYK has better recent form, which raises risk."
    if not any(expected_reason in reason for reason in reasons):
        raise ValueError("Expected conflicting recent-form reason to raise risk.")


def _validate_missing_field_error(engine: ExplanationEngine) -> None:
    context = _sample_context()
    del context["home_rest_days"]

    try:
        engine.generate_explanation(context)
    except ValueError as error:
        if "home_rest_days" not in str(error):
            raise ValueError("Missing-field error did not name home_rest_days.") from error
        return

    raise ValueError("Expected missing-field validation error.")


def test_explanation_engine() -> None:
    engine = ExplanationEngine()
    explanation = engine.generate_explanation(_sample_context())
    reasons = explanation["reasons"]

    _validate_expected_reasons(reasons)
    _validate_conflicting_signal_reason(engine)
    _validate_missing_field_error(engine)

    print("ExplanationEngine verification passed.")
    print(explanation)


if __name__ == "__main__":
    test_explanation_engine()
