from __future__ import annotations

from typing import Any

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"
REQUEST_TIMEOUT_SECONDS = 10
REQUIRED_RESPONSE_FIELDS = {
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "risk_score",
    "risk_level",
    "reasons",
}


def format_probability(probability: float) -> str:
    return f"{probability:.1%}"


def get_error_detail(response: requests.Response, fallback: str) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return fallback

    return str(detail) if detail else fallback


def fetch_prediction(game_id: str) -> dict[str, Any]:
    response = requests.post(
        API_URL,
        json={"game_id": game_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code == 404:
        detail = get_error_detail(response, "Game ID was not found.")
        raise ValueError(detail)

    if response.status_code >= 400:
        detail = get_error_detail(response, "Prediction request failed.")
        raise RuntimeError(detail)

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("Unexpected API response. Expected JSON.") from exc

    missing_fields = REQUIRED_RESPONSE_FIELDS.difference(result)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise RuntimeError(f"Unexpected API response. Missing fields: {missing}")

    if not isinstance(result["reasons"], list):
        raise RuntimeError("Unexpected API response. Reasons must be a list.")

    return result


def render_prediction(result: dict[str, Any]) -> None:
    st.success("Matchup analyzed successfully.")
    st.divider()

    st.header(f"{result['home_team']} vs {result['away_team']}")
    st.caption(f"Game date: {result['game_date']}")

    winner_column, risk_column = st.columns(2)
    winner_column.metric(
        "Predicted Winner",
        result["predicted_winner"],
    )
    risk_column.metric(
        "Risk Score",
        f"{result['risk_score']} / 10",
    )

    st.write("")
    probability_columns = st.columns(2)
    probability_columns[0].metric(
        "Home Win Probability",
        format_probability(result["home_win_probability"]),
    )
    probability_columns[1].metric(
        "Away Win Probability",
        format_probability(result["away_win_probability"]),
    )

    st.metric("Risk Level", result["risk_level"])

    st.subheader("Why this prediction?")
    for reason in result["reasons"]:
        st.markdown(f"- {reason}")


def main() -> None:
    st.set_page_config(page_title="BetSki", layout="centered")

    st.title("BetSki")
    st.subheader("NBA Betting Risk Analyzer")
    st.write("")
    st.info("Start the FastAPI backend before analyzing a matchup.")

    game_id = st.text_input(
        "Game ID",
        placeholder="Example: 0022300076",
    ).strip()

    if st.button("Analyze Matchup", type="primary"):
        if not game_id:
            st.warning("Enter a Game ID before analyzing a matchup.")
            st.stop()

        try:
            with st.spinner("Analyzing matchup..."):
                result = fetch_prediction(game_id)
        except requests.exceptions.ConnectionError:
            st.error("FastAPI backend is not running. Start it on port 8000 first.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("FastAPI backend did not respond in time. Try again.")
            st.stop()
        except requests.exceptions.RequestException:
            st.error("Could not reach the FastAPI backend.")
            st.stop()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        render_prediction(result)


if __name__ == "__main__":
    main()
