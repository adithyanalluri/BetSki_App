from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEATURE_COLUMNS_PATH, MODEL_PATH, PROCESSED_DATA_DIR  # noqa: E402
from src.risk.risk_engine import REQUIRED_CONTEXT_COLUMNS, RiskEngine  # noqa: E402


SEASON_SLUG = "2023_24"
FEATURES_FILE = PROCESSED_DATA_DIR / f"features_{SEASON_SLUG}.csv"
SAMPLE_GAME_ID = "0022300076"


class PredictionService:
    """Local service for generating model win probabilities."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_columns_path: Path = FEATURE_COLUMNS_PATH,
        features_path: Path = FEATURES_FILE,
    ) -> None:
        self.model_path = model_path
        self.feature_columns_path = feature_columns_path
        self.features_path = features_path

        self._validate_required_files()
        self.model = self._load_model()
        self.feature_columns = self._load_feature_columns()
        self.features = self._load_features()
        self.risk_engine = RiskEngine()
        self._validate_feature_columns_present()
        self._validate_risk_columns_present()

    def predict_by_game_id(self, game_id: str) -> dict[str, Any]:
        """Predict home and away win probabilities for a historical game."""
        game = self._get_game_row(game_id)
        model_input = game.loc[self.feature_columns].to_frame().T

        home_probability = float(self.model.predict_proba(model_input)[:, 1][0])
        away_probability = 1 - home_probability

        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        predicted_winner = home_team if home_probability >= 0.5 else away_team
        risk = self.risk_engine.calculate_risk(
            self._build_risk_input(game, home_probability)
        )

        return {
            "game_id": str(game["game_id"]),
            "game_date": str(game["game_date"]),
            "home_team": home_team,
            "away_team": away_team,
            "home_win_probability": home_probability,
            "away_win_probability": away_probability,
            "predicted_winner": predicted_winner,
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
        }

    def _validate_required_files(self) -> None:
        required_files = {
            "model artifact": self.model_path,
            "feature columns file": self.feature_columns_path,
            "feature dataset": self.features_path,
        }
        missing_files = [
            f"{label}: {path}"
            for label, path in required_files.items()
            if not path.exists()
        ]

        if missing_files:
            formatted_files = "\n".join(missing_files)
            raise FileNotFoundError(
                "PredictionService could not find required files:\n"
                f"{formatted_files}\n"
                "Run the data, feature, and model training pipeline before predicting."
            )

    def _load_model(self) -> Any:
        return joblib.load(self.model_path)

    def _load_feature_columns(self) -> list[str]:
        with self.feature_columns_path.open("r", encoding="utf-8") as file:
            feature_columns = json.load(file)

        if not isinstance(feature_columns, list) or not feature_columns:
            raise ValueError(
                f"Feature columns file must contain a non-empty list: {self.feature_columns_path}"
            )

        return feature_columns

    def _load_features(self) -> pd.DataFrame:
        features = pd.read_csv(self.features_path, dtype={"game_id": str})

        if features.empty:
            raise ValueError(f"Feature dataset is empty: {self.features_path}")

        return features

    def _validate_feature_columns_present(self) -> None:
        missing_columns = set(self.feature_columns).difference(self.features.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Feature dataset is missing columns required by the model: "
                f"{missing}"
            )

    def _validate_risk_columns_present(self) -> None:
        risk_feature_columns = set(REQUIRED_CONTEXT_COLUMNS) - {"home_win_probability"}
        missing_columns = risk_feature_columns.difference(self.features.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Feature dataset is missing columns required by RiskEngine: "
                f"{missing}"
            )

    def _build_risk_input(
        self,
        game: pd.Series,
        home_win_probability: float,
    ) -> dict[str, Any]:
        risk_input: dict[str, Any] = {"home_win_probability": home_win_probability}

        for column in REQUIRED_CONTEXT_COLUMNS:
            if column == "home_win_probability":
                continue
            risk_input[column] = game[column]

        return risk_input

    def _get_game_row(self, game_id: str) -> pd.Series:
        matches = self.features[self.features["game_id"] == str(game_id)]

        if matches.empty:
            raise ValueError(
                f"game_id {game_id} was not found in feature dataset: {self.features_path}"
            )

        if len(matches) > 1:
            raise ValueError(f"game_id {game_id} matched multiple feature rows.")

        return matches.iloc[0]


if __name__ == "__main__":
    service = PredictionService()
    print(service.predict_by_game_id(SAMPLE_GAME_ID))
