from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from src.services.prediction_service import PredictionService


app = FastAPI(
    title="BetSki API",
    description="API for NBA game prediction and betting risk analysis.",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str


class PredictRequest(BaseModel):
    game_id: str


class PredictResponse(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str
    risk_score: int
    risk_level: str
    reasons: list[str]


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    return PredictionService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        service = get_prediction_service()
        return service.predict_by_game_id(request.game_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from exc
