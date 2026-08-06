"""Pydantic response models used by the API."""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(ge=0, le=100)
    description: str
    symptoms: str
    causes: str
    prevention: str
    treatment: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: str | None = None


class ResearchAssetsResponse(BaseModel):
    assets: list[str]
