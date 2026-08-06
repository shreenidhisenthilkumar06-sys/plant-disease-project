"""FastAPI application for Plant Disease Detection."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from predict import ModelService, ModelUnavailableError
from preprocess import InvalidImageError, validate_upload
from schemas import HealthResponse, PredictionResponse, ResearchAssetsResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
ALLOWED_RESEARCH_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the TensorFlow model once before accepting requests."""
    service = ModelService(
    model_path=RESEARCH_DIR / "saved_model" / "pitlid_grape_best.h5",
    class_names_path=Path(__file__).with_name("class_names.json"),
)
    try:
        service.load()
    except ModelUnavailableError as error:
        # Keep non-inference endpoints available and return a clear 503 for predictions.
        app.state.model_error = str(error)
        app.state.model_service = None
    else:
        app.state.model_error = None
        app.state.model_service = service
    yield


app = FastAPI(
    title="Plant Disease Detection API",
    version="1.0.0",
    description="Inference API backed by the existing TensorFlow/Keras research model.",
    lifespan=lifespan,
)

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_model_service(request: Request) -> ModelService:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        detail = getattr(request.app.state, "model_error", "Model is not available.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return service


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    """Report whether the API is ready to serve predictions."""
    error = getattr(request.app.state, "model_error", None)
    return HealthResponse(status="ready" if error is None else "degraded", model_loaded=error is None, detail=error)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(request: Request, image: UploadFile = File(...)) -> PredictionResponse:
    """Validate an uploaded leaf image and return its predicted disease."""
    try:
        image_bytes = await validate_upload(image)
        return get_model_service(request).predict(image_bytes)
    except InvalidImageError as error:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(error)) from error
    finally:
        await image.close()


@app.get("/research-assets", response_model=ResearchAssetsResponse, tags=["research"])
def list_research_assets() -> ResearchAssetsResponse:
    """List viewable experiment images without exposing arbitrary project files."""
    if not RESEARCH_DIR.exists():
        return ResearchAssetsResponse(assets=[])
    assets = sorted(
        path.relative_to(RESEARCH_DIR).as_posix()
        for path in RESEARCH_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_RESEARCH_SUFFIXES
    )
    return ResearchAssetsResponse(assets=assets)


@app.get("/research-assets/{asset_path:path}", tags=["research"])
def get_research_asset(asset_path: str) -> FileResponse:
    """Serve a known image from the research directory for the research gallery."""
    candidate = (RESEARCH_DIR / asset_path).resolve()
    try:
        candidate.relative_to(RESEARCH_DIR.resolve())
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.") from error
    if not candidate.is_file() or candidate.suffix.lower() not in ALLOWED_RESEARCH_SUFFIXES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
    return FileResponse(candidate)
