import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from churn_mlops.config import configure_logging
from churn_mlops.serving.model_loader import load_model
from churn_mlops.serving.predictor import PredictionResult, Predictor
from churn_mlops.serving.schemas import InputFeatures

configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    """Load the serving predictor once for the lifetime of an API process."""
    application.state.predictor = None
    try:
        application.state.predictor = Predictor(load_model())
        logger.info("Serving predictor loaded successfully.")
    except Exception:
        logger.exception("Serving predictor failed to load during startup.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    """Return the process liveness status."""
    return JSONResponse(status_code=200, content={"status": "healthy"})


@app.get("/ready")
def ready(request: Request) -> JSONResponse:
    """Return whether the serving predictor is ready to accept predictions."""
    if request.app.state.predictor is None:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(status_code=200, content={"status": "ready"})


@app.post("/predict")
def predict(request: Request, features: InputFeatures) -> PredictionResult:
    """Predict churn for one validated customer feature record.

    Args:
        features: Validated request payload containing customer features.

    Returns:
        Prediction result produced by the configured registered model.
    """
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    return predictor.predict(features)
