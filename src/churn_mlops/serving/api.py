import logging
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from churn_mlops.config import ServingSettings, configure_logging
from churn_mlops.serving.model_loader import load_model
from churn_mlops.serving.prediction_logger import PredictionLogger
from churn_mlops.serving.predictor import PredictionResult, Predictor
from churn_mlops.serving.schemas import InputFeatures

configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    """Load the serving predictor once for the lifetime of an API process."""

    settings = ServingSettings()
    application.state.predictor = None
    application.state.settings = settings

    try:
        if settings.model_version is None:
            application.state.predictor = Predictor(load_model())
        else:
            application.state.predictor = Predictor(
                load_model(
                    model_name=settings.model_name,
                    model_version=settings.model_version,
                )
            )

        application.state.prediction_logger = PredictionLogger(
            prediction_log_path=settings.prediction_log_path,
            error_log_path=settings.error_log_path,
            log_features=settings.log_features,
        )

        logger.info("Serving predictor loaded successfully.")

    except Exception:
        logger.exception("Serving predictor failed to load during startup.")
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next):

    request_id_header = getattr(
        request.app.state,
        "settings",
        ServingSettings(),
    ).request_id_header
    request_id = request.headers.get(
        key=request_id_header,
        default=str(uuid4()),
    )

    request.state.request_id = request_id

    response = await call_next(request)

    response.headers[request_id_header] = request_id

    return response


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
def predict(
    request: Request, features: InputFeatures, reference_date: datetime | None = None
) -> PredictionResult:
    """Predict target class for one validated input feature record.

    Args:
        features: Validated request payload containing input features.
        reference_date: Optional reference date for the provided `features`.
            If not provided, set to UTC time at execution.

    Returns:
        Prediction result produced by the configured registered model.
    """
    predictor = request.app.state.predictor

    if predictor is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")

    prediction_logger: PredictionLogger = request.app.state.prediction_logger

    request_id = request.state.request_id

    start = perf_counter()

    try:
        prediction = predictor.predict(features=features, reference_date=reference_date)

        latency_ms = round(number=(perf_counter() - start) * 1000, ndigits=2)

        prediction_logger.log_prediction(
            request_id=request_id,
            latency_ms=latency_ms,
            prediction=prediction,
            features=features,
        )

        logger.info(
            "Prediction completed.",
            extra={
                "request_id": request_id,
                "latency_ms": latency_ms,
            },
        )

        return prediction

    except Exception as exc:
        latency_ms = round(number=(perf_counter() - start) * 1000, ndigits=2)

        try:
            prediction_logger.log_error(
                request_id=request_id,
                latency_ms=latency_ms,
                predictor=predictor,
                exception=exc,
                traceback_text=traceback.format_exc(),
                features=features,
            )
        except Exception:
            logger.exception("Failed to persist prediction error event.")

        logger.exception(
            "Prediction request failed.",
            extra={"request_id": request_id},
        )

        raise
