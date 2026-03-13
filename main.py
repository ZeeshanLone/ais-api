import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.ground_truth import router as ground_truth_router
from app.api.predict import router as prediction_router
from app.db.database import init_db
from app.models.model_loader import load_model

# ---------------------------------------------------------------------------
# Logging configuration
# Structured format: timestamp | level | logger | message
# Adjust level to DEBUG for local development.
# ---------------------------------------------------------------------------
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
        },
    }
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AIS Vessel State Prediction API...")

    init_db()
    logger.info("Database initialised.")

    app.state.model_bundle = load_model()
    logger.info(
        "Model bundle loaded: type=%s version=%s",
        app.state.model_bundle.metadata.model_type,
        app.state.model_bundle.metadata.version,
    )

    yield

    logger.info("Shutting down AIS Prediction API.")


app = FastAPI(
    title="AIS Vessel State Prediction API",
    description=(
        "Predicts vessel operational state (MOVING, DOCKED, DRIFTING, ANCHORED) "
        "from raw AIS position reports. Feature engineering and context window "
        "management are handled internally. "
        "Use POST /v1/ground-truth to attach true labels to past predictions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(prediction_router, prefix="/v1", tags=["Prediction"])
app.include_router(ground_truth_router, prefix="/v1", tags=["Ground Truth"])


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "ais-vessel-state-prediction"}
