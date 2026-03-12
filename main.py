from contextlib import asynccontextmanager

from app.api.predict import router as prediction_router
from app.db.database import init_db
from app.models.model_loader import load_model
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting AIS Vessel State Prediction API...")

    # Initialize database tables
    init_db()

    # Load ML model into memory
    app.state.model_bundle = load_model()

    print("Model loaded successfully.")
    print("Database initialized.")

    yield

    print("Shutting down AIS Prediction API...")


app = FastAPI(
    title="AIS Vessel State Prediction API",
    description=(
        "API for predicting vessel operational states "
        "(MOVING, DOCKED, DRIFTING, ANCHORED) from raw AIS data. "
        "The API accepts raw AIS messages and performs feature engineering "
        "and inference internally."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(
    prediction_router,
    prefix="/v1",
    tags=["Prediction"],
)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "ais-vessel-state-prediction",
    }
