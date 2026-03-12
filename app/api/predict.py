from app.services.inference_service import InferenceService
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas import AISMessage, PredictionResponse

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict vessel state from AIS data",
    description=(
        "Accepts a single raw AIS message and predicts the vessel "
        "operational state (MOVING, DOCKED, DRIFTING, ANCHORED). "
        "The API internally manages context windows and feature engineering."
    ),
)
def predict(
    message: AISMessage,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Main prediction endpoint.

    Workflow:
    1. Receive AIS message
    2. Store message in database
    3. Retrieve recent vessel history
    4. Generate features
    5. Run model inference
    6. Store prediction
    7. Return prediction response
    """

    # Retrieve model bundle loaded at startup
    model_bundle = request.app.state.model_bundle

    # Create inference service
    service = InferenceService(
        db=db,
        model_bundle=model_bundle,
    )

    # Run prediction pipeline
    result = service.predict(message)

    return result
