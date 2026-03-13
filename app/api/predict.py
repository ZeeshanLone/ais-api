from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas import AISMessage, PredictionResponse
from app.services.inference_service import InferenceService

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict vessel state from a single AIS message",
    description=(
        "Accepts a raw AIS position report and returns the predicted vessel "
        "operational state (MOVING, DOCKED, DRIFTING, ANCHORED) with class "
        "probabilities. The API manages context windows and feature engineering "
        "internally. Save the returned prediction_id to submit ground truth later."
    ),
)
def predict(
    message: AISMessage,
    request: Request,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    service = InferenceService(
        db=db,
        model_bundle=request.app.state.model_bundle,
    )
    return service.predict(message)
