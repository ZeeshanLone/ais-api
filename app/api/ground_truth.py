import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import GroundTruthDB, PredictionDB
from app.schemas import LABEL_TO_INT, GroundTruthRequest, GroundTruthResponse
from app.services.evaluation_service import EvaluationService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/ground-truth",
    response_model=GroundTruthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a ground truth label for a past prediction",
    description=(
        "Links a true vessel state label to a prediction using the prediction_id "
        "returned by /predict. Each prediction_id can only receive one ground truth "
        "label. Metrics are recomputed internally on every submission."
    ),
)
def submit_ground_truth(
    payload: GroundTruthRequest,
    db: Session = Depends(get_db),
) -> GroundTruthResponse:
    """
    Workflow
    --------
    1. Validate that the prediction_id exists.
    2. Reject if a ground truth label has already been submitted for this id.
    3. Persist the ground truth row.
    4. Trigger metric recomputation + logging (EvaluationService).
    5. Return simple acknowledgement.
    """
    prediction_id_str = str(payload.prediction_id)

    # --- 1. Check prediction exists ---
    prediction = db.execute(
        select(PredictionDB).where(PredictionDB.id == prediction_id_str)
    ).scalar_one_or_none()

    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"prediction_id '{prediction_id_str}' not found.",
        )

    # --- 2. Check not already labelled ---
    existing = db.execute(
        select(GroundTruthDB).where(GroundTruthDB.prediction_id == prediction_id_str)
    ).scalar_one_or_none()

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Ground truth for prediction_id '{prediction_id_str}' "
                f"already submitted (true_label='{existing.true_label}')."
            ),
        )

    # --- 3. Persist ground truth ---
    db_gt = GroundTruthDB(
        prediction_id=prediction_id_str,
        true_label=payload.true_label,
        true_class=LABEL_TO_INT[payload.true_label],
    )
    db.add(db_gt)
    db.commit()

    logger.info(
        "Ground truth submitted: prediction_id=%s true_label=%s",
        prediction_id_str,
        payload.true_label,
    )

    # --- 4. Recompute + log metrics ---
    try:
        EvaluationService(db).compute_and_store()
    except Exception as exc:
        # Never fail the client request due to metric computation errors.
        # The ground truth is already committed — metrics can be recomputed later.
        logger.error("Metric computation failed after ground truth submission: %s", exc)

    # --- 5. Acknowledge ---
    return GroundTruthResponse(
        status="accepted",
        prediction_id=payload.prediction_id,
    )
