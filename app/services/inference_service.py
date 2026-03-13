"""
Inference service.

Orchestrates the full prediction pipeline:
    1. Store incoming AIS message
    2. Fetch sliding window (last WINDOW_SIZE rows for this MMSI)
    3. Delegate feature engineering to the registered pipeline
    4. Delegate inference to the registered predictor
    5. Store prediction
    6. Return PredictionResponse

InferenceService has no knowledge of which model is active. It looks up the
(FeaturePipeline, BasePredictor) pair from the registry using model_type from
the ModelBundle. Adding a new model requires zero changes here.
"""

import logging
from typing import List
from uuid import UUID, uuid4

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AISMessageDB, PredictionDB
from app.models.model_bundle import ModelBundle
from app.pipeline.registry import get_pipeline_pair
from app.schemas import (
    INT_TO_LABEL,
    AISMessage,
    ModelInfo,
    Prediction,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Stateless service — one instance per request.
    """

    WINDOW_SIZE = 30

    def __init__(self, db: Session, model_bundle: ModelBundle):
        self.db = db
        self.bundle = model_bundle

    def predict(self, message: AISMessage) -> PredictionResponse:
        """
        Execute the full prediction pipeline for a single AIS message.
        """
        # 1. Persist incoming message
        self._store_message(message)

        # 2. Build sliding window (last WINDOW_SIZE messages for this MMSI)
        window = self._get_window(message.mmsi)

        # 3. Look up the correct pipeline + predictor pair
        pair = get_pipeline_pair(self.bundle.metadata.model_type)

        # 4. Feature engineering (model-specific)
        features = pair.pipeline.compute(window, self.bundle)

        # 5. Inference (model-specific)
        probs = pair.predictor.predict(features, self.bundle)

        # 6. Decode result
        class_id = int(np.argmax(probs))
        confidence = float(np.max(probs))
        label = INT_TO_LABEL[class_id]

        # 7. Persist prediction
        prediction_id = self._store_prediction(
            message, class_id, label, confidence, probs
        )

        logger.debug(
            "Prediction: mmsi=%d ts=%s label=%s confidence=%.3f",
            message.mmsi,
            message.timestamp,
            label,
            confidence,
        )

        # 8. Build response
        return PredictionResponse(
            prediction_id=prediction_id,
            timestamp=message.timestamp,
            mmsi=message.mmsi,
            prediction=Prediction(
                label=label,
                class_id=class_id,
                confidence=confidence,
            ),
            class_probabilities={
                "MOVING": float(probs[0]),
                "DOCKED": float(probs[1]),
                "DRIFTING": float(probs[2]),
                "ANCHORED": float(probs[3]),
            },
            model_info=ModelInfo(
                name=self.bundle.metadata.name,
                version=self.bundle.metadata.version,
                type=self.bundle.metadata.model_type,
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_message(self, message: AISMessage) -> None:
        db_msg = AISMessageDB(
            mmsi=message.mmsi,
            timestamp=message.timestamp,
            lat=message.lat,
            lon=message.lon,
            sog=message.sog,
            cog=message.cog,
            true_heading=message.true_heading,
            rot=message.rot,
            draught=message.draught,
            special_manoeuvre_indicator=message.special_manoeuvre_indicator,
            ship_type=message.ship_type,
            dim_bow=message.dim_bow,
            dim_stern=message.dim_stern,
            dim_port=message.dim_port,
            dim_starboard=message.dim_starboard,
        )
        self.db.add(db_msg)
        self.db.commit()

    def _get_window(self, mmsi: int) -> List[AISMessageDB]:
        """
        Fetch the last WINDOW_SIZE messages for this MMSI, sorted ascending
        by timestamp (oldest first). This is the same ordering Phase 3 uses
        (sort by time_epoch before rolling windows).
        """
        stmt = (
            select(AISMessageDB)
            .where(AISMessageDB.mmsi == mmsi)
            .order_by(AISMessageDB.timestamp.desc())
            .limit(self.WINDOW_SIZE)
        )
        rows = self.db.execute(stmt).scalars().all()
        # Reverse: DB returns newest-first; pipeline expects oldest-first
        return list(reversed(rows))

    def _store_prediction(
        self,
        message: AISMessage,
        class_id: int,
        label: str,
        confidence: float,
        probs: np.ndarray,
    ) -> UUID:
        prediction_id = uuid4()
        db_pred = PredictionDB(
            id=str(prediction_id),
            mmsi=message.mmsi,
            timestamp=message.timestamp,
            predicted_label=label,
            predicted_class=class_id,
            confidence=confidence,
            prob_moving=float(probs[0]),
            prob_docked=float(probs[1]),
            prob_drifting=float(probs[2]),
            prob_anchored=float(probs[3]),
            model_name=self.bundle.metadata.name,
            model_version=self.bundle.metadata.version,
            model_type=self.bundle.metadata.model_type,
        )
        self.db.add(db_pred)
        self.db.commit()
        return prediction_id
