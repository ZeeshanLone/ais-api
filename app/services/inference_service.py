from typing import List
from uuid import uuid4

import numpy as np
import polars as pl
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AISMessageDB, PredictionDB
from app.models.model_bundle import ModelBundle
from app.schemas import (
    INT_TO_LABEL,
    AISMessage,
    ModelInfo,
    Prediction,
    PredictionResponse,
)


class InferenceService:
    """
    Service responsible for executing the full prediction pipeline.
    """

    WINDOW_SIZE = 30

    def __init__(self, db: Session, model_bundle: ModelBundle):
        self.db = db
        self.model_bundle = model_bundle

    # PUBLIC ENTRYPOINT

    def predict(self, message: AISMessage) -> PredictionResponse:
        """
        Execute the complete prediction pipeline.
        """

        # 1. Store AIS message
        self._store_message(message)

        # 2. Retrieve sliding window
        window = self._get_recent_messages(message.mmsi)

        # 3. Compute raw feature dictionary
        feature_dict = self._compute_features(window)

        # 4. Align + scale features
        X = self._prepare_features(feature_dict)

        # 5. Model inference
        probs = self._predict_probabilities(X)

        # 6. Extract prediction
        class_id = int(np.argmax(probs))
        confidence = float(np.max(probs))
        label = INT_TO_LABEL[class_id]

        # 7. Store prediction
        prediction_id = self._store_prediction(
            message,
            class_id,
            label,
            confidence,
            probs,
        )

        # 8. Construct response
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
                name=self.model_bundle.metadata.name,
                version=self.model_bundle.metadata.version,
                type=self.model_bundle.metadata.model_type,
            ),
        )

    # STORE AIS MESSAGE

    def _store_message(self, message: AISMessage):

        db_message = AISMessageDB(
            mmsi=message.mmsi,
            timestamp=message.timestamp,
            lat=message.lat,
            lon=message.lon,
            sog=message.sog,
            cog=message.cog,
            heading=message.heading,
            rot=message.rot,
            draught=message.draught,
            special_manoeuvre_indicator=message.special_manoeuvre_indicator,
        )

        self.db.add(db_message)
        self.db.commit()

    # CONTEXT WINDOW

    def _get_recent_messages(self, mmsi: int) -> List[AISMessageDB]:
        """
        Fetch last WINDOW_SIZE AIS messages for a vessel.
        """

        stmt = (
            select(AISMessageDB)
            .where(AISMessageDB.mmsi == mmsi)
            .order_by(AISMessageDB.timestamp.desc())
            .limit(self.WINDOW_SIZE)
        )

        results = self.db.execute(stmt).scalars().all()

        return list(reversed(results))

    # FEATURE ENGINEERING

    def _compute_features(self, window: List[AISMessageDB]) -> dict:
        """
        Convert sliding window into raw feature dictionary.

        TODO: implement actual feature pipeline
        """

        last = window[-1]

        return {
            "sog": last.sog,
            "cog": last.cog,
            "rot": last.rot,
            "heading": last.heading,
            "draught": last.draught,
        }

    # FEATURE ALIGNMENT + SCALING

    def _prepare_features(self, features: dict) -> np.ndarray:

        feature_cols = self.model_bundle.feature_cols
        scaler_cols = self.model_bundle.scaler_cols
        medians = self.model_bundle.train_medians

        df = pl.DataFrame([features])

        # Ensure all model features exist
        for col in feature_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(medians.get(col, 0)).alias(col))

        # Correct feature order
        df = df.select(feature_cols)

        # Fill missing values
        fill_exprs = [
            pl.col(col).fill_null(medians.get(col, 0)) for col in feature_cols
        ]

        df = df.with_columns(fill_exprs)

        X = df.to_numpy()

        if self.model_bundle.scaler is not None:
            scaler_idx = [feature_cols.index(c) for c in scaler_cols]

            X[:, scaler_idx] = self.model_bundle.scaler.transform(X[:, scaler_idx])

        return X

    # MODEL INFERENCE

    def _predict_probabilities(self, X: np.ndarray):

        model = self.model_bundle.model

        probs = model.predict_proba(X)[0]

        return probs

    # STORE PREDICTION

    def _store_prediction(
        self,
        message: AISMessage,
        class_id: int,
        label: str,
        confidence: float,
        probs,
    ):

        prediction_id = uuid4()

        db_prediction = PredictionDB(
            id=prediction_id,
            mmsi=message.mmsi,
            timestamp=message.timestamp,
            predicted_label=label,
            predicted_class=class_id,
            confidence=confidence,
            prob_moving=float(probs[0]),
            prob_docked=float(probs[1]),
            prob_drifting=float(probs[2]),
            prob_anchored=float(probs[3]),
            model_name=self.model_bundle.metadata.name,
            model_version=self.model_bundle.metadata.version,
            model_type=self.model_bundle.metadata.model_type,
        )

        self.db.add(db_prediction)
        self.db.commit()

        return prediction_id
