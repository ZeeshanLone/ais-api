import json
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AISMessageDB(Base):
    """
    Stores raw AIS messages received by the API.
    These rows are the sole source of context for feature engineering —
    the sliding window is built from the last WINDOW_SIZE rows per MMSI.
    """

    __tablename__ = "ais_messages"

    __table_args__ = (
        # Composite index — all window lookups filter on mmsi then sort by timestamp
        Index("idx_mmsi_timestamp", "mmsi", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    mmsi: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    sog: Mapped[float] = mapped_column(Float, nullable=False)
    cog: Mapped[float] = mapped_column(Float, nullable=False)
    true_heading: Mapped[float] = mapped_column(Float, nullable=False)
    rot: Mapped[float] = mapped_column(Float, nullable=False)
    draught: Mapped[float] = mapped_column(Float, nullable=False)
    special_manoeuvre_indicator: Mapped[int] = mapped_column(Integer, nullable=False)

    # Optional fields — absent from many AIS feeds
    ship_type: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dim_bow: Mapped[float | None] = mapped_column(Float, nullable=True)
    dim_stern: Mapped[float | None] = mapped_column(Float, nullable=True)
    dim_port: Mapped[float | None] = mapped_column(Float, nullable=True)
    dim_starboard: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )


class PredictionDB(Base):
    """
    Stores model predictions.
    prediction_id is returned to the client and used later as the foreign key
    when submitting ground truth labels.
    """

    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    mmsi: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    predicted_label: Mapped[str] = mapped_column(String(16), nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    prob_moving: Mapped[float] = mapped_column(Float, nullable=False)
    prob_docked: Mapped[float] = mapped_column(Float, nullable=False)
    prob_drifting: Mapped[float] = mapped_column(Float, nullable=False)
    prob_anchored: Mapped[float] = mapped_column(Float, nullable=False)

    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )


class GroundTruthDB(Base):
    """
    Ground truth labels submitted by the client.
    Linked to PredictionDB via prediction_id (string UUID).
    Used to compute evaluation metrics on each submission.
    """

    __tablename__ = "ground_truth"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    prediction_id: Mapped[str] = mapped_column(
        String(36),
        index=True,
        unique=True,  # one ground truth per prediction
        nullable=False,
    )

    true_label: Mapped[str] = mapped_column(String(16), nullable=False)
    true_class: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )


class EvaluationSnapshotDB(Base):
    """
    Point-in-time evaluation metrics snapshot.

    One row is inserted every time a ground truth label is submitted.
    This lets us track how model performance estimates evolve as more
    ground truth accumulates — useful for deciding when the evaluation
    is statistically stable enough to trust.

    confusion_matrix_json : 4x4 matrix stored as JSON string.
                            Rows = true class, Columns = predicted class.
                            Order: MOVING, DOCKED, DRIFTING, ANCHORED.
    """

    __tablename__ = "evaluation_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    total_evaluated: Mapped[int] = mapped_column(Integer, nullable=False)
    coverage_pct: Mapped[float] = mapped_column(Float, nullable=False)

    f1_macro: Mapped[float] = mapped_column(Float, nullable=False)
    f1_weighted: Mapped[float] = mapped_column(Float, nullable=False)
    precision_macro: Mapped[float] = mapped_column(Float, nullable=False)
    recall_macro: Mapped[float] = mapped_column(Float, nullable=False)

    f1_moving: Mapped[float] = mapped_column(Float, nullable=False)
    f1_docked: Mapped[float] = mapped_column(Float, nullable=False)
    f1_drifting: Mapped[float] = mapped_column(Float, nullable=False)
    f1_anchored: Mapped[float] = mapped_column(Float, nullable=False)

    precision_moving: Mapped[float] = mapped_column(Float, nullable=False)
    precision_docked: Mapped[float] = mapped_column(Float, nullable=False)
    precision_drifting: Mapped[float] = mapped_column(Float, nullable=False)
    precision_anchored: Mapped[float] = mapped_column(Float, nullable=False)

    recall_moving: Mapped[float] = mapped_column(Float, nullable=False)
    recall_docked: Mapped[float] = mapped_column(Float, nullable=False)
    recall_drifting: Mapped[float] = mapped_column(Float, nullable=False)
    recall_anchored: Mapped[float] = mapped_column(Float, nullable=False)

    # JSON-serialised 4x4 list-of-lists
    confusion_matrix_json: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    def confusion_matrix(self) -> list[list[int]]:
        return json.loads(self.confusion_matrix_json)
