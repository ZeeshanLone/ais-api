from datetime import datetime
from typing import Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Label encoding — must stay in sync with Phase 4 LABEL_TO_INT at all times.
# Any change here requires retraining the model.
# ---------------------------------------------------------------------------
LABEL_TO_INT: Dict[str, int] = {
    "MOVING": 0,
    "DOCKED": 1,
    "DRIFTING": 2,
    "ANCHORED": 3,
}
INT_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}
VALID_LABELS = set(LABEL_TO_INT.keys())


# ===========================================================================
# INPUT SCHEMA
# ===========================================================================


class AISMessage(BaseModel):
    """
    A single raw AIS position report.

    Required fields map to AIS Type 1/2/3 (position reports).
    Optional fields map to AIS Type 5 (voyage-related static data) which
    is transmitted infrequently. When absent the feature pipeline imputes
    these with training-set medians — predictions remain valid.

    This schema is the frozen external contract. Adding a new model
    internally never requires changing this schema.
    """

    mmsi: int = Field(
        ...,
        description="Maritime Mobile Service Identity. Unique vessel identifier.",
        ge=100_000_000,
        le=999_999_999,
        examples=[538004383],
    )

    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of this AIS message (ISO-8601).",
        examples=["2026-03-12T12:15:21Z"],
    )

    lat: float = Field(
        ...,
        description="Latitude in decimal degrees.",
        ge=-90.0,
        le=90.0,
        examples=[37.7749],
    )

    lon: float = Field(
        ...,
        description="Longitude in decimal degrees.",
        ge=-180.0,
        le=180.0,
        examples=[-122.4194],
    )

    sog: float = Field(
        ...,
        description="Speed Over Ground in knots.",
        ge=0.0,
        le=102.2,
        examples=[0.3],
    )

    cog: float = Field(
        ...,
        description="Course Over Ground in degrees (0–359.9).",
        ge=0.0,
        lt=360.0,
        examples=[185.2],
    )

    # FIX: renamed from 'heading' to 'true_heading' to match Phase 3/4 FEATURE_COLS
    true_heading: float = Field(
        ...,
        description="True heading of the vessel in degrees (0–359).",
        ge=0.0,
        le=359.0,
        examples=[182.0],
    )

    rot: float = Field(
        ...,
        description=(
            "Rate of Turn in degrees per minute. "
            "Positive = starboard turn, negative = port turn."
        ),
        ge=-720.0,
        le=720.0,
        examples=[0.0],
    )

    draught: float = Field(
        ...,
        description="Maximum present static draught in metres.",
        ge=0.0,
        le=30.0,
        examples=[10.2],
    )

    special_manoeuvre_indicator: int = Field(
        ...,
        description="0 = not in special manoeuvre, 1 = in special manoeuvre.",
        ge=0,
        le=1,
        examples=[0],
    )

    # --- Optional AIS Type-5 fields ---
    # Absent from many real-time AIS feeds; pipeline imputes with train medians.

    ship_type: Optional[int] = Field(
        default=None,
        description=(
            "Raw AIS ship type code (0–99). "
            "Encoded into 11 IMO broad categories internally. "
            "Null → category 0 (Unknown/Reserved)."
        ),
        ge=0,
        le=99,
        examples=[70],
    )

    dim_bow: Optional[float] = Field(
        default=None,
        description="Distance from GPS antenna to bow (metres). Null → train median.",
        ge=0.0,
        le=511.0,
        examples=[145.0],
    )

    dim_stern: Optional[float] = Field(
        default=None,
        description="Distance from GPS antenna to stern (metres). Null → train median.",
        ge=0.0,
        le=511.0,
        examples=[45.0],
    )

    dim_port: Optional[float] = Field(
        default=None,
        description="Distance from GPS antenna to port side (metres). Null → train median.",
        ge=0.0,
        le=63.0,
        examples=[16.0],
    )

    dim_starboard: Optional[float] = Field(
        default=None,
        description="Distance from GPS antenna to starboard side (metres). Null → train median.",
        ge=0.0,
        le=63.0,
        examples=[16.0],
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        if v.year < 2000:
            raise ValueError("Timestamp appears invalid (year < 2000).")
        return v


# ===========================================================================
# OUTPUT SCHEMAS — frozen, model-agnostic
# ===========================================================================


class Prediction(BaseModel):
    """Core prediction result."""

    label: str = Field(
        ...,
        description="Predicted vessel state.",
        examples=["ANCHORED"],
    )

    class_id: int = Field(
        ...,
        description="Integer class identifier (MOVING=0, DOCKED=1, DRIFTING=2, ANCHORED=3).",
        examples=[3],
    )

    confidence: float = Field(
        ...,
        description="Confidence score — maximum predicted class probability.",
        ge=0.0,
        le=1.0,
        examples=[0.81],
    )


class ModelInfo(BaseModel):
    """Metadata about the model that generated the prediction."""

    name: str = Field(..., examples=["vessel_state_classifier"])
    version: str = Field(..., examples=["1.0.0"])
    type: str = Field(..., examples=["xgboost"])


class PredictionResponse(BaseModel):
    """
    Response returned by POST /v1/predict.

    This schema is the frozen external output contract.
    It does not change regardless of which model (XGBoost, transformer, etc.)
    is deployed internally.
    """

    prediction_id: UUID = Field(
        default_factory=uuid4,
        description=(
            "Unique identifier for this prediction. "
            "Return this when submitting a ground truth label."
        ),
    )

    timestamp: datetime = Field(
        ...,
        description="Timestamp of the AIS message used for this prediction.",
    )

    mmsi: int = Field(
        ...,
        description="MMSI of the vessel.",
    )

    prediction: Prediction = Field(
        ...,
        description="Predicted vessel state and confidence.",
    )

    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Full probability distribution across all four vessel states.",
        examples=[{"MOVING": 0.03, "DOCKED": 0.08, "DRIFTING": 0.08, "ANCHORED": 0.81}],
    )

    model_info: ModelInfo = Field(
        ...,
        description="Metadata about the model that produced this prediction.",
    )


# ===========================================================================
# GROUND TRUTH SCHEMAS
# ===========================================================================


class GroundTruthRequest(BaseModel):
    """
    Submitted by the client to attach a true label to a past prediction.
    The prediction_id is returned in PredictionResponse and acts as the link.
    """

    prediction_id: UUID = Field(
        ...,
        description="The prediction_id returned when the original prediction was made.",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )

    true_label: str = Field(
        ...,
        description="Ground truth vessel state label.",
        examples=["ANCHORED"],
    )

    @field_validator("true_label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in VALID_LABELS:
            raise ValueError(
                f"Invalid label '{v}'. Must be one of: {sorted(VALID_LABELS)}"
            )
        return v


class GroundTruthResponse(BaseModel):
    """Acknowledgement returned after a ground truth submission."""

    status: str = Field(default="accepted")
    prediction_id: UUID = Field(...)
