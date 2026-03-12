from datetime import datetime
from typing import Dict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

LABEL_TO_INT = {
    "MOVING": 0,
    "DOCKED": 1,
    "DRIFTING": 2,
    "ANCHORED": 3,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


class AISMessage(BaseModel):
    mmsi: int = Field(
        ...,
        description="Maritime Mobile Service Identity (MMSI). "
        "Unique identifier assigned to each vessel.",
        ge=100000000,
        le=999999999,
        examples=[538004383],
    )

    timestamp: datetime = Field(
        ...,
        description="Timestamp of the AIS message in ISO-8601 format (UTC recommended).",
        examples=["2026-03-12T12:15:21Z"],
    )

    lat: float = Field(
        ...,
        description="Latitude of the vessel position in decimal degrees.",
        ge=-90.0,
        le=90.0,
        examples=[37.7749],
    )

    lon: float = Field(
        ...,
        description="Longitude of the vessel position in decimal degrees.",
        ge=-180.0,
        le=180.0,
        examples=[-122.4194],
    )

    sog: float = Field(
        ...,
        description="Speed Over Ground (SOG) in knots.",
        ge=0,
        le=102.2,
        examples=[0.3],
    )

    cog: float = Field(
        ...,
        description="Course Over Ground (COG) in degrees relative to true north.",
        ge=0,
        lt=360,
        examples=[185.2],
    )

    heading: float = Field(
        ...,
        description="True heading of the vessel in degrees.",
        ge=0,
        le=359,
        examples=[182],
    )

    rot: float = Field(
        ...,
        description="Rate of Turn (ROT) in degrees per minute. "
        "Positive values indicate starboard turn, negative values indicate port turn.",
        ge=-720,
        le=720,
        examples=[0.0],
    )

    draught: float = Field(
        ...,
        description="Maximum present static draught of the vessel in meters.",
        ge=0,
        le=30,
        examples=[10.2],
    )

    special_manoeuvre_indicator: int = Field(
        ...,
        description=(
            "AIS Special Manoeuvre Indicator. Indicates whether the vessel "
            "is engaged in a special manoeuvre.\n\n"
            "Allowed values:\n"
            "0 → Not engaged in special manoeuvre\n"
            "1 → Engaged in special manoeuvre"
        ),
        ge=0,
        le=1,
        examples=[0],
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime):
        if v.year < 2000:
            raise ValueError("Timestamp appears invalid (year < 2000).")
        return v


class Prediction(BaseModel):
    """
    Core prediction result.
    """

    label: str = Field(
        ...,
        description="Predicted vessel state label.",
        examples=["ANCHORED"],
    )

    class_id: int = Field(
        ...,
        description="Integer identifier of the predicted class.",
        examples=[3],
    )

    confidence: float = Field(
        ...,
        description="Confidence score of the prediction. "
        "Usually equal to the maximum predicted class probability.",
        ge=0.0,
        le=1.0,
        examples=[0.81],
    )


class ModelInfo(BaseModel):
    """
    Metadata describing the model used to generate the prediction.
    """

    name: str = Field(
        ...,
        description="Logical name of the prediction model.",
        examples=["vessel_state_classifier"],
    )

    version: str = Field(
        ...,
        description="Version identifier of the deployed model.",
        examples=["1.0.0"],
    )

    type: str = Field(
        ...,
        description="Underlying model architecture (e.g. xgboost, transformer, lstm).",
        examples=["xgboost"],
    )


class PredictionResponse(BaseModel):
    """
    Response returned by the prediction API.
    """

    prediction_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier of this prediction. "
        "Used to link predictions with ground truth later.",
    )

    timestamp: datetime = Field(
        ...,
        description="Timestamp of the AIS message used for prediction.",
    )

    mmsi: int = Field(
        ...,
        description="MMSI of the vessel associated with this prediction.",
    )

    prediction: Prediction = Field(
        ...,
        description="Predicted vessel state and associated confidence.",
    )

    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all possible vessel states.",
        examples=[
            {
                "MOVING": 0.03,
                "DOCKED": 0.08,
                "DRIFTING": 0.08,
                "ANCHORED": 0.81,
            }
        ],
    )

    model_info: ModelInfo = Field(
        ...,
        description="Metadata about the model that generated the prediction.",
    )
