from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ModelMetadata:
    name: str
    version: str
    model_type: str


@dataclass
class ModelBundle:
    model: Any
    scaler: Any | None

    feature_cols: List[str]  # all model features
    scaler_cols: List[str]  # columns to scale

    train_medians: Dict[str, float]

    metadata: ModelMetadata
