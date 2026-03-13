from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelMetadata:
    name: str
    version: str
    model_type: str


@dataclass
class ModelBundle:
    """
    Self-contained bundle of everything a pipeline needs to produce predictions.

    Fields
    ------
    model          : Trained model object (XGBClassifier, torch.nn.Module, etc.)
    scaler         : Fitted StandardScaler (or None if model handles scaling internally)
    feature_cols   : Ordered list of all feature names the model expects.
                     The pipeline MUST produce features in exactly this order.
    scaler_cols    : Subset of feature_cols that the scaler was fitted on.
                     Categoricals (ship_type_encoded, hour_of_day, day_of_week)
                     are excluded.
    train_medians  : Median values from the training set used to impute missing
                     static fields (draught, dim_bow, dim_stern, dim_port,
                     dim_starboard) and as fallback for any unrecognised feature.
    port_tree      : Pre-built sklearn BallTree over NGA WPI port coordinates
                     (haversine metric, radians). Used to compute
                     dist_to_nearest_port at inference time.
                     None if the NGA WPI download failed at startup — the
                     pipeline falls back to -1.0 for that feature in that case.
    metadata       : Descriptive metadata about the model.
    """

    model: Any
    scaler: Optional[Any]

    feature_cols: List[str]
    scaler_cols: List[str]
    train_medians: Dict[str, float]

    port_tree: Optional[Any]  # sklearn.neighbors.BallTree | None

    metadata: ModelMetadata
