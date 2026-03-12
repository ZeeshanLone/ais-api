import pickle
from pathlib import Path

from app.models.model_bundle import ModelBundle, ModelMetadata

MODEL_PATH = Path("app/models/checkpoints/xgboost_best.pkl")
SCALER_PATH = Path("app/models/checkpoints/feature_scaler.pkl")


def load_model() -> ModelBundle:

    # -------------------------
    # Load model artifact
    # -------------------------

    with open(MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)

    model = model_bundle["model"]

    feature_cols = model_bundle["feature_cols"]

    # -------------------------
    # Load scaler artifact
    # -------------------------

    with open(SCALER_PATH, "rb") as f:
        scaler_bundle = pickle.load(f)

    scaler = scaler_bundle["scaler"]

    scaler_cols = scaler_bundle["cols"]

    train_medians = scaler_bundle["train_medians"]

    # -------------------------
    # Metadata
    # -------------------------

    metadata = ModelMetadata(
        name="vessel_state_classifier",
        version="1.0.0",
        model_type="xgboost",
    )

    return ModelBundle(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        scaler_cols=scaler_cols,
        train_medians=train_medians,
        metadata=metadata,
    )
