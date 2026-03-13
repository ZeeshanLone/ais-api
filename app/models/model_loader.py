import logging
import pickle
from pathlib import Path

import numpy as np
import requests

from app.models.model_bundle import ModelBundle, ModelMetadata

logger = logging.getLogger(__name__)

MODEL_PATH = Path("app/models/checkpoints/xgboost_best.pkl")
SCALER_PATH = Path("app/models/checkpoints/feature_scaler.pkl")

# ---------------------------------------------------------------------------
# NGA World Port Index
# Port coordinates are used to compute dist_to_nearest_port — a top-10
# SHAP feature, especially important for ANCHORED vs DRIFTING separation.
# ---------------------------------------------------------------------------
NGA_WPI_URL = (
    "https://msi.nga.mil/api/publications/download"
    "?type=view&key=16694622/SFH00000/UpdatedPub150.csv"
)
NGA_WPI_CACHE = Path("app/models/checkpoints/nga_wpi_ports.csv")
EARTH_RADIUS_M = 6_371_008.8


def _load_port_tree():
    """
    Load (or download + cache) the NGA World Port Index and build a
    BallTree for O(log N) nearest-port queries at inference time.

    Returns the BallTree on success, None on any failure.
    A None port_tree causes the pipeline to set dist_to_nearest_port = -1.0,
    which matches the Phase 3 fallback behaviour exactly.
    """
    try:
        # Lazy import — sklearn is large; keep startup fast if WPI unavailable
        import polars as pl
        from sklearn.neighbors import BallTree

        NGA_WPI_CACHE.parent.mkdir(parents=True, exist_ok=True)

        if not NGA_WPI_CACHE.exists():
            logger.info("NGA WPI cache not found — downloading...")
            try:
                resp = requests.get(NGA_WPI_URL, timeout=60)
                resp.raise_for_status()
                NGA_WPI_CACHE.write_bytes(resp.content)
                logger.info("NGA WPI downloaded and cached → %s", NGA_WPI_CACHE)
            except Exception as exc:
                logger.warning(
                    "NGA WPI download failed (%s). "
                    "dist_to_nearest_port will be -1.0 for all predictions.",
                    exc,
                )
                return None

        wpi = pl.read_csv(NGA_WPI_CACHE, ignore_errors=True)

        # Column name varies across WPI versions — find flexibly
        lat_col = next(
            (c for c in wpi.columns if "LAT" in c.upper() and "DEG" not in c.upper()),
            next((c for c in wpi.columns if "LAT" in c.upper()), None),
        )
        lon_col = next(
            (c for c in wpi.columns if "LONGI" in c.upper() and "DEG" not in c.upper()),
            next((c for c in wpi.columns if "LONGI" in c.upper()), None),
        )

        if lat_col is None or lon_col is None:
            logger.warning(
                "Could not find LAT/LON columns in NGA WPI CSV. "
                "dist_to_nearest_port will be -1.0 for all predictions."
            )
            return None

        lats = wpi[lat_col].cast(pl.Float64, strict=False).to_numpy()
        lons = wpi[lon_col].cast(pl.Float64, strict=False).to_numpy()

        valid = (
            ~np.isnan(lats)
            & ~np.isnan(lons)
            & (lats >= -90)
            & (lats <= 90)
            & (lons >= -180)
            & (lons <= 180)
        )
        lats, lons = lats[valid], lons[valid]

        coords_rad = np.radians(np.column_stack([lats, lons]))
        tree = BallTree(coords_rad, metric="haversine")

        logger.info("NGA WPI BallTree built: %d valid ports.", len(lats))
        return tree

    except Exception as exc:
        logger.warning(
            "Failed to build port BallTree (%s). "
            "dist_to_nearest_port will be -1.0 for all predictions.",
            exc,
        )
        return None


def load_model() -> ModelBundle:
    """
    Load all model artefacts and assemble a ModelBundle.
    Called once at application startup and stored in app.state.

    Raises
    ------
    FileNotFoundError  : if either checkpoint pkl is missing
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}. "
            "Place xgboost_best.pkl in app/models/checkpoints/."
        )
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler checkpoint not found: {SCALER_PATH}. "
            "Place feature_scaler.pkl in app/models/checkpoints/."
        )

    # --- Model artifact ---
    logger.info("Loading model from %s ...", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        model_bundle_raw = pickle.load(f)

    model = model_bundle_raw["model"]
    feature_cols = model_bundle_raw["feature_cols"]

    logger.info("Model loaded. Features: %d", len(feature_cols))

    # --- Scaler artifact ---
    logger.info("Loading scaler from %s ...", SCALER_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler_bundle = pickle.load(f)

    scaler = scaler_bundle["scaler"]
    scaler_cols = scaler_bundle["cols"]
    train_medians = scaler_bundle["train_medians"]

    logger.info("Scaler loaded. Scaled columns: %d", len(scaler_cols))

    # --- NGA WPI BallTree ---
    logger.info("Building NGA WPI BallTree for dist_to_nearest_port...")
    port_tree = _load_port_tree()
    if port_tree is None:
        logger.warning(
            "Port tree unavailable — dist_to_nearest_port will be -1.0. "
            "This will degrade ANCHORED vs DRIFTING separation. "
            "Ensure NGA WPI CSV is reachable or pre-cached."
        )

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
        port_tree=port_tree,
        metadata=metadata,
    )
