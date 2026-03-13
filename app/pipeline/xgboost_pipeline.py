"""
XGBoost feature pipeline.

Reproduces the exact Phase 3 feature engineering steps for a single
sliding window of AIS messages. Every computation mirrors the offline
training pipeline — same formulas, same boundary conditions, same fallbacks.

Phase 3 steps implemented here
-------------------------------
1.  time_delta         — seconds since previous message, capped at 3600s
2.  Kinematic          — sog_diff, cog_diff (wrapped), acceleration
3.  Spatial            — dist_from_prev (Geod WGS84), bearing_change, displacement_rate
4.  Temporal           — hour_of_day, day_of_week (UTC)
5.  Rolling N=10,30    — sog/cog/rot/pos stats, backwards only
6.  Ship type encoding — raw AIS code → 11 IMO category integer
7.  Static imputation  — dim_bow/stern/port/starboard with train medians
8.  dist_to_nearest_port — BallTree haversine query (or -1.0 fallback)
9.  Feature alignment  — exact FEATURE_COLS order from bundle
10. Scaling            — StandardScaler applied to scaler_cols only

Key invariants (matching Phase 3 exactly)
-----------------------------------------
- Segment boundary: time_delta == 0 OR time_delta >= 3600s →
  sog_diff, cog_diff, acceleration, dist_from_prev, bearing_change,
  displacement_rate are all zeroed for that row.
- cog_diff and bearing_change are wrapped to [-180, 180].
- Rolling stats use min_periods=1 for mean/min/max, min_periods=2 for std
  (single-point windows return 0.0 for std).
- pos_spread = lat_std + lon_std (NOT Euclidean distance).
- dist_to_nearest_port = -1.0 when port_tree is None.
"""

import logging
from typing import List

import numpy as np
from pyproj import Geod

from app.db.models import AISMessageDB
from app.models.model_bundle import ModelBundle
from app.pipeline.base import BasePredictor, FeaturePipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match Phase 3)
# ---------------------------------------------------------------------------
TIME_DELTA_CAP = 3600.0  # seconds — gaps > 1h treated as new segment
EARTH_RADIUS_M = 6_371_008.8  # for BallTree haversine → metres conversion

# Ship type → 11 IMO category mapping (identical to Phase 3)
_SHIP_TYPE_MAP: dict[int, int] = {}
_RANGES = {
    0: range(0, 20),
    1: range(20, 30),
    2: [30],
    3: [31, 32],
    4: range(33, 40),
    5: range(40, 50),
    6: range(50, 60),
    7: range(60, 70),
    8: range(70, 80),
    9: range(80, 90),
    10: range(90, 100),
}
for _cat, _codes in _RANGES.items():
    for _code in _codes:
        _SHIP_TYPE_MAP[_code] = _cat


def _encode_ship_type(raw: int | None) -> int:
    """Map raw AIS ship_type code to 11-category integer. Null/unknown → 0."""
    if raw is None:
        return 0
    return _SHIP_TYPE_MAP.get(raw, 0)


def _rolling_mean(arr: np.ndarray, n: int) -> float:
    subset = arr[-n:]
    return float(np.mean(subset))


def _rolling_std(arr: np.ndarray, n: int) -> float:
    subset = arr[-n:]
    if len(subset) < 2:
        return 0.0
    return float(np.std(subset, ddof=1))


def _rolling_min(arr: np.ndarray, n: int) -> float:
    return float(np.min(arr[-n:]))


def _rolling_max(arr: np.ndarray, n: int) -> float:
    return float(np.max(arr[-n:]))


def _wrap_to_180(angle: float) -> float:
    """Wrap an angle to the range [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


class XGBoostPipeline(FeaturePipeline):
    """
    Full Phase 3 feature engineering pipeline for XGBoost inference.

    Operates on a sorted window of up to 30 AISMessageDB rows and returns
    a (1, n_features) numpy array ready for XGBClassifier.predict_proba().
    """

    def compute(self, window: List[AISMessageDB], bundle: ModelBundle) -> np.ndarray:
        """
        Execute the full feature engineering pipeline.

        Parameters
        ----------
        window : Ascending-timestamp-sorted list of AISMessageDB rows.
                 Length is 1..WINDOW_SIZE (30).
        bundle : ModelBundle from app state.

        Returns
        -------
        np.ndarray : shape (1, n_features), dtype float32, scaled.
        """
        n = len(window)

        # ------------------------------------------------------------------
        # 1. Extract raw arrays from window
        # ------------------------------------------------------------------
        lats = np.array([r.lat for r in window], dtype=np.float64)
        lons = np.array([r.lon for r in window], dtype=np.float64)
        sogs = np.array([r.sog for r in window], dtype=np.float64)
        cogs = np.array([r.cog for r in window], dtype=np.float64)
        rots = np.array([r.rot for r in window], dtype=np.float64)
        headings = np.array([r.true_heading for r in window], dtype=np.float64)

        timestamps = [r.timestamp for r in window]

        # ------------------------------------------------------------------
        # 2. time_delta — seconds since previous message, capped at 3600s
        #    First row in window → 0 (mirrors Phase 3 first-row-per-vessel logic)
        # ------------------------------------------------------------------
        time_deltas = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            diff = (timestamps[i] - timestamps[i - 1]).total_seconds()
            # Clamp: negative diff (data anomaly) → 0; gap > cap → cap
            time_deltas[i] = float(np.clip(diff, 0.0, TIME_DELTA_CAP))

        # ------------------------------------------------------------------
        # 3. Segment boundary mask
        #    time_delta == 0  → first row OR data anomaly
        #    time_delta >= cap → new segment (>1h gap)
        #    At boundaries: kinematic + spatial features are zeroed.
        # ------------------------------------------------------------------
        is_boundary = (time_deltas == 0.0) | (time_deltas >= TIME_DELTA_CAP)

        # ------------------------------------------------------------------
        # 4. Kinematic features (for last row — uses current vs previous)
        # ------------------------------------------------------------------
        last = n - 1
        prev = last - 1 if n > 1 else None

        if n == 1 or is_boundary[last]:
            sog_diff = 0.0
            cog_diff = 0.0
            acceleration = 0.0
        else:
            sog_diff = float(sogs[last] - sogs[prev])
            cog_diff = _wrap_to_180(float(cogs[last] - cogs[prev]))
            td = time_deltas[last]
            acceleration = sog_diff / td if td > 0 else 0.0

        # ------------------------------------------------------------------
        # 5. Spatial features (for last row — pyproj.Geod WGS84)
        # ------------------------------------------------------------------
        geod = Geod(ellps="WGS84")

        if n == 1 or is_boundary[last]:
            dist_from_prev = 0.0
            bearing_change = 0.0
            displacement_rate = 0.0
        else:
            # Current leg: prev → last
            fwd_az_curr, _, dist = geod.inv(
                lons[prev],
                lats[prev],
                lons[last],
                lats[last],
            )
            dist_from_prev = float(abs(dist))

            td = time_deltas[last]
            displacement_rate = dist_from_prev / td if td > 0 else 0.0

            # Bearing change: need at least 3 non-boundary points
            prev2 = prev - 1 if prev > 0 else None
            if prev2 is not None and not is_boundary[prev]:
                fwd_az_prev, _, _ = geod.inv(
                    lons[prev2],
                    lats[prev2],
                    lons[prev],
                    lats[prev],
                )
                bearing_change = _wrap_to_180(fwd_az_curr - fwd_az_prev)
            else:
                bearing_change = 0.0

        # ------------------------------------------------------------------
        # 6. Temporal features — derived from last row's timestamp (UTC)
        # ------------------------------------------------------------------
        ts = timestamps[last]
        # Ensure timezone-aware for UTC extraction
        import datetime as dt_module

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt_module.timezone.utc)
        hour_of_day = ts.hour  # 0–23
        day_of_week = ts.weekday()  # 0=Monday … 6=Sunday

        # ------------------------------------------------------------------
        # 7. Rolling window statistics (backwards only — last N rows)
        #    N=10: last 10 rows (or fewer if window shorter)
        #    N=30: all rows in window (up to 30)
        #    Mirrors Phase 3 rolling_mean/std with min_periods=1/2.
        # ------------------------------------------------------------------
        def _window_stats(arr: np.ndarray, w: int) -> dict:
            subset = arr[-w:]
            mean = float(np.mean(subset))
            std = float(np.std(subset, ddof=1)) if len(subset) >= 2 else 0.0
            min_ = float(np.min(subset))
            max_ = float(np.max(subset))
            return mean, std, min_, max_

        sog_mean_10, sog_std_10, sog_min_10, sog_max_10 = _window_stats(sogs, 10)
        sog_mean_30, sog_std_30, sog_min_30, sog_max_30 = _window_stats(sogs, 30)

        cog_std_10 = float(np.std(cogs[-10:], ddof=1)) if min(n, 10) >= 2 else 0.0
        cog_std_30 = float(np.std(cogs[-30:], ddof=1)) if min(n, 30) >= 2 else 0.0

        rot_mean_10 = float(np.mean(rots[-10:]))
        rot_std_10 = float(np.std(rots[-10:], ddof=1)) if min(n, 10) >= 2 else 0.0
        rot_mean_30 = float(np.mean(rots[-30:]))
        rot_std_30 = float(np.std(rots[-30:], ddof=1)) if min(n, 30) >= 2 else 0.0

        # pos_spread = lat_std + lon_std (Phase 3 formula exactly)
        def _pos_spread(w: int) -> float:
            lat_sub = lats[-w:]
            lon_sub = lons[-w:]
            if len(lat_sub) < 2:
                return 0.0
            return float(np.std(lat_sub, ddof=1)) + float(np.std(lon_sub, ddof=1))

        pos_spread_10 = _pos_spread(10)
        pos_spread_30 = _pos_spread(30)

        # ------------------------------------------------------------------
        # 8. Ship type encoding (last row — static per vessel)
        # ------------------------------------------------------------------
        ship_type_encoded = _encode_ship_type(window[last].ship_type)

        # ------------------------------------------------------------------
        # 9. Static vessel dimensions — use row value if present, else median
        # ------------------------------------------------------------------
        medians = bundle.train_medians

        def _static(val: float | None, col: str) -> float:
            if val is None or val == 0.0:
                return float(medians.get(col, 0.0))
            return float(val)

        last_row = window[last]
        draught = _static(last_row.draught, "draught")
        dim_bow = _static(last_row.dim_bow, "dim_bow")
        dim_stern = _static(last_row.dim_stern, "dim_stern")
        dim_port = _static(last_row.dim_port, "dim_port")
        dim_starboard = _static(last_row.dim_starboard, "dim_starboard")

        # ------------------------------------------------------------------
        # 10. dist_to_nearest_port — BallTree haversine query
        #     Falls back to -1.0 when port_tree is None (matches Phase 3)
        # ------------------------------------------------------------------
        if bundle.port_tree is not None:
            lat_rad = np.radians([[lats[last], lons[last]]])
            dist_rad, _ = bundle.port_tree.query(lat_rad, k=1)
            dist_to_nearest_port = float(dist_rad[0, 0] * EARTH_RADIUS_M)
        else:
            dist_to_nearest_port = -1.0

        # ------------------------------------------------------------------
        # 11. Assemble raw feature dict
        # ------------------------------------------------------------------
        raw: dict[str, float] = {
            # Raw AIS signals
            "sog": float(sogs[last]),
            "cog": float(cogs[last]),
            "rot": float(rots[last]),
            "true_heading": float(headings[last]),
            # Kinematic
            "sog_diff": sog_diff,
            "cog_diff": cog_diff,
            "acceleration": acceleration,
            # Spatial
            "dist_from_prev": dist_from_prev,
            "bearing_change": bearing_change,
            "displacement_rate": displacement_rate,
            # Temporal
            "time_delta": float(time_deltas[last]),
            "hour_of_day": float(hour_of_day),
            "day_of_week": float(day_of_week),
            # Static
            "draught": draught,
            "dim_bow": dim_bow,
            "dim_stern": dim_stern,
            "dim_port": dim_port,
            "dim_starboard": dim_starboard,
            # Ship type
            "ship_type_encoded": float(ship_type_encoded),
            # Contextual
            "dist_to_nearest_port": dist_to_nearest_port,
            # Rolling N=10
            "sog_mean_10": sog_mean_10,
            "sog_std_10": sog_std_10,
            "sog_min_10": sog_min_10,
            "sog_max_10": sog_max_10,
            "cog_std_10": cog_std_10,
            "rot_mean_10": rot_mean_10,
            "rot_std_10": rot_std_10,
            "pos_spread_10": pos_spread_10,
            # Rolling N=30
            "sog_mean_30": sog_mean_30,
            "sog_std_30": sog_std_30,
            "sog_min_30": sog_min_30,
            "sog_max_30": sog_max_30,
            "cog_std_30": cog_std_30,
            "rot_mean_30": rot_mean_30,
            "rot_std_30": rot_std_30,
            "pos_spread_30": pos_spread_30,
        }

        # ------------------------------------------------------------------
        # 12. Align to exact FEATURE_COLS order, fill any unknown col with
        #     its train median (safety net — should never be needed).
        # ------------------------------------------------------------------
        feature_cols = bundle.feature_cols
        X = np.array(
            [raw.get(col, float(medians.get(col, 0.0))) for col in feature_cols],
            dtype=np.float32,
        ).reshape(1, -1)

        # Replace any residual inf/nan (data anomalies) with 0.0
        X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)

        # ------------------------------------------------------------------
        # 13. Scale — StandardScaler applied to scaler_cols only.
        #     Categoricals (ship_type_encoded, hour_of_day, day_of_week)
        #     are intentionally excluded (matching Phase 3 exactly).
        # ------------------------------------------------------------------
        if bundle.scaler is not None:
            scaler_idx = [
                feature_cols.index(c) for c in bundle.scaler_cols if c in feature_cols
            ]
            X[:, scaler_idx] = bundle.scaler.transform(X[:, scaler_idx])

        return X


class XGBoostPredictor(BasePredictor):
    """
    Runs XGBClassifier inference and returns a (4,) probability array.
    """

    def predict(self, features: np.ndarray, bundle: ModelBundle) -> np.ndarray:
        probs = bundle.model.predict_proba(features)[0]
        return probs.astype(np.float32)
