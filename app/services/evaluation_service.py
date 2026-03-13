"""
Evaluation service.

Triggered on every ground truth submission. Computes full classification
metrics over all predictions that have a matching ground truth label,
then:
    1. Inserts an EvaluationSnapshotDB row (persistent — queryable later)
    2. Logs a structured JSON record to logs/evaluation.log

The client receives only a simple acknowledgement — metrics are internal.
"""

import json
import logging
import logging.handlers
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import EvaluationSnapshotDB, GroundTruthDB, PredictionDB

# ---------------------------------------------------------------------------
# Evaluation logger — writes structured JSON lines to logs/evaluation.log
# Rotates at 10 MB, keeps 5 backups. Completely separate from the app logger
# so evaluation records are never mixed with request/debug output.
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_eval_logger = logging.getLogger("ais.evaluation")
_eval_logger.setLevel(logging.INFO)
_eval_logger.propagate = False  # do not bubble up to root logger

if not _eval_logger.handlers:
    _handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "evaluation.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _eval_logger.addHandler(_handler)

LABELS_ORDERED = ["MOVING", "DOCKED", "DRIFTING", "ANCHORED"]
N_CLASSES = len(LABELS_ORDERED)


class EvaluationService:
    """
    Computes and persists evaluation metrics after each ground truth submission.
    """

    def __init__(self, db: Session):
        self.db = db

    def compute_and_store(self) -> None:
        """
        1. Join predictions with ground truth.
        2. Compute classification metrics.
        3. Persist snapshot to DB.
        4. Log JSON record to evaluation.log.

        Silently returns if fewer than 2 classes are represented in ground
        truth (metrics are not meaningful yet — this happens in the very
        early phase when only a handful of labels have been submitted).
        """
        rows = self._fetch_evaluated_rows()

        if len(rows) == 0:
            return

        y_true = np.array([r["true_class"] for r in rows], dtype=np.int32)
        y_pred = np.array([r["predicted_class"] for r in rows], dtype=np.int32)

        # Need at least 2 distinct classes for meaningful metrics
        if len(np.unique(y_true)) < 2:
            _eval_logger.info(
                json.dumps(
                    {
                        "event": "skipped",
                        "reason": "fewer_than_2_classes_in_ground_truth",
                        "total_evaluated": len(rows),
                    }
                )
            )
            return

        metrics = self._compute_metrics(y_true, y_pred)
        total_preds = self._count_total_predictions()
        total_evaluated = len(rows)
        coverage_pct = (
            round(total_evaluated / total_preds * 100, 2) if total_preds > 0 else 0.0
        )

        self._store_snapshot(metrics, total_preds, total_evaluated, coverage_pct)
        self._log(metrics, total_preds, total_evaluated, coverage_pct)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_evaluated_rows(self) -> list[dict]:
        """
        Inner join: predictions that have a corresponding ground truth row.
        Returns list of dicts with true_class and predicted_class.
        """
        stmt = select(
            PredictionDB.predicted_class,
            GroundTruthDB.true_class,
        ).join(GroundTruthDB, PredictionDB.id == GroundTruthDB.prediction_id)
        result = self.db.execute(stmt).all()
        return [{"predicted_class": r[0], "true_class": r[1]} for r in result]

    def _count_total_predictions(self) -> int:
        result = self.db.execute(
            select(func.count()).select_from(PredictionDB)
        ).scalar()
        return int(result or 0)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute all classification metrics. Labels present in data only."""
        labels = list(range(N_CLASSES))

        f1_macro = float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        )
        f1_weighted = float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        )
        prec_macro = float(
            precision_score(
                y_true, y_pred, average="macro", labels=labels, zero_division=0
            )
        )
        rec_macro = float(
            recall_score(
                y_true, y_pred, average="macro", labels=labels, zero_division=0
            )
        )

        f1_per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        prec_per = precision_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )
        rec_per = recall_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        per_class = {}
        for i, label in enumerate(LABELS_ORDERED):
            per_class[label] = {
                "f1": round(float(f1_per[i]), 4),
                "precision": round(float(prec_per[i]), 4),
                "recall": round(float(rec_per[i]), 4),
            }

        return {
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "per_class": per_class,
            "confusion_matrix": cm,
        }

    def _store_snapshot(
        self,
        metrics: dict,
        total_predictions: int,
        total_evaluated: int,
        coverage_pct: float,
    ) -> None:
        pc = metrics["per_class"]
        snapshot = EvaluationSnapshotDB(
            total_predictions=total_predictions,
            total_evaluated=total_evaluated,
            coverage_pct=coverage_pct,
            f1_macro=metrics["f1_macro"],
            f1_weighted=metrics["f1_weighted"],
            precision_macro=metrics["precision_macro"],
            recall_macro=metrics["recall_macro"],
            f1_moving=pc["MOVING"]["f1"],
            f1_docked=pc["DOCKED"]["f1"],
            f1_drifting=pc["DRIFTING"]["f1"],
            f1_anchored=pc["ANCHORED"]["f1"],
            precision_moving=pc["MOVING"]["precision"],
            precision_docked=pc["DOCKED"]["precision"],
            precision_drifting=pc["DRIFTING"]["precision"],
            precision_anchored=pc["ANCHORED"]["precision"],
            recall_moving=pc["MOVING"]["recall"],
            recall_docked=pc["DOCKED"]["recall"],
            recall_drifting=pc["DRIFTING"]["recall"],
            recall_anchored=pc["ANCHORED"]["recall"],
            confusion_matrix_json=json.dumps(metrics["confusion_matrix"]),
        )
        self.db.add(snapshot)
        self.db.commit()

    def _log(
        self,
        metrics: dict,
        total_predictions: int,
        total_evaluated: int,
        coverage_pct: float,
    ) -> None:
        record = {
            "event": "evaluation_snapshot",
            "total_predictions": total_predictions,
            "total_evaluated": total_evaluated,
            "coverage_pct": coverage_pct,
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
        }
        _eval_logger.info(json.dumps(record))
