"""
Pipeline abstraction layer.

Two ABCs define the internal contract between InferenceService and any model:

    FeaturePipeline.compute(window, bundle) → features
    BasePredictor.predict(features, bundle) → probabilities (shape 4,)

Adding a new model (e.g. PatchTST) means:
    1. Create a new subclass of FeaturePipeline
    2. Create a new subclass of BasePredictor
    3. Register both in registry.py under a new model_type key

The API layer (schemas, endpoints, responses) never changes.
"""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

from app.db.models import AISMessageDB
from app.models.model_bundle import ModelBundle


class FeaturePipeline(ABC):
    """
    Transforms a sliding window of raw AIS messages into the feature
    representation expected by the corresponding model.

    Parameters
    ----------
    window  : List[AISMessageDB], sorted ascending by timestamp.
              Always contains at least 1 row (the current message).
              Maximum length == InferenceService.WINDOW_SIZE (30).
    bundle  : ModelBundle loaded at startup. Contains scaler, medians,
              port_tree, and feature column definitions.

    Returns
    -------
    Any     : Model-specific feature representation.
              - XGBoost  → np.ndarray of shape (1, n_features), float32, scaled
              - Transformer (future) → torch.Tensor of shape (seq_len, d_model)
    """

    @abstractmethod
    def compute(self, window: List[AISMessageDB], bundle: ModelBundle) -> Any: ...


class BasePredictor(ABC):
    """
    Runs model inference on pre-computed features.

    Parameters
    ----------
    features : Output of the corresponding FeaturePipeline.compute()
    bundle   : ModelBundle loaded at startup.

    Returns
    -------
    np.ndarray : Probability array of shape (4,), dtype float32.
                 Order: [MOVING, DOCKED, DRIFTING, ANCHORED]
                 Values sum to 1.0.
    """

    @abstractmethod
    def predict(self, features: Any, bundle: ModelBundle) -> np.ndarray: ...
