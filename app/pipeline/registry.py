"""
Pipeline registry.

Maps a model_type string (stored in ModelMetadata and in every PredictionDB row)
to the correct (FeaturePipeline, BasePredictor) pair.

Adding a new model
------------------
1. Create MyPipeline(FeaturePipeline) and MyPredictor(BasePredictor) in a
   new file under app/pipeline/.
2. Add a single entry to REGISTRY below.
3. No changes anywhere else — InferenceService, API schemas, DB models all
   remain untouched.
"""

from typing import NamedTuple

from app.pipeline.base import BasePredictor, FeaturePipeline
from app.pipeline.xgboost_pipeline import XGBoostPipeline, XGBoostPredictor


class PipelinePair(NamedTuple):
    pipeline: FeaturePipeline
    predictor: BasePredictor


# Instantiated once at import time — pipelines are stateless and thread-safe.
REGISTRY: dict[str, PipelinePair] = {
    "xgboost": PipelinePair(
        pipeline=XGBoostPipeline(),
        predictor=XGBoostPredictor(),
    ),
    # Future entries — uncomment when Phase 5 pipeline is implemented:
    # "patchtst": PipelinePair(
    #     pipeline=PatchTSTPipeline(),
    #     predictor=PatchTSTPredictor(),
    # ),
}


def get_pipeline_pair(model_type: str) -> PipelinePair:
    """
    Retrieve the (FeaturePipeline, BasePredictor) pair for a given model type.

    Raises
    ------
    KeyError : if model_type is not registered. This is a programming error
               (model_loader should always produce a registered model_type).
    """
    if model_type not in REGISTRY:
        raise KeyError(
            f"No pipeline registered for model_type='{model_type}'. "
            f"Registered types: {list(REGISTRY.keys())}"
        )
    return REGISTRY[model_type]
