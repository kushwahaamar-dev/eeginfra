"""
Inference subpackage for NeuroFormer.

Provides prediction and explainability utilities.
"""

from neuroformer.inference.predictor import Predictor
from neuroformer.inference.explainability import (
    AttentionVisualizer,
    compute_feature_importance,
)

__all__ = [
    "Predictor",
    "AttentionVisualizer",
    "compute_feature_importance",
]
