"""
Training subpackage for NeuroFormer.

Provides trainer, losses, and evaluation metrics.
"""

from neuroformer.training.losses import (
    FocalLoss,
    LabelSmoothingLoss,
    CombinedLoss,
)
from neuroformer.training.metrics import (
    compute_accuracy,
    compute_f1,
    compute_balanced_accuracy,
    compute_confusion_matrix,
)
from neuroformer.training.trainer import Trainer

__all__ = [
    "FocalLoss",
    "LabelSmoothingLoss", 
    "CombinedLoss",
    "compute_accuracy",
    "compute_f1",
    "compute_balanced_accuracy",
    "compute_confusion_matrix",
    "Trainer",
]
