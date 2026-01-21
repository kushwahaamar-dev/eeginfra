"""
Self-supervised pretraining subpackage.

Provides contrastive learning and masked signal modeling.
"""

from neuroformer.pretraining.contrastive import (
    ContrastiveLoss,
    SimCLREEGAugmentor,
    ContrastivePretrainer,
)
from neuroformer.pretraining.masked import (
    MaskedSignalModeling,
    MaskedPretrainer,
)

__all__ = [
    "ContrastiveLoss",
    "SimCLREEGAugmentor",
    "ContrastivePretrainer",
    "MaskedSignalModeling",
    "MaskedPretrainer",
]
