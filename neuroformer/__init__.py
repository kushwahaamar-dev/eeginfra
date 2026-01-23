"""
NeuroFormer: State-of-the-art EEG Analysis Framework
=====================================================

A cutting-edge deep learning framework for psychiatric disorder classification
from EEG data, featuring hybrid Transformer-GNN architectures.

Features:
- Hybrid Transformer-GNN architecture
- Self-supervised pretraining (contrastive + masked)
- Comprehensive data augmentation
- Real-time inference support
- Built-in explainability tools
"""

__version__ = "0.2.0"
__author__ = "Amar Kushwaha"

from neuroformer.config import NeuroFormerConfig, DataConfig, InferenceConfig

# Lazy imports for performance
def __getattr__(name: str):
    """Lazy import for heavy modules."""
    if name == "NeuroFormer":
        from neuroformer.models.neuroformer import NeuroFormer
        return NeuroFormer
    elif name == "NeuroFormerLite":
        from neuroformer.models.neuroformer import NeuroFormerLite
        return NeuroFormerLite
    elif name == "Trainer":
        from neuroformer.training.trainer import Trainer
        return Trainer
    elif name == "Predictor":
        from neuroformer.inference.predictor import Predictor
        return Predictor
    raise AttributeError(f"module 'neuroformer' has no attribute '{name}'")

__all__ = [
    "__version__",
    "NeuroFormerConfig",
    "DataConfig",
    "InferenceConfig",
    "NeuroFormer",
    "NeuroFormerLite",
    "Trainer",
    "Predictor",
]

