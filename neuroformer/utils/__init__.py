"""
Utility functions and classes for NeuroFormer.

Provides logging, validation, and error handling.
"""

from neuroformer.utils.logging import get_logger, setup_logging
from neuroformer.utils.validation import (
    validate_eeg_data,
    validate_config,
    check_device,
)
from neuroformer.utils.exceptions import (
    NeuroFormerError,
    DataValidationError,
    ModelConfigError,
    CheckpointError,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "validate_eeg_data",
    "validate_config",
    "check_device",
    "NeuroFormerError",
    "DataValidationError",
    "ModelConfigError",
    "CheckpointError",
]
