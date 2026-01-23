"""
Validation utilities for NeuroFormer.

Provides comprehensive input validation for data, configs, and models.
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple, List, Any
from pathlib import Path

from neuroformer.utils.exceptions import (
    DataValidationError,
    ModelConfigError,
    CheckpointError,
)


def validate_eeg_data(
    data: Union[np.ndarray, torch.Tensor],
    expected_electrodes: int = 19,
    expected_bands: int = 5,
    allow_batch: bool = True,
    name: str = "data"
) -> torch.Tensor:
    """
    Validate and normalize EEG data input.
    
    Args:
        data: Input data array or tensor
        expected_electrodes: Expected number of electrodes
        expected_bands: Expected number of frequency bands
        allow_batch: Whether batch dimension is allowed
        name: Name for error messages
        
    Returns:
        Validated torch.Tensor
        
    Raises:
        DataValidationError: If validation fails
    """
    # Convert to tensor if numpy
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        data = data.float()
    else:
        raise DataValidationError(
            f"Invalid data type for {name}",
            expected="np.ndarray or torch.Tensor",
            got=type(data).__name__
        )
    
    # Check for NaN/Inf
    if torch.isnan(data).any():
        raise DataValidationError(f"{name} contains NaN values")
    if torch.isinf(data).any():
        raise DataValidationError(f"{name} contains Inf values")
    
    # Check dimensions
    if data.dim() == 2:
        # (bands, electrodes) - add batch dimension
        if data.shape[0] != expected_bands:
            raise DataValidationError(
                f"Invalid number of bands in {name}",
                expected=expected_bands,
                got=data.shape[0],
                field="bands"
            )
        if data.shape[1] != expected_electrodes:
            raise DataValidationError(
                f"Invalid number of electrodes in {name}",
                expected=expected_electrodes,
                got=data.shape[1],
                field="electrodes"
            )
        data = data.unsqueeze(0)
        
    elif data.dim() == 3:
        if not allow_batch:
            raise DataValidationError(
                f"{name} should not have batch dimension"
            )
        if data.shape[1] != expected_bands:
            raise DataValidationError(
                f"Invalid number of bands in {name}",
                expected=expected_bands,
                got=data.shape[1],
                field="bands"
            )
        if data.shape[2] != expected_electrodes:
            raise DataValidationError(
                f"Invalid number of electrodes in {name}",
                expected=expected_electrodes,
                got=data.shape[2],
                field="electrodes"
            )
    else:
        raise DataValidationError(
            f"Invalid dimensions for {name}",
            expected="2D or 3D",
            got=f"{data.dim()}D"
        )
    
    return data


def validate_labels(
    labels: Union[np.ndarray, torch.Tensor, List],
    num_classes: int = 7,
    batch_size: Optional[int] = None,
    name: str = "labels"
) -> torch.Tensor:
    """
    Validate classification labels.
    
    Args:
        labels: Input labels
        num_classes: Number of valid classes
        batch_size: Expected batch size (optional)
        name: Name for error messages
        
    Returns:
        Validated torch.LongTensor
        
    Raises:
        DataValidationError: If validation fails
    """
    # Convert to tensor
    if isinstance(labels, (list, tuple)):
        labels = torch.tensor(labels, dtype=torch.long)
    elif isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()
    elif isinstance(labels, torch.Tensor):
        labels = labels.long()
    else:
        raise DataValidationError(
            f"Invalid type for {name}",
            expected="array-like",
            got=type(labels).__name__
        )
    
    # Check dimension
    if labels.dim() != 1:
        raise DataValidationError(
            f"{name} should be 1D",
            expected="1D",
            got=f"{labels.dim()}D"
        )
    
    # Check batch size
    if batch_size is not None and labels.size(0) != batch_size:
        raise DataValidationError(
            f"Batch size mismatch for {name}",
            expected=batch_size,
            got=labels.size(0)
        )
    
    # Check valid range
    if labels.min() < 0 or labels.max() >= num_classes:
        raise DataValidationError(
            f"Invalid label values in {name}",
            expected=f"0 to {num_classes - 1}",
            got=f"{labels.min().item()} to {labels.max().item()}"
        )
    
    return labels


def validate_config(config: Any) -> None:
    """
    Validate NeuroFormerConfig parameters.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ModelConfigError: If configuration is invalid
    """
    # Check required attributes
    required = [
        'num_electrodes', 'num_classes', 'd_model',
        'n_heads', 'n_layers', 'dropout'
    ]
    
    for attr in required:
        if not hasattr(config, attr):
            raise ModelConfigError(
                f"Missing required config parameter",
                param=attr
            )
    
    # Validate values
    if config.num_electrodes <= 0:
        raise ModelConfigError(
            "num_electrodes must be positive",
            param="num_electrodes",
            value=config.num_electrodes
        )
    
    if config.num_classes <= 1:
        raise ModelConfigError(
            "num_classes must be > 1",
            param="num_classes",
            value=config.num_classes
        )
    
    if config.d_model <= 0:
        raise ModelConfigError(
            "d_model must be positive",
            param="d_model",
            value=config.d_model
        )
    
    if config.d_model % config.n_heads != 0:
        raise ModelConfigError(
            "d_model must be divisible by n_heads",
            param="d_model/n_heads",
            value=f"{config.d_model}/{config.n_heads}"
        )
    
    if not 0 <= config.dropout < 1:
        raise ModelConfigError(
            "dropout must be in [0, 1)",
            param="dropout",
            value=config.dropout
        )


def validate_checkpoint(path: str) -> dict:
    """
    Validate and load a checkpoint file.
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        Loaded checkpoint dictionary
        
    Raises:
        CheckpointError: If checkpoint is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise CheckpointError(
            "Checkpoint file not found",
            path=str(path)
        )
    
    if not path.suffix in ['.pth', '.pt', '.ckpt']:
        raise CheckpointError(
            "Invalid checkpoint file extension",
            path=str(path)
        )
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except Exception as e:
        raise CheckpointError(
            f"Failed to load checkpoint: {e}",
            path=str(path)
        )
    
    # Check required keys
    if 'model_state_dict' not in checkpoint:
        raise CheckpointError(
            "Checkpoint missing 'model_state_dict'",
            path=str(path)
        )
    
    return checkpoint


def check_device(device: str = 'auto') -> str:
    """
    Check and resolve device specification.
    
    Args:
        device: Device string ('auto', 'cuda', 'cpu', 'cuda:0', etc.)
        
    Returns:
        Resolved device string
        
    Raises:
        ModelConfigError: If device is invalid or unavailable
    """
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ModelConfigError(
                "CUDA requested but not available",
                param="device",
                value=device
            )
        
        # Check specific device index
        if ':' in device:
            try:
                idx = int(device.split(':')[1])
                if idx >= torch.cuda.device_count():
                    raise ModelConfigError(
                        f"CUDA device {idx} not available",
                        param="device",
                        value=f"available: 0-{torch.cuda.device_count()-1}"
                    )
            except ValueError:
                raise ModelConfigError(
                    "Invalid CUDA device specification",
                    param="device",
                    value=device
                )
    
    elif device not in ['cpu', 'mps']:
        raise ModelConfigError(
            "Unknown device type",
            param="device",
            value=device
        )
    
    return device


def validate_sampling_rate(rate: int, min_rate: int = 64, max_rate: int = 2048) -> None:
    """
    Validate EEG sampling rate.
    
    Args:
        rate: Sampling rate in Hz
        min_rate: Minimum allowed rate
        max_rate: Maximum allowed rate
        
    Raises:
        DataValidationError: If rate is invalid
    """
    if not isinstance(rate, int):
        raise DataValidationError(
            "Sampling rate must be integer",
            expected="int",
            got=type(rate).__name__
        )
    
    if rate < min_rate or rate > max_rate:
        raise DataValidationError(
            "Sampling rate out of valid range",
            expected=f"{min_rate}-{max_rate} Hz",
            got=f"{rate} Hz"
        )
