"""
Data augmentation strategies for EEG data.

Implements various augmentation techniques to improve model generalization
and prevent overfitting.
"""

import numpy as np
from typing import Tuple, Optional


def time_shift(
    data: np.ndarray,
    max_shift: int = 50,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply random time shift augmentation.
    
    Shifts the signal along the time axis and wraps around.
    
    Args:
        data: EEG data of shape (n_channels, n_samples) or (n_samples,)
        max_shift: Maximum shift in samples (positive or negative)
        random_state: Random seed for reproducibility
        
    Returns:
        Time-shifted data
    """
    rng = np.random.default_rng(random_state)
    shift = rng.integers(-max_shift, max_shift + 1)
    
    return np.roll(data, shift, axis=-1)


def add_gaussian_noise(
    data: np.ndarray,
    noise_std: float = 0.1,
    relative: bool = True,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to EEG data.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        noise_std: Standard deviation of noise
        relative: If True, noise_std is relative to signal std
        random_state: Random seed for reproducibility
        
    Returns:
        Data with added noise
    """
    rng = np.random.default_rng(random_state)
    
    if relative:
        signal_std = np.std(data, axis=-1, keepdims=True)
        signal_std = np.where(signal_std == 0, 1, signal_std)
        noise = rng.normal(0, noise_std * signal_std, data.shape)
    else:
        noise = rng.normal(0, noise_std, data.shape)
    
    return data + noise


def channel_dropout(
    data: np.ndarray,
    dropout_prob: float = 0.1,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Randomly zero out entire channels.
    
    Simulates electrode failure or poor contact during training
    to make the model robust to missing channels.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        dropout_prob: Probability of dropping each channel
        random_state: Random seed for reproducibility
        
    Returns:
        Data with some channels zeroed out
    """
    rng = np.random.default_rng(random_state)
    
    if data.ndim == 1:
        return data  # Can't dropout single channel
    
    n_channels = data.shape[0]
    mask = rng.random(n_channels) > dropout_prob
    
    result = data.copy()
    result[~mask] = 0
    
    return result


def temporal_masking(
    data: np.ndarray,
    mask_ratio: float = 0.15,
    mask_length: int = 10,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly mask temporal segments for self-supervised learning.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        mask_ratio: Ratio of time points to mask
        mask_length: Length of each masked segment
        random_state: Random seed
        
    Returns:
        Tuple of (masked_data, mask) where mask indicates masked positions
    """
    rng = np.random.default_rng(random_state)
    
    n_samples = data.shape[-1]
    n_masks = int(n_samples * mask_ratio / mask_length)
    
    mask = np.ones(n_samples, dtype=bool)
    
    for _ in range(n_masks):
        start = rng.integers(0, max(1, n_samples - mask_length))
        mask[start:start + mask_length] = False
    
    masked_data = data.copy()
    if data.ndim == 1:
        masked_data[~mask] = 0
    else:
        masked_data[:, ~mask] = 0
    
    return masked_data, mask


def mixup(
    data1: np.ndarray,
    data2: np.ndarray,
    label1: np.ndarray,
    label2: np.ndarray,
    alpha: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply mixup augmentation between two samples.
    
    Interpolates between two samples and their labels.
    
    Args:
        data1: First EEG sample
        data2: Second EEG sample
        label1: One-hot label for first sample
        label2: One-hot label for second sample
        alpha: Beta distribution parameter (higher = more mixing)
        random_state: Random seed
        
    Returns:
        Tuple of (mixed_data, mixed_label)
    """
    rng = np.random.default_rng(random_state)
    
    lam = rng.beta(alpha, alpha)
    
    mixed_data = lam * data1 + (1 - lam) * data2
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_data, mixed_label


def cutmix_temporal(
    data1: np.ndarray,
    data2: np.ndarray,
    label1: np.ndarray,
    label2: np.ndarray,
    alpha: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply CutMix augmentation in temporal dimension.
    
    Replaces a temporal segment from one sample with another.
    
    Args:
        data1: First EEG sample (n_channels, n_samples)
        data2: Second EEG sample (n_channels, n_samples)
        label1: One-hot label for first sample
        label2: One-hot label for second sample
        alpha: Beta distribution parameter
        random_state: Random seed
        
    Returns:
        Tuple of (mixed_data, mixed_label)
    """
    rng = np.random.default_rng(random_state)
    
    n_samples = data1.shape[-1]
    lam = rng.beta(alpha, alpha)
    
    cut_len = int(n_samples * (1 - lam))
    cut_start = rng.integers(0, max(1, n_samples - cut_len))
    cut_end = cut_start + cut_len
    
    mixed_data = data1.copy()
    if mixed_data.ndim == 1:
        mixed_data[cut_start:cut_end] = data2[cut_start:cut_end]
    else:
        mixed_data[:, cut_start:cut_end] = data2[:, cut_start:cut_end]
    
    # Adjust labels based on actual mixing ratio
    actual_lam = 1 - cut_len / n_samples
    mixed_label = actual_lam * label1 + (1 - actual_lam) * label2
    
    return mixed_data, mixed_label


def scaling(
    data: np.ndarray,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply random amplitude scaling.
    
    Args:
        data: EEG data
        scale_range: (min_scale, max_scale) range for random scaling
        random_state: Random seed
        
    Returns:
        Scaled data
    """
    rng = np.random.default_rng(random_state)
    scale = rng.uniform(scale_range[0], scale_range[1])
    return data * scale


def permutation(
    data: np.ndarray,
    n_segments: int = 5,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Randomly permute temporal segments.
    
    Divides the signal into segments and shuffles their order.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        n_segments: Number of segments to create and shuffle
        random_state: Random seed
        
    Returns:
        Data with permuted segments
    """
    rng = np.random.default_rng(random_state)
    
    n_samples = data.shape[-1]
    segment_len = n_samples // n_segments
    
    # Split into segments
    segments = []
    for i in range(n_segments):
        start = i * segment_len
        end = start + segment_len if i < n_segments - 1 else n_samples
        if data.ndim == 1:
            segments.append(data[start:end])
        else:
            segments.append(data[:, start:end])
    
    # Shuffle
    rng.shuffle(segments)
    
    # Concatenate
    if data.ndim == 1:
        return np.concatenate(segments)
    else:
        return np.concatenate(segments, axis=-1)
