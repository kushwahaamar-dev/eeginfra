"""
Signal filtering utilities for EEG preprocessing.

Includes bandpass filtering, notch filtering, and ICA-based artifact removal.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Union


def bandpass_filter(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    sampling_rate: int = 256,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.
    
    Args:
        data: EEG data of shape (n_channels, n_samples) or (n_samples,)
        low_freq: Lower cutoff frequency in Hz
        high_freq: Upper cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        order: Filter order
        
    Returns:
        Filtered data with same shape as input
    """
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are valid
    low = max(low, 0.001)
    high = min(high, 0.999)
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, channel) for channel in data])


def notch_filter(
    data: np.ndarray,
    notch_freq: float = 60.0,
    sampling_rate: int = 256,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove power line interference.
    
    Args:
        data: EEG data of shape (n_channels, n_samples) or (n_samples,)
        notch_freq: Frequency to notch out (typically 50 or 60 Hz)
        sampling_rate: Sampling rate in Hz
        quality_factor: Quality factor of the notch filter
        
    Returns:
        Filtered data with same shape as input
    """
    nyquist = sampling_rate / 2
    w0 = notch_freq / nyquist
    
    if w0 >= 1.0:
        return data  # Notch frequency above Nyquist, return unchanged
    
    b, a = signal.iirnotch(w0, quality_factor)
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, channel) for channel in data])


def highpass_filter(
    data: np.ndarray,
    cutoff: float = 0.5,
    sampling_rate: int = 256,
    order: int = 4
) -> np.ndarray:
    """
    Apply highpass filter to remove DC offset and slow drifts.
    
    Args:
        data: EEG data of shape (n_channels, n_samples) or (n_samples,)
        cutoff: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        order: Filter order
        
    Returns:
        Filtered data with same shape as input
    """
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff / nyquist
    
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, channel) for channel in data])


def apply_ica(
    data: np.ndarray,
    n_components: Optional[int] = None,
    max_iter: int = 200,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Independent Component Analysis for artifact removal.
    
    Uses FastICA to decompose signals into independent components.
    Artifacts (eye blinks, muscle) typically appear as distinct components.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        n_components: Number of ICA components (defaults to n_channels)
        max_iter: Maximum iterations for ICA
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (sources, mixing_matrix) where:
            - sources: Independent components (n_components, n_samples)
            - mixing_matrix: Mixing matrix to reconstruct original data
    """
    from sklearn.decomposition import FastICA
    
    n_channels = data.shape[0]
    if n_components is None:
        n_components = n_channels
    
    # Transpose for sklearn (samples, features)
    data_T = data.T
    
    ica = FastICA(
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
        whiten='unit-variance'
    )
    
    sources = ica.fit_transform(data_T)
    mixing_matrix = ica.mixing_
    
    return sources.T, mixing_matrix


def remove_artifacts_ica(
    data: np.ndarray,
    artifact_components: list,
    n_components: Optional[int] = None,
    random_state: int = 42
) -> np.ndarray:
    """
    Remove artifacts by zeroing out specific ICA components.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        artifact_components: List of component indices to remove
        n_components: Number of ICA components
        random_state: Random seed
        
    Returns:
        Cleaned EEG data with artifacts removed
    """
    from sklearn.decomposition import FastICA
    
    n_channels = data.shape[0]
    if n_components is None:
        n_components = n_channels
        
    data_T = data.T
    
    ica = FastICA(
        n_components=n_components,
        random_state=random_state,
        whiten='unit-variance'
    )
    
    sources = ica.fit_transform(data_T)
    
    # Zero out artifact components
    for idx in artifact_components:
        if 0 <= idx < sources.shape[1]:
            sources[:, idx] = 0
    
    # Reconstruct
    cleaned = ica.inverse_transform(sources)
    
    return cleaned.T


def zscore_normalize(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Z-score normalize EEG data.
    
    Args:
        data: EEG data
        axis: Axis along which to normalize
        
    Returns:
        Normalized data with zero mean and unit variance
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    return (data - mean) / std


def resample(
    data: np.ndarray,
    original_rate: int,
    target_rate: int
) -> np.ndarray:
    """
    Resample EEG data to a different sampling rate.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        original_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        
    Returns:
        Resampled data
    """
    if original_rate == target_rate:
        return data
    
    n_samples = data.shape[-1]
    duration = n_samples / original_rate
    new_n_samples = int(duration * target_rate)
    
    if data.ndim == 1:
        return signal.resample(data, new_n_samples)
    else:
        return np.array([signal.resample(channel, new_n_samples) for channel in data])
