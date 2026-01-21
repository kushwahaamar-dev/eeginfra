"""
Feature extraction utilities for EEG data.

Implements spectral features (PSD, coherence), band powers, and asymmetry metrics.
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional, List


# Standard frequency bands for EEG analysis
FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100)
}


def compute_psd(
    data: np.ndarray,
    sampling_rate: int = 256,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Args:
        data: EEG data of shape (n_channels, n_samples) or (n_samples,)
        sampling_rate: Sampling rate in Hz
        nperseg: Length of each segment for Welch's method
        noverlap: Number of overlapping samples
        
    Returns:
        Tuple of (frequencies, psd) where:
            - frequencies: Array of frequency bins
            - psd: Power spectral density values
    """
    if nperseg is None:
        nperseg = min(256, data.shape[-1] // 4)
    if noverlap is None:
        noverlap = nperseg // 2
    
    if data.ndim == 1:
        freqs, psd = signal.welch(
            data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap
        )
    else:
        psds = []
        for channel in data:
            freqs, psd = signal.welch(
                channel, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap
            )
            psds.append(psd)
        psd = np.array(psds)
    
    return freqs, psd


def compute_band_powers(
    data: np.ndarray,
    sampling_rate: int = 256,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    relative: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute power in standard frequency bands.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        bands: Dictionary of band_name -> (low_freq, high_freq)
        relative: If True, compute relative (normalized) band powers
        
    Returns:
        Dictionary mapping band names to power values per channel
    """
    if bands is None:
        bands = FREQ_BANDS
    
    freqs, psd = compute_psd(data, sampling_rate)
    freq_resolution = freqs[1] - freqs[0]
    
    band_powers = {}
    total_power = np.trapz(psd, dx=freq_resolution, axis=-1) if relative else None
    
    for band_name, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) == 0:
            band_powers[band_name] = np.zeros(psd.shape[:-1])
        else:
            power = np.trapz(psd[..., idx], dx=freq_resolution, axis=-1)
            if relative and total_power is not None:
                power = power / (total_power + 1e-10)
            band_powers[band_name] = power
    
    return band_powers


def compute_coherence(
    data: np.ndarray,
    sampling_rate: int = 256,
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coherence between all electrode pairs.
    
    Coherence measures the linear relationship between two signals
    at each frequency. Values range from 0 to 1.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        nperseg: Segment length for spectral estimation
        
    Returns:
        Tuple of (frequencies, coherence_matrix) where:
            - frequencies: Array of frequency bins
            - coherence_matrix: Shape (n_channels, n_channels, n_freqs)
    """
    n_channels = data.shape[0]
    
    if nperseg is None:
        nperseg = min(256, data.shape[-1] // 4)
    
    # Get frequency bins from first pair
    freqs, _ = signal.coherence(
        data[0], data[1], fs=sampling_rate, nperseg=nperseg
    )
    n_freqs = len(freqs)
    
    coherence_matrix = np.zeros((n_channels, n_channels, n_freqs))
    
    for i in range(n_channels):
        coherence_matrix[i, i, :] = 1.0  # Self-coherence is 1
        for j in range(i + 1, n_channels):
            _, coh = signal.coherence(
                data[i], data[j], fs=sampling_rate, nperseg=nperseg
            )
            coherence_matrix[i, j, :] = coh
            coherence_matrix[j, i, :] = coh
    
    return freqs, coherence_matrix


def compute_band_coherence(
    data: np.ndarray,
    sampling_rate: int = 256,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute average coherence in each frequency band.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        bands: Dictionary of band_name -> (low_freq, high_freq)
        
    Returns:
        Dictionary mapping band names to coherence matrices (n_channels, n_channels)
    """
    if bands is None:
        bands = FREQ_BANDS
    
    freqs, coherence_matrix = compute_coherence(data, sampling_rate)
    
    band_coherence = {}
    for band_name, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) == 0:
            band_coherence[band_name] = np.zeros(coherence_matrix.shape[:2])
        else:
            band_coherence[band_name] = np.mean(coherence_matrix[:, :, idx], axis=-1)
    
    return band_coherence


def compute_asymmetry(
    data: np.ndarray,
    sampling_rate: int = 256,
    electrode_pairs: Optional[List[Tuple[int, int]]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute hemispheric asymmetry metrics.
    
    Frontal alpha asymmetry is a biomarker associated with depression
    and other mood disorders.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        electrode_pairs: List of (left_idx, right_idx) pairs for asymmetry
                        Defaults to standard frontal pairs
        
    Returns:
        Dictionary with asymmetry metrics per band
    """
    # Default frontal pairs for 19-channel 10-20 system
    # Fp1(0)-Fp2(1), F3(3)-F4(5), F7(2)-F8(6)
    if electrode_pairs is None:
        electrode_pairs = [(0, 1), (3, 5), (2, 6)]
    
    band_powers = compute_band_powers(data, sampling_rate)
    
    asymmetry = {}
    for band_name, powers in band_powers.items():
        asym_values = []
        for left_idx, right_idx in electrode_pairs:
            if left_idx < len(powers) and right_idx < len(powers):
                # Log asymmetry: ln(Right) - ln(Left)
                left_power = max(powers[left_idx], 1e-10)
                right_power = max(powers[right_idx], 1e-10)
                asym = np.log(right_power) - np.log(left_power)
                asym_values.append(asym)
        
        asymmetry[band_name] = np.array(asym_values) if asym_values else np.array([0.0])
    
    return asymmetry


def compute_differential_entropy(
    data: np.ndarray,
    sampling_rate: int = 256,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute Differential Entropy (DE) features for each band.
    
    DE is commonly used in EEG emotion recognition and is defined as:
    DE = 0.5 * log(2 * pi * e * variance)
    
    For band-limited signals, this approximates to:
    DE ≈ 0.5 * log(2 * pi * e * band_power)
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        bands: Dictionary of band_name -> (low_freq, high_freq)
        
    Returns:
        Dictionary mapping band names to DE values per channel
    """
    if bands is None:
        bands = FREQ_BANDS
    
    from neuroformer.preprocessing.filters import bandpass_filter
    
    de_features = {}
    
    for band_name, (low, high) in bands.items():
        # Filter to get band-limited signal
        filtered = bandpass_filter(data, low, high, sampling_rate)
        
        # Compute variance of filtered signal
        variance = np.var(filtered, axis=-1)
        variance = np.maximum(variance, 1e-10)  # Avoid log(0)
        
        # Differential entropy
        de = 0.5 * np.log(2 * np.pi * np.e * variance)
        de_features[band_name] = de
    
    return de_features


def extract_all_features(
    data: np.ndarray,
    sampling_rate: int = 256,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Extract comprehensive feature vector from EEG data.
    
    Combines band powers, DE features, and asymmetry into single vector.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        bands: Frequency bands to extract
        
    Returns:
        Feature vector of shape (n_features,)
    """
    if bands is None:
        bands = FREQ_BANDS
    
    features = []
    
    # Band powers (relative)
    band_powers = compute_band_powers(data, sampling_rate, bands, relative=True)
    for band_name in sorted(bands.keys()):
        features.extend(band_powers[band_name].flatten())
    
    # Differential entropy
    de_features = compute_differential_entropy(data, sampling_rate, bands)
    for band_name in sorted(bands.keys()):
        features.extend(de_features[band_name].flatten())
    
    # Alpha asymmetry (frontal)
    asymmetry = compute_asymmetry(data, sampling_rate)
    features.extend(asymmetry.get("alpha", [0.0]))
    
    return np.array(features)
