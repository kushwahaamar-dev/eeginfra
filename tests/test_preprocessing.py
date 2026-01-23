"""
Unit tests for preprocessing utilities.
"""

import pytest
import numpy as np
import torch

from neuroformer.preprocessing.filters import (
    bandpass_filter,
    notch_filter,
    highpass_filter,
    zscore_normalize,
    resample,
)
from neuroformer.preprocessing.augmentation import (
    time_shift,
    add_gaussian_noise,
    channel_dropout,
    mixup,
    scaling,
)
from neuroformer.preprocessing.features import (
    compute_psd,
    compute_band_powers,
    compute_coherence,
    compute_asymmetry,
    extract_all_features,
)


class TestFilters:
    """Tests for signal filtering functions."""
    
    @pytest.fixture
    def sample_signal(self):
        """Generate a sample EEG-like signal."""
        np.random.seed(42)
        # 19 channels, 1 second at 256 Hz
        return np.random.randn(19, 256).astype(np.float32)
    
    def test_bandpass_filter_1d(self):
        """Test bandpass on single channel."""
        signal = np.random.randn(256)
        filtered = bandpass_filter(signal, 1, 40, sampling_rate=256)
        assert filtered.shape == signal.shape
        assert not np.isnan(filtered).any()
    
    def test_bandpass_filter_2d(self, sample_signal):
        """Test bandpass on multi-channel signal."""
        filtered = bandpass_filter(sample_signal, 1, 40, sampling_rate=256)
        assert filtered.shape == sample_signal.shape
        assert not np.isnan(filtered).any()
    
    def test_notch_filter(self, sample_signal):
        """Test 60 Hz notch filter."""
        filtered = notch_filter(sample_signal, notch_freq=60, sampling_rate=256)
        assert filtered.shape == sample_signal.shape
    
    def test_highpass_filter(self, sample_signal):
        """Test highpass filter."""
        filtered = highpass_filter(sample_signal, cutoff=0.5, sampling_rate=256)
        assert filtered.shape == sample_signal.shape
    
    def test_zscore_normalize(self, sample_signal):
        """Test z-score normalization."""
        normalized = zscore_normalize(sample_signal, axis=-1)
        # Each channel should have ~0 mean and ~1 std
        assert np.allclose(normalized.mean(axis=-1), 0, atol=1e-5)
        assert np.allclose(normalized.std(axis=-1), 1, atol=1e-5)
    
    def test_resample(self, sample_signal):
        """Test resampling."""
        resampled = resample(sample_signal, original_rate=256, target_rate=128)
        expected_samples = 128
        assert resampled.shape[-1] == expected_samples


class TestAugmentation:
    """Tests for data augmentation functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample EEG features."""
        np.random.seed(42)
        return np.random.randn(19, 256).astype(np.float32)
    
    def test_time_shift(self, sample_data):
        """Test time shifting."""
        shifted = time_shift(sample_data, max_shift=10, random_state=42)
        assert shifted.shape == sample_data.shape
        # Should be different from original
        assert not np.allclose(shifted, sample_data)
    
    def test_add_gaussian_noise(self, sample_data):
        """Test noise addition."""
        noisy = add_gaussian_noise(sample_data, noise_std=0.1, random_state=42)
        assert noisy.shape == sample_data.shape
        assert not np.allclose(noisy, sample_data)
    
    def test_channel_dropout(self, sample_data):
        """Test channel dropout."""
        dropped = channel_dropout(sample_data, dropout_prob=0.5, random_state=42)
        assert dropped.shape == sample_data.shape
        # Some channels should be zeroed
        zero_channels = (dropped == 0).all(axis=-1).sum()
        assert zero_channels > 0
    
    def test_scaling(self, sample_data):
        """Test amplitude scaling."""
        scaled = scaling(sample_data, scale_range=(0.8, 1.2), random_state=42)
        assert scaled.shape == sample_data.shape
    
    def test_mixup(self):
        """Test mixup augmentation."""
        data1 = np.random.randn(19, 256).astype(np.float32)
        data2 = np.random.randn(19, 256).astype(np.float32)
        label1 = np.array([1, 0, 0, 0, 0, 0, 0])
        label2 = np.array([0, 0, 1, 0, 0, 0, 0])
        
        mixed_data, mixed_label = mixup(data1, data2, label1, label2, alpha=0.5, random_state=42)
        
        assert mixed_data.shape == data1.shape
        assert mixed_label.shape == label1.shape
        # Mixed label should sum to 1
        assert np.isclose(mixed_label.sum(), 1.0)


class TestFeatures:
    """Tests for feature extraction functions."""
    
    @pytest.fixture
    def sample_signal(self):
        """Generate sample multichannel signal."""
        np.random.seed(42)
        return np.random.randn(19, 512).astype(np.float32)
    
    def test_compute_psd(self, sample_signal):
        """Test PSD computation."""
        freqs, psd = compute_psd(sample_signal, sampling_rate=256)
        assert len(freqs) > 0
        assert psd.shape[0] == 19  # 19 channels
        assert not np.isnan(psd).any()
    
    def test_compute_band_powers(self, sample_signal):
        """Test band power computation."""
        band_powers = compute_band_powers(sample_signal, sampling_rate=256)
        
        assert 'delta' in band_powers
        assert 'alpha' in band_powers
        assert 'beta' in band_powers
        
        for band, powers in band_powers.items():
            assert powers.shape == (19,)  # One value per electrode
            assert (powers >= 0).all()  # Power should be non-negative
    
    def test_compute_coherence(self, sample_signal):
        """Test coherence computation."""
        freqs, coherence = compute_coherence(sample_signal, sampling_rate=256)
        
        assert len(freqs) > 0
        assert coherence.shape[:2] == (19, 19)
        # Coherence values should be in [0, 1]
        assert (coherence >= 0).all()
        assert (coherence <= 1).all()
        # Diagonal should be 1 (self-coherence)
        for i in range(19):
            assert np.allclose(coherence[i, i, :], 1.0)
    
    def test_compute_asymmetry(self, sample_signal):
        """Test asymmetry computation."""
        asymmetry = compute_asymmetry(sample_signal, sampling_rate=256)
        
        assert 'alpha' in asymmetry
        # Should have values for each electrode pair
        assert len(asymmetry['alpha']) > 0
    
    def test_extract_all_features(self, sample_signal):
        """Test comprehensive feature extraction."""
        features = extract_all_features(sample_signal, sampling_rate=256)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.isnan(features).any()


class TestValidation:
    """Tests for validation utilities."""
    
    def test_validate_eeg_data_valid(self):
        """Test validation with valid data."""
        from neuroformer.utils.validation import validate_eeg_data
        
        data = np.random.randn(5, 19).astype(np.float32)
        validated = validate_eeg_data(data)
        
        assert isinstance(validated, torch.Tensor)
        assert validated.shape == (1, 5, 19)  # Batch dim added
    
    def test_validate_eeg_data_invalid_shape(self):
        """Test validation with invalid shape."""
        from neuroformer.utils.validation import validate_eeg_data
        from neuroformer.utils.exceptions import DataValidationError
        
        data = np.random.randn(3, 10).astype(np.float32)  # Wrong dimensions
        
        with pytest.raises(DataValidationError):
            validate_eeg_data(data, expected_bands=5, expected_electrodes=19)
    
    def test_validate_eeg_data_with_nan(self):
        """Test validation catches NaN values."""
        from neuroformer.utils.validation import validate_eeg_data
        from neuroformer.utils.exceptions import DataValidationError
        
        data = np.random.randn(5, 19).astype(np.float32)
        data[0, 0] = np.nan
        
        with pytest.raises(DataValidationError):
            validate_eeg_data(data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
