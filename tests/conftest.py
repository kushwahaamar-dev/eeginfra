"""
Pytest configuration and fixtures.
"""

import pytest
import torch
import numpy as np


@pytest.fixture(scope="session")
def device():
    """Get available device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def sample_batch():
    """Generate sample batch of EEG data."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    batch_size = 4
    num_bands = 5
    num_electrodes = 19
    
    return {
        'features': torch.randn(batch_size, num_bands, num_electrodes),
        'labels': torch.randint(0, 7, (batch_size,)),
        'coherence': torch.randn(batch_size, num_bands, num_electrodes, num_electrodes)
    }


@pytest.fixture
def sample_signal():
    """Generate sample EEG signals."""
    np.random.seed(42)
    # 19 channels, 2 seconds at 256 Hz
    return np.random.randn(19, 512).astype(np.float32)


# Markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
