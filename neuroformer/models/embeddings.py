"""
Embedding layers for NeuroFormer.

Provides positional, electrode, and frequency band embeddings.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    
    Uses fixed sinusoidal patterns to encode position information,
    allowing the model to understand temporal order.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    
    Can adapt to data-specific patterns during training.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ElectrodeEmbedding(nn.Module):
    """
    Electrode-specific learned embeddings.
    
    Encodes spatial information about each electrode's position
    on the scalp using the standard 10-20 system.
    """
    
    # 10-20 system normalized coordinates (x, y, z)
    ELECTRODE_POSITIONS = {
        "Fp1": (-0.31, 0.95, 0.0), "Fp2": (0.31, 0.95, 0.0),
        "F7": (-0.81, 0.59, 0.0), "F3": (-0.55, 0.67, 0.47),
        "Fz": (0.0, 0.71, 0.71), "F4": (0.55, 0.67, 0.47),
        "F8": (0.81, 0.59, 0.0),
        "T3": (-1.0, 0.0, 0.0), "C3": (-0.71, 0.0, 0.71),
        "Cz": (0.0, 0.0, 1.0), "C4": (0.71, 0.0, 0.71),
        "T4": (1.0, 0.0, 0.0),
        "T5": (-0.81, -0.59, 0.0), "P3": (-0.55, -0.67, 0.47),
        "Pz": (0.0, -0.71, 0.71), "P4": (0.55, -0.67, 0.47),
        "T6": (0.81, -0.59, 0.0),
        "O1": (-0.31, -0.95, 0.0), "O2": (0.31, -0.95, 0.0),
    }
    
    def __init__(
        self,
        num_electrodes: int,
        d_model: int,
        use_positions: bool = True
    ):
        """
        Args:
            num_electrodes: Number of EEG electrodes
            d_model: Embedding dimension
            use_positions: Whether to use 3D position encoding
        """
        super().__init__()
        self.num_electrodes = num_electrodes
        self.d_model = d_model
        self.use_positions = use_positions
        
        # Learned electrode embeddings
        self.electrode_embeddings = nn.Embedding(num_electrodes, d_model)
        
        if use_positions:
            # Position projection
            self.position_proj = nn.Linear(3, d_model)
            
            # Register default positions
            positions = self._get_default_positions()
            self.register_buffer('positions', positions)
    
    def _get_default_positions(self) -> torch.Tensor:
        """Get default electrode positions from 10-20 system."""
        electrode_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "C3", "Cz", "C4", "T4",
            "T5", "P3", "Pz", "P4", "T6",
            "O1", "O2"
        ]
        
        positions = []
        for i in range(self.num_electrodes):
            if i < len(electrode_names) and electrode_names[i] in self.ELECTRODE_POSITIONS:
                positions.append(self.ELECTRODE_POSITIONS[electrode_names[i]])
            else:
                # Default position for unknown electrodes
                positions.append((0.0, 0.0, 0.0))
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def forward(self, electrode_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get electrode embeddings.
        
        Args:
            electrode_indices: Optional indices of electrodes to embed.
                             If None, returns all electrode embeddings.
                             
        Returns:
            Electrode embeddings of shape (num_electrodes, d_model) or
            (batch, num_selected, d_model)
        """
        if electrode_indices is None:
            electrode_indices = torch.arange(self.num_electrodes, device=self.electrode_embeddings.weight.device)
        
        embeddings = self.electrode_embeddings(electrode_indices)
        
        if self.use_positions:
            if electrode_indices.dim() == 1:
                pos_features = self.position_proj(self.positions[electrode_indices])
            else:
                # Handle batch dimension
                pos_features = self.position_proj(self.positions.unsqueeze(0).expand(electrode_indices.size(0), -1, -1))
            embeddings = embeddings + pos_features
        
        return embeddings


class BandEmbedding(nn.Module):
    """
    Frequency band embeddings.
    
    Provides learnable representations for each EEG frequency band.
    """
    
    BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
    
    def __init__(self, d_model: int, num_bands: int = 5):
        """
        Args:
            d_model: Embedding dimension
            num_bands: Number of frequency bands
        """
        super().__init__()
        self.num_bands = num_bands
        self.band_embeddings = nn.Embedding(num_bands, d_model)
        
        # Frequency range encoding
        self.freq_ranges = torch.tensor([
            [0.5, 4],    # delta
            [4, 8],      # theta
            [8, 13],     # alpha
            [13, 30],    # beta
            [30, 100],   # gamma
        ], dtype=torch.float32)
        
        self.freq_proj = nn.Linear(2, d_model)
        self.register_buffer('freq_ranges_buf', self.freq_ranges)
    
    def forward(self, band_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get frequency band embeddings.
        
        Args:
            band_indices: Optional indices of bands. If None, returns all.
            
        Returns:
            Band embeddings of shape (num_bands, d_model) or (batch, num_bands, d_model)
        """
        if band_indices is None:
            band_indices = torch.arange(self.num_bands, device=self.band_embeddings.weight.device)
        
        embeddings = self.band_embeddings(band_indices)
        freq_features = self.freq_proj(self.freq_ranges_buf[band_indices])
        
        return embeddings + freq_features


class PatchEmbedding(nn.Module):
    """
    Convert EEG signals into patch embeddings.
    
    Similar to Vision Transformer, divides the temporal signal into patches.
    """
    
    def __init__(
        self,
        num_electrodes: int,
        patch_size: int,
        d_model: int
    ):
        """
        Args:
            num_electrodes: Number of EEG channels
            patch_size: Size of each temporal patch
            d_model: Output embedding dimension
        """
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(num_electrodes * patch_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create patch embeddings from EEG data.
        
        Args:
            x: EEG data of shape (batch, n_channels, n_samples)
            
        Returns:
            Patch embeddings of shape (batch, n_patches, d_model)
        """
        batch_size, n_channels, n_samples = x.shape
        n_patches = n_samples // self.patch_size
        
        # Reshape into patches: (batch, n_patches, n_channels * patch_size)
        x = x[:, :, :n_patches * self.patch_size]
        x = x.reshape(batch_size, n_channels, n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, n_patches, -1)
        
        return self.proj(x)
