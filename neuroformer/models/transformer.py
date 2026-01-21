"""
Transformer encoder for temporal EEG processing.

Implements the temporal modeling component of NeuroFormer.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from neuroformer.models.attention import MultiHeadSelfAttention, EfficientAttention
from neuroformer.models.embeddings import PositionalEncoding


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Two linear transformations with GELU activation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.
    
    Consists of:
    1. Multi-head self-attention (with pre-norm)
    2. Feed-forward network (with pre-norm)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_efficient_attn: bool = False
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            use_efficient_attn: Use linear complexity attention
        """
        super().__init__()
        
        if use_efficient_attn:
            self.self_attn = EfficientAttention(d_model, n_heads, dropout=dropout)
        else:
            self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        
        if isinstance(self.self_attn, EfficientAttention):
            x = self.self_attn(x)
            attn_weights = None
        else:
            x, attn_weights = self.self_attn(x, mask, return_attention)
        
        x = residual + self.dropout(x)
        
        # Pre-norm feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x, attn_weights


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer encoder for EEG sequences.
    
    Captures long-range temporal dependencies in the signal.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
        use_efficient_attn: bool = False
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            use_efficient_attn: Use linear attention for long sequences
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, use_efficient_attn
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through transformer.
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, list of attention weights per layer)
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        attentions = [] if return_attention else None
        
        # Pass through transformer layers
        for layer in self.layers:
            x, attn = layer(x, mask, return_attention)
            if return_attention and attn is not None:
                attentions.append(attn)
        
        # Final normalization
        x = self.norm(x)
        
        return x, attentions


class BandwiseTransformer(nn.Module):
    """
    Separate transformer processing for each frequency band.
    
    Allows the model to learn band-specific temporal patterns.
    """
    
    def __init__(
        self,
        num_bands: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_bands = num_bands
        
        # Separate transformer for each band
        self.band_transformers = nn.ModuleList([
            TemporalTransformer(
                d_model, n_heads, n_layers, d_ff, dropout
            )
            for _ in range(num_bands)
        ])
        
        # Cross-band attention
        self.cross_band_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        band_features: list  # List of (batch, seq_len, d_model) per band
    ) -> torch.Tensor:
        """
        Process each band and fuse with cross-band attention.
        
        Args:
            band_features: List of features for each frequency band
            
        Returns:
            Fused representation (batch, seq_len, d_model)
        """
        # Process each band
        band_outputs = []
        for i, (transformer, features) in enumerate(zip(self.band_transformers, band_features)):
            out, _ = transformer(features)
            band_outputs.append(out)
        
        # Stack bands: (batch, num_bands, seq_len, d_model)
        stacked = torch.stack(band_outputs, dim=1)
        batch_size, num_bands, seq_len, d_model = stacked.shape
        
        # Reshape for cross-band attention: (batch * seq_len, num_bands, d_model)
        stacked = stacked.permute(0, 2, 1, 3).reshape(batch_size * seq_len, num_bands, d_model)
        
        # Cross-band attention
        cross_out, _ = self.cross_band_attn(stacked, stacked, stacked)
        cross_out = self.cross_norm(stacked + cross_out)
        
        # Average over bands
        cross_out = cross_out.mean(dim=1)  # (batch * seq_len, d_model)
        cross_out = cross_out.view(batch_size, seq_len, d_model)
        
        return cross_out
