"""
NeuroFormer: Hybrid Transformer-GNN model for EEG classification.

The main model that combines spatial GNN and temporal Transformer
for state-of-the-art psychiatric disorder classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from neuroformer.models.embeddings import (
    ElectrodeEmbedding,
    BandEmbedding,
    PatchEmbedding,
)
from neuroformer.models.attention import CrossAttention
from neuroformer.models.gnn_layers import SpatialGNN, build_adjacency_from_coherence
from neuroformer.models.transformer import TemporalTransformer


class SpatioTemporalFusion(nn.Module):
    """
    Fusion layer combining spatial GNN and temporal Transformer features.
    
    Uses cross-attention to let spatial and temporal representations
    inform each other.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Spatial attends to temporal
        self.spatial_to_temporal = CrossAttention(d_model, n_heads, dropout)
        
        # Temporal attends to spatial
        self.temporal_to_spatial = CrossAttention(d_model, n_heads, dropout)
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse spatial and temporal features.
        
        Args:
            spatial_features: From GNN (batch, n_electrodes, d_model)
            temporal_features: From Transformer (batch, seq_len, d_model)
            
        Returns:
            Fused features (batch, d_model)
        """
        # Cross-attention both ways
        spatial_enhanced = self.spatial_to_temporal(spatial_features, temporal_features)
        temporal_enhanced = self.temporal_to_spatial(temporal_features, spatial_features)
        
        # Global pooling
        spatial_pooled = spatial_enhanced.mean(dim=1)  # (batch, d_model)
        temporal_pooled = temporal_enhanced.mean(dim=1)  # (batch, d_model)
        
        # Gated fusion
        concat = torch.cat([spatial_pooled, temporal_pooled], dim=-1)
        gate = self.gate(concat)
        
        fused = gate * spatial_pooled + (1 - gate) * temporal_pooled
        
        return self.norm(fused)


class NeuroFormer(nn.Module):
    """
    NeuroFormer: State-of-the-art EEG classification model.
    
    Architecture:
    1. Input projection + electrode embeddings
    2. Spatial GNN processing electrode graphs (per band)
    3. Temporal Transformer processing time series
    4. Cross-attention fusion of spatial and temporal
    5. Classification head
    
    Supports multi-band processing for comprehensive frequency analysis.
    """
    
    def __init__(
        self,
        num_electrodes: int = 19,
        num_classes: int = 7,
        num_bands: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_transformer_layers: int = 4,
        n_gnn_layers: int = 3,
        d_ff: int = 1024,
        gnn_hidden: int = 128,
        dropout: float = 0.1,
        use_band_specific_gnn: bool = True,
        use_efficient_attn: bool = False
    ):
        """
        Args:
            num_electrodes: Number of EEG electrodes
            num_classes: Number of classification classes
            num_bands: Number of frequency bands
            d_model: Model dimension
            n_heads: Number of attention heads
            n_transformer_layers: Number of Transformer layers
            n_gnn_layers: Number of GNN layers
            d_ff: Feed-forward dimension
            gnn_hidden: GNN hidden dimension
            dropout: Dropout probability
            use_band_specific_gnn: Use separate GNN for each band
            use_efficient_attn: Use linear attention for long sequences
        """
        super().__init__()
        
        self.num_electrodes = num_electrodes
        self.num_classes = num_classes
        self.num_bands = num_bands
        self.d_model = d_model
        self.use_band_specific_gnn = use_band_specific_gnn
        
        # Embeddings
        self.electrode_embedding = ElectrodeEmbedding(num_electrodes, d_model)
        self.band_embedding = BandEmbedding(d_model, num_bands)
        
        # Input projection (from raw features to d_model)
        self.input_proj = nn.Linear(1, d_model)  # Single value per electrode/band
        
        # Spatial GNN pathway
        if use_band_specific_gnn:
            # Separate GNN for each frequency band
            self.spatial_gnns = nn.ModuleList([
                SpatialGNN(
                    in_features=d_model,
                    hidden_dim=gnn_hidden,
                    out_features=d_model,
                    n_layers=n_gnn_layers,
                    n_heads=4,
                    dropout=dropout
                )
                for _ in range(num_bands)
            ])
        else:
            # Shared GNN
            self.spatial_gnn = SpatialGNN(
                in_features=d_model,
                hidden_dim=gnn_hidden,
                out_features=d_model,
                n_layers=n_gnn_layers,
                n_heads=4,
                dropout=dropout
            )
        
        # Temporal Transformer pathway
        self.temporal_transformer = TemporalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_transformer_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_efficient_attn=use_efficient_attn
        )
        
        # Fusion layer
        self.fusion = SpatioTemporalFusion(d_model, n_heads, dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        band_powers: torch.Tensor,
        coherence_matrices: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through NeuroFormer.
        
        Args:
            band_powers: Band power features (batch, num_bands, num_electrodes)
            coherence_matrices: Coherence per band (batch, num_bands, num_electrodes, num_electrodes)
            adjacency: Pre-computed adjacency matrix (optional)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with 'logits' and optionally 'features', 'spatial_features', 'temporal_features'
        """
        batch_size = band_powers.size(0)
        device = band_powers.device
        
        # Build adjacency if not provided
        if adjacency is None:
            if coherence_matrices is not None:
                # Average coherence across bands
                avg_coherence = coherence_matrices.mean(dim=1)  # (batch, n_elec, n_elec)
                adjacency = build_adjacency_from_coherence(avg_coherence[0], threshold=0.3)
                adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # Fully connected as fallback
                adjacency = torch.ones(
                    batch_size, self.num_electrodes, self.num_electrodes, device=device
                )
        
        # Get electrode embeddings
        electrode_emb = self.electrode_embedding()  # (num_electrodes, d_model)
        electrode_emb = electrode_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Get band embeddings
        band_emb = self.band_embedding()  # (num_bands, d_model)
        
        # === Spatial Pathway (GNN) ===
        spatial_outputs = []
        
        for b in range(self.num_bands):
            # Input: band powers for this band -> (batch, num_electrodes, 1)
            band_input = band_powers[:, b, :].unsqueeze(-1)
            
            # Project to d_model
            band_features = self.input_proj(band_input)  # (batch, num_electrodes, d_model)
            
            # Add electrode and band embeddings
            band_features = band_features + electrode_emb + band_emb[b].unsqueeze(0).unsqueeze(0)
            
            # GNN processing
            if self.use_band_specific_gnn:
                spatial_out = self.spatial_gnns[b](band_features, adjacency)
            else:
                spatial_out = self.spatial_gnn(band_features, adjacency)
            
            spatial_outputs.append(spatial_out)
        
        # Combine band outputs
        spatial_features = torch.stack(spatial_outputs, dim=1)  # (batch, num_bands, num_electrodes, d_model)
        spatial_features = spatial_features.mean(dim=1)  # (batch, num_electrodes, d_model)
        
        # === Temporal Pathway (Transformer) ===
        # Reshape for temporal: treat (bands * electrodes) as sequence
        temporal_input = band_powers.view(batch_size, -1).unsqueeze(-1)  # (batch, seq_len, 1)
        temporal_input = self.input_proj(temporal_input)  # (batch, seq_len, d_model)
        
        temporal_features, attn_weights = self.temporal_transformer(temporal_input)
        
        # === Fusion ===
        fused_features = self.fusion(spatial_features, temporal_features)
        
        # === Classification ===
        logits = self.classifier(fused_features)
        
        output = {'logits': logits}
        
        if return_features:
            output['features'] = fused_features
            output['spatial_features'] = spatial_features
            output['temporal_features'] = temporal_features
            if attn_weights is not None:
                output['attention_weights'] = attn_weights
        
        return output
    
    def predict(self, band_powers: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get predicted class labels.
        
        Args:
            band_powers: Band power features
            **kwargs: Additional arguments for forward()
            
        Returns:
            Predicted class indices (batch,)
        """
        with torch.no_grad():
            output = self.forward(band_powers, **kwargs)
            return output['logits'].argmax(dim=-1)
    
    def predict_proba(self, band_powers: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            band_powers: Band power features
            **kwargs: Additional arguments for forward()
            
        Returns:
            Class probabilities (batch, num_classes)
        """
        with torch.no_grad():
            output = self.forward(band_powers, **kwargs)
            return F.softmax(output['logits'], dim=-1)


class NeuroFormerLite(nn.Module):
    """
    Lightweight version of NeuroFormer for faster inference.
    
    Reduces complexity while maintaining good accuracy.
    """
    
    def __init__(
        self,
        num_electrodes: int = 19,
        num_classes: int = 7,
        num_bands: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_electrodes = num_electrodes
        self.num_bands = num_bands
        
        # Simple input embedding
        self.input_embed = nn.Linear(num_electrodes * num_bands, d_model)
        
        # Single transformer layer
        self.transformer = TemporalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_model * 2,
            dropout=dropout,
            use_efficient_attn=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, band_powers: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = band_powers.size(0)
        
        # Flatten input
        x = band_powers.view(batch_size, -1)  # (batch, num_bands * num_electrodes)
        
        # Embed
        x = self.input_embed(x).unsqueeze(1)  # (batch, 1, d_model)
        
        # Transform
        x, _ = self.transformer(x)
        x = x.squeeze(1)  # (batch, d_model)
        
        # Classify
        logits = self.classifier(x)
        
        return {'logits': logits}
