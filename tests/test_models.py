"""
Unit tests for NeuroFormer models.
"""

import pytest
import torch
import numpy as np

from neuroformer.models import NeuroFormer
from neuroformer.models.embeddings import (
    PositionalEncoding,
    ElectrodeEmbedding,
    BandEmbedding,
)
from neuroformer.models.attention import (
    MultiHeadSelfAttention,
    CrossAttention,
)
from neuroformer.models.gnn_layers import GraphAttentionLayer, SpatialGNN
from neuroformer.models.transformer import TransformerEncoderLayer, TemporalTransformer


class TestPositionalEncoding:
    """Tests for positional encoding."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        pe = PositionalEncoding(d_model=128, max_len=1000)
        x = torch.randn(4, 50, 128)  # (batch, seq, d_model)
        out = pe(x)
        assert out.shape == x.shape
    
    def test_different_sequence_lengths(self):
        """Test with various sequence lengths."""
        pe = PositionalEncoding(d_model=64)
        for seq_len in [10, 100, 500]:
            x = torch.randn(2, seq_len, 64)
            out = pe(x)
            assert out.shape == x.shape


class TestElectrodeEmbedding:
    """Tests for electrode embedding."""
    
    def test_output_shape(self):
        """Test output dimensions."""
        embed = ElectrodeEmbedding(num_electrodes=19, d_model=128)
        out = embed()
        assert out.shape == (19, 128)
    
    def test_positions_included(self):
        """Test that 3D positions are incorporated."""
        embed = ElectrodeEmbedding(num_electrodes=19, d_model=128, use_positions=True)
        out = embed()
        assert out.shape == (19, 128)


class TestMultiHeadSelfAttention:
    """Tests for multi-head self-attention."""
    
    def test_output_shape(self):
        """Test output matches input shape."""
        attn = MultiHeadSelfAttention(d_model=128, n_heads=8)
        x = torch.randn(4, 50, 128)
        out, _ = attn(x)
        assert out.shape == x.shape
    
    def test_attention_weights(self):
        """Test attention weight retrieval."""
        attn = MultiHeadSelfAttention(d_model=128, n_heads=8)
        x = torch.randn(4, 50, 128)
        out, weights = attn(x, return_attention=True)
        assert weights is not None
        assert weights.shape == (4, 8, 50, 50)  # (batch, heads, seq, seq)


class TestGraphAttentionLayer:
    """Tests for graph attention layer."""
    
    def test_output_shape(self):
        """Test output dimensions."""
        gat = GraphAttentionLayer(in_features=64, out_features=32, n_heads=4)
        x = torch.randn(2, 19, 64)  # (batch, nodes, features)
        adj = torch.ones(2, 19, 19)  # fully connected
        out, _ = gat(x, adj)
        assert out.shape == (2, 19, 4 * 32)  # concat heads
    
    def test_with_sparse_adjacency(self):
        """Test with sparse connections."""
        gat = GraphAttentionLayer(in_features=64, out_features=32, n_heads=4)
        x = torch.randn(2, 19, 64)
        # Only connect neighboring electrodes
        adj = torch.eye(19).unsqueeze(0).expand(2, -1, -1)
        out, _ = gat(x, adj)
        assert out.shape == (2, 19, 4 * 32)


class TestSpatialGNN:
    """Tests for spatial GNN module."""
    
    def test_output_shape(self):
        """Test full GNN output."""
        gnn = SpatialGNN(
            in_features=64,
            hidden_dim=128,
            out_features=256,
            n_layers=3
        )
        x = torch.randn(2, 19, 64)
        adj = torch.ones(2, 19, 19)
        out = gnn(x, adj)
        assert out.shape == (2, 19, 256)


class TestTransformerEncoderLayer:
    """Tests for transformer encoder layer."""
    
    def test_output_shape(self):
        """Test single layer output."""
        layer = TransformerEncoderLayer(d_model=128, n_heads=8, d_ff=512)
        x = torch.randn(4, 50, 128)
        out, _ = layer(x)
        assert out.shape == x.shape


class TestTemporalTransformer:
    """Tests for temporal transformer."""
    
    def test_output_shape(self):
        """Test full transformer output."""
        transformer = TemporalTransformer(
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512
        )
        x = torch.randn(4, 50, 128)
        out, _ = transformer(x)
        assert out.shape == x.shape


class TestNeuroFormer:
    """Tests for the main NeuroFormer model."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return NeuroFormer(
            num_electrodes=19,
            num_classes=7,
            d_model=64,
            n_heads=4,
            n_transformer_layers=2,
            n_gnn_layers=2
        )
    
    def test_forward_pass(self, model):
        """Test basic forward pass."""
        x = torch.randn(4, 5, 19)  # (batch, bands, electrodes)
        output = model(x)
        
        assert 'logits' in output
        assert output['logits'].shape == (4, 7)
    
    def test_with_coherence(self, model):
        """Test forward pass with coherence matrices."""
        x = torch.randn(4, 5, 19)
        coherence = torch.randn(4, 5, 19, 19)
        output = model(x, coherence_matrices=coherence)
        
        assert output['logits'].shape == (4, 7)
    
    def test_return_features(self, model):
        """Test feature extraction mode."""
        x = torch.randn(4, 5, 19)
        output = model(x, return_features=True)
        
        assert 'features' in output
        assert 'spatial_features' in output
        assert 'temporal_features' in output
    
    def test_predict(self, model):
        """Test prediction method."""
        model.eval()
        x = torch.randn(4, 5, 19)
        preds = model.predict(x)
        
        assert preds.shape == (4,)
        assert preds.min() >= 0
        assert preds.max() < 7
    
    def test_predict_proba(self, model):
        """Test probability prediction."""
        model.eval()
        x = torch.randn(4, 5, 19)
        probs = model.predict_proba(x)
        
        assert probs.shape == (4, 7)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
    
    def test_gradient_flow(self, model):
        """Test that gradients flow correctly."""
        x = torch.randn(4, 5, 19, requires_grad=True)
        output = model(x)
        loss = output['logits'].sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestModelRobustness:
    """Tests for model robustness to edge cases."""
    
    @pytest.fixture
    def model(self):
        return NeuroFormer(
            num_electrodes=19,
            num_classes=7,
            d_model=64,
            n_heads=4,
            n_transformer_layers=2
        )
    
    def test_single_sample(self, model):
        """Test with batch size of 1."""
        x = torch.randn(1, 5, 19)
        output = model(x)
        assert output['logits'].shape == (1, 7)
    
    def test_large_batch(self, model):
        """Test with large batch size."""
        x = torch.randn(64, 5, 19)
        output = model(x)
        assert output['logits'].shape == (64, 7)
    
    def test_zero_input(self, model):
        """Test with zero input (should not crash)."""
        x = torch.zeros(4, 5, 19)
        output = model(x)
        assert not torch.isnan(output['logits']).any()
    
    def test_normalized_input(self, model):
        """Test with normalized input."""
        x = torch.randn(4, 5, 19)
        x = (x - x.mean()) / x.std()
        output = model(x)
        assert not torch.isnan(output['logits']).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
