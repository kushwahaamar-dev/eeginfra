"""
Attention mechanisms for NeuroFormer.

Implements multi-head self-attention and cross-attention modules.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Core component of the Transformer that allows the model to
    attend to different positions in the sequence.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights) where attention_weights
            is None if return_attention is False
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        # (batch, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # (batch, n_heads, seq_len, d_k)
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        # (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        if return_attention:
            return out, attn_weights
        return out, None


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for combining different representations.
    
    Used to fuse spatial (GNN) and temporal (Transformer) features.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        # Separate projections for query (from one modality)
        # and key/value (from another modality)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            query: Query tensor from first modality (batch, seq_q, d_model)
            key_value: Key/Value tensor from second modality (batch, seq_kv, d_model)
            mask: Optional attention mask
            
        Returns:
            Cross-attended output (batch, seq_q, d_model)
        """
        batch_size, seq_q, _ = query.shape
        seq_kv = key_value.size(1)
        
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_kv, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_kv, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)
        
        return self.out_proj(out)


class EfficientAttention(nn.Module):
    """
    Linear complexity attention using kernel approximation.
    
    More efficient for long sequences typical in EEG data.
    Uses random feature approximation of softmax kernel.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_features: int = 64,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_features: Number of random features for kernel approximation
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_features = n_features
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Random projection for kernel approximation
        self.register_buffer(
            'random_weights',
            torch.randn(n_heads, self.d_k, n_features) / math.sqrt(self.d_k)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random feature map."""
        # x: (batch, n_heads, seq_len, d_k)
        proj = torch.matmul(x, self.random_weights)  # (batch, n_heads, seq_len, n_features)
        return F.relu(proj) + 1e-6
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with linear attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply feature map
        q_prime = self._phi(q)
        k_prime = self._phi(k)
        
        # Linear attention: O(n) instead of O(n^2)
        # (K^T V) first, then Q @ (K^T V)
        kv = torch.matmul(k_prime.transpose(-2, -1), v)  # (batch, n_heads, n_features, d_k)
        qkv = torch.matmul(q_prime, kv)  # (batch, n_heads, seq_len, d_k)
        
        # Normalize
        normalizer = torch.matmul(q_prime, k_prime.sum(dim=-2, keepdim=True).transpose(-2, -1))
        out = qkv / (normalizer + 1e-6)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.dropout(self.out_proj(out))
