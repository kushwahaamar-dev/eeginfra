"""
Graph Neural Network layers for NeuroFormer.

Implements Graph Attention Networks (GAT) for modeling
spatial relationships between EEG electrodes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    
    Computes attention-weighted aggregation of neighbor features
    on the electrode graph.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            n_heads: Number of attention heads
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate heads (True) or average (False)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features))
        nn.init.xavier_uniform_(self.W, gain=1.414)
        
        # Attention parameters
        self.a_src = nn.Parameter(torch.empty(n_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_features, 1))
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        if concat:
            self.out_dim = n_heads * out_features
        else:
            self.out_dim = out_features
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Node features (batch, n_nodes, in_features)
            adj: Adjacency matrix (batch, n_nodes, n_nodes) or (n_nodes, n_nodes)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, n_nodes, _ = x.shape
        
        # Handle adjacency matrix dimensions
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Linear transformation: (batch, n_heads, n_nodes, out_features)
        h = torch.einsum('bni,hio->bhno', x, self.W)
        
        # Compute attention scores
        # Source scores: (batch, n_heads, n_nodes, 1)
        e_src = torch.matmul(h, self.a_src)
        # Destination scores: (batch, n_heads, n_nodes, 1)
        e_dst = torch.matmul(h, self.a_dst)
        
        # Attention coefficients: (batch, n_heads, n_nodes, n_nodes)
        e = e_src + e_dst.transpose(-2, -1)
        e = self.leaky_relu(e)
        
        # Mask based on adjacency (add large negative for non-edges)
        mask = adj.unsqueeze(1)  # (batch, 1, n_nodes, n_nodes)
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over neighbors
        attention = F.softmax(e, dim=-1)
        attention = torch.where(
            torch.isinf(e), 
            torch.zeros_like(attention), 
            attention
        )
        attention = self.dropout_layer(attention)
        
        # Aggregate neighbor features
        # (batch, n_heads, n_nodes, out_features)
        h_prime = torch.matmul(attention, h)
        
        # Combine heads
        if self.concat:
            # (batch, n_nodes, n_heads * out_features)
            out = h_prime.permute(0, 2, 1, 3).contiguous().view(batch_size, n_nodes, -1)
        else:
            # Average over heads
            out = h_prime.mean(dim=1)
        
        if return_attention:
            return out, attention.mean(dim=1)  # Average attention over heads
        return out, None


class SpatialGNN(nn.Module):
    """
    Multi-layer GNN for spatial electrode relationships.
    
    Uses coherence-based adjacency to model brain connectivity.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_features: Output feature dimension
            n_layers: Number of GAT layers
            n_heads: Number of attention heads per layer
            dropout: Dropout probability
            residual: Whether to use residual connections
        """
        super().__init__()
        self.n_layers = n_layers
        self.residual = residual
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.layers.append(
            GraphAttentionLayer(
                in_features, hidden_dim, n_heads, dropout, concat=True
            )
        )
        self.norms.append(nn.LayerNorm(n_heads * hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(
                GraphAttentionLayer(
                    n_heads * hidden_dim, hidden_dim, n_heads, dropout, concat=True
                )
            )
            self.norms.append(nn.LayerNorm(n_heads * hidden_dim))
        
        # Output layer
        if n_layers > 1:
            self.layers.append(
                GraphAttentionLayer(
                    n_heads * hidden_dim, out_features, n_heads, dropout, concat=False
                )
            )
            self.norms.append(nn.LayerNorm(out_features))
        
        # Input projection for residual
        self.input_proj = nn.Linear(in_features, out_features) if residual else None
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GNN layers.
        
        Args:
            x: Node features (batch, n_nodes, in_features)
            adj: Adjacency matrix (batch, n_nodes, n_nodes)
            
        Returns:
            Output features (batch, n_nodes, out_features)
        """
        residual_input = self.input_proj(x) if self.residual else None
        
        h = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h_new, _ = layer(h, adj)
            h_new = norm(h_new)
            
            if i < len(self.layers) - 1:  # Not last layer
                h_new = self.activation(h_new)
                h_new = self.dropout(h_new)
            
            h = h_new
        
        # Add residual connection
        if self.residual and residual_input is not None:
            h = h + residual_input
        
        return h


def build_adjacency_from_coherence(
    coherence_matrix: torch.Tensor,
    threshold: float = 0.3,
    top_k: Optional[int] = None,
    self_loops: bool = True
) -> torch.Tensor:
    """
    Build adjacency matrix from coherence values.
    
    Args:
        coherence_matrix: Coherence between electrodes (n_electrodes, n_electrodes)
        threshold: Minimum coherence to create edge
        top_k: Keep only top-k connections per node (optional)
        self_loops: Whether to include self-connections
        
    Returns:
        Binary adjacency matrix
    """
    adj = (coherence_matrix >= threshold).float()
    
    if top_k is not None:
        # Keep only top-k connections per node
        _, indices = torch.topk(coherence_matrix, min(top_k, coherence_matrix.size(-1)), dim=-1)
        mask = torch.zeros_like(coherence_matrix)
        mask.scatter_(-1, indices, 1.0)
        adj = adj * mask
    
    # Symmetrize
    adj = (adj + adj.T) / 2
    adj = (adj > 0).float()
    
    if self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        adj = (adj > 0).float()
    
    return adj


def build_distance_adjacency(
    positions: torch.Tensor,
    k: int = 5
) -> torch.Tensor:
    """
    Build adjacency based on spatial distance between electrodes.
    
    Args:
        positions: Electrode 3D positions (n_electrodes, 3)
        k: Number of nearest neighbors to connect
        
    Returns:
        Binary adjacency matrix
    """
    n_electrodes = positions.size(0)
    
    # Compute pairwise distances
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    distances = torch.norm(diff, dim=-1)
    
    # Get k nearest neighbors
    _, indices = torch.topk(distances, k + 1, dim=-1, largest=False)
    
    # Build adjacency
    adj = torch.zeros(n_electrodes, n_electrodes, device=positions.device)
    for i in range(n_electrodes):
        adj[i, indices[i]] = 1.0
    
    # Symmetrize
    adj = (adj + adj.T) / 2
    adj = (adj > 0).float()
    
    return adj
