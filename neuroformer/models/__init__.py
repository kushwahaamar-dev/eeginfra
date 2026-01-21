"""
Models subpackage for NeuroFormer.

Contains the hybrid Transformer-GNN architecture components.
"""

from neuroformer.models.embeddings import (
    PositionalEncoding,
    ElectrodeEmbedding,
    BandEmbedding,
)
from neuroformer.models.attention import (
    MultiHeadSelfAttention,
    CrossAttention,
)
from neuroformer.models.gnn_layers import (
    GraphAttentionLayer,
    SpatialGNN,
)
from neuroformer.models.transformer import (
    TransformerEncoderLayer,
    TemporalTransformer,
)
from neuroformer.models.neuroformer import NeuroFormer

__all__ = [
    "PositionalEncoding",
    "ElectrodeEmbedding",
    "BandEmbedding",
    "MultiHeadSelfAttention",
    "CrossAttention",
    "GraphAttentionLayer",
    "SpatialGNN",
    "TransformerEncoderLayer",
    "TemporalTransformer",
    "NeuroFormer",
]
