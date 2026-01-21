"""
Configuration module for NeuroFormer.

Defines hyperparameters, model configs, and training settings.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class NeuroFormerConfig:
    """Configuration for NeuroFormer model."""
    
    # Model architecture
    num_electrodes: int = 19
    num_classes: int = 7
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    
    # GNN settings
    gnn_hidden: int = 128
    gnn_layers: int = 3
    edge_threshold: float = 0.3  # Coherence threshold for edges
    
    # Frequency bands
    freq_bands: dict = field(default_factory=lambda: {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100)
    })
    
    # Preprocessing
    sampling_rate: int = 256
    filter_low: float = 0.5
    filter_high: float = 100.0
    notch_freq: float = 60.0  # Power line frequency
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    warmup_steps: int = 500
    early_stopping_patience: int = 15
    
    # Augmentation
    augment_time_shift: bool = True
    augment_noise: bool = True
    augment_channel_dropout: bool = True
    noise_std: float = 0.1
    channel_dropout_prob: float = 0.1
    
    # Self-supervised pretraining
    ssl_temperature: float = 0.07
    ssl_epochs: int = 50


@dataclass  
class DataConfig:
    """Configuration for data loading and processing."""
    
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Class labels
    class_names: List[str] = field(default_factory=lambda: [
        "Healthy",
        "Addictive Disorder",
        "Anxiety Disorder", 
        "Mood Disorder",
        "Obsessive Compulsive Disorder",
        "Schizophrenia",
        "Trauma and Stress Related Disorder"
    ])
    
    # Electrode positions (10-20 system)
    electrode_names: List[str] = field(default_factory=lambda: [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T3", "C3", "Cz", "C4", "T4",
        "T5", "P3", "Pz", "P4", "T6",
        "O1", "O2"
    ])


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    
    batch_size: int = 1
    use_gpu: bool = True
    precision: str = "fp32"  # fp32, fp16, int8
    explain: bool = False
