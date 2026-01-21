#!/usr/bin/env python
"""
Example: Training NeuroFormer for psychiatric disorder classification.

This script demonstrates the full training pipeline:
1. Loading and preprocessing EEG data
2. Creating the NeuroFormer model
3. Training with early stopping
4. Evaluating on test set
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import NeuroFormer components
from neuroformer import NeuroFormerConfig
from neuroformer.models import NeuroFormer
from neuroformer.data import EEGDataset, EEGDataModule
from neuroformer.training import Trainer, CombinedLoss
from neuroformer.training.metrics import compute_all_metrics


def generate_synthetic_data(n_samples: int = 1000, n_electrodes: int = 19, n_bands: int = 5, n_classes: int = 7):
    """Generate synthetic EEG data for demonstration."""
    np.random.seed(42)
    
    # Generate random band powers
    features = np.random.randn(n_samples, n_bands, n_electrodes).astype(np.float32)
    
    # Generate random labels
    labels = np.random.randint(0, n_classes, size=n_samples)
    
    # Add some class-specific patterns
    for c in range(n_classes):
        mask = labels == c
        # Each class has slightly different band power patterns
        features[mask, c % n_bands, :] += 0.5
    
    return features, labels


def main():
    print("=" * 60)
    print("NeuroFormer Training Example")
    print("=" * 60)
    
    # Configuration
    config = NeuroFormerConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate synthetic data (replace with real data loading)
    print("\n1. Loading data...")
    features, labels = generate_synthetic_data(n_samples=1000)
    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Class distribution: {np.bincount(labels)}")
    
    # Create data module with train/val/test splits
    print("\n2. Creating data splits...")
    data_module = EEGDataModule(
        features=features,
        labels=labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=32,
        num_workers=0
    )
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    print(f"   Train samples: {len(data_module.train_idx)}")
    print(f"   Val samples: {len(data_module.val_idx)}")
    print(f"   Test samples: {len(data_module.test_idx)}")
    
    # Create model
    print("\n3. Creating NeuroFormer model...")
    model = NeuroFormer(
        num_electrodes=config.num_electrodes,
        num_classes=config.num_classes,
        d_model=128,  # Smaller for demo
        n_heads=4,
        n_transformer_layers=2,
        n_gnn_layers=2,
        dropout=0.1
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {n_params:,}")
    
    # Setup training
    print("\n4. Setting up training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    class_weights = data_module.get_class_weights()
    criterion = CombinedLoss(
        num_classes=config.num_classes,
        use_focal=True,
        use_smoothing=True,
        class_weights=class_weights
    )
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir='./checkpoints',
        class_names=[
            "Healthy", "Addictive", "Anxiety", "Mood",
            "OCD", "Schizophrenia", "Trauma"
        ]
    )
    
    # Train
    print("\n5. Training model...")
    print("-" * 60)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,  # Short for demo
        early_stopping_patience=5
    )
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    print("-" * 60)
    test_metrics = trainer.validate(test_loader, epoch=0)
    
    print("\nTest Results:")
    print(f"   Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print(f"   Balanced Accuracy: {test_metrics.get('balanced_accuracy', 0):.4f}")
    print(f"   F1 (Macro): {test_metrics.get('f1_macro', 0):.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model saved to: ./checkpoints/best_model.pth")
    print("=" * 60)


if __name__ == '__main__':
    main()
