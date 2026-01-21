#!/usr/bin/env python
"""
Example: Self-supervised pretraining for NeuroFormer.

Demonstrates how to pretrain the model using:
1. Contrastive learning (SimCLR-style)
2. Masked signal modeling (BERT-style)

Pretraining can significantly improve performance when labeled data is limited.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from neuroformer import NeuroFormerConfig
from neuroformer.models import NeuroFormer
from neuroformer.pretraining import ContrastivePretrainer, MaskedPretrainer


def generate_unlabeled_data(n_samples: int = 5000):
    """Generate synthetic unlabeled EEG data."""
    np.random.seed(42)
    features = np.random.randn(n_samples, 5, 19).astype(np.float32)
    return features


def pretrain_contrastive(model, data, epochs=10, device='cpu'):
    """Run contrastive pretraining."""
    print("\n" + "=" * 50)
    print("Contrastive Pretraining (SimCLR)")
    print("=" * 50)
    
    pretrainer = ContrastivePretrainer(
        encoder=model,
        d_model=128,
        projection_dim=64,
        temperature=0.07
    ).to(device)
    
    optimizer = torch.optim.Adam(pretrainer.parameters(), lr=1e-4)
    
    dataset = TensorDataset(torch.tensor(data))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            loss = pretrainer.pretrain_step(x, optimizer)
            total_loss += loss
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - Contrastive Loss: {avg_loss:.4f}")
    
    return model


def pretrain_masked(model, data, epochs=10, device='cpu'):
    """Run masked signal modeling pretraining."""
    print("\n" + "=" * 50)
    print("Masked Signal Modeling Pretraining")
    print("=" * 50)
    
    pretrainer = MaskedPretrainer(
        encoder=model,
        d_model=128,
        num_electrodes=19,
        num_bands=5,
        mask_ratio=0.15
    ).to(device)
    
    optimizer = torch.optim.Adam(pretrainer.parameters(), lr=1e-4)
    
    dataset = TensorDataset(torch.tensor(data))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            loss = pretrainer.pretrain_step(x, optimizer)
            total_loss += loss
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - Masked Loss: {avg_loss:.4f}")
    
    return model


def main():
    print("=" * 60)
    print("NeuroFormer Self-Supervised Pretraining Example")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Generate unlabeled data
    print("\n1. Generating unlabeled data...")
    unlabeled_data = generate_unlabeled_data(n_samples=2000)
    print(f"   Data shape: {unlabeled_data.shape}")
    
    # Create model
    print("\n2. Creating NeuroFormer model...")
    model = NeuroFormer(
        num_electrodes=19,
        num_classes=7,
        d_model=128,
        n_heads=4,
        n_transformer_layers=2,
        dropout=0.1
    )
    
    # Option 1: Contrastive pretraining
    model = pretrain_contrastive(model, unlabeled_data, epochs=5, device=device)
    
    # Option 2: Masked pretraining (can be done after or instead of contrastive)
    model = pretrain_masked(model, unlabeled_data, epochs=5, device=device)
    
    # Save pretrained weights
    print("\n" + "=" * 50)
    print("Saving pretrained model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'pretrain_epochs': 10
    }, './pretrained_neuroformer.pth')
    print("✓ Saved to ./pretrained_neuroformer.pth")
    
    print("\nPretraining complete!")
    print("You can now fine-tune this model on labeled data using train_psychiatric.py")


if __name__ == '__main__':
    main()
