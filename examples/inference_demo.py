#!/usr/bin/env python
"""
Example: Inference with pretrained NeuroFormer.

Demonstrates:
1. Loading a trained model
2. Making predictions on new data
3. Interpreting results with explainability tools
"""

import numpy as np
import torch

from neuroformer.models import NeuroFormer
from neuroformer.inference import Predictor, AttentionVisualizer, compute_feature_importance


def generate_test_sample():
    """Generate a synthetic test sample."""
    np.random.seed(123)
    return np.random.randn(1, 5, 19).astype(np.float32)


def main():
    print("=" * 60)
    print("NeuroFormer Inference Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model (in practice, load from checkpoint)
    print("\n1. Loading model...")
    model = NeuroFormer(
        num_electrodes=19,
        num_classes=7,
        d_model=128,
        n_heads=4,
        n_transformer_layers=2,
        dropout=0.1
    )
    model.eval()
    print("   Model loaded (using random weights for demo)")
    
    # Create predictor
    predictor = Predictor(model, device=device)
    
    # Generate test data
    print("\n2. Preparing test data...")
    test_data = generate_test_sample()
    print(f"   Sample shape: {test_data.shape}")
    
    # Make prediction
    print("\n3. Making prediction...")
    result = predictor.predict(test_data, return_proba=True)
    
    print("\n   Prediction Results:")
    print("   " + "-" * 40)
    print(f"   Predicted class: {result['class_names'][0]}")
    print(f"   Confidence: {result['confidence'][0]:.3f}")
    
    print("\n   Class Probabilities:")
    for i, (name, prob) in enumerate(zip(predictor.class_names, result['probabilities'][0])):
        bar = "█" * int(prob * 30)
        print(f"   {name:35} {prob:.3f} {bar}")
    
    # Top-k predictions
    print("\n4. Top-3 predictions:")
    top_k = predictor.get_top_k_predictions(test_data, k=3)
    for rank, (name, prob) in enumerate(top_k[0], 1):
        print(f"   {rank}. {name}: {prob:.3f}")
    
    # Feature importance (gradient-based)
    print("\n5. Computing feature importance...")
    test_tensor = torch.tensor(test_data)
    target = torch.tensor([0])  # Assume true class is 0
    
    importance = compute_feature_importance(model, test_tensor, target, method='gradient')
    print(f"   Importance shape: {importance.shape}")
    
    # Show most important electrodes per band
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    print("\n   Most important electrodes per band:")
    for b, band in enumerate(band_names):
        top_electrode = np.argmax(importance[b])
        print(f"   {band}: Electrode {top_electrode} (importance: {importance[b, top_electrode]:.4f})")
    
    print("\n" + "=" * 60)
    print("Inference demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
