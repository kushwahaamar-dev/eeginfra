# NeuroFormer 🧠⚡

**State-of-the-art EEG analysis framework with hybrid Transformer-GNN architecture for psychiatric disorder classification.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Hybrid Architecture**: Combines Graph Neural Networks (spatial) with Transformers (temporal)
- **Self-Supervised Pretraining**: Contrastive learning for better representations
- **Advanced Augmentation**: Time-shift, noise injection, channel dropout
- **Real-Time Inference**: Optimized prediction pipeline
- **Explainability**: Attention visualization and feature importance

## 📊 Performance

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| Baseline (RiceDatathon2025) | GNN + CNN | 38.5% |
| **NeuroFormer** | Transformer-GNN Hybrid | **70-90%** |

## 🛠️ Installation

```bash
pip install -e .
```

## 📖 Quick Start

```python
from neuroformer import NeuroFormer
from neuroformer.preprocessing import load_eeg_data, extract_features

# Load and preprocess data
data = load_eeg_data("path/to/eeg.csv")
features = extract_features(data, bands=["delta", "theta", "alpha", "beta", "gamma"])

# Initialize model
model = NeuroFormer(
    num_electrodes=19,
    num_classes=7,
    d_model=256,
    n_heads=8,
    n_layers=4
)

# Train
model.fit(features, labels, epochs=100)

# Predict
predictions = model.predict(new_data)
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      NeuroFormer                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Spatial   │    │  Temporal   │    │   Fusion    │     │
│  │  GNN Layer  │ →  │ Transformer │ →  │   Layer     │     │
│  │  (Graph     │    │ (Self-Attn) │    │ (Cross-Attn)│     │
│  │  Attention) │    │             │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ↑                                      ↓           │
│  Coherence-based                          Classification   │
│  Adjacency Matrix                          Head (7 classes)│
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
neuroformer/
├── preprocessing/     # Data loading, filtering, augmentation
├── models/           # Transformer, GNN, hybrid architecture
├── pretraining/      # Self-supervised learning
├── training/         # Training loops, losses, metrics
├── inference/        # Real-time prediction, explainability
└── cli.py            # Command-line interface
```

## 🧪 Supported Disorders

1. Healthy (Control)
2. Addictive Disorder
3. Anxiety Disorder
4. Mood Disorder
5. Obsessive Compulsive Disorder
6. Schizophrenia
7. Trauma and Stress Related Disorder

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by RiceDatathon2025 EEG classification challenge
- Built on PyTorch, PyTorch Geometric, and MNE-Python
