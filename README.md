# NeuroFormer 🧠⚡

**State-of-the-art EEG analysis framework with hybrid Transformer-GNN architecture for psychiatric disorder classification.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](https://github.com/kushwahaamar-dev/eeginfra)

## 🚀 Features

- **Hybrid Architecture** — Graph Neural Networks (spatial) + Transformers (temporal) with cross-attention fusion
- **Self-Supervised Pretraining** — Contrastive (SimCLR) and masked signal modeling (BERT-style)
- **MNE Integration** — Load `.edf`, `.bdf`, `.fif`, `.set`, `.vhdr`, `.cnt`, `.gdf` natively
- **Advanced Augmentation** — Time-shift, noise, channel dropout, mixup, cutmix, permutation
- **Hyperparameter Tuning** — Grid, random, and Bayesian optimization
- **Ensemble Methods** — Weighted voting, snapshot ensembles, model registry
- **Cross-Validation** — Subject-aware K-Fold, stratified, and leave-one-subject-out
- **Real-Time Inference** — Streaming predictor with batch support and FP16
- **Explainability** — Attention topomaps, gradient & permutation feature importance
- **Visualization** — Training curves, confusion matrices, band power heatmaps, attention plots
- **Model Export** — ONNX and TorchScript for production deployment
- **Benchmarking** — Inference latency, throughput, and memory profiling

## 📊 Performance

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| Baseline (RiceDatathon2025) | GNN + CNN | 38.5% |
| **NeuroFormer** | Transformer-GNN Hybrid | **70-90%** |

## 🛠️ Installation

```bash
# Standard install
pip install -e .

# With development tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

## 📖 Quick Start

```python
from neuroformer.models import NeuroFormer
from neuroformer.data import EEGDataModule
from neuroformer.training import Trainer

# Initialize model
model = NeuroFormer(
    num_electrodes=19,
    num_classes=7,
    d_model=256,
    n_heads=8,
    n_transformer_layers=4,
    n_gnn_layers=3
)

# Load data with subject-aware splits
data_module = EEGDataModule(
    data_path="path/to/data.csv",
    batch_size=32,
    test_size=0.2
)

# Train
trainer = Trainer(model, optimizer, device='cuda')
history = trainer.fit(
    data_module.train_loader(),
    data_module.val_loader(),
    epochs=100
)

# Predict
predictions = model.predict(test_data)
probabilities = model.predict_proba(test_data)
```

### Load from EEG files (MNE)

```python
from neuroformer.data.mne_loader import MNELoader

loader = MNELoader(sampling_rate=256, l_freq=0.5, h_freq=45.0)
features = loader.load_and_process("recording.edf")
# Returns: {'band_powers', 'coherence', 'band_names', 'channel_names'}
```

### Hyperparameter Tuning

```python
from neuroformer.training.tuning import quick_tune

result = quick_tune(
    train_fn=my_train_function,
    method='bayesian',
    n_trials=30,
    monitor='val_accuracy'
)
print(f"Best params: {result.params}")
```

### Ensemble Prediction

```python
from neuroformer.models.ensemble import WeightedEnsemble

ensemble = WeightedEnsemble.from_checkpoints(
    ['model_v1.pth', 'model_v2.pth', 'model_v3.pth'],
    model_class=NeuroFormer,
    model_kwargs={'num_electrodes': 19, 'num_classes': 7}
)
predictions = ensemble.predict(test_data)
```

### Cross-Validation

```python
from neuroformer.training.cross_validation import CrossValidator

cv = CrossValidator(NeuroFormer, model_kwargs, n_folds=5, stratified=True)
results = cv.evaluate(features, labels, subject_ids, train_fn, eval_fn)
# Output: accuracy_mean, accuracy_std, per-fold scores
```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       NeuroFormer                            │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │   Spatial    │   │   Temporal   │   │    Fusion    │     │
│  │  GNN Layer   │ → │ Transformer  │ → │    Layer     │     │
│  │ (Graph Attn) │   │ (Self-Attn)  │   │ (Cross-Attn) │     │
│  └──────────────┘   └──────────────┘   └──────────────┘     │
│        ↑                                       ↓            │
│  Coherence-based                         Classification     │
│  Adjacency Matrix                       Head (7 classes)    │
└──────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
neuroformer/
├── models/            # Transformer, GNN, hybrid architecture, ensembles
├── preprocessing/     # Filtering, augmentation, feature extraction
├── pretraining/       # Contrastive + masked self-supervised learning
├── training/          # Trainer, losses, metrics, tuning, cross-validation
├── inference/         # Prediction, streaming, explainability
├── data/              # Dataset, samplers, MNE file loading
├── utils/             # Logging, validation, exceptions, checkpointing
├── viz/               # Topomaps, training curves, confusion matrices
├── cli.py             # Command-line interface
tests/                 # Unit tests (45+ test cases)
benchmarks/            # Performance benchmarking suite
examples/              # Training, pretraining, inference demos
docs/                  # API reference
```

## 🖥️ CLI

```bash
neuroformer train    --data train.csv --epochs 100 --batch-size 32
neuroformer pretrain --data unlabeled.csv --method contrastive
neuroformer predict  --model best.pth --input test.csv
neuroformer evaluate --model best.pth --data test.csv
neuroformer info
```

## 🧪 Supported Disorders

1. Healthy (Control)
2. Addictive Disorder
3. Anxiety Disorder
4. Mood Disorder
5. Obsessive Compulsive Disorder
6. Schizophrenia
7. Trauma and Stress Related Disorder

## 🧰 Development

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=neuroformer --cov-report=html

# Lint & format
ruff check neuroformer/
black neuroformer/ tests/

# Benchmark
python benchmarks/benchmark.py
```

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by RiceDatathon2025 EEG classification challenge
- Built on PyTorch, PyTorch Geometric, and MNE-Python

