# Changelog

All notable changes to NeuroFormer will be documented in this file.

## [0.2.0] - 2026-01-22

### Added
- **Robustness Enhancements**
  - Custom exception hierarchy (`NeuroFormerError`, `DataValidationError`, etc.)
  - Comprehensive logging system with file support
  - Input validation for data, configs, and checkpoints
  - 45+ unit tests for models and preprocessing

- **Model Checkpointing**
  - `ModelCheckpoint` class with versioning and top-k tracking
  - ONNX export support
  - TorchScript export support
  - Model summary utilities

- **Visualization**
  - EEG topomap plotting
  - Training curves visualization
  - Confusion matrix with normalization
  - Band power heatmaps
  - Attention weight visualization
  - `TrainingVisualizer` for real-time plotting

- **Documentation**
  - Contributing guide
  - API reference
  - Changelog

### Changed
- Bumped version to 0.2.0 (Beta status)
- Added Python 3.12 support
- Improved pytest configuration
- Added mypy and pytest-xdist to dev dependencies

---

## [0.1.0] - 2026-01-22

### Added
- **Foundation + Preprocessing**
  - Project structure and pyproject.toml
  - Configuration dataclasses
  - Signal filters (bandpass, notch, highpass)
  - ICA artifact removal
  - Data augmentation (time-shift, noise, mixup, cutmix)
  - Feature extraction (PSD, band powers, coherence, asymmetry)

- **Hybrid Architecture**
  - Positional and electrode embeddings (10-20 system)
  - Multi-head self-attention
  - Cross-attention for fusion
  - Efficient linear attention
  - Graph Attention Network (GAT) layers
  - Spatial GNN for electrode graphs
  - Temporal Transformer encoder
  - Main NeuroFormer hybrid model
  - NeuroFormerLite lightweight version

- **Self-Supervised Pretraining**
  - Contrastive learning (SimCLR-style)
  - NT-Xent loss
  - Masked signal modeling (BERT-style)
  - Combined pretraining support

- **Training Pipeline**
  - Focal loss for class imbalance
  - Label smoothing loss
  - Center loss for feature clustering
  - Comprehensive metrics (F1, balanced accuracy)
  - Trainer with early stopping
  - Learning rate scheduling
  - Automatic checkpointing

- **Inference**
  - Predictor with batch support
  - Real-time streaming predictor
  - Top-k predictions
  - Attention visualization
  - Feature importance computation

- **CLI**
  - `neuroformer train` command
  - `neuroformer pretrain` command
  - `neuroformer predict` command
  - `neuroformer evaluate` command
  - `neuroformer info` command

- **Examples**
  - Training pipeline example
  - Self-supervised pretraining example
  - Inference demo

### Target Performance
- 70-90%+ accuracy on psychiatric disorder classification
- Significant improvement over baseline (~38.5%)
