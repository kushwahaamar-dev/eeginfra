# NeuroFormer API Reference

## Quick Start

```python
from neuroformer import NeuroFormer, NeuroFormerConfig
from neuroformer.training import Trainer
from neuroformer.inference import Predictor

# Create model
model = NeuroFormer(num_electrodes=19, num_classes=7)

# Train
trainer = Trainer(model, optimizer, device='cuda')
trainer.fit(train_loader, val_loader, epochs=100)

# Predict
predictor = Predictor.from_checkpoint('best_model.pth', NeuroFormer)
results = predictor.predict(test_data)
```

---

## Core Classes

### NeuroFormer

Main hybrid Transformer-GNN model.

```python
from neuroformer.models import NeuroFormer

model = NeuroFormer(
    num_electrodes=19,      # Number of EEG electrodes
    num_classes=7,          # Classification classes
    num_bands=5,            # Frequency bands
    d_model=256,            # Hidden dimension
    n_heads=8,              # Attention heads
    n_transformer_layers=4, # Transformer depth
    n_gnn_layers=3,         # GNN depth
    dropout=0.1             # Dropout rate
)
```

**Methods:**
- `forward(band_powers, coherence_matrices=None, return_features=False)` → `Dict`
- `predict(band_powers)` → `Tensor` of class indices
- `predict_proba(band_powers)` → `Tensor` of probabilities

---

### Trainer

Training loop with early stopping and checkpointing.

```python
from neuroformer.training import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,      # Optional, defaults to CombinedLoss
    device='cuda',
    checkpoint_dir='./checkpoints'
)

history = trainer.fit(
    train_loader,
    val_loader,
    epochs=100,
    early_stopping_patience=15
)
```

---

### Predictor

Inference with batch support.

```python
from neuroformer.inference import Predictor

predictor = Predictor(model, device='cuda')

# Single prediction
result = predictor.predict(data, return_proba=True)
# Returns: {'class_indices', 'class_names', 'confidence', 'probabilities'}

# Top-k predictions
top_k = predictor.get_top_k_predictions(data, k=3)
```

---

## Preprocessing

### Filters

```python
from neuroformer.preprocessing import bandpass_filter, notch_filter, zscore_normalize

# Apply bandpass (1-40 Hz)
filtered = bandpass_filter(eeg_data, low=1, high=40, sampling_rate=256)

# Remove 60 Hz line noise
filtered = notch_filter(eeg_data, notch_freq=60, sampling_rate=256)

# Normalize
normalized = zscore_normalize(filtered, axis=-1)
```

### Augmentation

```python
from neuroformer.preprocessing import (
    time_shift, add_gaussian_noise, channel_dropout, mixup
)

# Time shift augmentation
augmented = time_shift(data, max_shift=10)

# Add noise
noisy = add_gaussian_noise(data, noise_std=0.1)

# Mixup two samples
mixed_data, mixed_label = mixup(data1, data2, label1, label2, alpha=0.5)
```

### Features

```python
from neuroformer.preprocessing import compute_band_powers, compute_coherence

# Extract band powers
band_powers = compute_band_powers(eeg_signal, sampling_rate=256)
# Returns: {'delta', 'theta', 'alpha', 'beta', 'gamma'}

# Compute coherence
freqs, coherence = compute_coherence(eeg_signal, sampling_rate=256)
```

---

## Pretraining

### Contrastive Learning

```python
from neuroformer.pretraining import ContrastivePretrainer

pretrainer = ContrastivePretrainer(
    encoder=model,
    d_model=256,
    projection_dim=128,
    temperature=0.07
)

# Training step
loss = pretrainer.pretrain_step(batch, optimizer)
```

### Masked Signal Modeling

```python
from neuroformer.pretraining import MaskedPretrainer

pretrainer = MaskedPretrainer(
    encoder=model,
    d_model=256,
    mask_ratio=0.15
)

loss, predictions, mask = pretrainer(batch)
```

---

## Visualization

```python
from neuroformer.viz import (
    plot_topomap,
    plot_training_curves,
    plot_confusion_matrix
)

# EEG topomap
fig = plot_topomap(values, title="Alpha Power")

# Training curves
fig = plot_training_curves(history, save_path='curves.png')

# Confusion matrix
fig = plot_confusion_matrix(cm, class_names=names, normalize=True)
```

---

## Utilities

### Checkpointing

```python
from neuroformer.utils.checkpoint import (
    save_checkpoint, load_checkpoint,
    export_to_onnx, export_to_torchscript
)

# Save
save_checkpoint(model, 'model.pth', optimizer=opt, epoch=50, metrics=metrics)

# Load
info = load_checkpoint('model.pth', model, device='cuda')

# Export to ONNX
export_to_onnx(model, 'model.onnx')

# Export to TorchScript
export_to_torchscript(model, 'model.pt')
```

### Validation

```python
from neuroformer.utils.validation import validate_eeg_data, check_device

# Validate input data
validated = validate_eeg_data(data, expected_electrodes=19)

# Check device availability
device = check_device('auto')  # Returns 'cuda' or 'cpu'
```

---

## CLI Usage

```bash
# Train model
neuroformer train --data train.csv --epochs 100 --batch-size 32

# Pretrain with SSL
neuroformer pretrain --data unlabeled.csv --method contrastive

# Make predictions
neuroformer predict --model best.pth --input test.csv --output predictions.csv

# Evaluate model
neuroformer evaluate --model best.pth --data test.csv
```
