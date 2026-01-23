# Contributing to NeuroFormer

Thank you for your interest in contributing to NeuroFormer! This guide will help you get started.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kushwahaamar-dev/eeginfra.git
   cd neuroformer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify installation:**
   ```bash
   pytest tests/ -v
   ```

## Code Standards

### Style Guide
- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use docstrings (Google style) for all public functions/classes

### Type Hints
```python
def process_eeg(
    data: np.ndarray,
    sampling_rate: int = 256,
    normalize: bool = True
) -> np.ndarray:
    """
    Process EEG data.
    
    Args:
        data: Raw EEG data (channels, samples)
        sampling_rate: Sampling rate in Hz
        normalize: Whether to normalize
        
    Returns:
        Processed EEG data
    """
    pass
```

### Running Formatters
```bash
# Format code
black neuroformer/ tests/

# Check linting
ruff check neuroformer/ tests/

# Type checking
mypy neuroformer/
```

## Testing

### Running Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_models.py -v

# With coverage
pytest tests/ --cov=neuroformer --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Writing Tests
- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures from `conftest.py`

Example:
```python
import pytest
from neuroformer.models import NeuroFormer

class TestNeuroFormer:
    @pytest.fixture
    def model(self):
        return NeuroFormer(num_electrodes=19, num_classes=7)
    
    def test_forward_pass(self, model):
        x = torch.randn(4, 5, 19)
        output = model(x)
        assert output['logits'].shape == (4, 7)
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   ```

3. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **PR Requirements:**
   - All tests must pass
   - Code must be formatted with Black
   - Include tests for new functionality
   - Update documentation if needed

## Commit Message Format

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `perf:` Performance improvements

## Architecture Overview

```
neuroformer/
├── models/          # Neural network architectures
│   ├── embeddings.py    # Positional, electrode embeddings
│   ├── attention.py     # Attention mechanisms
│   ├── gnn_layers.py    # Graph neural network layers
│   ├── transformer.py   # Transformer encoder
│   └── neuroformer.py   # Main hybrid model
├── preprocessing/   # Data preprocessing
├── training/        # Training utilities
├── pretraining/     # Self-supervised learning
├── inference/       # Prediction and explainability
├── utils/           # Logging, validation, checkpointing
└── viz/             # Visualization tools
```

## Questions?

Open an issue on GitHub for questions or discussions.
