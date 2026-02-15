"""
Ensemble methods for NeuroFormer.

Provides model ensembling strategies for improved accuracy and robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from neuroformer.utils.logging import get_logger

logger = get_logger(__name__)


class WeightedEnsemble(nn.Module):
    """
    Weighted ensemble of multiple NeuroFormer models.
    
    Combines predictions from multiple models using learnable
    or fixed weights for improved classification.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        strategy: str = 'soft_vote'
    ):
        """
        Args:
            models: List of trained models
            weights: Optional model weights (uniform if None)
            strategy: 'soft_vote', 'hard_vote', or 'stacking'
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.strategy = strategy
        
        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models
        
        self.weights = nn.Parameter(
            torch.tensor(weights, dtype=torch.float32),
            requires_grad=False
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            **kwargs: Additional model arguments
            
        Returns:
            Ensemble predictions
        """
        all_logits = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x, **kwargs)
                all_logits.append(output['logits'])
        
        stacked = torch.stack(all_logits, dim=0)  # (n_models, batch, classes)
        
        if self.strategy == 'soft_vote':
            # Weighted average of probabilities
            probs = F.softmax(stacked, dim=-1)
            weighted = torch.einsum('mbc,m->bc', probs, self.weights)
            logits = torch.log(weighted + 1e-8)
            
        elif self.strategy == 'hard_vote':
            # Majority voting
            predictions = stacked.argmax(dim=-1)  # (n_models, batch)
            logits = torch.zeros_like(stacked[0])
            for m in range(self.n_models):
                for b in range(predictions.shape[1]):
                    logits[b, predictions[m, b]] += self.weights[m]
                    
        else:
            # Simple average
            logits = (stacked * self.weights.view(-1, 1, 1)).sum(dim=0)
        
        return {
            'logits': logits,
            'individual_logits': stacked,
            'agreement': self._compute_agreement(stacked)
        }
    
    def _compute_agreement(self, stacked_logits: torch.Tensor) -> torch.Tensor:
        """Compute prediction agreement between models."""
        predictions = stacked_logits.argmax(dim=-1)  # (n_models, batch)
        modes = predictions.mode(dim=0).values
        agreement = (predictions == modes.unsqueeze(0)).float().mean(dim=0)
        return agreement
    
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        output = self.forward(x, **kwargs)
        return output['logits'].argmax(dim=-1)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: List[str],
        model_class: type,
        model_kwargs: Dict = None,
        device: str = 'cpu',
        **kwargs
    ) -> 'WeightedEnsemble':
        """
        Create ensemble from saved checkpoints.
        
        Args:
            checkpoint_paths: List of checkpoint paths
            model_class: Model class to instantiate
            model_kwargs: Model constructor arguments
            device: Device to load models on
            
        Returns:
            WeightedEnsemble instance
        """
        model_kwargs = model_kwargs or {}
        models = []
        
        for path in checkpoint_paths:
            model = model_class(**model_kwargs)
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
            logger.info(f"Loaded model from {path}")
        
        return cls(models, **kwargs)


class SnapshotEnsemble:
    """
    Snapshot ensemble using cyclic learning rate.
    
    Collects model snapshots at learning rate cycle minima
    for a free ensemble without extra training cost.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_cycles: int = 5,
        save_dir: str = './snapshots'
    ):
        self.model = model
        self.n_cycles = n_cycles
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots = []
    
    def get_lr_schedule(
        self,
        initial_lr: float,
        total_epochs: int
    ) -> List[float]:
        """
        Generate cyclic cosine annealing schedule.
        
        Args:
            initial_lr: Maximum learning rate
            total_epochs: Total training epochs
            
        Returns:
            List of learning rates per epoch
        """
        epochs_per_cycle = total_epochs // self.n_cycles
        lrs = []
        
        for epoch in range(total_epochs):
            cycle_epoch = epoch % epochs_per_cycle
            lr = initial_lr / 2 * (np.cos(np.pi * cycle_epoch / epochs_per_cycle) + 1)
            lrs.append(lr)
        
        return lrs
    
    def should_snapshot(self, epoch: int, total_epochs: int) -> bool:
        """Check if current epoch is a snapshot point."""
        epochs_per_cycle = total_epochs // self.n_cycles
        if epochs_per_cycle == 0:
            return False
        return (epoch + 1) % epochs_per_cycle == 0
    
    def save_snapshot(self, epoch: int):
        """Save model snapshot."""
        path = self.save_dir / f'snapshot_epoch{epoch}.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch
        }, path)
        self.snapshots.append(str(path))
        logger.info(f"Snapshot saved: {path}")
    
    def build_ensemble(
        self,
        model_class: type,
        model_kwargs: Dict = None,
        device: str = 'cpu'
    ) -> WeightedEnsemble:
        """Build ensemble from collected snapshots."""
        return WeightedEnsemble.from_checkpoints(
            self.snapshots,
            model_class,
            model_kwargs,
            device
        )


class ModelRegistry:
    """
    Registry for managing multiple trained model versions.
    """
    
    def __init__(self, registry_dir: str = './model_registry'):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / 'registry.json'
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict:
        import json
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        import json
        with open(self.registry_file, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def register(
        self,
        name: str,
        model: nn.Module,
        version: str,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a trained model.
        
        Args:
            name: Model name
            model: Trained model
            version: Version string
            metrics: Performance metrics
            config: Model configuration
            tags: Optional tags
            
        Returns:
            Path to saved model
        """
        import json
        from datetime import datetime
        
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / 'model.pth'
        torch.save({'model_state_dict': model.state_dict()}, model_path)
        
        # Save metadata
        metadata = {
            'name': name,
            'version': version,
            'metrics': metrics,
            'config': config or {},
            'tags': tags or [],
            'registered_at': datetime.now().isoformat(),
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry
        key = f"{name}/{version}"
        self.models[key] = metadata
        self._save_registry()
        
        logger.info(f"Registered model: {key}")
        return str(model_path)
    
    def load(
        self,
        name: str,
        version: str,
        model_class: type,
        model_kwargs: Dict = None,
        device: str = 'cpu'
    ) -> Tuple[nn.Module, Dict]:
        """Load a registered model."""
        model_kwargs = model_kwargs or {}
        model_dir = self.registry_dir / name / version
        model_path = model_dir / 'model.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = model_class(**model_kwargs)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        import json
        with open(model_dir / 'metadata.json') as f:
            metadata = json.load(f)
        
        return model, metadata
    
    def list_models(self) -> List[Dict]:
        """List all registered models."""
        return [
            {'key': k, **v}
            for k, v in sorted(self.models.items())
        ]
    
    def get_best(self, name: str, metric: str = 'val_accuracy') -> Optional[str]:
        """Get version with best metric for a model name."""
        best_version = None
        best_score = -float('inf')
        
        for key, meta in self.models.items():
            if key.startswith(f"{name}/"):
                score = meta.get('metrics', {}).get(metric, -float('inf'))
                if score > best_score:
                    best_score = score
                    best_version = meta['version']
        
        return best_version
