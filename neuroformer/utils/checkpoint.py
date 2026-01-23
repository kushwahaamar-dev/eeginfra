"""
Model checkpointing and export utilities.

Provides utilities for saving, loading, and exporting trained models.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
from datetime import datetime

from neuroformer.utils.logging import get_logger
from neuroformer.utils.exceptions import CheckpointError

logger = get_logger(__name__)


class ModelCheckpoint:
    """
    Advanced model checkpointing with versioning and metadata.
    """
    
    def __init__(
        self,
        save_dir: str,
        model_name: str = "neuroformer",
        save_top_k: int = 3,
        monitor: str = "val_accuracy",
        mode: str = "max"
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            model_name: Base name for checkpoint files
            save_top_k: Keep only top k checkpoints
            monitor: Metric to monitor for best models
            mode: 'max' or 'min' for the monitored metric
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        
        self.best_scores = []
        self.checkpoint_history = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optional optimizer state
            epoch: Current epoch
            metrics: Training/validation metrics
            config: Model configuration
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_epoch{epoch}_{timestamp}.pth"
        filepath = self.save_dir / filename
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": metrics or {},
            "config": config or {},
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        self.checkpoint_history.append(str(filepath))
        
        logger.info(f"Saved checkpoint: {filepath}")
        
        # Handle best model tracking
        if metrics and self.monitor in metrics:
            score = metrics[self.monitor]
            self._update_best(filepath, score, model, optimizer, epoch, metrics, config)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(filepath)
    
    def _update_best(self, filepath, score, model, optimizer, epoch, metrics, config):
        """Update best model tracking."""
        is_better = (
            len(self.best_scores) == 0 or
            (self.mode == "max" and score > min(s for s, _ in self.best_scores)) or
            (self.mode == "min" and score < max(s for s, _ in self.best_scores))
        )
        
        if is_better or len(self.best_scores) < self.save_top_k:
            best_path = self.save_dir / f"{self.model_name}_best.pth"
            
            # Save as best if it's the new best
            if len(self.best_scores) == 0 or (
                (self.mode == "max" and score >= max(s for s, _ in self.best_scores)) or
                (self.mode == "min" and score <= min(s for s, _ in self.best_scores))
            ):
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "config": config,
                    "best_score": score,
                    "monitor": self.monitor,
                }
                if optimizer:
                    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                
                torch.save(checkpoint, best_path)
                logger.info(f"New best model! {self.monitor}={score:.4f}")
            
            self.best_scores.append((score, str(filepath)))
            self.best_scores.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
            
            if len(self.best_scores) > self.save_top_k:
                self.best_scores = self.best_scores[:self.save_top_k]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond save_top_k."""
        if len(self.checkpoint_history) > self.save_top_k * 2:
            # Keep recent and best checkpoints
            best_paths = {p for _, p in self.best_scores}
            to_remove = []
            
            for path in self.checkpoint_history[:-self.save_top_k]:
                if path not in best_paths and Path(path).exists():
                    to_remove.append(path)
            
            for path in to_remove:
                Path(path).unlink()
                self.checkpoint_history.remove(path)
                logger.debug(f"Removed old checkpoint: {path}")
    
    def load_best(self, model: nn.Module, device: str = "cpu") -> Dict:
        """Load the best model checkpoint."""
        best_path = self.save_dir / f"{self.model_name}_best.pth"
        if not best_path.exists():
            raise CheckpointError("No best checkpoint found", path=str(best_path))
        
        return load_checkpoint(str(best_path), model, device)


def save_checkpoint(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> None:
    """
    Simple checkpoint saving function.
    
    Args:
        model: Model to save
        filepath: Path to save checkpoint
        optimizer: Optional optimizer
        epoch: Current epoch
        metrics: Optional metrics dictionary
        config: Optional config dictionary
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    device: str = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        device: Device to load on
        optimizer: Optional optimizer to restore
        strict: Whether to strictly enforce state dict keys
        
    Returns:
        Checkpoint dictionary with metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise CheckpointError("Checkpoint not found", path=str(filepath))
    
    try:
        checkpoint = torch.load(filepath, map_location=device)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint: {e}", path=str(filepath))
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    logger.info(f"Model loaded from {filepath}")
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
    }


def export_to_onnx(
    model: nn.Module,
    filepath: str,
    input_shape: tuple = (1, 5, 19),
    opset_version: int = 14,
    dynamic_axes: Optional[Dict] = None
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: Model to export
        filepath: Output path (.onnx)
        input_shape: Example input shape (batch, bands, electrodes)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable batch size
        
    Returns:
        Path to exported model
    """
    filepath = Path(filepath)
    if filepath.suffix != ".onnx":
        filepath = filepath.with_suffix(".onnx")
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    
    torch.onnx.export(
        model,
        dummy_input,
        str(filepath),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    
    logger.info(f"Model exported to ONNX: {filepath}")
    return str(filepath)


def export_to_torchscript(
    model: nn.Module,
    filepath: str,
    input_shape: tuple = (1, 5, 19),
    method: str = "trace"
) -> str:
    """
    Export model to TorchScript format.
    
    Args:
        model: Model to export
        filepath: Output path (.pt)
        input_shape: Example input shape
        method: 'trace' or 'script'
        
    Returns:
        Path to exported model
    """
    filepath = Path(filepath)
    if filepath.suffix != ".pt":
        filepath = filepath.with_suffix(".pt")
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    if method == "trace":
        dummy_input = torch.randn(*input_shape)
        scripted = torch.jit.trace(model, dummy_input)
    elif method == "script":
        scripted = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")
    
    scripted.save(str(filepath))
    logger.info(f"Model exported to TorchScript: {filepath}")
    
    return str(filepath)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get model summary information.
    
    Args:
        model: Model to summarize
        
    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "layers": len(list(model.modules())),
    }
