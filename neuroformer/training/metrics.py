"""
Evaluation metrics for EEG classification.

Provides accuracy, F1, balanced accuracy, and confusion matrix utilities.
"""

import torch
import numpy as np
from typing import Dict, Optional, Union, List
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def compute_accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted class indices or logits
        targets: Ground truth class indices
        
    Returns:
        Accuracy as float
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # If predictions are logits, take argmax
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    
    return accuracy_score(targets, predictions)


def compute_f1(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    average: str = 'macro',
    zero_division: int = 0
) -> float:
    """
    Compute F1 score.
    
    Args:
        predictions: Predicted class indices or logits
        targets: Ground truth class indices
        average: 'micro', 'macro', 'weighted', or None
        zero_division: Value to return when there's a zero division
        
    Returns:
        F1 score as float
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    
    return f1_score(targets, predictions, average=average, zero_division=zero_division)


def compute_balanced_accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute balanced accuracy (average recall per class).
    
    Better metric for imbalanced datasets than standard accuracy.
    
    Args:
        predictions: Predicted class indices or logits
        targets: Ground truth class indices
        
    Returns:
        Balanced accuracy as float
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    
    return balanced_accuracy_score(targets, predictions)


def compute_confusion_matrix(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: Optional[int] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices or logits
        targets: Ground truth class indices
        num_classes: Number of classes (for labels)
        normalize: 'true', 'pred', 'all', or None
        
    Returns:
        Confusion matrix as numpy array (num_classes, num_classes)
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    
    labels = list(range(num_classes)) if num_classes else None
    
    return sklearn_confusion_matrix(targets, predictions, labels=labels, normalize=normalize)


def compute_per_class_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1 per class.
    
    Args:
        predictions: Predicted class indices or logits
        targets: Ground truth class indices
        class_names: Optional list of class names
        
    Returns:
        Dictionary with metrics per class
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, zero_division=0
    )
    
    num_classes = len(precision)
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    metrics = {}
    for i, name in enumerate(class_names[:num_classes]):
        metrics[name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return metrics


def compute_all_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted class indices or logits
        targets: Ground truth class indices
        class_names: Optional list of class names
        
    Returns:
        Dictionary with all metrics
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=-1)
    
    num_classes = len(np.unique(targets))
    
    return {
        'accuracy': compute_accuracy(predictions, targets),
        'balanced_accuracy': compute_balanced_accuracy(predictions, targets),
        'f1_macro': compute_f1(predictions, targets, average='macro'),
        'f1_weighted': compute_f1(predictions, targets, average='weighted'),
        'confusion_matrix': compute_confusion_matrix(predictions, targets, num_classes),
        'per_class': compute_per_class_metrics(predictions, targets, class_names)
    }


class MetricTracker:
    """
    Track and aggregate metrics during training.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Args:
            class_names: Optional list of class names
        """
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all tracked predictions and targets."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None
    ):
        """
        Add batch predictions and targets.
        
        Args:
            predictions: Batch predictions (logits or indices)
            targets: Batch targets
            loss: Optional loss value
        """
        self.predictions.append(to_numpy(predictions))
        self.targets.append(to_numpy(targets))
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """
        Compute all metrics from accumulated predictions.
        
        Returns:
            Dictionary with all metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        predictions = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        
        metrics = compute_all_metrics(predictions, targets, self.class_names)
        
        if len(self.losses) > 0:
            metrics['loss'] = np.mean(self.losses)
        
        return metrics
