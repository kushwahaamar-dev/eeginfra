"""
Cross-validation utilities for NeuroFormer.

Subject-aware k-fold cross-validation to ensure proper evaluation
of EEG classification models.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Generator
from collections import defaultdict

from neuroformer.utils.logging import get_logger

logger = get_logger(__name__)


class SubjectKFold:
    """
    Subject-aware K-Fold cross-validation.
    
    Ensures that all samples from one subject are in the same fold,
    preventing data leakage in EEG classification.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        subject_ids: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for subject-aware splits.
        
        Args:
            subject_ids: Subject ID for each sample
            labels: Optional labels for stratification info
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        if n_subjects < self.n_folds:
            raise ValueError(
                f"Cannot split {n_subjects} subjects into {self.n_folds} folds"
            )
        
        rng = np.random.RandomState(self.random_state)
        
        if self.shuffle:
            rng.shuffle(unique_subjects)
        
        fold_sizes = np.full(self.n_folds, n_subjects // self.n_folds)
        fold_sizes[:n_subjects % self.n_folds] += 1
        
        current = 0
        for fold_idx in range(self.n_folds):
            fold_subjects = unique_subjects[current:current + fold_sizes[fold_idx]]
            current += fold_sizes[fold_idx]
            
            test_mask = np.isin(subject_ids, fold_subjects)
            train_indices = np.where(~test_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            logger.debug(
                f"Fold {fold_idx+1}: {len(train_indices)} train, "
                f"{len(test_indices)} test samples "
                f"({len(fold_subjects)} test subjects)"
            )
            
            yield train_indices, test_indices


class StratifiedSubjectKFold:
    """
    Stratified subject-aware cross-validation.
    
    Balances class distribution across folds while maintaining
    subject integrity.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        subject_ids: np.ndarray,
        labels: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate stratified subject-aware splits.
        
        Args:
            subject_ids: Subject ID for each sample
            labels: Class labels for stratification
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        rng = np.random.RandomState(self.random_state)
        
        # Determine majority label per subject
        unique_subjects = np.unique(subject_ids)
        subject_labels = {}
        
        for subj in unique_subjects:
            mask = subject_ids == subj
            subj_labels = labels[mask]
            # Use most common label
            unique, counts = np.unique(subj_labels, return_counts=True)
            subject_labels[subj] = unique[counts.argmax()]
        
        # Group subjects by their majority label
        label_subjects = defaultdict(list)
        for subj, label in subject_labels.items():
            label_subjects[label].append(subj)
        
        if self.shuffle:
            for label in label_subjects:
                rng.shuffle(label_subjects[label])
        
        # Distribute subjects across folds, balancing labels
        folds = [[] for _ in range(self.n_folds)]
        
        for label in sorted(label_subjects.keys()):
            subjects = label_subjects[label]
            for i, subj in enumerate(subjects):
                folds[i % self.n_folds].append(subj)
        
        # Generate splits
        for fold_idx in range(self.n_folds):
            test_subjects = set(folds[fold_idx])
            
            test_mask = np.array([s in test_subjects for s in subject_ids])
            train_indices = np.where(~test_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            yield train_indices, test_indices


class CrossValidator:
    """
    Run cross-validation evaluation for NeuroFormer.
    """
    
    def __init__(
        self,
        model_class: type,
        model_kwargs: Dict,
        n_folds: int = 5,
        stratified: bool = True,
        device: str = 'auto'
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.n_folds = n_folds
        self.stratified = stratified
        self.device = device if device != 'auto' else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
        train_fn,
        eval_fn,
        epochs: int = 50
    ) -> Dict[str, List[float]]:
        """
        Run cross-validation.
        
        Args:
            features: Input features (n_samples, ...)
            labels: Class labels (n_samples,)
            subject_ids: Subject IDs (n_samples,)
            train_fn: Function(model, train_data, train_labels, epochs) -> model
            eval_fn: Function(model, test_data, test_labels) -> metrics dict
            epochs: Training epochs per fold
            
        Returns:
            Dict mapping metric names to lists of per-fold scores
        """
        if self.stratified:
            splitter = StratifiedSubjectKFold(self.n_folds)
        else:
            splitter = SubjectKFold(self.n_folds)
        
        all_metrics = defaultdict(list)
        
        for fold_idx, (train_idx, test_idx) in enumerate(
            splitter.split(subject_ids, labels)
        ):
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold_idx+1}/{self.n_folds}")
            logger.info(f"{'='*50}")
            
            # Split data
            X_train = features[train_idx]
            y_train = labels[train_idx]
            X_test = features[test_idx]
            y_test = labels[test_idx]
            
            # Train
            model = self.model_class(**self.model_kwargs).to(self.device)
            model = train_fn(model, X_train, y_train, epochs)
            
            # Evaluate
            metrics = eval_fn(model, X_test, y_test)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            logger.info(f"Fold {fold_idx+1}: {metrics}")
        
        # Compute summary statistics
        summary = {}
        for key, values in all_metrics.items():
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))
            summary[f'{key}_folds'] = values
        
        logger.info(f"\nCross-Validation Results:")
        for key in all_metrics:
            mean = np.mean(all_metrics[key])
            std = np.std(all_metrics[key])
            logger.info(f"  {key}: {mean:.4f} ± {std:.4f}")
        
        return summary


class LeaveOneSubjectOut:
    """
    Leave-one-subject-out cross-validation.
    
    Uses each subject as a test set once, providing the most
    rigorous evaluation for generalization.
    """
    
    def __init__(self):
        pass
    
    def split(
        self,
        subject_ids: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate LOSO splits.
        
        Args:
            subject_ids: Subject ID for each sample
            labels: Unused, for API compatibility
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        unique_subjects = np.unique(subject_ids)
        
        for subj in unique_subjects:
            test_mask = subject_ids == subj
            train_indices = np.where(~test_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            yield train_indices, test_indices
