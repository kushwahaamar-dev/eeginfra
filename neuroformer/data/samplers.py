"""
Custom samplers for EEG data.

Provides balanced and subject-aware sampling strategies.
"""

import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List, Optional


class SubjectAwareSampler(Sampler):
    """
    Sampler that ensures samples from the same subject
    are grouped together within each batch.
    
    Useful for contrastive learning where we want to
    create positive pairs from the same subject.
    """
    
    def __init__(
        self,
        subject_ids: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Args:
            subject_ids: Subject identifier for each sample
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle subjects
            drop_last: Whether to drop last incomplete batch
        """
        self.subject_ids = subject_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group indices by subject
        self.subject_to_indices = {}
        for idx, sid in enumerate(subject_ids):
            if sid not in self.subject_to_indices:
                self.subject_to_indices[sid] = []
            self.subject_to_indices[sid].append(idx)
    
    def __iter__(self) -> Iterator[int]:
        subjects = list(self.subject_to_indices.keys())
        
        if self.shuffle:
            np.random.shuffle(subjects)
        
        indices = []
        for subject in subjects:
            subject_indices = self.subject_to_indices[subject].copy()
            if self.shuffle:
                np.random.shuffle(subject_indices)
            indices.extend(subject_indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        return len(self.subject_ids)


class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures each batch has balanced class representation.
    
    Useful for highly imbalanced datasets like psychiatric disorder
    classification where some disorders are rare.
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        num_batches: Optional[int] = None
    ):
        """
        Args:
            labels: Class labels for each sample
            batch_size: Total batch size (should be divisible by num_classes)
            num_batches: Number of batches per epoch (defaults to covering all samples)
        """
        self.labels = labels
        self.batch_size = batch_size
        
        # Group indices by class
        unique_classes = np.unique(labels)
        self.num_classes = len(unique_classes)
        self.samples_per_class = batch_size // self.num_classes
        
        self.class_to_indices = {}
        for c in unique_classes:
            self.class_to_indices[c] = np.where(labels == c)[0].tolist()
        
        if num_batches is None:
            # Enough batches to see all samples on average
            min_class_size = min(len(v) for v in self.class_to_indices.values())
            num_batches = max(1, len(labels) // batch_size)
        
        self.num_batches = num_batches
    
    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            batch = []
            for class_idx, indices in self.class_to_indices.items():
                # Sample with replacement if class is small
                replace = len(indices) < self.samples_per_class
                selected = np.random.choice(
                    indices, 
                    size=self.samples_per_class, 
                    replace=replace
                )
                batch.extend(selected.tolist())
            
            np.random.shuffle(batch)
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches


class StratifiedSampler(Sampler):
    """
    Stratified sampler maintaining class proportions.
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        shuffle: bool = True
    ):
        """
        Args:
            labels: Class labels
            shuffle: Whether to shuffle within each class
        """
        self.labels = labels
        self.shuffle = shuffle
        
        self.class_to_indices = {}
        for c in np.unique(labels):
            self.class_to_indices[c] = np.where(labels == c)[0].tolist()
    
    def __iter__(self) -> Iterator[int]:
        indices_per_class = []
        
        for c, indices in self.class_to_indices.items():
            class_indices = indices.copy()
            if self.shuffle:
                np.random.shuffle(class_indices)
            indices_per_class.append(class_indices)
        
        # Interleave classes
        max_len = max(len(idx) for idx in indices_per_class)
        all_indices = []
        
        for i in range(max_len):
            for class_indices in indices_per_class:
                if i < len(class_indices):
                    all_indices.append(class_indices[i])
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return len(self.labels)
