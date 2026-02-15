"""
PyTorch Dataset for EEG data.

Supports loading preprocessed EEG features for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG psychiatric disorder classification.
    
    Loads preprocessed band power and coherence features.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        coherence: Optional[np.ndarray] = None,
        subject_ids: Optional[np.ndarray] = None,
        transform: Optional[callable] = None
    ):
        """
        Args:
            features: Band power features (n_samples, n_bands, n_electrodes)
            labels: Class labels (n_samples,)
            coherence: Optional coherence matrices (n_samples, n_bands, n_elec, n_elec)
            subject_ids: Optional subject identifiers for subject-aware splits
            transform: Optional data augmentation transform
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        if coherence is not None:
            self.coherence = torch.tensor(coherence, dtype=torch.float32)
        else:
            self.coherence = None
        
        self.subject_ids = subject_ids
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            features = self.transform(features)
        
        if self.coherence is not None:
            return features, self.coherence[idx], label
        
        return features, label
    
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        feature_cols: Optional[List[str]] = None,
        label_col: str = 'label',
        num_electrodes: int = 19,
        num_bands: int = 5
    ) -> 'EEGDataset':
        """
        Load dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            feature_cols: Columns to use as features (auto-detected if None)
            label_col: Column containing labels
            num_electrodes: Number of EEG electrodes
            num_bands: Number of frequency bands
            
        Returns:
            EEGDataset instance
        """
        df = pd.read_csv(csv_path)
        
        if feature_cols is None:
            # Auto-detect: all columns except label
            feature_cols = [c for c in df.columns if c != label_col and df[c].dtype in [np.float64, np.float32, np.int64]]
        
        features = df[feature_cols].values
        labels = df[label_col].values
        
        # Reshape features to (n_samples, n_bands, n_electrodes)
        n_samples = len(labels)
        expected_features = num_bands * num_electrodes
        
        if features.shape[1] >= expected_features:
            features = features[:, :expected_features].reshape(n_samples, num_bands, num_electrodes)
        else:
            # Pad if needed
            padded = np.zeros((n_samples, expected_features))
            padded[:, :features.shape[1]] = features
            features = padded.reshape(n_samples, num_bands, num_electrodes)
        
        return cls(features, labels)


class EEGDataModule:
    """
    Data module for managing train/val/test splits.
    
    Provides subject-aware splitting to prevent data leakage.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        coherence: Optional[np.ndarray] = None,
        subject_ids: Optional[np.ndarray] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4,
        random_state: int = 42
    ):
        """
        Args:
            features: All features
            labels: All labels
            coherence: Optional coherence matrices
            subject_ids: Subject identifiers for subject-aware splits
            train_ratio: Training set ratio
            val_ratio: Validation set ratio  
            test_ratio: Test set ratio
            batch_size: Batch size for DataLoaders
            num_workers: Number of data loading workers
            random_state: Random seed for splitting
        """
        self.features = features
        self.labels = labels
        self.coherence = coherence
        self.subject_ids = subject_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Perform split
        if subject_ids is not None:
            train_idx, val_idx, test_idx = self._subject_aware_split(
                subject_ids, train_ratio, val_ratio, random_state
            )
        else:
            train_idx, val_idx, test_idx = self._random_split(
                len(labels), train_ratio, val_ratio, random_state
            )
        
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
    
    def _subject_aware_split(
        self,
        subject_ids: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split by subject to avoid data leakage."""
        rng = np.random.default_rng(random_state)
        
        unique_subjects = np.unique(subject_ids)
        rng.shuffle(unique_subjects)
        
        n_subjects = len(unique_subjects)
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)
        
        train_subjects = set(unique_subjects[:n_train])
        val_subjects = set(unique_subjects[n_train:n_train + n_val])
        test_subjects = set(unique_subjects[n_train + n_val:])
        
        train_idx = np.where(np.isin(subject_ids, list(train_subjects)))[0]
        val_idx = np.where(np.isin(subject_ids, list(val_subjects)))[0]
        test_idx = np.where(np.isin(subject_ids, list(test_subjects)))[0]
        
        return train_idx, val_idx, test_idx
    
    def _random_split(
        self,
        n_samples: int,
        train_ratio: float,
        val_ratio: float,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random stratified split."""
        rng = np.random.default_rng(random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]
    
    def _get_subset(self, indices: np.ndarray) -> EEGDataset:
        """Create dataset subset from indices."""
        features = self.features[indices]
        labels = self.labels[indices]
        coherence = self.coherence[indices] if self.coherence is not None else None
        
        return EEGDataset(features, labels, coherence)
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        dataset = self._get_subset(self.train_idx)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        dataset = self._get_subset(self.val_idx)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        dataset = self._get_subset(self.test_idx)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced data."""
        train_labels = self.labels[self.train_idx]
        unique, counts = np.unique(train_labels, return_counts=True)
        
        # Inverse frequency weighting
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(unique)
        
        full_weights = np.ones(max(unique) + 1)
        full_weights[unique] = weights
        
        return torch.tensor(full_weights, dtype=torch.float32)
