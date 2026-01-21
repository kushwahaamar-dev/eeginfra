"""
Trainer class for NeuroFormer.

Provides complete training loop with early stopping, learning rate
scheduling, checkpointing, and logging.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable, List
from tqdm import tqdm

from neuroformer.training.losses import CombinedLoss
from neuroformer.training.metrics import MetricTracker, compute_all_metrics


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation metric and stops training when it
    stops improving.
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            scale = self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


import numpy as np


class Trainer:
    """
    Full-featured trainer for NeuroFormer.
    
    Handles:
    - Training and validation loops
    - Early stopping
    - Learning rate scheduling
    - Checkpointing (best + periodic)
    - Logging and progress bars
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[object] = None,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: NeuroFormer model
            optimizer: Optimizer
            criterion: Loss function (defaults to CombinedLoss)
            scheduler: Learning rate scheduler
            device: 'cuda' or 'cpu'
            checkpoint_dir: Directory for saving checkpoints
            class_names: Names of classes for metrics
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion or CombinedLoss()
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.class_names = class_names
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.train_metrics = MetricTracker(class_names)
        self.val_metrics = MetricTracker(class_names)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.current_epoch = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch in pbar:
            # Unpack batch
            if len(batch) == 2:
                inputs, targets = batch
                coherence = None
            else:
                inputs, coherence, targets = batch
                coherence = coherence.to(self.device)
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs, coherence_matrices=coherence)
            logits = outputs['logits']
            
            # Compute loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track metrics
            self.train_metrics.update(logits, targets, loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return self.train_metrics.compute()
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        for batch in pbar:
            if len(batch) == 2:
                inputs, targets = batch
                coherence = None
            else:
                inputs, coherence, targets = batch
                coherence = coherence.to(self.device)
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs, coherence_matrices=coherence)
            logits = outputs['logits']
            
            loss = self.criterion(logits, targets)
            
            self.val_metrics.update(logits, targets, loss.item())
        
        return self.val_metrics.compute()
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        
        print(f"Training on {self.device}")
        print(f"Total epochs: {epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print("-" * 50)
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Log metrics
            train_loss = train_metrics.get('loss', 0)
            train_acc = train_metrics.get('accuracy', 0)
            val_loss = val_metrics.get('loss', 0)
            val_acc = val_metrics.get('accuracy', 0)
            val_balanced_acc = val_metrics.get('balanced_accuracy', 0)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                  f"Balanced: {val_balanced_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth', val_metrics)
                print(f"  ✓ New best model saved (acc: {val_acc:.4f})")
            
            # Early stopping check
            if early_stopping(val_acc):
                print(f"\n✓ Early stopping triggered at epoch {epoch}")
                break
            
            # Periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', val_metrics)
        
        print(f"\nTraining complete. Best validation accuracy: {self.best_val_acc:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics': metrics,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
