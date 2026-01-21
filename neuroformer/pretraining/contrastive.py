"""
Contrastive learning for EEG self-supervised pretraining.

Implements SimCLR-style contrastive learning adapted for EEG data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    Used in SimCLR for contrastive learning. Pulls together
    representations of augmented views of the same sample while
    pushing apart representations of different samples.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature scaling parameter (lower = sharper)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two views.
        
        Args:
            z_i: First view representations (batch, d)
            z_j: Second view representations (batch, d)
            
        Returns:
            Scalar loss value
        """
        batch_size = z_i.size(0)
        device = z_i.device
        
        # Normalize representations
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)  # (2*batch, d)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # (2*batch, 2*batch)
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ], dim=0).to(device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Cross entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class SimCLREEGAugmentor(nn.Module):
    """
    EEG-specific augmentation module for contrastive learning.
    
    Generates two different augmented views of the same EEG sample.
    """
    
    def __init__(
        self,
        time_shift_max: int = 50,
        noise_std: float = 0.1,
        channel_dropout_prob: float = 0.1,
        scaling_range: Tuple[float, float] = (0.8, 1.2)
    ):
        """
        Args:
            time_shift_max: Maximum samples to shift in time
            noise_std: Gaussian noise standard deviation (relative)
            channel_dropout_prob: Probability of dropping each channel
            scaling_range: Range for random amplitude scaling
        """
        super().__init__()
        self.time_shift_max = time_shift_max
        self.noise_std = noise_std
        self.channel_dropout_prob = channel_dropout_prob
        self.scaling_range = scaling_range
    
    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to EEG data.
        
        Args:
            x: EEG data (batch, channels, samples) or (batch, bands, electrodes)
            
        Returns:
            Augmented data with same shape
        """
        batch_size = x.size(0)
        device = x.device
        
        x_aug = x.clone()
        
        # Random time shift (if temporal dimension exists)
        if x.dim() == 3 and x.size(-1) > self.time_shift_max * 2:
            shifts = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (batch_size,))
            for i, shift in enumerate(shifts):
                x_aug[i] = torch.roll(x_aug[i], shift.item(), dims=-1)
        
        # Add Gaussian noise
        noise = torch.randn_like(x_aug) * self.noise_std
        x_aug = x_aug + noise * x_aug.std(dim=-1, keepdim=True)
        
        # Channel dropout
        if x.dim() >= 2:
            mask = torch.rand(batch_size, x.size(1), device=device) > self.channel_dropout_prob
            mask = mask.unsqueeze(-1).expand_as(x_aug) if x.dim() == 3 else mask
            x_aug = x_aug * mask.float()
        
        # Random scaling
        scale = torch.empty(batch_size, 1, 1 if x.dim() == 3 else 1, device=device).uniform_(
            self.scaling_range[0], self.scaling_range[1]
        )
        x_aug = x_aug * scale
        
        return x_aug
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views.
        
        Args:
            x: Input EEG data
            
        Returns:
            Tuple of (view1, view2)
        """
        view1 = self.augment(x)
        view2 = self.augment(x)
        return view1, view2


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Projects representations to a lower-dimensional space
    where contrastive loss is computed.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ContrastivePretrainer(nn.Module):
    """
    Full contrastive pretraining pipeline.
    
    Combines encoder, augmentor, and projection head for
    self-supervised pretraining of NeuroFormer.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        d_model: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        """
        Args:
            encoder: NeuroFormer encoder backbone
            d_model: Encoder output dimension
            projection_dim: Contrastive projection dimension
            temperature: Contrastive loss temperature
        """
        super().__init__()
        self.encoder = encoder
        self.augmentor = SimCLREEGAugmentor()
        self.projection_head = ProjectionHead(d_model, d_model, projection_dim)
        self.criterion = ContrastiveLoss(temperature)
    
    def forward(
        self,
        x: torch.Tensor,
        coherence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive pretraining.
        
        Args:
            x: Input EEG band powers (batch, bands, electrodes)
            coherence: Optional coherence matrices
            
        Returns:
            Tuple of (loss, projections for view1)
        """
        # Generate two views
        view1, view2 = self.augmentor(x)
        
        # Encode both views
        out1 = self.encoder(view1, coherence_matrices=coherence, return_features=True)
        out2 = self.encoder(view2, coherence_matrices=coherence, return_features=True)
        
        # Get features
        z1 = out1['features']
        z2 = out2['features']
        
        # Project
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)
        
        # Compute loss
        loss = self.criterion(p1, p2)
        
        return loss, p1
    
    def pretrain_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Single pretraining step.
        
        Args:
            batch: EEG data batch
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        optimizer.zero_grad()
        loss, _ = self.forward(batch)
        loss.backward()
        optimizer.step()
        return loss.item()
