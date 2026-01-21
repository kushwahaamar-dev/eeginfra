"""
Masked signal modeling for EEG self-supervised pretraining.

Implements BERT-style masked pretraining adapted for EEG signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class MaskedSignalModeling(nn.Module):
    """
    Masked Signal Modeling (MSM) objective.
    
    Randomly masks portions of the EEG signal and trains
    the model to reconstruct the masked values. Similar to
    BERT's masked language modeling.
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_length: int = 1,
        replace_with_noise: float = 0.1,
        replace_with_random: float = 0.1
    ):
        """
        Args:
            mask_ratio: Proportion of positions to mask
            mask_length: Length of each masked span
            replace_with_noise: Probability of replacing with noise instead of zero
            replace_with_random: Probability of replacing with random value
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_length = mask_length
        self.replace_with_noise = replace_with_noise
        self.replace_with_random = replace_with_random
    
    def create_mask(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create random mask for input tensor.
        
        Args:
            x: Input tensor (batch, seq_len, features) or (batch, channels, samples)
            
        Returns:
            Tuple of (masked_x, mask) where mask is True for masked positions
        """
        batch_size = x.size(0)
        seq_len = x.size(1) if x.dim() == 3 else x.numel() // batch_size
        device = x.device
        
        # Determine number of masks
        n_masks = int(seq_len * self.mask_ratio / self.mask_length)
        
        # Create mask
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            # Random starting positions
            starts = torch.randperm(max(1, seq_len - self.mask_length))[:n_masks]
            for start in starts:
                mask[i, start:start + self.mask_length] = True
        
        # Apply mask
        masked_x = x.clone()
        
        if x.dim() == 3:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
        else:
            mask_expanded = mask
        
        # 80% zero, 10% noise, 10% random
        rand_vals = torch.rand(batch_size, device=device)
        
        for i in range(batch_size):
            if rand_vals[i] < 0.8:
                # Zero out
                masked_x[i][mask[i]] = 0
            elif rand_vals[i] < 0.9:
                # Add noise
                noise = torch.randn_like(masked_x[i][mask[i]]) * x[i].std()
                masked_x[i][mask[i]] = noise
            # else: keep original (10%)
        
        return masked_x, mask
    
    def forward(
        self,
        x: torch.Tensor,
        predictions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss on masked positions.
        
        Args:
            x: Original input
            predictions: Model predictions
            mask: Boolean mask (True = masked)
            
        Returns:
            MSE loss on masked positions
        """
        if x.dim() == 3:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
        else:
            mask_expanded = mask
        
        # Only compute loss on masked positions
        masked_targets = x[mask_expanded]
        masked_preds = predictions[mask_expanded]
        
        if masked_targets.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        
        loss = F.mse_loss(masked_preds, masked_targets)
        return loss


class MaskedPretrainer(nn.Module):
    """
    Full masked pretraining pipeline.
    
    Combines encoder, decoder, and masking for self-supervised
    signal reconstruction pretraining.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        d_model: int = 256,
        num_electrodes: int = 19,
        num_bands: int = 5,
        mask_ratio: float = 0.15
    ):
        """
        Args:
            encoder: NeuroFormer encoder
            d_model: Model dimension
            num_electrodes: Number of EEG electrodes
            num_bands: Number of frequency bands
            mask_ratio: Proportion to mask
        """
        super().__init__()
        self.encoder = encoder
        self.masker = MaskedSignalModeling(mask_ratio)
        
        # Decoder to reconstruct masked values
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, num_electrodes * num_bands),
        )
        
        self.num_electrodes = num_electrodes
        self.num_bands = num_bands
    
    def forward(
        self,
        x: torch.Tensor,
        coherence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for masked pretraining.
        
        Args:
            x: Input EEG band powers (batch, bands, electrodes)
            coherence: Optional coherence matrices
            
        Returns:
            Tuple of (loss, predictions, mask)
        """
        batch_size = x.size(0)
        
        # Flatten for masking
        x_flat = x.view(batch_size, -1)  # (batch, bands * electrodes)
        
        # Create mask
        masked_x, mask = self.masker.create_mask(x_flat)
        
        # Reshape back
        masked_x = masked_x.view(batch_size, self.num_bands, self.num_electrodes)
        
        # Encode
        output = self.encoder(masked_x, coherence_matrices=coherence, return_features=True)
        features = output['features']
        
        # Decode
        predictions = self.decoder(features)  # (batch, bands * electrodes)
        
        # Compute loss
        loss = self.masker(x_flat, predictions, mask)
        
        # Reshape predictions
        predictions = predictions.view(batch_size, self.num_bands, self.num_electrodes)
        
        return loss, predictions, mask
    
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
        loss, _, _ = self.forward(batch)
        loss.backward()
        optimizer.step()
        return loss.item()


class CombinedPretrainer(nn.Module):
    """
    Combined contrastive + masked pretraining.
    
    Uses both objectives for richer representations.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        d_model: int = 256,
        num_electrodes: int = 19,
        num_bands: int = 5,
        contrastive_weight: float = 1.0,
        masked_weight: float = 0.5
    ):
        super().__init__()
        
        from neuroformer.pretraining.contrastive import ContrastivePretrainer
        
        self.contrastive = ContrastivePretrainer(encoder, d_model)
        self.masked = MaskedPretrainer(encoder, d_model, num_electrodes, num_bands)
        
        self.contrastive_weight = contrastive_weight
        self.masked_weight = masked_weight
    
    def forward(
        self,
        x: torch.Tensor,
        coherence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Combined forward pass.
        
        Returns:
            Tuple of (total_loss, dict with individual losses)
        """
        # Contrastive loss
        contrastive_loss, _ = self.contrastive(x, coherence)
        
        # Masked loss
        masked_loss, _, _ = self.masked(x, coherence)
        
        # Combined loss
        total_loss = (
            self.contrastive_weight * contrastive_loss +
            self.masked_weight * masked_loss
        )
        
        return total_loss, {
            'contrastive': contrastive_loss.item(),
            'masked': masked_loss.item(),
            'total': total_loss.item()
        }
