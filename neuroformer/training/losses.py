"""
Loss functions for EEG classification.

Includes Focal Loss and Label Smoothing for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Down-weights easy examples and focuses training on hard examples.
    Particularly useful for EEG psychiatric disorder classification
    where some disorders may be underrepresented.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights (num_classes,). If None, uniform weights.
            gamma: Focusing parameter. Higher = more focus on hard examples.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits (batch, num_classes)
            targets: Class indices (batch,)
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss.
    
    Softens hard labels to prevent overconfidence and improve
    generalization. Instead of [0, 0, 1, 0], uses [ε/K, ε/K, 1-ε, ε/K].
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0 = no smoothing)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Logits (batch, num_classes)
            targets: Class indices (batch,)
            
        Returns:
            Loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed labels
        smooth_labels = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # KL divergence
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple objectives.
    
    Supports combining classification loss with auxiliary losses
    like center loss for better feature clustering.
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        use_focal: bool = True,
        use_smoothing: bool = True,
        focal_gamma: float = 2.0,
        smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            num_classes: Number of classes
            use_focal: Whether to use focal loss
            use_smoothing: Whether to use label smoothing
            focal_gamma: Focal loss gamma
            smoothing: Label smoothing factor
            class_weights: Optional class weights for imbalance
        """
        super().__init__()
        
        self.use_focal = use_focal
        self.use_smoothing = use_smoothing
        
        if use_focal:
            self.focal = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        
        if use_smoothing:
            self.smoothing = LabelSmoothingLoss(num_classes, smoothing)
        
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            
        Returns:
            Combined loss value
        """
        losses = []
        
        if self.use_focal:
            losses.append(self.focal(logits, targets))
        elif not self.use_smoothing:
            losses.append(self.ce(logits, targets))
        
        if self.use_smoothing:
            losses.append(self.smoothing(logits, targets))
        
        if len(losses) == 0:
            return self.ce(logits, targets)
        
        return sum(losses) / len(losses)


class CenterLoss(nn.Module):
    """
    Center Loss for learning discriminative features.
    
    Minimizes intra-class variation by penalizing distance
    from class centers in feature space.
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        alpha: float = 0.5
    ):
        """
        Args:
            num_classes: Number of classes
            feature_dim: Dimension of feature vectors
            alpha: Learning rate for center updates
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        
        # Initialize centers
        self.centers = nn.Parameter(
            torch.randn(num_classes, feature_dim),
            requires_grad=False
        )
    
    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute center loss.
        
        Args:
            features: Feature vectors (batch, feature_dim)
            targets: Class labels (batch,)
            
        Returns:
            Center loss value
        """
        batch_size = features.size(0)
        
        # Get centers for each sample's class
        centers_batch = self.centers[targets]
        
        # Compute distances
        loss = (features - centers_batch).pow(2).sum() / (2 * batch_size)
        
        # Update centers (only during training)
        if self.training:
            diff = centers_batch - features
            
            # Count samples per class
            unique_labels, counts = torch.unique(targets, return_counts=True)
            
            for label, count in zip(unique_labels, counts):
                mask = targets == label
                center_diff = diff[mask].mean(dim=0)
                self.centers.data[label] -= self.alpha * center_diff
        
        return loss
