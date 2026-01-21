"""
Explainability utilities for NeuroFormer.

Provides attention visualization and feature importance analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Union


class AttentionVisualizer:
    """
    Visualize attention weights from NeuroFormer.
    
    Helps understand which electrodes and time points
    the model focuses on for predictions.
    """
    
    # 10-20 electrode positions for visualization
    ELECTRODE_POSITIONS = {
        "Fp1": (0.31, 0.93), "Fp2": (0.69, 0.93),
        "F7": (0.15, 0.75), "F3": (0.35, 0.75),
        "Fz": (0.5, 0.75), "F4": (0.65, 0.75), "F8": (0.85, 0.75),
        "T3": (0.05, 0.5), "C3": (0.3, 0.5),
        "Cz": (0.5, 0.5), "C4": (0.7, 0.5), "T4": (0.95, 0.5),
        "T5": (0.15, 0.25), "P3": (0.35, 0.25),
        "Pz": (0.5, 0.25), "P4": (0.65, 0.25), "T6": (0.85, 0.25),
        "O1": (0.35, 0.07), "O2": (0.65, 0.07),
    }
    
    ELECTRODE_NAMES = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T3", "C3", "Cz", "C4", "T4",
        "T5", "P3", "Pz", "P4", "T6",
        "O1", "O2"
    ]
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Args:
            model: NeuroFormer model
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def get_attention_weights(
        self,
        band_powers: torch.Tensor,
        coherence: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from forward pass.
        
        Args:
            band_powers: Input features (batch, bands, electrodes)
            coherence: Optional coherence matrices
            
        Returns:
            Dictionary with attention weight tensors
        """
        band_powers = band_powers.to(self.device)
        if coherence is not None:
            coherence = coherence.to(self.device)
        
        outputs = self.model(band_powers, coherence_matrices=coherence, return_features=True)
        
        attention_weights = {}
        if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
            attention_weights['transformer'] = outputs['attention_weights']
        
        return attention_weights
    
    def plot_electrode_importance(
        self,
        importance_scores: np.ndarray,
        title: str = "Electrode Importance",
        cmap: str = 'RdYlBu_r',
        figsize: Tuple[int, int] = (8, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot electrode importance as a topomap.
        
        Args:
            importance_scores: Importance score per electrode
            title: Plot title
            cmap: Colormap
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Draw head outline
        head_circle = plt.Circle((0.5, 0.5), 0.45, fill=False, linewidth=2, color='black')
        ax.add_patch(head_circle)
        
        # Draw nose indicator
        ax.plot([0.5, 0.5], [0.95, 1.0], 'k-', linewidth=2)
        
        # Normalize scores
        scores = np.array(importance_scores)
        if scores.max() > scores.min():
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores_norm = np.ones_like(scores) * 0.5
        
        # Plot electrodes
        positions = []
        for i, name in enumerate(self.ELECTRODE_NAMES[:len(importance_scores)]):
            if name in self.ELECTRODE_POSITIONS:
                x, y = self.ELECTRODE_POSITIONS[name]
                positions.append((x, y))
                
                color = plt.cm.get_cmap(cmap)(scores_norm[i])
                circle = plt.Circle((x, y), 0.04, color=color, ec='black', linewidth=1)
                ax.add_patch(circle)
                
                ax.annotate(name, (x, y), ha='center', va='center', fontsize=8)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(scores.min(), scores.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Importance')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_band_importance(
        self,
        importance_per_band: Dict[str, float],
        title: str = "Frequency Band Importance",
        figsize: Tuple[int, int] = (10, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot importance of each frequency band.
        
        Args:
            importance_per_band: Dictionary mapping band names to importance
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        bands = list(importance_per_band.keys())
        values = list(importance_per_band.values())
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bands)))
        
        bars = ax.bar(bands, values, color=colors, edgecolor='black')
        
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Importance')
        ax.set_title(title)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def compute_feature_importance(
    model: nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    method: str = 'gradient',
    device: str = 'cpu'
) -> np.ndarray:
    """
    Compute feature importance scores.
    
    Args:
        model: Trained model
        data: Input features (batch, bands, electrodes)
        targets: True labels
        method: 'gradient' or 'permutation'
        device: Computation device
        
    Returns:
        Importance scores (bands, electrodes)
    """
    model = model.to(device)
    data = data.to(device)
    targets = targets.to(device)
    
    if method == 'gradient':
        return _gradient_importance(model, data, targets)
    elif method == 'permutation':
        return _permutation_importance(model, data, targets)
    else:
        raise ValueError(f"Unknown method: {method}")


def _gradient_importance(
    model: nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor
) -> np.ndarray:
    """Compute gradient-based feature importance."""
    model.eval()
    data.requires_grad_(True)
    
    outputs = model(data)
    logits = outputs['logits']
    
    # Get gradients w.r.t. correct class
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    
    # Absolute gradient as importance
    importance = data.grad.abs().mean(dim=0)  # (bands, electrodes)
    
    return importance.detach().cpu().numpy()


def _permutation_importance(
    model: nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    n_repeats: int = 10
) -> np.ndarray:
    """Compute permutation-based feature importance."""
    model.eval()
    
    with torch.no_grad():
        baseline_outputs = model(data)
        baseline_acc = (baseline_outputs['logits'].argmax(dim=1) == targets).float().mean().item()
    
    n_bands, n_electrodes = data.shape[1], data.shape[2]
    importance = np.zeros((n_bands, n_electrodes))
    
    for b in range(n_bands):
        for e in range(n_electrodes):
            drops = []
            for _ in range(n_repeats):
                data_perm = data.clone()
                # Shuffle this feature across batch
                perm_idx = torch.randperm(data.size(0))
                data_perm[:, b, e] = data[perm_idx, b, e]
                
                with torch.no_grad():
                    perm_outputs = model(data_perm)
                    perm_acc = (perm_outputs['logits'].argmax(dim=1) == targets).float().mean().item()
                
                drops.append(baseline_acc - perm_acc)
            
            importance[b, e] = np.mean(drops)
    
    return importance
