"""
Visualization tools for NeuroFormer.

Provides plotting utilities for EEG topomaps, training curves,
confusion matrices, and model interpretability.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Wedge
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path


# Standard 10-20 electrode positions (normalized to 0-1)
ELECTRODE_POSITIONS_2D = {
    "Fp1": (0.31, 0.93), "Fp2": (0.69, 0.93),
    "F7": (0.15, 0.75), "F3": (0.35, 0.75),
    "Fz": (0.5, 0.75), "F4": (0.65, 0.75), "F8": (0.85, 0.75),
    "T3": (0.05, 0.5), "C3": (0.3, 0.5),
    "Cz": (0.5, 0.5), "C4": (0.7, 0.5), "T4": (0.95, 0.5),
    "T5": (0.15, 0.25), "P3": (0.35, 0.25),
    "Pz": (0.5, 0.25), "P4": (0.65, 0.25), "T6": (0.85, 0.25),
    "O1": (0.35, 0.07), "O2": (0.65, 0.07),
}

ELECTRODE_NAMES = list(ELECTRODE_POSITIONS_2D.keys())


def plot_topomap(
    values: np.ndarray,
    electrode_names: Optional[List[str]] = None,
    title: str = "EEG Topomap",
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (8, 8),
    show_values: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot EEG values as a topographic map.
    
    Args:
        values: Values per electrode (n_electrodes,)
        electrode_names: Names of electrodes (defaults to 10-20 system)
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        show_values: Whether to show numeric values
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    electrode_names = electrode_names or ELECTRODE_NAMES[:len(values)]
    
    # Draw head outline
    head_circle = Circle((0.5, 0.5), 0.45, fill=False, linewidth=2, color='black')
    ax.add_patch(head_circle)
    
    # Draw nose
    nose_x = [0.5, 0.45, 0.5, 0.55, 0.5]
    nose_y = [0.95, 0.98, 1.02, 0.98, 0.95]
    ax.plot(nose_x, nose_y, 'k-', linewidth=2)
    
    # Draw ears
    left_ear = Wedge((0.02, 0.5), 0.06, 90, 270, fill=False, linewidth=2)
    right_ear = Wedge((0.98, 0.5), 0.06, 270, 90, fill=False, linewidth=2)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)
    
    # Normalize values for coloring
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        norm_values = np.ones_like(values) * 0.5
    else:
        norm_values = (values - vmin) / (vmax - vmin)
    
    colormap = plt.cm.get_cmap(cmap)
    
    # Plot electrodes
    for i, name in enumerate(electrode_names[:len(values)]):
        if name in ELECTRODE_POSITIONS_2D:
            x, y = ELECTRODE_POSITIONS_2D[name]
            color = colormap(norm_values[i])
            
            circle = Circle((x, y), 0.04, color=color, ec='black', linewidth=1.5)
            ax.add_patch(circle)
            
            # Label
            ax.annotate(name, (x, y), ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Show value
            if show_values:
                ax.annotate(f'{values[i]:.2f}', (x, y + 0.06), ha='center', fontsize=7)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', fontsize=10)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Accuracy plot
    ax2 = axes[1]
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train Acc', color='#2ca02c', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Val Acc', color='#d62728', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix with annotations.
    
    Args:
        confusion_matrix: Confusion matrix (n_classes, n_classes)
        class_names: Class names for labels
        normalize: Whether to normalize by true class
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = '.2f'
    else:
        cm = confusion_matrix
        fmt = 'd'
    
    n_classes = len(cm)
    class_names = class_names or [f'Class {i}' for i in range(n_classes)]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Proportion' if normalize else 'Count', rotation=-90, va="bottom")
    
    # Labels
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            text = f"{val:{fmt}}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_band_powers(
    band_powers: Dict[str, np.ndarray],
    electrode_names: Optional[List[str]] = None,
    title: str = "Band Power Distribution",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot band power heatmap across electrodes.
    
    Args:
        band_powers: Dict mapping band names to power arrays
        electrode_names: Electrode names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    bands = list(band_powers.keys())
    n_electrodes = len(list(band_powers.values())[0])
    electrode_names = electrode_names or ELECTRODE_NAMES[:n_electrodes]
    
    # Create matrix
    data = np.array([band_powers[b] for b in bands])
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    im = ax.imshow(data, aspect='auto', cmap='viridis')
    
    ax.set_xticks(np.arange(n_electrodes))
    ax.set_yticks(np.arange(len(bands)))
    ax.set_xticklabels(electrode_names, rotation=45, ha='right')
    ax.set_yticklabels([b.capitalize() for b in bands])
    
    ax.set_xlabel('Electrode', fontsize=11)
    ax.set_ylabel('Frequency Band', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Power (μV²)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot class distribution as a bar chart.
    
    Args:
        labels: Array of class labels
        class_names: Names for each class
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique]
    else:
        class_names = [class_names[i] for i in unique]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(unique)))
    bars = ax.bar(class_names, counts, color=colors, edgecolor='black', linewidth=1)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_attention_weights(
    attention: np.ndarray,
    electrode_names: Optional[List[str]] = None,
    title: str = "Attention Weights",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot attention matrix as heatmap.
    
    Args:
        attention: Attention weights (seq_len, seq_len) or (heads, seq_len, seq_len)
        electrode_names: Names for sequence positions
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    if attention.ndim == 3:
        # Average over heads
        attention = attention.mean(axis=0)
    
    seq_len = attention.shape[0]
    electrode_names = electrode_names or [f'{i}' for i in range(seq_len)]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    im = ax.imshow(attention, cmap='hot', aspect='auto')
    
    if seq_len <= 30:
        ax.set_xticks(np.arange(seq_len))
        ax.set_yticks(np.arange(seq_len))
        ax.set_xticklabels(electrode_names[:seq_len], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(electrode_names[:seq_len], fontsize=8)
    
    ax.set_xlabel('Key Position', fontsize=11)
    ax.set_ylabel('Query Position', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


class TrainingVisualizer:
    """
    Real-time training visualization.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update history with new metrics."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def plot(self, save: bool = True) -> plt.Figure:
        """Generate training curves."""
        fig = plot_training_curves(self.history)
        
        if save and self.save_dir:
            fig.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def save_history(self, path: Optional[str] = None):
        """Save history to JSON."""
        import json
        
        path = path or (self.save_dir / 'training_history.json' if self.save_dir else 'training_history.json')
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
