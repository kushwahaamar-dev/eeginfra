"""
Prediction engine for NeuroFormer.

Provides optimized inference with batching and optional quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Dict, List, Optional, Tuple
from pathlib import Path


class Predictor:
    """
    High-performance prediction engine for NeuroFormer.
    
    Features:
    - Automatic batching for large datasets
    - GPU acceleration
    - Optional FP16 inference
    - Probability calibration
    """
    
    # Class names for psychiatric disorders
    CLASS_NAMES = [
        "Healthy",
        "Addictive Disorder",
        "Anxiety Disorder",
        "Mood Disorder",
        "Obsessive Compulsive Disorder",
        "Schizophrenia",
        "Trauma and Stress Related Disorder"
    ]
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        precision: str = 'fp32',
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: Trained NeuroFormer model
            device: 'auto', 'cuda', or 'cpu'
            precision: 'fp32' or 'fp16'
            class_names: Custom class names (optional)
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.precision = precision
        self.class_names = class_names or self.CLASS_NAMES
        
        self.model = model.to(device)
        self.model.eval()
        
        if precision == 'fp16' and device == 'cuda':
            self.model = self.model.half()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_class: type,
        model_kwargs: Optional[Dict] = None,
        device: str = 'auto',
        precision: str = 'fp32'
    ) -> 'Predictor':
        """
        Load predictor from checkpoint file.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            model_class: NeuroFormer model class
            model_kwargs: Arguments for model initialization
            device: Device to load model on
            precision: Inference precision
            
        Returns:
            Predictor instance
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model_kwargs = model_kwargs or {}
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, device, precision)
    
    def _prepare_input(
        self,
        data: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Convert input to proper tensor format."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        data = data.to(self.device)
        
        if self.precision == 'fp16':
            data = data.half()
        else:
            data = data.float()
        
        # Ensure batch dimension
        if data.dim() == 2:
            data = data.unsqueeze(0)
        
        return data
    
    @torch.no_grad()
    def predict(
        self,
        band_powers: Union[np.ndarray, torch.Tensor],
        coherence: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_proba: bool = False
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Make predictions on EEG data.
        
        Args:
            band_powers: Band power features (batch, bands, electrodes) or (bands, electrodes)
            coherence: Optional coherence matrices
            return_proba: Whether to return class probabilities
            
        Returns:
            Dictionary with 'class_indices', 'class_names', and optionally 'probabilities'
        """
        band_powers = self._prepare_input(band_powers)
        
        if coherence is not None:
            coherence = self._prepare_input(coherence)
        
        # Forward pass
        outputs = self.model(band_powers, coherence_matrices=coherence)
        logits = outputs['logits']
        
        # Get predictions
        probabilities = F.softmax(logits, dim=-1)
        class_indices = probabilities.argmax(dim=-1)
        
        # Convert to numpy
        class_indices = class_indices.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        result = {
            'class_indices': class_indices,
            'class_names': [self.class_names[i] for i in class_indices],
            'confidence': probabilities.max(axis=-1)
        }
        
        if return_proba:
            result['probabilities'] = probabilities
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_proba: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions on a full dataset.
        
        Args:
            dataloader: DataLoader with EEG data
            return_proba: Whether to return probabilities
            
        Returns:
            Dictionary with aggregated predictions
        """
        all_indices = []
        all_probas = []
        all_targets = []
        
        for batch in dataloader:
            if len(batch) == 2:
                inputs, targets = batch
                coherence = None
            else:
                inputs, coherence, targets = batch
            
            inputs = self._prepare_input(inputs)
            if coherence is not None:
                coherence = self._prepare_input(coherence)
            
            outputs = self.model(inputs, coherence_matrices=coherence)
            logits = outputs['logits']
            
            probas = F.softmax(logits, dim=-1).cpu().numpy()
            indices = probas.argmax(axis=-1)
            
            all_indices.append(indices)
            all_probas.append(probas)
            all_targets.append(targets.numpy())
        
        result = {
            'class_indices': np.concatenate(all_indices),
            'targets': np.concatenate(all_targets),
        }
        
        if return_proba:
            result['probabilities'] = np.concatenate(all_probas, axis=0)
        
        return result
    
    def get_top_k_predictions(
        self,
        band_powers: Union[np.ndarray, torch.Tensor],
        k: int = 3,
        coherence: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> List[List[Tuple[str, float]]]:
        """
        Get top-k predictions with confidence scores.
        
        Args:
            band_powers: Input features
            k: Number of top predictions to return
            coherence: Optional coherence matrices
            
        Returns:
            List of [(class_name, probability), ...] for each sample
        """
        result = self.predict(band_powers, coherence, return_proba=True)
        probabilities = result['probabilities']
        
        top_k_results = []
        for probs in probabilities:
            top_indices = np.argsort(probs)[::-1][:k]
            top_k = [
                (self.class_names[i], float(probs[i]))
                for i in top_indices
            ]
            top_k_results.append(top_k)
        
        return top_k_results


class RealTimePredictor(Predictor):
    """
    Real-time predictor with streaming support.
    
    Optimized for low-latency inference on streaming EEG data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        precision: str = 'fp32',
        buffer_size: int = 10
    ):
        super().__init__(model, device, precision)
        self.buffer_size = buffer_size
        self.prediction_buffer = []
    
    def predict_streaming(
        self,
        band_powers: Union[np.ndarray, torch.Tensor],
        coherence: Optional[Union[np.ndarray, torch.Tensor]] = None,
        smoothing: bool = True
    ) -> Dict:
        """
        Make prediction on streaming data with optional smoothing.
        
        Args:
            band_powers: Current frame features
            coherence: Optional coherence
            smoothing: Whether to smooth predictions over buffer
            
        Returns:
            Prediction result
        """
        result = self.predict(band_powers, coherence, return_proba=True)
        
        if smoothing:
            self.prediction_buffer.append(result['probabilities'][0])
            if len(self.prediction_buffer) > self.buffer_size:
                self.prediction_buffer.pop(0)
            
            # Average probabilities
            avg_probs = np.mean(self.prediction_buffer, axis=0)
            smoothed_idx = avg_probs.argmax()
            
            result = {
                'class_indices': np.array([smoothed_idx]),
                'class_names': [self.class_names[smoothed_idx]],
                'confidence': np.array([avg_probs.max()]),
                'probabilities': avg_probs.reshape(1, -1),
                'raw_probabilities': result['probabilities']
            }
        
        return result
    
    def reset_buffer(self):
        """Clear the prediction buffer."""
        self.prediction_buffer = []
