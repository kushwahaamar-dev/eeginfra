"""
Hyperparameter tuning utilities for NeuroFormer.

Provides grid search, random search, and Bayesian optimization
for finding optimal model configurations.
"""

import copy
import json
import itertools
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

import numpy as np

from neuroformer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrialResult:
    """Result from a single hyperparameter trial."""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    duration_seconds: float
    trial_id: int
    
    @property
    def score(self) -> float:
        return self.metrics.get('val_accuracy', self.metrics.get('val_loss', 0))


@dataclass
class SearchSpace:
    """Define hyperparameter search space."""
    
    # Model architecture
    d_model: List[int] = field(default_factory=lambda: [128, 256, 512])
    n_heads: List[int] = field(default_factory=lambda: [4, 8])
    n_transformer_layers: List[int] = field(default_factory=lambda: [2, 4, 6])
    n_gnn_layers: List[int] = field(default_factory=lambda: [2, 3, 4])
    dropout: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    
    # Training
    learning_rate: List[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64])
    weight_decay: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    warmup_epochs: List[int] = field(default_factory=lambda: [5, 10])
    
    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary."""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_transformer_layers': self.n_transformer_layers,
            'n_gnn_layers': self.n_gnn_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
        }
    
    def total_combinations(self) -> int:
        """Total number of hyperparameter combinations."""
        d = self.to_dict()
        total = 1
        for vals in d.values():
            total *= len(vals)
        return total


class GridSearch:
    """
    Exhaustive grid search over hyperparameter space.
    """
    
    def __init__(
        self,
        search_space: Dict[str, List],
        train_fn: Callable,
        monitor: str = 'val_accuracy',
        mode: str = 'max',
        save_dir: Optional[str] = None
    ):
        """
        Args:
            search_space: Dict mapping param names to lists of values
            train_fn: Function(params) -> metrics dict
            monitor: Metric to optimize
            mode: 'max' or 'min'
            save_dir: Directory to save results
        """
        self.search_space = search_space
        self.train_fn = train_fn
        self.monitor = monitor
        self.mode = mode
        self.save_dir = Path(save_dir) if save_dir else None
        
        self.results: List[TrialResult] = []
        self.best_result: Optional[TrialResult] = None
    
    def run(self, max_trials: Optional[int] = None) -> TrialResult:
        """
        Run grid search.
        
        Args:
            max_trials: Maximum number of trials (None = all combinations)
            
        Returns:
            Best trial result
        """
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        combinations = list(itertools.product(*values))
        
        if max_trials:
            combinations = combinations[:max_trials]
        
        total = len(combinations)
        logger.info(f"Grid Search: {total} combinations")
        
        for trial_id, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            # Skip invalid configurations
            if not self._validate_params(params):
                continue
            
            logger.info(f"Trial {trial_id+1}/{total}: {params}")
            
            start = time.time()
            try:
                metrics = self.train_fn(params)
            except Exception as e:
                logger.warning(f"Trial {trial_id+1} failed: {e}")
                continue
            duration = time.time() - start
            
            result = TrialResult(
                params=params,
                metrics=metrics,
                duration_seconds=duration,
                trial_id=trial_id
            )
            self.results.append(result)
            
            # Update best
            score = metrics.get(self.monitor, 0)
            if self.best_result is None or (
                (self.mode == 'max' and score > self.best_result.metrics.get(self.monitor, 0)) or
                (self.mode == 'min' and score < self.best_result.metrics.get(self.monitor, float('inf')))
            ):
                self.best_result = result
                logger.info(f"New best! {self.monitor}={score:.4f}")
        
        if self.save_dir:
            self._save_results()
        
        return self.best_result
    
    def _validate_params(self, params: Dict) -> bool:
        """Check if parameter combination is valid."""
        d_model = params.get('d_model', 256)
        n_heads = params.get('n_heads', 8)
        
        if d_model % n_heads != 0:
            return False
        
        return True
    
    def _save_results(self):
        """Save results to file."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        results_data = []
        for r in self.results:
            results_data.append({
                'trial_id': r.trial_id,
                'params': r.params,
                'metrics': r.metrics,
                'duration': r.duration_seconds
            })
        
        with open(self.save_dir / 'grid_search_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)


class RandomSearch:
    """
    Random search over hyperparameter space.
    
    More efficient than grid search for high-dimensional spaces.
    """
    
    def __init__(
        self,
        search_space: Dict[str, List],
        train_fn: Callable,
        n_trials: int = 20,
        monitor: str = 'val_accuracy',
        mode: str = 'max',
        seed: int = 42,
        save_dir: Optional[str] = None
    ):
        self.search_space = search_space
        self.train_fn = train_fn
        self.n_trials = n_trials
        self.monitor = monitor
        self.mode = mode
        self.rng = random.Random(seed)
        self.save_dir = Path(save_dir) if save_dir else None
        
        self.results: List[TrialResult] = []
        self.best_result: Optional[TrialResult] = None
    
    def _sample_params(self) -> Dict[str, Any]:
        """Sample random parameter combination."""
        params = {}
        for key, values in self.search_space.items():
            params[key] = self.rng.choice(values)
        return params
    
    def run(self) -> TrialResult:
        """Run random search."""
        logger.info(f"Random Search: {self.n_trials} trials")
        
        for trial_id in range(self.n_trials):
            # Sample until valid
            for _ in range(100):
                params = self._sample_params()
                d_model = params.get('d_model', 256)
                n_heads = params.get('n_heads', 8)
                if d_model % n_heads == 0:
                    break
            
            logger.info(f"Trial {trial_id+1}/{self.n_trials}: {params}")
            
            start = time.time()
            try:
                metrics = self.train_fn(params)
            except Exception as e:
                logger.warning(f"Trial {trial_id+1} failed: {e}")
                continue
            duration = time.time() - start
            
            result = TrialResult(
                params=params,
                metrics=metrics,
                duration_seconds=duration,
                trial_id=trial_id
            )
            self.results.append(result)
            
            score = metrics.get(self.monitor, 0)
            if self.best_result is None or (
                (self.mode == 'max' and score > self.best_result.metrics.get(self.monitor, 0)) or
                (self.mode == 'min' and score < self.best_result.metrics.get(self.monitor, float('inf')))
            ):
                self.best_result = result
                logger.info(f"New best! {self.monitor}={score:.4f}")
        
        if self.save_dir:
            self._save_results()
        
        return self.best_result
    
    def _save_results(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        results_data = [{
            'trial_id': r.trial_id,
            'params': r.params,
            'metrics': r.metrics,
            'duration': r.duration_seconds
        } for r in self.results]
        
        with open(self.save_dir / 'random_search_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)


class BayesianOptimizer:
    """
    Simple Bayesian-inspired optimization using surrogate scoring.
    
    Uses historical results to prioritize promising regions of
    the hyperparameter space.
    """
    
    def __init__(
        self,
        search_space: Dict[str, List],
        train_fn: Callable,
        n_trials: int = 30,
        n_initial: int = 5,
        monitor: str = 'val_accuracy',
        mode: str = 'max',
        exploration_rate: float = 0.3,
        seed: int = 42,
        save_dir: Optional[str] = None
    ):
        self.search_space = search_space
        self.train_fn = train_fn
        self.n_trials = n_trials
        self.n_initial = n_initial
        self.monitor = monitor
        self.mode = mode
        self.exploration_rate = exploration_rate
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.save_dir = Path(save_dir) if save_dir else None
        
        self.results: List[TrialResult] = []
        self.best_result: Optional[TrialResult] = None
        self._param_scores: Dict[str, Dict[Any, List[float]]] = {
            key: {v: [] for v in values}
            for key, values in search_space.items()
        }
    
    def _sample_initial(self) -> Dict[str, Any]:
        """Random sampling for initial exploration."""
        params = {}
        for key, values in self.search_space.items():
            params[key] = self.rng.choice(values)
        return params
    
    def _sample_informed(self) -> Dict[str, Any]:
        """Sample using learned parameter scores."""
        params = {}
        
        for key, values in self.search_space.items():
            if self.rng.random() < self.exploration_rate:
                # Explore
                params[key] = self.rng.choice(values)
            else:
                # Exploit: choose value with best average score
                scores = self._param_scores[key]
                avg_scores = {}
                for v in values:
                    if scores[v]:
                        avg_scores[v] = np.mean(scores[v])
                    else:
                        avg_scores[v] = float('inf') if self.mode == 'max' else float('-inf')
                
                if self.mode == 'max':
                    best_val = max(avg_scores, key=avg_scores.get)
                else:
                    best_val = min(avg_scores, key=avg_scores.get)
                
                params[key] = best_val
        
        return params
    
    def _update_scores(self, params: Dict, score: float):
        """Update parameter performance tracking."""
        for key, value in params.items():
            if key in self._param_scores:
                self._param_scores[key][value].append(score)
    
    def run(self) -> TrialResult:
        """Run Bayesian optimization."""
        logger.info(f"Bayesian Optimization: {self.n_trials} trials "
                     f"({self.n_initial} initial)")
        
        for trial_id in range(self.n_trials):
            # Sample params
            if trial_id < self.n_initial:
                params = self._sample_initial()
            else:
                params = self._sample_informed()
            
            # Validate d_model / n_heads compatibility
            d_model = params.get('d_model', 256)
            n_heads = params.get('n_heads', 8)
            if d_model % n_heads != 0:
                # Fix by choosing compatible n_heads
                valid_heads = [h for h in self.search_space.get('n_heads', [8]) if d_model % h == 0]
                if valid_heads:
                    params['n_heads'] = self.rng.choice(valid_heads)
                else:
                    continue
            
            logger.info(f"Trial {trial_id+1}/{self.n_trials}: {params}")
            
            start = time.time()
            try:
                metrics = self.train_fn(params)
            except Exception as e:
                logger.warning(f"Trial {trial_id+1} failed: {e}")
                continue
            duration = time.time() - start
            
            score = metrics.get(self.monitor, 0)
            self._update_scores(params, score)
            
            result = TrialResult(
                params=params,
                metrics=metrics,
                duration_seconds=duration,
                trial_id=trial_id
            )
            self.results.append(result)
            
            if self.best_result is None or (
                (self.mode == 'max' and score > self.best_result.metrics.get(self.monitor, 0)) or
                (self.mode == 'min' and score < self.best_result.metrics.get(self.monitor, float('inf')))
            ):
                self.best_result = result
                logger.info(f"New best! {self.monitor}={score:.4f}")
        
        if self.save_dir:
            self._save_results()
        
        return self.best_result
    
    def _save_results(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        results_data = [{
            'trial_id': r.trial_id,
            'params': r.params,
            'metrics': r.metrics,
            'duration': r.duration_seconds
        } for r in self.results]
        
        with open(self.save_dir / 'bayesian_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def get_param_importance(self) -> Dict[str, float]:
        """
        Estimate parameter importance based on score variance.
        
        Returns:
            Dict mapping parameter names to importance scores
        """
        importance = {}
        
        for key in self.search_space:
            scores_per_value = self._param_scores[key]
            value_means = []
            
            for v, scores in scores_per_value.items():
                if scores:
                    value_means.append(np.mean(scores))
            
            if len(value_means) >= 2:
                importance[key] = float(np.std(value_means))
            else:
                importance[key] = 0.0
        
        # Normalize
        total = sum(importance.values()) or 1
        importance = {k: v / total for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def quick_tune(
    train_fn: Callable,
    search_space: Optional[Dict[str, List]] = None,
    method: str = 'random',
    n_trials: int = 20,
    monitor: str = 'val_accuracy',
    save_dir: Optional[str] = None
) -> TrialResult:
    """
    Quick hyperparameter tuning wrapper.
    
    Args:
        train_fn: Function(params) -> metrics dict
        search_space: Parameter search space
        method: 'grid', 'random', or 'bayesian'
        n_trials: Number of trials
        monitor: Metric to optimize
        save_dir: Results save directory
        
    Returns:
        Best trial result
    """
    if search_space is None:
        search_space = SearchSpace().to_dict()
    
    if method == 'grid':
        searcher = GridSearch(search_space, train_fn, monitor=monitor, save_dir=save_dir)
        return searcher.run(max_trials=n_trials)
    elif method == 'random':
        searcher = RandomSearch(search_space, train_fn, n_trials=n_trials, monitor=monitor, save_dir=save_dir)
        return searcher.run()
    elif method == 'bayesian':
        searcher = BayesianOptimizer(search_space, train_fn, n_trials=n_trials, monitor=monitor, save_dir=save_dir)
        return searcher.run()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'grid', 'random', or 'bayesian'")
