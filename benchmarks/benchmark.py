#!/usr/bin/env python
"""
Benchmark suite for NeuroFormer.

Measures inference speed, memory usage, and throughput.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple
import gc


def measure_inference_time(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 5, 19),
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure model inference time.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        n_warmup: Warmup iterations
        n_runs: Measurement iterations
        device: Device to run on
        
    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
    
    # Synchronize if CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    times = np.array(times)
    
    return {
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'throughput_samples_per_sec': float(1000 / times.mean() * input_shape[0])
    }


def measure_batch_throughput(
    model: torch.nn.Module,
    batch_sizes: List[int] = [1, 4, 8, 16, 32, 64],
    n_runs: int = 50,
    device: str = 'cpu'
) -> Dict[int, Dict[str, float]]:
    """
    Measure throughput at different batch sizes.
    
    Args:
        model: Model to benchmark
        batch_sizes: Batch sizes to test
        n_runs: Runs per batch size
        device: Device to run on
        
    Returns:
        Dictionary mapping batch size to timing stats
    """
    results = {}
    
    for batch_size in batch_sizes:
        try:
            stats = measure_inference_time(
                model,
                input_shape=(batch_size, 5, 19),
                n_runs=n_runs,
                device=device
            )
            results[batch_size] = stats
            print(f"Batch {batch_size}: {stats['mean_ms']:.2f}ms, "
                  f"{stats['throughput_samples_per_sec']:.1f} samples/sec")
        except RuntimeError as e:
            print(f"Batch {batch_size}: OOM or error - {e}")
            break
        
        # Clear cache
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    return results


def measure_memory_usage(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 5, 19),
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Measure GPU memory usage.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        device: Device (must be 'cuda')
        
    Returns:
        Memory statistics in MB
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    
    model = model.to(device)
    model.eval()
    
    baseline = torch.cuda.memory_allocated() / (1024 ** 2)
    
    dummy_input = torch.randn(input_shape, device=device)
    
    with torch.no_grad():
        _ = model(dummy_input)
        torch.cuda.synchronize()
    
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    current = torch.cuda.memory_allocated() / (1024 ** 2)
    
    return {
        'baseline_mb': baseline,
        'peak_mb': peak,
        'current_mb': current,
        'model_mb': current - baseline,
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: Model to analyze
        
    Returns:
        Parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'size_mb': total * 4 / (1024 ** 2)
    }


def benchmark_model(
    model: torch.nn.Module,
    device: str = 'auto',
    batch_sizes: List[int] = [1, 8, 32]
) -> Dict:
    """
    Run comprehensive benchmark.
    
    Args:
        model: Model to benchmark
        device: Device ('auto', 'cuda', 'cpu')
        batch_sizes: Batch sizes to test
        
    Returns:
        Complete benchmark results
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running benchmark on {device}")
    print("=" * 50)
    
    # Parameters
    params = count_parameters(model)
    print(f"\nParameters: {params['total']:,} ({params['size_mb']:.2f} MB)")
    
    # Inference time
    print(f"\nInference Time (batch=1):")
    single_sample = measure_inference_time(model, (1, 5, 19), device=device)
    print(f"  Mean: {single_sample['mean_ms']:.3f}ms")
    print(f"  P95:  {single_sample['p95_ms']:.3f}ms")
    print(f"  Throughput: {single_sample['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Batch throughput
    print(f"\nBatch Throughput:")
    batch_results = measure_batch_throughput(model, batch_sizes, device=device)
    
    # Memory (CUDA only)
    memory = {}
    if device == 'cuda':
        print(f"\nMemory Usage:")
        memory = measure_memory_usage(model, (32, 5, 19), device=device)
        print(f"  Model: {memory.get('model_mb', 0):.2f} MB")
        print(f"  Peak:  {memory.get('peak_mb', 0):.2f} MB")
    
    return {
        'device': device,
        'parameters': params,
        'single_sample': single_sample,
        'batch_throughput': batch_results,
        'memory': memory
    }


if __name__ == '__main__':
    from neuroformer.models import NeuroFormer
    
    print("NeuroFormer Benchmark")
    print("=" * 50)
    
    # Standard model
    model = NeuroFormer(
        num_electrodes=19,
        num_classes=7,
        d_model=256,
        n_heads=8,
        n_transformer_layers=4
    )
    
    results = benchmark_model(model, device='auto')
    print("\n✓ Benchmark complete!")
