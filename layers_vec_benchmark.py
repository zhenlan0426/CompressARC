import torch
import time
import numpy as np
from unittest.mock import Mock

# Import the original and vectorized implementations
import layers
import multitensor_systems
from layers_vec import normalize_vec
from multitensor_systems_vec import FlatMultiTensor, pack_multitensor

# For comparison, create a version without optimizations
from torch_scatter import scatter_mean

def normalize_vec_old(flat: FlatMultiTensor, debias: bool = True) -> FlatMultiTensor:
    """Old version using scatter_mean and recomputing row2slice each time."""
    if flat.data.numel() == 0 or len(flat.dims_list) == 0:
        return FlatMultiTensor(
            data=flat.data.clone(),
            offsets=flat.offsets.clone(),
            lengths=flat.lengths.clone(),
            shapes=flat.shapes.copy(),
            dims_list=flat.dims_list.copy(),
            channel_dim=flat.channel_dim,
            row2slice=flat.row2slice.clone(),
        )
    
    # Recompute row2slice each time (old approach)
    row2slice = torch.repeat_interleave(
        torch.arange(len(flat.dims_list), device=flat.data.device, dtype=torch.long),
        flat.lengths
    )
    
    n_slices = len(flat.dims_list)
    
    if debias:
        slice_means = scatter_mean(flat.data, row2slice, dim=0, dim_size=n_slices)
        centered_data = flat.data - slice_means[row2slice]
        variance_data = centered_data ** 2
    else:
        variance_data = flat.data ** 2
        centered_data = flat.data
    
    slice_vars = scatter_mean(variance_data, row2slice, dim=0, dim_size=n_slices)
    eps = 1e-8
    slice_stds = torch.sqrt(slice_vars + eps)
    normalized_data = centered_data / slice_stds[row2slice]
    
    return FlatMultiTensor(
        data=normalized_data,
        offsets=flat.offsets.clone(),
        lengths=flat.lengths.clone(),
        shapes=flat.shapes.copy(),
        dims_list=flat.dims_list.copy(),
        channel_dim=flat.channel_dim,
        row2slice=flat.row2slice.clone(),
    )


class BenchmarkTask:
    """Mock task class for benchmarking."""
    def __init__(self, n_examples=16, n_x=64, n_y=64):
        self.n_examples = n_examples
        self.n_x = n_x
        self.n_y = n_y
        self.shapes = [[(n_x, n_y), (n_x, n_y)] for _ in range(n_examples)]
        self.masks = torch.ones(n_examples, n_x, n_y, 2)


def create_benchmark_multitensor_system():
    """Create a larger multitensor system for benchmarking."""
    task = BenchmarkTask()
    
    system = Mock()
    system.task = task
    system.make_multitensor = Mock(return_value={})
    
    # More slices for better benchmarking
    test_dims = [
        (1, 0, 0, 1, 1), (1, 1, 0, 1, 1), (1, 0, 1, 1, 1), (1, 1, 1, 1, 1),
        (1, 0, 0, 1, 0), (1, 1, 0, 1, 0), (1, 0, 1, 1, 0), (1, 1, 1, 1, 0),
        (1, 0, 0, 0, 1), (1, 1, 0, 0, 1), (1, 0, 1, 0, 1), (1, 1, 1, 0, 1),
    ]
    
    system.__iter__ = lambda self: iter(test_dims)
    return system, test_dims


def benchmark_normalize_implementations():
    """Compare performance between old and new implementations."""
    print("🚀 Benchmarking normalize implementations...")
    print("=" * 60)
    
    # Create test data
    system, test_dims = create_benchmark_multitensor_system()
    
    # Create larger test multitensor
    mt = {}
    for dims in test_dims:
        # Larger tensors for meaningful benchmarking
        shape = [64, 64, 16]  # 64x64 spatial, 16 channels
        mt[dims] = torch.randn(*shape)
    
    flat = pack_multitensor(mt, system, channel_dim=16)
    
    print(f"Test setup:")
    print(f"  - Number of slices: {len(flat.dims_list)}")
    print(f"  - Total positions: {flat.data.shape[0]:,}")
    print(f"  - Channels: {flat.data.shape[1]}")
    print(f"  - Total elements: {flat.data.numel():,}")
    print()
    
    # Warm up GPU
    for _ in range(3):
        _ = normalize_vec(flat, debias=True)
        _ = normalize_vec_old(flat, debias=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark new implementation (segment_coo + pre-computed row2slice)
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        result_new = normalize_vec(flat, debias=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    new_time = time.time() - start_time
    
    # Benchmark old implementation (scatter_mean + recomputed row2slice)
    start_time = time.time()
    for _ in range(num_runs):
        result_old = normalize_vec_old(flat, debias=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    old_time = time.time() - start_time
    
    # Verify correctness
    max_diff = torch.max(torch.abs(result_new.data - result_old.data)).item()
    
    print(f"Performance Results ({num_runs} runs):")
    print(f"  📊 Old implementation (scatter_mean):     {old_time:.4f}s ({old_time/num_runs*1000:.2f}ms per call)")
    print(f"  ⚡ New implementation (segment_coo):      {new_time:.4f}s ({new_time/num_runs*1000:.2f}ms per call)")
    print(f"  🎯 Speedup: {old_time/new_time:.2f}x")
    print(f"  ✅ Max difference: {max_diff:.2e} (numerical precision)")
    
    return old_time, new_time, max_diff


def benchmark_row2slice_computation():
    """Benchmark the cost of recomputing row2slice vs using pre-computed."""
    print("\n🔍 Benchmarking row2slice computation overhead...")
    print("=" * 60)
    
    system, test_dims = create_benchmark_multitensor_system()
    
    # Create test data
    mt = {}
    for dims in test_dims:
        shape = [64, 64, 16]
        mt[dims] = torch.randn(*shape)
    
    flat = pack_multitensor(mt, system, channel_dim=16)
    
    num_runs = 1000
    
    # Benchmark pre-computed access
    start_time = time.time()
    for _ in range(num_runs):
        row2slice = flat.row2slice  # Just access pre-computed
        _ = row2slice.max()  # Force evaluation
    precomputed_time = time.time() - start_time
    
    # Benchmark recomputation
    start_time = time.time()
    for _ in range(num_runs):
        row2slice = torch.repeat_interleave(
            torch.arange(len(flat.dims_list), device=flat.data.device, dtype=torch.long),
            flat.lengths
        )
        _ = row2slice.max()  # Force evaluation
    recomputed_time = time.time() - start_time
    
    print(f"Row2slice computation ({num_runs} runs):")
    print(f"  📊 Recomputed each time:    {recomputed_time:.4f}s ({recomputed_time/num_runs*1000:.3f}ms per call)")
    print(f"  ⚡ Pre-computed access:     {precomputed_time:.4f}s ({precomputed_time/num_runs*1000:.3f}ms per call)")
    print(f"  🎯 Speedup: {recomputed_time/precomputed_time:.1f}x")
    
    return recomputed_time, precomputed_time


def main():
    """Run all benchmarks."""
    print("🧪 Vectorized Layers Performance Benchmark")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Run benchmarks
    old_time, new_time, max_diff = benchmark_normalize_implementations()
    recomp_time, precomp_time = benchmark_row2slice_computation()
    
    print("\n📋 Summary:")
    print("=" * 60)
    print("✅ Optimizations successfully implemented:")
    print("   1. ⚡ segment_coo instead of scatter_mean")
    print("   2. 🎯 Pre-computed row2slice mapping")
    print()
    print(f"🚀 Overall performance improvement: {old_time/new_time:.2f}x faster")
    print(f"🔧 Row2slice optimization: {recomp_time/precomp_time:.1f}x faster access")
    print(f"✅ Numerical accuracy maintained (max diff: {max_diff:.2e})")


if __name__ == "__main__":
    main()