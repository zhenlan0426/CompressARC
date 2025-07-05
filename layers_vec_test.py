import torch
import numpy as np
import pytest
from unittest.mock import Mock

# Import the original and vectorized implementations
import layers
import multitensor_systems
from layers_vec import normalize_vec, affine_vec
from multitensor_systems_vec import FlatMultiTensor, pack_multitensor, unpack_flat


class TestTask:
    """Mock task class for testing."""
    def __init__(self, n_examples=2, n_x=4, n_y=4):
        self.n_examples = n_examples
        self.n_x = n_x
        self.n_y = n_y
        self.shapes = [[(3, 3), (3, 3)] for _ in range(n_examples)]
        self.masks = torch.ones(n_examples, n_x, n_y, 2)


def create_test_multitensor_system():
    """Create a simple multitensor system for testing."""
    task = TestTask()
    
    # Create a mock multitensor system
    system = Mock()
    system.task = task
    system.make_multitensor = Mock(return_value={})
    
    # Define test dimensions - simplified for testing
    test_dims = [(1, 0, 0, 1, 1), (1, 1, 0, 1, 1), (1, 0, 1, 1, 1)]
    
    # Make system iterable
    system.__iter__ = lambda self: iter(test_dims)
    
    return system, test_dims


def test_normalize_vec_basic():
    """Test basic normalize_vec functionality."""
    print("Testing normalize_vec basic functionality...")
    
    # Create test data
    system, test_dims = create_test_multitensor_system()
    
    # Create test multitensor
    mt = {}
    for dims in test_dims:
        # Create random tensor with shape corresponding to dims
        shape = [3, 3, 16]  # spatial + channel
        mt[dims] = torch.randn(*shape)
    
    # Pack into FlatMultiTensor
    flat = pack_multitensor(mt, system, channel_dim=16)
    
    # Apply vectorized normalize
    normalized_flat = normalize_vec(flat, debias=True)
    
    # Apply original normalize to each tensor
    original_results = {}
    for dims in system:
        tensor = mt[dims]
        # Apply original normalize directly (without the decorator)
        all_but_last = list(range(len(tensor.shape)-1))
        normalized = tensor - torch.mean(tensor, dim=all_but_last)
        normalized = normalized / torch.sqrt(1e-8+torch.mean(normalized**2, dim=all_but_last))
        original_results[dims] = normalized
    
    # Compare results
    for i, dims in enumerate(test_dims):
        vec_result = normalized_flat.view(i)
        orig_result = original_results[dims]
        
        print(f"Dims {dims}: vec_result shape = {vec_result.shape}, orig_result shape = {orig_result.shape}")
        
        # Check if results are close
        assert torch.allclose(vec_result, orig_result, atol=1e-5), f"Results don't match for dims {dims}"
    
    print("✓ Basic normalize_vec test passed")


def test_normalize_vec_no_debias():
    """Test normalize_vec with debias=False."""
    print("Testing normalize_vec with debias=False...")
    
    system, test_dims = create_test_multitensor_system()
    
    # Create test multitensor
    mt = {}
    for dims in test_dims:
        shape = [3, 3, 16]
        mt[dims] = torch.randn(*shape)
    
    # Pack and normalize
    flat = pack_multitensor(mt, system, channel_dim=16)
    normalized_flat = normalize_vec(flat, debias=False)
    
    # Apply original normalize without debias
    original_results = {}
    for dims in system:
        tensor = mt[dims]
        all_but_last = list(range(len(tensor.shape)-1))
        normalized = tensor / torch.sqrt(1e-8+torch.mean(tensor**2, dim=all_but_last))
        original_results[dims] = normalized
    
    # Compare results
    for i, dims in enumerate(test_dims):
        vec_result = normalized_flat.view(i)
        orig_result = original_results[dims]
        assert torch.allclose(vec_result, orig_result, atol=1e-5), f"Results don't match for dims {dims}"
    
    print("✓ normalize_vec no debias test passed")


def test_normalize_vec_edge_cases():
    """Test normalize_vec with edge cases."""
    print("Testing normalize_vec edge cases...")
    
    system, test_dims = create_test_multitensor_system()
    
    # Test with very small values
    mt = {}
    for dims in test_dims:
        shape = [3, 3, 16]
        mt[dims] = torch.randn(*shape) * 1e-10
    
    flat = pack_multitensor(mt, system, channel_dim=16)
    normalized_flat = normalize_vec(flat, debias=True)
    
    # Check that we don't get NaN or Inf
    assert torch.isfinite(normalized_flat.data).all(), "Normalized data contains NaN or Inf"
    
    # Test with constant values
    mt = {}
    for dims in test_dims:
        shape = [3, 3, 16]
        mt[dims] = torch.ones(*shape) * 2.5
    
    flat = pack_multitensor(mt, system, channel_dim=16)
    normalized_flat = normalize_vec(flat, debias=True)
    
    # With debias=True, constant values should become zero
    assert torch.allclose(normalized_flat.data, torch.zeros_like(normalized_flat.data), atol=1e-5)
    
    print("✓ normalize_vec edge cases test passed")


def test_affine_vec_basic():
    """Test basic affine_vec functionality."""
    print("Testing affine_vec basic functionality...")
    
    system, test_dims = create_test_multitensor_system()
    
    # Create test multitensor
    mt = {}
    for dims in test_dims:
        shape = [3, 3, 16]
        mt[dims] = torch.randn(*shape)
    
    flat = pack_multitensor(mt, system, channel_dim=16)
    
    # Test global weights
    weights = torch.randn(16, 16)
    bias = torch.randn(16)
    
    result = affine_vec(flat, weights, bias)
    
    # Manual verification
    expected_data = torch.matmul(flat.data, weights) + bias
    assert torch.allclose(result.data, expected_data, atol=1e-5)
    
    print("✓ Basic affine_vec test passed")





if __name__ == "__main__":
    test_normalize_vec_basic()
    test_normalize_vec_no_debias()
    test_normalize_vec_edge_cases()
    test_affine_vec_basic()
    print("\n🎉 All tests passed!") 