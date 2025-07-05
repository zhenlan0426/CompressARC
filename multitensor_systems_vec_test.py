"""
Test suite for multitensor_systems_vec.py

This test verifies that the pack_multitensor() and unpack_flat() functions correctly
preserve data through round-trip conversions. The tests ensure that:

1. Basic round-trip conversion preserves all data exactly
2. Different channel dimensions work correctly (1, 4, 16, 32)
3. Different tensor data types are preserved (float32, float64, int32, int64)
4. Metadata (shapes, offsets, lengths, dims_list) is correctly preserved
5. FlatMultiTensor view() and write() methods work correctly
6. Shape validation prevents incorrect tensor writes
7. Empty multitensor systems are handled gracefully
8. Device consistency is maintained (CPU/CUDA)

The core principle being tested is that:
    unpack_flat(pack_multitensor(original_mt, system, C), system) == original_mt

This is critical for the vectorized multitensor framework to work correctly.
"""

import pytest
import torch
import numpy as np
from multitensor_systems_vec import pack_multitensor, unpack_flat, FlatMultiTensor
import multitensor_systems


class TestPackUnpackRoundTrip:
    """Test that pack_multitensor followed by unpack_flat preserves the original data."""
    
    def setup_method(self):
        """Set up common test fixtures."""
        # Create a simple mock task for testing
        self.task = MockTask()
        
        # Create a test multitensor system
        self.n_examples = 2
        self.n_colors = 3
        self.n_x = 4
        self.n_y = 3
        self.channel_dim = 8
        
        self.multitensor_system = multitensor_systems.MultiTensorSystem(
            self.n_examples, self.n_colors, self.n_x, self.n_y, self.task
        )
    
    def create_test_multitensor(self, channel_dim):
        """Create a MultiTensor with random data for testing."""
        mt = self.multitensor_system.make_multitensor(default=None)
        
        # Fill each valid slice with random tensors
        for dims in self.multitensor_system:
            shape = self.multitensor_system.shape(dims, channel_dim)
            tensor = torch.randn(shape, dtype=torch.float32)
            mt[dims] = tensor
            
        return mt
    
    def test_pack_unpack_roundtrip_basic(self):
        """Test basic pack/unpack roundtrip preserves data."""
        # Create test data
        original_mt = self.create_test_multitensor(self.channel_dim)
        
        # Pack the multitensor
        flat_mt = pack_multitensor(original_mt, self.multitensor_system, self.channel_dim)
        
        # Verify FlatMultiTensor structure
        assert isinstance(flat_mt, FlatMultiTensor)
        assert flat_mt.channel_dim == self.channel_dim
        assert len(flat_mt.dims_list) > 0
        assert flat_mt.data.shape[1] == self.channel_dim
        
        # Unpack back to nested structure
        reconstructed_mt = unpack_flat(flat_mt, self.multitensor_system)
        
        # Verify the reconstructed data matches original
        for dims in self.multitensor_system:
            original_tensor = original_mt[dims]
            reconstructed_tensor = reconstructed_mt[dims]
            
            # Check shapes match
            assert original_tensor.shape == reconstructed_tensor.shape, \
                f"Shape mismatch for dims {dims}: {original_tensor.shape} vs {reconstructed_tensor.shape}"
            
            # Check data matches (allowing for small floating point errors)
            torch.testing.assert_close(
                original_tensor, reconstructed_tensor,
                msg=f"Data mismatch for dims {dims}"
            )
    
    def test_pack_unpack_different_channel_dims(self):
        """Test pack/unpack with different channel dimensions."""
        for channel_dim in [1, 4, 16, 32]:
            original_mt = self.create_test_multitensor(channel_dim)
            
            flat_mt = pack_multitensor(original_mt, self.multitensor_system, channel_dim)
            reconstructed_mt = unpack_flat(flat_mt, self.multitensor_system)
            
            for dims in self.multitensor_system:
                torch.testing.assert_close(
                    original_mt[dims], reconstructed_mt[dims],
                    msg=f"Mismatch for dims {dims} with channel_dim {channel_dim}"
                )
    
    def test_pack_unpack_different_dtypes(self):
        """Test pack/unpack with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            mt = self.multitensor_system.make_multitensor(default=None)
            
            for dims in self.multitensor_system:
                shape = self.multitensor_system.shape(dims, self.channel_dim)
                if dtype in [torch.float32, torch.float64]:
                    tensor = torch.randn(shape, dtype=dtype)
                else:
                    tensor = torch.randint(0, 10, shape, dtype=dtype)
                mt[dims] = tensor
            
            flat_mt = pack_multitensor(mt, self.multitensor_system, self.channel_dim)
            reconstructed_mt = unpack_flat(flat_mt, self.multitensor_system)
            
            for dims in self.multitensor_system:
                assert mt[dims].dtype == reconstructed_mt[dims].dtype
                torch.testing.assert_close(
                    mt[dims], reconstructed_mt[dims],
                    msg=f"Mismatch for dims {dims} with dtype {dtype}"
                )
    
    def test_pack_preserves_metadata(self):
        """Test that packing preserves correct metadata."""
        original_mt = self.create_test_multitensor(self.channel_dim)
        flat_mt = pack_multitensor(original_mt, self.multitensor_system, self.channel_dim)
        
        # Check that all valid dims are represented
        expected_dims = list(self.multitensor_system)
        actual_dims = [list(dims) for dims in flat_mt.dims_list]
        
        assert len(expected_dims) == len(actual_dims)
        for dims in expected_dims:
            assert dims in actual_dims, f"Missing dims {dims} in packed data"
        
        # Check that shapes are correctly stored
        for idx, dims in enumerate(flat_mt.dims_list):
            expected_shape = self.multitensor_system.shape(list(dims))
            actual_shape = flat_mt.shapes[idx]
            assert expected_shape == actual_shape, \
                f"Shape mismatch for dims {dims}: expected {expected_shape}, got {actual_shape}"
        
        # Check that offsets and lengths are consistent
        total_positions = 0
        for idx in range(len(flat_mt.dims_list)):
            assert flat_mt.offsets[idx] == total_positions
            expected_length = int(np.prod(flat_mt.shapes[idx]))
            assert flat_mt.lengths[idx] == expected_length
            total_positions += expected_length
        
        assert flat_mt.data.shape[0] == total_positions
    
    def test_flat_multitensor_view_write(self):
        """Test FlatMultiTensor view and write methods."""
        original_mt = self.create_test_multitensor(self.channel_dim)
        flat_mt = pack_multitensor(original_mt, self.multitensor_system, self.channel_dim)
        
        # Test view method
        for idx, dims in enumerate(flat_mt.dims_list):
            view_tensor = flat_mt.view(idx)
            original_tensor = original_mt[dims]
            
            assert view_tensor.shape == original_tensor.shape
            torch.testing.assert_close(view_tensor, original_tensor)
        
        # Test write method
        if len(flat_mt.dims_list) > 0:
            idx = 0
            dims = flat_mt.dims_list[idx]
            original_shape = flat_mt.shapes[idx] + [self.channel_dim]
            
            # Create new test data
            new_tensor = torch.ones(original_shape, dtype=flat_mt.data.dtype)
            
            # Write the new data
            flat_mt.write(idx, new_tensor)
            
            # Verify the write worked
            view_tensor = flat_mt.view(idx)
            torch.testing.assert_close(view_tensor, new_tensor)
    
    def test_write_shape_validation(self):
        """Test that write method validates tensor shapes."""
        original_mt = self.create_test_multitensor(self.channel_dim)
        flat_mt = pack_multitensor(original_mt, self.multitensor_system, self.channel_dim)
        
        if len(flat_mt.dims_list) > 0:
            idx = 0
            
            # Try to write tensor with wrong shape
            wrong_shape = [1, 1, 1]  # Definitely wrong shape
            wrong_tensor = torch.ones(wrong_shape)
            
            with pytest.raises(ValueError, match="Tensor shape mismatch"):
                flat_mt.write(idx, wrong_tensor)
    
    def test_empty_multitensor_system(self):
        """Test behavior with a multitensor system that has no valid dims."""
        # Test with different dimension sizes to ensure robustness
        test_configs = [
            (1, 1, 1, 1),      # All small dimensions
            (3, 4, 5, 6),      # Different sizes
            (10, 2, 15, 8),    # Larger dimensions
            (1, 10, 1, 10),    # Mixed small and large
        ]
        
        for n_examples, n_colors, n_x, n_y in test_configs:
            empty_task = MockTask(n_examples, n_x, n_y)
            empty_system = EmptyMultiTensorSystem(empty_task, n_examples, n_colors, n_x, n_y)
            
            mt = empty_system.make_multitensor(default=None)
            
            # This should work but result in empty structures
            flat_mt = pack_multitensor(mt, empty_system, self.channel_dim)
            reconstructed_mt = unpack_flat(flat_mt, empty_system)
            
            assert flat_mt.data.shape[0] == 0, f"Failed for config {(n_examples, n_colors, n_x, n_y)}"
            assert flat_mt.data.shape[1] == self.channel_dim, f"Channel dim wrong for config {(n_examples, n_colors, n_x, n_y)}"
            assert len(flat_mt.dims_list) == 0, f"dims_list not empty for config {(n_examples, n_colors, n_x, n_y)}"
            assert len(flat_mt.shapes) == 0, f"shapes not empty for config {(n_examples, n_colors, n_x, n_y)}"
            assert flat_mt.offsets.numel() == 0, f"offsets not empty for config {(n_examples, n_colors, n_x, n_y)}"
            assert flat_mt.lengths.numel() == 0, f"lengths not empty for config {(n_examples, n_colors, n_x, n_y)}"
    
    def test_large_dimensions(self):
        """Test with larger dimension sizes to stress test the system."""
        # Create a larger test system
        large_task = MockTask()
        large_task.masks = torch.ones((5, 20, 15, 2))  # Adjust mask size
        
        large_system = multitensor_systems.MultiTensorSystem(5, 8, 20, 15, large_task)
        
        original_mt = large_system.make_multitensor(default=None)
        for dims in large_system:
            shape = large_system.shape(dims, self.channel_dim)
            original_mt[dims] = torch.randn(shape)
        
        # Pack and unpack
        flat_mt = pack_multitensor(original_mt, large_system, self.channel_dim)
        reconstructed_mt = unpack_flat(flat_mt, large_system)
        
        # Verify integrity with larger system
        for dims in large_system:
            torch.testing.assert_close(
                original_mt[dims], reconstructed_mt[dims],
                msg=f"Mismatch for dims {dims} in large system"
            )
    
    def test_single_valid_dim(self):
        """Test with a system that has only one valid dimension combination."""
        class SingleDimSystem:
            def __init__(self, task):
                self.task = task
                self.n_examples = 2
                self.n_colors = 3
                self.n_directions = 8
                self.n_x = 4
                self.n_y = 5
                self.dim_lengths = [2, 3, 8, 4, 5]
                self._valid_dims = [0, 1, 0, 0, 0]  # Only one specific combination
            
            def dims_valid(self, dims):
                return dims == self._valid_dims
            
            def shape(self, dims, extra_dim=None):
                shape = []
                for dim_index, length in enumerate(self.dim_lengths):
                    if dims[dim_index]:
                        shape.append(length)
                if extra_dim is not None:
                    shape.append(extra_dim)
                return shape
            
            def __iter__(self):
                if self.dims_valid(self._valid_dims):
                    yield self._valid_dims
            
            def make_multitensor(self, default=None):
                from multitensor_systems import MultiTensor
                return MultiTensor(self._make_multitensor(default, 0), self)
            
            def _make_multitensor(self, default, index):
                if index == 5:
                    return default
                return [self._make_multitensor(default, index+1) for _ in range(2)]
        
        single_task = MockTask()
        single_system = SingleDimSystem(single_task)
        
        mt = single_system.make_multitensor(default=None)
        for dims in single_system:
            shape = single_system.shape(dims, self.channel_dim)
            mt[dims] = torch.randn(shape)
        
        # Pack and unpack
        flat_mt = pack_multitensor(mt, single_system, self.channel_dim)
        reconstructed_mt = unpack_flat(flat_mt, single_system)
        
        # Should have exactly one slice
        assert len(flat_mt.dims_list) == 1
        assert flat_mt.dims_list[0] == tuple(single_system._valid_dims)
        
        # Verify data integrity
        for dims in single_system:
            torch.testing.assert_close(
                mt[dims], reconstructed_mt[dims],
                msg=f"Mismatch for dims {dims} in single dim system"
            )
    
    def test_device_consistency(self):
        """Test that device is preserved through pack/unpack."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        mt = self.multitensor_system.make_multitensor(default=None)
        
        for dims in self.multitensor_system:
            shape = self.multitensor_system.shape(dims, self.channel_dim)
            tensor = torch.randn(shape, dtype=torch.float32, device=device)
            mt[dims] = tensor
        
        flat_mt = pack_multitensor(mt, self.multitensor_system, self.channel_dim)
        reconstructed_mt = unpack_flat(flat_mt, self.multitensor_system)
        
        # Check that all tensors are on the correct device
        assert flat_mt.data.device.type == device.type
        assert flat_mt.offsets.device.type == device.type
        assert flat_mt.lengths.device.type == device.type
        
        for dims in self.multitensor_system:
            assert reconstructed_mt[dims].device.type == device.type


class MockTask:
    """Mock task class for testing."""
    def __init__(self, n_examples=2, n_x=4, n_y=3):
        self.in_out_same_size = False
        self.all_out_same_size = False
        self.masks = torch.ones((n_examples, n_x, n_y, 2))  # (examples, x, y, in/out)


class EmptyMultiTensorSystem:
    """A multitensor system with no valid dimension combinations."""
    def __init__(self, task, n_examples=3, n_colors=4, n_x=5, n_y=6):
        self.task = task
        self.n_examples = n_examples
        self.n_colors = n_colors
        self.n_directions = 8
        self.n_x = n_x
        self.n_y = n_y
        self.dim_lengths = [self.n_examples, self.n_colors, self.n_directions, self.n_x, self.n_y]
    
    def dims_valid(self, dims):
        """Always return False - no valid dims."""
        return False
    
    def shape(self, dims, extra_dim=None):
        """Return shape based on dims, even though no dims are valid."""
        shape = []
        for dim_index, length in enumerate(self.dim_lengths):
            if dims[dim_index]:
                shape.append(length)
        if extra_dim is not None:
            shape.append(extra_dim)
        return shape
    
    def __iter__(self):
        """Yield no valid dims."""
        return iter([])
    
    def make_multitensor(self, default=None):
        """Create an empty multitensor."""
        from multitensor_systems import MultiTensor
        return MultiTensor(self._make_multitensor(default, 0), self)
    
    def _make_multitensor(self, default, index):
        """Create nested structure."""
        if index == 5:  # NUM_DIMENSIONS
            return default
        return [self._make_multitensor(default, index+1) for _ in range(2)]


if __name__ == "__main__":
    pytest.main([__file__]) 