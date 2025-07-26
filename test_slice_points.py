import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Tuple

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def vectorized_best_slice_points_batch(mask: torch.Tensor) -> torch.Tensor:
    """Vectorized implementation from bayesian_logger_batch.py
    
    mask : (B, n_test, L)
    output: (B, n_test, 2) where the last dim is (start, end)
    """
    B, n_test, L = mask.shape
    total_sum = mask.sum(dim=2, keepdim=True)                # (B, n_test, 1)

    # Prepare tensors to record best values
    best_score = torch.full((B, n_test), -float('inf'))      # (B, n_test)
    best_start = torch.zeros((B, n_test), dtype=torch.long)
    best_end = torch.ones((B, n_test), dtype=torch.long)

    search_lengths = range(1, L + 1)
    mask_ = mask.unsqueeze(1)                                # (B,1,n_test,L)
    mask_ = mask_.reshape(B * n_test, 1, L)                  # merge for conv1d
    for length in search_lengths:
        kernel = torch.ones(1, 1, length, device=mask.device)
        seg_sum = F.conv1d(mask_, kernel, stride=1)          # (B*n_test, 1, L-length+1)
        seg_sum = seg_sum.squeeze(1)                         # (B*n_test, offsets)

        score = 2 * seg_sum - total_sum.view(-1, 1)          # broadcast
        # Find best offset for this length
        max_score, max_idx = torch.max(score, dim=1)         # (B*n_test,)

        update_mask = max_score > best_score.view(-1)
        if update_mask.any():
            best_score.view(-1)[update_mask] = max_score[update_mask]
            best_start.view(-1)[update_mask] = max_idx[update_mask]
            best_end.view(-1)[update_mask] = max_idx[update_mask] + length

    return torch.stack([best_start, best_end], dim=-1)       # (B, n_test, 2)


def non_vectorized_best_slice_point(mask: torch.Tensor) -> Tuple[int, int]:
    """Non-vectorized implementation from solution_selection_batch.py
    
    mask : (L,) - single mask
    output: (start, end) tuple
    """
    L = mask.shape[0]
    search_lengths = list(range(1, L + 1))
    max_logprob, best_slice_start, best_slice_end = None, None, None

    for length in search_lengths:
        logprobs = torch.stack([
            -torch.sum(mask[:offset]) + torch.sum(mask[offset:offset + length]) - torch.sum(mask[offset + length:])
            for offset in range(mask.shape[0] - length + 1)
        ])
        if max_logprob is None or torch.max(logprobs) > max_logprob:
            max_logprob = torch.max(logprobs)
            best_slice_start = torch.argmax(logprobs).item()
            best_slice_end = best_slice_start + length

    return best_slice_start, best_slice_end


def non_vectorized_batch_implementation(mask: torch.Tensor) -> torch.Tensor:
    """Apply non-vectorized implementation using loops over batch and examples
    
    mask : (B, n_test, L)
    output: (B, n_test, 2) where the last dim is (start, end)
    """
    B, n_test, L = mask.shape
    results = torch.zeros((B, n_test, 2), dtype=torch.long)
    
    for b in range(B):
        for n in range(n_test):
            start, end = non_vectorized_best_slice_point(mask[b, n, :])
            results[b, n, 0] = start
            results[b, n, 1] = end
    
    return results


def generate_random_test_data(B: int, n_test: int, L: int) -> torch.Tensor:
    """Generate random binary mask data for testing"""
    # Create random binary masks with varying sparsity
    masks = []
    for _ in range(B):
        batch_masks = []
        for _ in range(n_test):
            # Random sparsity between 0.1 and 0.8
            sparsity = np.random.uniform(0.1, 0.8)
            mask = torch.bernoulli(torch.full((L,), sparsity))
            batch_masks.append(mask)
        masks.append(torch.stack(batch_masks))
    return torch.stack(masks)


def test_correctness():
    """Test that both implementations produce identical results"""
    print("Testing correctness...")
    
    # Test with different sizes
    test_cases = [
        (2, 3, 10),   # Small case
        (1, 1, 5),    # Minimal case
        (3, 4, 15),   # Medium case
        (5, 2, 20),   # Larger case
    ]
    
    all_passed = True
    
    for B, n_test, L in test_cases:
        print(f"  Testing B={B}, n_test={n_test}, L={L}...")
        
        # Generate test data
        mask = generate_random_test_data(B, n_test, L)
        
        # Run both implementations
        vectorized_result = vectorized_best_slice_points_batch(mask)
        non_vectorized_result = non_vectorized_batch_implementation(mask)
        
        # Compare results
        if torch.equal(vectorized_result, non_vectorized_result):
            print(f"    ✓ PASSED")
        else:
            print(f"    ✗ FAILED")
            print(f"      Vectorized result shape: {vectorized_result.shape}")
            print(f"      Non-vectorized result shape: {non_vectorized_result.shape}")
            print(f"      Vectorized result:\n{vectorized_result}")
            print(f"      Non-vectorized result:\n{non_vectorized_result}")
            print(f"      Difference:\n{vectorized_result - non_vectorized_result}")
            all_passed = False
    
    return all_passed


def test_performance():
    """Compare performance of both implementations"""
    print("\nTesting performance...")
    
    # Test with larger data
    B, n_test, L = 10, 20, 50
    mask = generate_random_test_data(B, n_test, L)
    
    # Warm up
    _ = vectorized_best_slice_points_batch(mask)
    _ = non_vectorized_batch_implementation(mask)
    
    # Time vectorized implementation
    start_time = time.time()
    for _ in range(10):  # Run multiple times for better timing
        vectorized_result = vectorized_best_slice_points_batch(mask)
    vectorized_time = (time.time() - start_time) / 10
    
    # Time non-vectorized implementation
    start_time = time.time()
    for _ in range(10):  # Run multiple times for better timing
        non_vectorized_result = non_vectorized_batch_implementation(mask)
    non_vectorized_time = (time.time() - start_time) / 10
    
    print(f"  Vectorized implementation: {vectorized_time:.6f} seconds")
    print(f"  Non-vectorized implementation: {non_vectorized_time:.6f} seconds")
    print(f"  Speedup: {non_vectorized_time / vectorized_time:.2f}x")
    
    # Verify results are still the same
    if torch.equal(vectorized_result, non_vectorized_result):
        print("  ✓ Results match in performance test")
    else:
        print("  ✗ Results don't match in performance test")


def test_edge_cases():
    """Test edge cases"""
    print("\nTesting edge cases...")
    
    # Test case 1: All zeros
    mask = torch.zeros(2, 3, 10)
    vec_result = vectorized_best_slice_points_batch(mask)
    nonvec_result = non_vectorized_batch_implementation(mask)
    
    if torch.equal(vec_result, nonvec_result):
        print("  ✓ All zeros case passed")
    else:
        print("  ✗ All zeros case failed")
        print(f"    Vectorized: {vec_result}")
        print(f"    Non-vectorized: {nonvec_result}")
    
    # Test case 2: All ones
    mask = torch.ones(2, 3, 10)
    vec_result = vectorized_best_slice_points_batch(mask)
    nonvec_result = non_vectorized_batch_implementation(mask)
    
    if torch.equal(vec_result, nonvec_result):
        print("  ✓ All ones case passed")
    else:
        print("  ✗ All ones case failed")
        print(f"    Vectorized: {vec_result}")
        print(f"    Non-vectorized: {nonvec_result}")
    
    # Test case 3: Single element
    mask = torch.tensor([[[1.0]]])
    vec_result = vectorized_best_slice_points_batch(mask)
    nonvec_result = non_vectorized_batch_implementation(mask)
    
    if torch.equal(vec_result, nonvec_result):
        print("  ✓ Single element case passed")
    else:
        print("  ✗ Single element case failed")
        print(f"    Vectorized: {vec_result}")
        print(f"    Non-vectorized: {nonvec_result}")


def detailed_example():
    """Show a detailed example with intermediate steps"""
    print("\nDetailed example:")
    
    # Create a simple test case
    mask = torch.tensor([[[1.0, 0.0, 1.0, 1.0, 0.0]]])  # B=1, n_test=1, L=5
    print(f"Input mask: {mask.squeeze()}")
    
    vec_result = vectorized_best_slice_points_batch(mask)
    nonvec_result = non_vectorized_batch_implementation(mask)
    
    print(f"Vectorized result: start={vec_result[0,0,0].item()}, end={vec_result[0,0,1].item()}")
    print(f"Non-vectorized result: start={nonvec_result[0,0,0].item()}, end={nonvec_result[0,0,1].item()}")
    
    # Show the slice
    start, end = vec_result[0,0,0].item(), vec_result[0,0,1].item()
    print(f"Selected slice: {mask.squeeze()[start:end]}")


if __name__ == "__main__":
    print("=== Slice Points Implementation Comparison Test ===")
    
    # Run correctness tests
    correctness_passed = test_correctness()
    
    # Run performance tests
    test_performance()
    
    # Run edge case tests
    test_edge_cases()
    
    # Show detailed example
    detailed_example()
    
    print("\n=== Summary ===")
    if correctness_passed:
        print("✓ All correctness tests PASSED - Implementations are equivalent!")
    else:
        print("✗ Some correctness tests FAILED - Implementations differ!")
    
    print("\nTest completed.") 