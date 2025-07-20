# Batch Performance Evaluation Tools

This package provides comprehensive tools for evaluating the performance of batched vs unbatched neural network layer implementations based on the findings in `batch_parallelization_analysis.md`.

## Key Findings Summary

The analysis revealed that **tensor size is the critical factor** determining batching efficiency:

- **Small tensors** (≤100 elements/grid): **2-4x speedup** with batching
- **Medium tensors** (100-500 elements): **1.5-2x speedup** with batching  
- **Large tensors** (≥900 elements): **0.5-0.7x speedup** (batching is slower!)

## Available Tools

### 1. Simple Evaluation Function

```python
from simple_batch_benchmark import evaluate_batch_performance_ratios

# Test all available layers with different batch sizes
results = evaluate_batch_performance_ratios(
    batch_sizes=[1, 2, 4, 8],
    problem_sizes=["small", "medium", "large"]
)

# Get optimal batch size recommendations
recommendations = results['recommendations']
print(f"Small problems: use batch_size={recommendations['small']['optimal_batch_size']}")
print(f"Medium problems: use batch_size={recommendations['medium']['optimal_batch_size']}")
print(f"Large problems: use batch_size={recommendations['large']['optimal_batch_size']}")
```

### 2. Test Specific Layers

```python
# Test only specific layers
specific_results = evaluate_batch_performance_ratios(
    layer_list=["normalize", "share_up_shared_weights_mask_true"],
    batch_sizes=[1, 2, 4, 8]
)
```

### 3. Single Layer Testing

```python
from simple_batch_benchmark import evaluate_layer_performance

# Test a single layer across all problem sizes and batch sizes
layer_results = evaluate_layer_performance("normalize", batch_sizes=[1, 2, 4, 8])
```

## Example Results

Based on our comprehensive testing:

```
🎯 OPTIMAL BATCH SIZE RECOMMENDATIONS:
   ✅  SMALL problems: batch_size=8 (2.82x speedup, 35.3% efficiency)
   ✅ MEDIUM problems: batch_size=8 (1.66x speedup, 20.8% efficiency)
   ⚠️  LARGE problems: batch_size=1 (1.00x speedup, 100.0% efficiency)
```

## Performance Categories

### Small Problems (Task 0: 6×6 grids, ~36 elements)
- **Best batch size**: 8
- **Expected speedup**: 2-4x
- **Use case**: Aggressive batching recommended

### Medium Problems (Task 37: 10×8 grids, ~80 elements)  
- **Best batch size**: 4-8
- **Expected speedup**: 1.5-2x
- **Use case**: Moderate batching beneficial

### Large Problems (Task 21: 30×30 grids, ~900 elements)
- **Best batch size**: 1 (no batching)
- **Expected speedup**: <1x (batching hurts performance)
- **Use case**: Process individually for best performance

## Adaptive Batching Strategy

Based on the results, use this adaptive strategy:

```python
def get_optimal_batch_size(tensor_elements_per_grid):
    """Get optimal batch size based on tensor complexity."""
    if tensor_elements_per_grid <= 100:
        return 8  # Aggressive batching for small tensors
    elif tensor_elements_per_grid <= 500:
        return 4  # Moderate batching for medium tensors
    else:
        return 1  # No batching for large tensors

# Usage example
grid_size = input_height * input_width
optimal_batch = get_optimal_batch_size(grid_size)
```

## Why This Happens

### Memory Bandwidth Bottleneck
- **Small tensors**: Fit in GPU cache, computation becomes the bottleneck
- **Large tensors**: Exceed cache capacity, memory bandwidth becomes the bottleneck

### Cache Hierarchy Effects  
- **L1/L2 Cache**: ~1-32MB (fast access)
- **GPU DRAM**: Gigabytes (slow access)
- **Batch overhead**: Increases memory pressure and reduces cache efficiency

## Running the Benchmarks

### Quick Test
```bash
python simple_batch_benchmark.py
```

### Full Comprehensive Analysis
```bash  
python batch_performance_benchmark.py
```

## Integration with Your Code

To use these findings in your training pipeline:

```python
# In your training loop
def get_batch_size_for_task(task):
    """Dynamically determine optimal batch size based on task complexity."""
    max_grid_elements = max([
        np.prod(example_shape) 
        for example_shape in task.shapes
    ])
    
    if max_grid_elements <= 100:
        return 8
    elif max_grid_elements <= 500: 
        return 4
    else:
        return 1

# Usage
optimal_batch_size = get_batch_size_for_task(current_task)
model = create_batched_model(batch_size=optimal_batch_size)
```

## Files

- `simple_batch_benchmark.py`: Simple, robust evaluation functions
- `batch_performance_benchmark.py`: Comprehensive evaluation with detailed reporting
- `batch_parallelization_analysis.md`: Original detailed analysis findings

## Performance Validation

The tools validate the core hypothesis from the analysis:

✅ **Task 0** (small): 2.82x speedup with batch_size=8  
✅ **Task 37** (medium): 1.66x speedup with batch_size=8  
✅ **Task 21** (large): No benefit from batching (use batch_size=1)

This demonstrates that **batching is not universally beneficial** - it depends critically on the memory footprint of your tensors. 