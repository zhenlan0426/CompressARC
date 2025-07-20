# Adaptive Batch Size Function Usage

## Overview

The `get_optimal_batch_size()` function takes a preprocessed task and returns an optimal batch size recommendation based on empirical performance analysis from your batch parallelization study.

## Key Function

### `get_optimal_batch_size(task, conservative=False, verbose=False)`

**Purpose**: Analyze a task and recommend optimal batch size for maximum performance.

**Parameters**:
- `task`: A preprocessed task object from `preprocessing.preprocess_tasks()`
- `conservative`: Use safer (lower) batch sizes if True
- `verbose`: Print detailed analysis if True

**Returns**: Dictionary with:
- `recommended_batch_size`: Optimal batch size (1, 2, 4, or 8)
- `expected_speedup`: Predicted performance improvement
- `confidence`: Reliability of recommendation ("high", "medium", "low")
- `complexity_category`: Task category ("small", "medium", "large")
- `reasoning`: Explanation of the recommendation
- `task_stats`: Detailed task metrics

## Quick Usage Examples

### Basic Usage
```python
import preprocessing
from adaptive_batch_size import get_optimal_batch_size

# Load your task
tasks = preprocessing.preprocess_tasks('training', [0])
task = tasks[0]

# Get recommendation
result = get_optimal_batch_size(task, verbose=True)
optimal_batch_size = result['recommended_batch_size']

# Use in your model
# model = create_model(task, batch_size=optimal_batch_size)
```

### Multiple Tasks
```python
from adaptive_batch_size import get_batch_sizes_for_tasks

# Analyze multiple tasks at once
tasks = preprocessing.preprocess_tasks('training', [0, 37, 21, 150])
results = get_batch_sizes_for_tasks(tasks, verbose=True)

# Access recommendations
for task_name, rec in results['recommendations'].items():
    batch_size = rec['recommended_batch_size']
    speedup = rec['expected_speedup']
    print(f"{task_name}: use batch_size={batch_size} ({speedup:.1f}x speedup)")
```

## Decision Logic

The function analyzes the maximum grid size (`task.n_x × task.n_y`) and categorizes tasks:

### Small Tasks (≤100 elements per grid)
- **Recommended batch size**: 8 (or 4 if conservative)
- **Expected speedup**: ~2.8x
- **Examples**: 6×6 grids (36 elements), 10×8 grids (80 elements)
- **Reasoning**: Small tensors fit in GPU cache, batching provides excellent speedup

### Medium Tasks (100-900 elements per grid)  
- **Recommended batch size**: 4 (or 2 if conservative)
- **Expected speedup**: ~1.7x
- **Examples**: 11×11 grids (121 elements), 20×20 grids (400 elements)
- **Reasoning**: Moderate tensor sizes benefit from batching but with diminishing returns

### Large Tasks (>900 elements per grid)
- **Recommended batch size**: 1 (no batching)
- **Expected speedup**: 1.0x (no improvement)
- **Examples**: 30×30 grids (900 elements), 40×40 grids (1600 elements)
- **Reasoning**: Large tensors exceed cache capacity, batching hurts due to memory bandwidth limits

## Real-World Example Results

From testing with actual ARC tasks:

```
📋 Task-Specific Recommendations:
  00576224: batch_size=8 (small, 6×6 grid)    # 2.8x speedup expected
  0a1d4ef5: batch_size=4 (medium, 30×30 grid) # 1.7x speedup expected  
  10fcaaa3: batch_size=8 (small, 10×8 grid)   # 2.8x speedup expected
  29623171: batch_size=4 (medium, 11×11 grid) # 1.7x speedup expected
  36d67576: batch_size=4 (medium, 14×15 grid) # 1.7x speedup expected
```

## Integration into Training Pipeline

```python
def train_with_adaptive_batching(task_list):
    for task in task_list:
        # Get optimal batch size
        recommendation = get_optimal_batch_size(task)
        batch_size = recommendation['recommended_batch_size'] 
        expected_speedup = recommendation['expected_speedup']
        
        print(f"Training {task.task_name} with batch_size={batch_size}")
        print(f"Expected {expected_speedup:.1f}x speedup")
        
        # Create model with optimal batch size
        model = create_batched_model(task, batch_size=batch_size)
        train_model(model, task)
```

## Conservative vs Aggressive Strategies

```python
# Aggressive (maximum performance)
aggressive = get_optimal_batch_size(task, conservative=False)
# → Recommends higher batch sizes for max speedup

# Conservative (safer, more stable)
conservative = get_optimal_batch_size(task, conservative=True) 
# → Recommends lower batch sizes to avoid memory issues
```

## Files in This Package

- `adaptive_batch_size.py` - Main implementation
- `example_adaptive_batching.py` - Complete usage examples
- `USAGE_adaptive_batch_size.md` - This documentation
- `simple_batch_benchmark.py` - Performance benchmarking tools
- `batch_parallelization_analysis.md` - Original empirical analysis

## Performance Validation

This function is based on comprehensive empirical testing that confirmed:

✅ **Small tasks (Task 0, 6×6)**: 2.82x speedup with batch_size=8  
✅ **Medium tasks (Task 37, 10×8)**: 1.66x speedup with batch_size=8  
✅ **Large tasks (Task 21, 30×30)**: No benefit from batching  

## Memory Considerations

The function also estimates memory usage and automatically reduces batch sizes if memory consumption is projected to exceed safe limits (~100MB per batch).

## Why This Works

The core insight from your analysis is that **tensor size, not algorithmic complexity, determines batching efficiency**:

- **Small tensors**: Computation-bound → batching helps by parallelizing operations
- **Large tensors**: Memory bandwidth-bound → batching hurts by exceeding cache capacity

This function encodes these findings into a practical tool for production use.

---

**Key Takeaway**: Use this function to automatically determine optimal batch sizes instead of using fixed batching strategies. Task complexity matters more than you might expect! 