"""
Simple Batch Performance Evaluation

A streamlined version that evaluates performance ratios between batched and unbatched 
implementations for different batch sizes (1, 2, 4, 8) and different problem sizes.
Focuses on reliable layer implementations and provides clean output.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Optional
import warnings

# Import existing modules  
import preprocessing
import layers
import layers_batch
import initializers_batch
import my_test

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def evaluate_layer_performance(layer_name: str, batch_sizes: List[int] = [1, 2, 4, 8], 
                              iterations: int = 5) -> Dict:
    """
    Evaluate a single layer's performance across different batch sizes and problem sizes.
    
    Args:
        layer_name: Name of the layer to test (from LAYER_TEST_REGISTRY)
        batch_sizes: List of batch sizes to test
        iterations: Number of timing iterations
        
    Returns:
        Dictionary with performance results
    """
    if layer_name not in my_test.LAYER_TEST_REGISTRY:
        return {"error": f"Layer {layer_name} not found in registry"}
    
    layer_info = my_test.LAYER_TEST_REGISTRY[layer_name]
    
    # Test tasks (based on analysis findings)
    test_tasks = {
        "small": {"task_id": 0, "expected_elements": 36},    # 6×6 grids
        "medium": {"task_id": 37, "expected_elements": 80},  # 10×8 grids  
        "large": {"task_id": 21, "expected_elements": 900}   # 30×30 grids
    }
    
    # Load tasks
    tasks = {}
    for size_name, task_info in test_tasks.items():
        try:
            task_list = preprocessing.preprocess_tasks('training', [task_info['task_id']])
            tasks[size_name] = task_list[0]
        except Exception:
            continue
    
    if not tasks:
        return {"error": "No tasks could be loaded"}
    
    results = {}
    
    for task_size, task in tasks.items():
        task_results = {}
        print(f"  📏 {task_size} (Task {test_tasks[task_size]['task_id']}): ", end="")
        
        for batch_size in batch_sizes:
            try:
                # Generate test data
                if 'share' in layer_name:
                    task_id = test_tasks[task_size]['task_id']
                    batch_weights = 'batched_weights' in layer_name
                    batched_args, _ = my_test.generate_share_data_with_task(task_id, batch_size, batch_weights)
                else:
                    system = task.multitensor_system
                    generate_fn = layer_info['generate']
                    batched_args = generate_fn(system, batch_size)
                
                # Prepare functions
                ref_func = layer_info['ref']
                batched_func = layer_info['batched']
                kwargs = layer_info.get('kwargs', {})
                
                # Split for unbatched execution
                per_batch_args = []
                for b in range(batch_size):
                    batch_args = []
                    for arg, is_batched in batched_args:
                        if is_batched:
                            batch_args.append(my_test.split_multitensor_batch(arg, batch_size)[b])
                        else:
                            batch_args.append(arg)
                    per_batch_args.append(batch_args)
                
                # Time both implementations
                def time_execution(func, args, kwargs, iters):
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start = time.time()
                    for _ in range(iters):
                        result = func(*args, **kwargs)
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                    return (time.time() - start) / iters
                
                # Time unbatched (loop over batch)
                def run_unbatched():
                    return [ref_func(*args, **kwargs) for args in per_batch_args]
                
                # Time batched  
                def run_batched():
                    args = [arg for arg, _ in batched_args]
                    return batched_func(*args, **kwargs)
                
                unbatched_time = time_execution(run_unbatched, [], {}, iterations)
                batched_time = time_execution(run_batched, [], {}, iterations)
                
                if batched_time > 0:
                    speedup = unbatched_time / batched_time
                else:
                    speedup = 0.0
                
                task_results[batch_size] = {
                    "unbatched_time": unbatched_time,
                    "batched_time": batched_time,
                    "speedup": speedup,
                    "efficiency": speedup / batch_size if batch_size > 0 else 0
                }
                
                print(f"bs{batch_size}:{speedup:.1f}x ", end="")
                
            except Exception as e:
                task_results[batch_size] = {"error": str(e), "speedup": 0.0}
                print(f"bs{batch_size}:❌ ", end="")
        
        print()  # New line after each task size
        results[task_size] = task_results
    
    return results


def batch_performance_summary(batch_sizes: List[int] = [1, 2, 4, 8], 
                             layers_to_test: Optional[List[str]] = None,
                             iterations: int = 5) -> Dict:
    """
    Comprehensive batch performance evaluation across multiple layers and problem sizes.
    
    Args:
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8])
        layers_to_test: List of layer names to test (defaults to reliable ones)
        iterations: Number of timing iterations for stability
        
    Returns:
        Dictionary with comprehensive results and recommendations
    """
    print("🎯 Batch Performance Evaluation")
    print("=" * 50)
    
    # Default to reliable layers if none specified
    if layers_to_test is None:
        reliable_layers = [
            "normalize",
            "affine_batched_weights", 
            "softmax_batched_weights",
            "share_up_batched_weights_mask_true",
            "share_up_shared_weights_mask_true",
            "share_down_shared_weights_mask_true"
        ]
        # Filter to only available layers
        layers_to_test = [layer for layer in reliable_layers 
                         if layer in my_test.LAYER_TEST_REGISTRY]
    
    print(f"📊 Testing {len(layers_to_test)} layers")
    print(f"🔄 Testing batch sizes: {batch_sizes}")
    print(f"⏱️  Using {iterations} iterations per test")
    
    all_results = {}
    total_times = {"unbatched": {}, "batched": {}}
    
    for layer_name in layers_to_test:
        print(f"\n📊 {layer_name}")
        layer_results = evaluate_layer_performance(layer_name, batch_sizes, iterations)
        
        if "error" not in layer_results:
            all_results[layer_name] = layer_results
            
            # Accumulate total times
            for task_size, task_results in layer_results.items():
                if task_size not in total_times["unbatched"]:
                    total_times["unbatched"][task_size] = {bs: 0.0 for bs in batch_sizes}
                    total_times["batched"][task_size] = {bs: 0.0 for bs in batch_sizes}
                
                for batch_size, result in task_results.items():
                    if "error" not in result:
                        total_times["unbatched"][task_size][batch_size] += result.get("unbatched_time", 0)
                        total_times["batched"][task_size][batch_size] += result.get("batched_time", 0)
        else:
            print(f"  ❌ {layer_results['error']}")
    
    # Calculate total performance and recommendations
    print("\n" + "=" * 60)
    print("📈 TOTAL PERFORMANCE ACROSS ALL LAYERS")
    print("=" * 60)
    
    recommendations = {}
    
    for task_size in ["small", "medium", "large"]:
        if task_size in total_times["unbatched"]:
            print(f"\n🏷️  {task_size.upper()} Problems:")
            
            best_batch_size = 1
            best_speedup = 1.0
            
            for batch_size in batch_sizes:
                total_unbatched = total_times["unbatched"][task_size].get(batch_size, 0)
                total_batched = total_times["batched"][task_size].get(batch_size, 0)
                
                if total_batched > 0:
                    speedup = total_unbatched / total_batched
                    efficiency = speedup / batch_size
                    
                    status = "✅" if speedup > 1.0 else "⚠️ "
                    print(f"   Batch Size {batch_size}: {speedup:.2f}x speedup "
                          f"({efficiency:.1%} efficiency) {status}")
                    
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_batch_size = batch_size
            
            recommendations[task_size] = {
                "optimal_batch_size": best_batch_size,
                "speedup": best_speedup,
                "efficiency": best_speedup / best_batch_size
            }
            
            recommendation_status = "✅" if best_speedup > 1.0 else "⚠️"
            print(f"   {recommendation_status} RECOMMENDATION: Use batch_size={best_batch_size} "
                  f"({best_speedup:.2f}x total speedup)")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 OPTIMAL BATCH SIZE RECOMMENDATIONS:")
    for task_size, rec in recommendations.items():
        status = "✅" if rec["speedup"] > 1.0 else "⚠️"
        print(f"   {status} {task_size.upper():>6} problems: batch_size={rec['optimal_batch_size']} "
              f"({rec['speedup']:.2f}x speedup, {rec['efficiency']:.1%} efficiency)")
    
    return {
        "layer_results": all_results,
        "total_times": total_times,
        "recommendations": recommendations,
        "summary": {
            "layers_tested": len(all_results),
            "batch_sizes_tested": len(batch_sizes),
            "total_successful_tests": sum(
                len([r for r in task_results.values() if "error" not in r])
                for layer_results in all_results.values()
                for task_results in layer_results.values()
            )
        }
    }


# Specific function matching user's exact request
def evaluate_batch_performance_ratios(layer_list: Optional[List[str]] = None, 
                                     problem_sizes: List[str] = ["small", "medium", "large"],
                                     batch_sizes: List[int] = [1, 2, 4, 8]) -> Dict:
    """
    The exact function the user requested: evaluate performance ratio between batched 
    and unbatched with different batch sizes for different problem sizes.
    
    Args:
        layer_list: List of layers to test (defaults to available working layers)
        problem_sizes: List of problem sizes to test
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8])
        
    Returns:
        Dictionary with performance ratios and optimal batch size recommendations
    """
    return batch_performance_summary(
        batch_sizes=batch_sizes, 
        layers_to_test=layer_list,
        iterations=10  # More iterations for stable results
    )


if __name__ == "__main__":
    # Run the evaluation as requested by the user
    print("🚀 Running Batch Performance Ratio Evaluation")
    results = evaluate_batch_performance_ratios(batch_sizes=[1, 2, 4, 8])
    
    print(f"\n📊 Final Summary:")
    print(f"   - Layers successfully tested: {results['summary']['layers_tested']}")
    print(f"   - Total successful benchmarks: {results['summary']['total_successful_tests']}")
    
    # Example of accessing specific results
    recommendations = results['recommendations']
    for problem_size, rec in recommendations.items():
        print(f"   - {problem_size} problems: optimal batch_size = {rec['optimal_batch_size']}") 