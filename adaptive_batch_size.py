"""
Adaptive Batch Size Recommendation

This module provides functions to determine optimal batch sizes for tasks based on 
empirical performance analysis findings from batch_parallelization_analysis.md.

Key findings:
- Small tensors (≤100 elements/grid): batch_size=8 (2.82x speedup)
- Medium tensors (100-500 elements): batch_size=8 (1.66x speedup)  
- Large tensors (≥900 elements): batch_size=1 (1.00x speedup - no batching benefit)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import preprocessing


def get_optimal_batch_size(task: preprocessing.Task, 
                          conservative: bool = False,
                          verbose: bool = False) -> Dict:
    """
    Determine optimal batch size for a task based on empirical performance analysis.
    
    Args:
        task: A preprocessed task object from preprocessing.preprocess_tasks()
        conservative: If True, use more conservative batch sizes (lower risk)
        verbose: If True, print detailed reasoning
        
    Returns:
        Dictionary with recommended batch_size, reasoning, and task statistics
    """
    
    # Calculate task complexity metrics
    max_grid_elements = task.n_x * task.n_y
    avg_grid_elements = np.mean([
        shape[0][0] * shape[0][1] if shape[0] else 0 
        for shape in task.shapes
    ])
    max_input_elements = max([
        shape[0][0] * shape[0][1] if shape[0] else 0 
        for shape in task.shapes
    ])
    max_output_elements = max([
        shape[1][0] * shape[1][1] if shape[1] else 0 
        for shape in task.shapes
    ])
    
    # Determine complexity category based on empirical findings
    complexity_threshold_small = 100   # ≤100: aggressive batching works well
    complexity_threshold_large = 900   # ≥900: batching hurts performance
    
    # Use the most conservative metric (maximum elements that model needs to handle)
    primary_metric = max_grid_elements
    secondary_metric = max(max_input_elements, max_output_elements)
    
    # Task complexity classification
    if primary_metric <= complexity_threshold_small:
        complexity_category = "small"
        recommended_batch_size = 4 if conservative else 8
        expected_speedup = 2.8 if not conservative else 1.8
        confidence = "high"
        reasoning = f"Small task: max grid {task.n_x}×{task.n_y}={primary_metric} elements. Batching provides excellent speedup."
        
    elif primary_metric <= complexity_threshold_large:
        complexity_category = "medium" 
        recommended_batch_size = 2 if conservative else 4
        expected_speedup = 1.7 if not conservative else 1.3
        confidence = "medium"
        reasoning = f"Medium task: max grid {task.n_x}×{task.n_y}={primary_metric} elements. Moderate batching beneficial."
        
    else:
        complexity_category = "large"
        recommended_batch_size = 1
        expected_speedup = 1.0
        confidence = "high"
        reasoning = f"Large task: max grid {task.n_x}×{task.n_y}={primary_metric} elements. Batching hurts performance due to memory bandwidth limits."
    
    # Additional considerations
    memory_estimate_mb = (primary_metric * 4 * 16 * recommended_batch_size) / (1024**2)  # Rough estimate
    
    # Conservative adjustments
    if memory_estimate_mb > 100:  # If estimated memory > 100MB, be more conservative
        if recommended_batch_size > 1:
            recommended_batch_size = max(1, recommended_batch_size // 2)
            reasoning += " Reduced due to high memory usage estimate."
            confidence = "medium"
    
    result = {
        "recommended_batch_size": recommended_batch_size,
        "expected_speedup": expected_speedup,
        "confidence": confidence,
        "complexity_category": complexity_category,
        "reasoning": reasoning,
        "task_stats": {
            "task_name": task.task_name,
            "max_grid_size": (task.n_x, task.n_y),
            "max_grid_elements": primary_metric,
            "avg_grid_elements": avg_grid_elements,
            "max_input_elements": max_input_elements,
            "max_output_elements": max_output_elements,
            "estimated_memory_mb": memory_estimate_mb,
            "n_examples": task.n_examples
        }
    }
    
    if verbose:
        print(f"📊 Task Analysis: {task.task_name}")
        print(f"   Grid size: {task.n_x}×{task.n_y} ({primary_metric} elements)")
        print(f"   Complexity: {complexity_category}")
        print(f"   ✅ Recommended batch_size: {recommended_batch_size}")
        print(f"   Expected speedup: {expected_speedup:.1f}x")
        print(f"   Confidence: {confidence}")
        print(f"   Reasoning: {reasoning}")
        print(f"   Estimated memory usage: {memory_estimate_mb:.1f} MB")
    
    return result


def get_batch_sizes_for_tasks(tasks: list, 
                             conservative: bool = False, 
                             verbose: bool = False) -> Dict:
    """
    Get optimal batch sizes for multiple tasks.
    
    Args:
        tasks: List of preprocessed task objects
        conservative: Use more conservative (safer) batch sizes
        verbose: Print detailed analysis for each task
        
    Returns:
        Dictionary mapping task names to batch size recommendations
    """
    
    recommendations = {}
    summary_stats = {
        "total_tasks": len(tasks),
        "small_tasks": 0,
        "medium_tasks": 0, 
        "large_tasks": 0,
        "batch_8_recommended": 0,
        "batch_4_recommended": 0,
        "batch_2_recommended": 0,
        "batch_1_recommended": 0
    }
    
    if verbose:
        print("🎯 Analyzing Multiple Tasks for Optimal Batch Sizes")
        print("=" * 60)
    
    for task in tasks:
        result = get_optimal_batch_size(task, conservative=conservative, verbose=verbose)
        recommendations[task.task_name] = result
        
        # Update summary statistics
        category = result["complexity_category"]
        batch_size = result["recommended_batch_size"]
        
        summary_stats[f"{category}_tasks"] += 1
        summary_stats[f"batch_{batch_size}_recommended"] += 1
        
        if verbose:
            print()  # Empty line between tasks
    
    if verbose:
        print("📈 SUMMARY ACROSS ALL TASKS")
        print("=" * 60)
        print(f"Small tasks (batch_size=4-8):   {summary_stats['small_tasks']}")
        print(f"Medium tasks (batch_size=2-4):  {summary_stats['medium_tasks']}")
        print(f"Large tasks (batch_size=1):     {summary_stats['large_tasks']}")
        print()
        print("Batch size distribution:")
        for batch_size in [8, 4, 2, 1]:
            count = summary_stats[f"batch_{batch_size}_recommended"]
            if count > 0:
                print(f"  batch_size={batch_size}: {count} tasks")
    
    return {
        "recommendations": recommendations,
        "summary": summary_stats
    }


def adaptive_batch_size_strategy(grid_elements: int) -> int:
    """
    Simple function to get batch size based on grid complexity.
    
    Args:
        grid_elements: Number of elements in the largest grid for the task
        
    Returns:
        Recommended batch size (int)
    """
    if grid_elements <= 100:
        return 8  # Small grids: aggressive batching
    elif grid_elements <= 500:
        return 4  # Medium grids: moderate batching
    elif grid_elements <= 900:
        return 2  # Large grids: conservative batching
    else:
        return 1  # Very large grids: no batching


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Adaptive Batch Size Recommendations")
    print("=" * 60)
    
    # Test with the same tasks used in analysis
    test_task_ids = [0, 37, 21]  # Small, medium, large
    
    try:
        tasks = preprocessing.preprocess_tasks('training', test_task_ids)
        
        print(f"Loaded {len(tasks)} test tasks")
        print()
        
        # Get recommendations for individual tasks
        for task in tasks:
            result = get_optimal_batch_size(task, verbose=True)
            print()
        
        print("=" * 60)
        
        # Get batch recommendations for all tasks
        all_results = get_batch_sizes_for_tasks(tasks, verbose=False)
        
        print("📋 Quick Reference:")
        for task_name, result in all_results["recommendations"].items():
            batch_size = result["recommended_batch_size"]
            category = result["complexity_category"]
            speedup = result["expected_speedup"]
            print(f"  {task_name}: batch_size={batch_size} ({category}, {speedup:.1f}x speedup)")
            
    except Exception as e:
        print(f"❌ Could not load test tasks: {e}")
        print("Testing with synthetic examples...")
        
        # Test the simple strategy function
        test_cases = [
            (36, "Small (6×6)"),
            (80, "Medium (10×8)"), 
            (900, "Large (30×30)"),
            (1600, "Very large (40×40)")
        ]
        
        print("\n🔧 Simple Strategy Function Tests:")
        for elements, description in test_cases:
            batch_size = adaptive_batch_size_strategy(elements)
            print(f"  {description} - {elements} elements → batch_size={batch_size}") 