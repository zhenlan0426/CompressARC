"""
Example: Using Adaptive Batch Sizing in Your Training Pipeline

This example demonstrates how to integrate the adaptive batch size recommendations
into your existing training workflow.
"""

import preprocessing
from adaptive_batch_size import get_optimal_batch_size, get_batch_sizes_for_tasks


def example_single_task():
    """Example: Get optimal batch size for a single task"""
    print("🎯 Example 1: Single Task Analysis")
    print("=" * 50)
    
    # Load a task (replace with your actual task loading)
    tasks = preprocessing.preprocess_tasks('training', [0])  # Load task 0
    task = tasks[0]
    
    # Get recommendation
    recommendation = get_optimal_batch_size(task, verbose=True)
    
    # Use the recommendation
    optimal_batch_size = recommendation['recommended_batch_size']
    expected_speedup = recommendation['expected_speedup']
    
    print(f"\n💡 Integration: Use batch_size={optimal_batch_size} for {expected_speedup:.1f}x speedup")
    return optimal_batch_size


def example_multiple_tasks():
    """Example: Analyze multiple tasks for batch planning"""
    print("\n🎯 Example 2: Multiple Task Analysis")
    print("=" * 50)
    
    # Load multiple tasks
    task_ids = [0, 37, 21, 150, 200]  # Mix of different complexity tasks
    try:
        tasks = preprocessing.preprocess_tasks('training', task_ids)
        print(f"✅ Loaded {len(tasks)} tasks")
        
        # Get recommendations for all tasks
        results = get_batch_sizes_for_tasks(tasks, verbose=False)
        
        print("\n📊 Batch Size Distribution:")
        summary = results['summary']
        for batch_size in [8, 4, 2, 1]:
            count = summary.get(f'batch_{batch_size}_recommended', 0)
            if count > 0:
                percentage = (count / summary['total_tasks']) * 100
                print(f"   batch_size={batch_size}: {count}/{summary['total_tasks']} tasks ({percentage:.0f}%)")
        
        # Show detailed recommendations
        print("\n📋 Task-Specific Recommendations:")
        for task_name, rec in results['recommendations'].items():
            batch_size = rec['recommended_batch_size']
            category = rec['complexity_category']
            grid_size = rec['task_stats']['max_grid_size']
            print(f"   {task_name}: batch_size={batch_size} ({category}, {grid_size[0]}×{grid_size[1]} grid)")
            
        return results
        
    except Exception as e:
        print(f"❌ Could not load some tasks: {e}")
        return None


def example_integration_training_loop():
    """Example: How to integrate into your training pipeline"""
    print("\n🎯 Example 3: Training Pipeline Integration")
    print("=" * 50)
    
    # Simulated training loop setup
    def train_with_adaptive_batching(task_list):
        """Simulated training function with adaptive batching"""
        
        for task in task_list:
            # Get optimal batch size for this task
            recommendation = get_optimal_batch_size(task, conservative=False)
            batch_size = recommendation['recommended_batch_size']
            expected_speedup = recommendation['expected_speedup']
            
            print(f"📚 Training {task.task_name}:")
            print(f"   Grid: {task.n_x}×{task.n_y} → batch_size={batch_size} ({expected_speedup:.1f}x speedup)")
            
            # Here you would create your model with the appropriate batch size
            # model = create_batched_model(task, batch_size=batch_size)
            # train_model(model, task, batch_size=batch_size)
            
            # Simulated training result
            simulated_training_time = 100.0 / expected_speedup  # Base time divided by speedup
            print(f"   ⚡ Simulated training time: {simulated_training_time:.1f}s")
            print()
    
    # Demo with a few tasks
    try:
        tasks = preprocessing.preprocess_tasks('training', [0, 37, 21])
        train_with_adaptive_batching(tasks)
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def example_conservative_vs_aggressive():
    """Example: Conservative vs Aggressive batching strategies"""
    print("\n🎯 Example 4: Conservative vs Aggressive Strategies")  
    print("=" * 50)
    
    try:
        tasks = preprocessing.preprocess_tasks('training', [0, 37])  # Small and medium task
        
        for task in tasks:
            print(f"\n📊 Task {task.task_name} ({task.n_x}×{task.n_y} grid):")
            
            # Aggressive strategy (default)
            aggressive = get_optimal_batch_size(task, conservative=False)
            print(f"   🚀 Aggressive: batch_size={aggressive['recommended_batch_size']} ({aggressive['expected_speedup']:.1f}x speedup)")
            
            # Conservative strategy  
            conservative = get_optimal_batch_size(task, conservative=True)
            print(f"   🛡️  Conservative: batch_size={conservative['recommended_batch_size']} ({conservative['expected_speedup']:.1f}x speedup)")
            
            # Recommendation
            if aggressive['recommended_batch_size'] > conservative['recommended_batch_size']:
                print(f"   💡 Use aggressive for max speed, conservative for stability")
            else:
                print(f"   💡 Both strategies agree: use batch_size={aggressive['recommended_batch_size']}")
                
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    print("🚀 Adaptive Batch Sizing Examples")
    print("=" * 60)
    
    # Run all examples
    example_single_task()
    example_multiple_tasks() 
    example_integration_training_loop()
    example_conservative_vs_aggressive()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("💡 Key takeaway: Always analyze your tasks first, then adapt batch sizes accordingly.")
    print("   Small grids (≤100 elements): Use batch_size=8")
    print("   Medium grids (100-900 elements): Use batch_size=4") 
    print("   Large grids (>900 elements): Use batch_size=1") 