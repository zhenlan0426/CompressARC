"""
Comprehensive Performance Evaluation for Batch vs Unbatched Layer Performance

This module provides a comprehensive benchmarking system that evaluates performance 
ratios between batched and unbatched implementations across different:
- Batch sizes (1, 2, 4, 8)  
- Problem sizes (small, medium, large tensors)
- Available layer implementations

Based on the findings in batch_parallelization_analysis.md, this helps determine
optimal batch sizes depending on problem characteristics.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from functools import partial
import warnings
import json

# Import existing modules  
import preprocessing
import multitensor_systems
import layers
import layers_batch
import initializers_batch
import my_test

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def evaluate_batch_performance(layer_functions: List[str] = None, 
                              problem_sizes: List[str] = None,
                              batch_sizes: List[int] = [1, 2, 4, 8],
                              iterations: int = 10) -> Dict:
    """
    Simple function to evaluate performance ratio between batched and unbatched implementations.
    
    Args:
        layer_functions: List of layer names to test (defaults to all available)  
        problem_sizes: List of problem sizes to test ['small', 'medium', 'large']
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8])
        iterations: Number of timing iterations for stable results
        
    Returns:
        Dictionary with performance results and optimal batch size recommendations
    """
    print("🎯 Simple Batch Performance Evaluation")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = BatchPerformanceEvaluator(warmup_iterations=2)
    evaluator.TIMING_ITERATIONS = iterations
    evaluator.BATCH_SIZES = batch_sizes
    
    if not evaluator.tasks:
        print("❌ No tasks loaded. Cannot proceed.")
        return {}
    
    # Filter available layers
    available_layers = evaluator.get_available_layers()
    if layer_functions:
        available_layers = {k: v for k, v in available_layers.items() if k in layer_functions}
    
    # Filter problem sizes  
    test_tasks = evaluator.tasks
    if problem_sizes:
        test_tasks = {k: v for k, v in test_tasks.items() if k in problem_sizes}
    
    print(f"📊 Testing {len(available_layers)} layers")
    print(f"📏 Testing {len(test_tasks)} problem sizes: {list(test_tasks.keys())}")
    print(f"🔄 Testing batch sizes: {batch_sizes}")
    
    # Run streamlined benchmarks
    results = {}
    total_times = {"unbatched": {}, "batched": {}}
    
    for layer_name, layer_info in available_layers.items():
        print(f"\n📊 {layer_name}")
        layer_results = {}
        
        for task_size in test_tasks.keys():
            task_results = {}
            print(f"  📏 {task_size}: ", end="")
            
            for batch_size in batch_sizes:
                result = evaluator.benchmark_layer_single_config(
                    layer_name, layer_info, task_size, batch_size
                )
                
                task_results[batch_size] = result
                
                if result.get('success'):
                    # Accumulate total times
                    if task_size not in total_times["unbatched"]:
                        total_times["unbatched"][task_size] = {}
                        total_times["batched"][task_size] = {}
                    if batch_size not in total_times["unbatched"][task_size]:
                        total_times["unbatched"][task_size][batch_size] = 0
                        total_times["batched"][task_size][batch_size] = 0
                        
                    total_times["unbatched"][task_size][batch_size] += result['unbatched_time']
                    total_times["batched"][task_size][batch_size] += result['batched_time']
                    
                    print(f"bs{batch_size}:{result['speedup']:.1f}x ", end="")
                else:
                    print(f"bs{batch_size}:❌ ", end="")
                    
            print()  # New line
            layer_results[task_size] = task_results
        results[layer_name] = layer_results
    
    # Calculate total time speedups and optimal batch sizes
    print("\n📈 SUMMARY - Total Time Across All Layers:")
    print("=" * 60)
    
    optimal_recommendations = {}
    for task_size in test_tasks.keys():
        print(f"\n🏷️  {task_size.upper()} Problems:")
        task_info = evaluator.TEST_TASKS[task_size]
        print(f"   Task {task_info['task_id']} (~{task_info['expected_elements']} elements/grid)")
        
        best_batch_size = 1
        best_speedup = 1.0
        
        for batch_size in batch_sizes:
            if (task_size in total_times["unbatched"] and 
                batch_size in total_times["unbatched"][task_size]):
                
                total_unbatched = total_times["unbatched"][task_size][batch_size]
                total_batched = total_times["batched"][task_size][batch_size] 
                
                if total_batched > 0:
                    total_speedup = total_unbatched / total_batched
                    efficiency = total_speedup / batch_size
                    
                    print(f"   Batch Size {batch_size}: {total_speedup:.2f}x speedup ({efficiency:.1%} efficiency)")
                    
                    if total_speedup > best_speedup:
                        best_speedup = total_speedup
                        best_batch_size = batch_size
        
        # Determine recommendation confidence
        expected_speedup = task_info["expected_speedup"]
        confidence = "high" if abs(best_speedup - expected_speedup) < 0.5 else "medium"
        if best_speedup < 1.0:
            confidence = "low"
        
        optimal_recommendations[task_size] = {
            "optimal_batch_size": best_batch_size,
            "speedup": best_speedup,
            "efficiency": best_speedup / best_batch_size,
            "confidence": confidence
        }
        
        recommendation = "✅" if best_speedup > 1.0 else "⚠️"
        print(f"   {recommendation} RECOMMENDATION: Use batch_size={best_batch_size} ({best_speedup:.2f}x speedup)")
    
    print("\n" + "=" * 60)
    print("🎯 OPTIMAL BATCH SIZES BY PROBLEM SIZE:")
    for task_size, rec in optimal_recommendations.items():
        print(f"   {task_size:>6}: batch_size={rec['optimal_batch_size']} "
              f"({rec['speedup']:.2f}x speedup, {rec['efficiency']:.1%} efficiency)")
    
    return {
        "results": results,
        "total_times": total_times,
        "recommendations": optimal_recommendations,
        "summary": {
            "layers_tested": len(available_layers),
            "problem_sizes_tested": len(test_tasks),
            "batch_sizes_tested": len(batch_sizes)
        }
    }


class BatchPerformanceEvaluator:
    """Comprehensive performance evaluator for batched vs unbatched layer implementations."""
    
    # Test tasks based on analysis findings
    TEST_TASKS = {
        "small": {"task_id": 0, "expected_elements": 36, "expected_speedup": 3.4},    # Task 0: 6×6 grids
        "medium": {"task_id": 37, "expected_elements": 80, "expected_speedup": 2.1},  # Task 37: 10×8 grids  
        "large": {"task_id": 21, "expected_elements": 900, "expected_speedup": 0.63}  # Task 21: 30×30 grids
    }
    
    # Batch sizes to test
    BATCH_SIZES = [1, 2, 4, 8]
    
    # Number of timing iterations for stable results
    TIMING_ITERATIONS = 10
    
    def __init__(self, warmup_iterations: int = 3):
        """
        Initialize the performance evaluator.
        
        Args:
            warmup_iterations: Number of warmup runs before timing
        """
        self.warmup_iterations = warmup_iterations
        self.results = {}
        
        # Load test tasks
        self.tasks = {}
        for size_name, task_info in self.TEST_TASKS.items():
            try:
                tasks = preprocessing.preprocess_tasks('training', [task_info['task_id']])
                self.tasks[size_name] = tasks[0] 
                print(f"✓ Loaded {size_name} task (Task {task_info['task_id']})")
            except Exception as e:
                print(f"✗ Failed to load {size_name} task: {e}")
                
    def calculate_tensor_stats(self, multitensor_system) -> Dict:
        """Calculate statistics about tensor sizes in a multitensor system."""
        stats = {
            "total_elements": 0,
            "total_tensors": 0,
            "largest_tensor_elements": 0,
            "tensor_shapes": []
        }
        
        for dims in multitensor_system:
            # Get representative shape (channel_dim=16 as typical)
            shape = multitensor_system.shape(dims, 16)
            elements = np.prod(shape)
            
            stats["total_elements"] += elements
            stats["total_tensors"] += 1
            stats["largest_tensor_elements"] = max(stats["largest_tensor_elements"], elements)
            stats["tensor_shapes"].append((dims, shape, elements))
            
        return stats
        
    def get_available_layers(self) -> Dict:
        """Get all available layers for benchmarking from the test registry."""
        return {
            name: info for name, info in my_test.LAYER_TEST_REGISTRY.items()
            if hasattr(info.get('batched'), '__call__') and hasattr(info.get('ref'), '__call__')
        }
    
    def time_function_execution(self, func, args, kwargs, iterations: int) -> float:
        """Time function execution with warmup and multiple iterations."""
        # Warmup
        for _ in range(self.warmup_iterations):
            try:
                _ = func(*args, **kwargs)
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
            except Exception:
                pass
                
        # Actual timing
        start_time = time.time()
        for _ in range(iterations):
            try:
                result = func(*args, **kwargs) 
                # Ensure CUDA operations complete
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
                # Force backward pass for complete timing
                if hasattr(result, 'multitensor_system'):
                    loss = sum(result[dims].sum() for dims in result.multitensor_system)
                    loss.backward()
            except Exception as e:
                print(f"Warning: Error during timing: {e}")
                return float('inf')
                
        end_time = time.time()
        return (end_time - start_time) / iterations
    
    def benchmark_layer_single_config(self, layer_name: str, layer_info: Dict, 
                                    task_size: str, batch_size: int) -> Dict:
        """Benchmark a single layer configuration."""
        task = self.tasks.get(task_size)
        if not task:
            return {"error": f"Task {task_size} not available"}
            
        try:
            # Generate test data
            if 'share' in layer_name and hasattr(my_test, 'generate_share_data_with_task'):
                # Special handling for share layers that need task context
                task_id = self.TEST_TASKS[task_size]['task_id']
                batch_weights = 'batched_weights' in layer_name
                batched_args, _ = my_test.generate_share_data_with_task(task_id, batch_size, batch_weights)
            else:
                # Standard data generation
                system = task.multitensor_system
                generate_fn = layer_info['generate']
                batched_args = generate_fn(system, batch_size)
                
            # Prepare arguments for both implementations
            ref_func = layer_info['ref']
            batched_func = layer_info['batched'] 
            kwargs = layer_info.get('kwargs', {})
            
            # Split batched args for unbatched execution
            per_batch_args = []
            for b in range(batch_size):
                batch_args = []
                for arg, is_batched in batched_args:
                    if is_batched:
                        batch_args.append(my_test.split_multitensor_batch(arg, batch_size)[b])
                    else:
                        batch_args.append(arg)
                per_batch_args.append(batch_args)
            
            # Time unbatched execution (loop over batch)
            def run_unbatched():
                results = []
                for args_b in per_batch_args:
                    result = ref_func(*args_b, **kwargs)
                    results.append(result)
                return results
                
            # Time batched execution  
            def run_batched():
                args = [arg for arg, _ in batched_args]
                return batched_func(*args, **kwargs)
            
            unbatched_time = self.time_function_execution(run_unbatched, [], {}, self.TIMING_ITERATIONS)
            batched_time = self.time_function_execution(run_batched, [], {}, self.TIMING_ITERATIONS)
            
            # Calculate metrics
            if batched_time > 0 and unbatched_time > 0:
                speedup = unbatched_time / batched_time
                efficiency = speedup / batch_size  # How much of theoretical speedup achieved
            else:
                speedup = 0.0
                efficiency = 0.0
                
            # Get tensor statistics
            first_arg = batched_args[0][0] if batched_args else None
            tensor_stats = self.calculate_tensor_stats(first_arg.multitensor_system) if first_arg else {}
            
            return {
                "unbatched_time": unbatched_time,
                "batched_time": batched_time, 
                "speedup": speedup,
                "efficiency": efficiency,
                "tensor_stats": tensor_stats,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def benchmark_all_layers(self) -> Dict:
        """Benchmark all available layers across all configurations."""
        available_layers = self.get_available_layers()
        print(f"\n🚀 Starting comprehensive benchmark of {len(available_layers)} layers...")
        print(f"   Testing batch sizes: {self.BATCH_SIZES}")
        print(f"   Testing problem sizes: {list(self.TEST_TASKS.keys())}")
        print(f"   Timing iterations: {self.TIMING_ITERATIONS}")
        
        results = {}
        
        for layer_name, layer_info in available_layers.items():
            print(f"\n📊 Benchmarking layer: {layer_name}")
            layer_results = {}
            
            for task_size in self.tasks.keys():
                task_results = {}
                print(f"  📏 Problem size: {task_size}")
                
                for batch_size in self.BATCH_SIZES:
                    print(f"    🔄 Batch size: {batch_size}...", end=" ")
                    
                    result = self.benchmark_layer_single_config(
                        layer_name, layer_info, task_size, batch_size
                    )
                    
                    task_results[batch_size] = result
                    
                    if result.get('success'):
                        speedup = result['speedup']
                        print(f"Speedup: {speedup:.2f}x")
                    else:
                        print(f"❌ {result.get('error', 'Unknown error')}")
                        
                layer_results[task_size] = task_results
            results[layer_name] = layer_results
            
        return results
    
    def find_optimal_batch_sizes(self, results: Dict) -> Dict:
        """Analyze results to find optimal batch sizes for different problem sizes."""
        optimal_configs = {}
        
        for task_size in self.TEST_TASKS.keys():
            optimal_configs[task_size] = {
                "recommended_batch_size": 1,
                "expected_speedup": 1.0,
                "confidence": "low",
                "reasoning": "No successful benchmarks"
            }
            
            # Collect speedups across all layers for this task size
            speedups_by_batch = {bs: [] for bs in self.BATCH_SIZES}
            
            for layer_name, layer_results in results.items():
                task_results = layer_results.get(task_size, {})
                for batch_size, result in task_results.items():
                    if result.get('success') and result.get('speedup', 0) > 0:
                        speedups_by_batch[batch_size].append(result['speedup'])
            
            # Find batch size with best average speedup
            avg_speedups = {}
            for batch_size, speedups in speedups_by_batch.items():
                if speedups:
                    avg_speedups[batch_size] = np.mean(speedups)
            
            if avg_speedups:
                best_batch_size = max(avg_speedups.keys(), key=lambda x: avg_speedups[x])
                best_speedup = avg_speedups[best_batch_size]
                
                # Determine confidence based on consistency and theoretical expectations
                expected = self.TEST_TASKS[task_size]["expected_speedup"]
                consistency = 1.0 - (np.std(speedups_by_batch[best_batch_size]) / best_speedup) if speedups_by_batch[best_batch_size] else 0
                theory_match = 1.0 - abs(best_speedup - expected) / max(best_speedup, expected)
                
                confidence_score = (consistency + theory_match) / 2
                if confidence_score > 0.8:
                    confidence = "high"
                elif confidence_score > 0.5:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                optimal_configs[task_size] = {
                    "recommended_batch_size": best_batch_size,
                    "expected_speedup": best_speedup,
                    "confidence": confidence,
                    "reasoning": f"Average {best_speedup:.2f}x speedup across {len(speedups_by_batch[best_batch_size])} layers"
                }
        
        return optimal_configs
    
    def generate_performance_report(self, results: Dict, optimal_configs: Dict) -> str:
        """Generate a comprehensive performance report."""
        report_lines = [
            "# Comprehensive Batch Performance Evaluation Report",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Summary statistics
        total_benchmarks = sum(
            len(task_results) * len(batch_results)
            for layer_results in results.values()
            for task_results in layer_results.values()  
            for batch_results in task_results.values()
        )
        
        successful_benchmarks = sum(
            1 for layer_results in results.values()
            for task_results in layer_results.values()
            for batch_results in task_results.values()
            for result in batch_results.values()
            if result.get('success')
        )
        
        report_lines.extend([
            f"- **Total Benchmarks**: {total_benchmarks}",
            f"- **Successful Benchmarks**: {successful_benchmarks} ({successful_benchmarks/total_benchmarks*100:.1f}%)",
            f"- **Layers Tested**: {len(results)}",
            f"- **Problem Sizes**: {len(self.TEST_TASKS)}",
            f"- **Batch Sizes**: {len(self.BATCH_SIZES)}",
            "",
            "## Optimal Batch Size Recommendations",
            ""
        ])
        
        for task_size, config in optimal_configs.items():
            task_info = self.TEST_TASKS[task_size]
            report_lines.extend([
                f"### {task_size.title()} Problems (Task {task_info['task_id']}, ~{task_info['expected_elements']} elements/grid)",
                f"- **Recommended Batch Size**: {config['recommended_batch_size']}",
                f"- **Expected Speedup**: {config['expected_speedup']:.2f}x",
                f"- **Confidence**: {config['confidence']}",  
                f"- **Reasoning**: {config['reasoning']}",
                ""
            ])
        
        # Layer-specific results
        report_lines.extend([
            "## Detailed Results by Layer",
            ""
        ])
        
        for layer_name, layer_results in results.items():
            report_lines.append(f"### {layer_name}")
            report_lines.append("")
            
            # Create results table
            report_lines.extend([
                "| Problem Size | Batch Size | Speedup | Efficiency | Status |",
                "|--------------|------------|---------|------------|--------|"
            ])
            
            for task_size, task_results in layer_results.items():
                for batch_size, result in task_results.items():
                    if result.get('success'):
                        speedup = result['speedup']
                        efficiency = result['efficiency']
                        status = "✅"
                        report_lines.append(
                            f"| {task_size} | {batch_size} | {speedup:.2f}x | {efficiency:.1%} | {status} |"
                        )
                    else:
                        error = result.get('error', 'Unknown')[:20] + "..." if len(result.get('error', '')) > 20 else result.get('error', 'Unknown')
                        report_lines.append(
                            f"| {task_size} | {batch_size} | - | - | ❌ {error} |"
                        )
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_results(self, results: Dict, optimal_configs: Dict, 
                    filename: str = "batch_performance_results.json"):
        """Save benchmark results to JSON file."""
        output = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "batch_sizes_tested": self.BATCH_SIZES,
                "timing_iterations": self.TIMING_ITERATIONS,
                "warmup_iterations": self.warmup_iterations
            },
            "task_info": self.TEST_TASKS,
            "results": results,
            "optimal_configs": optimal_configs
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}")


def run_comprehensive_benchmark(save_report: bool = True, 
                               report_filename: str = "batch_performance_report.md",
                               results_filename: str = "batch_performance_results.json") -> Dict:
    """
    Run comprehensive batch performance evaluation.
    
    Args:
        save_report: Whether to save the performance report to a markdown file
        report_filename: Filename for the performance report  
        results_filename: Filename for the raw results JSON
        
    Returns:
        Dictionary containing all benchmark results and optimal configurations
    """
    print("🎯 Initializing Comprehensive Batch Performance Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = BatchPerformanceEvaluator()
    
    if not evaluator.tasks:
        print("❌ No tasks loaded successfully. Cannot proceed with benchmark.")
        return {}
    
    # Run benchmarks  
    results = evaluator.benchmark_all_layers()
    
    # Find optimal configurations
    print("\n🧠 Analyzing results to find optimal batch sizes...")
    optimal_configs = evaluator.find_optimal_batch_sizes(results)
    
    # Generate report
    print("\n📋 Generating performance report...")
    report = evaluator.generate_performance_report(results, optimal_configs)
    
    # Save results
    evaluator.save_results(results, optimal_configs, results_filename)
    
    if save_report:
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to {report_filename}")
    
    # Print summary
    print("\n" + "=" * 60)  
    print("🎉 BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"📊 Benchmarked {len(results)} layers across {len(evaluator.tasks)} problem sizes")
    print(f"🎯 Optimal configurations:")
    
    for task_size, config in optimal_configs.items():
        print(f"   {task_size:>6} problems: batch_size={config['recommended_batch_size']} "
              f"({config['expected_speedup']:.2f}x speedup, {config['confidence']} confidence)")
    
    return {
        "results": results,
        "optimal_configs": optimal_configs,
        "report": report
    }


if __name__ == "__main__":
    # Example: Simple evaluation (what the user requested)
    print("🚀 Running Simple Batch Performance Evaluation")
    simple_results = evaluate_batch_performance(
        batch_sizes=[1, 2, 4, 8],
        iterations=5  # Fewer iterations for quick testing
    )
    
    # Example: Full comprehensive benchmark 
    # comprehensive_results = run_comprehensive_benchmark() 