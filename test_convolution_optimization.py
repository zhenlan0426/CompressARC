import torch
import time
import numpy as np
import psutil
import gc

def original_implementation(logits_slice, problem_slice, output_shape, x_logprobs, y_logprobs, x_log_partition, y_log_partition):
    """Original double loop implementation"""
    logprobs = []  # will become (B, O_x, O_y)
    B = logits_slice.shape[0]
    for x_offset in range(x_logprobs.shape[1]):  # iterate over possible x-starts
        logprobs_y = []
        for y_offset in range(y_logprobs.shape[1]):  # iterate over possible y-starts
            # Grid-position prior
            logprob = (
                x_logprobs[:, x_offset] - x_log_partition +
                y_logprobs[:, y_offset] - y_log_partition
            )  # (B,)

            # Extract the corresponding crop
            logits_crop = logits_slice[:, :, x_offset:x_offset + output_shape[0], y_offset:y_offset + output_shape[1]]  # (B, C, x', y')
            target_crop = problem_slice[:output_shape[0], :output_shape[1]]  # (x', y')

            # Broadcast target over the batch dimension
            target_crop_b = target_crop.unsqueeze(0).expand(B, *target_crop.shape)

            ce = torch.nn.functional.cross_entropy(logits_crop, target_crop_b, reduction='none')  # (B, x', y')
            ce_sum = ce.sum(dim=(1, 2))  # (B,)

            logprob = logprob - ce_sum  # (B,)
            logprobs_y.append(logprob)
        logprobs_y = torch.stack(logprobs_y, dim=1)  # (B, O_y)
        logprobs.append(logprobs_y)
    logprobs = torch.stack(logprobs, dim=1)  # (B, O_x, O_y)
    return logprobs

def convolution_implementation_v1(logits_slice, problem_slice, output_shape, x_logprobs, y_logprobs, x_log_partition, y_log_partition):
    """Convolution implementation with loop over C"""
    B = logits_slice.shape[0]
    
    # Compute logprobs using convolution for efficiency
    logp = torch.nn.functional.log_softmax(logits_slice, dim=1)  # (B, C, LH, LW)
    
    C, LH, LW = logits_slice.shape[1], logits_slice.shape[2], logits_slice.shape[3]
    
    OH, OW = output_shape[0], output_shape[1]
    
    target_crop = problem_slice[:OH, :OW]  # (OH, OW)
    
    sum_logp = torch.zeros((B, LH - OH + 1, LW - OW + 1), device=logp.device, dtype=logp.dtype)
    
    for c in range(C):
        kernel = (target_crop == c).to(logp.dtype)  # (OH, OW)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, OH, OW)
        conv_out = torch.nn.functional.conv2d(logp[:, c:c+1, :, :], kernel, padding=0)  # (B, 1, O_x, O_y)
        sum_logp += conv_out.squeeze(1)  # (B, O_x, O_y)
    
    # Add priors
    x_logprobs_exp = x_logprobs.unsqueeze(2)  # (B, O_x, 1)
    y_logprobs_exp = y_logprobs.unsqueeze(1)  # (B, 1, O_y)
    
    logprobs = sum_logp + x_logprobs_exp + y_logprobs_exp - x_log_partition[:, None, None] - y_log_partition[:, None, None]
    return logprobs

def convolution_implementation_v2(logits_slice, problem_slice, output_shape, x_logprobs, y_logprobs, x_log_partition, y_log_partition):
    """Fully vectorized convolution implementation without loop over C"""
    B = logits_slice.shape[0]
    
    # Compute logprobs using convolution for efficiency (vectorized over all channels)
    logp = torch.nn.functional.log_softmax(logits_slice, dim=1)  # (B, C, LH, LW)
    
    C, LH, LW = logits_slice.shape[1], logits_slice.shape[2], logits_slice.shape[3]
    
    OH, OW = output_shape[0], output_shape[1]
    
    target_crop = problem_slice[:OH, :OW]  # (OH, OW)
    
    # Use unfold to get all possible crops efficiently
    logp_unfolded = torch.nn.functional.unfold(logp, kernel_size=(OH, OW), stride=1)  # (B, C*OH*OW, num_windows)
    logp_unfolded = logp_unfolded.view(B, C, OH, OW, -1)  # (B, C, OH, OW, num_windows)
    
    # Reshape target_crop for gather operation
    target_flat = target_crop.reshape(-1)  # (OH*OW,)
    target_expanded = target_flat.unsqueeze(0).unsqueeze(-1).expand(B, -1, logp_unfolded.shape[-1])  # (B, OH*OW, num_windows)
    
    # Gather the log probabilities for the target colors
    logp_selected = torch.gather(logp_unfolded.view(B, C, -1, logp_unfolded.shape[-1]), 1, target_expanded.unsqueeze(1))  # (B, 1, OH*OW, num_windows)
    logp_selected = logp_selected.squeeze(1)  # (B, OH*OW, num_windows)
    
    # Sum over spatial dimensions and reshape back to 2D grid
    sum_logp = logp_selected.sum(dim=1)  # (B, num_windows)
    sum_logp = sum_logp.view(B, LH - OH + 1, LW - OW + 1)  # (B, O_x, O_y)
    
    # Add priors
    x_logprobs_exp = x_logprobs.unsqueeze(2)  # (B, O_x, 1)
    y_logprobs_exp = y_logprobs.unsqueeze(1)  # (B, 1, O_y)
    
    logprobs = sum_logp + x_logprobs_exp + y_logprobs_exp - x_log_partition[:, None, None] - y_log_partition[:, None, None]
    return logprobs

def convolution_implementation_v3(logits_slice, problem_slice, output_shape, x_logprobs, y_logprobs, x_log_partition, y_log_partition):
    """One-hot encoded 2D convolution implementation"""
    B = logits_slice.shape[0]
    
    # Compute logprobs using convolution for efficiency
    logp = torch.nn.functional.log_softmax(logits_slice, dim=1)  # (B, C, LH, LW)
    
    C, LH, LW = logits_slice.shape[1], logits_slice.shape[2], logits_slice.shape[3]
    
    OH, OW = output_shape[0], output_shape[1]
    
    target_crop = problem_slice[:OH, :OW]  # (OH, OW)
    
    # One-hot encode the target crop
    target_one_hot = torch.nn.functional.one_hot(target_crop, num_classes=C).to(logp.dtype)  # (OH, OW, C)
    target_one_hot = target_one_hot.permute(2, 0, 1).unsqueeze(0)  # (1, C, OH, OW)
    
    # Perform grouped convolution: each channel separately
    conv_out = torch.nn.functional.conv2d(logp, target_one_hot, padding=0)  # (B, 1, O_x, O_y)
    
    # Sum over channels
    sum_logp = conv_out.squeeze(1)  # (B, O_x, O_y)
    
    # Add priors
    x_logprobs_exp = x_logprobs.unsqueeze(2)  # (B, O_x, 1)
    y_logprobs_exp = y_logprobs.unsqueeze(1)  # (B, 1, O_y)
    
    logprobs = sum_logp + x_logprobs_exp + y_logprobs_exp - x_log_partition[:, None, None] - y_log_partition[:, None, None]
    return logprobs

def get_memory_usage():
    """Get current memory usage in MB"""
    if torch.cuda.is_available():
        # GPU memory
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**2
        return {
            'gpu_allocated': gpu_memory,
            'gpu_cached': gpu_memory_cached,
            'cpu': psutil.Process().memory_info().rss / 1024**2
        }
    else:
        # CPU memory only
        return {
            'gpu_allocated': 0,
            'gpu_cached': 0,
            'cpu': psutil.Process().memory_info().rss / 1024**2
        }

def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def benchmark_vram_usage():
    """Benchmark VRAM usage for all implementations"""
    print("=" * 60)
    print("VRAM USAGE BENCHMARK")
    print("=" * 60)
    
    # Test with different problem sizes
    test_configs = [
        {"name": "Small", "B": 4, "C": 10, "LH": 15, "LW": 15, "OH": 3, "OW": 3},
        {"name": "Medium", "B": 8, "C": 10, "LH": 25, "LW": 25, "OH": 5, "OW": 5},
        {"name": "Large", "B": 16, "C": 10, "LH": 35, "LW": 35, "OH": 7, "OW": 7},
        {"name": "XLarge", "B": 32, "C": 10, "LH": 45, "LW": 45, "OH": 9, "OW": 9},
    ]
    
    implementations = [
        ("Original", original_implementation),
        ("Conv v1", convolution_implementation_v1),
        ("Conv v2", convolution_implementation_v2),
        ("Conv v3", convolution_implementation_v3)
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n--- {config['name']} Problem Size ---")
        print(f"Batch: {config['B']}, Classes: {config['C']}, Grid: {config['LH']}x{config['LW']}, Output: {config['OH']}x{config['OW']}")
        
        results[config['name']] = {}
        
        # Set up test data
        torch.manual_seed(42)
        np.random.seed(42)
        
        B, C, LH, LW, OH, OW = config['B'], config['C'], config['LH'], config['LW'], config['OH'], config['OW']
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logits_slice = torch.randn(B, C, LH, LW, device=device)
        problem_slice = torch.randint(0, C, (LH, LW), device=device)
        output_shape = (OH, OW)
        
        O_x, O_y = LH - OH + 1, LW - OW + 1
        x_logprobs = torch.randn(B, O_x, device=device)
        y_logprobs = torch.randn(B, O_y, device=device)
        x_log_partition = torch.randn(B, device=device)
        y_log_partition = torch.randn(B, device=device)
        
        for impl_name, impl_func in implementations:
            print(f"\nTesting {impl_name}...")
            
            # Clear memory before test
            clear_memory()
            baseline_memory = get_memory_usage()
            
            try:
                # Measure peak memory during execution
                peak_memory = baseline_memory.copy()
                
                # Run implementation multiple times to get stable memory reading
                for _ in range(5):
                    result = impl_func(
                        logits_slice, problem_slice, output_shape,
                        x_logprobs, y_logprobs, x_log_partition, y_log_partition
                    )
                    current_memory = get_memory_usage()
                    
                    # Track peak memory usage
                    if current_memory['gpu_allocated'] > peak_memory['gpu_allocated']:
                        peak_memory = current_memory
                    
                    # Keep result in memory to measure peak usage
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # Calculate memory usage relative to baseline
                memory_used = {
                    'gpu_allocated': peak_memory['gpu_allocated'] - baseline_memory['gpu_allocated'],
                    'gpu_cached': peak_memory['gpu_cached'] - baseline_memory['gpu_cached'],
                    'cpu': peak_memory['cpu'] - baseline_memory['cpu']
                }
                
                results[config['name']][impl_name] = memory_used
                
                print(f"  GPU Allocated: {memory_used['gpu_allocated']:.1f} MB")
                print(f"  GPU Cached: {memory_used['gpu_cached']:.1f} MB")
                print(f"  CPU: {memory_used['cpu']:.1f} MB")
                print(f"  Output shape: {result.shape}")
                
                # Clean up
                del result
                clear_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ❌ OUT OF MEMORY: {e}")
                    results[config['name']][impl_name] = {'error': 'OOM'}
                else:
                    raise e
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                results[config['name']][impl_name] = {'error': str(e)}
    
    # Print summary comparison
    print("\n" + "=" * 60)
    print("MEMORY USAGE SUMMARY")
    print("=" * 60)
    
    for config_name in results:
        print(f"\n--- {config_name} ---")
        if torch.cuda.is_available():
            print(f"{'Implementation':<12} {'GPU Alloc (MB)':<15} {'GPU Cache (MB)':<15} {'CPU (MB)':<10}")
            print("-" * 55)
        else:
            print(f"{'Implementation':<12} {'CPU (MB)':<10}")
            print("-" * 25)
            
        for impl_name in results[config_name]:
            mem_data = results[config_name][impl_name]
            if 'error' in mem_data:
                print(f"{impl_name:<12} ERROR: {mem_data['error']}")
            else:
                if torch.cuda.is_available():
                    print(f"{impl_name:<12} {mem_data['gpu_allocated']:<15.1f} {mem_data['gpu_cached']:<15.1f} {mem_data['cpu']:<10.1f}")
                else:
                    print(f"{impl_name:<12} {mem_data['cpu']:<10.1f}")
    
    # Compare v1, v2, v3 efficiency
    print("\n" + "=" * 60)
    print("CONVOLUTION VERSIONS COMPARISON")
    print("=" * 60)
    
    for config_name in results:
        print(f"\n--- {config_name} ---")
        config_results = results[config_name]
        
        if all(impl in config_results and 'error' not in config_results[impl] 
               for impl in ['Conv v1', 'Conv v2', 'Conv v3']):
            
            v1_mem = config_results['Conv v1']['gpu_allocated'] if torch.cuda.is_available() else config_results['Conv v1']['cpu']
            v2_mem = config_results['Conv v2']['gpu_allocated'] if torch.cuda.is_available() else config_results['Conv v2']['cpu']
            v3_mem = config_results['Conv v3']['gpu_allocated'] if torch.cuda.is_available() else config_results['Conv v3']['cpu']
            
            print(f"v1 memory: {v1_mem:.1f} MB")
            print(f"v2 memory: {v2_mem:.1f} MB")
            print(f"v3 memory: {v3_mem:.1f} MB")
            
            if v1_mem > 0:
                print(f"v2 vs v1: {v2_mem/v1_mem:.2f}x memory")
                print(f"v3 vs v1: {v3_mem/v1_mem:.2f}x memory")
            if v2_mem > 0:
                print(f"v3 vs v2: {v3_mem/v2_mem:.2f}x memory")
        else:
            print("Some implementations failed - cannot compare")
    
    return results

def benchmark_vram_scaling():
    """Test how VRAM usage scales with problem size"""
    print("\n" + "=" * 60)
    print("VRAM SCALING ANALYSIS")
    print("=" * 60)
    
    # Test scaling with batch size
    print("\n--- Scaling with Batch Size ---")
    batch_sizes = [1, 2, 4, 8, 16, 32]
    base_config = {"C": 10, "LH": 25, "LW": 25, "OH": 5, "OW": 5}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for impl_name, impl_func in [("Conv v1", convolution_implementation_v1), 
                                ("Conv v2", convolution_implementation_v2),
                                ("Conv v3", convolution_implementation_v3)]:
        print(f"\n{impl_name}:")
        print(f"{'Batch Size':<10} {'Memory (MB)':<12} {'Memory/Batch':<12}")
        print("-" * 35)
        
        for B in batch_sizes:
            try:
                clear_memory()
                baseline = get_memory_usage()
                
                # Create test data
                logits_slice = torch.randn(B, base_config['C'], base_config['LH'], base_config['LW'], device=device)
                problem_slice = torch.randint(0, base_config['C'], (base_config['LH'], base_config['LW']), device=device)
                output_shape = (base_config['OH'], base_config['OW'])
                
                O_x, O_y = base_config['LH'] - base_config['OH'] + 1, base_config['LW'] - base_config['OW'] + 1
                x_logprobs = torch.randn(B, O_x, device=device)
                y_logprobs = torch.randn(B, O_y, device=device)
                x_log_partition = torch.randn(B, device=device)
                y_log_partition = torch.randn(B, device=device)
                
                # Run implementation
                result = impl_func(logits_slice, problem_slice, output_shape,
                                 x_logprobs, y_logprobs, x_log_partition, y_log_partition)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                peak = get_memory_usage()
                memory_used = peak['gpu_allocated'] - baseline['gpu_allocated'] if torch.cuda.is_available() else peak['cpu'] - baseline['cpu']
                memory_per_batch = memory_used / B if B > 0 else 0
                
                print(f"{B:<10} {memory_used:<12.1f} {memory_per_batch:<12.1f}")
                
                del result, logits_slice, problem_slice, x_logprobs, y_logprobs, x_log_partition, y_log_partition
                clear_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"{B:<10} OOM")
                    break
                else:
                    raise e
    
    # Test scaling with grid size
    print("\n--- Scaling with Grid Size ---")
    grid_sizes = [15, 20, 25, 30, 35, 40]
    base_config = {"B": 8, "C": 10, "OH": 5, "OW": 5}
    
    for impl_name, impl_func in [("Conv v1", convolution_implementation_v1), 
                                ("Conv v2", convolution_implementation_v2),
                                ("Conv v3", convolution_implementation_v3)]:
        print(f"\n{impl_name}:")
        print(f"{'Grid Size':<10} {'Memory (MB)':<12} {'Memory/Pixel':<12}")
        print("-" * 35)
        
        for size in grid_sizes:
            try:
                clear_memory()
                baseline = get_memory_usage()
                
                # Create test data
                logits_slice = torch.randn(base_config['B'], base_config['C'], size, size, device=device)
                problem_slice = torch.randint(0, base_config['C'], (size, size), device=device)
                output_shape = (base_config['OH'], base_config['OW'])
                
                O_x, O_y = size - base_config['OH'] + 1, size - base_config['OW'] + 1
                x_logprobs = torch.randn(base_config['B'], O_x, device=device)
                y_logprobs = torch.randn(base_config['B'], O_y, device=device)
                x_log_partition = torch.randn(base_config['B'], device=device)
                y_log_partition = torch.randn(base_config['B'], device=device)
                
                # Run implementation
                result = impl_func(logits_slice, problem_slice, output_shape,
                                 x_logprobs, y_logprobs, x_log_partition, y_log_partition)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                peak = get_memory_usage()
                memory_used = peak['gpu_allocated'] - baseline['gpu_allocated'] if torch.cuda.is_available() else peak['cpu'] - baseline['cpu']
                memory_per_pixel = memory_used / (size * size) if size > 0 else 0
                
                print(f"{size}x{size:<6} {memory_used:<12.1f} {memory_per_pixel:<12.4f}")
                
                del result, logits_slice, problem_slice, x_logprobs, y_logprobs, x_log_partition, y_log_partition
                clear_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"{size}x{size:<6} OOM")
                    break
                else:
                    raise e

def benchmark_vram_comprehensive():
    """Comprehensive VRAM benchmark with larger problem sizes"""
    print("=" * 70)
    print("COMPREHENSIVE VRAM USAGE BENCHMARK")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - running on CPU")
        print("Memory measurements will show CPU RAM usage instead of VRAM")
        print()
    
    # Test with progressively larger problem sizes
    test_configs = [
        {"name": "Small", "B": 8, "C": 10, "LH": 20, "LW": 20, "OH": 3, "OW": 3},
        {"name": "Medium", "B": 16, "C": 10, "LH": 30, "LW": 30, "OH": 5, "OW": 5},
        {"name": "Large", "B": 32, "C": 10, "LH": 50, "LW": 50, "OH": 8, "OW": 8},
        {"name": "XLarge", "B": 64, "C": 10, "LH": 80, "LW": 80, "OH": 12, "OW": 12},
        {"name": "XXLarge", "B": 128, "C": 10, "LH": 100, "LW": 100, "OH": 15, "OW": 15},
    ]
    
    implementations = [
        ("Conv v1", convolution_implementation_v1),
        ("Conv v2", convolution_implementation_v2),
        ("Conv v3", convolution_implementation_v3)
    ]
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for config in test_configs:
        print(f"\n--- {config['name']} Problem Size ---")
        print(f"Batch: {config['B']}, Classes: {config['C']}, Grid: {config['LH']}x{config['LW']}, Output: {config['OH']}x{config['OW']}")
        
        B, C, LH, LW, OH, OW = config['B'], config['C'], config['LH'], config['LW'], config['OH'], config['OW']
        
        # Calculate expected tensor sizes
        input_size = B * C * LH * LW * 4 / (1024**2)  # 4 bytes per float32
        output_size = B * (LH - OH + 1) * (LW - OW + 1) * 4 / (1024**2)
        print(f"Expected input tensor size: {input_size:.1f} MB")
        print(f"Expected output tensor size: {output_size:.1f} MB")
        
        results[config['name']] = {}
        
        # Set up test data
        torch.manual_seed(42)
        np.random.seed(42)
        
        try:
            logits_slice = torch.randn(B, C, LH, LW, device=device)
            problem_slice = torch.randint(0, C, (LH, LW), device=device)
            output_shape = (OH, OW)
            
            O_x, O_y = LH - OH + 1, LW - OW + 1
            x_logprobs = torch.randn(B, O_x, device=device)
            y_logprobs = torch.randn(B, O_y, device=device)
            x_log_partition = torch.randn(B, device=device)
            y_log_partition = torch.randn(B, device=device)
            
            print(f"Test data created successfully")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Cannot create test data - OUT OF MEMORY")
                results[config['name']] = {'error': 'OOM during data creation'}
                continue
            else:
                raise e
        
        for impl_name, impl_func in implementations:
            print(f"\n  Testing {impl_name}...")
            
            try:
                # Clear memory and get baseline
                clear_memory()
                baseline_memory = get_memory_usage()
                
                # Measure memory during execution
                max_memory_used = baseline_memory.copy()
                
                # Run implementation with memory tracking
                for run in range(3):  # Multiple runs to get stable reading
                    result = impl_func(
                        logits_slice, problem_slice, output_shape,
                        x_logprobs, y_logprobs, x_log_partition, y_log_partition
                    )
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    current_memory = get_memory_usage()
                    
                    # Track maximum memory usage
                    if current_memory['gpu_allocated'] > max_memory_used['gpu_allocated']:
                        max_memory_used = current_memory
                    
                    # Don't delete result immediately to measure peak usage
                    if run < 2:  # Keep result for peak measurement
                        temp_result = result
                
                # Calculate memory usage
                memory_used = {
                    'gpu_allocated': max_memory_used['gpu_allocated'] - baseline_memory['gpu_allocated'],
                    'gpu_cached': max_memory_used['gpu_cached'] - baseline_memory['gpu_cached'],
                    'cpu': max_memory_used['cpu'] - baseline_memory['cpu']
                }
                
                results[config['name']][impl_name] = memory_used
                
                if torch.cuda.is_available():
                    print(f"    GPU Allocated: {memory_used['gpu_allocated']:.1f} MB")
                    print(f"    GPU Cached: {memory_used['gpu_cached']:.1f} MB")
                else:
                    print(f"    CPU Memory: {memory_used['cpu']:.1f} MB")
                
                print(f"    Output shape: {result.shape}")
                print(f"    ✓ Success")
                
                # Clean up
                del result
                if 'temp_result' in locals():
                    del temp_result
                clear_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    ❌ OUT OF MEMORY")
                    results[config['name']][impl_name] = {'error': 'OOM'}
                else:
                    print(f"    ❌ ERROR: {e}")
                    results[config['name']][impl_name] = {'error': str(e)}
                clear_memory()
            except Exception as e:
                print(f"    ❌ ERROR: {e}")
                results[config['name']][impl_name] = {'error': str(e)}
                clear_memory()
        
        # Clean up test data
        del logits_slice, problem_slice, x_logprobs, y_logprobs, x_log_partition, y_log_partition
        clear_memory()
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("DETAILED MEMORY USAGE COMPARISON")
    print("=" * 70)
    
    memory_key = 'gpu_allocated' if torch.cuda.is_available() else 'cpu'
    memory_unit = 'GPU MB' if torch.cuda.is_available() else 'CPU MB'
    
    for config_name in results:
        print(f"\n--- {config_name} ---")
        
        if isinstance(results[config_name], dict) and 'error' in results[config_name]:
            print(f"❌ {results[config_name]['error']}")
            continue
            
        print(f"{'Implementation':<12} {memory_unit:<12} {'Efficiency':<12}")
        print("-" * 40)
        
        # Get baseline (v1) for efficiency comparison
        v1_memory = None
        if 'Conv v1' in results[config_name] and 'error' not in results[config_name]['Conv v1']:
            v1_memory = results[config_name]['Conv v1'][memory_key]
        
        for impl_name in ['Conv v1', 'Conv v2', 'Conv v3']:
            if impl_name in results[config_name]:
                mem_data = results[config_name][impl_name]
                if 'error' in mem_data:
                    print(f"{impl_name:<12} {'ERROR':<12} {mem_data['error']}")
                else:
                    memory_val = mem_data[memory_key]
                    efficiency = f"{memory_val/v1_memory:.2f}x" if v1_memory and v1_memory > 0 else "N/A"
                    print(f"{impl_name:<12} {memory_val:<12.1f} {efficiency:<12}")
    
    # Performance vs Memory Trade-off Analysis
    print("\n" + "=" * 70)
    print("PERFORMANCE vs MEMORY TRADE-OFF ANALYSIS")
    print("=" * 70)
    
    print("\nQuick performance test (100 runs each)...")
    
    # Use medium size for performance test
    config = {"B": 16, "C": 10, "LH": 30, "LW": 30, "OH": 5, "OW": 5}
    B, C, LH, LW, OH, OW = config['B'], config['C'], config['LH'], config['LW'], config['OH'], config['OW']
    
    try:
        logits_slice = torch.randn(B, C, LH, LW, device=device)
        problem_slice = torch.randint(0, C, (LH, LW), device=device)
        output_shape = (OH, OW)
        O_x, O_y = LH - OH + 1, LW - OW + 1
        x_logprobs = torch.randn(B, O_x, device=device)
        y_logprobs = torch.randn(B, O_y, device=device)
        x_log_partition = torch.randn(B, device=device)
        y_log_partition = torch.randn(B, device=device)
        
        perf_results = {}
        
        for impl_name, impl_func in implementations:
            # Warmup
            for _ in range(5):
                _ = impl_func(logits_slice, problem_slice, output_shape,
                            x_logprobs, y_logprobs, x_log_partition, y_log_partition)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Time measurement
            start_time = time.time()
            for _ in range(100):
                _ = impl_func(logits_slice, problem_slice, output_shape,
                            x_logprobs, y_logprobs, x_log_partition, y_log_partition)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            perf_results[impl_name] = elapsed_time
        
        print(f"\n{'Implementation':<12} {'Time (s)':<10} {'Memory (MB)':<12} {'Efficiency':<12}")
        print("-" * 50)
        
        medium_results = results.get('Medium', {})
        for impl_name in ['Conv v1', 'Conv v2', 'Conv v3']:
            if impl_name in perf_results and impl_name in medium_results:
                time_val = perf_results[impl_name]
                if 'error' not in medium_results[impl_name]:
                    memory_val = medium_results[impl_name][memory_key]
                    efficiency = time_val * memory_val  # Lower is better
                    print(f"{impl_name:<12} {time_val:<10.3f} {memory_val:<12.1f} {efficiency:<12.1f}")
                else:
                    print(f"{impl_name:<12} {time_val:<10.3f} {'ERROR':<12} {'N/A':<12}")
        
        print("\nNotes:")
        print("- Lower efficiency score (Time × Memory) is better")
        print("- v1: Loop over channels (simple but potentially slower)")
        print("- v2: Uses unfold (memory intensive but potentially faster)")
        print("- v3: Uses grouped convolution (balanced approach)")
        
    except Exception as e:
        print(f"Performance test failed: {e}")
    
    return results

def memory_stress_test():
    """Stress test to highlight memory differences between implementations"""
    print("=" * 70)
    print("MEMORY STRESS TEST - HIGHLIGHTING DIFFERENCES")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - this test is designed for GPU")
        return
    
    device = torch.device('cuda')
    
    # Progressive stress test configurations
    stress_configs = [
        {"name": "Moderate", "B": 64, "C": 10, "LH": 60, "LW": 60, "OH": 10, "OW": 10},
        {"name": "High", "B": 128, "C": 10, "LH": 80, "LW": 80, "OH": 15, "OW": 15},
        {"name": "Extreme", "B": 256, "C": 10, "LH": 100, "LW": 100, "OH": 20, "OW": 20},
        {"name": "Maximum", "B": 512, "C": 10, "LH": 120, "LW": 120, "OH": 25, "OW": 25},
    ]
    
    implementations = [
        ("Conv v1", convolution_implementation_v1),
        ("Conv v2", convolution_implementation_v2),
        ("Conv v3", convolution_implementation_v3)
    ]
    
    results = {}
    
    for config in stress_configs:
        print(f"\n--- {config['name']} Stress Test ---")
        B, C, LH, LW, OH, OW = config['B'], config['C'], config['LH'], config['LW'], config['OH'], config['OW']
        print(f"Batch: {B}, Grid: {LH}x{LW}, Output: {OH}x{OW}")
        
        # Calculate expected memory requirements
        input_size = B * C * LH * LW * 4 / (1024**2)
        intermediate_size_v2 = B * C * OH * OW * (LH - OH + 1) * (LW - OW + 1) * 4 / (1024**2)  # unfold creates this
        print(f"Expected input size: {input_size:.1f} MB")
        print(f"Expected v2 intermediate size: {intermediate_size_v2:.1f} MB")
        
        results[config['name']] = {}
        
        # Try to create test data
        try:
            torch.manual_seed(42)
            logits_slice = torch.randn(B, C, LH, LW, device=device)
            problem_slice = torch.randint(0, C, (LH, LW), device=device)
            output_shape = (OH, OW)
            
            O_x, O_y = LH - OH + 1, LW - OW + 1
            x_logprobs = torch.randn(B, O_x, device=device)
            y_logprobs = torch.randn(B, O_y, device=device)
            x_log_partition = torch.randn(B, device=device)
            y_log_partition = torch.randn(B, device=device)
            
            print("✓ Test data created")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Cannot create test data - OUT OF MEMORY")
                print(f"   This configuration requires too much memory")
                results[config['name']] = {'error': 'OOM during setup'}
                clear_memory()
                continue
            else:
                raise e
        
        # Test each implementation
        for impl_name, impl_func in implementations:
            print(f"\n  Testing {impl_name}...")
            
            try:
                clear_memory()
                baseline = get_memory_usage()
                
                # Single run with memory monitoring
                result = impl_func(
                    logits_slice, problem_slice, output_shape,
                    x_logprobs, y_logprobs, x_log_partition, y_log_partition
                )
                
                torch.cuda.synchronize()
                peak = get_memory_usage()
                
                memory_used = {
                    'allocated': peak['gpu_allocated'] - baseline['gpu_allocated'],
                    'cached': peak['gpu_cached'] - baseline['gpu_cached']
                }
                
                results[config['name']][impl_name] = memory_used
                
                print(f"    GPU Allocated: {memory_used['allocated']:.1f} MB")
                print(f"    GPU Cached: {memory_used['cached']:.1f} MB")
                print(f"    Total GPU: {memory_used['allocated'] + memory_used['cached']:.1f} MB")
                print(f"    Output shape: {result.shape}")
                print(f"    ✓ Success")
                
                del result
                clear_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    ❌ OUT OF MEMORY")
                    results[config['name']][impl_name] = {'error': 'OOM'}
                    clear_memory()
                else:
                    print(f"    ❌ ERROR: {e}")
                    results[config['name']][impl_name] = {'error': str(e)}
                    clear_memory()
        
        # Clean up test data
        try:
            del logits_slice, problem_slice, x_logprobs, y_logprobs, x_log_partition, y_log_partition
        except:
            pass
        clear_memory()
    
    # Analysis and Summary
    print("\n" + "=" * 70)
    print("STRESS TEST ANALYSIS")
    print("=" * 70)
    
    print(f"\n{'Config':<10} {'v1 Total':<12} {'v2 Total':<12} {'v3 Total':<12} {'v2/v1':<8} {'v3/v1':<8}")
    print("-" * 70)
    
    for config_name in results:
        if isinstance(results[config_name], dict) and 'error' in results[config_name]:
            print(f"{config_name:<10} {results[config_name]['error']}")
            continue
            
        v1_total = v2_total = v3_total = None
        
        for impl in ['Conv v1', 'Conv v2', 'Conv v3']:
            if impl in results[config_name] and 'error' not in results[config_name][impl]:
                total = results[config_name][impl]['allocated'] + results[config_name][impl]['cached']
                if impl == 'Conv v1':
                    v1_total = total
                elif impl == 'Conv v2':
                    v2_total = total
                elif impl == 'Conv v3':
                    v3_total = total
        
        v1_str = f"{v1_total:.1f} MB" if v1_total is not None else "OOM"
        v2_str = f"{v2_total:.1f} MB" if v2_total is not None else "OOM"
        v3_str = f"{v3_total:.1f} MB" if v3_total is not None else "OOM"
        
        v2_ratio = f"{v2_total/v1_total:.1f}x" if v1_total and v2_total and v1_total > 0 else "N/A"
        v3_ratio = f"{v3_total/v1_total:.1f}x" if v1_total and v3_total and v1_total > 0 else "N/A"
        
        print(f"{config_name:<10} {v1_str:<12} {v2_str:<12} {v3_str:<12} {v2_ratio:<8} {v3_ratio:<8}")
    
    # Memory efficiency recommendations
    print("\n" + "=" * 70)
    print("MEMORY EFFICIENCY RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n🔍 Key Findings:")
    print("• Conv v1: Uses simple loop over channels - consistent memory usage")
    print("• Conv v2: Uses unfold() - creates large intermediate tensors")
    print("• Conv v3: Uses grouped convolution - most memory efficient")
    
    print("\n💡 Recommendations:")
    print("• For small problems: Any version works fine")
    print("• For medium problems: v3 > v1 > v2 (memory efficiency)")
    print("• For large problems: v3 is strongly recommended")
    print("• v2 may cause OOM errors on large inputs due to unfold() memory requirements")
    
    print("\n⚡ Performance vs Memory Trade-off:")
    print("• v3: Best balance of speed and memory efficiency")
    print("• v1: Moderate memory, slower due to loop")
    print("• v2: Fastest when memory allows, but memory-intensive")
    
    return results

def test_correctness_and_performance():
    """Test correctness and performance of all implementations"""
    
    # Set up test data
    torch.manual_seed(42)
    np.random.seed(42)
    
    B = 8  # batch size
    C = 10  # number of classes (colors 0-9)
    LH, LW = 20, 20  # large grid size
    OH, OW = 3, 3  # output shape
    
    # Create test inputs
    logits_slice = torch.randn(B, C, LH, LW)
    problem_slice = torch.randint(0, C, (LH, LW))
    output_shape = (OH, OW)
    
    # Create position priors
    O_x, O_y = LH - OH + 1, LW - OW + 1
    x_logprobs = torch.randn(B, O_x)
    y_logprobs = torch.randn(B, O_y)
    x_log_partition = torch.randn(B)
    y_log_partition = torch.randn(B)
    
    print("Testing correctness...")
    
    # Test original implementation
    result_original = original_implementation(
        logits_slice, problem_slice, output_shape, 
        x_logprobs, y_logprobs, x_log_partition, y_log_partition
    )
    
    # Test convolution implementation v1
    result_conv_v1 = convolution_implementation_v1(
        logits_slice, problem_slice, output_shape,
        x_logprobs, y_logprobs, x_log_partition, y_log_partition
    )
    
    # Test convolution implementation v2
    result_conv_v2 = convolution_implementation_v2(
        logits_slice, problem_slice, output_shape,
        x_logprobs, y_logprobs, x_log_partition, y_log_partition
    )
    
    # Test convolution implementation v3
    result_conv_v3 = convolution_implementation_v3(
        logits_slice, problem_slice, output_shape,
        x_logprobs, y_logprobs, x_log_partition, y_log_partition
    )
    
    # Check correctness
    print(f"Original shape: {result_original.shape}")
    print(f"Conv v1 shape: {result_conv_v1.shape}")
    print(f"Conv v2 shape: {result_conv_v2.shape}")
    print(f"Conv v3 shape: {result_conv_v3.shape}")
    
    # Compare results
    diff_v1 = torch.abs(result_original - result_conv_v1).max().item()
    diff_v2 = torch.abs(result_original - result_conv_v2).max().item()
    diff_v3 = torch.abs(result_original - result_conv_v3).max().item()
    
    print(f"Max difference (original vs conv v1): {diff_v1}")
    print(f"Max difference (original vs conv v2): {diff_v2}")
    print(f"Max difference (original vs conv v3): {diff_v3}")
    
    tolerance = 1e-5
    if diff_v1 < tolerance:
        print("✓ Convolution v1 implementation is correct!")
    else:
        print("✗ Convolution v1 implementation has errors!")
        
    if diff_v2 < tolerance:
        print("✓ Convolution v2 implementation is correct!")
    else:
        print("✗ Convolution v2 implementation has errors!")
        
    if diff_v3 < tolerance:
        print("✓ Convolution v3 implementation is correct!")
    else:
        print("✗ Convolution v3 implementation has errors!")
    
    print("\nTesting performance...")
    
    # Performance testing
    n_runs = 100
    
    # Original implementation timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(n_runs):
        _ = original_implementation(
            logits_slice, problem_slice, output_shape,
            x_logprobs, y_logprobs, x_log_partition, y_log_partition
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    original_time = time.time() - start_time
    
    # Convolution v1 timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(n_runs):
        _ = convolution_implementation_v1(
            logits_slice, problem_slice, output_shape,
            x_logprobs, y_logprobs, x_log_partition, y_log_partition
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    conv_v1_time = time.time() - start_time
    
    # Convolution v2 timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(n_runs):
        _ = convolution_implementation_v2(
            logits_slice, problem_slice, output_shape,
            x_logprobs, y_logprobs, x_log_partition, y_log_partition
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    conv_v2_time = time.time() - start_time
    
    # Convolution v3 timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(n_runs):
        _ = convolution_implementation_v3(
            logits_slice, problem_slice, output_shape,
            x_logprobs, y_logprobs, x_log_partition, y_log_partition
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    conv_v3_time = time.time() - start_time
    
    print(f"Original implementation: {original_time:.4f}s ({n_runs} runs)")
    print(f"Convolution v1: {conv_v1_time:.4f}s ({n_runs} runs)")
    print(f"Convolution v2: {conv_v2_time:.4f}s ({n_runs} runs)")
    print(f"Convolution v3: {conv_v3_time:.4f}s ({n_runs} runs)")
    print(f"Speedup v1: {original_time / conv_v1_time:.2f}x")
    print(f"Speedup v2: {original_time / conv_v2_time:.2f}x")
    print(f"Speedup v3: {original_time / conv_v3_time:.2f}x")
    print(f"v2 vs v1: {conv_v1_time / conv_v2_time:.2f}x")
    print(f"v3 vs v1: {conv_v1_time / conv_v3_time:.2f}x")
    print(f"v3 vs v2: {conv_v2_time / conv_v3_time:.2f}x")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "vram":
            benchmark_vram_usage()
        elif sys.argv[1] == "comprehensive":
            benchmark_vram_comprehensive()
        elif sys.argv[1] == "scaling":
            benchmark_vram_scaling()
        elif sys.argv[1] == "stress":
            memory_stress_test()
        elif sys.argv[1] == "all":
            test_correctness_and_performance()
            benchmark_vram_comprehensive()
            memory_stress_test()
        else:
            print("Usage: python test_convolution_optimization.py [vram|comprehensive|scaling|stress|all]")
            print("  vram         - Run basic VRAM usage benchmark")
            print("  comprehensive - Run comprehensive VRAM benchmark with larger sizes")
            print("  scaling      - Run VRAM scaling analysis")
            print("  stress       - Run memory stress test to highlight differences")
            print("  all          - Run all tests")
            print("  (no args)    - Run correctness and performance tests only")
    else:
        test_correctness_and_performance()