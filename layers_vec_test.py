import preprocessing
from multitensor_systems import multify
from multitensor_systems_vec import flat_multitensor, unpack_flat
import torch
import layers_vec
import layers
import time

def create_mts(channel_dim=16, n_tasks=5):
    """
    Create real-sized multitensors for testing purposes.
    
    Args:
        channel_dim (int, optional): Number of channels for each multitensor. Defaults to 16.
        n_tasks (int, optional): Number of tasks to create multitensors for. Defaults to 5.
    
    Returns:
        list: List of multitensors, one for each task, with gradients enabled for testing
              backward pass functionality.
    """
    # create real sized multitensor for testing
    task_nums = list(range(n_tasks))
    split = "training"  # "training", "evaluation, or "test"
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    MTs = []

    @multify
    def init(dims, _, multitensor_system, channel_dim):
        shape = multitensor_system.shape(dims, channel_dim)
        mean = torch.randn(shape,device='cuda')
        mean.requires_grad=True
        return mean

    for task in tasks:
        multitensor_system = task.multitensor_system
        mt = init(multitensor_system.make_multitensor(channel_dim), multitensor_system, channel_dim)
        MTs.append(mt)
    return MTs

@multify
def get_grads(dims, mt):
    return mt.grad

def abs_diff(a, b):
    return torch.abs(a - b).mean()

def test_meta(fn, fn_vec, **kwargs):
    """
    Meta-testing function that compares regular and vectorized implementations of layer functions.
    
    This function tests both correctness (forward and backward passes) and performance
    between a regular multitensor implementation and its vectorized counterpart.
    
    Args:
        fn: Regular multitensor function to test.
        fn_vec: Vectorized version of the function to test.
        **kwargs: Additional keyword arguments to pass to both functions.
    
    Raises:
        AssertionError: If forward or backward pass results differ significantly
                       between regular and vectorized implementations.
    
    Performance:
        Prints timing comparison and speedup factor between implementations.
    """
    print(f"\n===== Testing {fn_vec.__name__} =====")
    if kwargs:
        print(f"With arguments: {kwargs}")
    
    MTs = create_mts()
    FTs = [flat_multitensor(mt, debug=True) for mt in MTs]
    print(f"Testing with {len(MTs)} multitensors")
    
    for i, (mt, ft) in enumerate(zip(MTs, FTs)):
        # forward pass
        mt2 = fn(mt, **kwargs)
        ft2 = fn_vec(ft, **kwargs)
        forward_diff = abs_diff(flat_multitensor(mt2).data, ft2.data)
        assert forward_diff < 1e-4, f"forward pass failed for multitensor {i}, diff: {forward_diff}"
        
        # backward pass
        flat_multitensor(mt2).data.sum().backward()
        ft2.data.sum().backward()
        backward_diff = abs_diff(flat_multitensor(get_grads(mt)).data, ft.data.grad)
        assert backward_diff < 1e-3, f"backward pass failed for multitensor {i}, diff: {backward_diff}"
    
    print("✓ Forward and backward pass correctness tests passed")
    
    # benchmark performance
    num_runs=10
    print(f"Benchmarking performance with {num_runs} runs...")
    
    start_time = time.time()
    for _ in range(num_runs):
        for mt in MTs:
            mt2 = fn(mt, **kwargs)
            flat_multitensor(mt2).data.sum().backward()
    regular_time = time.time() - start_time

    start_time = time.time()
    for _ in range(num_runs):
        for ft in FTs:
            ft2 = fn_vec(ft, **kwargs)
            ft2.data.sum().backward()
    vectorized_time = time.time() - start_time
    
    print(f"Regular implementation: {regular_time:.4f}s")
    print(f"Vectorized implementation: {vectorized_time:.4f}s")
    print(f"Speedup: {regular_time/vectorized_time:.2f}x")
    print("=" * 50)


test_meta(layers.normalize, layers_vec.normalize)
# test_meta(layers.affine, layers_vec.affine, weight=(torch.randn(16, 16, device='cuda'),torch.randn(16, device='cuda')))
