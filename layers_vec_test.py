import preprocessing
from multitensor_systems import multify, MultiTensor
from multitensor_systems_vec import flat_multitensor, unpack_flat
import torch
import initializers
import layers_vec
import layers
import time

def make_affine_weights(channel_dim: int = 16,
                        split: str = "training",
                        task_index: int = 0,
                        device: str = "cuda",
                        up_down_weights: bool = False) -> MultiTensor:
    """
    Build a `MultiTensor` whose 18 leaves each contain a (W, b) pair with
    shapes (channel_dim, channel_dim) and (channel_dim,).

    Nothing else needs to be constructed by the caller.

    Parameters
    ----------
    channel_dim : int          Width of each weight matrix / bias vector.
    split       : str          Which ARC split to load via `preprocessing`.
    task_index  : int          Which task number within the split to use just
                               for its `MultiTensorSystem`.
    device      : str          CUDA / CPU device string.

    Returns
    -------
    MultiTensor
        Per-slice weights ready to pass as the `weight` argument to
        `layers.affine` and `layers_vec.affine`.
    """
    # Obtain a task object solely for its multitensor_system metadata
    task_nums = [task_index]
    task = preprocessing.preprocess_tasks(split, task_nums)[0]
    mts = task.multitensor_system          # <-- we need only this

    # Create an empty multitensor to hold the (W, b) pairs
    weights_mt = mts.make_multitensor()

    for dims in mts:                       # iterates over the 18 valid slices
        if up_down_weights:
            W1 = torch.randn(channel_dim, channel_dim, device=device) / channel_dim**0.5
            b1 = torch.randn(channel_dim,               device=device)
            W2 = torch.randn(channel_dim, channel_dim, device=device) / channel_dim**0.5
            b2 = torch.randn(channel_dim,               device=device)
            weights_mt[dims] = ((W1, b1), (W2, b2))
        else:
            W = torch.randn(channel_dim, channel_dim, device=device) / channel_dim**0.5
            b = torch.randn(channel_dim,               device=device)
            weights_mt[dims] = (W, b)

    return weights_mt

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

def test_meta(fn, fn_vec, *args,**kwargs):
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
    MTs = create_mts()
    start_time = time.time()
    FTs = [flat_multitensor(mt, debug=True) for mt in MTs]
    print(f"Time taken to create multitensors: {time.time() - start_time:.4f}s")
    print(f"Testing with {len(MTs)} multitensors")
    
    for i, (mt, ft) in enumerate(zip(MTs, FTs)):
        # forward pass
        mt2 = fn(mt, *args, **kwargs)
        ft2 = fn_vec(ft, *args, **kwargs)
        forward_diff = abs_diff(flat_multitensor(mt2).data, ft2.data)
        assert forward_diff < 1e-4, f"forward pass failed for multitensor {i}, diff: {forward_diff}"
        
        # backward pass
        flat_multitensor(mt2).data.sum().backward()
        ft2.data.sum().backward()
        backward_diff = abs_diff(flat_multitensor(get_grads(mt)).data, ft.data.grad)
        assert backward_diff < 3e-3, f"backward pass failed for multitensor {i}, diff: {backward_diff}"
    
    print("✓ Forward and backward pass correctness tests passed")
    
    # benchmark performance
    num_runs=10
    print(f"Benchmarking performance with {num_runs} runs...")
    
    # --- Regular implementation timing ---
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        for mt in MTs:
            mt2 = fn(mt, *args, **kwargs)
            flat_multitensor(mt2).data.sum().backward()
    torch.cuda.synchronize()
    regular_time = time.time() - start_time

    # --- Vectorized implementation timing ---
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        for ft in FTs:
            ft2 = fn_vec(ft, *args, **kwargs)
            ft2.data.sum().backward()
    torch.cuda.synchronize()
    vectorized_time = time.time() - start_time
    
    print(f"Regular implementation: {regular_time:.4f}s")
    print(f"Vectorized implementation: {vectorized_time:.4f}s")
    print(f"Speedup: {regular_time/vectorized_time:.2f}x")
    print("=" * 50)


# test_meta(layers.normalize, layers_vec.normalize)

# test single weight
# test_meta(layers.affine, layers_vec.affine, weight=(torch.randn(16, 16, device='cuda'),torch.randn(16, device='cuda')))

# test multitensor weight
# test_meta(layers.affine, layers_vec.affine, weight=make_affine_weights(channel_dim=16))

test_meta(layers.share_up, layers_vec.share_up, make_affine_weights(channel_dim=16, up_down_weights=True)) 
