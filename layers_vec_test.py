import preprocessing
from multitensor_systems import multify
from multitensor_systems_vec import flat_multitensor, unpack_flat
import torch
import layers_vec
import layers
import time
def create_mts(channel_dim=16, n_tasks=5):
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

MTs = create_mts()
def test_meta(MTs, fn, fn_vec, num_runs=10):
    FTs = [flat_multitensor(mt, debug=True) for mt in MTs]
    for mt, ft in zip(MTs, FTs):
        # forward pass
        mt2 = fn(mt)
        ft2 = fn_vec(ft)
        assert abs_diff(flat_multitensor(mt2).data, ft2.data) < 1e-4, f"forward pass failed"
        # backward pass
        flat_multitensor(mt2).data.sum().backward()
        ft2.data.sum().backward()
        assert abs_diff(flat_multitensor(get_grads(mt)).data, ft.data.grad) < 1e-3, f"backward pass failed"
    
    # benchmark performance
    start_time = time.time()
    for _ in range(num_runs):
        for mt in MTs:
            mt2 = fn(mt)
            flat_multitensor(mt2).data.sum().backward()
    regular_time = time.time() - start_time

    start_time = time.time()
    for _ in range(num_runs):
        for ft in FTs:
            ft2 = fn_vec(ft)
            ft2.data.sum().backward()
    vectorized_time = time.time() - start_time
    
    print(f"Regular implementation: {regular_time:.4f}s")
    print(f"Vectorized implementation: {vectorized_time:.4f}s")
    print(f"Speedup: {regular_time/vectorized_time:.2f}x")


test_meta(MTs, layers.normalize, layers_vec.normalize)

