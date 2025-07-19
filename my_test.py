"""
Utilities and simple meta-testing framework for verifying that the new
batched implementations of layers behave identically (forward & backward)
 to the original un-batched versions.

Usage pattern
-------------
Assume you have implemented a module `layers_batch.py` that mirrors the
API of the existing `layers.py` but supports a leading batch dimension
on every leaf tensor of a `MultiTensor`.

```
python test.py            # runs a quick smoke-test on layers.normalize
python test.py full       # runs all registered meta-tests
```

The script provides:
====================
* split_multitensor_batch   – view-only slice of a batched MultiTensor
* stack_multitensor_batch   – inverse of split_* (torch.stack along batch)
* multitensor_allclose      – element-wise comparison of two MultiTensors
* meta_tester               – generic forward/backward equivalence test

You can register additional layer pairs to `LAYER_TEST_REGISTRY` at the
bottom of the file.
"""

import sys
from typing import List, Tuple
import time

import numpy as np
import torch

import multitensor_systems as mtsys
import layers

import initializers_batch
import layers_batch


################################################################################
# Helper utilities                                                                
################################################################################
# clone_slices: if True, each slice is clone+detached (no shared grads)
#               if False, slices are views into shared storage (fast / memory-light)
def is_tensor(obj):
    """Check if an object is a torch.Tensor."""
    return isinstance(obj, torch.Tensor)

def split_nested_batch(nested_tensor, batch_idx, clone_slice=True):
    """nested, batch,... -> nested, slice, ..."""
    if is_tensor(nested_tensor):
        slice_result = nested_tensor[batch_idx]
        if clone_slice:
            slice_result = slice_result.detach().clone().requires_grad_(True)
        return slice_result
    elif isinstance(nested_tensor, (list, tuple)):
        container_type = type(nested_tensor)
        return container_type(split_nested_batch(item, batch_idx, clone_slice) 
                             for item in nested_tensor)
    elif nested_tensor is None:
        return None
    else:
        raise TypeError(f"Unsupported type for nested tensor: {type(nested_tensor)}")
    
def split_multitensor_batch(
    mt_batched: mtsys.MultiTensor,
    batch_size: int = 8,
    clone_slices: bool = True,
) -> List[mtsys.MultiTensor]:
    """ dim, nested, batch,... -> batch, dim, nested,...
    """
    system = mt_batched.multitensor_system

    split_mt: List[mtsys.MultiTensor] = [system.make_multitensor() for _ in range(batch_size)]
    for dims in system:
        batched_leaf = mt_batched[dims]  # Could be tensor or nested structure
        for b in range(batch_size):
            split_mt[b][dims] = split_nested_batch(batched_leaf, b, clone_slices)
    return split_mt

def nested_allclose(a, b, **kwargs):
    """Compare two nested structures with tensors using allclose."""
    if is_tensor(a) and is_tensor(b):
        return torch.allclose(a, b, **kwargs)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(nested_allclose(a_i, b_i, **kwargs) for a_i, b_i in zip(a, b))
    elif a is None and b is None:
        return True
    else:
        raise TypeError(f"Unsupported type for nested structure: {type(a)} vs {type(b)}")

def multitensor_allclose(mt1: mtsys.MultiTensor, mt2: mtsys.MultiTensor,
                        **kwargs) -> Tuple[bool, Tuple[int, ...] | None]:
    """Element-wise allclose across every leaf. Returns (ok, bad_dims)."""
    system = mt1.multitensor_system
    assert mt2.multitensor_system is system, "Systems differ"
    
    for dims in system:
        if not nested_allclose(mt1[dims], mt2[dims], **kwargs):
            return False, tuple(dims)
            
    return True, None

def get_nested_grad(obj):
    if is_tensor(obj):
        return obj.grad
    elif isinstance(obj, (list, tuple)):
        container_type = type(obj)
        return container_type(get_nested_grad(item) for item in obj)
    else:
        raise TypeError(f"Unsupported type for getting grad: {type(obj)}")

################################################################################
# Meta-tester                                                                    
################################################################################

def meta_tester(
    name,
    fn_ref,
    fn_batched,
    batched_args: Tuple,
    batch_size: int = 8,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    **kwargs,
):
    """Compare *fn_ref* (loop over batch) vs *fn_batched* (vectorised).

    Parameters
    ----------
    name : str
        Name of the layer to test.
    fn_ref : callable
        Original layer – expects *un-batched* MultiTensor inputs.
    fn_batched : callable
        Batched layer implementation.
    batched_args : Tuple
        Arguments to pass into *fn_batched*.  must all be MultiTensors.
        non-MultiTensor args are passed in as kwargs.
    """

    # Build per-batch argument tuples.
    #   All batched_args are MultiTensors -> split into list
    #   Non-MultiTensor args come via kwargs (shared across batches)
    per_batch_args: List[List[mtsys.MultiTensor]] = [[] for _ in range(batch_size)] # (b, args) -> (dims, )
    
    # Keep handles on all split MultiTensors for grad comparisons later
    all_splits: List[List[mtsys.MultiTensor]] = [] # (args, b) -> (dims, )
    
    for arg in batched_args:
        split_list = split_multitensor_batch(arg, batch_size=batch_size, clone_slices=True)
        all_splits.append(split_list)
        for b in range(batch_size):
            per_batch_args[b].append(split_list[b])

    # Forward pass – reference (loop) and batched.
    out_ref_list = [fn_ref(*args_b, **kwargs) for args_b in per_batch_args]
    out_batched = fn_batched(*batched_args, **kwargs)

    # Split batched output into per-slice views for comparison.
    out_batched_splits = split_multitensor_batch(out_batched, batch_size=batch_size, clone_slices=False)

    for b in range(batch_size):
        ok, bad_dims = multitensor_allclose(out_ref_list[b], out_batched_splits[b], atol=atol*0.1, rtol=rtol*0.1)
        assert ok, f"Forward mismatch (batch idx {b}) at dims={bad_dims} in {name}"

    # Backward – use same random gradient tensor for both paths.
    # Build a gradient MultiTensor with matching leaves.
    grad_mt = out_batched.multitensor_system.make_multitensor()
    for dims in out_batched.multitensor_system:
        grad_mt[dims] = torch.randn(*out_batched[dims].shape)

    # Batched backward.
    loss_batched = sum((out_batched[dims] * grad_mt[dims]).sum() for dims in out_batched.multitensor_system)
    loss_batched.backward()

    # Reference backward – loop over B and reuse corresponding slice of grad.
    for b in range(batch_size):
        loss_ref_b = sum(
            (out_ref_list[b][dims] * grad_mt[dims][b]).sum() for dims in out_batched.multitensor_system
        )
        loss_ref_b.backward()

    # Compare gradients on all input MultiTensors per slice.
    for arg_idx, (arg, splits) in enumerate(zip(batched_args, all_splits)):
        batched_grad_mt = arg.multitensor_system.make_multitensor()
        for dims in arg.multitensor_system:
            batched_grad_mt[dims] = get_nested_grad(arg[dims])
        split_grads = split_multitensor_batch(batched_grad_mt, batch_size=batch_size, clone_slices=False)
        for b in range(batch_size):
            ref_grad_mt = arg.multitensor_system.make_multitensor()
            for dims in arg.multitensor_system:
                ref_grad_mt[dims] = get_nested_grad(splits[b][dims])
            ok, bad_dims = multitensor_allclose(split_grads[b], ref_grad_mt, atol=atol, rtol=rtol)
            assert ok, f"Gradients mismatch for arg {arg_idx} at batch {b} in Function {name} at dims={bad_dims}"

    print(f"[PASSED] Function \033[1m{name}\033[0m: forward & backward match")

    # time batched vs unbatched
    start_time = time.time()
    for _ in range(10):
        out_ref_list = [fn_ref(*args_b, **kwargs) for args_b in per_batch_args]
        for b in range(batch_size):
            loss_ref_b = sum(
                (out_ref_list[b][dims] * grad_mt[dims][b]).sum() for dims in out_batched.multitensor_system
            )
            loss_ref_b.backward()
    unbatched_time = time.time() - start_time

    start_time = time.time()
    for _ in range(10):
        out_batched = fn_batched(*batched_args, **kwargs)
        loss_batched = sum((out_batched[dims] * grad_mt[dims]).sum() for dims in out_batched.multitensor_system)
        loss_batched.backward()
    batched_time = time.time() - start_time
    print(f"Time for unbatched: {unbatched_time:.2f} seconds, time for batched: {batched_time:.2f} seconds, ratio: {unbatched_time/batched_time:.2f}")
    

################################################################################
# registry                                                          
################################################################################

def generate_decode_latents_data(system, batch_size=8, decoding_dim=4, channel_dim_fn=lambda dims: 16 if dims[2] == 0 else 8):
    """Generate dummy batched MultiTensors using BatchInitializer."""
    initializer = initializers_batch.Initializer(system, channel_dim_fn, batch_size=batch_size, batch_weights=True)

    target_capacities = initializer.initialize_multizeros([decoding_dim])
    decode_weights = initializer.initialize_multilinear([decoding_dim, channel_dim_fn])
    multiposteriors = initializer.initialize_multiposterior(decoding_dim)

    # Clear weights_list as it's just for dummy data
    initializer.weights_list.clear()

    return target_capacities, decode_weights, multiposteriors

def generate_single_tensor_data(system, batch_size=8, channel_dim_fn=16):
    """Generate dummy MultiTensor with a single tensor for testing."""
    initializer = initializers_batch.Initializer(system, channel_dim_fn, batch_size=batch_size, batch_weights=True)
    return initializer.initialize_multisingle_tensor(16), # needs to be a tuple

def generate_affine_data(system, batch_size=8, batch_weights=True):
    in_channels = 16
    out_channels = 32
    channel_dim_fn = 16
    initializer = initializers_batch.Initializer(system, channel_dim_fn, batch_size=batch_size, batch_weights=batch_weights)
    x = initializer.initialize_multisingle_tensor(16)
    weight = initializer.initialize_multilinear([in_channels, out_channels])
    return x, weight

LAYER_TEST_REGISTRY = {
    "normalize": {
        "ref": layers.normalize,
        "batched": layers_batch.normalize,
        "generate": generate_single_tensor_data,
        "kwargs": {},
    },
    "affine_batched_weights": {
        "ref": layers.affine,
        "batched": layers_batch.affine,
        "generate": lambda s, bs=8: generate_affine_data(s, batch_size=bs, batch_weights=True),
        "kwargs": {"use_bias": True},
    },
    "affine_batched_weights_no_bias": {
        "ref": layers.affine,
        "batched": layers_batch.affine,
        "generate": lambda s, bs=8: generate_affine_data(s, batch_size=bs, batch_weights=True),
        "kwargs": {"use_bias": False},
    },    
    "affine_shared_weights": {
        "ref": layers.affine,
        "batched": layers_batch.affine,
        "generate": lambda s, bs=8: generate_affine_data(s, batch_size=bs, batch_weights=False),
        "kwargs": {"use_bias": True},
    },
}

if __name__ == "__main__":
    dummy_system = mtsys.MultiTensorSystem(3, 4, 7, 7, None)

    for name, info in LAYER_TEST_REGISTRY.items():
        generate = info['generate']
        batched_args = generate(dummy_system)
        meta_tester(name, info['ref'], info['batched'], batched_args, **info['kwargs'])