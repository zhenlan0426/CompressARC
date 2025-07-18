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

def nested_allclose(a, b, checkGrad=False, **kwargs):
    """Compare two nested structures with tensors using allclose."""
    if is_tensor(a) and is_tensor(b):
        if checkGrad:
            # Check if tensors have gradients or are views with base tensors having gradients
            a_grad = a.grad if a.is_leaf else a._base.grad
            b_grad = b.grad if b.is_leaf else b._base.grad
            return a_grad is not None and b_grad is not None and torch.allclose(a_grad, b_grad, **kwargs)
        return torch.allclose(a, b, **kwargs)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(nested_allclose(a_i, b_i, checkGrad=checkGrad, **kwargs) for a_i, b_i in zip(a, b))
    else:
        raise TypeError(f"Unsupported type for nested structure: {type(a)} vs {type(b)}")

def multitensor_allclose(mt1: mtsys.MultiTensor, mt2: mtsys.MultiTensor, checkGrad=False,
                        **kwargs) -> Tuple[bool, Tuple[int, ...] | None]:
    """Element-wise allclose across every leaf. Returns (ok, bad_dims)."""
    system = mt1.multitensor_system
    assert mt2.multitensor_system is system, "Systems differ"
    
    for dims in system:
        if not nested_allclose(mt1[dims], mt2[dims], checkGrad=checkGrad, **kwargs):
            return False, tuple(dims)
            
    return True, None

################################################################################
# Meta-tester                                                                    
################################################################################

def meta_tester(
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
        assert ok, f"Forward mismatch (batch idx {b}) at dims={bad_dims} in {fn_ref.__name__}"

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
        arg_split = split_multitensor_batch(arg, batch_size=batch_size, clone_slices=False)
        for b in range(batch_size):
            assert multitensor_allclose(
                arg_split[b], splits[b], checkGrad=True, atol=atol, rtol=rtol
            )[0], f"Gradients mismatch for arg {arg_idx} at batch {b} in {fn_ref.__name__}"

    print(f"[OK] {fn_ref.__name__}: forward & backward match")

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

LAYER_TEST_REGISTRY = {
    # "decode_latents": (layers.decode_latents, layers_batch.decode_latents, generate_decode_latents_data),
    "normalize": (layers.normalize, layers_batch.normalize, generate_single_tensor_data),
}

if __name__ == "__main__":
    dummy_system = mtsys.MultiTensorSystem(3, 4, 7, 7, None)

    for _, (ref_fn, batch_fn, generate_fn) in LAYER_TEST_REGISTRY.items():
        batched_args = generate_fn(dummy_system)
        meta_tester(ref_fn, batch_fn, batched_args)