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

try:
    import layers_batch
except ModuleNotFoundError:
    layers_batch = None  # continue even if batched impl not yet present


################################################################################
# Helper utilities                                                                
################################################################################
# clone_slices: if True, each slice is clone+detached (no shared grads)
#               if False, slices are views into shared storage (fast / memory-light)
def split_multitensor_batch(
    mt_batched: mtsys.MultiTensor,
    batch_size: int = 8,
    clone_slices: bool = True,
) -> List[mtsys.MultiTensor]:
    """Split the leading batch dimension into a list of *views*.

    Each returned MultiTensor mirrors the structure of *mt_batched* but the
    leaf tensors correspond to *mt_batched[dims][b]*.

    Gradients flowing through the un-batched views accumulate into the shared
    storage of *mt_batched*, which is what we want for reference testing.
    """
    system = mt_batched.multitensor_system

    split_mt: List[mtsys.MultiTensor] = [system.make_multitensor() for _ in range(batch_size)]
    for dims in system:
        batched_leaf = mt_batched[dims]  # shape (B, ...)
        for b in range(batch_size):
            leaf_slice = batched_leaf[b]
            if clone_slices:
                leaf_slice = leaf_slice.detach().clone().requires_grad_(True)
            split_mt[b][dims] = leaf_slice
    return split_mt

def multitensor_allclose(mt1: mtsys.MultiTensor, mt2: mtsys.MultiTensor, **kwargs) -> Tuple[bool, Tuple[int, ...] | None]:
    """Element-wise allclose across every leaf.  Returns (ok, bad_dims)."""
    system = mt1.multitensor_system
    assert mt2.multitensor_system is system, "Systems differ"
    for dims in system:
        if not torch.allclose(mt1[dims], mt2[dims], **kwargs):
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
    atol: float = 1e-6,
    rtol: float = 1e-4,
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
        ok, bad_dims = multitensor_allclose(out_ref_list[b], out_batched_splits[b], atol=atol, rtol=rtol)
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
        for dims in arg.multitensor_system:
            grad_batched_full = arg[dims].grad  # shape (B, ...)
            for b in range(batch_size):
                grad_ref_slice = splits[b][dims].grad
                assert torch.allclose(grad_batched_full[b], grad_ref_slice, atol=atol, rtol=rtol), (
                    f"Gradient mismatch at arg {arg_idx}, dims={dims}, batch idx {b} in {fn_ref.__name__}")

    print(f"[OK] {fn_ref.__name__}: forward & backward match (batch={batch_size})")

################################################################################
# Quick demo / registry                                                          
################################################################################

LAYER_TEST_REGISTRY = {
    "normalize": (layers.normalize, getattr(layers_batch, "normalize", None)),
}


def _make_random_latents(batch_size: int = 8):
    """Create a random MultiTensor with a leading batch dimension on leaves."""
    n_examples, n_colors, n_x, n_y = 2, 3, 5, 5
    system = mtsys.MultiTensorSystem(n_examples, n_colors, n_x, n_y, task=None)
    channel_dim = 4

    latents = system.make_multitensor()
    for dims in system:
        shape = system.shape(dims, extra_dim=channel_dim)
        latents[dims] = torch.randn(batch_size, *shape, requires_grad=True)
    return latents


def run_smoke_test():
    if layers_batch is None:
        print("layers_batch not found – skipping smoke test.")
        return
    latents = _make_random_latents(batch_size=8)
    meta_tester(layers.normalize, layers_batch.normalize, (latents,), batch_size=8)


def run_full():
    if layers_batch is None:
        print("layers_batch not found – no tests to run.")
        return
    for name, (fn_ref, fn_batched) in LAYER_TEST_REGISTRY.items():
        if fn_batched is None:
            print(f"[SKIP] {name}: batched implementation missing")
            continue
        latents = _make_random_latents(batch_size=8)
        meta_tester(fn_ref, fn_batched, (latents,), batch_size=8)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    if len(sys.argv) > 1 and sys.argv[1] == "full":
        run_full()
    else:
        run_smoke_test() 