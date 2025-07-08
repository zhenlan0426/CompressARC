import torch
import torch.nn.functional as F
from torch_scatter import segment_csr, scatter_mean
from typing import Optional, Union
import numpy as np

from multitensor_systems_vec import FlatMultiTensor
from multitensor_systems import MultiTensor
import multitensor_systems
from functools import partial

"""
Vectorized implementations of layers that operate on FlatMultiTensor.
This file contains GPU-friendly versions of operations from layers.py.
"""

np.random.seed(0)
torch.manual_seed(0)

def add_residual(layer, use_bias=False, pre_norm=False, post_norm=False):
    """
    Surround a layer/operation with a residual connection, up and down projections,
    and pre/post-norms.
    Args:
        layer (Callable): The layer/operation to modify.
    Returns:
        Callable: Another layer/operation that applies the original layer with the
                above modifications.
    """
    def layer_with_residual(x, residual_weights, *args,
                            use_bias=use_bias, pre_norm=pre_norm, post_norm=post_norm, **kwargs):
        if isinstance(residual_weights, MultiTensor):
            down_project_weights = multitensor_systems.multify(lambda dims, weights: weights[0])(residual_weights)
            up_project_weights = multitensor_systems.multify(lambda dims, weights: weights[1])(residual_weights)
        else:
            down_project_weights = residual_weights[0]
            up_project_weights = residual_weights[1]
        if pre_norm:
            z = normalize(x)
        z = affine(x, down_project_weights, use_bias=use_bias)
        z = layer(z, *args, **kwargs)
        if post_norm:
            z = normalize(z)
        z = affine(z, up_project_weights, use_bias=use_bias)
        return x + z
    return layer_with_residual

def normalize(flat: FlatMultiTensor, debias: bool = True) -> FlatMultiTensor:
    """
    Vectorized normalize operation using segment_csr for efficient segment reduction.
    
    Normalizes each slice to have variance one, computed independently per slice and channel.
    Uses CSR pointers derived from slice lengths for optimal performance.
    
    Args:
        flat: FlatMultiTensor with data shape (total_positions, channel_dim)
        debias: If True, subtract mean before normalizing variance
        
    Returns:
        FlatMultiTensor with normalized data
    """
    # TODO: improve performance, right now it's slower than the regular implementation
    n_slices = len(flat.dims_list)
    
    # ------------------------------------------------------------------
    # Use pre-computed CSR indptr for segment_csr operations
    slice_lengths = flat.lengths
    indptr = flat.indptr

    # Compute per-slice statistics using `segment_csr`.
    if debias:
        # Mean per slice per channel (directly computed)
        slice_means = segment_csr(flat.data, indptr, reduce="mean")  # (n_slices, C)

        # Center the data
        broadcast_means = torch.repeat_interleave(slice_means, slice_lengths, dim=0)
        centered_data = flat.data - broadcast_means

        # Squared centred values for variance
        variance_data = centered_data ** 2
    else:
        centered_data = flat.data
        variance_data = centered_data ** 2

    # Variance (mean of squared centred values) per slice per channel
    slice_vars = segment_csr(variance_data, indptr, reduce="mean")
    
    # Compute standard deviation with numerical stability
    eps = 1e-8
    slice_stds = torch.sqrt(slice_vars + eps)
    
    # Normalize: divide centered data by standard deviation using efficient broadcasting
    broadcast_stds = torch.repeat_interleave(slice_stds, slice_lengths, dim=0)
    normalized_data = centered_data / broadcast_stds
    
    # Create new FlatMultiTensor with normalized data - avoid unnecessary copying
    return FlatMultiTensor(
        data=normalized_data,
        offsets=flat.offsets,
        lengths=flat.lengths,
        shapes=flat.shapes,
        dims_list=flat.dims_list,
        channel_dim=flat.channel_dim,
    )

def affine(flat: FlatMultiTensor, weight: Union[torch.Tensor, MultiTensor, list, tuple], use_bias: bool = False) -> FlatMultiTensor:
    """
    Affine transformation for a *FlatMultiTensor*.

    Two usage modes are supported.
    1. Shared weight  (``weight`` is a ``Tensor`` **or** a two–element ``(W, b)`` list/tuple):
       Every slice is multiplied by the same matrix ``W`` (and optional bias ``b``).
    2. Per-slice weight (``weight`` is a ``MultiTensor`` whose leaves hold ``(W, b)`` pairs):
       Each slice *i* picks its own ``W_i`` (and optional ``b_i``) and we loop over the
       *k* slices.  Only *k* GEMM calls are issued; memory footprint stays
       O(total_len · d).

    Args
    -----
    flat   : FlatMultiTensor
        Input buffer of shape (total_len, d).
    weight : Union[Tensor, MultiTensor, Sequence[Tensor]]
        See modes above.
    use_bias : bool, default False
        Whether to add the bias vector(s) after the matrix multiplication.
    """

    # ------------------------------------------------------------------
    # Fast path – one global weight for all slices.
    # ------------------------------------------------------------------
    if not isinstance(weight, MultiTensor):
        # Accept both (W, b) container or bare W tensor.
        if isinstance(weight, (list, tuple)):
            W = weight[0]
            b = weight[1] if use_bias and len(weight) > 1 else None
        else:
            W = weight
            b = None

        out_data = torch.matmul(flat.data, W)
        if use_bias and b is not None:
            out_data = out_data + b  # broadcast add

        return FlatMultiTensor(
            data=out_data,
            offsets=flat.offsets,
            lengths=flat.lengths,
            shapes=flat.shapes,
            dims_list=flat.dims_list,
            channel_dim=flat.channel_dim,
        )

    # ------------------------------------------------------------------
    # Per-slice weights – loop over *k* slices (k = 18 here).
    # ------------------------------------------------------------------
    out_data = torch.empty_like(flat.data)

    for idx, dims in enumerate(flat.dims_list):
        # Retrieve the (W, b) pair (or bare W) for this slice.
        wb = weight[dims]
        if isinstance(wb, (list, tuple)):
            W_i = wb[0]
            b_i = wb[1] if use_bias and len(wb) > 1 else None
        else:
            W_i = wb
            b_i = None

        offset = int(flat.offsets[idx].item())
        length = int(flat.lengths[idx].item())

        slice_in = flat.data.narrow(0, offset, length)
        slice_out = torch.matmul(slice_in, W_i)
        if use_bias and b_i is not None:
            slice_out = slice_out + b_i

        out_data.narrow(0, offset, length).copy_(slice_out)

    return FlatMultiTensor(
        data=out_data,
        offsets=flat.offsets,
        lengths=flat.lengths,
        shapes=flat.shapes,
        dims_list=flat.dims_list,
        channel_dim=flat.channel_dim,
    )

@partial(add_residual, post_norm=True)
def share_up(flat: FlatMultiTensor) -> FlatMultiTensor:
    """Vectorized *share_up* operation implemented via a cached CSR SpMM.

    The heavy lifting is delegated to a pre-built sparse matrix *S* (constructed
    once by ``FlatMultiTensor.build_share_up_metadata``).  The forward pass then
    reduces to a single call to ``torch.sparse.mm`` which is handled by
    cuSPARSE/hipSPARSE and supports autograd out-of-the-box.
    """
    # Ensure the metadata (and thus the CSR matrix) is available
    # flat.build_share_up_metadata()
    S = flat._share_up_S  # (N, N) sparse CSR on the correct device

    # Sparse matrix multiplication: (N, N) @ (N, C) → (N, C)
    out_data = torch.sparse.mm(S, flat.data)

    # Return a new FlatMultiTensor sharing all metadata but with the updated data
    return FlatMultiTensor(
        data=out_data,
        offsets=flat.offsets,
        lengths=flat.lengths,
        shapes=flat.shapes,
        dims_list=flat.dims_list,
        channel_dim=flat.channel_dim,
    )
