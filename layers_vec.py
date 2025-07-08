import torch
import torch.nn.functional as F
from torch_scatter import segment_csr, scatter_mean
from typing import Optional
import numpy as np

from multitensor_systems_vec import FlatMultiTensor

"""
Vectorized implementations of layers that operate on FlatMultiTensor.
This file contains GPU-friendly versions of operations from layers.py.
"""

np.random.seed(0)
torch.manual_seed(0)


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

def affine(flat: FlatMultiTensor, weight, use_bias=False) -> FlatMultiTensor:
    """
    Apply a linear layer to a tensor, along the channel dimension.
    Args:
        x (Tensor): Input to the linear layer.
        weight (list[Tensor]): A weight matrix and a bias vector.
    Returns:
        Tensor: Output of the linear layer.
    """
    x = torch.matmul(flat.data, weight[0])
    if use_bias:
        x = x + weight[1]
    return FlatMultiTensor(
        data=x,
        offsets=flat.offsets,
        lengths=flat.lengths,
        shapes=flat.shapes,
        dims_list=flat.dims_list,
        channel_dim=flat.channel_dim,
    )

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

