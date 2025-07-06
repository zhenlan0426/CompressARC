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
    Vectorized normalize operation using scatter_mean for efficient segment reduction.
    
    Normalizes each slice to have variance one, computed independently per slice and channel.
    Uses pre-computed row2slice mapping for optimal performance.
    
    Args:
        flat: FlatMultiTensor with data shape (total_positions, channel_dim)
        debias: If True, subtract mean before normalizing variance
        
    Returns:
        FlatMultiTensor with normalized data
    """
    # TODO: improve performance, right now it's slower than the regular implementation
    n_slices = len(flat.dims_list)
    
    # Use pre-computed row2slice mapping
    row2slice = flat.row2slice
    slice_lengths = flat.lengths
    
    # Compute per-slice statistics using scatter_mean
    if debias:
        # Compute mean per slice per channel: shape (n_slices, channel_dim)
        slice_means = scatter_mean(flat.data, row2slice, dim=0, dim_size=n_slices)
        
        # Subtract mean from each position using efficient broadcasting
        broadcast_means = torch.repeat_interleave(slice_means, slice_lengths, dim=0)
        centered_data = flat.data - broadcast_means
        
        # Compute variance from centered data
        variance_data = centered_data ** 2
    else:
        # Skip mean subtraction, compute variance directly
        variance_data = flat.data ** 2
        centered_data = flat.data
    
    # Compute mean of squared values (variance) per slice per channel
    slice_vars = scatter_mean(variance_data, row2slice, dim=0, dim_size=n_slices)
    
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
        row2slice=flat.row2slice,
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
        row2slice=flat.row2slice,
    )

def share_up(flat: FlatMultiTensor) -> FlatMultiTensor:
    """Vectorised **share-up** communication.

    The input ``flat`` must already contain the *down-projected* residual buffer
    for *all* slices concatenated together.  The routine uses pre-computed
    metadata inside ``FlatMultiTensor`` (``repeat_counts`` and ``dst_rows``)
    to perform the scatter-add without any Python-level loops.

    Returns a *new* ``FlatMultiTensor`` whose ``data`` tensor holds the
    aggregated result.  All metadata (offsets, shapes, etc.) are reused so the
    caller can continue to treat the output as a valid `FlatMultiTensor`.
    """
    # Ensure metadata present – it will be built exactly once per instance.
    flat.build_share_up_metadata()

    # --------------------------------------------------------------
    # Fast path using CSR metadata (pre-computed once during initialisation).
    # --------------------------------------------------------------
    # 1. Gather *all* source contributions in the order expected by CSR.
    src_expanded = flat.data[flat.src_rows.to(torch.long)]  # (M, C)

    # 2. Aggregate into destination rows via ``segment_csr`` (no sorting or
    #    additional indexing work is required).
    out_data = segment_csr(src_expanded, flat.csr_ptr.to(torch.long), reduce="sum")

    # Preserve type information by constructing a new FlatMultiTensor that
    # re-uses all structural metadata from ``flat`` but replaces the data.
    return FlatMultiTensor(
        data=out_data,
        offsets=flat.offsets,
        lengths=flat.lengths,
        shapes=flat.shapes,
        dims_list=flat.dims_list,
        channel_dim=flat.channel_dim,
        row2slice=flat.row2slice,
    )