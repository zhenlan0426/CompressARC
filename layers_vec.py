import torch
import torch.nn.functional as F
from torch_scatter import segment_coo, scatter_mean
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
    
    n_slices = len(flat.dims_list)
    
    # Use pre-computed row2slice mapping
    row2slice = flat.row2slice
    
    # Compute per-slice statistics using scatter_mean
    if debias:
        # Compute mean per slice per channel: shape (n_slices, channel_dim)
        slice_means = scatter_mean(flat.data, row2slice, dim=0, dim_size=n_slices)
        
        # Subtract mean from each position
        centered_data = flat.data - slice_means[row2slice]
        
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
    
    # Normalize: divide centered data by standard deviation
    normalized_data = centered_data / slice_stds[row2slice]
    
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

