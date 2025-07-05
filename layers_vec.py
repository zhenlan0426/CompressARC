import torch
import torch.nn.functional as F
from torch_scatter import segment_coo
from typing import Optional
import numpy as np

from multitensor_systems_vec import FlatMultiTensor

"""
Vectorized implementations of layers that operate on FlatMultiTensor.
This file contains GPU-friendly versions of operations from layers.py.
"""

np.random.seed(0)
torch.manual_seed(0)


def normalize_vec(flat: FlatMultiTensor, debias: bool = True) -> FlatMultiTensor:
    """
    Vectorized normalize operation using pytorch_scatter segment_coo for efficient segment reduction.
    
    Normalizes each slice to have variance one, computed independently per slice and channel.
    Uses pre-computed row2slice mapping for optimal performance.
    
    Args:
        flat: FlatMultiTensor with data shape (total_positions, channel_dim)
        debias: If True, subtract mean before normalizing variance
        
    Returns:
        FlatMultiTensor with normalized data
    """
    
    n_slices = len(flat.dims_list)
    
    # Use pre-computed row2slice mapping - indices are already sorted
    row2slice = flat.row2slice
    
    # Compute per-slice statistics using segment_coo (optimized for sorted indices)
    if debias:
        # Compute mean per slice per channel: shape (n_slices, channel_dim)
        slice_means = segment_coo(flat.data, row2slice, dim_size=n_slices, reduce='mean')
        
        # Subtract mean from each position
        centered_data = flat.data - slice_means[row2slice]
        
        # Compute variance from centered data
        variance_data = centered_data ** 2
    else:
        # Skip mean subtraction, compute variance directly
        variance_data = flat.data ** 2
        centered_data = flat.data
    
    # Compute mean of squared values (variance) per slice per channel
    slice_vars = segment_coo(variance_data, row2slice, dim_size=n_slices, reduce='mean')
    
    # Compute standard deviation with numerical stability
    eps = 1e-8
    slice_stds = torch.sqrt(slice_vars + eps)
    
    # Normalize: divide centered data by standard deviation
    normalized_data = centered_data / slice_stds[row2slice]
    
    # Create new FlatMultiTensor with normalized data
    return FlatMultiTensor(
        data=normalized_data,
        offsets=flat.offsets.clone(),
        lengths=flat.lengths.clone(),
        shapes=flat.shapes.copy(),
        dims_list=flat.dims_list.copy(),
        channel_dim=flat.channel_dim,
        row2slice=flat.row2slice.clone(),
    )


def affine_vec(flat: FlatMultiTensor, weights: torch.Tensor, bias: Optional[torch.Tensor] = None) -> FlatMultiTensor:
    """
    Vectorized affine transformation for FlatMultiTensor.
    
    Args:
        flat: Input FlatMultiTensor
        weights: Weight tensor of shape (n_slices, channel_dim, channel_dim) or (channel_dim, channel_dim)
        bias: Optional bias tensor of shape (n_slices, channel_dim) or (channel_dim,)
        
    Returns:
        FlatMultiTensor with transformed data
    """
    
    # Use pre-computed row2slice mapping
    row2slice = flat.row2slice
    
    if weights.dim() == 3:
        # Per-slice weights: (n_slices, channel_dim, channel_dim)
        # Gather weights for each position: (total_positions, channel_dim, channel_dim)
        pos_weights = weights[row2slice]  # (total_positions, channel_dim, channel_dim)
        
        # Apply transformation: (P, C) @ (P, C, C) -> (P, C)
        # Using einsum for batched matrix multiplication
        transformed_data = torch.einsum('pc,pco->po', flat.data, pos_weights)
    else:
        # Global weights: (channel_dim, channel_dim)
        transformed_data = torch.matmul(flat.data, weights)
    
    if bias is not None:
        if bias.dim() == 2:
            # Per-slice bias: (n_slices, channel_dim)
            pos_bias = bias[row2slice]  # (total_positions, channel_dim)
            transformed_data = transformed_data + pos_bias
        else:
            # Global bias: (channel_dim,)
            transformed_data = transformed_data + bias
    
    return FlatMultiTensor(
        data=transformed_data,
        offsets=flat.offsets.clone(),
        lengths=flat.lengths.clone(),
        shapes=flat.shapes.copy(),
        dims_list=flat.dims_list.copy(),
        channel_dim=flat.channel_dim,
        row2slice=flat.row2slice.clone(),
    ) 