import itertools

import numpy as np
import torch

import multitensor_systems

from layers import affine

"""
This file contains batched versions of some layers, vectorized over the batch dimension.
"""

np.random.seed(0)
torch.manual_seed(0)

@multitensor_systems.multify
def normalize(dims, x, debias=True):
    """
    Normalize the tensor to have variance one, for every index along the channel dimension.
    Args:
        dims (list[int]): Tells you which tensor in the multitensor system we're normalizing
        x (Tensor): Tensor to normalize.
    Returns:
        Tensor: Normalized tensor.
    """
    all_but_last = list(range(1, len(x.shape)-1)) # all but batch and channel dimensions
    if debias:
        x = x - torch.mean(x, dim=all_but_last, keepdim=True)
    x = x / torch.sqrt(1e-8+torch.mean(x**2, dim=all_but_last, keepdim=True))
    return x

def channel_layer(target_capacity, posterior):
    """
    Batched version of channel_layer. Assumes inputs have a prepended batch dimension.
    Operations are vectorized over the batch dim.
    """
    mean, local_capacity_adjustment = posterior

    batch_size = mean.shape[0]
    all_but_batch_and_channel = list(range(1, len(mean.shape) - 1))
    num_spatial_dims = len(mean.shape) - 2

    dimensionality = 1
    for axis_length in mean.shape[1:]:
        dimensionality *= axis_length

    min_capacity = torch.tensor(0.5, device=mean.device)
    init_capacity = torch.tensor(10000, device=mean.device)

    target_capacity = 10 * target_capacity

    desired_global_capacity = torch.exp(target_capacity) * init_capacity + min_capacity
    output_scaling = 1 - torch.exp(-desired_global_capacity / dimensionality * 2)

    local_mean = torch.mean(local_capacity_adjustment, dim=all_but_batch_and_channel, keepdim=True)
    target_capacity_viewed = target_capacity.view(batch_size, *((1,) * num_spatial_dims), -1)
    local_capacity_adjustment = (target_capacity_viewed + local_capacity_adjustment - local_mean)

    desired_local_capacity = torch.exp(local_capacity_adjustment) * init_capacity + min_capacity

    noise_std = torch.exp(-desired_local_capacity / dimensionality)
    noise_var = noise_std ** 2

    stable_sqrt1memx = lambda x: torch.where(x > 20, 1, torch.sqrt(1 - torch.exp(-x)))
    signal_std = stable_sqrt1memx(desired_local_capacity / dimensionality * 2)
    signal_var = 1 - noise_var

    mean_mean = torch.mean(mean, dim=all_but_batch_and_channel, keepdim=True)
    normalized_mean = mean - mean_mean
    normalized_mean = normalized_mean / torch.sqrt(torch.mean(normalized_mean ** 2 + 1e-8, dim=all_but_batch_and_channel, keepdim=True))

    z = signal_std * normalized_mean

    output_scaling = output_scaling.view(batch_size, *((1,) * num_spatial_dims), -1)
    z = output_scaling * z

    KL = 0.5 * (noise_var + signal_var * normalized_mean ** 2 - 1) + desired_local_capacity / dimensionality
    return z, KL


def batched_affine(x, weight, use_bias=False):
    num_spatial = len(x.shape) - 2
    w = weight[0].view(*weight[0].shape[:-2], *((1,) * num_spatial), *weight[0].shape[-2:])
    x = torch.matmul(x, w)
    if use_bias:
        b = weight[1].view(*weight[1].shape[:-1], *((1,) * num_spatial), weight[1].shape[-1])
        x = x + b
    return x


def decode_latents(target_capacities, decode_weights, multiposteriors):
    """
    Batched version of decode_latents. Uses channel_layer and the original affine,
    which handles batched inputs via broadcasting.
    """
    KL_amounts = []
    KL_names = []

    @multitensor_systems.multify
    def decode_latents_(dims, target_capacity, decode_weight, posterior):
        z, KL = channel_layer(target_capacity, posterior)
        x = batched_affine(z, decode_weight, use_bias=True)
        KL_amounts.append(KL)
        KL_names.append(str(dims))
        return x

    x = decode_latents_(target_capacities, decode_weights, multiposteriors)
    return x, KL_amounts, KL_names 