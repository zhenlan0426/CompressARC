import itertools

import numpy as np
import torch

import multitensor_systems

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

@multitensor_systems.multify
def affine(dims, x, weight, use_bias=False):
    batch_size = x.shape[0]
    num_spatial = len(x.shape) - 3
    w = weight[0]
    if w.dim() == 3:
        # batch matmul
        w = w.view(batch_size, *((1,) * num_spatial), *w.shape[-2:])
    x = torch.matmul(x, w)
    if use_bias:
        b = weight[1]
        if b.dim() == 2:
            b = b.view(batch_size, *((1,) * (num_spatial+1)), b.shape[-1])
        x = x + b
    return x

# layer is a leaf function, i.e. it takes one of the multitensor and returns the same type of multitensor
# leaf functions needs to take dims as the first argument. add_residual takes a leaf function and returns a new leaf function.
# @multitensor_systems.multify
# @add_residual
# def softmax(dims, x):...
# softmax is a leaf function -> add_residual(softmax) is a new leaf function. -> multify(add_residual(softmax)) applies
# to all the tensors in the multitensor system.
def add_residual(layer):
    """
    Surround a layer/operation with a residual connection, up and down projections,
    and pre/post-norms.
    Args:
        layer (Callable): The layer/operation to modify.
    Returns:
        Callable: Another layer/operation that applies the original layer with the
                above modifications.
    """
    def layer_with_residual(dims, x, residual_weights, *args,
                            use_bias=False, pre_norm=False, post_norm=False, **kwargs):
        if pre_norm:
            z = normalize(x)
        z = affine(x, residual_weights[0], use_bias=use_bias)
        z = layer(dims, z, *args, **kwargs)
        if post_norm:
            z = normalize(z)
        z = affine(z, residual_weights[1], use_bias=use_bias)
        return x + z
    return layer_with_residual

def channel_layer(target_capacity, posterior):
    """
    Batched version of channel_layer. Assumes inputs have a prepended batch dimension.
    Operations are vectorized over the batch dim.
    Args:
        target_capacity (Tensor): of shape (batch_size, decoding_dim)
        posterior (Tensor, Tensor): of shape (batch_size, *spatial_dims, decoding_dim)
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
    target_capacity = target_capacity.view(batch_size, *((1,) * num_spatial_dims), -1)
    target_capacity = 10 * target_capacity

    desired_global_capacity = torch.exp(target_capacity) * init_capacity + min_capacity
    output_scaling = 1 - torch.exp(-desired_global_capacity / dimensionality * 2)

    local_mean = torch.mean(local_capacity_adjustment, dim=all_but_batch_and_channel, keepdim=True)
    local_capacity_adjustment = (target_capacity + local_capacity_adjustment - local_mean)

    desired_local_capacity = torch.exp(local_capacity_adjustment) * init_capacity + min_capacity

    noise_std = torch.exp(-desired_local_capacity / dimensionality)
    noise_var = noise_std ** 2

    stable_sqrt1memx = lambda x: torch.where(x > 20, 1, torch.sqrt(1 - torch.exp(-x)))
    signal_std = stable_sqrt1memx(desired_local_capacity / dimensionality * 2)
    signal_var = 1 - noise_var

    normalized_mean = mean - torch.mean(mean, dim=all_but_batch_and_channel, keepdim=True)
    normalized_mean = normalized_mean / torch.sqrt(torch.mean(normalized_mean ** 2 + 1e-8, dim=all_but_batch_and_channel, keepdim=True))

    z = signal_std * normalized_mean + noise_std * torch.randn_like(mean)

    output_scaling = output_scaling.view(batch_size, *((1,) * num_spatial_dims), -1)
    z = output_scaling * z

    KL = 0.5 * (noise_var + signal_var * normalized_mean ** 2 - 1) + desired_local_capacity / dimensionality
    return z, KL

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
        x = affine(z, decode_weight, use_bias=True)
        KL_amounts.append(KL)
        KL_names.append(str(dims))
        return x

    x = decode_latents_(target_capacities, decode_weights, multiposteriors)
    return x, KL_amounts, KL_names 

@multitensor_systems.multify
@add_residual
def softmax(dims, x):
    """
    Batched version of softmax. x has a prepended batch dimension. Do not softmax over batch.
    """
    axes = list(range(1, 1 + sum(dims)))
    if dims[0]==1:
        axes.pop(0)  # don't softmax over examples
    subsets_of_axes = []
    for subset_size in range(1, len(axes)+1):
        subsets_of_axes = subsets_of_axes + list(itertools.combinations(axes, subset_size))
    softmaxxes = []
    for subset in subsets_of_axes:
        offsets = torch.amax(x, dim=subset, keepdim=True)
        softmax = torch.exp(x-offsets)
        softmax = softmax / torch.sum(softmax, dim=subset, keepdim=True)
        softmaxxes.append(softmax)
    return torch.cat(softmaxxes, dim=-1)

def share_direction(residual, share_weights, direction):
    """
    Apply the multitensor communication layer.
    Args:
        residual (MultiTensor[Tensor]): The residual stream.
        share_weights (Multitensor[list[list[Tensor]]]): Multiresidual projection weights.
        direction (int): 1 for up, -1 for down.
    Returns:
        MultiTensor[Tensor]: The output of the multitensor communication layer.
    """
    
    # Split the multiresidual into two multilinears
    down_project_weights = multitensor_systems.multify(lambda dims, weights: weights[0])(share_weights)
    up_project_weights = multitensor_systems.multify(lambda dims, weights: weights[1])(share_weights)

    multitensor_system = residual.multitensor_system
    
    # Create a copy of the original masks to prevent any mutations
    original_masks = residual.multitensor_system.task.masks

    x = affine(residual, down_project_weights, use_bias=False)  # down-project

    # Define a different communication method depending on which way we're communicating.
    if direction == 1:  # share up
        def share(dims, _):
            lower_xs = []
            for lower_dims in multitensor_system:  # get information from all lower tensors
                # check that lower_dims lower than dims in all indices
                if all([lower_naxes <= naxes for lower_naxes, naxes in zip(lower_dims, dims)]):
                    lower_x = x[lower_dims]
                    # unsqueeze all the dimensions of lower_x until it's the same rank as x
                    for dim, (lower_naxes, naxes) in enumerate(zip(lower_dims, dims)):
                        if lower_naxes < naxes:
                            axis = sum(dims[:dim]) + 1
                            lower_x = torch.unsqueeze(lower_x, axis)
                    lower_xs.append(lower_x)
            return sum(lower_xs)
    else:  # share down
        def share(dims, _):
            higher_xs = []
            for higher_dims in multitensor_system:  # get information from all higher tensors
                # check that higher_dims higher than dims in all indices
                if all([higher_naxes >= naxes for higher_naxes, naxes in zip(higher_dims, dims)]):
                    higher_x = x[higher_dims]
                    # aggregate all the dimensions of higher_x until it's the same rank as x
                    for dim, (higher_naxes, naxes) in reversed(list(enumerate(zip(higher_dims, dims)))):
                        if higher_naxes > naxes:
                            axis = sum(higher_dims[:dim]) + 1
                            # only average over non-masked elements (top left corner)
                            if (x.multitensor_system.task.in_out_same_size or x.multitensor_system.task.all_out_same_size) and dim==3:  # be careful aggregating the x axis
                                # expand/contract masks to make the dims the same as higher_x
                                masks = original_masks # (example, x, y, in/out)
                                masks = 1-(1-masks[...,0])*(1-masks[...,1]) # 1 if either in or out is 1, (example, x, y)
                                for i in range(sum(higher_dims[1:3])):  # insert color and direction dims
                                    masks = masks[:,None,...]
                                if dims[4] == 0:  # remove y dim
                                    masks = masks[...,0]
                                masks = masks[...,None]  # add channel dim
                                masks = masks.unsqueeze(0)  # add batch dim
                                higher_x = torch.sum(higher_x*masks, dim=axis) / (torch.sum(masks, dim=axis)+1e-4)
                            elif (x.multitensor_system.task.in_out_same_size or x.multitensor_system.task.all_out_same_size) and dim==4:  # be careful aggregating the y axis
                                # expand/contract masks to make the dims the same as higher_x
                                masks = original_masks
                                masks = 1-(1-masks[...,0])*(1-masks[...,1])
                                for i in range(sum(higher_dims[1:3])):  # insert color and direction dims
                                    masks = masks[:,None,...]
                                if higher_dims[3] == 0:  # remove x dim
                                    masks = masks[...,0,:]
                                masks = masks[...,None]  # add channel dim
                                masks = masks.unsqueeze(0)  # add batch dim
                                higher_x = torch.sum(higher_x*masks, dim=axis) / (torch.sum(masks, dim=axis)+1e-4)
                            else:
                                higher_x = torch.mean(higher_x, dim=axis)
                    higher_xs.append(higher_x)
            return sum(higher_xs)
    x = multitensor_systems.multify(share)(x)  # perform the cross-tensor communication
    x = normalize(x)  # post-norm
    x = affine(x, up_project_weights, use_bias=False)  # up-project
    residual = multitensor_systems.multify(lambda dims, x, y: x+y)(residual, x)  # add residual
    return residual

def share_up(residual, share_up_weights):
    """
    Apply the multitensor communication layer, upwards.
    Args:
        residual (MultiTensor[Tensor]): The residual stream.
        share_up_weights (Multitensor[list[list[Tensor]]]): Multiresidual projection weights.
    Returns:
        MultiTensor[Tensor]: The output of the multitensor communication layer.
    """
    return share_direction(residual, share_up_weights, 1)

def share_down(residual, share_down_weights):
    """
    Apply the multitensor communication layer, downwards.
    Args:
        residual (MultiTensor[Tensor]): The residual stream.
        share_down_weights (Multitensor[list[list[Tensor]]]): Multiresidual projection weights.
    Returns:
        MultiTensor[Tensor]: The output of the multitensor communication layer.
    """
    return share_direction(residual, share_down_weights, -1)

def only_do_for_certain_shapes(*shapes):
    """
    Decorator which takes a function that is applied to every tensor in a multitensor,
    and replaces that function with the identity for select tensors in the multitensor.
    Args:
        *shapes (list[list[int]]): A list of MultiTensor dims, for which the function
                should be applied. Don't do the function if the dims for the tensor isn't
                in the list.
    """
    def decorator(fn):
        def filtered_fn(dims, x, *args, **kwargs):
            if tuple(dims) in shapes:
                return fn(dims, x, *args, **kwargs)
            else:
                return x
        return filtered_fn
    return decorator

def make_directional_layer(fn, diagonal_fn):
    """
    Take a directional function (one version made for cardinal directions and another for diagonal)
    and use it to create a directional layer that works on tensors that have a direction
    dimension.
    fn works on one direction. Use dim (x vs y) and flip (+x vs -x) to make it work on the other four directions.
    diagonal_fn works on one diagonal direction. use flip x (yes or no), flip y (yes or no) to make it work on the other three diagonal directions.
    channel split is used for each of the 8 directions to include both forward and backward.
    Args:
        fn (Callable): A directional function that takes a tensor and a dim argument.
        diagonal_fn (Callable): A directional function that takes a tensor and two dim arguments.
    Returns:
        Callable: A function that takes a tensor with a direction dimension and applies fn and
                diagonal_fn in a different direction for each slice of the tensor along the
                direction dimension.
    """
    def directional_layer(dims, x, masks):
        """
        Args:
            dims (list[int]): Ignore this argument. It will be filled in by the multify decorator.
            x (Tensor): The input to the directional layer.
            masks (Tensor): A (example, x, y, in/out) tensor of zeros and ones telling you which pixels are in-bounds.
        Returns:
            Tensor: The output of the directional layer.
        """

        batch_size = x.shape[0]
        # Create a copy of masks before modifying to avoid mutating the original
        masks = masks.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # add batch dim and expand

        # rearrange mask to fit same shape as x
        masks = 1-(1-masks[...,0])*(1-masks[...,1]) # (batch, example, x, y)
        if dims[4]==0:
            masks = masks[:,:,:,0]
        if dims[3]==0:
            masks = masks[:,:,0,:]
        # dims (example - 0, color - 1, direction - 2, x - 3, y - 4)
        for _ in range(sum(dims[1:3])):
            masks = masks.unsqueeze(2)
        masks = masks.unsqueeze(-1) # hidden channel dim
        # mask out x
        x = x*masks

        # figure out which dimension the direction dimension is
        direction_dim = 1 + sum(dims[:2])

        # make a default output tensor in case we try to do cumulative ops on a dimension that
        # is not present in the tensor x
        zero_tensor = torch.zeros_like(torch.select(x, direction_dim, 0))

        # split the channel dimension into two.
        # split the direction dimension into two.
        # for each half of the direction dimension, each index of the direction dimension corresponds
        # to either x or y, and we accumulate in those respective dimensions.
        # do the other half of the channel dimension in the reverse direction.
        # do the other half of the direction dimension in the reverse direction.
        result_tensors = []
        for channel_split in range(2):  # forward, backward
            result_list = []
            for direction_split in range(2):  # forward, backward
                for direction_ind in range(4):  # x, x+y, y, y-x
                    if direction_ind % 2 == 0:  # cardinal direction
                        cardinal_direction_ind = int(direction_ind//2)
                        if dims[3+cardinal_direction_ind]>0:
                            x_slice = torch.select(x, direction_dim, 4*direction_split+direction_ind)
                            x_slice = x_slice[...,channel_split::2]
                            masks_flipped = torch.select(masks, direction_dim, 0)
                            if direction_split + channel_split == 1:
                                # below: decrement index to account for slicing, increment index to go from direction to x
                                x_slice = torch.flip(x_slice, [direction_dim+cardinal_direction_ind])
                                masks_flipped = torch.flip(masks_flipped, [direction_dim+cardinal_direction_ind])
                            result = fn(x_slice, direction_dim+cardinal_direction_ind, masks_flipped)
                            if direction_split + channel_split == 1:
                                result = torch.flip(result, [direction_dim+cardinal_direction_ind])
                        else:
                            result = zero_tensor
                    else:  # diagonal direction
                        if dims[3] == 1 and dims[4] == 1:
                            diagonal_direction_ind = int(direction_ind//2)  # 0 for x+y, 1 for y-x
                            x_slice = torch.select(x, direction_dim, 4*direction_split+direction_ind)
                            x_slice = x_slice[...,channel_split::2]
                            masks_flipped = torch.select(masks, direction_dim, 0)
                            if (direction_split + channel_split + diagonal_direction_ind) % 2 == 1:
                                # below: decrement index to account for slicing, increment index to go from direction to x
                                x_slice = torch.flip(x_slice, [direction_dim])
                                masks_flipped = torch.flip(masks_flipped, [direction_dim])
                            if direction_split + channel_split == 1:
                                x_slice = torch.flip(x_slice, [direction_dim+1])
                                masks_flipped = torch.flip(masks_flipped, [direction_dim+1])
                            result = diagonal_fn(x_slice, direction_dim, direction_dim+1, masks_flipped)
                            if (direction_split + channel_split + diagonal_direction_ind) % 2 == 1:
                                result = torch.flip(result, [direction_dim])
                            if direction_split + channel_split == 1:
                                result = torch.flip(result, [direction_dim+1])
                        else:
                            result = zero_tensor
                    result_list.append(result)
            result_list = torch.stack(result_list, dim=direction_dim)  # stack direction dim together
            result_tensors.append(result_list)
        return torch.cat(result_tensors, dim=-1)  # cat channel dim together
    return directional_layer

"""
Function cummax

Apply the directional cummax layer.
Args:
    x (MultiTensor[Tensor]): The input to the cummax layer.
    weights (MultiTensor[list[list[Tensor]]]): Multiresidual projection weights surrounding the cummax operations.
            Implicitly introduced by the add_residual decorator.
    masks (Tensor): A (example, x, y, in/out) tensor of zeros and ones telling you which pixels are in-bounds.
    Other boolean kwargs such as pre_norm, post_norm, use_bias, introduced by the add_residual decorator.
Returns:
    MultiTensor[Tensor]: The output of the cummax layer.
"""
def cummax_(x, dim, masks):
    masks = 1e3*(1-masks)
    max_ = torch.max(x-masks, dim=dim, keepdim=True)[0] + masks + 1e-3
    min_ = torch.min(x+masks, dim=dim, keepdim=True)[0] - masks - 1e-3
    x = torch.cummax(x-masks, dim=dim)[0] + masks
    return (x - min_) / (max_-min_) * 2 - 1
def diagonal_cummax_(x, dim1, dim2, masks):
    masks_ = 1e3*(1-masks)
    min_dim = min(x.shape[dim1], x.shape[dim2])
    n_iters = int(np.ceil(np.log2(min_dim)))
    # compute the cummax and max via forward+backward associative scan
    # unlike typical parallel scan, we don't have sparse updates, e.g. for length 8, instead of updating 1,3,5,7 in first iteration
    # and then 3, 7 and then 7, we update 0~7 in first iteration, 1~7 in second iteration, 3~7 in third iteration
    # as a result, we dont need backward scan to get the prefix max. it is only needed to "broadcast" the max per diagnal to normalize.
    max_x = x - masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign*2**i
            shifted_x = diagonal_shift_(max_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=-1e3)
            # M[i,j] = max(M[i,j], M[i-2^iterations,j-2^iterations]) for i,j >= 2^iterations
            max_x = torch.max(max_x, shifted_x)
        if sign == 1:  # save the cummax after the forward associative scan
            cummax_x = max_x + masks_
    max_x = max_x + masks_
    # compute the min via forward+backward associative scan
    min_x = x + masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign*2**i
            shifted_x = diagonal_shift_(min_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=1e3)
            min_x = torch.min(min_x, shifted_x)
    min_x = min_x - masks_
    return ((cummax_x - min_x) / (max_x-min_x+1e-5) * 2 - 1)*masks  # rescale the cummax to fit the max and min
cummax = multitensor_systems.multify(  # apply decorators
         only_do_for_certain_shapes((1,1,1,1,1), (1,0,1,1,1))(
         add_residual(
         make_directional_layer(
         cummax_, diagonal_cummax_
         ))))

"""
Function shift

Apply the directional shift layer.
Args:
    x (MultiTensor[Tensor]): The input to the shift layer.
    weights (MultiTensor[list[list[Tensor]]]): Multiresidual projection weights surrounding the shift operations.
            Implicitly introduced by the add_residual decorator.
    masks (Tensor): A (example, x, y, in/out) tensor of zeros and ones telling you which pixels are in-bounds.
    Other boolean kwargs such as pre_norm, post_norm, use_bias, introduced by the add_residual decorator.
Returns:
    MultiTensor[Tensor]: The output of the shift layer.
"""
def shift_(x, dim, masks):
    padding = torch.zeros_like(torch.narrow(x, dim, 0, 1))
    narrowed = torch.narrow(x, dim, 0, x.shape[dim]-1)
    return torch.cat([padding, narrowed], dim=dim)
def diagonal_shift_(x, dim1, dim2, masks, shift_amount=1, pad_value=0):
    for dim in (dim1, dim2):
        padding = pad_value+torch.zeros_like(torch.narrow(x, dim, 0, abs(shift_amount)))
        if shift_amount >= 0:
            narrowed = torch.narrow(x, dim, 0, x.shape[dim]-shift_amount)
            x = torch.cat([padding, narrowed], dim=dim)
        else:
            narrowed = torch.narrow(x, dim, -shift_amount, x.shape[dim]+shift_amount)
            x = torch.cat([narrowed, padding], dim=dim)
    return x
shift = multitensor_systems.multify(  # apply decorators
        only_do_for_certain_shapes((1,1,1,1,1), (1,0,1,1,1))(
        add_residual(
        make_directional_layer(
        shift_, diagonal_shift_
        ))))

directional_dims = [(i,j,1,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
# @multitensor_systems.multify
# @only_do_for_certain_shapes(*directional_dims)
# def direction_share(dims, x, weights, pre_norm=True, use_bias=False):
#     """
#     Apply the directional communication layer.
#     Args:
#         dims (list[int]): Ignore this argument. It will be filled in by the multify decorator.
#         x (MultiTensor[Tensor]): The input to the directional communication layer.
#         weights (MultiTensor[list[list[list[Tensor]]]]): A multitensor full of linear layer weights
#                 for every pair of directions.
#     Returns:
#         MultiTensor[Tensor]: The output of the directional communication layer.
#     """
#     # Optionally normalize the input
#     z = normalize(x) if pre_norm else x
#     x_new = x.clone()
#     n_directions = dims[3] + dims[4]
#     direction_dim = -2 - n_directions

#     # Precomputed coefficients for the directional shift.
#     coefficients = [1, 0.2, 0.4, 0.2, 1, 0.2, 0.4, 0.2]

#     # Loop over all pairs of directions.
#     for d1 in range(8):
#         for d2 in range(8):
#             # Determine the appropriate coefficient.
#             c = coefficients[(d2 - d1) % 8]
#             # Apply the affine transformation for this pair and accumulate.
#             update = c * affine(z.narrow(direction_dim, d2, 1), weights[d1][d2], use_bias=use_bias)
#             x_new.narrow(direction_dim, d1, 1).add_(update)

#     # Reassemble the tensor along the original direction dimension.
#     return x_new

@multitensor_systems.multify
@only_do_for_certain_shapes(*directional_dims)
def direction_share(dims, x, weights, pre_norm=True, use_bias=False):
    """
    Apply the directional communication layer.
    Args:
        dims (list[int]): Ignore this argument. It will be filled in by the multify decorator.
        x (MultiTensor[Tensor]): The input to the directional communication layer.
        weights (MultiTensor[list[list[list[Tensor]]]]): A multitensor full of linear layer weights
                for every pair of directions.
    Returns:
        MultiTensor[Tensor]: The output of the directional communication layer.
    """
    # Optionally normalize the input
    z = normalize(x) if pre_norm else x
    num_spatial = dims[3] + dims[4]
    direction_dim = -2 - num_spatial
    # Move direction dim to -2 for vectorized operations
    x_stacked = x.movedim(direction_dim, -2)
    z_stacked = z.movedim(direction_dim, -2)

    # Create coefficients tensor (8x8, based on circular differences)
    coefficients = [1, 0.2, 0.4, 0.2, 1, 0.2, 0.4, 0.2]
    c_tensor = torch.tensor(
        [coefficients[(j - i) % 8] for i in range(8) for j in range(8)],
        device=x.device,
        dtype=x.dtype,
    ).view(8, 8)

    # Stack weights into tensors for vectorized matmul (8 d1, 8 d2, in, out)
    w_list = [[weights[d1][d2][0] for d2 in range(8)] for d1 in range(8)] # pick out weight and ignore bias
    w_stacked = torch.stack([torch.stack(d2_weights, dim=0) for d2_weights in w_list], dim=0) 
    k = w_stacked.ndim
    # Incorporate coefficients into weights
    while c_tensor.ndim < k:
        c_tensor = c_tensor[..., None]
    w_stacked = w_stacked * c_tensor
    
    # Vectorized matmul: sum_d2 c[d1,d2] * (z_d2 @ w[d1,d2])
    if k == 4: # unbatched weights, (8, 8, d_in, d_out)
        addition = torch.einsum("... j m, i j m n -> ... i n", z_stacked, w_stacked)
    else: # batched, (8, 8, b, d_in, d_out)
        addition = torch.einsum("b ... j m, i j b m n -> b ... i n", z_stacked, w_stacked)

    # Add to original x and move direction dim back
    output_stacked = x_stacked + addition
    return output_stacked.movedim(-2, direction_dim)

@multitensor_systems.multify
@add_residual
def nonlinear(dims, x):
    """
    Apply the nonlinear layer.
    Args:
        dims (list[int]): Ignore this argument. It will be filled in by the multify decorator.
        x (MultiTensor[Tensor]): The input to the nonlinear layer.
        weights (MultiTensor[list[list[Tensor]]]): Multiresidual projection weights surrounding the nonlinear operations.
                Implicitly introduced by the add_residual decorator.
        Other boolean kwargs such as pre_norm, post_norm, use_bias, introduced by the add_residual decorator.
    Returns:
        MultiTensor[Tensor]: The output of the nonlinear layer.
    """
    return torch.nn.functional.silu(x)