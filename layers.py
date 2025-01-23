import itertools

import numpy as np
import torch

import multitensor_systems


np.random.seed(0)
torch.manual_seed(0)


@multitensor_systems.multify
def normalize(dims, x, debias=True):
    all_but_last = list(range(len(x.shape)-1))
    if debias:
        x = x - torch.mean(x, dim=all_but_last)
    x = x / torch.sqrt(1e-8+torch.mean(x**2, dim=all_but_last))
    return x

@multitensor_systems.multify
def normalize_range(dims, x, debias=True):
    all_but_last = list(range(len(x.shape)-1))
    x = x - torch.mean(x, dim=all_but_last)
    max_x = torch.abs(x)
    for dim in reversed(all_but_last):
        max_x = torch.max(max_x, dim=dim)[0]
    x = x / (1e-8+max_x)
    return x

@multitensor_systems.multify
def affine(dims, x, weight, use_bias=False):
    x = torch.matmul(x, weight[0])
    if use_bias:
        x = x + weight[1]
    return x

def add_residual(layer):
    def layer_with_residual(dims, x, residual_weights, *args, use_bias=False, pre_norm=False, post_norm=False, **kwargs):
        if pre_norm:
            z = normalize(x)
        z = affine(x, residual_weights[0], use_bias=use_bias)
        z = layer(dims, z, *args, **kwargs)
        if post_norm:
            z = normalize(z)
        z = affine(z, residual_weights[1], use_bias=use_bias)
        return x + z
    return layer_with_residual

def channel_layer(global_capacity_adjustment, posterior):
    mean, local_capacity_adjustment = posterior

    all_but_last_dim = tuple(range(len(mean.shape)-1))
    dimensionality = 1
    for axis_length in mean.shape:
        dimensionality *= axis_length
    min_capacity = 0.5
    init_capacity = 10000
    min_capacity = torch.tensor(min_capacity)
    init_capacity = torch.tensor(init_capacity)

    global_capacity_adjustment = 10*global_capacity_adjustment

    desired_global_capacity = torch.exp(global_capacity_adjustment)*init_capacity + min_capacity
    output_scaling = 1-torch.exp(-desired_global_capacity / dimensionality * 2)

    local_capacity_adjustment = (global_capacity_adjustment + 
                                 local_capacity_adjustment - 
                                 torch.mean(local_capacity_adjustment, dim=all_but_last_dim))
    desired_local_capacity = torch.exp(local_capacity_adjustment)*init_capacity + min_capacity
    noise_std = torch.exp(-desired_local_capacity / dimensionality)
    noise_var = noise_std**2
    stable_sqrt1memx = lambda x: torch.where(x>20, 1, torch.sqrt(1-torch.exp(-x)))
    signal_std = stable_sqrt1memx(desired_local_capacity / dimensionality * 2)
    signal_var = 1-noise_var
    normalized_mean = mean - torch.mean(mean, dim=all_but_last_dim)
    normalized_mean = normalized_mean / torch.sqrt(torch.mean(normalized_mean**2+1e-8, dim=all_but_last_dim))

    z = signal_std*normalized_mean + noise_std*torch.randn(normalized_mean.shape)
    z = output_scaling*z

    KL = 0.5*(noise_var + signal_var*normalized_mean**2 - 1) + desired_local_capacity/dimensionality
    return z, KL

def decode_latents(global_capacity_adjustments, decode_weights, multiposteriors):
    KL_amounts = []
    KL_names = []

    @multitensor_systems.multify
    def decode_latents_(dims, global_capacity_adjustment, decode_weight, posterior):
        z, KL = channel_layer(global_capacity_adjustment, posterior)
        x = affine(z, decode_weight, use_bias=True)
        KL_amounts.append(KL)
        KL_names.append(str(dims))
        return x
    x = decode_latents_(global_capacity_adjustments, decode_weights, multiposteriors)
    return x, KL_amounts, KL_names


def share_direction(residual, share_weights, direction):
    down_project_weights = multitensor_systems.multify(lambda dims, weights: weights[0])(share_weights)
    up_project_weights = multitensor_systems.multify(lambda dims, weights: weights[1])(share_weights)

    multitensor_system = residual.multitensor_system

    x = affine(residual, down_project_weights, use_bias=False)
    if direction == 1:  # share up
        def share(dims, _):
            lower_xs = []
            for lower_dims in multitensor_system:
                # check that lower_dims lower than dims in all indices
                if all([lower_naxes <= naxes for lower_naxes, naxes in zip(lower_dims, dims)]):
                    lower_x = x[lower_dims]
                    # unsqueeze all the dimensions of lower_x until it's the same rank as x
                    for dim, (lower_naxes, naxes) in enumerate(zip(lower_dims, dims)):
                        if lower_naxes < naxes:
                            axis = sum(dims[:dim], 0)
                            lower_x = torch.unsqueeze(lower_x, axis)
                    lower_xs.append(lower_x)
            return sum(lower_xs)
    else:  # share down
        def share(dims, _):
            higher_xs = []
            for higher_dims in multitensor_system:
                # check that higher_dims higher than dims in all indices
                if all([higher_naxes >= naxes for higher_naxes, naxes in zip(higher_dims, dims)]):
                    higher_x = x[higher_dims]
                    # aggregate all the dimensions of higher_x until it's the same rank as x
                    for dim, (higher_naxes, naxes) in reversed(list(enumerate(zip(higher_dims, dims)))):
                        if higher_naxes > naxes:
                            axis = sum(higher_dims[:dim], 0)
                            if (x.multitensor_system.task.in_out_same_size or x.multitensor_system.task.all_out_same_size) and dim==3:  # be careful aggregating the x axis
                                # expand/contract masks to make the dims the same as higher_x
                                masks = x.multitensor_system.task.masks
                                masks = 1-(1-masks[...,0])*(1-masks[...,1])
                                for i in range(sum(higher_dims[1:3])):  # insert color and direction dims
                                    masks = masks[:,None,...]
                                if dims[4] == 0:  # remove y dim
                                    masks = masks[...,0]
                                masks = masks[...,None]  # add vector dim
                                higher_x = torch.sum(higher_x*masks, dim=axis) / (torch.sum(masks, dim=axis)+1e-4)
                            elif (x.multitensor_system.task.in_out_same_size or x.multitensor_system.task.all_out_same_size) and dim==4:  # be careful aggregating the y axis
                                # expand/contract masks to make the dims the same as higher_x
                                masks = x.multitensor_system.task.masks
                                masks = 1-(1-masks[...,0])*(1-masks[...,1])
                                for i in range(sum(higher_dims[1:3])):  # insert color and direction dims
                                    masks = masks[:,None,...]
                                if higher_dims[3] == 0:  # remove x dim
                                    masks = masks[...,0,:]
                                masks = masks[...,None]  # add vector dim
                                higher_x = torch.sum(higher_x*masks, dim=axis) / (torch.sum(masks, dim=axis)+1e-4)
                            else:
                                higher_x = torch.mean(higher_x, dim=axis)
                    higher_xs.append(higher_x)
            return sum(higher_xs)
    x = multitensor_systems.multify(share)(x)
    x = normalize(x)
    x = affine(x, up_project_weights, use_bias=False)
    residual = multitensor_systems.multify(lambda dims, x, y: x+y)(residual, x)
    return residual

def share_up(residual, share_up_weights):
    return share_direction(residual, share_up_weights, 1)

def share_down(residual, share_down_weights):
    return share_direction(residual, share_down_weights, -1)


def only_do_for_certain_shapes(*shapes):
    def decorator(fn):
        def filtered_fn(dims, x, *args, **kwargs):
            if tuple(dims) in shapes:
                return fn(dims, x, *args, **kwargs)
            else:
                return x
        return filtered_fn
    return decorator


@multitensor_systems.multify
@add_residual
def softmax(dims, x):
    axes = list(range(sum(dims)))
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


def make_cumulative_layer(fn, diagonal_fn):
    def cumulative_layer(dims, x, masks):

        # rearrange mask to fit same shape as x
        masks = 1-(1-masks[...,0])*(1-masks[...,1])
        if dims[4]==0:
            masks = masks[:,:,0]
        if dims[3]==0:
            masks = masks[:,0,...]
        for i in range(sum(dims[1:3])):
            masks = masks[:,None,...]
        masks = masks[...,None]
        # mask out x
        x = x*masks

        # figure out which dimension the direction dimension is
        n_directions = dims[3]+dims[4]
        direction_dim = sum(dims[:2])

        # make a default output tensor in case we try to do cumulative ops on a dimension that
        # is not present in the tensor x
        zero_tensor = torch.zeros_like(torch.select(x, direction_dim, 0))

        # split the vector dimension into two.
        # split the direction dimension into two.
        # for each half of the direction dimension, each index of the direction dimension corresponds
        # to either x or y, and we accumulate in those respective dimensions.
        # do the other half of the vector dimension in the reverse direction.
        # do the other half of the direction dimension in the reverse direction.
        result_tensors = []
        for vector_split in range(2):  # forward, backward
            result_list = []
            for direction_split in range(2):  # forward, backward
                for direction_ind in range(4):  # x, x+y, y, y-x
                    if direction_ind % 2 == 0:  # cardinal direction
                        cardinal_direction_ind = int(direction_ind//2)
                        if dims[3+cardinal_direction_ind]>0:
                            x_slice = torch.select(x, direction_dim, 4*direction_split+direction_ind)
                            x_slice = x_slice[...,vector_split::2]
                            masks_flipped = torch.select(masks, direction_dim, 0)
                            if direction_split + vector_split == 1:
                                # below: decrement index to account for slicing, increment index to go from direction to x
                                x_slice = torch.flip(x_slice, [direction_dim+cardinal_direction_ind])
                                masks_flipped = torch.flip(masks_flipped, [direction_dim+cardinal_direction_ind])
                            result = fn(x_slice, direction_dim+cardinal_direction_ind, masks_flipped)
                            if direction_split + vector_split == 1:
                                result = torch.flip(result, [direction_dim+cardinal_direction_ind])
                        else:
                            result = zero_tensor
                    else:  # diagonal direction
                        if dims[3] == 1 and dims[4] == 1:
                            diagonal_direction_ind = int(direction_ind//2)  # 0 for x+y, 1 for y-x
                            x_slice = torch.select(x, direction_dim, 4*direction_split+direction_ind)
                            x_slice = x_slice[...,vector_split::2]
                            masks_flipped = torch.select(masks, direction_dim, 0)
                            if (direction_split + vector_split + diagonal_direction_ind) % 2 == 1:
                                # below: decrement index to account for slicing, increment index to go from direction to x
                                x_slice = torch.flip(x_slice, [direction_dim])
                                masks_flipped = torch.flip(masks_flipped, [direction_dim])
                            if direction_split + vector_split == 1:
                                x_slice = torch.flip(x_slice, [direction_dim+1])
                                masks_flipped = torch.flip(masks_flipped, [direction_dim+1])
                            result = diagonal_fn(x_slice, direction_dim, direction_dim+1, masks_flipped)
                            if (direction_split + vector_split + diagonal_direction_ind) % 2 == 1:
                                result = torch.flip(result, [direction_dim])
                            if direction_split + vector_split == 1:
                                result = torch.flip(result, [direction_dim+1])
                        else:
                            result = zero_tensor
                    result_list.append(result)
            result_list = torch.stack(result_list, dim=direction_dim)  # stack direction dim together
            result_tensors.append(result_list)
        return torch.cat(result_tensors, dim=-1)  # cat vector dim together
    return cumulative_layer

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
    max_x = x - masks_
    for sign in (1, -1):
        for i in range(n_iters):
            shift_amount = sign*2**i
            shifted_x = diagonal_shift_(max_x, dim1, dim2, masks_, shift_amount=shift_amount, pad_value=-1e3)
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
    return ((cummax_x - min_x) / (max_x-min_x+1e-5) * 2 - 1)*masks
cummax = multitensor_systems.multify(  # apply decorators
         only_do_for_certain_shapes((1,1,1,1,1), (1,0,1,1,1))(
         add_residual(
         make_cumulative_layer(
         cummax_, diagonal_cummax_
         ))))

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
        make_cumulative_layer(
        shift_, diagonal_shift_
        ))))

directional_dims = [(i,j,1,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
@multitensor_systems.multify
@only_do_for_certain_shapes(*directional_dims)
def direction_share(dims, x, weights, pre_norm=True, use_bias=False):
    if pre_norm:
        z = normalize(x)
    else:
        z = x
    n_directions = dims[3]+dims[4]
    direction_dim = -2-n_directions
    x = [torch.select(x, direction_dim, direction_ind) for direction_ind in range(8)]
    for direction_ind1 in range(8):
        for direction_ind2 in range(8):
            coefficient = [1, 0.2, 0.4, 0.2, 1, 0.2, 0.4, 0.2][(direction_ind2-direction_ind1) % 8]
            z_slice = torch.select(z, direction_dim, direction_ind2)
            z_slice = affine(z_slice, weights[direction_ind1][direction_ind2], use_bias=use_bias)
            x[direction_ind1] = x[direction_ind1] + coefficient*z_slice
    x = torch.stack(x, dim=direction_dim)
    return x

@multitensor_systems.multify
@add_residual
def nonlinear(dims, x):
    return torch.nn.functional.silu(x)

def postprocess_mask(task, x_mask, y_mask):
    x_mask_modifier = np.zeros([task.n_examples, task.n_x, 2])
    y_mask_modifier = np.zeros([task.n_examples, task.n_y, 2])
    for example_num in range(task.n_examples):
        max_length = max(task.shapes[example_num][0][0], task.shapes[example_num][1][0])
        for in_out_mode in range(2):
            x_mask_modifier[example_num,max_length:,in_out_mode] = -1000
        max_length = max(task.shapes[example_num][0][1], task.shapes[example_num][1][1])
        for in_out_mode in range(2):
            y_mask_modifier[example_num,max_length:,in_out_mode] = -1000
    x_mask = x_mask+torch.from_numpy(x_mask_modifier).to(x_mask.device).to(x_mask.dtype)
    y_mask = y_mask+torch.from_numpy(y_mask_modifier).to(y_mask.device).to(y_mask.dtype)
    return x_mask, y_mask
