import numpy as np
import torch
import multitensor_systems

import initializers

np.random.seed(0)
torch.manual_seed(0)


class Initializer(initializers.Initializer):
    def __init__(self, multitensor_system, channel_dim_fn, batch_size=1, batch_weights=False):
        super().__init__(multitensor_system, channel_dim_fn)
        self.batch_size = batch_size
        self.batch_weights = batch_weights

    def initialize_zeros(self, dims, shape):
        if callable(shape):
            shape = shape(dims)
        batched_shape = (self.batch_size,) + tuple(shape)
        zeros = torch.zeros(batched_shape, requires_grad=True)
        self.weights_list.append(zeros)
        return zeros

    def initialize_posterior(self, dims, channel_dim):
        if callable(channel_dim):
            channel_dim = channel_dim(dims)
        shape = self.multitensor_system.shape(dims, channel_dim)
        batched_shape = (self.batch_size,) + tuple(shape)
        mean = 0.01 * torch.randn(batched_shape)
        mean.requires_grad = True
        local_capacity_adjustment = self.initialize_zeros(dims, shape)
        self.weights_list.append(mean)
        return [mean, local_capacity_adjustment]

    def initialize_linear(self, dims, shape):
        if callable(shape):
            shape = shape(dims)
        n_in, n_out = shape
        if callable(n_in):
            n_in = n_in(dims)
        if callable(n_out):
            n_out = n_out(dims)
        scale = 1 / np.sqrt(n_in)
        if self.batch_weights:
            weight = scale * torch.randn(self.batch_size, n_in, n_out)
            bias = scale * torch.randn(self.batch_size, n_out)
        else:
            weight = scale * torch.randn(n_in, n_out)
            bias = scale * torch.randn(n_out)
        weight.requires_grad = True
        bias.requires_grad = True
        self.weights_list.extend([weight, bias])
        return [weight, bias] 