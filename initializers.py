import numpy as np
import torch

import tensor_algebra

class Initializer():

    def __init__(self, multitensor_system, vector_dim_fn):
        self.multitensor_system = multitensor_system
        self.vector_dim_fn = vector_dim_fn
        self.weights_list = []

    def initialize_zeros(self, dims, shape):
        if callable(shape):
            shape = shape(dims)
        zeros = torch.zeros(shape, requires_grad=True)
        self.weights_list.append(zeros)
        return zeros

    def initialize_linear(self, dims, shape):
        n_in, n_out = shape
        if callable(n_in):
            n_in = n_in(dims)
        if callable(n_out):
            n_out = n_out(dims)
        weight = 1/np.sqrt(n_in)*torch.randn([n_in, n_out])
        bias = 1/np.sqrt(n_in)*torch.randn([n_out])
        weight.requires_grad = True
        bias.requires_grad = True
        self.weights_list.append(weight)
        self.weights_list.append(bias)
        return [weight, bias]

    def initialize_residual(self, dims, n_in, n_out):
        if callable(n_in):
            n_in = n_in(dims)
        if callable(n_out):
            n_out = n_out(dims)
        vector_dim_fn = self.vector_dim_fn
        linear_1 = self.initialize_linear(dims, [vector_dim_fn, n_in])
        linear_2 = self.initialize_linear(dims, [n_out, vector_dim_fn])
        return [linear_1, linear_2]

    def initialize_posterior(self, dims, channel_dim):
        if callable(channel_dim):
            channel_dim = channel_dim(dims)
        mean = 0.01*torch.randn(self.multitensor_system.shape(dims, channel_dim))
        local_capacity_adjustment = self.initialize_zeros(dims, self.multitensor_system.shape(dims, channel_dim))
        mean.requires_grad = True
        self.weights_list.append(mean)
        return [mean, local_capacity_adjustment]

    def initialize_multizeros(self, shape):
        return tensor_algebra.use_multitensor_system_to_multify(self.multitensor_system, self.initialize_zeros)(shape)
    def initialize_multilinear(self, shape):
        return tensor_algebra.use_multitensor_system_to_multify(self.multitensor_system, self.initialize_linear)(shape)
    def initialize_multiresidual(self, n_in, n_out):
        return tensor_algebra.use_multitensor_system_to_multify(self.multitensor_system, self.initialize_residual)(n_in, n_out)
    def initialize_multiposterior(self, channel_dim):
        return tensor_algebra.use_multitensor_system_to_multify(self.multitensor_system, self.initialize_posterior)(channel_dim)
