import numpy as np
import torch
import multitensor_systems

import initializers

np.random.seed(0)
torch.manual_seed(0)


class Initializer(initializers.Initializer):
    def __init__(self, multitensor_system, channel_dim_fn, batch_size=8, batch_weights=True):
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
    
    def initialize_single_tensor(self, dims, channel_dim):
        if callable(channel_dim):
            channel_dim = channel_dim(dims)
        shape = self.multitensor_system.shape(dims, channel_dim)
        batched_shape = (self.batch_size,) + tuple(shape)
        single_tensor = torch.randn(batched_shape, requires_grad=True)
        self.weights_list.append(single_tensor)
        return single_tensor
    
    def initialize_multisingle_tensor(self, channel_dim):
        return multitensor_systems.multify(self.initialize_single_tensor)(
            self.multitensor_system.make_multitensor(default=channel_dim)
        )

    def initialize_head(self):
        """Batched version of initialize_head that includes a leading batch dimension.
        This is the same as the base Initializer, only difference being that self.initialize_linear is batched version.
        """
        dims = [1, 1, 0, 1, 1]

        # Use the overridden initialize_linear so weight/bias follow the batching setting
        head_weights = self.initialize_linear(dims, [self.channel_dim_fn(dims), 2])

        # Enforce symmetry w.r.t swapping x and y dimensions on the weight tensor
        W = head_weights[0]  # shape: (batch, n_in, 2) if batched else (n_in, 2)
        # Temporarily disable grad so that W_sym is a leaf node, rather than intermediate node (grad will be backproped into W)
        W.requires_grad = False
        W_sym = torch.stack([W[..., 0]] * 2, dim=-1)  # Duplicate first output channel
        head_weights[0] = W_sym
        head_weights[0].requires_grad = True  # Re-enable gradients
        if self.batch_weights:
            # + 100 * self.head_weights[1] in arc_compressor.py
            head_weights[1] = head_weights[1].view(self.batch_size, 1, 1, 1, 1, 2)
        # Keep weights_list consistent (weight is the second-to-last element that was appended)
        self.weights_list[-2] = head_weights[0]
        return head_weights
        