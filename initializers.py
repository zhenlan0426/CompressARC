import numpy as np
import torch

import multitensor_systems


np.random.seed(0)
torch.manual_seed(0)


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
        vector_dim_fn = self.vector_dim_fn
        linear_1 = self.initialize_linear(dims, [vector_dim_fn, n_in])
        linear_2 = self.initialize_linear(dims, [n_out, vector_dim_fn])
        return [linear_1, linear_2]

    def initialize_posterior(self, dims, channel_dim):
        if callable(channel_dim):
            channel_dim = channel_dim(dims)
        shape = self.multitensor_system.shape(dims, channel_dim)
        mean = 0.01*torch.randn(shape)
        local_capacity_adjustment = self.initialize_zeros(dims, shape)
        mean.requires_grad = True
        self.weights_list.append(mean)
        return [mean, local_capacity_adjustment]

    def initialize_direction_share(self, dims, _):
        vector_dim_fn = self.vector_dim_fn
        linears = []
        for direction_dim1 in range(8):
            linears.append([])
            for direction_dim2 in range(8):
                linears[direction_dim1].append(self.initialize_linear(dims, [vector_dim_fn, vector_dim_fn]))

        return linears

    def initialize_head(self):
        head_weights = self.initialize_linear([1, 1, 0, 1, 1], [self.vector_dim_fn([1, 1, 0, 1, 1]), 2])
        self.weights_list.pop()  # remove bias
        self.weights_list.pop()  # remove weight
        head_weights[0].requires_grad = False
        head_weights[0] = torch.stack([head_weights[0][...,0]]*2, dim=-1)
        head_weights[0].requires_grad = True
        self.weights_list.append(head_weights[0])
        self.weights_list.append(head_weights[1])
        return head_weights


    def initialize_multizeros(self, shape):
        return multitensor_systems.multify(self.initialize_zeros)(self.multitensor_system.make_multitensor(default=shape))
    def initialize_multilinear(self, shape):
        return multitensor_systems.multify(self.initialize_linear)(self.multitensor_system.make_multitensor(default=shape))
    def initialize_multiresidual(self, n_in, n_out):
        return multitensor_systems.multify(self.initialize_residual)(n_in, self.multitensor_system.make_multitensor(default=n_out))
    def initialize_multiposterior(self, decoding_dim):
        return multitensor_systems.multify(self.initialize_posterior)(self.multitensor_system.make_multitensor(default=decoding_dim))
    def initialize_multidirection_share(self):
        return multitensor_systems.multify(self.initialize_direction_share)(self.multitensor_system.make_multitensor())


    def symmetrize_xy(self, multiweights):
        for dims in self.multitensor_system:
            if dims[3] == 0 and dims[4] == 1:
                from_dims = dims[:3] + [1, 0]
                multiweights[dims] = multiweights[from_dims]

    def symmetrize_direction_sharing(self, multiweights):
        for dims in self.multitensor_system:
            for direction_ind1 in range(8):
                for direction_ind2 in range(8):
                    from_direction_ind1 = direction_ind1
                    from_direction_ind2 = direction_ind2
                    if dims[3] + dims[4] == 1:  # one spatial dimension, symmetry wrt horizontal and vertical flip
                        from_dims = dims[:3] + [1, 0]
                        if dims[4] == 1:  # if y axis is the spatial one, tie the tensor to the x axis one but rotated 90 degrees
                            from_direction_ind1 = (2+from_direction_ind1) % 8
                            from_direction_ind2 = (2+from_direction_ind2) % 8
                        # figure out which tensor is the canonical one we should tie the weights to
                        flip_y = from_direction_ind1 > 4 or from_direction_ind1 in (0, 4) and from_direction_ind2 > 4
                        if flip_y:
                            from_direction_ind1 = (8-from_direction_ind1) % 8
                            from_direction_ind2 = (8-from_direction_ind2) % 8
                        flip_x = 2 < from_direction_ind1 < 6 or from_direction_ind1 in (2, 6) and 2 < from_direction_ind2 < 6
                        if flip_x:
                            from_direction_ind1 = (4-from_direction_ind1) % 8
                            from_direction_ind2 = (4-from_direction_ind2) % 8
                    else:  # no spatial dimensions, full D4 symmetry
                        from_dims = dims
                        # figure out which tensor is the canonical one we should tie the weights to
                        rotation = int((from_direction_ind1//2)*2)
                        from_direction_ind1 = (from_direction_ind1 - rotation) % 8
                        from_direction_ind2 = (from_direction_ind2 - rotation) % 8
                        flip = (from_direction_ind2 - from_direction_ind1) % 8 > 4
                        if flip:
                            from_direction_ind2 = (8+2*from_direction_ind1 - from_direction_ind2) % 8
                    multiweights[dims][direction_ind1][direction_ind2] = multiweights[from_dims][from_direction_ind1][from_direction_ind2]
