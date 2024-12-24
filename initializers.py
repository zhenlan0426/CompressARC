import torch

def Initializer():

    def __init__(self, multitensor_system, vector_dim_fn):
        self.multitensor_system = multitensor_system
        self.vector_dim_fn = vector_dim_fn
        self.weights_list = []

    def initalize_zeros(self, dims, shape):
        if callable(shape):
            shape = shape(dims)
        zeros = torch.zeros(shape, requires_grad=True)
        self.weights_list.append(zeros)
        return zeros

    def initialize_linear(self, dims, shape):
        if callable(shape[0]):
            shape[0] = shape[0](dims)
        if callable(shape[1]):
            shape[1] = shape[1](dims)
        weight = 1/np.sqrt(shape[0])*torch.randn(shape)
        bias = 1/np.sqrt(shape[0])*torch.randn([shape[1]])
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
        linear_1 = self.initialize_linear([self.vector_dim_fn(dims), n_in])
        linear_2 = self.initialize_linear([n_out, self.vector_dim_fn(dims)])
        return [linear_1, linear_2]

    def initialize_posterior(self, dims, channel_dim):
        if callable(channel_dim):
            channel_dim = channel_dim(dims)
        mean = 0.01*torch.randn(self.multitensor_system.shape(dims, channel_dim))
        local_capacity_adjustment = self.initialize_zeros(multitensor_system.shape(dims, channel_dim))
        mean.requires_grad = True
        self.weights_list.append(mean)
        return [mean, local_capacity_adjustment]

    def initialize_multizeros(self, shape):
        shape = self.multitensor_system.make_multitensor(default=shape)
        return tensor_algebra.multify(self.initialize_zeros)(shape)
    def initialize_multilinear(self, shape):
        shape = self.multitensor_system.make_multitensor(default=shape)
        return tensor_algebra.multify(self.initialize_linear)(shape)
    def initialize_multiresidual(self, n_in, n_out):
        n_in = self.multitensor_system.make_multitensor(default=n_in)
        n_out = self.multitensor_system.make_multitensor(default=n_out)
        return tensor_algebra.multify(self.initialize_residual)(n_in, n_out)
    def initialize_multiposterior(self, channel_dim):
        channel_dim = self.multitensor_system.make_multitensor(default=channel_dim)
        return tensor_algebra.multify(self.initialize_posterior)(channel_dim)
