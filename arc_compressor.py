import numpy as np
import torch

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')


class ARCCompressor():

    n_layers = 3
    vector_dim_fn = lambda dims: (16 if dims[2] == 0 else 8)
    upgrade_dim = 16  # must be even
    downgrade_dim = 8  # must be even
    channel_dim = 4
    softmax_dim = 2
    cummax_dim = 4
    shift_dim = 4
    reversal_dim = 2
    nonlinear_dim = 16

    def __init__(self, task):
        self.multitensor_system = task.multitensor_system

        # Initialize all the weights
        initializer = initializers.Initializer(multitensor_system, self.vector_dim_fn)

        self.multiposterior = initializer.initialize_multiposterior(self.channel_dim)
        self.global_capacity_adjustments = initializer.initialize_multizeros([self.channel_dim])

        self.share_up_weights = []
        self.share_down_weights = []
        self.softmax_weights = []
        self.cummax_weights = []
        self.shift_weights = []
        self.reversal_weights = []
        self.nonlinear_weights = []
        for layer_num in range(n_layers):
            self.share_up_weights.append(initializer.initialize_multiresidual(upgrade_dim, upgrade_dim))
            self.share_down_weights.append(initializer.initialize_multiresidual(downgrade_dim, downgrade_dim))
            output_scaling_fn = lambda dims: softmax_dim*(2**(dims[1]+dims[2]+dims[3]+dims[4])-1)
            self.softmax_weights.append(initializer.initialize_multiresidual(softmax_dim, output_scaling_fn))
            self.cummax_weights.append(initializer.initialize_multiresidual(cummax_dim, cummax_dim))
            self.shift_weights.append(initializer.initialize_multiresidual(shift_dim, shift_dim))
            self.reversal_weights.append(initializer.initialize_multilinear(self.vector_dim_fn, self.vector_dim_fn))
            self.nonlinear_weights.append(initializer.initialize_multiresidual(nonlinear_dim, nonlinear_dim))

        self.head_weights = initializer.initialize_linear([self.vector_dim_fn([1, 1, 0, 1, 1]), 2])
        self.x_mask_weights = initializer.initialize_linear([self.vector_dim_fn([1, 0, 0, 1, 0]), 2])
        self.y_mask_weights = initializer.initialize_linear([self.vector_dim_fn([1, 0, 0, 0, 1]), 2])

        self.weights_list = initializer.weights_list

    def forward(self):

        x, KL_amounts, KL_names = layers.decode_latents(self.global_capacity_adjustments, self.decode_weights, self.multiposteriors)

        for layer_num in range(self.n_layers):
            # Multitensor layer
            x = layers.share_up(x, self.share_up_weights[layer_num])

            # Softmax layer
            x = layers.softmax(x, self.softmax_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)

            # Directional layers
            x = layers.cummax(x, self.cummax_weights[layer_num], pre_norm=False, post_norm=True, use_bias=False)
            x = layers.shift(x, self.shift_weights[layer_num], pre_norm=False, post_norm=True, use_bias=False)
            x = layers.reverse(x, self.reverse_weights[layer_num], pre_norm=True, use_bias=False)

            # Nonlinear layer
            x = layers.nonlinear(x, self.nonlinear_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)

            # Multitensor layer
            x = layers.share_down(x, self.share_down_weights[layer_num])

            # Normalization layer
            x = layers.normalize(x)

        # Head
        output = layers.affine(x[[1,1,0,1,1]], self.head_weights, use_bias=True)-5
        x_mask = layers.affine(x[[1,0,0,1,0]], self.x_mask_weights, use_bias=True)
        y_mask = layers.affine(x[[1,0,0,0,1]], self.y_mask_weights, use_bias=True)

        return output, x_mask, y_mask, KL_amounts, KL_names
