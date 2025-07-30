import numpy as np
import torch

import initializers_batch
import layers_batch as layers


np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)



class ARCCompressor:
    """
    The main model class for the VAE Decoder in our solution to ARC.
    """

    # Define the channel dimensions that all the layers use
    n_layers = 4
    share_up_dim = 16
    share_down_dim = 8
    decoding_dim = 4
    softmax_dim = 2
    cummax_dim = 4
    shift_dim = 4
    nonlinear_dim = 16

    # This function gives the channel dimension of the residual stream depending on
    # which dimensions are present, for every tensor in the multitensor.
    def channel_dim_fn(self, dims):
        return 16 if dims[2] == 0 else 8

    def __init__(self, task, batch_size=8, shared_model=None, task_model=None, device='cpu'):
        """
        Create a model that is tailored to the given task, and initialize all the weights.
        The weights are symmetrized such that swapping the x and y dimension ordering should
        make the output's dimension ordering also swapped, for the same weights. This may not
        be exactly correct since symmetrizing all operations is difficult.
        Args:
            task (preprocessing.Task): The task which the model is to be made for solving.
            batch_size (int): Batch size for initialization.
            shared_model (ARCCompressor): Existing model to reuse shared weights from.
            task_model (ARCCompressor): Existing model to reuse task-specific weights from.
            device (str): Device to initialize tensors on ('cuda' or 'cpu'). Defaults to 'cpu'.
        """
        # Set default device for tensor creation during initialization
        torch.set_default_device(device)
        
        self.multitensor_system = task.multitensor_system
        if task_model is None:
            # task-specific weights
            initializer = initializers_batch.Initializer(self.multitensor_system, self.channel_dim_fn, batch_size, True)
            self.multiposteriors = initializer.initialize_multiposterior(self.decoding_dim)
            self.target_capacities = initializer.initialize_multizeros([self.decoding_dim])
            self.task_params = initializer.weights_list
        else:
            self.multiposteriors = task_model.multiposteriors
            self.target_capacities = task_model.target_capacities
            self.task_params = task_model.task_params
        # Pin task-specific parameters if we are on CPU so later GPU copies use faster pinned memory
        if device == 'cpu':
            self.pin_task_params()
        
        if shared_model is None:
            # shared weights
            initializer = initializers_batch.Initializer(self.multitensor_system, self.channel_dim_fn, batch_size, False)
            self.decode_weights = initializer.initialize_multilinear([self.decoding_dim, self.channel_dim_fn])
            initializer.symmetrize_xy(self.decode_weights)
            

            self.share_up_weights = []
            self.share_down_weights = []
            self.softmax_weights = []
            self.cummax_weights = []
            self.shift_weights = []
            self.direction_share_weights = []
            self.nonlinear_weights = []

            for layer_num in range(self.n_layers):
                self.share_up_weights.append(initializer.initialize_multiresidual(self.share_up_dim, self.share_up_dim))
                self.share_down_weights.append(initializer.initialize_multiresidual(self.share_down_dim, self.share_down_dim))
                output_scaling_fn = lambda dims: self.softmax_dim * (2 ** (dims[1] + dims[2] + dims[3] + dims[4]) - 1)
                self.softmax_weights.append(initializer.initialize_multiresidual(self.softmax_dim, output_scaling_fn))
                self.cummax_weights.append(initializer.initialize_multiresidual(self.cummax_dim, self.cummax_dim))
                self.shift_weights.append(initializer.initialize_multiresidual(self.shift_dim, self.shift_dim))
                self.direction_share_weights.append(initializer.initialize_multidirection_share())
                self.nonlinear_weights.append(initializer.initialize_multiresidual(self.nonlinear_dim, self.nonlinear_dim))

            self.head_weights = initializer.initialize_head()
            self.mask_weights = initializer.initialize_linear(
                [1, 0, 0, 1, 0], [self.channel_dim_fn([1, 0, 0, 1, 0]), 2]
            )

            # Symmetrize weights so that their behavior is equivariant to swapping x and y dimension ordering
            for weight_list in [
                self.share_up_weights,
                self.share_down_weights,
                self.softmax_weights,
                self.cummax_weights,
                self.shift_weights,
                self.nonlinear_weights,
            ]:
                for layer_num in range(self.n_layers):
                    initializer.symmetrize_xy(weight_list[layer_num])

            for layer_num in range(self.n_layers):
                initializer.symmetrize_direction_sharing(self.direction_share_weights[layer_num])
            self.shared_params = initializer.weights_list
        else:
            # Re-use existing shared weights (references, not copies)
            self.decode_weights = shared_model.decode_weights
            self.share_up_weights = shared_model.share_up_weights
            self.share_down_weights = shared_model.share_down_weights
            self.softmax_weights = shared_model.softmax_weights
            self.cummax_weights = shared_model.cummax_weights
            self.shift_weights = shared_model.shift_weights
            self.direction_share_weights = shared_model.direction_share_weights
            self.nonlinear_weights = shared_model.nonlinear_weights
            self.head_weights = shared_model.head_weights
            self.mask_weights = shared_model.mask_weights
            self.shared_params = shared_model.shared_params
        self.weights_list = self.shared_params + self.task_params

    # ------------------------------------------------------------------
    # Utility helpers for device management of task-specific params
    # ------------------------------------------------------------------
    def _move_task_params(self, device, non_blocking=False):
        """Move only the task-specific parameters (multiposteriors & capacities)"""
        for p in self.task_params:
            p.data = p.data.to(device, non_blocking=non_blocking)
            if p.grad is not None:
                p.grad = p.grad.to(device, non_blocking=non_blocking)

    def to_task_cpu(self, non_blocking=False):
        self._move_task_params("cpu", non_blocking=non_blocking)

    def to_task_cuda(self, non_blocking=False):
        self._move_task_params("cuda", non_blocking=non_blocking)

        # ------------------------------------------------------------------
    # Utility helpers for device management of shared params

    def pin_task_params(self):
        """Pin CPU memory for all task-specific parameters to accelerate host->GPU DMA."""
        for p in self.task_params:
            if p.data.device.type == "cpu":
                p.data = p.data.pin_memory()

    # ------------------------------------------------------------------
    def _move_shared_params(self, device, non_blocking=False):
        """Move only the shared parameters"""
        for p in self.shared_params:
            p.data = p.data.to(device, non_blocking=non_blocking)
            if p.grad is not None:
                p.grad = p.grad.to(device, non_blocking=non_blocking)

    def to_shared_cpu(self):
        self._move_shared_params("cpu")

    def to_shared_cuda(self):
        self._move_shared_params("cuda")


    def forward(self):
        """
        Compute the forward pass of the VAE decoder. Start by using internally stored latents,
        and process from there. Output an [example, color, x, y, channel] tensor for the colors,
        and an [example, x, channel] and [example, y, channel] tensor for the masks.
        Returns:
            Tensor: An [example, color, x, y, channel] tensor, where for every example,
                    input/output (picked by channel dimension), and every pixel (picked
                    by x and y dimensions), we have a vector full of logits for that
                    pixel being each possible color.
            Tensor: An [example, x, channel] tensor, where for every example, input/output
                    (picked by channel dimension), and every x, we assign a score that
                    contributes to the likelihood that that index of the x dimension is not
                    masked out in the prediction.
            Tensor: An [example, y, channel] tensor, used in the same way as above.
            list[Tensor]: A list of tensors indicating the amount of KL contributed by each component
                    tensor in the layers.decode_latents() step.
            list[str]: A list of tensor names that correspond to each tensor in the aforementioned output.
        """
        # Decoding layer
        # x does not have input / output dimension. as test output is not known, we will have to a shared representation for both input and output.
        x, KL_amounts, KL_names = layers.decode_latents(
            self.target_capacities, self.decode_weights, self.multiposteriors
        )

        for layer_num in range(self.n_layers):
            # Multitensor communication layer
            x = layers.share_up(x, self.share_up_weights[layer_num])

            # Softmax layer
            x = layers.softmax(x, self.softmax_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)

            # Directional layers
            x = layers.cummax(
                x, self.cummax_weights[layer_num], self.multitensor_system.task.masks,
                pre_norm=False, post_norm=True, use_bias=False
            )
            x = layers.shift(
                x, self.shift_weights[layer_num], self.multitensor_system.task.masks,
                pre_norm=False, post_norm=True, use_bias=False
            )

            # Directional communication layer
            x = layers.direction_share(x, self.direction_share_weights[layer_num], pre_norm=True, use_bias=False)

            # Nonlinear layer
            x = layers.nonlinear(x, self.nonlinear_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)

            # Multitensor communication layer
            x = layers.share_down(x, self.share_down_weights[layer_num])

            # Normalization layer
            x = layers.normalize(x)

        # Linear Heads
        output = (
            layers.affine(x[[1, 1, 0, 1, 1]], self.head_weights, use_bias=False)
            + 100 * self.head_weights[1]
        )
        x_mask = layers.affine(x[[1, 0, 0, 1, 0]], self.mask_weights, use_bias=True)
        y_mask = layers.affine(x[[1, 0, 0, 0, 1]], self.mask_weights, use_bias=True)

        # Postprocessing - mask out out of bounds pixels based on example shapes (max length of input/output)
        # note this leaks information about the shapes of the examples to the model but would fail for outputN - disable for now
        # x_mask, y_mask = layers.postprocess_mask(self.multitensor_system.task, x_mask, y_mask)

        return output, x_mask, y_mask, KL_amounts, KL_names

