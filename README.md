ARC-AGI Problem: each task is represented as tuples of examples (input1, output1),(input2, output2),... (inputN, outputN), where input and output are 2d grid of integer between 0 and 9. Our goal is to given all examples up till inputN, and predict outputN.

model follows a VAE framework, without the encoder. It starts with latent code z (multi-tensor) of shape (example, color, direction, x, y, channel) and 18 combinations of sub-dim tensors (valid_dims.md), go through many layers of transformations that maps from multi-tensor to multi-tensor (same space), and outputs (via decoder f) a tensor of shape (example, color, x, y, 2), where the last dimension is for input and output of a given example in the task. In addition, model outputs two masks of shape (example, x, 2) and (example, y, 2) as we dont know the exact shape of the input and output ahead of time and x, y of the tensor are chosen as upper bound of possible input and output dimensions.

model is trained by jointly optimizing the latent code z and the decoder f, where the objective is ELBO and each task has its own z, and weights f. An alternative is to have a shared f but different z for each task. The training would be done via stochastic gradient descent iterating over all tasks.










