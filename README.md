ARC-AGI Problem: each task is represented as tuples of examples (input1, output1),(input2, output2),... (inputN, outputN), where input and output are 2d grid of integer between 0 and 9. Our goal is to given all examples up till inputN, and predict outputN.

model follows a VAE framework, without the encoder. It starts with latent code z (multi-tensor) of shape (example, color, direction, x, y, channel) and 18 combinations of sub-dim tensors (valid_dims.md), go through many layers of transformations that maps from multi-tensor to multi-tensor (same space), and outputs (via decoder f) a tensor of shape (example, color, x, y, 2), where the last dimension is for input and output of a given example in the task. In addition, model outputs two masks of shape (example, x, 2) and (example, y, 2) as we dont know the exact shape of the input and output ahead of time and x, y of the tensor are chosen as upper bound of possible input and output dimensions.

model is trained by jointly optimizing the latent code z and the decoder f, where the objective is ELBO. for the re-construction term, since we dont know the grid size, we enumerate all possible grid sizes and sum them out, i.e. sum over size P(size|z) * P(x|size, z), the P(size|z) is given by the mask output. details are in take_step function in train.py

there are certain cases where we know the grid size, e.g. input and output always have the same size. in this case, we can use the mask (example, x, y) to better inform the model, e.g. in share_down and cummax functions in layers.py, we would only aggregate over the valid patches. and in the objective function, we would only sum over different offsets but not over different lengths. Other cases are such grid size is unknown and needs to be inferred by the model.

I am thinking of an alternative that we combine the two cases above. latent variable z should conceptually contain information about the grid size, which lives in the tensor of shape (example, x, y, channel). we can have soft mask as sigmoid of this tensor.mean(-1) and multiply all multi-tensor that are compatible with the mask. Compatible means multi-tensor that has at least example, x or example, y. This mask multiplication layer can happen before any layers and once every block of layers.

- **Amortization** is tricky with multi-processing and shared f. work on **Batched** first then dont need multi-processing.
- **mask** needs to happen after Amortization as without shared f, use perfect mask (when grid size is known) would be best. With sharing, learnt mask is better for the case where grid size is unknown. We can train with known and unknown grid size using uncertain mask and for test time known grid size, predict with perfect mask.
- **Amortization** need to know which parameters have task-specific shape.










