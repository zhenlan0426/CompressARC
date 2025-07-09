Problem: each task is represented as tuples of examples (input1, output1),(input2, output2),... (inputN, outputN), where input and output are 2d grid of integer between 0 and 9. Our goal is to given all examples up till inputN, and predict outputN.

model follows a VAE framework, without the encoder. It starts with latent code z of shape (example, color, direction, x, y, z) and 18 combinations of sub-dim tensors and outputs via decoder f a tensor of shape (example, color, x, y, 2), where the last dimension is for input and output of a given example in the task. In addition, model outputs two masks of shape (example, x, 2) and (example, y, 2) as we dont know the exact shape of the input and output ahead of time and x, y of the tensor are chosen as upper bound of possible input and output dimensions.

model is trained by jointly optimizing the latent code z and the decoder f, where the objective is ELBO. for the re-construction term, since we dont know the grid size, we enumerate all possible grid sizes and sum them out, i.e. sum over size P(size|z) * P(x|size, z), the P(size|z) is given by the mask output.











