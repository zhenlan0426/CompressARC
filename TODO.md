# Project TODOs

This document outlines potential future development directions and improvements for the project.

## Optimization Strategy Enhancements

The current approach optimizes latent state `z` and model weights `f` independently for each task. To improve efficiency and leverage shared knowledge across tasks, two primary enhancement strategies are proposed.

### 1. Meta-Learning: Universal Initialization

This approach involves finding a single set of initial parameters `(z_meta, f_meta)` that serves as an effective starting point for all tasks. This is a classic meta-learning strategy, similar in principle to algorithms like MAML.

**Concept:**
- Pre-train a single `(z_meta, f_meta)` across a variety of tasks.
- This "meta-model" is not optimized for any single task but is primed for rapid adaptation.
- For a new task, initialize the model with `(z_meta, f_meta)` and perform a few steps of fine-tuning to find the task-specific optimal parameters.

**Pros:**
- **Robust and Proven:** A well-established and powerful technique.
- **Faster Optimization:** Significantly reduces the number of optimization steps required for new tasks compared to random initialization.
- **Simpler Implementation:** Less complex than the encoder-based approach.

**Cons:**
- **Optimization Still Required:** Does not eliminate the need for per-task optimization at inference time.
- **Generic Starting Point:** May be a suboptimal compromise if tasks fall into very distinct clusters.

### 2. Amortized Inference via Task Encoder

This approach trains a separate encoder network that maps a description of a task directly to its predicted optimal parameters `(z_pred, f_pred)`. This *amortizes* the cost of optimization into the training of the encoder.

**Concept:**
- Design and train an encoder that takes a task representation (e.g., a dataset sample, a configuration vector) as input.
- The encoder outputs the predicted parameters `(z_pred, f_pred)`.
- **Option A (Pure Amortization):** Use the predicted parameters directly, replacing iterative optimization with a single forward pass.
- **Option B (Hybrid):** Use the predicted parameters as a high-quality starting point for a few steps of further optimization.

**Pros:**
- **Extremely Fast Inference:** Drastically reduces or eliminates the need for test-time optimization.
- **Learned Task Structure:** The encoder can learn a sophisticated mapping from tasks to solutions.
- **State-of-the-Art Potential:** The hybrid approach often yields the best performance by combining the speed of the encoder with the precision of fine-tuning.

**Cons:**
- **Increased Complexity:** Requires designing, implementing, and training a second, non-trivial model (the encoder).
- **Task Representation Challenge:** The effectiveness is highly dependent on how a "task" is represented as an input to the encoder.

1. grid size uncertainty weight - coefficient
2. loss = total_KL + 10*reconstruction_error

