#### Understand Code Base

Class **Task**:
1. Color Mapping (_construct_multitensor_system)
   Raw colors [0, 7] become:
   colors: [0, 2, 4, 6, 7]  # All colors found in task
   Mapped to indices: 0→0, 2→1, 4→2, 6→3, 7→4

2. attibutes
    - problem:
        Shape: [examples, x, y, modes] = [6, 9, 9, 2]
        modes: 0=input, 1=output
    - Mask:
        Shape: [6, 9, 9, 2] - same as problem tensor
        1.0 = valid region, 0.0 = padding

3. multitensor_system: created based on task

Class **MultiTensorSystem**:
- dims: 0/1 vector for (examples, colors, directions, x, y)
- __iter__: iterate over all possible dims, subset of (2^5=32)
- _make_multitensor: creates a nested list (tree-like) of shape [2 x 2 x 2 x 2 x 2]
- make_multitensor: MultiTensor wrapper for _make_multitensor
- shape: given dims, return the shape of the tensor

Class **MultiTensor**:
- __getitem__ / __setitem__: allow simple indexing (e.g. multitensor[dims]) into the nested list


#### other considerations
- Multi-Task Learning (Your new proposal)
  - The Goal: To train a single, generalist model that performs well on average across all tasks.
  - The Process: Exactly as you said. You would have one model and one optimizer. You'd iterate through a shuffled set of tasks, and for each task, you'd perform a standard gradient step on your single model.
  - The Outcome: You end up with one set of weights (z, f) that represents a compromise or a blend of the requirements for all tasks. This model is intended to be used directly for inference, without any further fine-tuning.
- Meta-Learning (The "Universal Starting Point" I implemented)
  - The Goal: To train a model that is an excellent starting point for rapid, task-specific fine-tuning. The model isn't trained to solve any task directly; it's trained to be adaptable.
  - The Process: This is the more complex two-loop structure in meta_train.py:
    - Inner Loop: Simulate fine-tuning by taking a few "test steps" on a copy of the model for a specific task.
    - Outer Loop: See how well that fine-tuning simulation worked, and use that result to update the original model, making it a better starting point.
  - The Outcome: The resulting weights (z_meta, f_meta) are likely not good for solving any specific task out-of-the-box. Their value is that only a few additional training steps are needed to achieve high performance on a new task.