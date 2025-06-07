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
