# Understanding function integration and task preprocessing
_Exported on 6/19/2025 at 15:22:18 EDT from Cursor (1.1.3)_

---

**User**

explain how different function comes together and how task is preprocessed in details

---

**Cursor**

Below is a walkthrough of the preprocessing pipeline, starting from the public entry-point and drilling all the way down to the per-pixel tensors that will later be fed to the neural network.

────────────────────────────────────────────────────
1.  Entry-point: `preprocess_tasks`
────────────────────────────────────────────────────
• Signature: `preprocess_tasks(split, task_nums_or_task_names)`

1.  Loads the JSON file that contains the ARC-AGI challenges for the requested data split  
   (`dataset/arc-agi_<split>_challenges.json`).  
   –  For `"test"` split only problems are loaded.  
   –  For `"train"` / `"evaluation"` an additional solutions file is loaded so the
      ground-truth outputs are available for scoring.

2.  Builds a list of task names, then instantiates a `Task` object
    for every task that matches either
    • a string in `task_nums_or_task_names`, **or**  
    • an integer index in that list (handy when you just want “task #17”).

The heavy lifting now moves inside the `Task` constructor.

────────────────────────────────────────────────────
2.  `class Task.__init__`
────────────────────────────────────────────────────
Order of operations (exactly the order in which they are called):

1. `_collect_problem_shapes`  
   Collects the raw `[height, width]` of every input and output grid.  
   Result: `self.shapes` is a list of length `n_examples` whose elements are  
   `[in_shape, out_shape]`.

2. `_predict_solution_shapes`  
   ARC test examples do **not** include outputs, so their shapes must be guessed.
   Heuristics applied in priority order:

   a. If every *training* example has identical input and output shapes
      (`in_out_same_size`) → test output shape = its own input shape.

   b. Else, if every training example has the *same* output shape
      (`all_out_same_size`) → use that common shape for the tests.

   c. Else fallback to the maximum height / width seen anywhere in the task.

3. `_construct_multitensor_system`  
   a. Determines the global canvas size  
      `self.n_x = max height`, `self.n_y = max width`  
      so every example can fit in a shared tensor.

   b. Scans **all** cells of every grid (input + output) to collect the set
      of colors that appear, always forcing   `0`  = black background.  
      `self.colors` is the sorted list,  
      `self.n_colors = len(colors) - 1` (minus background).

   c. Instantiates a `MultiTensorSystem`, a wrapper (defined elsewhere)
      that later orchestrates weight-sharing across examples and modes.

4. `_compute_mask`  
   Builds a boolean mask of shape  
   `(n_examples, n_x, n_y, 2)` where the last index is  
   `0 = input 1 = output`.  
   • Inside the true grid area ⇒ `1`  
   • Padding pixels outside the original grid ⇒ `0`  
   This mask is reused later for both forward activations and the
   cross-entropy loss so the network is never penalised for predictions on
   “imaginary” padding pixels.

5. `_create_problem_tensor`

      Shape before argmax: `(n_examples, n_colors+1, n_x, n_y, 2)`  
      Steps:
      a. For every example & mode (input/output) build a one-hot tensor with
         `_create_grid_tensor` (see below).  
      b. Write it into the canvas at the correct coordinates.  
      c. At the end, `np.argmax(axis=1)` collapses the color
         dimension so what we store is the integer color index per pixel:  
         final `self.problem` shape → `(n_examples, n_x, n_y, 2)`  
         and immediately moved to the default torch device.

6. `_create_solution_tensor` (only if ground-truth is available)  
   Converts the list of test outputs into the same integer-encoded format
   as above, stores it in `self.solution`, and keeps a hash of the full
   solution tuple (`self.solution_hash`) so later code can quickly detect
   whether it has already found the right answer.

────────────────────────────────────────────────────
3.  Helper: `_create_grid_tensor`
────────────────────────────────────────────────────
Turns a single 2-D grid of color indices into a **one-hot** tensor:

    output shape: (n_colors + 1, grid_height, grid_width)

Channel 0 is black/background, channel c is “1” wherever color c appears.

This helper is used both by `_create_problem_tensor`
(train + test inputs, train outputs) and `_create_solution_tensor`
(test outputs).

────────────────────────────────────────────────────
4.  How everything fits together
────────────────────────────────────────────────────
• `preprocess_tasks` returns a list of fully-initialised `Task` objects.

• Each `Task` now exposes:
  – `problem` integer tensor with both input and (train) output grids  
  – `masks`  boolean tensor marking valid cells only  
  – `solution` (evaluation / train split only) ground-truth test outputs  
  – `multitensor_system` struct that defines tensor groups for the network  
  – Misc. metadata (`n_examples`, `n_colors`, etc.)

• Down-stream training code (`train.py`, `parallel_train.py`, …) can iterate
  over tasks, feed `problem` through the neural model, and compute the loss
  on the masked regions. At inference time the same tensors serve as input,
  but only `problem[..., 0]` (the input mode) is given; the model is asked
  to predict `problem[..., 1]` (output mode) for the unseen test examples.

────────────────────────────────────────────────────
5.  Why the design matters
────────────────────────────────────────────────────
1.  A *single* fixed-size tensor per task means the network can be scripted
    with static shapes — important for JIT compilation and batching.

2.  The explicit `masks` avoid spurious gradients coming from the padding
    area and allow cross-tasks weight sharing by aligning every example on
    the same canvas.

3.  Colour one-hot + argmax → integer encoding keeps memory usage low:  
    one byte per pixel instead of 10-20 floats, yet fully differentiable
    where it matters (before argmax during the forward pass).

By following the above pipeline you can trace any cell of the original ARC
JSON description through a deterministic sequence of transformations until
it lands as a single integer inside a PyTorch tensor ready for the model.

