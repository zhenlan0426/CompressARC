# Understanding the cummax function
_Exported on 6/22/2025 at 16:24:21 EDT from Cursor (1.1.3)_

---

**User**

explain cummax in details

---

**Cursor**

`cummax` is a fairly sophisticated layer that lets the network ask, for every pixel, “what is the (masked) cumulative-maximum of the values I have seen so far in some chosen scan-direction?”  
Because ARC images can contain monotone “growing” patterns (e.g., lines that propagate until they hit an obstacle), this operation is a powerful inductive bias.

Below is a step-by-step view of how it is built and what happens at runtime.

────────────────────────────────────────────────────────────────────────
1.  The core scalar kernels
   • `cummax_(x, dim, masks)`   (```415:430:layers.py```)
        – Re-interprets the boolean input mask as ±1 000 offsets so that masked
          positions are effectively −∞ for a max and +∞ for a min.  
        – Computes the cumulative maximum along the chosen axis (`torch.cummax`)
          and then rescales the result into the range [-1, 1] using the global
          max/min for that slice so that gradients stay well behaved.

   • `diagonal_cummax_(x, dim1, dim2, masks)`   (```431:445:layers.py```)
        – Handles 45° diagonals, which PyTorch cannot scan directly.  
        – Uses a **parallel prefix / associative scan**: repeatedly shifts the
          tensor by 1, 2, 4, … cells (log₂ passes) in both forward and backward
          directions, updating running maxima at each step.  
        – Derives minima with the same trick, then rescales to [-1, 1].

────────────────────────────────────────────────────────────────────────
2.  Turning the scalar kernels into a *directional* layer  
   `make_directional_layer` (defined earlier in the file) wraps the two scalar
   kernels so that a single call can process an 8-way direction tensor:

   • The input has an explicit **direction dimension** with eight slices
     (→, ↗, ↑, ↖, ←, ↙, ↓, ↘).  
   • The wrapper loops over half-channels and forward/backward scans so that:
        – Even-numbered channels propagate in the “forward” sense of each
          direction, odd-numbered channels propagate in the reverse sense.
   • Channels belonging to directions that do not physically exist for the
     current tensor shape (for example, diagonal directions when `dims[3]` or
     `dims[4]` is 0) are replaced by a zero tensor, so nothing crashes.

────────────────────────────────────────────────────────────────────────
3.  Adding residual plumbing  
   `add_residual` inserts the standard Transformer-style pre-norm, linear
   down-projection → *layer* → post-norm → linear up-projection, plus a skip
   connection:

        residual_out = x + W_up( Layer( W_down( pre_norm(x) )))
   Optional flags (`pre_norm`, `post_norm`, `use_bias`) come from the caller.

────────────────────────────────────────────────────────────────────────
4.  Running only where it makes sense  
   `only_do_for_certain_shapes((1,1,1,1,1), (1,0,1,1,1))` tells the multitensor
   system to *replace the layer with an identity* unless the tensor’s shape
   pattern belongs to one of the two whitelists:
        • (1 × 1 × 1 × 1 × 1)  – full 8-direction tensor
        • (1 × 0 × 1 × 1 × 1) – no colour channel, still directional

   This keeps computation focused on tensors that actually carry spatial
   information; e.g. scalar global-context tensors will skip the layer.

────────────────────────────────────────────────────────────────────────
5.  Broadcasting across the whole multitensor system  
   The outermost `multitensor_systems.multify` decorator means:

   • If the network contains, say, 12 tensors of different ranks/sizes, a *new*
     copy of all the logic above is applied **independently to each tensor**.  
   • The outputs are reassembled into a `MultiTensor` object with the same
     indexing scheme.  
   • For a particular tensor, `dims` (its signature of axes) is passed so the
     layer knows where the example, colour, direction and channel dimensions
     physically live inside that tensor.

────────────────────────────────────────────────────────────────────────
6.  Putting it all together (definition line)  
```446:453:layers.py
cummax = multitensor_systems.multify(
         only_do_for_certain_shapes(...)(
         add_residual(
         make_directional_layer(
         cummax_, diagonal_cummax_
         ))))
```
Reading inside-out:

   1. `make_directional_layer` gives an 8-direction scan that calls
      `cummax_` (cardinals) or `diagonal_cummax_` (diagonals).  
   2. `add_residual` wraps it in residual projections/norms.  
   3. `only_do_for_certain_shapes` gates execution by tensor shape pattern.  
   4. `multify` maps the whole construction over every tensor in the network’s
      multitensor system.

────────────────────────────────────────────────────────────────────────
Key take-aways
• `cummax` is a **direction-aware cumulative-max** with proper masking,
  normalisation, and residual wiring.  
• Its implementation blends classical parallel-scan techniques with PyTorch
  primitives and network engineering conveniences (shape gating, multitensor
  mapping).  
• In tasks like ARC, this allows the model to extrapolate from partial lines or
  grow shapes until they hit borders, which is why it is part of this
  architecture’s inductive-bias toolkit.

```python
cummax = multitensor_systems.multify(
         only_do_for_certain_shapes(...)(
         add_residual(
         make_directional_layer(
         cummax_, diagonal_cummax_
         ))))
```

---

**User**

why do we need masks?

---

**Cursor**

In these directional layers the network is asked to scan over a 2-D grid that can have “holes”, padding, or regions that simply do not belong to the current image (ARC tasks have wildly different input-output sizes).  
If we ran cumulative-max (or any directional reduction) straight over the raw tensor, those irrelevant positions would leak information and distort the statistics that the layer is supposed to model.  
The mask solves exactly that problem.

How the mask is used

1. Converting “invalid” cells into extreme values  
   In `cummax_` (cardinals) and `diagonal_cummax_` (diagonals) you see

       masks = 1e3 * (1 – masks)        # masks = 0 ➞ add 0      (valid)
                                        # masks = 1 ➞ add ±1000  (invalid)

   • For the max-scan we subtract `masks` so invalid cells become ≈ −∞.  
   • For the min-scan we add   `masks` so invalid cells become ≈ +∞.  
   In both cases they are guaranteed never to be selected as the running
   extreme, yet gradients can still flow through them (they are finite).

2. Preventing normalisation artefacts  
   After the cumulative scan the code rescales

       (x – min) / (max – min) * 2 – 1

   If invalid pixels were included in `max`/`min`, the denominator could blow
   up or shrink to ~0, destroying dynamic range. Masking keeps the extrema
   meaningful for the *real* part of the image.

3. Guarding prefix-scan logic on diagonals  
   The parallel-prefix algorithm in `diagonal_cummax_` repeatedly shifts the
   tensor and takes element-wise maxima/minima. Without masking, padding
   introduced by each shift would contaminate neighbours on every iteration.

4. Supporting variable-size training batches  
   In ARC a batch contains tasks whose “true canvas” widths/heights differ.
   The mask allows all of them to be packed into a single rectangular tensor
   while ensuring that computations never cross the true boundaries of any
   individual example.

5. Allowing the same layer to serve both **input-side** and **output-side** grids  
   The multitensor system keeps tensors for “input colours”, “predicted
   colours”, “direction logits”, etc. with potentially different shapes.  
   Supplied masks (built once in `Task`) adapt the directional layers to every
   one of those shapes without any special-case code inside the layer itself.

In short, the mask is the layer’s guarantee that it only processes meaningful
pixels and that statistical operations (max, min, normalisation) remain well
defined even when the physical tensor is padded or contains multiple tasks of
different geometry.

