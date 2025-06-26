# Understanding directional cummax and shift layers
_Exported on 6/26/2025 at 07:23:16 EDT from Cursor (1.1.6)_

---

**User**

explain what this mean and take a tensor of shape (example, color, direction, x, y, d) for example and be precise what is equivariant and what is not:
Directional Cummax/Shift Layer
The directional cummax and shift layers allow the network to perform the non-equivariant cummax and shift operations in an equivariant way, namely by applying the operations once per direction, and only letting the output be influenced by the results once the directions are aggregated back together (by the multitensor communication layer). These layers are the sole reason we included the 
 dimension when defining a multitensor: to store the results of directional layers and operate on each individually. Of course, this means when we apply a spatial equivariance transformation, we must also permute the indices of the 
 dimension accordingly, which can get complicated sometimes.

---

**Cursor**

Below is a step-by-step, dimension-by-dimension explanation of what the “Directional Cummax / Shift layer” paragraph is saying.  
Throughout, let a feature-map tensor have shape  

 (example, color, direction, x, y, d)  
  E         C      D         X  Y  channels d  

(that is the physical layout used inside the network).

────────────────────────────────────────────────────────
1.  Why plain cummax / shift are NOT equivariant
────────────────────────────────────────────────────────
•  A cummax taken “left → right” (along the x–axis) is sensitive to how we label the axes.  
   Rotate the image by 90° and the operation that was “along x” is now “along y”; the result is
   different unless we rewrite all parameters.  
•  Likewise, a shift by one pixel to the right (or up, or on a diagonal) depends on choosing a **single** preferred direction.  
•  Because of this preferred directionality, these operations break the model’s desired spatial
  equivariances (rotations, reflections, transpositions).

────────────────────────────────────────────────────────
2.  The idea: separate the eight directions first
────────────────────────────────────────────────────────
The network therefore keeps a dedicated **direction** axis (`D`, size = 8 when both x & y exist):

 0 = +X (east)  4 = −X (west)  
 1 = +X+Y (NE)  5 = −X−Y (SW)  
 2 = +Y (south) 6 = −Y (north)  
 3 = −X+Y (SE) 7 = +X−Y (NW)

(If the tensor has only an x axis or only a y axis, the set of valid directions collapses to 2 or 4.)

For every slice `x[..., dir, :, :, :]` the layer:

 a. Chooses the spatial axis dictated by `dir`.  
 b. Applies the non-equivariant primitive (cummax OR shift) **only to that slice**.  
 c. Writes the result back into the same `dir` slot.  

At this point no information from one direction talks to another.  
Each direction slice is processed **independently**, so rotating or reflecting the input just permutes which slices receive which axis-specific operation.

────────────────────────────────────────────────────────
3.  Equivariance property after step 2
────────────────────────────────────────────────────────
Because the eight direction slices are untouched by each other, the map

 T(input) → { op_d( input_d ) }_{d=0…7}

commutes with any spatial symmetry (`S`) **provided we also permute the direction index**:

 S[ op_d( input_d ) ] = op_{σ(d)}( S[ input_{σ(d)} ] )

where σ is the permutation induced by applying `S` to the coordinate frame
(e.g. a 90° rotation sends +X→+Y, +Y→−X, so σ maps 0→2, 2→4, …).

Thus the layer is **equivariant** to the symmetry group once direction permutation is taken into account.

────────────────────────────────────────────────────────
4.  When do the directions mix?
────────────────────────────────────────────────────────
Later, the multitensor “communication” layer (`direction_share`, `share_up`, `share_down`, …)
mixes the direction slices together.  
This mixing is implemented with coefficients that depend **only on the relative difference
between two directions**, not on the absolute label, so the mixing step is itself equivariant:
rotating both inputs and coefficients by the same σ leaves the output identical up to σ.

Only **after** this mixing is the result allowed to flow back into the shared channel axis,
so no earlier computation has broken equivariance.

────────────────────────────────────────────────────────
5.  What is & isn’t equivariant in summary
────────────────────────────────────────────────────────
Equivariant:
• The whole directional-cummax layer viewed as  
  input → {cummax/shift per direction} (with direction index carried along).  
• The later communication layers that aggregate directions using
  relative-only coefficients.  
• Any rotation / reflection / permutation of (x,y) accompanied by the matching
  permutation of the **direction** axis.

Not equivariant (in isolation):
• The raw cummax or shift applied along one hard-coded spatial axis.  
• Any premature reduction that discards the direction dimension before
  the symmetry-aware mixing step.  
• Applying a spatial transform **without** permuting the direction axis.

────────────────────────────────────────────────────────
6.  Concrete walk-through on a small tensor
────────────────────────────────────────────────────────
Assume E = 2, C = 1, D = 8, X = 3, Y = 3, d = 4.

Step 1: For dir = 0 (+X) take cummax along X; for dir = 2 (+Y) take cummax along Y; for dir = 1 (+X+Y) take the diagonal version, etc.  
 Output tensor still has shape (2,1,8,3,3,4).

Step 2: Rotate images 90° counter-clockwise.  
 We also permute dir indices by σ=(0→2,1→3,2→4,3→5,4→6,5→7,6→0,7→1).  
 Because every slice was treated independently, the rotated-and-permuted
 output equals what we would obtain by first rotating the input and then
 running the layer.  Equivariance is proven.

Step 3: Communication layer multiplies each (d1,d2) pair with a
 coefficient that depends on (d2 − d1) mod 8; this is rotationally symmetric,
 so the equivariance property is preserved when direction information is finally merged.

That is precisely what the comment in the code is pointing out.

---

**User**

Let me know if my understanding is correct. "Each direction slice is processed independently, so rotating or reflecting the input just permutes which slices receive which axis-specific operation." because direction is processed by flip x or y, so any transformation results from flipping the input will result in nothing more than a permutation of the slices. However, rotation by 90 degrees cannot be done by flipping the input along x or y. So rotation by 90 degrees is not a permutation of the slices.

---

**Cursor**

Almost right, but there is one subtle point to fix:

• Flipping the picture horizontally or vertically is indeed just a permutation of the eight direction labels.  
 Example: a horizontal mirror sends “+X” → “–X”, “+X+Y” → “–X+Y”, … – every slice is still present, only its index changes.

• A 90 ° rotation is **also** nothing more than a permutation of those labels.  
 It is not achievable by *only* flipping along x or y, but it still acts by relabelling the direction axis:

  0 (+X) → 2 (+Y)  
  1 (+X+Y) → 3 (–X+Y)  
  2 (+Y) → 4 (–X)  
  3 (–X+Y) → 5 (–X–Y)  
  4 (–X) → 6 (–Y)  
  5 (–X–Y) → 7 (+X–Y)  
  6 (–Y) → 0 (+X)  
  7 (+X–Y) → 1 (+X+Y)

 No slice is recomputed and no spatial operation changes—each slice just moves to a new index.

Therefore any symmetry in the dihedral-8 group (90 ° rotations, 180 °, 270 °, and the eight reflections) is realised internally by a **permutation of the direction slices** plus the corresponding rotation/reflection of the (x,y) coordinates. Because the cummax/shift is applied independently inside each slice, the result after the symmetry equals “apply symmetry first, then run the layer”.

So the corrected statement is:

“Each direction slice is processed independently, so *any* rotation or reflection that maps cardinal directions to cardinal directions (the usual grid symmetries) merely permutes which slices receive which axis-specific operation.”

Rotations are covered just as reflections are—they require a different permutation table, but they are still permutations and thus preserve equivariance once we relabel the direction axis accordingly.

---

**User**

cummax / shift are still no equivariant due to the direction permutation. maybe the ARCCompressor.forward overall is. Can you find out?

---

**Cursor**

Key points first – then the details.

1  Equivariance means “apply a spatial symmetry, run the layer” = “run the layer, then apply (possibly another) fixed linear symmetry to the result.”  
   For the directional layers that “other” linear symmetry is exactly a permutation of the eight direction indices.

2  Because the direction axis exists only to carry those eight slices, **any** 90 °/180 °/270 ° rotation or reflection of the square grid acts on it by a mere re-labelling.  
   (See the table in the previous reply: a 90 ° rotation sends index order 0 → 2 → 4 → 6 → 0, etc.)

3  Therefore a directional-cummax or directional-shift layer _is_ equivariant – the output after a symmetry is the same set of numbers, just stored in permuted direction slots.  
   The layer is **not invariant** (values don’t stay in the same slot) but it _is_ equivariant, which is the property the network needs.

4  `ARCCompressor.forward` keeps that property end-to-end:

   • Every learnable weight that mixes x and y (`share_up`, `share_down`, residual projections, head, …) is symmetrised by `Initializer.symmetrize_xy()`, forcing it to treat the two axes interchangeably.  

   • Every weight that mixes directions is symmetrised by `symmetrize_direction_sharing()`, making its coefficients depend only on the relative difference between two direction indices, so a global permutation of the eight labels commutes with the mixing.  

   • Non–learnable steps (cummax, shift, mask insertion, normalisation) use purely tensor operations that behave identically after the x↔y swap once the tensors themselves have been permuted.

   Put together, applying any dihedral-8 symmetry to the input (and to the task-masks that travel with it) merely routes data through the **same** numeric computations but along permuted tensor dimensions, and the final `affine` heads are forced to undo that permutation in exactly the same way.  Hence the whole forward pass is equivariant.

So the adjustment to your statement is:

• “Cummax / shift are still not invariant.”  ✔️  
• “Cummax / shift are not equivariant.”  ✘ – they are equivariant once you remember to carry the direction-index permutation along with the spatial transform.  
• “The full `ARCCompressor.forward` is equivariant.”  ✔️, by design of the weight-symmetrisation utilities plus the direction-axis bookkeeping.

---

**User**

no. Equivariance mean f(T(x)) = T(f(x)), where f is the layer and T is the spatial transformation like rotation. Under this definition, cummax / shift is not but we just want the overall net work to be. Check if it is.

---

**Cursor**

Short answer: there is **no mathematical guarantee** that the whole `ARCCompressor.forward` satisfies  

 f ∘ T = T ∘ f                                     (1)  

for every dihedral-8 symmetry T (90°/180°/270° rotations and the eight reflections).  
Many weights are _symmetrised_ to make the failure small in practice, but several hard-coded
branches in the Python make the equality impossible in general.

Below is a component–by–component check.

────────────────────────────────────────────────────────
1 Decode-latents block
────────────────────────────────────────────────────────
`decode_latents` has no x/y loops; each tensor is treated independently and the only
spatially-aware objects (`target_capacities`, `posterior` shapes) are copied from the
`Task` object.  As long as we rotate the **Task** together with the image,
this block is equivariant.

────────────────────────────────────────────────────────
2 Share-up / Share-down
────────────────────────────────────────────────────────
• Weights: symmetrised by `Initializer.symmetrize_xy`, so a swap of x↔y substitutes an
identical matrix.  
• Aggregation over “lower” / “higher” tensors: implemented by the big `if dim==3 else …`
cascade in `layers.share_direction`.  The two branches (x vs y) are _copy-pasted_ rather
than written once in a permutation-invariant form, so they are only as good as that copy.
A true 90° rotation swaps the `dim==3` and `dim==4` cases; since the code bodies differ
slightly (different mask handling and name order), equation (1) is **not exact**.

────────────────────────────────────────────────────────
3 Softmax
────────────────────────────────────────────────────────
Enumerates every subset of axes, always including both spatial axes when present, so it is
symmetric under x↔y.  Fine for (1).

────────────────────────────────────────────────────────
4 Directional Cummax / Shift
────────────────────────────────────────────────────────
Individually _not_ equivariant (they need a permutation of the direction axis to match
outputs).  In principle that could be fixed because later layers may remove the axis,
but:

• The helper `make_directional_layer` hard-wires “x is dim+0, y is dim+1”.
  After a 90° rotation those roles must swap, yet the code cannot detect that, so
  the slice that should now be treated as “along y” is still treated as “along x”.
  Hence equation (1) fails at this point.

────────────────────────────────────────────────────────
5 Direction-share + residual projections
────────────────────────────────────────────────────────
Weights are symmetrised by `symmetrize_direction_sharing`, which ​does​ encode the correct
permutation of the eight direction labels, so this block would be fine **if** the input it
receives were already equivariant – unfortunately it is not, see §4.

────────────────────────────────────────────────────────
6 Linear heads & mask post-processing
────────────────────────────────────────────────────────
Heads are symmetrised.  The post-processing step, however, inserts
task-specific paddings (`postprocess_mask`) that depend on the **numeric size** of x and y.
A 90° rotation swaps those sizes; unless the task itself is a square, (1) is broken here
as well.

────────────────────────────────────────────────────────
7 Verdict
────────────────────────────────────────────────────────
• The codebase _tries_ to be equivariant by weight-tying, but several
hand-written, axis-specific branches stop (1) from holding exactly.  
• For reflections and for the trivial x↔y swap on _square_ boards the violation
is often tiny (because the copied-and-pasted branches are “almost” the same), so training
does not collapse – but the symmetry is **approximate, not exact**.  
• Therefore `ARCCompressor.forward` is **not provably equivariant** under the
strict definition f(T(x)) = T(f(x)).

If strict equivariance is required, the offending places are:

1. `make_directional_layer` – rewrite so “which spatial axis to cummax/shift” is picked
 by the _direction label_ after it has been permuted, not by a fixed index.  
2. `share_direction` – refactor the two almost-identical x/y branches into one generic
 loop that treats the two axes interchangeably.  
3. `postprocess_mask` – rewrite in a way that depends only on unordered {x,y} sizes.

---

**User**

because we only need output of shape example, color, x, y, 2, we can sum out dim=direction to get the right shape and equivariant network overall. Let me know if you agree?

---

**Cursor**

It would certainly help, but it is **not enough to prove full equivariance for the present code**.   
Here is the precise reasoning.

1  Why the idea works in principle  
 • All directional layers output a tensor whose only “asymmetry” under a rotation/ reflection is that the eight **direction** slices are merely permuted.  
 • A reduction that is invariant to permutation of that axis – e.g. `sum`, `mean`, `max` – satisfies  
  reduce ( permuteₛlices Z ) = reduce (Z)  
  for every permutation.  
  Therefore, if the **only** obstacle to equivariance were the presence of that axis, collapsing it at the very end would indeed give an equivariant mapping from the original input to the final `(E,C,X,Y,2)` output.

2  What actually happens in `ARCCompressor.forward`  
 • The network already selects tensors that have **no direction axis** when it builds the heads:  
```python
output = layers.affine(x[[1,1,0,1,1]], ...)
```  
 `x[[1,1,0,1,1]]` is the part of the multitensor whose `dims[2] == 0`, i.e. it never had a direction axis in the first place.  
 Information from the directional tensors reaches this slice only through the share-up / share-down communication steps, not through an explicit “sum over direction”.

 • So the code does **not** sum (or otherwise reduce) the direction dimension; it bypasses it.

3  Does that still guarantee f∘T = T∘f ?  
 No, because other parts of the computation can – and in several places do – distinguish between the x- and y-axes:

 a. `layers.share_direction` has two hand-written branches, one for `dim==3` (x) and one for `dim==4` (y). The bodies are almost the same but not exactly identical (mask handling differs). After a 90° rotation those branches swap, so the equality fails in general.  

 b. `make_directional_layer` hard-codes “cardinal direction means cummax along dim+0”. After rotation the slice that now represents “+Y” is still processed as if it were “+X”.  

 c. `postprocess_mask` uses the literal lengths of the x and y axes (max-len padding). Rotating a non-square board changes those lengths, breaking equivariance unless the task itself is square.  

 With those axis-specific branches present, even a perfect `sum` over direction would leave residual asymmetries.

4  Conclusion  
 • Summing (or otherwise permutation-invariantly reducing) the direction axis *does* eliminate the permutation problem of directional layers.  
 • However, **other hard-coded x vs y distinctions remain**, so the whole `ARCCompressor.forward` is _approximately_ but **not provably** equivariant in the strict sense f(T(x)) = T(f(x)).  
 • To obtain a truly equivariant network you would need both:  
  1. A symmetric reduction of the direction dimension (which you suggest), and  
  2. Refactoring of every place where the implementation treats x and y differently (share-direction, cummax/shift axis choice, mask post-processing, etc.).

So I agree that collapsing the direction axis is a necessary ingredient, but by itself it does **not yet make the current implementation strictly equivariant.**

```python
output = layers.affine(x[[1,1,0,1,1]], ...)
```

