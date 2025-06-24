# Understanding directional cummax and shift layers
_Exported on 6/24/2025 at 11:20:11 EDT from Cursor (1.1.5)_

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

