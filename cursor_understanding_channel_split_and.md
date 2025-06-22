# Understanding channel_split and direction_split usage
_Exported on 6/22/2025 at 16:23:36 EDT from Cursor (1.1.3)_

---

**User**

why do we need both channel_split and direction_split in make_directional_layer? we only have 8 directions.

---

**Cursor**

`direction_split` and `channel_split` are serving two orthogonal purposes.

1. direction_split (the outer offset 4\*direction\_split)  
   • Chooses which half of the **direction dimension** we are reading from:  
      – direction_split = 0 → the first four slices (→, ↗, ↑, ↖)  
      – direction_split = 1 → the next four slices (←, ↙, ↓, ↘)  
   • Those eight slices are the only elements we store in the tensor’s *direction* axis, so the loop is strictly picking between the “forward” and the “reverse” set of arrows.

2. channel_split (the even/odd slicing `[..., channel_split::2]`)  
   • Chooses which half of the **hidden-channel dimension** the result is written into.  
   • It is used to run the same directional primitive twice, but with the spatial scan order reversed:  
      – channel_split = 0  → run the op in the natural scan order for that arrow.  
      – channel_split = 1  → flip the slice, run the op, then flip the result back (see the `torch.flip` guarded by `direction_split + channel_split == 1`).  
   • Doing it this way lets us keep only eight direction slices while still having *both* “left-to-right” and “right-to-left” (or “near-to-far” / “far-to-near”) accumulations available. We simply park the second scan in the alternate half of the channel dimension instead of doubling the direction dimension to 16.

Why is that useful?  
– Operations such as `cummax` or `shift` are not symmetric: scanning from the boundary toward the pixel gives a different value than scanning away from the boundary. Many ARC tasks need information coming from both sides of a pixel along the same arrow.  
– Storing the two scans in separate channels is cheaper (and keeps downstream logic simple) compared to extending the direction dimension, which would force every subsequent layer that relies on exactly eight directions to change.

So even though we only *store* eight directions, we intentionally compute sixteen directional features (8 arrows × 2 scan-orders). The `direction_split` chooses the arrow, the `channel_split` decides whether we are putting the forward or the reverse scan for that arrow into the output tensor.

---

**User**

conceptually, doesn't -> and <- of the 8 directions already corresponds to left to right and right to left scan?

---

**Cursor**

“→” and “←” indeed give you two opposite arrows, but having just those two slices does **not** automatically give every pixel access to both scan orders for the *same* arrow.

What the network wants from a directional primitive such as `cummax` or `shift` is usually two separate features for **each** arrow d:

• f₊(d) – the cumulative value when you sweep *with* the arrow (e.g. left→right for “→”).  
• f₋(d) – the cumulative value when you sweep *against* the arrow (right→left for “→”).

If we tried to store f₊(→) and f₋(→) in different **direction** slices (say f₊(→) under “→” and f₋(→) under “←”), then:

1. They would now live under *different* direction indices.  
2. Any later layer that wants to combine or compare the two has to look across the direction axis, which is awkward (it breaks the nice “8-arrow” semantics many other layers assume).

Instead the code keeps the direction axis fixed at eight arrows and pushes the *scan-order* distinction into the **channel** axis:

• `direction_split` (0 or 1) picks which of the eight arrows we are processing (→, ↗, ↑, ↖, ←, ↙, ↓, ↘).  
• `channel_split` (0 or 1) decides whether we run the primitive in the natural order (f₊) or in the reversed order (f₋) for that very same arrow. The result is written into even or odd channels, so both scans of the same arrow sit side-by-side in the channel dimension.

That way every arrow slice still represents the same geometric direction, but now carries *two* complementary features. Down-stream 1×1 “mixing” layers can fuse f₊ and f₋ by a simple linear map on the channel axis, without touching or reshaping the direction axis. It’s a small redundancy (f₋(→) equals f₊(←) up to a boundary effect), but it greatly simplifies later computation and keeps the architecture symmetric.

---

**User**

I am still confused as to why you have to have both. can you create a simply example that shows we need both?

---

**Cursor**

Below is a toy experiment that makes the need for the two loops — `direction_split` *and* `channel_split` — tangible.  We’ll look at the `cummax` primitive, but exactly the same argument holds for `shift`, `cumsum`, etc.

––––––––––––  SET-UP  ––––––––––––  
We create a 1-D “image” (width = 7) and ask for two features for the **same** arrow (→):

• f₊ = cummax scanned **left → right**  
• f₋ = cummax scanned **right → left**

```python
import torch
row = torch.tensor([1, 5, 2, 4, 3, 6, 0], dtype=torch.float32)   #  ←  example data

f_plus  = torch.cummax(row, dim=0)[0]                            # left→right
f_minus = torch.flip(torch.cummax(torch.flip(row, [0]), 0)[0],   # right→left
                     [0])

print("f₊:", f_plus.tolist())
print("f₋:", f_minus.tolist())
```

Output

```
f₊: [1, 5, 5, 5, 5, 6, 6]
f₋: [6, 6, 6, 6, 6, 6, 0]
```

Every pixel **needs both numbers** if, say, the next layer wants
`g = (f₊ > 3) & (f₋ < 6)`.

––––––––––––  OPTION A : put them in two direction slices  ––––––––––––  

If we stored

• f₊ under the direction slice “→” (index 0)  
• f₋ under the direction slice “←” (index 4)

then the tensor would look (schematically) like

```
dir=0(→) :  [1 5 5 5 5 6 6]
dir=4(←) :  [6 6 6 6 6 6 0]
```

A later layer that wants both numbers for *each* arrow now has to read
from **two different direction indices** and somehow align them.
That forces a convolution (or other mixing op) *across* the direction
dimension, which almost every other layer in the model tries hard to
avoid: the eight-arrow axis is intended to be a *structural* axis, not a
feature axis mixed freely.

––––––––––––  OPTION B : keep the direction axis fixed, split by channels  ––––––––––––  

With the current design we leave the eight arrows alone and write the
two scans into **even/odd channels** of the same arrow slice:

```
dir=0(→) , chan even :  [1 5 5 5 5 6 6]   (f₊)
dir=0(→) , chan  odd :  [6 6 6 6 6 6 0]   (f₋)
```

Now any 1×1 (per-pixel) linear layer can combine f₊ and f₋ by a *simple
weight matrix on the channel axis*; nothing ever has to look across the
direction axis.  The price is one extra factor-of-2 in channel count
(the loop `for channel_split in range(2)`), but the network stays
perfectly factorised into the intended tensor product
“examples × colours × arrows × spatial × channels”.

––––––––––––  Bringing it back to `make_directional_layer`  ––––––––––––  

* `direction_split` (0 or 1) tells us whether we are handling the first
  four arrows (→ ↗ ↑ ↖) or the opposite four (← ↙ ↓ ↘).  
  That is purely about **which geometric arrow** the slice belongs to.

* `channel_split` (0 or 1) duplicates the computation for the *same*
  arrow but with the spatial scan reversed, and stores the second result
  in the alternate half of the channel dimension.

Because many downstream layers are written to mix only **within**
channels (and not across the eight arrows), having both loops is the
straightforward way to expose *two scan orders per arrow* without
touching the structure of the direction axis.

```python
import torch
row = torch.tensor([1, 5, 2, 4, 3, 6, 0], dtype=torch.float32)   #  ←  example data

f_plus  = torch.cummax(row, dim=0)[0]                            # left→right
f_minus = torch.flip(torch.cummax(torch.flip(row, [0]), 0)[0],   # right→left
                     [0])

print("f₊:", f_plus.tolist())
print("f₋:", f_minus.tolist())
```

```plaintext
f₊: [1, 5, 5, 5, 5, 6, 6]
f₋: [6, 6, 6, 6, 6, 6, 0]
```

```plaintext
dir=0(→) :  [1 5 5 5 5 6 6]
dir=4(←) :  [6 6 6 6 6 6 0]
```

```plaintext
dir=0(→) , chan even :  [1 5 5 5 5 6 6]   (f₊)
dir=0(→) , chan  odd :  [6 6 6 6 6 6 0]   (f₋)
```

