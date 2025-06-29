# Explain function concept and code
_Exported on 6/29/2025 at 19:37:03 EDT from Cursor (1.1.6)_

---

**User**

explain this function, first conceptually, then go over the codes

---

**Cursor**

Conceptual explanation  
------------------------------------------------  
A “direction-sharing” layer contains 8 × 8 independent linear maps: for every incoming direction dir1 (0-7) we have a separate weight matrix that tells the network how to communicate information toward outgoing direction dir2 (0-7).  
Because the overall model is meant to be equivariant under  

1. swapping the x- and y-axes, and  
2. 90° rotations and reflections of the compass directions,

many of those 64 maps must actually be identical – they are just different views of the same underlying transformation seen after applying the symmetry operations.  
`symmetrize_direction_sharing` walks through every stored weight tensor and, for each `(dir1, dir2)` pair, finds a single “representative” map that all symmetry-related pairs must share. It then overwrites the current weight with a reference to that representative so that every time we later update the representative map, every equivalent pair is updated too – enforcing weight tying and guaranteeing the equivariance of the layer.

High-level steps  
1. Iterate over every set of tensor indices `dims` produced by the `MultiTensorSystem`.  
2. For every combination of incoming and outgoing directions `(dir1, dir2)` (0–7):  
   a. Work out which symmetry transformations (axis swap, 180° flip, 90° rotations, reflections) can turn the current pair into a canonical “representative” pair.  
   b. Compute the canonical dimensions and directions `(from_dims, from_dir1, from_dir2)`.  
   c. Point `multiweights[dims][dir1][dir2]` at the weight that lives at that canonical location so all equivalent pairs literally share the same tensor object.

Code walk-through  
------------------------------------------------  

```140:162:initializers.py
for dims in self.multitensor_system:
    for dir1 in range(8):
        for dir2 in range(8):
            from_dims = dims
            from_dir1, from_dir2 = dir1, dir2
```
• Loop over every tensor position and all 64 direction pairs.  
• Begin by assuming the representative is the current position.

```146:155:initializers.py
if dims[3] + dims[4] == 1:
    from_dims = dims[:3] + [1, 0]       # enforce x-y order
    if dims[4] == 1:                     # if x and y were swapped…
        from_dir1 = (2 + from_dir1) % 8  # …rotate directions 180°
        from_dir2 = (2 + from_dir2) % 8
```
Case 1: the coordinates differ only by an x/y swap (exactly one of the two flags is 1).  
• Put the dims into the canonical “y before x” ordering `[1,0]`.  
• Swapping x and y is equivalent to a 180° rotation of directions, so both directions are rotated by +2 (mod 8).

```149:153:initializers.py
if from_dir1 > 4 or (from_dir1 in {0,4} and from_dir2 > 4):
    from_dir1 = (8 - from_dir1) % 8
    from_dir2 = (8 - from_dir2) % 8
```
• If the first direction points into the “western” half-plane (indices 5-7) or lies on an axis (0 or 4) with the second direction in that half, mirror both across the y-axis to land in the eastern half (0-4).

```154:155:initializers.py
if 2 < from_dir1 < 6 or (from_dir1 in {2,6} and 2 < from_dir2 < 6):
    from_dir1 = (4 - from_dir1) % 8
    from_dir2 = (4 - from_dir2) % 8
```
• Similarly, if the first direction lies in the “southern” half-plane (indices 3-5) mirror across the x-axis to bring it into the northern half (7-1).

```156:162:initializers.py
else:
    rotation = (from_dir1 // 2) * 2      # quantise dir1 to nearest axis (0,2,4,6)
    from_dir1 = (from_dir1 - rotation) % 8
    from_dir2 = (from_dir2 - rotation) % 8
    if (from_dir2 - from_dir1) % 8 > 4:
        from_dir2 = (8 + 2*from_dir1 - from_dir2) % 8
```
Case 2: dims already obey xy ordering.  
• Rotate the coordinate system so that `from_dir1` is in the canonical ‘east’ (0) or ‘north’ (2) position.  
• After rotation ensure `from_dir2` is not the “long way round” (>180°) from `from_dir1`; if it is, reflect across the bisector.

```163:164:initializers.py
multiweights[dims][dir1][dir2] = \
    multiweights[from_dims][from_dir1][from_dir2]
```
• Finally tie the current weight to the representative weight found above.

Result: every symmetry-equivalent `(dims, dir1, dir2)` triple now literally holds the same tensor object, and subsequent training will automatically keep them identical, giving the layer the desired x/y-swap and rotation/reflection equivariance.

```python
for dims in self.multitensor_system:
    for dir1 in range(8):
        for dir2 in range(8):
            from_dims = dims
            from_dir1, from_dir2 = dir1, dir2
```

```python
if dims[3] + dims[4] == 1:
    from_dims = dims[:3] + [1, 0]       # enforce x-y order
    if dims[4] == 1:                     # if x and y were swapped…
        from_dir1 = (2 + from_dir1) % 8  # …rotate directions 180°
        from_dir2 = (2 + from_dir2) % 8
```

```python
if from_dir1 > 4 or (from_dir1 in {0,4} and from_dir2 > 4):
    from_dir1 = (8 - from_dir1) % 8
    from_dir2 = (8 - from_dir2) % 8
```

```python
if 2 < from_dir1 < 6 or (from_dir1 in {2,6} and 2 < from_dir2 < 6):
    from_dir1 = (4 - from_dir1) % 8
    from_dir2 = (4 - from_dir2) % 8
```

```python
else:
    rotation = (from_dir1 // 2) * 2      # quantise dir1 to nearest axis (0,2,4,6)
    from_dir1 = (from_dir1 - rotation) % 8
    from_dir2 = (from_dir2 - rotation) % 8
    if (from_dir2 - from_dir1) % 8 > 4:
        from_dir2 = (8 + 2*from_dir1 - from_dir2) % 8
```

```python
multiweights[dims][dir1][dir2] = \
    multiweights[from_dims][from_dir1][from_dir2]
```

---

**User**

explain conceptually which dir1 and dir2 share weights with which and why

---

**Cursor**

The eight directions are labelled counter-clockwise starting at the positive x axis  

0 = E, 1 = NE, 2 = N, 3 = NW, 4 = W, 5 = SW, 6 = S, 7 = SE.

The network is supposed to be equivariant under the whole D4 symmetry
group of the square (rotations by 90° k and reflections in the x-, y-
and diagonal axes) ​plus​ the bookkeeping detail that “swapping the x-
and y-fields” (dims\[3\]↔dims\[4\]) is itself a symmetry.  
If two ⟨dir1, dir2⟩ pairs can be carried into each other by one of those
symmetries they must share the **same** weight matrix.

How the code creates those ties
--------------------------------
1. 180° turn when the x/y fields are swapped  
   • ‟swap x↔y” ⇝ rotate every direction by +2 (mod 8).

2. Two reflections to push the *first* direction into the north-east
   quadrant  
   • mirror in the ​y​-axis if dir1 points to the west half-plane  
   • mirror in the ​x​-axis if it points to the south half-plane.  

   After these flips the first direction is guaranteed to lie in the
   sector bounded by −45° … +135° (indices 0-3).

3. A rotation that makes the first direction **canonical**:  
   rotation = 2 × ⌊dir1/2⌋  
   • Any cardinal (even) direction becomes 0 (E).  
   • Any diagonal (odd) direction becomes 1 (NE).

4. Possibly one last reflection so that the *second* direction sits
   within ±180° of the first (difference ≤ 4).

Once that four-step reduction is done the pair
⟨from_dir1, from_dir2⟩ is taken as the representative; all other
instances are overwritten to point at its tensor object.

The resulting equivalence classes
---------------------------------
Because every cardinal input direction collapses to E (0) and every
diagonal input collapses to NE (1), there are only **two** canonical
values for dir1.  For each of those, dir2 can differ by at most 4
steps, giving 5 possibilities.  Hence the 64 original pairs fall into
just

 2 (dir1 options) × 5 (relative dir2 options) = 10 weight-sharing
classes:

Cardinal first direction (dir1 canonical = 0)
• (0, 0)  “straight ahead”  
• (0, 1)  turn +45° (E→NE)  
• (0, 2)  turn +90° (E→N)  
• (0, 3)  turn +135° (E→NW)  
• (0, 4)  reverse (E→W)

Diagonal first direction (dir1 canonical = 1)
• (1, 1)  straight (NE→NE)  
• (1, 2)  turn +45° (NE→N)  
• (1, 3)  turn +90° (NE→NW)  
• (1, 4)  turn +135° (NE→W)  
• (1, 5)  reverse (NE→SW)

Examples of weight sharing
• Weight for W → S (4→6) = weight for E → N (0→2)  (180° rotation).  
• Weight for NW → W (3→4) = weight for NE → N (1→2) (reflection in y-axis).  
• Weight for S → SE (6→7) = weight for E → NE (0→1) (-90° rotation).

So when you update the representative tensor for, say, (0, 2), every
equivalent pair—(2, 4), (4, 6), (6, 0), (1, 3), (3, 5), (5, 7), (7, 1),
and their x/y-swapped versions—automatically uses exactly that same
matrix, guaranteeing the desired dihedral-equivariance of the layer.

