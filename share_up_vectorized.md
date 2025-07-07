# Vectorized `share_up` — How It Works Under the Hood

The `share_up` operation is the **core broadcast-style communication primitive** used throughout the ARC compressor.  Conceptually, it takes every logical tensor slice in the `MultiTensor` hierarchy and **adds into it the values of all *ancestor* slices**—those whose dimension mask is a subset of the slice’s own mask—using NumPy/PyTorch broadcasting rules.

Traditionally this was implemented with nested Python loops and many `unsqueeze`/`mean` operations (see `layers.py`).  The vectorised rewrite found in `layers_vec.py` replaces these loops with **one sparse matrix multiplication (SpMM)**:

```python
out_data = torch.sparse.mm(S, flat.data)  # (N, N) ⋅ (N, C) → (N, C)
```
where
* `flat.data` holds every logical slice flattened and stacked (total **N** rows, **C** channels), and
* `S ∈ {0,1}^{N×N}` is a (highly-structured) **binary CSR matrix** with exactly one row per *destination* row and one column per *source* row.

The rest of this note dives into how that matrix is built, cached, and applied.

---
## 1.  Flattening the hierarchy
All tensor slices live inside a `FlatMultiTensor` object created by

```python
flat = flat_multitensor(mt)  # helper in multitensor_systems_vec.py
```

Key pieces of metadata captured during flattening:

| field                | shape / type                         | purpose |
|----------------------|--------------------------------------|---------|
| `data`               | `(N, C)`                             | concatenated slice contents |
| `offsets`            | `(n_slices,)` **long**               | start row of each slice inside `data` |
| `lengths`            | `(n_slices,)` **long**               | number of rows in each slice |
| `shapes`             | `List[List[int]]`                    | original spatial shape of every slice (no channel dim) |
| `dims_list`          | `List[Tuple[int,int,int,int,int]]`   | 5-bit mask describing which logical axes this slice owns <br/>(examples, colours, directions, x, y) |
| `row2slice`          | `(N,)` **long**                      | maps global row → owning slice |

Having all slices in *one* matrix means a single SpMM can handle the full hierarchy.

---
## 2.  Semantics of `share_up`
For every destination slice **s** and every position **p** inside that slice, we must **sum** the value at `p` with the values coming from all *ancestor* slices **t** satisfying

```
∀ axis a:  dims_t[a] ≤ dims_s[a]
```

If an ancestor is missing an axis that the destination has, its value is broadcast (copied) along that axis.  Therefore each destination row ends up adding between `1` and `2⁵ = 32` source rows (one per possible ancestor mask).

---
## 3.  Building the sparse matrix S
`FlatMultiTensor.build_share_up_metadata()` constructs `S` **once per distinct layout** and stores it in a class-level cache:

```python
key = (tuple(lengths.tolist()), tuple(dims_list))
```

Steps (all vectorised on CPU for speed):

1. **Pre-compute axis strides per slice** – let us convert quickly between a row’s *local linear index* and its multi-index coordinates.
2. **Ancestor lookup table** – for every slice `s` list the indices of slices `t` that satisfy the subset check above.
3. **Loop over slices (not rows)** – for each destination slice `s`:
   a. create a tensor `dest_local_idx = 0..length_s-1`.<br/>
   b. recover the full 5-D *logical coordinates* of every row in `s` using integer division & modulo with the pre-computed strides.
   c. for each ancestor `t` convert those coordinates back into *t*’s local row indices, shift by `offsets[t]`, and collect them in a matrix of shape `(length_s, n_anc)`.
4. **Assemble CSR buffers**:
   * The flattened `(length_s, n_anc)` block becomes a chunk of the `indices` array.
   * `indptr` is extended by `length_s` elements, each increasing by `n_anc`.

<!-----  BEGIN DEEP-DIVE  ‑---->

### 3.1  Pre-computing per-slice strides
For every slice **s** we need a quick way to map a **local linear index** `i ∈ [0, length_s)` back to its **multi-index coordinates** `(e, c, d, x, y)`.
This is just the classic row-major offset formula:

```text
idx = ((((e * C + c) * D + d) * X + x) * Y) + y
```

Taking the partial derivatives of this formula yields the stride for each *present* axis.
We store a small dict:

```
axis_info_s[axis] = (position_in_shape, stride)
```

so we can recover coordinates with two fused integer ops per axis (division & modulo).

### 3.2  Enumerating ancestors
Two slices are in an ancestor/descendant relation iff their 5-bit masks satisfy
`dims_t ≤ dims_s` component-wise.  This is evaluated once per pair and the indices cached
in `ancestor_lists[s]`.

### 3.3  Vectorised coordinate lifting
Instead of iterating over **every row** we do the following *per destination slice*:

```python
L = length_s
idx = torch.arange(L)              # (L,)
coords_present = idx[:, None] // strides_dest   # broadcast division gives all axes at once
coords_present %=  strides_dest                  # keep modulo to peel off lower dims
```

The result is a `(L, |axes_s|)` tensor of coordinates which we then expand to full 5-D by scattering into a zero tensor `(L, 5)`.

### 3.4  Projecting into ancestor index spaces
Given the full coordinates we compute the *ancestor*’s local linear index via a dot product with its own stride vector:

```python
src_local_idx = (coords_all[:, anc_axes] * anc_strides).sum(1)
```

and shift by the slice’s global offset to get a **global row id** usable in the CSR matrix.
Because this is applied to the entire `(L,)` vector at once it stays in highly-optimised C++/OpenMP code.

### 3.5  Assembling CSR pieces
We stack the `n_anc` column vectors side-by-side -> `(L, n_anc)`; flatten it row-major, and append to the running `indices` list.  The corresponding `indptr` extension is simply `torch.arange(1, L+1) * n_anc + running_nnz`.

> **Micro-example** – suppose slice *s* has `length_s = 4` and two ancestors.
> We might build the tiny block
>
> ```text
> indices_block = [ 7,  3,   8,  4,   9,  5,  10,  6 ]
>                 |---- row 0 ----|  |---- row 1 ----|  ...
> ```
>
> and extend `indptr` by `[0, 2, 4, 6, 8] + running_nnz`.

On completion of the outer loop the concatenated `indices_t` and `indptr_t` already form *valid* CSR buffers—no further sorting or de-duplication is required because ancestor enumeration guarantees uniqueness.

<!-----  END DEEP-DIVE  ‑---->

Because all non-zero values are *1*, the `values` array is just `torch.ones(nnz)`.

The result is an `(N, N)` CSR tensor containing ~`N·⟨#ancestors⟩` ones (the average number of ancestors is ≤ 32, usually much lower).

---
## 4.  Applying `share_up`
Once `S` is cached the forward path is tiny:

```python
def share_up(flat: FlatMultiTensor) -> FlatMultiTensor:
    S = flat._share_up_S  # device-appropriate copy
    out_data = torch.sparse.mm(S, flat.data)  # SpMM does the summation
    return FlatMultiTensor(  # share metadata, replace data
        data=out_data,
        offsets=flat.offsets,
        lengths=flat.lengths,
        shapes=flat.shapes,
        dims_list=flat.dims_list,
        channel_dim=flat.channel_dim,
        row2slice=flat.row2slice,
    )
```

Highlights:
* **No Python loops** – all heavy work is delegated to the highly-optimised cuSPARSE/hipSPARSE kernels.
* **Autograd-ready** – PyTorch already supports gradients through `torch.sparse.mm`, so nothing extra is required.
* The returned `FlatMultiTensor` is *shallow*—it reuses all metadata so downstream ops remain zero-copy.

---
## 5.  Performance & memory considerations
* **Time complexity**:  `O(nnz · C)`, where `nnz = Σ length_s · n_anc(s)`.  In practice, vectorisation reduces Python overhead by 2–3 orders of magnitude compared to the loop implementation.
* **Memory footprint**: `indptr` (`N+1` ints) + `indices` (`nnz` ints) + `values` (`nnz` scalars).  Storing on CPU and only moving to GPU when needed avoids duplicating the structure across devices.
* **Cache reuse**:  as long as the sequence of slices (masks + lengths) is identical, *any* `FlatMultiTensor`—training, evaluation, gradients—shares the same `S` object.

---
## 6.  Worked example (2-D toy)
Assume only two logical axes `(examples, x)` and three slices:

| slice id | mask | shape | flattened rows |
|----------|------|-------|----------------|
| 0        | (1,0) | `(B,)`   | rows `0..B-1`      |
| 1        | (0,1) | `(W,)`   | rows `B..B+W-1`    |
| 2        | (1,1) | `(B,W)` | rows `B+W..B+W+BW-1` |

`share_up` needs to add slices 0 and 1 into slice 2.  Matrix `S` therefore looks like:

```
   ┌── dest rows (B) ─┐ ┌─── dest rows (W) ─┐ ┌──────── dest rows (BW) ────────┐
     identity(B)          identity(W)         [I_B ⊗ 1_W | 1_B ⊗ I_W | I_BW ]
```

When multiplied with `data`, each `BW` row receives one value from the `(1,0)` ancestor, one from `(0,1)`, and one from itself—exactly the desired broadcast-sum behaviour.

---
## 7.  Summary
`share_up` condenses an otherwise expensive nested-loop broadcast into a **single sparse matrix multiplication**.  The key ingredients are:

1.  A *flat* view of the multi-tensor hierarchy,
2.  A lazily-cached CSR matrix encoding ancestor relationships, and
3.  Leveraging GPU-accelerated SpMM for both speed and automatic differentiation.

This design removes virtually all Python overhead, scales to millions of tensor positions, and plays nicely with PyTorch’s autograd.
