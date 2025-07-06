# Vectorized `share_up` Framework (Loop-Free)

This document outlines the two core stages required to replace the double-loop implementation of `share_up` with a pure-tensor, GPU-efficient approach that uses `torch.repeat_interleave` and `torch_scatter.segment_coo`.

---

## Step 1  – One-Time Metadata Construction (CPU)

Performed once at model initialisation; Python loops are fine because the result is cached.

1. **Gather slice stats**  
   From `flat_multitensor` obtain for every slice `t` (dims tuple):
   • `offset[t]`, `length[t]`, `shape[t]`.

2. **Compute broadcast factors**  
   For each pair `(t → s)` where `dims(t) ≤ dims(s)` (component-wise):
   ```text
   tile_factor[t→s] = ∏ axis_length_s   # product of axis lengths that are missing in t
   ```

3. **Slice-level repeat totals**  
   ```text
   repeat_total[t] = Σ_{s ancestor of t} tile_factor[t→s]
   ```
   Every row inside slice `t` will be duplicated exactly `repeat_total[t]` times.

4. **Populate global index tensors in **source-row order**

   For each slice `t` (iterate once):
   * `repeat_counts_t`  – a length-`length[t]` vector filled with `repeat_total[t]`.
   * `dst_rows_t`       – explicit destination row indices for **all** repeated copies of rows in `t`.  Uses `tile_factor[t→s]` and `shape[s]` to enumerate destinations.

   Append to global lists and finally concatenate:
   ```text
   repeat_counts ∈ ℕᴺ            # N = total rows, sorted by src rows
   dst_rows      ∈ ℕᴹ            # M = Σ repeat_counts, jointly sorted
   ```

   Store both tensors on GPU (int32).  Size: O(N · avg_fan-in) – a few MB.

---

## Step 2  – Loop-Free Forward Pass (GPU)

Executed every iteration; no Python loops.

```python
# flat : (N, C)  – down-projected residual buffer

src_expanded = torch.repeat_interleave(flat, repeat_counts, dim=0)  # (M, C)

out = torch_scatter.segment_coo(
    src_expanded,                     # values to add
    dst_rows,                         # COO row indices (already sorted)
    dim_size=N,                       # output rows
    reduce='sum'                      # aggregate
)  # -> (N, C)
```

• `repeat_interleave` duplicates each source row exactly the required number of times because `repeat_counts` is pre-aligned with source ordering.  
• `segment_coo` performs a single fused scatter-add, summing contributions into their destination rows.  
• Result `out` is the communicated tensor; gradients propagate automatically.

---

### Key Benefits

* **Zero runtime Python loops** – all heavy work in two CUDA kernels.
* **Tiny, immutable metadata** – int32 vectors instead of a huge sparse matrix.
* **Scales with channels** – arithmetic cost O(M · C), where M ≪ N². 