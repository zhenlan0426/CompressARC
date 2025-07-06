# Vectorized `share_up` Framework (Loop-Free)

### CSR Matrix Formulation

Some teams prefer classical sparse-matrix algebra instead of the repeat-interleave + scatter pattern above.  The same communication pattern can be expressed as a binary row-sparse matrix **S** that is multiplied with the flattened buffer at run-time.

```text
    out = S  @  flat.data        # (N×N  ·  N×C) → (N×C)
```

where `N = total_positions` and the non-zero pattern of **S** encodes exactly the ancestor relationships.

#### A. One-time CSR build (Python / CPU)
1.  **Enumerate valid slices**  — load `valid_dims.md`, build the list `all_dims`.
2.  **Create row-ptr vector**  (`indptr`, length `N+1`).  For every *destination* slice `s`
    * iterate over all *source* slices `t` such that `dims(t) ≤ dims(s)`.
    * each physical row inside `t` contributes **one** non-zero at column index `(row_offset_t + local_row_idx)`.
    * append column indices into `indices` list.
    * push the running length into `indptr`.
3.  **Values**  — all ones (`torch.ones_like(indices, dtype=torch.float32)`).
4.  **Transfer to GPU**  — convert to `torch.sparse_csr_tensor(indptr, indices, values, size=(N,N))` **once** and cache inside `FlatMultiTensor._share_up_cache`.

Memory footprint ≤ 12 bytes × `nnz`; in practice a few MB.

#### B. Forward pass (GPU)
```python
S = cached_csr   # (N, N)  on device
out = torch.sparse.mm(S, flat.data)   # (N, C)
```
•  Uses cuSPARSE/hipSPARSE CSRMM under the hood.
•  Autograd: backward calls the same kernel with transposed operands, nothing to implement.
•  Supports mixed precision (FP16/BF16) from PyTorch 2.1 onwards.

#### Pros / Cons vs. Scatter approach
|                               | CSR SpMM                 | repeat-interleave + scatter |
|-------------------------------|--------------------------|-----------------------------|
| Kernel count per step         | 1                        | 2                           |
| Peak GPU bandwidth utilisation| slightly higher (cuSPARSE tuned) | very good (fused scatter) |
| Metadata size                 | `nnz` × 12 B             | `M` × 4 B (`dst_rows`) + `N` × 4 B (`repeat_counts`) |
| Dynamic sparsity              | expensive (need rebuild) | cheap (just recompute `repeat_counts`) |

Choose the variant that best matches your update frequency and framework constraints.

--- 