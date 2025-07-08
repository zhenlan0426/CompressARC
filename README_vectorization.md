# CompressARC ‑ Vectorization Branch

## Overview
This experimental branch set out to **vectorize** the core tensor-processing pipeline used in ARC task solvers.
The motivation was to replace many small, per-slice tensor operations with batched, GPU-friendly kernels so that the same amount of work could be executed with fewer Python loops and kernel launches.

## Main Ideas
1. **Flat representation (`FlatMultiTensor`)**  
   All logical tensor slices are packed into one tall matrix accompanied by lightweight metadata (`offsets`, `lengths`, `dims_list`, …).  This is implemented in `multitensor_systems_vec.py`.
2. **Vectorized layers (`layers_vec.py`)**  
   Core building blocks such as `normalize`, `affine`, and `share_up` were rewritten to operate on the flat buffer using:
   * `torch_scatter.segment_csr` for fast per-slice reductions
   * Sparse CSR matrices for broadcast / gather operations (e.g. `share_up`)
3. **Helper utilities & tests**  
   * `flat_multitensor`, `unpack_flat` helpers to convert back-and-forth.  
   * `layers_vec_test.py` validates forward & backward correctness and benchmarks speed.

## Benchmark Results
Running `python layers_vec_test.py` on an NVIDIA RTX 4090 (CUDA 11.8, PyTorch 2.1) produced the following numbers for **10 forward-&-backward iterations on 5 real-sized tasks** (`channel_dim = 16`).

| Operation | Regular | Vectorized | Speed-up |
|-----------|---------|-----------|----------|
| `normalize` | 0.153 s | 0.133 s | **1.15×** |
| `affine` (shared W) | 0.082 s | 0.041 s | **1.99×** |
| `affine` (per-slice W) | 0.081 s | 0.102 s | 0.80× |
| `share_up` (+ residual) | 0.315 s | 0.411 s | 0.77× |

The headline speed-up peaks around **1.1–1.2×** and occasionally the vectorized kernels are *slower* than their slice-wise counterparts due to:
* Conversion overhead (packing/unpacking `FlatMultiTensor`)
* Kernel launch latency already amortized in the highly optimised baseline
* Extra memory traffic introduced by CSR scatter/gather operations

## Outcome & Decision
While the exercise produced cleaner mathematical kernels, the real-world wall-clock gains are **insufficient to justify the added complexity**:

* Considerable engineering effort to maintain a parallel set of vectorised layers
* Larger memory footprint (dense buffer + CSR metadata)
* Debugging difficulty when stepping outside the well-tested slice-wise code path

**Therefore this branch will not be merged into `main` and is effectively *abandoned*.**  The code remains available for archival / educational purposes, but no further development is planned.

## Lessons Learned
1. Micro-kernel vectorisation must deliver >2× speed-up to overcome engineering cost.
2. Converting irregular, sparse problems into dense formats can backfire once packing/unpacking overhead is counted.
3. Profiling early and often prevents sunk-cost fallacy – the `layers_vec_test.py` harness was invaluable here.

## Next Steps (if revisited)
* Investigate *fused* CUDA kernels that avoid the intermediate CSR indirection.
* Explore higher-level algorithmic improvements (e.g. caching, smarter batching) which may give larger wins than low-level vectorisation.
* Keep a close eye on forthcoming PyTorch 2.x compiler optimisations that could make naïve Python loops competitive without manual vectorisation.

---
**Status:** ⛔ Abandoned – kept for reference only. 