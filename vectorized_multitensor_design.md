# Vectorised MultiTensor Framework (Uniform Channel Dimension)

This document captures the **current blueprint** for replacing the nested-list `MultiTensor` with a single, flat, GPU-friendly representation.  It will evolve as individual layers are ported.

---

## 1. Rationale
* Original code keeps one `MultiTensor` *per valid dims combination* – a nested python list of small tensors.
* That incurs Python-level iteration and launches many CUDA kernels per layer, limiting throughput.
* We standardise all logical slices to the **same channel width `C = 16`** and pack them into **one contiguous buffer**.  
  Slices that conceptually need fewer channels simply ignore the unused ones.

## 2. Core Abstraction – `FlatMultiTensor`
| Field          | Shape / Type                                  | Purpose |
| -------------- | --------------------------------------------- | ------------------------------------------------------------ |
| `data`         | `(total_positions, C)` **tensor**             | All activations concatenated along a new *position* axis. |
| `offsets`      | `(n_slices,) long`                            | Starting row in `data` for each logical slice. |
| `lengths`      | `(n_slices,) long`                            | Number of rows belonging to the slice. |
| `shapes`       | `List[List[int]]`                             | Spatial extent (E, Col, Dir, X, Y) per slice (no channel). |
| `dims_list`    | `List[Tuple[int, …]]`                         | The 5-bit mask that identifies the slice. |
| `channel_dim`  | `int`                                         | The uniform channel width (`16`). |

Helper methods:
* `view(idx)` – returns a **view** of slice *idx* with its original spatial shape + channel.
* `write(idx, tensor)` – in-place update of a slice with shape checks.
* `as_nested_list()` – debugging utility that reconstructs the original structure.

## 3. Packing & Unpacking
| Utility               | Description |
| ----------------------| ----------- |
| `pack_multitensor(mt, system, C)` | Walks the old nested structure, builds metadata, allocates one big buffer and copies data. |
| `unpack_flat(flat, system)`       | Re-creates a nested `MultiTensor` (slow, for regression tests). |

Both live in `multitensor_systems_vec.py`.

## 4. Naming Convention
* Vectorised rewrites mirror the existing file names and append `*_vec.py`  
  e.g. `multitensor_systems_vec.py`, later `layers_vec.py`, etc.
* Test files are named `<original_file_name>_test.py`.

## 5. Incremental Porting & Testing
1. **Layer by layer** we implement a vectorised version that operates directly on `FlatMultiTensor`.
2. Test harness (pytest):
   * Build small random tasks → create both nested and flat inputs.
   * Run reference layer vs. vectorised layer.
   * Assert forward equality (`allclose`) and, optionally, gradient equality.
3. End-to-end model test will be enabled once every sub-layer has a vectorised counterpart.

## 6. Future Extensions
* Pre-computed index tables (`coord_x`, `parent_indices`, …) will be added once needed by share/directive layers.
* Support for different `C` via compile-time constant if memory becomes a concern.
* Potential JIT / TorchScript wrapping once all kernels are consolidated.

---
*Last updated: initial scaffolding committed; will expand as porting progresses.* 