import numpy as np
import torch
from typing import List, Tuple, Optional
import itertools

# Keep the same dimensional semantics: examples, colors, directions, x, y
NUM_DIMENSIONS = 5

class FlatMultiTensor:
    """A minimal flat representation that stores *all* logical tensors in one buffer.

    Each logical slice has the same channel width (``channel_dim``).  Metadata
    arrays map a slice index (corresponding 1-to-1 with a valid ``dims`` mask)
    to its row range inside ``data``.
    """

    def __init__(
        self,
        data: torch.Tensor,                # (total_positions, channel_dim)
        offsets: torch.Tensor,             # (n_slices,) start row for each slice
        lengths: torch.Tensor,             # (n_slices,) number of rows for each slice
        shapes: List[List[int]],           # list of full spatial shapes (len = n_slices)
        dims_list: List[Tuple[int, ...]],  # list of the 5-bit masks that identify each slice
        channel_dim: int,
        indptr: torch.Tensor = None,
    ):
        self.data = data
        self.offsets = offsets
        self.lengths = lengths
        self.shapes = shapes
        self.dims_list = dims_list
        self.channel_dim = channel_dim
        if indptr is None:
            # Pre-compute CSR indptr for segment_csr operations
            self.indptr = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0)])
        else:
            self.indptr = indptr
        self.build_share_up_metadata()

    # ------------------------------------------------------------------
    # Class-level cache for expensive share-up metadata.
    # ------------------------------------------------------------------
    _share_up_cache: dict = {}

    def __add__(self, other):
        return FlatMultiTensor(
            data=self.data + other.data,
            offsets=self.offsets,
            lengths=self.lengths,
            shapes=self.shapes,
            dims_list=self.dims_list,
            channel_dim=self.channel_dim,
            indptr=self.indptr,
        )

    def build_share_up_metadata(self):
        """Construct (or fetch from cache) the CSR matrix *S* that realises the
        ``share_up`` operation for the current *FlatMultiTensor* layout.

        The matrix has shape (N, N) where *N = total_positions*.  Row *i*
        gathers (sums) the rows of all ancestor slices *t* (those whose
        dimension mask satisfies ``dims(t) \u2264 dims(s)``) that map to the
        same logical coordinates as row *i* once broadcasting rules are taken
        into account.

        The build happens once on CPU and the result is cached at the
        class-level so subsequent tensors that share the same metadata (i.e.
        same ``dims_list``/``lengths`` ordering) can reuse it immediately.
        """
        # Already built for this instance – nothing to do.
        if hasattr(self, "_share_up_S"):
            return

        # ------------------------------------------------------------------
        # Decide on a cache key.  The pair (lengths, dims_list) uniquely
        # identifies the logical layout irrespective of the *data* tensor.
        # ------------------------------------------------------------------
        key = (tuple(self.lengths.tolist()), tuple(self.dims_list))

        # Fetch from cache if available
        if key in FlatMultiTensor._share_up_cache:
            self._share_up_S = FlatMultiTensor._share_up_cache[key].to(self.data.device)
            return

        # ------------------------------------------------------------------
        # Derive global constants
        # ------------------------------------------------------------------
        N = int(self.data.shape[0])  # total rows across all slices

        # Determine the length of each of the 5 logical axes (examples, colors,
        # directions, x, y) by inspecting the recorded shapes.
        axis_lengths = [None] * NUM_DIMENSIONS  # type: List[Optional[int]]
        for dims, shape in zip(self.dims_list, self.shapes):
            pos = 0
            for axis in range(NUM_DIMENSIONS):
                if dims[axis]:
                    length_val = shape[pos]
                    if axis_lengths[axis] is None:
                        axis_lengths[axis] = length_val
                    else:
                        # Sanity check – layouts must be consistent across slices
                        assert axis_lengths[axis] == length_val
                    pos += 1
        # Replace any still-None entries with 1 (should not occur)
        axis_lengths = [l if l is not None else 1 for l in axis_lengths]

        # ------------------------------------------------------------------
        # Pre-compute per-slice helpers: mapping from logical axis -> (position
        # within `shape`, stride) to convert between flattened index and
        # multi-index coordinates quickly.
        # ------------------------------------------------------------------
        slice_axis_meta = []  # list[dict[axis -> (pos_in_shape, stride)]]
        for dims, shape in zip(self.dims_list, self.shapes):
            # Compute strides under row-major (C) order for this *shape*.
            strides_c = []
            stride_val = 1
            for length_val in reversed(shape):
                strides_c.append(stride_val)
                stride_val *= length_val
            strides_c = list(reversed(strides_c))

            axis_info = {}
            pos = 0
            for axis in range(NUM_DIMENSIONS):
                if dims[axis]:
                    axis_info[axis] = (pos, strides_c[pos])
                    pos += 1
            slice_axis_meta.append(axis_info)

        # ------------------------------------------------------------------
        # Pre-compute ancestor list for each slice (dims(t) <= dims(s)).
        # ------------------------------------------------------------------
        ancestor_lists = []  # list[list[int]]
        for s_dims in self.dims_list:
            ancestors = [t_idx for t_idx, t_dims in enumerate(self.dims_list)
                         if all(t <= s for t, s in zip(t_dims, s_dims))]
            ancestor_lists.append(ancestors)

        # ------------------------------------------------------------------
        # Build CSR data structures **much faster** by vectorising over each
        # slice instead of iterating row-by-row in Python.  This reduces the
        # Python-level loop count from *total_positions* to *num_slices*.
        # ------------------------------------------------------------------
        indptr: List[int] = [0]
        indices_blocks: List[torch.Tensor] = []
        running_nnz = 0
        cpu_device = torch.device("cpu")

        for s_idx, (offset_s, length_s, dims_s, shape_s) in enumerate(
            zip(self.offsets.tolist(), self.lengths.tolist(), self.dims_list, self.shapes)
        ):
            ancestors = ancestor_lists[s_idx]
            n_anc = len(ancestors)
            if n_anc == 0 or length_s == 0:
                # Should never happen but guard for safety
                indptr.extend([indptr[-1]] * length_s)
                continue

            # Vector of local indices 0 .. length_s-1
            dest_local_idx = torch.arange(length_s, dtype=torch.int64, device=cpu_device)

            # Compute per-axis coordinates for destination slice
            axis_info = slice_axis_meta[s_idx]
            present_axes = [axis for axis in range(NUM_DIMENSIONS) if dims_s[axis]]
            strides_dest = torch.tensor(
                [axis_info[axis][1] for axis in present_axes], dtype=torch.int64, device=cpu_device
            )

            remainder = dest_local_idx.unsqueeze(1)  # (L, 1)
            coords_present = torch.empty(length_s, len(strides_dest), dtype=torch.int64, device=cpu_device)
            for col, stride_val in enumerate(strides_dest):
                coords_present[:, col] = remainder[:, 0] // stride_val
                remainder = remainder % stride_val

            # Map to full 5-D global coordinates
            coords_all = torch.zeros(length_s, NUM_DIMENSIONS, dtype=torch.int64, device=cpu_device)
            pos = 0
            for axis in present_axes:
                coords_all[:, axis] = coords_present[:, pos]
                pos += 1

            # Build indices for all ancestors in a vectorised manner
            src_globals_per_anc = []
            for anc_idx in ancestors:
                anc_info = slice_axis_meta[anc_idx]
                anc_axes = [axis for axis in range(NUM_DIMENSIONS) if self.dims_list[anc_idx][axis]]
                anc_strides = torch.tensor(
                    [anc_info[axis][1] for axis in anc_axes], dtype=torch.int64, device=cpu_device
                )
                src_local_idx = (coords_all[:, anc_axes] * anc_strides).sum(dim=1)
                src_global_idx = src_local_idx + int(self.offsets[anc_idx].item())
                src_globals_per_anc.append(src_global_idx)

            # Stack to shape (L, n_anc) then flatten row-major
            src_globals_mat = torch.stack(src_globals_per_anc, dim=1)  # (L, n_anc)
            indices_block = src_globals_mat.reshape(-1)  # row-major flatten
            indices_blocks.append(indices_block)

            # Update indptr entries for these rows
            indptr.extend((running_nnz + (torch.arange(1, length_s + 1, dtype=torch.int64, device=cpu_device) * n_anc)).tolist())
            running_nnz += length_s * n_anc

        # Concatenate all index blocks from every slice
        indices_t = torch.cat(indices_blocks).to(cpu_device)

        # ------------------------------------------------------------------
        # Assemble CSR tensor and cache it.
        # ------------------------------------------------------------------
        indptr_t = torch.tensor(indptr, dtype=torch.int64, device=cpu_device)
        values_t = torch.ones(indices_t.numel(), dtype=self.data.dtype, device=cpu_device)

        csr = torch.sparse_csr_tensor(indptr_t, indices_t, values_t, size=(N, N))

        # Store CPU version in global cache and device-specific version on instance
        FlatMultiTensor._share_up_cache[key] = csr.cpu()
        self._share_up_S = csr.to(self.data.device)

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def view(self, idx: int) -> torch.Tensor:
        """Return the *view* of slice ``idx`` with its original spatial shape."""
        start = int(self.offsets[idx].item())
        end = start + int(self.lengths[idx].item())
        shape = self.shapes[idx] + [self.channel_dim]
        return self.data.narrow(0, start, end - start).view(*shape)

    def write(self, idx: int, tensor: torch.Tensor) -> None:
        """In-place write ``tensor`` into slice ``idx`` (shape checked)."""
        start = int(self.offsets[idx].item())
        length = int(self.lengths[idx].item())
        expected_shape = self.shapes[idx] + [self.channel_dim]
        if list(tensor.shape) != expected_shape:
            raise ValueError(f"Tensor shape mismatch: expected {expected_shape}, got {list(tensor.shape)}")
        self.data.narrow(0, start, length).copy_(tensor.view(length, self.channel_dim))

    # -------------------------- debug utilities --------------------------
    def as_nested_list(self):
        """Reconstruct a python nested list with the original MultiTensor layout.
        This is *slow* and intended only for debugging.
        """
        result = {}
        for idx, dims in enumerate(self.dims_list):
            result[tuple(dims)] = self.view(idx)
        return result


def flat_multitensor(mt, debug=False) -> FlatMultiTensor:
    """Convert the nested ``MultiTensor`` *mt* into a ``FlatMultiTensor``.

    Args
    -----
    mt : multitensor_systems.MultiTensor
        The original nested structure we want to flatten.

    Returns
    -------
    FlatMultiTensor
        Flattened buffer + metadata.
    """

    offsets: List[int] = []
    lengths: List[int] = []
    shapes: List[List[int]] = []
    dims_list: List[Tuple[int, ...]] = []
    tensor_list: List[torch.Tensor] = []
    multitensor_system = mt.multitensor_system
    channel_dim = mt[1,1,1,1,1].shape[-1]
    # Build metadata first – iterate in the same order as multitensor_system
    running_total = 0
    for dims in multitensor_system:
        tensor = mt[dims]
        spatial_shape = list(tensor.shape[:-1])  # exclude channel dim
        num_pos = int(np.prod(spatial_shape))
        offsets.append(running_total)
        lengths.append(num_pos)
        shapes.append(spatial_shape)
        dims_list.append(tuple(dims))
        
        # Flatten the tensor and add to list for concatenation
        flattened = tensor.view(num_pos, channel_dim)
        tensor_list.append(flattened)
        
        running_total += num_pos

    # Concatenate all tensors to preserve gradients
    data = torch.cat(tensor_list, dim=0)
    if debug:
        data = data.detach().requires_grad_(True)

    return FlatMultiTensor(
        data=data,
        offsets=torch.tensor(offsets, device=data.device, dtype=torch.long),
        lengths=torch.tensor(lengths, device=data.device, dtype=torch.long),
        shapes=shapes,
        dims_list=dims_list,
        channel_dim=channel_dim,
    )


def unpack_flat(flat: FlatMultiTensor, multitensor_system):
    """Convert a ``FlatMultiTensor`` back into the nested `MultiTensor` structure.

    Creates a fresh `MultiTensor` instance with the same layout as
    *multitensor_system* and fills it with copies of each logical slice that is
    currently stored in *flat*.

    Parameters
    ----------
    flat : FlatMultiTensor
        The flat representation to unpack.
    multitensor_system : multitensor_systems.MultiTensorSystem
        The system providing the target nested layout.

    Returns
    -------
    multitensor_systems.MultiTensor
        A new `MultiTensor` instance whose leaves hold clones of the tensors
        from *flat*.
    """
    nested = multitensor_system.make_multitensor(default=None)
    for idx, dims in enumerate(flat.dims_list):
        nested[dims] = flat.view(idx).clone()
    return nested