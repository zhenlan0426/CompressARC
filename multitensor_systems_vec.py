import numpy as np
import torch
from typing import List, Tuple
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
        row2slice: torch.Tensor,           # (total_positions,) mapping each row to its slice index
    ):
        self.data = data
        self.offsets = offsets
        self.lengths = lengths
        self.shapes = shapes
        self.dims_list = dims_list
        self.channel_dim = channel_dim
        self.row2slice = row2slice
        self.build_share_up_metadata()

    # ------------------------------------------------------------------
    # Class-level cache for expensive share-up metadata.
    # ------------------------------------------------------------------
    _share_up_cache: dict = {}

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

    def build_share_up_metadata(self):
        """Pre-compute the tensors required by the vectorised ``share_up`` routine.

        The method follows **Step 1** from ``share_up_vectorized.md``:

        1.  Compute the per-slice repeat totals ``repeat_total[t]`` – how many
            times every *row* inside slice *t* must be duplicated so that it can
            be scattered to *all* of its ancestor slices (including itself).
        2.  Construct a length-``N`` vector ``repeat_counts`` that lists
            ``repeat_total[t]`` *for every row* of slice *t*, respecting the
            global (source-row) ordering of the flat buffer.
        3.  Enumerate the explicit destination row indices ``dst_rows`` for the
            repeated rows.  These indices are produced **in the same order** as
            they will appear after ``torch.repeat_interleave`` is applied to the
            source buffer with ``repeat_counts`` – i.e. we iterate *slice → row
            → copy*.

        The resulting int32 tensors are stored as:
            • ``self.repeat_counts`` – shape ``(N,)``
            • ``self.dst_rows``     – shape ``(M,)``, where ``M = repeat_counts.sum()``
        """
        # Re-entry guard – if metadata is already attached to *this* instance.
        if hasattr(self, "csr_ptr") and self.csr_ptr is not None:
            return

        # ------------------------------------------------------------------
        # 0) Try to **reuse** cached metadata.  The structural layout of the
        #    tensor system (dims_list + per-axis lengths) is invariant across
        #    training steps, while the *data* buffer changes every iteration.
        #    Re-computing the mapping every time destroys any performance
        #    advantage of the vectorised implementation.  We therefore cache
        #    the CSR pointer + source index vector the *first* time they are
        #    built and reuse them for subsequent ``FlatMultiTensor`` objects
        #    created from the same multitensor layout.
        # ------------------------------------------------------------------

        # The cache **key** must uniquely identify the logical structure but
        # remain hashable.  We combine:
        #   • tuple(dims_list)          – 5-bit masks per slice (order matters)
        #   • tuple(global_dim_lengths) – lengths of the 5 physical axes
        # This is sufficient because shapes inside each slice are fully
        # determined by these two pieces of information.

        # Recover global axis lengths (examples, colours, dirs, x, y).
        global_lengths = []
        for i in range(NUM_DIMENSIONS):
            for dims, shape in zip(self.dims_list, self.shapes):
                if dims[i]:
                    # Pick the first occurrence – all slices share the same
                    # length along every physical axis.
                    shape_idx = sum(dims[:i])  # position of this axis in shape
                    global_lengths.append(shape[shape_idx])
                    break
        cache_key = (tuple(self.dims_list), tuple(global_lengths))

        cached = FlatMultiTensor._share_up_cache.get(cache_key)
        if cached is not None:
            # Move cached tensors to the *current* device if needed.
            self.src_rows = cached[0].to(self.data.device)
            self.csr_ptr = cached[1].to(self.data.device)
            return  # <-- done – no need to recompute

        # ------------------------------------------------------------------
        # Helper: recover the *global* length of every physical axis.  We scan
        # the existing slice shapes – all slices share the same length per axis
        # so we just pick the first occurrence.
        # ------------------------------------------------------------------
        dim_lengths = [None] * NUM_DIMENSIONS  # examples, colours, dirs, x, y
        for dims, shape in zip(self.dims_list, self.shapes):
            shape_idx = 0
            for axis in range(NUM_DIMENSIONS):
                if dims[axis]:
                    if dim_lengths[axis] is None:
                        dim_lengths[axis] = shape[shape_idx]
                    shape_idx += 1
            if all(l is not None for l in dim_lengths):
                break  # all discovered
        if any(l is None for l in dim_lengths):
            raise RuntimeError("Failed to infer global axis lengths from slices.")

        # Pre-compute strides for every slice so that we can convert between
        # flat row indices <–> multi-dim coordinates efficiently.
        slice_strides = []  # list[ list[int] ] parallel to self.shapes
        for shape in self.shapes:
            strides = []
            running = 1
            for length in reversed(shape):
                strides.insert(0, running)
                running *= length
            slice_strides.append(strides)

        # Convenience: fast lookup – axis index -> (shape_index_in_slice)
        axis_to_shape_idx = []  # list[ dict[int, int] ] per slice
        for dims in self.dims_list:
            mapping = {}
            shape_pos = 0
            for axis in range(NUM_DIMENSIONS):
                if dims[axis]:
                    mapping[axis] = shape_pos
                    shape_pos += 1
            axis_to_shape_idx.append(mapping)

        N = int(self.data.shape[0])

        # Containers for the global metadata being built.
        # We no longer rely on per-source repeat counts during the actual
        # communication step.  Instead, we construct two parallel lists that
        # enumerate – _for every copy_ – the *source* row index and its
        # *destination* row index.  These will later be sorted by destination
        # so that we can build a CSR representation expected by
        # ``torch_scatter.segment_csr``.

        src_rows_list: List[int] = []  # (M,)
        dst_rows_list: List[int] = []  # (M,)

        # No need to accumulate per-source repeat counts in the new CSR path.
        # We'll keep the variable name only to avoid refactoring larger loops,
        # but it remains unused.
        # repeat_counts_list: List[int] = []  # deprecated

        # ------------------------------------------------------------------
        # Iterate over *source* slices – this is the outermost iteration because
        # both ``repeat_counts`` and ``dst_rows`` have to be ordered by source
        # rows.
        # ------------------------------------------------------------------
        for src_idx, src_dims in enumerate(self.dims_list):
            src_offset = int(self.offsets[src_idx].item())
            src_length = int(self.lengths[src_idx].item())
            src_shape = self.shapes[src_idx]
            src_strides = slice_strides[src_idx]
            src_axis_map = axis_to_shape_idx[src_idx]

            # Identify *ancestor* slices – those whose dims superset src_dims.
            ancestor_indices = []
            for dst_idx, dst_dims in enumerate(self.dims_list):
                is_ancestor = all((not src_dims[ax]) or dst_dims[ax] for ax in range(NUM_DIMENSIONS))
                if is_ancestor:
                    ancestor_indices.append(dst_idx)

            # ------------------------------------------------------------------
            # Compute per-ancestor tile_factors – product of lengths for axes
            # that are in dst but not in src.  We no longer need the slice-wide
            # repeat_total, only individual tile factors for enumeration.
            # ------------------------------------------------------------------
            per_ancestor_tile_factors = {}
            for dst_idx in ancestor_indices:
                dst_dims = self.dims_list[dst_idx]
                # tile_factor = product of lengths for axes that are in dst but not in src
                tf = 1
                for ax in range(NUM_DIMENSIONS):
                    if (not src_dims[ax]) and dst_dims[ax]:
                        tf *= dim_lengths[ax]
                per_ancestor_tile_factors[dst_idx] = tf

            # ------------------------------------------------------------------
            # Enumerate destination rows for every *row* inside this slice.
            # ------------------------------------------------------------------
            # Precompute coordinate arrays for the source slice to avoid repeated
            # div/mod in innermost loops.  We convert every flat row idx -> list
            # of coords (len = |src_dims|).
            if src_length > 0:
                coords_tensor = torch.arange(src_length, dtype=torch.long)
                coords_list = [[] for _ in range(src_length)]
                for axis_pos, length in enumerate(src_shape):
                    stride = src_strides[axis_pos]
                    coord_vals = coords_tensor // stride
                    coords_tensor = coords_tensor % stride
                    for r in range(src_length):
                        coords_list[r].append(int(coord_vals[r].item()))

            # Iterate rows in *source-row* order.
            for local_row_idx in range(src_length):
                src_row_global = src_offset + local_row_idx
                src_coords = coords_list[local_row_idx]

                # For each ancestor slice produce its contribution indices.
                for dst_idx in ancestor_indices:
                    dst_dims = self.dims_list[dst_idx]
                    dst_offset = int(self.offsets[dst_idx].item())
                    dst_shape = self.shapes[dst_idx]
                    dst_strides = slice_strides[dst_idx]
                    dst_axis_map = axis_to_shape_idx[dst_idx]

                    tf = per_ancestor_tile_factors[dst_idx]
                    if tf == 1:
                        # Fast-path: dst == src (same dims) → index is identical.
                        dst_rows_list.append(src_row_global)
                        src_rows_list.append(src_row_global)
                        continue

                    # Axes that need to be enumerated (present in dst, absent in src)
                    missing_axes = [ax for ax in range(NUM_DIMENSIONS) if (not src_dims[ax]) and dst_dims[ax]]

                    # Prepare coordinate template – will fill in missing axes later.
                    dst_coords_template = [None] * len(dst_shape)
                    # Copy shared axis coordinates from src.
                    for ax in range(NUM_DIMENSIONS):
                        if src_dims[ax]:
                            dst_shape_idx = dst_axis_map[ax]
                            src_shape_idx = src_axis_map[ax]
                            dst_coords_template[dst_shape_idx] = src_coords[src_shape_idx]

                    # Enumerate cartesian product over missing axes.
                    missing_axes_shape_idxs = [dst_axis_map[ax] for ax in missing_axes]
                    missing_axes_lengths = [dim_lengths[ax] for ax in missing_axes]

                    for prod_coords in itertools.product(*[range(l) for l in missing_axes_lengths]):
                        for idx, coord in enumerate(prod_coords):
                            dst_coords_template[missing_axes_shape_idxs[idx]] = coord
                        # Now compute flat index in dst slice.
                        flat_idx_in_slice = 0
                        for dim_pos, coord in enumerate(dst_coords_template):
                            flat_idx_in_slice += coord * dst_strides[dim_pos]
                        dst_global = dst_offset + flat_idx_in_slice
                        dst_rows_list.append(dst_global)
                        src_rows_list.append(src_row_global)

        # ----------------------------------------------------------------------
        # Convert Python lists to tensors and *sort by destination row* so that
        # we can build an efficient CSR pointer once and re-use it in every
        # subsequent ``share_up`` call.
        # ----------------------------------------------------------------------
        dst_tensor = torch.tensor(dst_rows_list, dtype=torch.int32, device=self.data.device)
        src_tensor = torch.tensor(src_rows_list, dtype=torch.int32, device=self.data.device)

        sort_idx = torch.argsort(dst_tensor)
        dst_sorted = dst_tensor[sort_idx]
        src_sorted = src_tensor[sort_idx]

        # Build CSR pointer: counts per destination row → prefix sum.
        counts = torch.bincount(dst_sorted, minlength=N).to(torch.int32)
        ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=self.data.device), torch.cumsum(counts, dim=0)])

        # Store metadata tensors – they will be reused by the fast path.
        self.src_rows = src_sorted
        self.csr_ptr = ptr

        # ------------------------------------------------------------------
        # 6) **Cache** the freshly computed metadata for reuse.
        # ------------------------------------------------------------------
        # Store CPU copies to maximise portability across devices; we'll move
        # them to the correct device when loading from the cache.
        FlatMultiTensor._share_up_cache[cache_key] = (
            self.src_rows.cpu(),
            self.csr_ptr.cpu(),
        )

        # No longer expose ``repeat_counts`` or ``dst_rows`` – the CSR metadata
        # supersedes them.


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
    # Pre-compute row2slice mapping for efficient scatter operations
    row2slice = torch.repeat_interleave(
        torch.arange(len(dims_list), device=data.device, dtype=torch.long),
        torch.tensor(lengths, device=data.device, dtype=torch.long)
    )

    return FlatMultiTensor(
        data=data,
        offsets=torch.tensor(offsets, device=data.device, dtype=torch.long),
        lengths=torch.tensor(lengths, device=data.device, dtype=torch.long),
        shapes=shapes,
        dims_list=dims_list,
        channel_dim=channel_dim,
        row2slice=row2slice,
    )


def unpack_flat(flat: FlatMultiTensor, multitensor_system):
    """Convert a ``FlatMultiTensor`` back into the nested list structure.

    Returns a new ``MultiTensor`` instance filled with cloned tensors.
    """
    nested = multitensor_system.make_multitensor(default=None)
    for idx, dims in enumerate(flat.dims_list):
        nested[dims] = flat.view(idx).clone()
    return nested 