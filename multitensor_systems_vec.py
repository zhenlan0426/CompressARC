import numpy as np
import torch
from typing import List, Tuple

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
    ):
        self.data = data
        self.offsets = offsets
        self.lengths = lengths
        self.shapes = shapes
        self.dims_list = dims_list
        self.channel_dim = channel_dim

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


def _generate_all_dims():
    """Yield every 5-bit mask and its integer index (0..31)."""
    for i in range(1 << NUM_DIMENSIONS):
        yield [(i >> bit) & 1 for bit in range(NUM_DIMENSIONS)]


def pack_multitensor(mt, multitensor_system, channel_dim: int) -> FlatMultiTensor:
    """Convert the nested ``MultiTensor`` *mt* into a ``FlatMultiTensor``.

    Args
    -----
    mt : multitensor_systems.MultiTensor
        The original nested structure we want to flatten.
    multitensor_system : multitensor_systems.MultiTensorSystem
        Source system – gives us valid dimension combos and shape helper.
    channel_dim : int
        Uniform channel width C for *all* slices.

    Returns
    -------
    FlatMultiTensor
        Flattened buffer + metadata.
    """

    offsets: List[int] = []
    lengths: List[int] = []
    shapes: List[List[int]] = []
    dims_list: List[Tuple[int, ...]] = []

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
        running_total += num_pos

    total_positions = running_total
    
    # Handle empty case
    if total_positions == 0:
        # Create empty tensors with default device/dtype
        data = torch.zeros((0, channel_dim), dtype=torch.float32)
        return FlatMultiTensor(
            data=data,
            offsets=torch.tensor([], dtype=torch.long),
            lengths=torch.tensor([], dtype=torch.long),
            shapes=[],
            dims_list=[],
            channel_dim=channel_dim,
        )
    
    # Allocate big buffer and copy data
    # Get device and dtype from first tensor
    first_dims = next(iter(multitensor_system))
    first_tensor = mt[first_dims]
    device = next(first_tensor.parameters(), torch.tensor([])).device if hasattr(first_tensor, 'parameters') else first_tensor.device
    dtype = first_tensor.dtype
    data = torch.zeros((total_positions, channel_dim), dtype=dtype, device=device)

    for idx, dims in enumerate(multitensor_system):
        tensor = mt[dims]
        start = offsets[idx]
        length = lengths[idx]
        data[start : start + length].copy_(tensor.view(length, channel_dim))

    return FlatMultiTensor(
        data=data,
        offsets=torch.tensor(offsets, device=data.device, dtype=torch.long),
        lengths=torch.tensor(lengths, device=data.device, dtype=torch.long),
        shapes=shapes,
        dims_list=dims_list,
        channel_dim=channel_dim,
    )


def unpack_flat(flat: FlatMultiTensor, multitensor_system):
    """Convert a ``FlatMultiTensor`` back into the nested list structure.

    Returns a new ``MultiTensor`` instance filled with cloned tensors.
    """
    nested = multitensor_system.make_multitensor(default=None)
    for idx, dims in enumerate(flat.dims_list):
        nested[dims] = flat.view(idx).clone()
    return nested 