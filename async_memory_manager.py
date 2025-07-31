import torch


def optimizer_to(optimizer, device, non_blocking=False):
    """Move optimizer state to the specified device (CPU or CUDA).

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose internal state will be migrated.
        device (str): Destination device, e.g. "cpu" or "cuda".
        non_blocking (bool): If True the copy may run asynchronously w.r.t. the host.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=non_blocking)


def pin_optimizer_memory(optimizer):
    """Pin CPU memory for optimizer state to accelerate CPU->GPU transfers.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose state tensors will be pinned.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.device.type == "cpu":
                state[k] = v.pin_memory()


class AsyncMemoryManager:
    """Overlap CPU↔GPU transfers with compute via a dedicated CUDA stream.

    This simplified version drops ThreadPoolExecutor because launching the
    non-blocking `.to(device)` operations is already fast (copies execute on
    the GPU’s DMA engines).  Keeping the same public method names avoids
    changes to training scripts that already rely on them.
    """

    def __init__(self, use_pinned_memory=True):
        # High-priority stream exclusively for parameter transfers
        self.transfer_stream = torch.cuda.Stream(priority=0)
        self.use_pinned_memory = use_pinned_memory
        self._pinned_optimizers = set()  # Track which optimizers have been pinned

    # ------------------------------------------------------------------
    # Public API (compatible with previous thread-based implementation)
    # ------------------------------------------------------------------
    def move_to_gpu_async(self, model, optimizer):
        """Schedule the model & optimizer to be moved to CUDA.

        Parameters are copied with `non_blocking=True` so the call returns
        immediately; the actual DMA happens on ``self.transfer_stream``.
        Returns ``None`` for backward compatibility (previously a Future).
        """
        if model is None or optimizer is None:
            return None

        # Ensure optimizer memory is pinned for faster transfers
        if self.use_pinned_memory and id(optimizer) not in self._pinned_optimizers:
            pin_optimizer_memory(optimizer)
            self._pinned_optimizers.add(id(optimizer))

        with torch.cuda.stream(self.transfer_stream):
            model.to_task_cuda(non_blocking=True)
            optimizer_to(optimizer, "cuda", non_blocking=True)
        return None

    def move_to_cpu_async(self, model, optimizer):
        """Move model & optimizer back to CPU (blocking operation).
        
        Note: After moving to CPU, optimizer memory will be automatically
        pinned on the next GPU transfer for faster subsequent transfers.
        """
        if model is None or optimizer is None:
            return None
        model.to_task_cpu()
        optimizer_to(optimizer, "cpu")
        
        # Pin optimizer memory immediately after CPU transfer for next GPU move
        if self.use_pinned_memory:
            pin_optimizer_memory(optimizer)
            self._pinned_optimizers.add(id(optimizer))
        
        return None

    def wait_for_gpu_ready(self, _future=None):
        """Block until all transfers on the dedicated stream are finished."""
        self.transfer_stream.synchronize()

    def cleanup_completed(self):  # retained for API compatibility
        pass

    def wait_all_complete(self):
        self.transfer_stream.synchronize()

    def shutdown(self):  # retained for API compatibility
        pass

    # ------------------------------------------------------------------
    # Pinned memory management methods
    # ------------------------------------------------------------------
    def enable_pinned_memory(self):
        """Enable pinned memory for faster transfers."""
        self.use_pinned_memory = True

    def disable_pinned_memory(self):
        """Disable pinned memory (useful for debugging or memory-constrained systems)."""
        self.use_pinned_memory = False

    def get_pinned_optimizer_count(self):
        """Return the number of optimizers currently using pinned memory."""
        return len(self._pinned_optimizers)

    def clear_pinned_tracking(self):
        """Clear the tracking of pinned optimizers (useful for cleanup)."""
        self._pinned_optimizers.clear()
