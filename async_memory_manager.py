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


class AsyncMemoryManager:
    """Overlap CPU↔GPU transfers with compute via a dedicated CUDA stream.

    This simplified version drops ThreadPoolExecutor because launching the
    non-blocking `.to(device)` operations is already fast (copies execute on
    the GPU’s DMA engines).  Keeping the same public method names avoids
    changes to training scripts that already rely on them.
    """

    def __init__(self):
        # High-priority stream exclusively for parameter transfers
        self.transfer_stream = torch.cuda.Stream(priority=0)

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

        with torch.cuda.stream(self.transfer_stream):
            model.to_task_cuda(non_blocking=True)
            optimizer_to(optimizer, "cuda", non_blocking=True)
        return None

    def move_to_cpu_async(self, model, optimizer):
        """Move model & optimizer back to CPU (blocking operation)."""
        if model is None or optimizer is None:
            return None
        model.to_task_cpu()
        optimizer_to(optimizer, "cpu")
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
