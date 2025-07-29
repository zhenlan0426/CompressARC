import torch
from concurrent.futures import ThreadPoolExecutor


def optimizer_to(optimizer, device):
    """Move optimizer state to specified device"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


class AsyncMemoryManager:
    """Handles asynchronous GPU/CPU memory transfers for training"""
    
    def __init__(self, max_workers=5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="memory_mgmt")
        self.pending_operations = []
    
    def move_to_gpu_async(self, model, optimizer):
        """Start moving model and optimizer to GPU asynchronously"""
        def _move():
            try:
                model.to_task_cuda()
                optimizer_to(optimizer, 'cuda')
                return model, optimizer
            except Exception as e:
                print(f"Error moving to GPU: {e}")
                raise
        
        return self.executor.submit(_move)
    
    def move_to_cpu_async(self, model, optimizer):
        """Start moving model and optimizer to CPU asynchronously"""
        def _move():
            try:
                model.to_task_cpu()
                optimizer_to(optimizer, 'cpu')
                return model, optimizer
            except Exception as e:
                print(f"Error moving to CPU: {e}")
                raise
        
        future = self.executor.submit(_move)
        self.pending_operations.append(future)
        return future
    
    def wait_for_gpu_ready(self, future):
        """Wait for async GPU transfer to complete"""
        if future is not None:
            try:
                future.result()
            except Exception as e:
                print(f"Error preparing task on GPU: {e}")
                raise
    
    def cleanup_completed(self):
        """Clean up completed background operations"""
        completed = [f for f in self.pending_operations if f.done()]
        for f in completed:
            try:
                f.result()
            except Exception as e:
                print(f"Error in background operation: {e}")
        self.pending_operations = [f for f in self.pending_operations if not f.done()]
    
    def wait_all_complete(self):
        """Wait for all pending operations to complete"""
        for future in self.pending_operations:
            try:
                future.result()
            except Exception as e:
                print(f"Error in final cleanup: {e}")
        self.pending_operations.clear()
    
    def shutdown(self):
        """Clean shutdown of the memory manager"""
        print("Shutting down memory management thread pool...")
        self.executor.shutdown(wait=True)