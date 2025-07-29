# Multi-Process Design for Shared Decoder `f`

## Overview

This document outlines a multi-process framework for training a shared decoder `f` (implemented in `ARCCompressor`) across ~1000 ARC-AGI tasks, while maintaining per-task latent codes `z`. The design uses Python's `multiprocessing` (mp.Pool with 4-8 workers) and supports GPU sharing via NVIDIA MPS. It follows a parameter-server architecture to ensure safe read/write access to the global `f` and its optimizer, tolerating minor staleness for efficiency. Per-task `z` is stored on CPU and loaded to GPU only when needed.

Key goals:
- Share one `f` and optimizer across all tasks for meta-learning.
- Use asynchronous updates to minimize overhead.
- Avoid disk I/O for syncing; rely on in-memory queues and shared memory.

## Components

### 1. Server Process
- **Role**: Owns and manages the global `f` (single `ARCCompressor` instance) and its optimizer (e.g., Adam on `f.weights_list`).
- **Responsibilities**:
  - Initializes global `f` from `initializers.py`.
  - Listens on a `multiprocessing.Queue` for gradient updates from workers.
  - Applies received gradients to global `f`, calls `optimizer.step()`, and clears gradients.
  - Optionally aggregates (e.g., averages) if multiple gradients arrive in quick succession.
  - Responds to worker requests for the latest `state_dict` via a response queue.
- **Resource Use**: Runs on the shared GPU (via MPS) but is mostly idle, applying updates asynchronously.

### 2. Worker Processes (4-8 via mp.Pool)
- **Role**: Handle training for assigned tasks or task batches.
- **Responsibilities**:
  - For each task: Load per-task `z` (multiposteriors) from CPU storage (e.g., shared `Manager.dict` or files) and its local optimizer.
  - Request latest global `f` state from server (via queue; load to a temporary local model on GPU).
  - Perform forward/backward passes (e.g., in `take_step`), accumulating gradients over 1-5 steps for `f` and `z`.
  - Update `z` locally using its per-task optimizer and save back to storage.
  - Send accumulated gradients for `f` (as a `state_dict` of tensors) to the server's queue.
- **Local Temp Model**: Workers use a short-lived `ARCCompressor` instance for computations, syncing state from/to server as needed.

### 3. Shared Elements
- **Global `f` and Optimizer**: Managed solely by the server to avoid races.
- **Per-Task `z`**: Stored in a shared `Manager.dict` (keyed by task ID) or per-task files on CPU. Not globally shared during updates—workers handle locally.
- **Queues**:
  - Request Queue: Workers send requests for fresh `f` state.
  - Update Queue: Workers send gradients to server.
  - All in-memory; tensors serialized if needed (e.g., via `torch.save` to bytes).
- **Multiprocessing Tools**: `mp.Manager` for shared dicts, `mp.Queue` for communication, `mp.Pool` for workers.

## Training Flow
1. **Initialization (Main Process)**:
   - Spawn server process with global `f` and optimizer.
   - Preprocess tasks, initialize per-task `z` if needed.
   - Create mp.Pool with 4-8 workers.

2. **Epoch Loop (Multi-Epoch Training)**:
   - Shuffle tasks and divide into batches.
   - Use pool.map to assign batches to workers.

3. **Per-Worker Task Processing**:
   - Load task data and `z`.
   - Request and load global `f` state.
   - Train: Forward/backward, accumulate grads, update `z` locally.
   - Send `f` grads to server.
   - Repeat for next task in batch.

4. **Server Update**:
   - Receive grads, apply to global `f`, step optimizer.
   - Send updated state if requested.

5. **End of Epoch**: Optional sync point (e.g., main process queries server for metrics).

## Handling Staleness and Asynchrony
- Workers may train on slightly stale `f` (e.g., if server updates mid-request), but with 4-8 workers, this is minimal and acceptable.
- Gradient accumulation reduces update frequency, lowering contention.
- If queues backlog, workers can skip sends or use timeouts.

## Pros and Cons
### Pros
- **Safe Sharing**: Central server prevents races on weights/optimizer state.
- **Efficient**: Async updates, in-memory communication, GPU sharing via MPS.
- **Scalable for Small Workers**: Low overhead for 4-8 processes.
- **Generalization**: Shared `f` learns across tasks, as per meta-learning goals.

### Cons
- **Overhead**: Queue communication adds minor latency (~ms per update); mitigate with larger accumulation.
- **Complexity**: Adds server process and queues; but integrates well with existing mp.Pool.
- **Staleness**: Tolerated, but monitor via logs (e.g., version counters).

## Codebase Impacts
- **`parallel_train.py`** or **`multiGPUs.py`**: Add server spawning, queue setup, pass to workers.
- **`train.py`** / **`solve_task.py`**: Modify loops for queue-based sync, grad sending, local z updates.
- **`arc_compressor.py`**: Helpers for grad extraction/loading state.
- **`initializers.py`**: Separate global f init from per-task z.
- Test with dummy tasks to verify.

This design enables efficient sharing while aligning with the project's VAE-like training (ELBO objective)." 