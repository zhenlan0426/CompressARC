import time
import os

import numpy as np
import torch

import preprocessing
# Use the batched ARCCompressor implementation which prepends a batch dimension
import arc_compressor_batch as arc_compressor
import solution_selection_batch as solution_selection
from async_memory_manager import AsyncMemoryManager, optimizer_to
from checkpoint_utils import save_checkpoint, load_checkpoint

from utils_batch import compute_grid_size_log_partition, compute_grid_logprob

"""
This file trains a model for every ARC-AGI task in a split.
"""

np.random.seed(0)
torch.manual_seed(0)


def get_optimal_batch_size(task):
    grid_size = task.n_examples * task.n_colors * task.n_x * task.n_y
    if grid_size < 5000:
        return 16
    elif grid_size < 10000:
        return 8
    elif grid_size < 20000:
        return 4
    else:
        return 2


def take_step(task, model, optimizer_task, optimizer_shared, train_step):
    """
    Runs a forward pass of the model on the ARC-AGI task.
    Args:
        task (Task): The ARC-AGI task containing the problem.
        model (ArcCompressor): The VAE decoder model to run the forward pass with.
        optimizer (torch.optim.Optimizer): The optimizer used to take the step on the model weights.
        train_step (int): The training iteration number.
        train_history_logger (Logger): A logger object used for logging the forward pass outputs
                of the model, as well as accuracy and other things.
    """
    # TODO: revisit coefficient schedule for amortization / grid size uncertainty
    optimizer_task.zero_grad()
    optimizer_shared.zero_grad()
    logits, x_mask, y_mask, KL_amounts, KL_names, = model.forward()
    # Shape now: (B, E, C, X, Y, in_out)
    # Pre-append a zero-logit channel for the black colour along the colour dimension (dim=2)
    logits = torch.cat([torch.zeros_like(logits[:, :, :1, ...]), logits], dim=2)

    B = logits.shape[0]
    per_sample_KL = torch.zeros(B, device=logits.device)
    for KL_amount in KL_amounts:
        per_sample_KL += KL_amount.sum(dim=list(range(1, KL_amount.ndim)))
    total_KL = per_sample_KL.mean()

    reconstruction_error_per_sample = torch.zeros(B, device=logits.device)
    test_reconstruction_error = 0

    # ------------------------------------------------------------
    # Fully vectorised pre-computation over examples *and* in/out modes.
    # ------------------------------------------------------------
    small_coeff = 1.0

    # Bring the length dimension to the end so the helper treats everything
    # before it as vectorisable.
    x_grid_log_partitions = compute_grid_size_log_partition(x_mask.transpose(2, 3), small_coeff)  # (B, E, 2)
    y_grid_log_partitions = compute_grid_size_log_partition(y_mask.transpose(2, 3), small_coeff)  # (B, E, 2)

    for example_num in range(task.n_examples):  # sum over examples
        for in_out_mode in range(2):  # sum over in/out grid per example
            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            coeff_mask = 1.0

            logits_slice = logits[:, example_num, :, :, :, in_out_mode]  # (B, C, x, y)
            problem_slice = task.problem[example_num, :, :, in_out_mode]  # (x, y)

            x_mask_slice = x_mask[:, example_num, :, in_out_mode]
            y_mask_slice = y_mask[:, example_num, :, in_out_mode]

            precomp_x = x_grid_log_partitions[:, example_num, in_out_mode]
            precomp_y = y_grid_log_partitions[:, example_num, in_out_mode]
                        
            if example_num >= task.n_train and in_out_mode == 1: # test set no backward pass
                with torch.no_grad():
                    logprob = compute_grid_logprob(logits_slice, problem_slice, x_mask_slice, y_mask_slice, grid_size_uncertain, coeff_mask, coeff_mask, precomp_x, precomp_y)
                    test_reconstruction_error -= logprob.sum()
            else:
                logprob = compute_grid_logprob(logits_slice, problem_slice, x_mask_slice, y_mask_slice, grid_size_uncertain, coeff_mask, coeff_mask, precomp_x, precomp_y)
                reconstruction_error_per_sample = reconstruction_error_per_sample - logprob

    reconstruction_error = reconstruction_error_per_sample.mean()
    scalar_loss = total_KL + 10 * reconstruction_error
    scalar_loss.backward()
    optimizer_task.step()
    optimizer_shared.step()

    # Return scalar metrics for tracking if desired
    return scalar_loss.item(), test_reconstruction_error.item()

if __name__ == "__main__":
    start_time = time.time()
    ######################## hyperparameters for training ##################################
    task_nums = list(range(1000))
    split = "training"  # "training", "evaluation, or "test"
    only_same_size_tasks = False  # Set to True to only run for tasks where task.in_out_same_size or task.all_out_same_size
    # burn_in = 100
    # track_freq = 10
    n_epochs = 10
    
    # Checkpoint options
    resume_from_checkpoint = None  # Set to checkpoint path to resume training
    # resume_from_checkpoint = "run_results/2025-07-30_09-34-08/checkpoint"  # Set to checkpoint path to resume training
    continue_training = True  # If True, save everything needed to continue training (shared + task params/optimizers)
    ########################################################################################

    # Preprocess all tasks, make models, optimizers, and loggers. Make plots.
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    if only_same_size_tasks:
        tasks = [task for task in tasks if task.in_out_same_size or task.all_out_same_size]
    models = []  # One ARCCompressor per task
    task_optimizers = []  # Optimisers that handle ONLY task-specific parameters
    shared_optimizer = None  # Created once, handles shared weights

    for i, task in enumerate(tasks):
        batch_size = get_optimal_batch_size(task)
        if i == 0:
            # First task – create brand-new weights on GPU initially
            model = arc_compressor.ARCCompressor(task, batch_size, device='cuda')
            # Shared optimiser – only ever created once, keeps reference to shared_params
            shared_optimizer = torch.optim.Adam(model.shared_params, lr=0.007)
            
        else:
            # Subsequent tasks – reuse shared weights from the first model, init on CPU
            model = arc_compressor.ARCCompressor(task, batch_size, shared_model=models[0], device='cpu')        
        models.append(model)
        # Optimiser dedicated to task-specific latents (starts on CPU)
        task_opt = torch.optim.Adam(model.task_params, lr=0.007)
        task_optimizers.append(task_opt)

    # load checkpoint to continue training
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_epoch = load_checkpoint(resume_from_checkpoint, models, shared_optimizer, task_optimizers, continue_training=continue_training)
        # Delete the old checkpoint after successful loading to save disk space
        os.remove(resume_from_checkpoint)
        print(f"Deleted old checkpoint: {resume_from_checkpoint}")
    
    # Initialize async memory manager with pinned memory for faster transfers
    memory_mgr = AsyncMemoryManager(use_pinned_memory=True)
    print(f"AsyncMemoryManager initialized with pinned memory support")
    
    for epoch in range(start_epoch, start_epoch+n_epochs):
        order = list(range(len(tasks)))
        np.random.shuffle(order)
        
        epoch_loss = 0.0
        epoch_test_recon = 0.0
        
        # Pre-load first task to GPU using memory manager for pinned memory benefits
        if len(order) > 0:
            first_idx = order[0]
            memory_mgr.move_to_gpu_async(models[first_idx], task_optimizers[first_idx])
            memory_mgr.wait_for_gpu_ready()  # Ensure first task is ready before training loop
        
        # Main training loop with async memory management
        for i, idx in enumerate(order):
            task = tasks[idx]
            model = models[idx]
            task_optimizer = task_optimizers[idx]
            
            # Prepare next task asynchronously (if exists)
            next_gpu_future = None
            if i + 1 < len(order):
                next_idx = order[i + 1]
                next_gpu_future = memory_mgr.move_to_gpu_async(
                    models[next_idx], task_optimizers[next_idx]
                )
            
            # Core training step (current task already on GPU)
            scalar_loss, test_recon = take_step(task, model, task_optimizer, shared_optimizer, epoch)
            epoch_loss += scalar_loss
            epoch_test_recon += test_recon
            
            # Start moving current task to CPU (async)
            memory_mgr.move_to_cpu_async(model, task_optimizer)
            
            # Ensure next task is ready before continuing
            memory_mgr.wait_for_gpu_ready(next_gpu_future)
        
        # Wait for all background operations to complete
        memory_mgr.wait_all_complete()
        
        # Print epoch results
        avg_loss = epoch_loss / len(tasks)
        avg_test = epoch_test_recon / len(tasks)
        print(f"Epoch {epoch+1}/{start_epoch+n_epochs} - avg_loss={avg_loss:.4f} avg_testRecon={avg_test:.4f}")

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = f"run_results/{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    
    # save final checkpoint
    checkpoint_path = os.path.join(dir_path, "checkpoint")
    save_checkpoint(models, shared_optimizer, task_optimizers, epoch, checkpoint_path, continue_training=continue_training)

    # Save training summary
    end_time = time.time()
    total_time = end_time - start_time
    summary_path = os.path.join(dir_path, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Total training time: {total_time / 60 / 60:.4f} hours\n")
        f.write(f"Last epoch average loss: {avg_loss:.4f}\n")
        f.write(f"Last epoch average test loss: {avg_test:.4f}\n")
        f.write(f"Pinned memory optimizers: {len(task_optimizers)}/{len(task_optimizers)} (pinned memory enabled)\n")
        f.write(f"Number of tasks: {len(tasks)}\n")
        f.write(f"Number of epochs: {n_epochs}\n")
