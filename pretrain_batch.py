import time

import numpy as np
import torch

import preprocessing
# Use the batched ARCCompressor implementation which prepends a batch dimension
import arc_compressor_batch as arc_compressor
import solution_selection_batch as solution_selection

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

def take_step(task, model, optimizer_task, optimizer_shared, train_step, track_last=True):
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
    small_coeff = 0.01 ** max(0, 1 - train_step / 100)

    # Bring the length dimension to the end so the helper treats everything
    # before it as vectorisable.
    x_grid_log_partitions = compute_grid_size_log_partition(x_mask.transpose(2, 3), small_coeff)  # (B, E, 2)
    y_grid_log_partitions = compute_grid_size_log_partition(y_mask.transpose(2, 3), small_coeff)  # (B, E, 2)

    for example_num in range(task.n_examples):  # sum over examples
        for in_out_mode in range(2):  # sum over in/out grid per example
            if example_num >= task.n_train and in_out_mode == 1:
                if not track_last:
                    continue

            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            coeff_mask = 0.01 ** max(0, 1 - train_step / 100) if grid_size_uncertain else 1.0

            logits_slice = logits[:, example_num, :, :, :, in_out_mode]  # (B, C, x, y)
            problem_slice = task.problem[example_num, :, :, in_out_mode]  # (x, y)

            x_mask_slice = x_mask[:, example_num, :, in_out_mode]
            y_mask_slice = y_mask[:, example_num, :, in_out_mode]

            precomp_x = x_grid_log_partitions[:, example_num, in_out_mode]
            precomp_y = y_grid_log_partitions[:, example_num, in_out_mode]

            logprob = compute_grid_logprob(logits_slice, problem_slice, x_mask_slice, y_mask_slice, grid_size_uncertain, coeff_mask, coeff_mask, precomp_x, precomp_y)

            if example_num >= task.n_train and in_out_mode == 1:
                test_reconstruction_error -= logprob.sum()
            else:
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
    split = "evaluation"  # "training", "evaluation, or "test"
    only_same_size_tasks = False  # Set to True to only run for tasks where task.in_out_same_size or task.all_out_same_size
    burn_in = 100
    track_freq = 10
    n_epochs = 500
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
            # First task – create brand-new weights that will be shared.
            model = arc_compressor.ARCCompressor(task, batch_size, batch_weights=False)
            # Shared optimiser – only ever created once, keeps reference to shared_params
            shared_optimizer = torch.optim.Adam(model.shared_params, lr=0.01, betas=(0.5, 0.9))
        else:
            # Subsequent tasks – reuse shared weights from the first model
            model = arc_compressor.ARCCompressor(task, batch_size, shared_model=models[0], batch_weights=False)
        models.append(model)
        # Optimiser dedicated to task-specific latents
        task_opt = torch.optim.Adam(model.task_params, lr=0.01, betas=(0.5, 0.9))
        task_optimizers.append(task_opt)

        # visualization.plot_problem(train_history_logger)
        # train_history_loggers.append(train_history_logger)

    task_stats = []

    # Get the solution hashes so that we can check for correctness
    true_solution_hashes = [task.solution_hash for task in tasks]
    def optimizer_to(optimizer, device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    import random
    for epoch in range(n_epochs):
        order = list(range(len(tasks)))
        random.shuffle(order)

        epoch_loss = 0.0
        epoch_test_recon = 0.0

        for idx in order:
            task = tasks[idx]
            model = models[idx]
            task_optimizer = task_optimizers[idx]

            # ------ Move task resources to GPU ------
            model.to_task_cuda()
            optimizer_to(task_optimizer, 'cuda')

            scalar_loss, test_recon = take_step(task, model, task_optimizer, shared_optimizer, epoch)
            epoch_loss += scalar_loss
            epoch_test_recon += test_recon

            # ------ Off-load task back to CPU ------
            model.to_task_cpu()
            optimizer_to(task_optimizer, 'cpu')

        avg_loss = epoch_loss / len(tasks)
        avg_test = epoch_test_recon / len(tasks)
        print(f"Epoch {epoch+1}/{n_epochs} - avg_loss={avg_loss:.4f} avg_testRecon={avg_test:.4f}")

    import os
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = f"run_results/{timestamp}"
    os.makedirs(dir_path, exist_ok=True)

    # Save shared weights
    torch.save(models[0].shared_params, os.path.join(dir_path, 'shared_weights.pt'))

    # Save task-specific weights
    for i, model in enumerate(models):
        task_id = tasks[i].task_name
        task_specific_weights_path = os.path.join(dir_path, f"task_{task_id}_weights.pt")
        torch.save(model.task_params, task_specific_weights_path)

    # Save training summary
    end_time = time.time()
    total_time = end_time - start_time
    summary_path = os.path.join(dir_path, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Total training time: {total_time / 60 / 60:.4f} hours\n")
        f.write(f"Last epoch average loss: {avg_loss:.4f}\n")
        f.write(f"Last epoch average test loss: {avg_test:.4f}\n")
