import time

import numpy as np
import torch

import preprocessing
# Use the batched ARCCompressor implementation which prepends a batch dimension
import arc_compressor_batch as arc_compressor
import solution_selection_batch as solution_selection
# import bayesian_logger_batch as bl
import solution_selection_batch as solution_selection

from utils_batch import compute_grid_size_log_partition, compute_grid_logprob
from checkpoint_utils import load_checkpoint
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

def take_step(task, model, optimizer_task, optimizer_shared, train_step, train_history_logger: solution_selection.Logger):
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
            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            coeff_mask = 0.01 ** max(0, 1 - train_step / 100) if grid_size_uncertain else 1.0

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
    optimizer_task.zero_grad()
    optimizer_shared.zero_grad()

    # Performance recording
    train_history_logger.log(train_step,
                             logits,
                             x_mask,
                             y_mask,
                             KL_amounts,
                             KL_names,
                             total_KL,
                             reconstruction_error,
                             scalar_loss,
                             test_reconstruction_error)


if __name__ == "__main__":
    start_time = time.time()
    ######################## hyperparameters for training ##################################
    task_nums = list(range(1000))
    split = "evaluation"  # "training", "evaluation, or "test"
    only_same_size_tasks = False  # Set to True to only run for tasks where task.in_out_same_size or task.all_out_same_size
    n_iterations = 500
    lr = 0.01
    shared_lr_factor = 0.1
    checkpoint = "run_results/2025-07-30_16-14-19/shared_params"
    ########################################################################################

    # Preprocess all tasks
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    if only_same_size_tasks:
        tasks = [task for task in tasks if task.in_out_same_size or task.all_out_same_size]

    task_stats = []

    # Get the solution hashes so that we can check for correctness
    true_solution_hashes = [task.solution_hash for task in tasks]
    checkpoint = torch.load(checkpoint, map_location='cuda')
    # Train the models one by one, creating them on the fly
    for i, task in enumerate(tasks):        
        # Create model, optimizer, and logger for this task
        batch_size = get_optimal_batch_size(task)
        model = arc_compressor.ARCCompressor(task, batch_size, device='cuda')
        optimizer_task = torch.optim.Adam(model.task_params, lr=lr, betas=(0.5, 0.9))
        optimizer_shared = torch.optim.Adam(model.shared_params, lr=lr*shared_lr_factor, betas=(0.5, 0.9))
        train_history_logger = solution_selection.Logger(task)
        load_checkpoint(checkpoint, model, None, None, continue_training=False)

        task_start_time = time.time()
        for train_step in range(n_iterations):
            take_step(task, model, optimizer_task, optimizer_shared, train_step, train_history_logger)

        time_spent = time.time() - task_start_time

        # Compute the best and second best solutions
        # train_history_logger.finalize_solutions()
        stats = train_history_logger.compute_stats()
        stats['task_num'] = task.task_name
        stats['time_spent'] = time_spent
        stats['n_iterations'] = n_iterations
        task_stats.append(stats)


    import os
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = f"run_results/{timestamp}"
    os.makedirs(dir_path, exist_ok=True)

    # Save task stats
    import csv
    import pickle  # Added for saving complex data structures
    keys = task_stats[0].keys()
    with open(f'{dir_path}/task_stats.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(task_stats)

    # Write down how long it all took
    with open(f'{dir_path}/timing_result.txt', 'w') as f:
        f.write("Time elapsed in seconds: " + str(time.time() - start_time))

    fixed_stats = [stat for stat in task_stats if stat['is_shape_fixed']]
    different_stats = [stat for stat in task_stats if not stat['is_shape_fixed']]
    total_stats = task_stats

    summary_data = []

    def compute_averages(stats, category):
        if not stats:
            return None
        n = len(stats)
        avgs = {
            'category': category,
            'num_tasks': n,
        }
        columns = ['avg_total_loss', 'last_total_loss', 'last_test_recon', 'top1_match_pct', 'top2_match_pct', 'time_spent']
        for col in columns:
            avgs[f'avg_{col}'] = sum(stat[col] for stat in stats) / n
        avgs['total_time_spent'] = sum(stat['time_spent'] for stat in stats)
        return avgs

    for group, cat in [(fixed_stats, 'fixed'), (different_stats, 'different'), (total_stats, 'total')]:
        avg = compute_averages(group, cat)
        if avg:
            summary_data.append(avg)

    if summary_data:
        keys = summary_data[0].keys()
        with open(f'{dir_path}/summary.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_data)

    # ------------------------------------------------------------------
    # Save Bayesian samples and unique solutions for all tasks in ONE file
    # Structure: {task_name: {'samples': ..., 'unique_solutions': ..., 'unique_index_solutions': ...}, ...}
    # all_bayesian_data = {}
    # for task, logger in zip(tasks, train_history_loggers):
    #     all_bayesian_data[task.task_name] = {
    #         'samples': logger.get_samples(),
    #         'unique_solutions': logger.get_unique_solutions(),
    #         'unique_index_solutions': logger.get_unique_index_solutions(),
    #     }

    # with open(f'{dir_path}/bayesian_data.pkl', 'wb') as f:
    #     pickle.dump(all_bayesian_data, f)
