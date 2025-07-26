import time

import numpy as np
import torch

import preprocessing
# Use the batched ARCCompressor implementation which prepends a batch dimension
import arc_compressor_batch as arc_compressor
import solution_selection_batch as solution_selection
import bayesian_logger_batch as bl


"""
This file trains a model for every ARC-AGI task in a split.
"""

np.random.seed(0)
torch.manual_seed(0)


def mask_select_logprobs(mask, length):
    """
    Compute the (unnormalised) log-probabilities for selecting every contiguous slice of
    the specified length, **vectorised over the batch dimension**.

    Args:
        mask (Tensor): Tensor of shape (B, L) where B is the batch size and L the
                       maximum possible length. Larger (more positive) entries mean the
                       corresponding index is *less* likely to be masked out.
        length (int):  Desired slice length.

    Returns:
        Tensor: log_partition of shape (B,) – log-partition-function for each batch element.
        Tensor: logprobs      of shape (B, L-length+1) – unnormalised log-probability for
                choosing a slice starting at every possible offset.
    """

    # mask: (B, L)
    B, L = mask.shape

    logprobs = []
    for offset in range(L - length + 1):
        before_sum = mask[:, :offset].sum(dim=1)                    # (B,)
        middle_sum = mask[:, offset:offset + length].sum(dim=1)     # (B,)
        after_sum  = mask[:, offset + length:].sum(dim=1)           # (B,)
        logprobs.append(-before_sum + middle_sum - after_sum)       # (B,)

    logprobs = torch.stack(logprobs, dim=1)                         # (B, O)
    log_partition = torch.logsumexp(logprobs, dim=1)                # (B,)
    return log_partition, logprobs

def get_optimal_batch_size(task):
    grid_size = task.n_examples * task.n_colors * task.n_x * task.n_y
    if grid_size < 5000:
        return 16
    elif grid_size < 10000:
        return 8
    elif grid_size < 20000:
        return 4
    else:
        return 1

def take_step(task, model, optimizer, train_step, train_history_logger: bl.BayesianLogger, track_last=False):
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
    optimizer.zero_grad()
    logits, x_mask, y_mask, KL_amounts, KL_names, = model.forward()
    # Shape now: (B, E, C, X, Y, in_out)
    # Pre-append a zero-logit channel for the black colour along the colour dimension (dim=2)
    logits = torch.cat([torch.zeros_like(logits[:, :, :1, ...]), logits], dim=2)

    B = logits.shape[0]
    per_sample_KL = torch.zeros(B, device=logits.device)
    for KL_amount in KL_amounts:
        per_sample_KL += KL_amount.sum(dim=list(range(1, KL_amount.ndim)))
    total_KL = per_sample_KL.sum()

    reconstruction_error_per_sample = torch.zeros(B, device=logits.device)
    test_reconstruction_error = 0

    for example_num in range(task.n_examples):  # sum over examples
        for in_out_mode in range(2):  # sum over in/out grid per example
            if example_num >= task.n_train and in_out_mode == 1:
                if not track_last:
                    continue

            # Determine whether the grid size is already known.
            # If not, there is an extra term in the reconstruction error, corresponding to
            # the probability of reconstructing the correct grid size.
            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            if grid_size_uncertain:
                coefficient = 0.01**max(0, 1-train_step/100)
            else:
                coefficient = 1
            logits_slice = logits[:, example_num, :, :, :, in_out_mode]  # (B, C, x, y)
            problem_slice = task.problem[example_num, :, :, in_out_mode]  # (x, y)
            output_shape = task.shapes[example_num][in_out_mode]
            x_log_partition, x_logprobs = mask_select_logprobs(
                coefficient * x_mask[:, example_num, :, in_out_mode],
                output_shape[0]
            )  # (B,), (B, O_x)
            y_log_partition, y_logprobs = mask_select_logprobs(
                coefficient * y_mask[:, example_num, :, in_out_mode],
                output_shape[1]
            )  # (B,), (B, O_y)
            # Account for probability of getting right grid size, if grid size is not known
            if grid_size_uncertain:
                x_log_partitions = []
                y_log_partitions = []
                for length in range(1, x_mask.shape[2] + 1):
                    # this is log-sum-exp over grids of the given length
                    x_log_partitions.append(
                        mask_select_logprobs(coefficient * x_mask[:, example_num, :, in_out_mode], length)[0]
                    )
                for length in range(1, y_mask.shape[2] + 1):
                    y_log_partitions.append(
                        mask_select_logprobs(coefficient * y_mask[:, example_num, :, in_out_mode], length)[0]
                    )
                # this is log sum exp over all possible lengths
                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

            # log P(correct colors) = log sum over all possible starts P(grid that starts at x_offset, y_offset) * P(correct colors given grid)
            # log sum exp log P(above) = log sum exp (log P1 + log P2)
            # this two loops calculate log P1 + log P2
            logprobs = []  # will become (B, O_x, O_y)
            B = logits_slice.shape[0]
            for x_offset in range(x_logprobs.shape[1]):  # iterate over possible x-starts
                logprobs_y = []
                for y_offset in range(y_logprobs.shape[1]):  # iterate over possible y-starts
                    # Grid-position prior
                    logprob = (
                        x_logprobs[:, x_offset] - x_log_partition +
                        y_logprobs[:, y_offset] - y_log_partition
                    )  # (B,)

                    # Extract the corresponding crop
                    logits_crop = logits_slice[:, :, x_offset:x_offset + output_shape[0], y_offset:y_offset + output_shape[1]]  # (B, C, x', y')
                    target_crop = problem_slice[:output_shape[0], :output_shape[1]]  # (x', y')

                    # Broadcast target over the batch dimension
                    target_crop_b = target_crop.unsqueeze(0).expand(B, *target_crop.shape)

                    ce = torch.nn.functional.cross_entropy(logits_crop, target_crop_b, reduction='none')  # (B, x', y')
                    ce_sum = ce.sum(dim=(1, 2))  # (B,)

                    logprob = logprob - ce_sum  # (B,)
                    logprobs_y.append(logprob)
                logprobs_y = torch.stack(logprobs_y, dim=1)  # (B, O_y)
                logprobs.append(logprobs_y)
            logprobs = torch.stack(logprobs, dim=1)  # (B, O_x, O_y)
            if grid_size_uncertain:
                coefficient = 0.1**max(0, 1-train_step/100)
            else:
                coefficient = 1
            logprob = torch.logsumexp(coefficient * logprobs, dim=(1, 2)) / coefficient  # (B,)

            if example_num >= task.n_train and in_out_mode == 1:
                test_reconstruction_error += -logprob.sum()
            else:
                reconstruction_error_per_sample = reconstruction_error_per_sample - logprob

    reconstruction_error = reconstruction_error_per_sample.sum()
    scalar_loss = total_KL + 10 * reconstruction_error
    scalar_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Performance recording
    train_history_logger.log(train_step,
                             logits,
                             x_mask,
                             y_mask,
                             KL_amounts,
                             KL_names,
                             total_KL,
                             reconstruction_error,
                             per_sample_KL + 10 * reconstruction_error_per_sample,
                             test_reconstruction_error if track_last else None)


if __name__ == "__main__":
    start_time = time.time()
    ######################## hyperparameters for training ##################################
    task_nums = list(range(1))
    split = "evaluation"  # "training", "evaluation, or "test"
    only_same_size_tasks = False  # Set to True to only run for tasks where task.in_out_same_size or task.all_out_same_size
    burn_in = 100
    track_freq = 10
    n_iterations = 500
    ########################################################################################

    # Preprocess all tasks, make models, optimizers, and loggers. Make plots.
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    if only_same_size_tasks:
        tasks = [task for task in tasks if task.in_out_same_size or task.all_out_same_size]
    models = []
    optimizers = []
    train_history_loggers = []
    for task in tasks:
        batch_size = get_optimal_batch_size(task)
        model = arc_compressor.ARCCompressor(task, batch_size)
        models.append(model)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        optimizers.append(optimizer)

        train_history_logger = bl.BayesianLogger(task, burn_in_steps=burn_in, track_frequency=track_freq)
        # visualization.plot_problem(train_history_logger)
        train_history_loggers.append(train_history_logger)

    task_stats = []

    # Get the solution hashes so that we can check for correctness
    true_solution_hashes = [task.solution_hash for task in tasks]

    # Train the models one by one
    for i, (task, model, optimizer, train_history_logger) in enumerate(zip(tasks, models, optimizers, train_history_loggers)):
        task_start_time = time.time()

        for train_step in range(n_iterations):
            track_last = (train_step == n_iterations - 1)
            take_step(task, model, optimizer, train_step, train_history_logger, track_last=track_last)

        time_spent = time.time() - task_start_time

        # visualization.plot_solution(train_history_logger)

        stats = train_history_logger.compute_stats()
        stats['task_num'] = task.task_name
        stats['time_spent'] = time_spent
        stats['n_iterations'] = n_iterations
        task_stats.append(stats)

        # solution_selection.save_predictions(train_history_loggers[:i+1])
        # solution_selection.plot_accuracy(true_solution_hashes)

    # Save final states for all tasks
    # solution_selection.save_final_states(train_history_loggers, 'final_states.npz')

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
    all_bayesian_data = {}
    for task, logger in zip(tasks, train_history_loggers):
        all_bayesian_data[task.task_name] = {
            'samples': logger.get_samples(),
            'unique_solutions': logger.get_unique_solutions(),
            'unique_index_solutions': logger.get_unique_index_solutions(),
        }

    with open(f'{dir_path}/bayesian_data.pkl', 'wb') as f:
        pickle.dump(all_bayesian_data, f)
