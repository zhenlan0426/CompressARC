import time

import numpy as np
import torch

import preprocessing
import arc_compressor
import initializers
import tensor_algebra
import layers
import solution_selection
import visualization


def mask_select_logprobs(mask, length):
    logprobs = []
    for offset in range(mask.shape[0]-length+1):
        logprob = -torch.sum(mask[:offset])
        logprob = logprob + torch.sum(mask[offset:offset+length])
        logprob = logprob - torch.sum(mask[offset+length:])
        logprobs.append(logprob)
    logprobs = torch.stack(logprobs, dim=0)
    log_partition = torch.logsumexp(logprobs, dim=0)
    return log_partition, logprobs

def take_step(task, model, optimizer, train_step, train_history_logger):

    optimizer.zero_grad()
#    with torch.no_grad():            ####################################  test
#        noise_norm = torch.sqrt(torch.mean(model.head_weights[0]**2))
#        noise = noise_norm*torch.randn_like(noise_norm)
#        model.head_weights[0].data = 0.99*model_head_weights[0] + 0.01*noise
    logits, x_mask, y_mask, KL_amounts, KL_names, kernel_KL_amounts, kernel_KL_names = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)  # add black

    total_KL = 0
#    for KL_amount in KL_amounts + kernel_KL_amounts:                ############################
#        total_KL = total_KL + torch.sum(KL_amount)
    for KL_amount in KL_amounts:
        total_KL = total_KL + torch.sum(KL_amount)
    for KL_amount in kernel_KL_amounts:
        total_KL = total_KL + 0.1*torch.sum(KL_amount)

    reconstruction_error = 0
    for example_num in range(task.n_examples):
        for in_out_mode in range(2):
            if example_num >= task.n_train and in_out_mode == 1:
                continue
            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            if grid_size_uncertain:
                coefficient = 0.01**max(0, 1-train_step/100)
            else:
                coefficient = 1
            logits_slice = logits[example_num,:,:,:,in_out_mode]  # color, x, y
            problem_slice = task.problem[example_num,:,:,in_out_mode]  # x, y
            output_shape = task.shapes[example_num][in_out_mode]
            x_log_partition, x_logprobs = mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], output_shape[0])
            y_log_partition, y_logprobs = mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], output_shape[1])
            # Account for probability of getting right grid size, if grid size is not known
            if grid_size_uncertain:
                x_log_partitions = []
                y_log_partitions = []
                for length in range(1, x_mask.shape[1]+1):
                    x_log_partitions.append(mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], length)[0])
                for length in range(1, y_mask.shape[1]+1):
                    y_log_partitions.append(mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], length)[0])
                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

            logprobs = [[] for x_offset in range(x_logprobs.shape[0])]  # x, y
            for x_offset in range(x_logprobs.shape[0]):
                for y_offset in range(y_logprobs.shape[0]):
                    logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition
                    logits_crop = logits_slice[:,x_offset:x_offset+output_shape[0],y_offset:y_offset+output_shape[1]]  # c, x, y
                    target_crop = problem_slice[:output_shape[0],:output_shape[1]]  # x, y
                    logprob = logprob - torch.nn.functional.cross_entropy(logits_crop[None,...], target_crop[None,...], reduction='sum')
                    logprobs[x_offset].append(logprob)
            logprobs = torch.stack([torch.stack(logprobs_, dim=0) for logprobs_ in logprobs], dim=0)  # x, y
            if grid_size_uncertain:
                coefficient = 0.1**max(0, 1-train_step/100)
            else:
                coefficient = 1
#            coefficient = 1
#            if grid_size_uncertain:                       ####################################################################
#                coefficient = 0.1**max(0, 1-train_step/200)
#            else:
#                coefficient = 1
            logprob = torch.logsumexp(coefficient*logprobs, dim=(0,1))/coefficient
            reconstruction_error = reconstruction_error - logprob

    loss = total_KL + 10*reconstruction_error
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Performance recording
    train_history_logger.log(train_step,
                             logits,
                             x_mask,
                             y_mask,
                             KL_amounts,
                             KL_names,
                             kernel_KL_amounts,
                             kernel_KL_names,
                             total_KL,
                             reconstruction_error,
                             loss)


if __name__ == "__main__":
    start_time = time.time()

    task_nums = list(range(400))[99:]
    split = "training"  # "training", "evaluation, or "test"
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    models = []
    optimizers = []
    train_steps = []
    train_history_loggers = []
    for task in tasks:
        model = arc_compressor.ARCCompressor(task)
        models.append(model)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        optimizers.append(optimizer)
        train_steps.append(0)
        train_history_logger = solution_selection.Logger(task)
#        visualization.plot_problem(train_history_logger)
        train_history_loggers.append(train_history_logger)

    for i, (task, model, optimizer, train_step, train_history_logger) in enumerate(zip(tasks, models, optimizers, train_steps, train_history_loggers)):
        print(task.task_name)
        n_iterations = 1500
        for _ in range(n_iterations):
            take_step(task, model, optimizer, train_steps[i], train_history_logger)
            train_steps[i] = train_steps[i] + 1
            if train_steps[i] % n_iterations == 0:
                visualization.plot_solution(train_history_logger)
                solution_selection.save_accuracy(train_history_loggers[:i+1])
                solution_selection.plot_accuracy()

    print("Time elapsed in seconds: " + str(time.time() - start_time))
