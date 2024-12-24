import numpy as np
import torch


import preprocessing
import arc_compressor
import initializers
import tensor_algebra
import layers


def take_step(task, model, optimizer, train_step, train_history_logger):

    logits, x_mask, y_mask, KL_amounts, KL_names = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)  # add black

    total_KL = 0
    for KL_amount in KL_amounts:
        total_KL = total_KL + KL_amount

    reconstruction_error = 0
    for example_num in range(task.n_examples):
        for in_out_mode in range(2):
            if example_num >= task.n_train and in_out_mode == 1:
                continue
            logits_slice = logits[example_num,:,:,:,in_out_mode]  # color, x, y
            problem_slice = task.problem[example_num,:,:,in_out_mode]  # x, y
            output_shape = task.shapes[example_num][in_out_mode]
            x_logprobs = []
            for x_offset in range(x_mask.shape[1]-output_shape[0]+1):
                logprob = -torch.sum(x_mask[example_num,:x_offset,in_out_mode])
                logprob = logprob + torch.sum(x_mask[example_num,x_offset:x_offset+output_shape[0],in_out_mode])
                logprob = logprob - torch.sum(x_mask[example_num,x_offset+output_shape[0]:,in_out_mode])
                x_logprobs.append(logprob)
            for y_offset in range(y_mask.shape[1]-output_shape[1]+1):
                logprob = -torch.sum(y_mask[example_num,:y_offset,in_out_mode])
                logprob = logprob + torch.sum(y_mask[example_num,y_offset:y_offset+output_shape[1],in_out_mode])
                logprob = logprob - torch.sum(y_mask[example_num,y_offset+output_shape[1]:,in_out_mode])
                y_logprobs.append(logprob)
            x_logprobs = torch.stack(x_logprobs, dim=0)
            y_logprobs = torch.stack(y_logprobs, dim=0)
            x_log_partition = torch.logsumexp(x_logprobs)
            y_log_partition = torch.logsumexp(y_logprobs)
            logprobs = [[] for x_offset in range(x_logprobs.shape[0])]  # x, y
            for x_offset in range(x_logprobs.shape[0]):
                for y_offset in range(y_logprobs.shape[0]):
                    logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition
                    logits_crop = logits_slice[:,x_offset:x_offset+output_shape[0],y_offset:y_offset+output_shape[1]]  # c, x, y
                    target_crop = problem_slice[:output_shape[0],:output_shape[1]]  # x, y
                    logprob = logprob - torch.nn.functional.crossentropy(logits_crop, target_crop, reduction='sum')
                    logprobs[x_offset].append(logprob)
            logprobs = torch.stack([torch.stack(logprobs_, dim=0) for logprobs_ in logprobs], dim=0)  # x, y
            logprob = torch.logsumexp(logprobs)
            reconstruction_error = reconstruction_error - logprob

    loss = total_KL + 10*reconstruction_error
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Performance recording
    train_history_logger.log(train_step, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, reconstruction_error, loss)


if __name__ == "__main__":
    task_nums = list(range(100))
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
        train_history_loggers.append(train_history_logger)

    for i, (task, model, optimizer, train_step, train_history_logger) in enumerate(zip(tasks, models, optimizers, train_steps, train_history_loggers)):
        for _ in range(500):
            take_step(task, model, optimizer, train_step, train_history_logger)
            train_steps[i] = train_step + 1
