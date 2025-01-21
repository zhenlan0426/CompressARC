import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

import train
import preprocessing
import arc_compressor
import initializers
import tensor_algebra
import layers
import solution_selection
import visualization


if __name__ == "__main__":
    split = input('Enter which split you want to find the task in (training, evaluation, test): ')
    task_name = input('Enter which task you want to analyze (eg. 272f95fa): ')
    folder = task_name + '/'
    print('Performing a training run on task ' + task_name + ' and placing the results in ' + folder)

    os.makedirs(folder, exist_ok=True)

    task = preprocessing.preprocess_tasks(split, [task_name])[0]
    if False:

        model = arc_compressor.ARCCompressor(task)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        train_history_logger = solution_selection.Logger(task)
        visualization.plot_problem(train_history_logger)

        n_iterations = 1500
        for train_step in range(n_iterations):
            train.take_step(task, model, optimizer, train_step, train_history_logger)
            if (train_step+1) % 50 == 0:
                visualization.plot_solution(train_history_logger, fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.png')
                visualization.plot_solution(train_history_logger, fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.pdf')

        np.savez(folder + task_name + '_KL_curves.npz',
                 KL_curves={key:np.array(val) for key, val in train_history_logger.KL_curves.items()},
                 reconstruction_error_curve=np.array(train_history_logger.reconstruction_error_curve),
                 multiposteriors=model.multiposteriors,
                 global_capacity_adjustments=model.global_capacity_adjustments,
                 decode_weights=model.decode_weights,
                 solution_hashes_over_time=train_history_logger.solution_hashes_over_time,
                 true_solution_hash=task.solution_hash)

    stored_data = np.load(folder + task_name + '_KL_curves.npz', allow_pickle=True)
    KL_curves = stored_data['KL_curves'][()]
    reconstruction_error_curve = stored_data['reconstruction_error_curve']
    multiposteriors = stored_data['multiposteriors'][()]
    global_capacity_adjustments = stored_data['global_capacity_adjustments'][()]
    decode_weights = stored_data['decode_weights'][()]
#    solution_hashes_over_time = stored_data['solution_hashes_over_time'][()]
#    true_solution_hash = stored_data['true_solution_hash'][()]


    
    fig, ax = plt.subplots()
    for component_name, curve in KL_curves.items():
        if tuple(eval(component_name)) == (1,0,0,1,0):
            color = (1, 0, 0)
            label = '(example, x, channel)'
        elif tuple(eval(component_name)) == (1,0,0,0,1):
            color = (0, 1, 0)
            label = '(example, y, channel)'
        elif tuple(eval(component_name)) == (0,1,1,0,0):
            color = (0, 0.5, 1)
            label = '(color, direction, channel)'
        elif tuple(eval(component_name)) == (0,1,0,0,0):
            color = (0.5, 0, 1)
            label = '(color, channel)'
        else:
            color = (0.5, 0.5, 0.5)
            label = None
        ax.plot(np.arange(curve.shape[0]), curve, color=color, label=label)
    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('KL contribution')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(folder + task_name + '_KL_components.png', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    total_KL = 0
    for component_name, curve in KL_curves.items():
        total_KL = total_KL + curve
    fig, ax = plt.subplots()
    ax.plot(np.arange(total_KL.shape[0]), total_KL, label='KL from z', color='k')
    ax.plot(np.arange(reconstruction_error_curve.shape[0]), reconstruction_error_curve, label='reconstruction error', color='r')
    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('total KL or reconstruction error')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(folder + task_name + '_KL_vs_reconstruction.png', bbox_inches='tight')
    plt.close()


    samples = []
    for i in range(100):
        sample, KL_amounts, KL_names = layers.decode_latents(global_capacity_adjustments, decode_weights, multiposteriors)
        samples.append(sample)

    if task_name == '272f95fa':
        def average_samples(dims, *items):
            mean = torch.mean(torch.stack(items, dim=0), dim=0).detach().cpu().numpy()
            all_but_last_dim = tuple(range(len(mean.shape) - 1))
            mean = mean - np.mean(mean, axis=all_but_last_dim)
            return mean
        means = tensor_algebra.multify(average_samples)(*samples)
        example_x_mean = means[1,0,0,1,0]
        example_y_mean = means[1,0,0,0,1]
        color_direction_mean = means[0,1,1,0,0]
        color_mean = means[0,1,0,0,0]

        dims_to_plot = []
        for KL_amount, KL_name in zip(KL_amounts, KL_names):
            dims = tuple(eval(KL_name))
            if torch.sum(KL_amount).detach().cpu().numpy() > 1:
                dims_to_plot.append(dims)

        for dims in dims_to_plot:
            tensor = means[dims]

            orig_shape = tensor.shape
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)
            for component_num in range(3):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]
                if len(component.shape) == 1:
                    component = component[None,:]
                if len(component.shape) != 2:
                    continue
                fig, ax = plt.subplots()
                ax.imshow(component, cmap='gray', vmin=-1, vmax=1)
                axis_names = ['example', 'color', 'direction', 'height', 'width']
                tensor_name = '_'.join([axis_name for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                if sum(dims) == 2:
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                else:
                    x_dim = None
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                plt.ylabel(x_dim)
                plt.xlabel(y_dim)

                if x_dim is None:
                    ax.set_yticks([])
                    ax.set_xticks([], minor=True)
                if y_dim is None:
                    ax.set_xticks([])
                    ax.set_xticks([], minor=True)

                if x_dim == 'example':
                    ax.set_yticks(np.arange(task.n_examples))
                if y_dim == 'example':
                    ax.set_xticks(np.arange(task.n_examples))

                color_names = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'orange', 'light blue', 'brown']
                restricted_color_names = [color_names[i] for i in task.colors]
                restricted_color_codes = [tuple((visualization.color_list[i]/255).tolist()) for i in task.colors]
                if x_dim == 'color':
                    ax.set_yticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_yticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_yticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")
                if y_dim == 'color':
                    ax.set_xticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_xticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_xticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")

                direction_names = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"]
                if x_dim == 'direction':
                    ax.set_yticks(np.arange(8))
                    ax.set_yticklabels(direction_names)
                    ax.tick_params(axis='y', which='major', labelsize=22)
                if y_dim == 'direction':
                    ax.set_xticks(np.arange(8))
                    ax.set_xticklabels(direction_names)
                    ax.tick_params(axis='x', which='major', labelsize=22)

                ax.set_title('component ' + str(component_num) + ', strength = ' + str(float(strength)))
#                plt.savefig(folder + task_name + '_' + tensor_name + '_component_' + str(component_num) + '_strength_' + str(float(strength)) + '.png', bbox_inches='tight')
                plt.savefig(folder + task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png', bbox_inches='tight')
                plt.close()


    print('done')
