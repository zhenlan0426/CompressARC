import matplotlib.pyplot as plt
import numpy as np
import torch


color_list = np.array([
    [0, 0, 0],
    [30, 147, 255],
    [249, 60, 49],
    [79, 204, 48],
    [255, 220, 0],
    [153, 153, 153],
    [229, 58, 163],
    [255, 133, 27],
    [135, 216, 241],
    [146, 18, 49],
])

def convert_color(grid):  # grid dims must end in c
    return np.clip(np.matmul(grid, color_list), 0, 255).astype(np.uint8)

def plot_problem(logger):
    n_train = logger.task.n_train
    n_test = logger.task.n_test
    n_examples = logger.task.n_examples
    n_x = logger.task.n_x
    n_y = logger.task.n_y
    pixels = 128+np.zeros([n_train+n_test, n_x, 2, n_y, 3], dtype=np.uint8)
    for example_num in range(n_examples):
        if example_num < n_train:
            subsplit = 'train'
            subsplit_example_num = example_num
        else:
            subsplit = 'test'
            subsplit_example_num = example_num - n_train
        for mode_num, mode in enumerate(('input', 'output')):
            if subsplit == 'test' and mode == 'output':
                continue
            grid = np.array(logger.task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
            grid = convert_color(grid)  # x, y, c
            pixels[example_num,:grid.shape[0],mode_num,:grid.shape[1],:] = grid
    pixels = pixels.reshape([(n_train+n_test)*n_x, 2*n_y, 3])
    
    fig, ax = plt.subplots()
    ax.imshow(pixels)
    for example_num in range(n_examples+1):
        ax.plot((-0.5, 2*n_y-0.5), (n_x*example_num-0.5, n_x*example_num-0.5), color=(0.5,0.5,0.5))
    for divider_num in range(3):
        ax.plot((n_y*divider_num-0.5, n_y*divider_num-0.5), (-0.5, n_x*n_examples-0.5), color=(0.5,0.5,0.5))
    plt.savefig('plots/' + logger.task.task_name + '_problem.pdf', bbox_inches='tight')
    plt.close()

def plot_solution(logger):
    n_train = logger.task.n_train
    n_test = logger.task.n_test
    n_examples = logger.task.n_examples
    n_x = logger.task.n_x
    n_y = logger.task.n_y
    n_plotted_solutions = 5
    pixels = 128+np.zeros([n_test, n_x, n_plotted_solutions, n_y, 3], dtype=np.uint8)
    for example_num in range(n_train, n_examples):
        subsplit = 'test'
        subsplit_example_num = example_num - n_train

        for solution_num, solution in enumerate([
            logger.current_solution,
            logger.ema_logits_solution,
            logger.ema_probabilities_solution,
            logger.solution_most_frequent,
            logger.solution_second_most_frequent,
            ]):

            grid = np.array(solution[subsplit_example_num])  # x, y
            grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c

            grid = convert_color(grid)  # x, y, c
            pixels[subsplit_example_num,:grid.shape[0],solution_num,:grid.shape[1],:] = grid
    pixels = pixels.reshape([n_test*n_x, n_plotted_solutions*n_y, 3])
    
    fig, ax = plt.subplots()
    ax.imshow(pixels)
    for example_num in range(n_test+1):
        ax.plot((-0.5, n_plotted_solutions*n_y-0.5), (n_x*example_num-0.5, n_x*example_num-0.5), color=(0.5,0.5,0.5))
    for divider_num in range(n_plotted_solutions+1):
        ax.plot((n_y*divider_num-0.5, n_y*divider_num-0.5), (-0.5, n_x*n_test-0.5), color=(0.5,0.5,0.5))
    plt.savefig('plots/' + logger.task.task_name + '_solutions.pdf', bbox_inches='tight')
    plt.close()
