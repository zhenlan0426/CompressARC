import json

import numpy as np
import torch

import tensor_algebra


class Task():
    def __init__(self, task_name, problem, solution):
        self.task_name = task_name
        self.n_train = len(problem['train'])
        self.n_test = len(problem['test'])
        self.n_examples = self.n_train + self.n_test

        self.unprocessed_problem = problem
        self.collect_problem_shapes(problem)
        self.predict_solution_shapes(problem)
        self.construct_multitensor_system(problem)
        self.compute_mask()
        self.create_problem_tensor(problem)
        if solution is not None:
            self.create_solution_tensor(solution)
        else:
            self.solution = None

    def collect_problem_shapes(self, problem):
        # get all the shapes of all the problem grids
        self.shapes = []
        for example_num in range(self.n_train):
            in_shape = list(np.array(problem['train'][example_num]['input']).shape)
            out_shape = list(np.array(problem['train'][example_num]['output']).shape)
            self.shapes.append([in_shape, out_shape])
        for example_num in range(self.n_test):
            in_shape = list(np.array(problem['test'][example_num]['input']).shape)
            self.shapes.append([in_shape, None])

    def predict_solution_shapes(self, problem):
        # try to predict the solution size based on problem sizes
        self.in_out_same_size = True
        self.all_in_same_size = True
        self.all_out_same_size = True
        for example_num in range(self.n_train):
            if tuple(self.shapes[example_num][0]) != tuple(self.shapes[example_num][1]):
                self.in_out_same_size = False
        for example_num in range(self.n_train+self.n_test-1):
            if tuple(self.shapes[example_num][0]) != tuple(self.shapes[example_num-1][0]):
                self.all_in_same_size = False
        for example_num in range(self.n_train-1):
            if tuple(self.shapes[example_num][1]) != tuple(self.shapes[example_num+1][1]):
                self.all_out_same_size = False
        if self.in_out_same_size:
            for example_num in range(self.n_train, self.n_examples):
                self.shapes[example_num][1] = self.shapes[example_num][0]
        elif self.all_out_same_size:
            for example_num in range(self.n_examples-1):
                self.shapes[example_num+1][1] = self.shapes[example_num][1]
        else:
            max_n_x = 0
            max_n_y = 0
            for example_num in range(self.n_examples):
                for in_out_mode in range(2):
                    if example_num >= self.n_train and in_out_mode == 1:
                        continue
                    max_n_x = max(max_n_x, self.shapes[example_num][in_out_mode][0])
                    max_n_y = max(max_n_y, self.shapes[example_num][in_out_mode][1])
            for example_num in range(self.n_train, self.n_examples):
                self.shapes[example_num][1] = [max_n_x, max_n_y]
        
    def construct_multitensor_system(self, problem):
        # figure out the size of the tensors we'll process in our neural network: colors, x, y
        self.n_x = 0
        self.n_y = 0
        for example_num in range(self.n_examples):
            self.n_x = max(self.n_x, self.shapes[example_num][0][0], self.shapes[example_num][1][0])
            self.n_y = max(self.n_y, self.shapes[example_num][0][1], self.shapes[example_num][1][1])
        colors = set()
        for subsplit_name, n_examples in (('train', self.n_train), ('test', self.n_test)):
            for example_num in range(n_examples):
                for mode in ('input', 'output'):
                    if subsplit_name == 'test' and mode == 'output':
                        continue
                    else:
                        for row in problem[subsplit_name][example_num][mode]:
                            for color in row:
                                colors.add(color)
        colors.add(0)  # we'll treat black differently than other colors, as it is often used as a background color
        self.colors = list(sorted(colors))
        self.n_colors = len(self.colors)-1  # don't count black
        self.multitensor_system = tensor_algebra.MultiTensorSystem(self.n_examples, self.n_colors, self.n_x, self.n_y, self)

    def create_problem_tensor(self, problem):
        # load the data into a tensor self.problem (note: separate from problem!!) for crossentropy evaluation
        self.problem = np.zeros([self.n_examples, self.n_colors+1, self.n_x, self.n_y, 2])  # example, color, x, y, in/out. Black is included.
        for subsplit_name, n_examples in (('train', self.n_train), ('test', self.n_test)):
            for example_num in range(n_examples):
                for mode in ('input', 'output'):
                    new_example_num = example_num if subsplit_name=='train' else self.n_train+example_num
                    if subsplit_name == 'test' and mode == 'output':
                        grid = np.zeros([self.n_colors+1] + self.shapes[new_example_num][1])  # color, x, y
                    else:
                        grid = problem[subsplit_name][example_num][mode]  # x, y
                        grid = [[[1 if self.colors.index(color)==ref_color else 0
                                  for color in row]
                                 for row in grid]
                                for ref_color in range(self.n_colors+1)]  # color, x, y
                        grid = np.array(grid)  # color, x, y
                    mode_num = 0 if mode=='input' else 1
                    self.problem[new_example_num,:,:grid.shape[1],:grid.shape[2],mode_num] = grid

        self.problem = np.argmax(self.problem, axis=1)  # example, x, y, in/out. Black is included.
        self.problem = torch.from_numpy(self.problem).to(torch.get_default_device())  # example, x, y, in/out

    def create_solution_tensor(self, solution):
        # load the data into a tensor self.solution (note: separate from solution!!) for crossentropy evaluation
        self.solution = np.zeros((self.n_test, self.n_colors+1, self.n_x, self.n_y))  # example, color, x, y
        self.solution_tuple = ()
        for example_num in range(self.n_test):
            grid = solution[example_num]  # x, y
            self.solution_tuple = self.solution_tuple + (tuple(tuple(row) for row in grid),)
            grid = [[[1 if self.colors.index(color)==ref_color else 0
                      for color in row]
                     for row in grid]
                    for ref_color in range(self.n_colors+1)]  # color, x, y
            grid = np.array(grid)  # color, x, y
            # unfortunately sometimes the solution tensor will be bigger than (n_x, n_y), and in these cases
            # we'll never get the solution.
            min_x = min(grid.shape[1], self.n_x)
            min_y = min(grid.shape[2], self.n_y)
            self.solution[example_num,:,:min_x,:min_y] = grid[:,:min_x,:min_y]
        self.solution = np.argmax(self.solution, axis=1)  # example, x, y. Black is included.
        self.solution = torch.from_numpy(self.solution).to(torch.get_default_device())  # example, x, y
        self.solution_hash = hash(self.solution_tuple)

    def compute_mask(self):
        # compute a mask to zero out activations and crossentropies
        self.masks = np.zeros([self.n_examples, self.n_x, self.n_y, 2])  # example, x, y, in/out.
        for example_num in range(self.n_examples):
            for mode in ('input', 'output'):
                mode_num = 0 if mode=='input' else 1
                x_mask = (np.arange(self.n_x) < self.shapes[example_num][mode_num][0])
                y_mask = (np.arange(self.n_y) < self.shapes[example_num][mode_num][1])
                mask = x_mask[:,None]*y_mask[None,:]
                self.masks[example_num,:,:,mode_num] = mask
        self.masks = torch.from_numpy(self.masks).to(torch.get_default_dtype()).to(torch.get_default_device())


def preprocess_tasks(split, task_nums_or_task_names):
    """
    split: one of "training", "evaluation", and "test"
    """
    with open('dataset/arc-agi_' + split + '_challenges.json', 'r') as f:
        problems = json.load(f)
    if split != "test":
        with open('dataset/arc-agi_' + split + '_solutions.json', 'r') as f:
            solutions = json.load(f)
    else:
        solutions = None

    task_names = list(problems.keys())

    tasks = []
    for task_num, task_name in enumerate(task_names):
        if task_num in task_nums_or_task_names or task_name in task_nums_or_task_names:
            problem = problems[task_name]
            solution = None if solutions is None else solutions[task_name]
            task = Task(task_name, problem, solution)
            tasks.append(task)

    return tasks
