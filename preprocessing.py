import numpy as np
import torch


class Task():
    def __init__(self, problem, solution):
        self.n_train = len(problem['train'])
        self.n_test = len(problem['test'])
        self.n_examples = self.n_train + self.n_test

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
        for example_num in range(self.n_train+n_test-1):
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
            for example_num in range(self.n_train, self.n_examples):
                self.shapes[example_num][1] = [30, 30]
        
    def construct_multitensor_system(self):
        # figure out the size of the tensors we'll process in our neural network: colors, x, y
        self.n_x = 0
        self.n_y = 0
        for example_num in range(self.n_examples):
            self.n_x = max(self.n_x, self.shapes[example_num][0][0], self.shapes[example_num][1][0])
            self.n_y = max(self.n_y, self.shapes[example_num][0][1], self.shapes[example_num][1][1])
        colors = {}
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
        self.multitensor_system = MultiTensorSystem(self.n_examples, self.n_colors, self.n_x, self.n_y, self)

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
                        grid = [[[1 if self.colors.index(color)==ref_color
                                  for color in row]
                                 for row in grid]
                                for ref_color in range(self.n_colors+1)]  # color, x, y
                        grid = np.array(grid)  # color, x, y
                    mode_num = 0 if mode=='input' else 1
                    self.problem[new_example_num,:,:grid.shape[0],:grid.shape[1],mode_num] = grid

        self.problem = np.argmax(self.problem, axis=1)  # example, x, y, in/out. Black is included.
        self.problem = torch.from_numpy(self.problem).to(torch.get_default_device())  # example, x, y, in/out

    def create_solution_tensor(self, solution):
        # load the data into a tensor self.solution (note: separate from solution!!) for crossentropy evaluation
        self.solution = np.zeros((self.n_test, self.n_colors+1, self.n_x, self.n_y))  # example, color, x, y
        for example_num in range(n_test):
            grid = solution[example_num]  # x, y
            grid = [[[1 if self.colors.index(color)==ref_color
                      for color in row]
                     for row in grid]
                    for ref_color in range(self.n_colors+1)]  # color, x, y
            grid = np.array(grid)  # color, x, y
            self.solution[self.n_train+example_num,:,:grid.shape[0],:grid.shape[1]] = grid
        self.solution = np.argmax(self.solution, axis=1)  # example, x, y. Black is included.
        self.solution = torch.from_numpy(self.solution).to(torch.get_default_device())  # example, x, y

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
        self.masks = torch.from_numpy(masks).to(torch.get_default_dtype()).to(torch.get_default_device())


def preprocess_tasks(split, task_nums):
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

    problem_names = list(problems.values())

    tasks = []
    for problem_name in problem_names:
        problem = problems[problem_name]
        solution = None if solutions is None elsesolutions[problem_name]
        task = Task(problem, solution)
        tasks.append(task)

    return tasks
