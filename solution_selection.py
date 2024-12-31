import matplotlib.pyplot as plt
import numpy as np
import torch


class Logger():

    decay = 0.97

    def __init__(self, task):
        self.task = task
        self.KL_curves = dict()
        self.total_KL_curve = []
        self.reconstruction_error_curve = []
        self.loss_curve = []

        n_test = self.task.n_test
        n_colors = self.task.n_colors
        n_x = self.task.n_x
        n_y = self.task.n_y
        self.current_logits = torch.zeros([n_test, n_colors+1, n_x, n_y])  # example, color, x, y
        self.current_probabilities = 1/(self.task.n_colors+1) + self.current_logits  # example, color, x, y
        self.current_x_mask = torch.zeros([self.task.n_test, self.task.n_x])  # example, x
        self.current_y_mask = torch.zeros([self.task.n_test, self.task.n_y])  # example, y

        self.ema_logits = torch.zeros_like(self.current_logits)  # example, color, x, y
        self.ema_probabilities = 1/(self.task.n_colors+1) + self.current_logits  # example, color, x, y
        self.ema_x_mask = torch.zeros([self.task.n_test, self.task.n_x])  # example, x
        self.ema_y_mask = torch.zeros([self.task.n_test, self.task.n_y])  # example, y

        self.ema_logits_solution = None
        self.ema_probabilities_solution = None

        self.solution_hashes_count = dict()
        self.solution_most_frequent = None
        self.solution_second_most_frequent = None

        self.mode_frequencies = []

        if self.task.solution is not None:
            self.solution_accuracies = []

    def log(self, train_step, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, reconstruction_error, loss):
        if train_step == 0:
            for KL_name in KL_names:
                self.KL_curves[KL_name] = []
        for KL_amount, KL_name in zip(KL_amounts, KL_names):
            self.KL_curves[KL_name].append(float(torch.sum(KL_amount.detach()).cpu().numpy()))
        self.total_KL_curve.append(float(total_KL.detach().cpu().numpy()))
        self.reconstruction_error_curve.append(float(reconstruction_error.detach().cpu().numpy()))
        self.loss_curve.append(float(loss.detach().cpu().numpy()))

        self.track_solution(logits.detach(), x_mask.detach(), y_mask.detach())

    def track_solution(self, logits, x_mask, y_mask):
#        self.current_logits = logits[self.task.n_train-1:-1,:,:,:,0]  # example, color, x, y    #################################################################
#        self.current_x_mask = x_mask[self.task.n_train-1:-1,:,1]  # example, x
#        self.current_y_mask = y_mask[self.task.n_train-1:-1,:,1]  # example, y
        self.current_logits = logits[self.task.n_train:,:,:,:,1]  # example, color, x, y
        self.current_x_mask = x_mask[self.task.n_train:,:,1]  # example, x
        self.current_y_mask = y_mask[self.task.n_train:,:,1]  # example, y
        self.current_probabilities = torch.softmax(self.current_logits, dim=1)  # example, color, x, y

        self.ema_logits = self.decay*self.ema_logits + (1-self.decay)*self.current_logits
        self.ema_probabilities = self.decay*self.ema_probabilities + (1-self.decay)*self.current_probabilities
        self.ema_x_mask = self.decay*self.ema_x_mask + (1-self.decay)*self.current_x_mask
        self.ema_y_mask = self.decay*self.ema_y_mask + (1-self.decay)*self.current_y_mask

        self.ema_logits_solution = self.postprocess_solution(self.ema_logits, self.ema_x_mask, self.ema_y_mask)
        self.ema_probabilities_solution = self.postprocess_solution(self.ema_probabilities, self.ema_x_mask, self.ema_y_mask)

        # Track the first and second most common solution
        self.current_solution = self.postprocess_solution(self.current_logits, self.current_x_mask, self.current_y_mask)
        hashed = hash(self.current_solution)
        if hashed in self.solution_hashes_count:
            self.solution_hashes_count[hashed] += 1
        else:
            self.solution_hashes_count[hashed] = 1
        if self.solution_most_frequent is None:
            self.solution_most_frequent = self.current_solution
        if self.solution_second_most_frequent is None:
            self.solution_second_most_frequent = self.current_solution
        if hashed != hash(self.solution_most_frequent):
            if self.solution_hashes_count[hashed] >= self.solution_hashes_count[hash(self.solution_second_most_frequent)]:
                self.solution_second_most_frequent = self.current_solution
                if self.solution_hashes_count[hashed] >= self.solution_hashes_count[hash(self.solution_most_frequent)]:
                    self.solution_second_most_frequent = self.solution_most_frequent
                    self.solution_most_frequent = self.current_solution

        self.solution_names = ['current', 'ema_logits', 'ema_probabilities', 'mode1', 'mode2']
        self.solutions = [
                self.current_solution,
                self.ema_logits_solution,
                self.ema_probabilities_solution,
                self.solution_most_frequent,
                self.solution_second_most_frequent
        ]
        self.solution_hashes = [hash(solution) for solution in self.solutions]
        self.mode_frequencies.append([self.solution_hashes_count[self.solution_hashes[3]], self.solution_hashes_count[self.solution_hashes[4]]])

        if self.task.solution is not None:
            self.solution_accuracies.append([int(self.task.solution_hash == solution_hash) for solution_hash in self.solution_hashes])
    
    def best_slice_point(self, mask, length):
        if self.task.in_out_same_size or self.task.all_out_same_size:
            search_lengths = [length]
        else:
            search_lengths = list(range(1, mask.shape[0]+1))
        max_logprob = None
        for length in search_lengths:
            logprobs = []
            for offset in range(mask.shape[0]-length+1):
                logprob = -torch.sum(mask[:offset])
                logprob = logprob + torch.sum(mask[offset:offset+length])
                logprob = logprob - torch.sum(mask[offset+length:])
                logprobs.append(logprob)
            logprobs = torch.stack(logprobs, dim=0)
            if max_logprob is None or max_logprob < torch.max(logprobs):
                max_logprob = torch.max(logprobs)
                best_slice_start = torch.argmax(logprobs)
                best_slice_end = best_slice_start + length
        return best_slice_start, best_slice_end
    def best_crop(self, prediction, x_mask, x_length, y_mask, y_length):
        x_start, x_end = self.best_slice_point(x_mask, x_length)
        y_start, y_end = self.best_slice_point(y_mask, y_length)
        return prediction[...,x_start:x_end,y_start:y_end]
    def postprocess_solution(self, prediction, x_mask, y_mask):  # prediction must be example, color, x, y
        colors = torch.argmax(prediction, dim=1)  # example, x, y
        solution_slices = []  # example, x, y
        for example_num in range(self.task.n_test):
            if self.task.in_out_same_size or self.task.all_out_same_size:
                x_length = self.task.shapes[self.task.n_train+example_num][1][0]
                y_length = self.task.shapes[self.task.n_train+example_num][1][1]
            else:
                x_length = None
                y_length = None
            solution_slice = self.best_crop(colors[example_num,:,:], x_mask[example_num,:], x_length, y_mask[example_num,:], y_length)  # x, y
            solution_slices.append(solution_slice.cpu().numpy().tolist())
        for example in solution_slices:
            for row in example:
                for i, val in enumerate(row):
                    row[i] = self.task.colors[val]
        solution_slices = tuple(tuple(tuple(row) for row in example) for example in solution_slices)
        return solution_slices

def save_accuracy(loggers, accuracy_curve_fname='accuracy_curve.npz'):
    n_iterations = len(loggers[0].solution_accuracies)
    n_solution_types = 5
    n_tasks = len(loggers)
    accuracy_table = np.zeros([n_iterations, n_tasks, n_solution_types])
    mode_frequency_table = np.zeros([n_iterations, n_tasks, 2])
    for iteration in range(n_iterations):
        for task_num in range(n_tasks):
            for solution_type in range(n_solution_types):
                if loggers[task_num].solution_accuracies[iteration][solution_type]:
                    accuracy_table[iteration,task_num,solution_type] = 1
            mode_frequency_table[iteration,task_num,0] = loggers[task_num].mode_frequencies[iteration][0]
            mode_frequency_table[iteration,task_num,1] = loggers[task_num].mode_frequencies[iteration][1]

    np.savez(accuracy_curve_fname, accuracy_table=accuracy_table, mode_frequency_table=mode_frequency_table)

def plot_accuracy(accuracy_curve_fname='accuracy_curve.npz'):
    stored_data = np.load(accuracy_curve_fname, allow_pickle=True)
    accuracy_table, mode_frequency_table = stored_data['accuracy_table'], stored_data['mode_frequency_table']
    n_iterations, n_tasks, n_solution_types = accuracy_table.shape
    fig, ax = plt.subplots()
    
    solution_pick_1 = np.where(mode_frequency_table[:,:,0]>1, accuracy_table[:,:,3], accuracy_table[:,:,1])  # iteration, task
    solution_pick_2 = np.where(mode_frequency_table[:,:,1]>1, accuracy_table[:,:,4], accuracy_table[:,:,2])  # iteration, task
    either_correct = 1-(1-solution_pick_1)*(1-solution_pick_2)  # iteration, task
    accuracy_curve = np.mean(either_correct, axis=1)  # iteration
    ax.plot(np.arange(n_iterations), accuracy_curve, 'k-')

    solution_pick_1 = np.where(mode_frequency_table[:,:,0]>1, accuracy_table[:,:,3], accuracy_table[:,:,2])  # iteration, task
    solution_pick_2 = np.where(mode_frequency_table[:,:,1]>1, accuracy_table[:,:,4], accuracy_table[:,:,1])  # iteration, task
    either_correct = 1-(1-solution_pick_1)*(1-solution_pick_2)  # iteration, task
    accuracy_curve = np.mean(either_correct, axis=1)  # iteration
    ax.plot(np.arange(n_iterations), accuracy_curve, 'r-')

    solution_pick_1 = np.where(mode_frequency_table[:,:,0]>5, accuracy_table[:,:,3], accuracy_table[:,:,1])  # iteration, task
    solution_pick_2 = np.where(mode_frequency_table[:,:,1]>5, accuracy_table[:,:,4], accuracy_table[:,:,2])  # iteration, task
    either_correct = 1-(1-solution_pick_1)*(1-solution_pick_2)  # iteration, task
    accuracy_curve = np.mean(either_correct, axis=1)  # iteration
    ax.plot(np.arange(n_iterations), accuracy_curve, 'g-')

    solution_pick_1 = np.where(mode_frequency_table[:,:,0]>5, accuracy_table[:,:,3], accuracy_table[:,:,2])  # iteration, task
    solution_pick_2 = np.where(mode_frequency_table[:,:,1]>5, accuracy_table[:,:,4], accuracy_table[:,:,1])  # iteration, task
    either_correct = 1-(1-solution_pick_1)*(1-solution_pick_2)  # iteration, task
    accuracy_curve = np.mean(either_correct, axis=1)  # iteration
    ax.plot(np.arange(n_iterations), accuracy_curve, 'b-')

    plt.savefig('accuracy_curve.pdf', bbox_inches='tight')
    plt.close()

