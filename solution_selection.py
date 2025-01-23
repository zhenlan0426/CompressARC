import matplotlib.pyplot as plt
import numpy as np
import torch


np.random.seed(0)
torch.manual_seed(0)


class Logger():

    ema_decay = 0.97

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
        self.current_x_mask = torch.zeros([self.task.n_test, self.task.n_x])  # example, x
        self.current_y_mask = torch.zeros([self.task.n_test, self.task.n_y])  # example, y

        self.ema_logits = torch.zeros_like(self.current_logits)  # example, color, x, y
        self.ema_x_mask = torch.zeros([self.task.n_test, self.task.n_x])  # example, x
        self.ema_y_mask = torch.zeros([self.task.n_test, self.task.n_y])  # example, y

        self.solution_hashes_count = dict()
        self.solution_most_frequent = None
        self.solution_second_most_frequent = None

        self.solution_contributions_log = []
        self.solution_picks_history = []

    def log(self,
            train_step,
            logits,
            x_mask,
            y_mask,
            KL_amounts,
            KL_names,
            total_KL,
            reconstruction_error,
            loss):
        if train_step == 0:
            for KL_name in KL_names:
                self.KL_curves[KL_name] = []
        for KL_amount, KL_name in zip(KL_amounts, KL_names):
            self.KL_curves[KL_name].append(float(torch.sum(KL_amount.detach()).cpu().numpy()))
        self.total_KL_curve.append(float(total_KL.detach().cpu().numpy()))
        self.reconstruction_error_curve.append(float(reconstruction_error.detach().cpu().numpy()))
        self.loss_curve.append(float(loss.detach().cpu().numpy()))

        self.track_solution(train_step, logits.detach(), x_mask.detach(), y_mask.detach())

    def track_solution(self, train_step, logits, x_mask, y_mask):
        self.current_logits = logits[self.task.n_train:,:,:,:,1]  # example, color, x, y
        self.current_x_mask = x_mask[self.task.n_train:,:,1]  # example, x
        self.current_y_mask = y_mask[self.task.n_train:,:,1]  # example, y

        self.ema_logits = self.ema_decay*self.ema_logits + (1-self.ema_decay)*self.current_logits
        self.ema_x_mask = self.ema_decay*self.ema_x_mask + (1-self.ema_decay)*self.current_x_mask
        self.ema_y_mask = self.ema_decay*self.ema_y_mask + (1-self.ema_decay)*self.current_y_mask

        # Track the first and second most common solution
        solution_contributions = []
        for logits, x_mask, y_mask in ((self.current_logits, self.current_x_mask, self.current_y_mask),
                                       (self.ema_logits, self.ema_x_mask, self.ema_y_mask)):
            solution, uncertainty = self.postprocess_solution(logits, x_mask, y_mask)
            hashed = hash(solution)
            score = -10*uncertainty
            if train_step < 150:
                score = score - 10
            if logits is self.ema_logits:
                score = score - 4

            solution_contributions.append((hashed, score))

            if hashed in self.solution_hashes_count:
                self.solution_hashes_count[hashed] = float(np.logaddexp(self.solution_hashes_count[hashed], score))
            else:
                self.solution_hashes_count[hashed] = float(score)
            if self.solution_most_frequent is None:
                self.solution_most_frequent = solution
            if self.solution_second_most_frequent is None:
                self.solution_second_most_frequent = solution
            if hashed != hash(self.solution_most_frequent):
                if self.solution_hashes_count[hashed] >= self.solution_hashes_count[hash(self.solution_second_most_frequent)]:
                    self.solution_second_most_frequent = solution
                    if self.solution_hashes_count[hashed] >= self.solution_hashes_count[hash(self.solution_most_frequent)]:
                        self.solution_second_most_frequent = self.solution_most_frequent
                        self.solution_most_frequent = solution

        self.solution_contributions_log.append(solution_contributions)


        self.solutions = [
                self.solution_most_frequent,
                self.solution_second_most_frequent
        ]
        self.solution_hashes = [hash(solution) for solution in self.solutions]
        self.solution_picks_history.append(self.solution_hashes)

    
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
        uncertainties = torch.logsumexp(prediction, dim=1) - torch.amax(prediction, dim=1)  # example, x, y
        solution_slices = []  # example, x, y
        uncertainty = []  # example
        for example_num in range(self.task.n_test):
            if self.task.in_out_same_size or self.task.all_out_same_size:
                x_length = self.task.shapes[self.task.n_train+example_num][1][0]
                y_length = self.task.shapes[self.task.n_train+example_num][1][1]
            else:
                x_length = None
                y_length = None
            solution_slice = self.best_crop(colors[example_num,:,:], x_mask[example_num,:], x_length, y_mask[example_num,:], y_length)  # x, y
            uncertainty_slice = self.best_crop(uncertainties[example_num,:,:], x_mask[example_num,:], x_length, y_mask[example_num,:], y_length)  # x, y
            solution_slices.append(solution_slice.cpu().numpy().tolist())
            uncertainty.append(float(np.mean(uncertainty_slice.cpu().numpy())))
        for example in solution_slices:
            for row in example:
                for i, val in enumerate(row):
                    row[i] = self.task.colors[val]
        solution_slices = tuple(tuple(tuple(row) for row in example) for example in solution_slices)
        uncertainty = np.mean(uncertainty)
        return solution_slices, uncertainty


def save_predictions(loggers, fname='predictions.npz'):
    solution_contribution_logs = [logger.solution_contributions_log for logger in loggers]
    solution_picks_histories = [logger.solution_picks_history for logger in loggers]
    np.savez(fname,
             solution_contribution_logs=solution_contribution_logs,
             solution_picks_histories=solution_picks_histories
    )

def plot_accuracy(true_solution_hashes, fname='predictions.npz'):
    stored_data = np.load(fname, allow_pickle=True)
    solution_contribution_logs = stored_data['solution_contribution_logs']
    solution_picks_histories = stored_data['solution_picks_histories']

    n_tasks = len(solution_contribution_logs)
    n_iterations = len(solution_contribution_logs[0])

    correct = np.zeros([n_tasks, n_iterations])
    
    for task_num in range(n_tasks):
        for step_num in range(n_iterations):
            if any([hash_==true_solution_hashes[task_num] for hash_ in solution_picks_histories[task_num][step_num]]):
                correct[task_num,step_num] = 1

    accuracy_curve = np.mean(correct, axis=0)

    fig, ax = plt.subplots()

    ax.plot(np.arange(n_iterations), accuracy_curve, 'k-')
    
    plt.savefig('accuracy_curve.pdf', bbox_inches='tight')
    plt.close()
