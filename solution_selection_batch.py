import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

class Logger:
    """
    This class contains functionalities relating to the recording of model outputs, postprocessing,
    selection of most frequently sampled/highest scoring solutions, accuracy computations, and more.
    """
    ema_decay = 0.97

    def __init__(self, task):
        self.task = task
        self.KL_curves = {}
        self.total_KL_curve = []
        self.reconstruction_error_curve = []
        self.loss_curve = []
        self.test_reconstruction_error_curve = []

        self.ema_logits = None
        self.ema_x_mask = None
        self.ema_y_mask = None

        self.solution_hashes_count = {}
        self.solution_most_frequent = None
        self.solution_second_most_frequent = None

        self.solution_contributions_log = []
        self.solution_picks_history = []

    def log(self, train_step, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, reconstruction_error, loss, test_reconstruction_error=None):
        """Logs training progress and tracks solutions from one forward pass."""
        if train_step == 0:
            self.KL_curves = {KL_name: [] for KL_name in KL_names}

        for KL_amount, KL_name in zip(KL_amounts, KL_names):
            kl_sum_per_batch = torch.sum(KL_amount.detach(), dim=tuple(range(1, KL_amount.dim())))
            avg_kl = torch.mean(kl_sum_per_batch).item()
            self.KL_curves[KL_name].append(avg_kl)

        B = logits.shape[0]
        self.total_KL_curve.append((total_KL.detach() / B).item())

        self.reconstruction_error_curve.append((reconstruction_error.detach() / B).item())
        self.loss_curve.append((loss.detach() / B).item())
        if test_reconstruction_error is not None:
            self.test_reconstruction_error_curve.append((test_reconstruction_error.detach() / B).item())

        self._track_solution(train_step, logits.detach(), x_mask.detach(), y_mask.detach())

    def _track_solution(self, train_step, logits, x_mask, y_mask):
        """Postprocess and score solutions and keep track of the top two solutions with highest scores."""
        test_logits = logits[:, self.task.n_train:, :, :, :, 1]  # (B, n_test, C+1, max_x, max_y)
        x_mask_test = x_mask[:, self.task.n_train:, :, 1]  # (B, n_test, max_x)
        y_mask_test = y_mask[:, self.task.n_train:, :, 1]  # (B, n_test, max_y)

        B = logits.shape[0]

        if self.ema_logits is None:
            self.ema_logits = torch.zeros_like(test_logits)
            self.ema_x_mask = torch.zeros_like(x_mask_test)
            self.ema_y_mask = torch.zeros_like(y_mask_test)

        self.ema_logits = self.ema_decay * self.ema_logits + (1 - self.ema_decay) * test_logits
        self.ema_x_mask = self.ema_decay * self.ema_x_mask + (1 - self.ema_decay) * x_mask_test
        self.ema_y_mask = self.ema_decay * self.ema_y_mask + (1 - self.ema_decay) * y_mask_test

        solution_contributions = []
        for b in range(B):
            for is_ema, score_adjust in [(False, 0), (True, -4)]:
                if is_ema:
                    pred = self.ema_logits[b]
                    xm = self.ema_x_mask[b]
                    ym = self.ema_y_mask[b]
                else:
                    pred = test_logits[b]
                    xm = x_mask_test[b]
                    ym = y_mask_test[b]

                solution, uncertainty = self._postprocess_solution(pred, xm, ym)
                hashed_solution = hash(solution)
                score = -10 * uncertainty
                if train_step < 150:
                    score = score - 10
                score = score + score_adjust

                solution_contributions.append((hashed_solution, score))

                self.solution_hashes_count[hashed_solution] = float(np.logaddexp(
                    self.solution_hashes_count.get(hashed_solution, -np.inf), score))

                self._update_most_frequent_solutions(hashed_solution, solution)

        self.solution_contributions_log.append(solution_contributions)
        self.solution_picks_history.append([hash(sol) for sol in [
            self.solution_most_frequent, self.solution_second_most_frequent]])

    def _update_most_frequent_solutions(self, hashed, solution):
        """Keeps track of the top two solutions with highest scores."""
        if self.solution_most_frequent is None:
            self.solution_most_frequent = solution
        if self.solution_second_most_frequent is None:
            self.solution_second_most_frequent = solution

        if hashed != hash(self.solution_most_frequent):
            if self.solution_hashes_count[hashed] >= self.solution_hashes_count.get(
                    hash(self.solution_second_most_frequent), -np.inf):
                self.solution_second_most_frequent = solution
                if self.solution_hashes_count[hashed] >= self.solution_hashes_count.get(
                        hash(self.solution_most_frequent), -np.inf):
                    self.solution_second_most_frequent = self.solution_most_frequent
                    self.solution_most_frequent = solution

    def best_crop(self, prediction, x_mask, x_length, y_mask, y_length):
        x_start, x_end = self._best_slice_point(x_mask, x_length)
        y_start, y_end = self._best_slice_point(y_mask, y_length)
        return prediction[..., x_start:x_end, y_start:y_end]

    def _best_slice_point(self, mask, length):
        if self.task.in_out_same_size or self.task.all_out_same_size:
            search_lengths = [length]
        else:
            search_lengths = list(range(1, mask.shape[0]+1))
        max_logprob, best_slice_start, best_slice_end = None, None, None

        for length in search_lengths:
            logprobs = torch.stack([
                -torch.sum(mask[:offset]) + torch.sum(mask[offset:offset + length]) - torch.sum(mask[offset + length:])
                for offset in range(mask.shape[0] - length + 1)
            ])
            if max_logprob is None or torch.max(logprobs) > max_logprob:
                max_logprob = torch.max(logprobs)
                best_slice_start = torch.argmax(logprobs).item()
                best_slice_end = best_slice_start + length

        return best_slice_start, best_slice_end

    def _postprocess_solution(self, prediction, x_mask, y_mask):  # prediction must be example, color, x, y
        """Postprocess a solution and compute some variables that are used to calculate the score."""
        colors = torch.argmax(prediction, dim=1)  # example, x, y
        uncertainties = torch.logsumexp(prediction, dim=1) - torch.amax(prediction, dim=1)  # example, x, y
        solution_slices, uncertainty_values = [], []  # example, x, y; example

        for example_num in range(self.task.n_test):
            x_length = None
            y_length = None
            if self.task.in_out_same_size or self.task.all_out_same_size:
                x_length = self.task.shapes[self.task.n_train+example_num][1][0]
                y_length = self.task.shapes[self.task.n_train+example_num][1][1]
            solution_slice = self.best_crop(colors[example_num],
                                            x_mask[example_num],
                                            x_length,
                                            y_mask[example_num],
                                            y_length)  # x, y
            uncertainty_slice = self.best_crop(uncertainties[example_num],
                                               x_mask[example_num],
                                               x_length,
                                               y_mask[example_num],
                                               y_length)  # x, y

            solution_slices.append(solution_slice.cpu().numpy().tolist())
            uncertainty_values.append(float(np.mean(uncertainty_slice.cpu().numpy())))

        for example in solution_slices:
            for row in example:
                for i, val in enumerate(row):
                    row[i] = self.task.colors[val]

        solution_slices = tuple(tuple(tuple(row) for row in example) for example in solution_slices)
        return solution_slices, np.mean(uncertainty_values)

    def compute_stats(self):
        import numpy as np

        avg_total_loss = np.mean([r + k for r, k in zip(self.reconstruction_error_curve, self.total_KL_curve)])

        last_total_loss = self.reconstruction_error_curve[-1] + self.total_KL_curve[-1]

        last_test_recon = self.test_reconstruction_error_curve[-1]

        is_shape_fixed = self.task.in_out_same_size or self.task.all_out_same_size

        def has_right_shape(solution):
            if solution is None:
                return False
            for ex in range(self.task.n_test):
                pred_grid = solution[ex]
                true_shape = self.task.true_test_shapes[ex]
                if len(pred_grid) != true_shape[0] or (len(pred_grid) > 0 and len(pred_grid[0]) != true_shape[1]):
                    return False
            return True

        top1_right_shape = has_right_shape(self.solution_most_frequent)
        top2_right_shape = has_right_shape(self.solution_second_most_frequent)

        def compute_match_pct(solution, right_shape):
            if not right_shape:
                return 0.0
            total_matches = 0
            total_cells = 0
            for ex in range(self.task.n_test):
                pred_grid = solution[ex]
                true_shape = self.task.true_test_shapes[ex]
                h, w = true_shape
                pred_array = np.array(pred_grid)
                # Get ground truth indices and convert to actual colors
                true_indices = self.task.solution[ex, :h, :w].cpu().numpy()
                true_colors = np.array(self.task.colors)[true_indices]
                # Both pred_array and true_colors now contain actual color values
                matches = np.sum(true_colors == pred_array)
                total_matches += matches
                total_cells += h * w
            return (total_matches / total_cells * 100) if total_cells > 0 else 0.0

        top1_match_pct = compute_match_pct(self.solution_most_frequent, top1_right_shape)
        top2_match_pct = compute_match_pct(self.solution_second_most_frequent, top2_right_shape)

        return {
            'avg_total_loss': float(avg_total_loss),
            'last_total_loss': float(last_total_loss),
            'last_test_recon': float(last_test_recon),
            'is_shape_fixed': is_shape_fixed,
            'top1_right_shape': top1_right_shape,
            'top2_right_shape': top2_right_shape,
            'top1_match_pct': top1_match_pct,
            'top2_match_pct': top2_match_pct,
        }


def save_predictions(loggers, fname='predictions.npz'):
    """Saves solution score contributions and history of chosen solutions."""
    np.savez(fname,
             solution_contribution_logs=[logger.solution_contributions_log for logger in loggers],
             solution_picks_histories=[logger.solution_picks_history for logger in loggers])


def plot_accuracy(true_solution_hashes, fname='predictions.npz'):
    """Plots accuracy curve over training iterations."""
    stored_data = np.load(fname, allow_pickle=True)
    solution_picks_histories = stored_data['solution_picks_histories']

    n_tasks = len(solution_picks_histories)
    n_iterations = len(solution_picks_histories[0])

    correct = np.array([[
        int(any(hash_ == true_solution_hashes[task_num] for hash_ in solution_pair))
        for solution_pair in task_history
    ] for task_num, task_history in enumerate(solution_picks_histories)])

    accuracy_curve = correct.mean(axis=0)

    plt.figure()
    plt.plot(np.arange(n_iterations), accuracy_curve, 'k-')
    plt.savefig('accuracy_curve.pdf', bbox_inches='tight')
    plt.close()

