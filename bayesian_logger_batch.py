import torch
import torch.nn.functional as F
from typing import List, Tuple, Any

from solution_selection_batch import Logger as _BaseLogger

from utils_batch import compute_grid_logprob, compute_grid_size_log_partition

# TODO: 1. length = self.task.shapes[self.task.n_train + ex][1][dim_index]. make sure it is infered rather than leaked from ground truth.
#       2. Inference - _make_solutions_batch - should be done in the same way as in train, i.e. P(size) * P(colors | size), difference is 
#          use argmax for inference, but sum for training over all possible sizes.
class BayesianLogger(_BaseLogger):
    """Extended Logger that collects posterior samples for Bayesian ensembling.

    Additional features over _BaseLogger:
    1. Burn-in: ignore the first *burn_in_steps* optimisation steps.
    2. Tracking frequency: keep only one snapshot every *track_frequency* steps.
    3. Storage:
       • *unique_solutions*: list of *actual* output grids Y (not hashes).
       • *samples*: flat list of tuples ``(elbo_loss_i, logits_i, x_mask_i, y_mask_i)``
         already detached & on CPU so they do not consume VRAM.

    The class keeps all analytics provided by the parent logger (curves, stats…)
    by calling ``super().log`` on every iteration. Extra sampling information is
    only gathered at the specified tracking steps.
    """

    def __init__(self, task: Any, *, burn_in_steps: int = 0, track_frequency: int = 1):
        super().__init__(task)
        self.burn_in_steps = max(int(burn_in_steps), 0)
        self.track_frequency = max(int(track_frequency), 1)

        # Bayesian-specific containers
        self.unique_solutions: List[Tuple] = []           # actual colour grids
        self.unique_index_solutions: List[List[torch.Tensor]] = []     # colour-index grids
        self._solution_hash_set = set()
        self.losses: List[float] = []
        self.logits_samples: List[torch.Tensor] = []
        self.x_mask_samples: List[torch.Tensor] = []
        self.y_mask_samples: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Public API: override parent log
    # ------------------------------------------------------------------
    @torch.no_grad()
    def log(self,
            train_step: int,
            logits: torch.Tensor,
            x_mask: torch.Tensor,
            y_mask: torch.Tensor,
            KL_amounts,
            KL_names,
            total_KL,
            reconstruction_error,
            loss: torch.Tensor,
            test_reconstruction_error=None):
        """Same signature as parent but *loss* can be a vector (one per sample)."""

        # Keep parent behaviour for curves/statistics (aggregate losses as before)
        # If *loss* is a vector, aggregate with .sum() so the parent sees a scalar.
        loss_scalar = loss.sum() if isinstance(loss, torch.Tensor) and loss.ndim > 0 else loss
        if train_step == 0:
            self.KL_curves = {KL_name: [] for KL_name in KL_names}

        for KL_amount, KL_name in zip(KL_amounts, KL_names):
            kl_sum_per_batch = torch.sum(KL_amount.detach(), dim=tuple(range(1, KL_amount.dim())))
            avg_kl = torch.mean(kl_sum_per_batch).item()
            self.KL_curves[KL_name].append(avg_kl)

        B = logits.shape[0]
        self.total_KL_curve.append((total_KL.detach() / B).item())

        self.reconstruction_error_curve.append((reconstruction_error.detach() / B).item())
        self.loss_curve.append((loss_scalar.detach() / B).item())
        if test_reconstruction_error is not None:
            self.test_reconstruction_error_curve.append((test_reconstruction_error.detach() / B).item())

        # ------------------------------------------------------------------
        # Additional Bayesian sampling (only for selected steps)
        # ------------------------------------------------------------------
        if train_step < self.burn_in_steps:
            return
        if (train_step - self.burn_in_steps) % self.track_frequency != 0:
            return

        # Detach tensors without moving to CPU
        logits_det = logits.detach()
        x_mask_det = x_mask.detach()
        y_mask_det = y_mask.detach()
        loss_det = loss.detach()

        # Extract test-part tensors
        logits_t = logits_det[:, self.task.n_train:, :, :, :, 1]
        x_mask_t = x_mask_det[:, self.task.n_train:, :, 1]
        y_mask_t = y_mask_det[:, self.task.n_train:, :, 1]

        # Compute predicted grids for the whole batch *vectorised*
        solutions_mapped, solutions_index = self._make_solutions_batch(logits_t, x_mask_t, y_mask_t)

        # Store information sample-wise ------------------------------------
        B = logits_det.shape[0]
        for b in range(B):
            # 1) unique Y
            sol_mapped = solutions_mapped[b]
            sol_index = solutions_index[b]
            sol_hash = hash(sol_mapped)
            if sol_hash not in self._solution_hash_set:
                self._solution_hash_set.add(sol_hash)
                self.unique_solutions.append(sol_mapped)
                self.unique_index_solutions.append(sol_index)

            # 2) Append to separate lists
            self.losses.append(float(loss_det[b].item()))
            self.logits_samples.append(logits_t[b])
            self.x_mask_samples.append(x_mask_t[b])
            self.y_mask_samples.append(y_mask_t[b])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_solutions_batch(self,
                              pred: torch.Tensor,
                              x_mask: torch.Tensor,
                              y_mask: torch.Tensor):
        """Vectorised conversion of model outputs into discrete grids.

        Returns two parallel lists (mapped_colors, index_colors), each of length B.
        mapped_colors uses actual colour values; index_colors keeps colour indices.
        returns: (B, n_test, X, Y), (B, n_test, X, Y)
        """
        B, n_test, _, _, _ = pred.shape

        # Colours & uncertainties
        colors = torch.argmax(pred, dim=2)                       # (B, n_test, X, Y)

        # Pre-compute best slice (start,end) for x & y per (B, n_test)
        x_slice = self._best_slice_points_batch(x_mask, dim_index=0)          # (B, n_test, 2)
        y_slice = self._best_slice_points_batch(y_mask, dim_index=1)          # (B, n_test, 2)

        mapped_solutions: List[Tuple] = []
        index_solutions: List[List[torch.Tensor]] = []

        for b in range(B):
            mapped_ex_slices = []
            index_ex_slices = []

            for ex in range(n_test):
                xs, xe = x_slice[b, ex]
                ys, ye = y_slice[b, ex]

                grid_tensor = colors[b, ex, xs:xe, ys:ye]

                index_ex_slices.append(grid_tensor)

                grid_list = grid_tensor.cpu().tolist()

                mapped_grid = [[self.task.colors[val] for val in row] for row in grid_list]

                mapped_ex_slices.append(tuple(tuple(r) for r in mapped_grid))

            mapped_solutions.append(tuple(mapped_ex_slices))
            index_solutions.append(index_ex_slices)

        return mapped_solutions, index_solutions

    # ------------------------------------------------------------------
    # Vectorised slice search
    # ------------------------------------------------------------------
    def _best_slice_points_batch(self, mask: torch.Tensor, dim_index: int) -> torch.Tensor:
        """Return optimal (start,end) indices for every (B, n_test) mask.

        mask : (B, n_test, L)
        output: (B, n_test, 2) where the last dim is (start, end)
        """
        B, n_test, L = mask.shape
        total_sum = mask.sum(dim=2, keepdim=True)                # (B, n_test, 1)

        # Prepare tensors to record best values
        best_score = torch.full((B, n_test), -float('inf'), device=mask.device)      # (B, n_test)
        best_start = torch.zeros((B, n_test), dtype=torch.long, device=mask.device)
        best_end = torch.ones((B, n_test), dtype=torch.long, device=mask.device)

        is_fixed = self.task.in_out_same_size or self.task.all_out_same_size
        if not is_fixed:
            search_lengths = range(1, L + 1)
            mask_ = mask.unsqueeze(1)                                # (B,1,n_test,L)
            mask_ = mask_.reshape(B * n_test, 1, L)                  # merge for conv1d
            for length in search_lengths:
                kernel = torch.ones(1, 1, length, device=mask.device)
                seg_sum = F.conv1d(mask_, kernel, stride=1)          # (B*n_test, 1, L-length+1)
                seg_sum = seg_sum.squeeze(1)                         # (B*n_test, offsets)

                score = 2 * seg_sum - total_sum.view(-1, 1)          # broadcast
                # Find best offset for this length
                max_score, max_idx = torch.max(score, dim=1)         # (B*n_test,)

                update_mask = max_score > best_score.view(-1)
                if update_mask.any():
                    best_score.view(-1)[update_mask] = max_score[update_mask]
                    best_start.view(-1)[update_mask] = max_idx[update_mask]
                    best_end.view(-1)[update_mask] = max_idx[update_mask] + length
        else:
            for ex in range(n_test):
                length = self.task.shapes[self.task.n_train + ex][1][dim_index]
                mask_ex = mask[:, ex, :].unsqueeze(1)  # (B, 1, L)
                kernel = torch.ones(1, 1, length, device=mask.device)
                seg_sum = F.conv1d(mask_ex, kernel, stride=1).squeeze(1)  # (B, L-length+1)
                score = 2 * seg_sum - total_sum[:, ex, 0].unsqueeze(1)  # (B, offsets)
                max_score, max_idx = torch.max(score, dim=1)  # (B,)
                best_score[:, ex] = max_score
                best_start[:, ex] = max_idx
                best_end[:, ex] = max_idx + length

        return torch.stack([best_start, best_end], dim=-1)       # (B, n_test, 2)

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------
    def get_unique_solutions(self) -> List[Tuple]:
        return self.unique_solutions

    def get_unique_index_solutions(self) -> List[List[torch.Tensor]]:
        return self.unique_index_solutions

    def get_samples(self):
        return [(loss, logits.cpu(), x_mask.cpu(), y_mask.cpu()) for loss, logits, x_mask, y_mask in zip(self.losses, self.logits_samples, self.x_mask_samples, self.y_mask_samples)]

    # ------------------------------------------------------------------
    # Offline Bayesian aggregation ---------------------------------------------------
    # ------------------------------------------------------------------
    def finalize_solutions(self):
        """Compute `solution_most_frequent` and `solution_second_most_frequent`.

        This must be called **after** training is finished for the task. It
        evaluates *every* candidate grid in ``unique_index_solutions`` under
        *every* posterior sample stored in ``self.samples``. The score used is

            score(Y) = log ∑_i exp( - [ ELBO_i  +  RE_i(Y) ] )

        where *ELBO_i* is the per-sample loss that was logged during training
        and *RE_i(Y)* is the negative log-likelihood ("test reconstruction
        error") of candidate *Y* under sample *i*.
        """
        if not self.unique_index_solutions:
            # Nothing to do
            return

        # 1. Stack sample tensors -----------------------------------------------------
        losses, logits_t, x_mask_t, y_mask_t = self._stack_samples()
        device = logits_t.device  # CPU

        n_candidates = len(self.unique_index_solutions)
        candidate_scores = torch.full((n_candidates,), -float('inf'))

        for idx, Y_index in enumerate(self.unique_index_solutions):
            recon_error = self._candidate_reconstruction_error(
                Y_index, logits_t, x_mask_t, y_mask_t, device=device
            )  # (N_samples,)
            # P(Y|H) = sum_mu P(mu|H) * P(Y|mu), mu is logits
            total_score = torch.logsumexp((-losses - recon_error), dim=0)
            candidate_scores[idx] = total_score

        # Get best two candidates -----------------------------------------------------
        idx = torch.argsort(candidate_scores, descending=True)
        best_idx = idx[0].item()
        second_idx = idx[1].item()
        self.solution_most_frequent = self.unique_solutions[best_idx]
        self.solution_second_most_frequent = self.unique_solutions[second_idx]

    # ------------------------------------------------------------------
    # Helper: stack samples into big tensors
    # ------------------------------------------------------------------
    def _stack_samples(self):
        """Return stacked (losses, logits, x_mask, y_mask)."""
        if not self.losses:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        device = self.logits_samples[0].device
        losses = torch.tensor(self.losses, dtype=torch.float32, device=device)
        logits = torch.stack(self.logits_samples, dim=0).to(device)
        x_mask = torch.stack(self.x_mask_samples, dim=0).to(device)
        y_mask = torch.stack(self.y_mask_samples, dim=0).to(device)
        return losses, logits, x_mask, y_mask

    # ------------------------------------------------------------------
    # Helper: compute reconstruction error of a candidate grid across samples
    # ------------------------------------------------------------------
    def _candidate_reconstruction_error(self,
                                        candidate_index_sol: List[torch.Tensor],
                                        logits_t: torch.Tensor,
                                        x_mask_t: torch.Tensor,
                                        y_mask_t: torch.Tensor,
                                        *,
                                        device=torch.device('cuda')) -> torch.Tensor:
        """Return vector of size (N_samples,) with negative log-likelihood per sample."""

        N = logits_t.shape[0]
        recon_error = torch.zeros(N, dtype=torch.float32, device=device)

        grid_size_uncertain = not (self.task.in_out_same_size or self.task.all_out_same_size)
        coeff_mask = 1.0
        coeff_softmax = 1.0

        if grid_size_uncertain:
            x_grid_log_partitions = compute_grid_size_log_partition(x_mask_t, coeff_mask)  # (N, n_test)
            y_grid_log_partitions = compute_grid_size_log_partition(y_mask_t, coeff_mask)  # (N, n_test)
        else:
            x_grid_log_partitions = None
            y_grid_log_partitions = None

        # Loop over test examples -----------------------------------------------------
        for ex, target in enumerate(candidate_index_sol):
            logits_slice = logits_t[:, ex, :, :, :]
            x_mask_ex = x_mask_t[:, ex, :]
            y_mask_ex = y_mask_t[:, ex, :]

            precomp_x = x_grid_log_partitions[:, ex] if grid_size_uncertain else None
            precomp_y = y_grid_log_partitions[:, ex] if grid_size_uncertain else None

            logprob = compute_grid_logprob(logits_slice, target, x_mask_ex, y_mask_ex, grid_size_uncertain, coeff_mask, coeff_softmax, precomp_x, precomp_y)
            recon_error = recon_error - logprob  # subtract because we aggregate negative log-likelihood

        return recon_error 