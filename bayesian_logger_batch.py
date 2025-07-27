import torch
import torch.nn.functional as F
from typing import List, Tuple, Any

from solution_selection_batch import Logger as _BaseLogger


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
        self.unique_index_solutions: List[Tuple] = []     # colour-index grids
        self._solution_hash_set = set()
        # Each element: (elbo_loss (float), logits (Tensor), x_mask (Tensor), y_mask (Tensor))
        self.samples: List[Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]] = []

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

        # Move tensors to CPU & detach to free VRAM
        logits_cpu = logits.detach().cpu()
        x_mask_cpu = x_mask.detach().cpu()
        y_mask_cpu = y_mask.detach().cpu()
        loss_cpu = loss.detach().cpu()

        # Extract test-part tensors: shapes
        #   logits_t   -> (B, n_test, C+1, X, Y)
        #   x_mask_t   -> (B, n_test, X)
        #   y_mask_t   -> (B, n_test, Y)
        logits_t = logits_cpu[:, self.task.n_train:, :, :, :, 1]
        x_mask_t = x_mask_cpu[:, self.task.n_train:, :, 1]
        y_mask_t = y_mask_cpu[:, self.task.n_train:, :, 1]

        # Compute predicted grids for the whole batch *vectorised*
        solutions_mapped, solutions_index = self._make_solutions_batch(logits_t, x_mask_t, y_mask_t)

        # Store information sample-wise ------------------------------------
        B = logits_cpu.shape[0]
        for b in range(B):
            # 1) unique Y
            sol_mapped = solutions_mapped[b]
            sol_index = solutions_index[b]
            sol_hash = hash(sol_mapped)
            if sol_hash not in self._solution_hash_set:
                self._solution_hash_set.add(sol_hash)
                self.unique_solutions.append(sol_mapped)
                self.unique_index_solutions.append(sol_index)

            # 2) raw tensors + corresponding per-sample loss
            self.samples.append((float(loss_cpu[b].item()),
                                 logits_t[b], x_mask_t[b], y_mask_t[b]))

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
        index_solutions: List[Tuple] = []

        for b in range(B):
            mapped_ex_slices = []
            index_ex_slices = []

            for ex in range(n_test):
                xs, xe = x_slice[b, ex]
                ys, ye = y_slice[b, ex]

                # Extract colour indices for this slice once
                grid_idx = colors[b, ex, xs:xe, ys:ye].cpu().numpy().tolist()
                index_ex_slices.append(tuple(tuple(r) for r in grid_idx))

                # Map indices → actual colours using task.colors
                mapped_grid = [[self.task.colors[val] for val in row] for row in grid_idx]
                mapped_ex_slices.append(tuple(tuple(r) for r in mapped_grid))

            mapped_solutions.append(tuple(mapped_ex_slices))
            index_solutions.append(tuple(index_ex_slices))

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

    def get_unique_index_solutions(self) -> List[Tuple]:
        return self.unique_index_solutions

    def get_samples(self):
        return self.samples 

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

            total_score = torch.logsumexp(-(losses + recon_error), dim=0)
            candidate_scores[idx] = total_score

        # Get best two candidates -----------------------------------------------------
        best_idx = torch.argmax(candidate_scores).item()
        second_idx = torch.argsort(candidate_scores, descending=True)[1].item() if n_candidates > 1 else best_idx

        self.solution_most_frequent = self.unique_solutions[best_idx]
        self.solution_second_most_frequent = self.unique_solutions[second_idx]

    # ------------------------------------------------------------------
    # Helper: stack samples into big tensors
    # ------------------------------------------------------------------
    def _stack_samples(self):
        """Return stacked (losses, logits, x_mask, y_mask)."""
        losses = torch.tensor([s[0] for s in self.samples], dtype=torch.float32)
        logits = torch.stack([s[1] for s in self.samples], dim=0)      # (N, n_test, C+1, X, Y)
        x_mask = torch.stack([s[2] for s in self.samples], dim=0)      # (N, n_test, X)
        y_mask = torch.stack([s[3] for s in self.samples], dim=0)      # (N, n_test, Y)
        return losses, logits, x_mask, y_mask

    # ------------------------------------------------------------------
    # Helper: compute reconstruction error of a candidate grid across samples
    # ------------------------------------------------------------------
    def _candidate_reconstruction_error(self,
                                        candidate_index_sol: Tuple,
                                        logits_t: torch.Tensor,
                                        x_mask_t: torch.Tensor,
                                        y_mask_t: torch.Tensor,
                                        *,
                                        device=torch.device('cpu')) -> torch.Tensor:
        """Return vector of size (N_samples,) with negative log-likelihood per sample."""
        from train_batch import mask_select_logprobs  # local import to avoid cycle

        N, n_test, C, X, Y = logits_t.shape
        recon_error = torch.zeros(N, dtype=torch.float32, device=device)

        # Loop over test examples -----------------------------------------------------
        for ex, grid in enumerate(candidate_index_sol):
            if len(grid) == 0 or len(grid[0]) == 0:
                continue  # empty grid, skip
            h = len(grid)
            w = len(grid[0])

            target = torch.tensor(grid, dtype=torch.long, device=device)  # (h, w)

            # Compute log priors for x/y offsets for *all* samples at once
            x_log_partition, x_logprobs = mask_select_logprobs(x_mask_t[:, ex, :], h)  # (N,), (N, O_x)
            y_log_partition, y_logprobs = mask_select_logprobs(y_mask_t[:, ex, :], w)  # (N,), (N, O_y)

            Ox = x_logprobs.shape[1]
            Oy = y_logprobs.shape[1]

            # Prepare broadcast versions
            x_prior = (x_logprobs - x_log_partition.unsqueeze(1)).unsqueeze(2)  # (N, Ox, 1)
            y_prior = (y_logprobs - y_log_partition.unsqueeze(1)).unsqueeze(1)  # (N, 1, Oy)
            prior = x_prior + y_prior                                           # (N, Ox, Oy)

            # Cross-entropy component: iterate over offsets -------------------
            ll_offsets = []  # list of (N, Ox, Oy) partial log-likelihood per offset
            for x_off in range(Ox):
                logits_x = logits_t[:, ex, :, x_off:x_off + h, :]  # (N, C, h, Y)
                slice_rows = []
                for y_off in range(Oy):
                    logits_crop = logits_x[:, :, :, y_off:y_off + w]            # (N, C, h, w)
                    # target broadcast over batch
                    ce = torch.nn.functional.cross_entropy(
                        logits_crop, target.unsqueeze(0).expand(N, -1, -1), reduction='none'
                    )  # (N, h, w)
                    ce_sum = ce.sum(dim=(1, 2))  # (N,)
                    slice_rows.append(-ce_sum)   # log-likelihood (without prior)
                ll_offsets.append(torch.stack(slice_rows, dim=1))  # (N, Oy)
            ll_offsets = torch.stack(ll_offsets, dim=1)            # (N, Ox, Oy)

            logprob = torch.logsumexp(prior + ll_offsets, dim=(1, 2))  # (N,)
            recon_error = recon_error - logprob  # subtract because we aggregate negative log-likelihood

        return recon_error 