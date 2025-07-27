import torch
import torch.nn.functional as F

def compute_sum_logp(logits_slice, target_crop):
    """
    Compute the sum of log-probabilities for a given target crop sliding over the logits_slice.
    Args:
        logits_slice (Tensor): Tensor of shape (B, C, X, Y)
        target_crop (Tensor): Tensor of shape (x, y)
    Returns:
        Tensor: sum_logp of shape (B, X - x + 1, Y - y + 1)
    """
    C = logits_slice.shape[1]
    logp = F.log_softmax(logits_slice, dim=1)  # (B, C, LH, LW)
    target_one_hot = F.one_hot(target_crop.long(), num_classes=C).to(logp.dtype)  # (OH, OW, C)
    target_one_hot = target_one_hot.permute(2, 0, 1).unsqueeze(0)  # (1, C, OH, OW)
    conv_out = torch.nn.functional.conv2d(logp, target_one_hot, padding=0)  # (B, 1, O_x, O_y)
    sum_logp = conv_out.squeeze(1)  # (B, O_x, O_y)
    return sum_logp

def mask_select_logprobs(mask, length):
    """
    Compute the (unnormalised) log-probabilities for selecting every contiguous slice of
    the specified length, **vectorised over the batch dimension**.

    Args:
        mask (Tensor): Tensor of shape (B, L) where B is the batch size and L the
                       maximum possible length. Larger (more positive) entries mean the
                       corresponding index is *less* likely to be masked out.
        length (int):  Desired slice length.

    Returns:
        Tensor: log_partition of shape (B,) – log-partition-function for each batch element.
        Tensor: logprobs      of shape (B, L-length+1) – unnormalised log-probability for
                choosing a slice starting at every possible offset.
    """

    # Compute the sum over every contiguous window of size `length` using a 1-D convolution.
    # The convolution reduces the problem to a single call that is fully vectorised over the batch.
    #   middle_sum: (B, L - length + 1)
    kernel = torch.ones((1, 1, length), dtype=mask.dtype, device=mask.device) # (1, 1, length)
    middle_sum = torch.nn.functional.conv1d(mask.unsqueeze(1), kernel).squeeze(1) # (B, L - length + 1)

    # Total sum per batch element – used to avoid explicitly computing before/after sums.
    # See derivation in commit message:  logprob(offset) = 2 * middle_sum - total_sum
    total_sum = mask.sum(dim=1, keepdim=True)  # (B, 1)

    logprobs = 2 * middle_sum - total_sum  # (B, L - length + 1)
    log_partition = torch.logsumexp(logprobs, dim=1)  # (B,)
    return log_partition, logprobs

def compute_grid_size_log_partition(mask: torch.Tensor, coefficient: float):
    """Vectorised helper to compute the log-partition when the grid size is unknown.

    This replaces the triple-nested loop over `(example_num, in_out_mode, length)`
    with a single loop over `length`, vectorised over the first two dimensions.

    Args:
        mask (Tensor): Tensor of shape (B, E, L) – *without* the `in_out` dimension.
        coefficient (float): Scaling coefficient used in the original code (the
            small value when the grid size is uncertain, otherwise `1`).

    Returns:
        Tensor: log-partition of shape (B, E) containing the log-sum-exp over all
            possible grid sizes for every `(batch, example)` pair.
    """

    # We assume the *last* dimension is `L` (the maximum possible grid length).
    # All remaining dimensions (except the batch dim) are treated uniformly and
    # vectorised over.
    L = mask.shape[-1]

    # Flatten all but the last dimension so we can call `mask_select_logprobs`
    # once per possible length.
    mask_flat = (coefficient * mask).reshape(-1, L)  # (B * R, L),  R = prod(other dims)

    # Accumulate log-partitions for every possible slice length.
    parts = []
    for length in range(1, L + 1):
        part, _ = mask_select_logprobs(mask_flat, length)  # (B * R,)
        parts.append(part)

    # Combine over all lengths (log-sum-exp) and reshape back to the original
    # non-length dimensions.
    log_partition_flat = torch.logsumexp(torch.stack(parts, dim=0), dim=0)  # (B * R,)
    return log_partition_flat.reshape(*mask.shape[:-1])  # (B, *other_dims)

def compute_grid_logprob(logits_slice: torch.Tensor,
                         target: torch.Tensor,
                         x_mask: torch.Tensor,
                         y_mask: torch.Tensor,
                         grid_size_uncertain: bool,
                         coeff_mask: float = 1.0,
                         coeff_softmax: float = 1.0,
                         precomp_x_partition: torch.Tensor = None,
                         precomp_y_partition: torch.Tensor = None) -> torch.Tensor:
    """Compute the log-probability of reconstructing the target grid.

    Args:
        logits_slice: (B, C, X, Y)
        target: (h, w)
        x_mask: (B, max_h)
        y_mask: (B, max_w)
        grid_size_uncertain: bool
        coeff_mask: float
        coeff_softmax: float
        precomp_x_partition: Optional (B,)
        precomp_y_partition: Optional (B,)

    Returns:
        logprob: (B,)
    """
    h, w = target.shape
    x_log_partition, x_logprobs = mask_select_logprobs(coeff_mask * x_mask, h)
    y_log_partition, y_logprobs = mask_select_logprobs(coeff_mask * y_mask, w)
    if grid_size_uncertain:
        x_log_partition = precomp_x_partition
        y_log_partition = precomp_y_partition
    # Prepare broadcast versions
    x_prior = (x_logprobs - x_log_partition.unsqueeze(1)).unsqueeze(2)  # (B, Ox, 1)
    y_prior = (y_logprobs - y_log_partition.unsqueeze(1)).unsqueeze(1)  # (B, 1, Oy)
    prior = x_prior + y_prior                                           # (B, Ox, Oy)
    sum_logp = compute_sum_logp(logits_slice, target)                   # (B, Ox, Oy)
    logprobs = prior + sum_logp
    logprob = torch.logsumexp(coeff_softmax * logprobs, dim=(1, 2)) / coeff_softmax  # (B,)
    return logprob 