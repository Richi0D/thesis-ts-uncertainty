# scaling.py
import torch


def compute_scale_stats(context: torch.Tensor, eps: float = 1e-5):
    """
    Per-series mean and std over the time axis, ignoring NaNs.

    Args:
        context: [..., T]   e.g. [B, T] or [B, V, T]; may contain NaN for missing values
    Returns:
        loc:   [..., 1]
        scale: [..., 1]
    """
    observed = ~torch.isnan(context)                                       # [..., T] bool
    count = observed.sum(dim=-1, keepdim=True).clamp_min(1)                # [..., 1]  avoid /0 for all-NaN series

    # Replace NaN by 0 so sums ignore them (they also don't count in `count`)
    filled = torch.where(observed, context, torch.zeros_like(context))     # [..., T]
    loc = filled.sum(dim=-1, keepdim=True) / count                         # [..., 1]

    # Variance only over observed points
    centered = torch.where(observed, context - loc, torch.zeros_like(context))  # [..., T]
    var = (centered ** 2).sum(dim=-1, keepdim=True) / count                # [..., 1]
    scale = var.sqrt().clamp_min(eps)                                      # [..., 1]  flat series -> no /0

    return loc, scale


def apply_scaling(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
    """
    Standardize with GIVEN stats, then arcsinh.
    Used for the context AND for the target / known future covariates
    (always with the stats from the context).
    """
    return torch.asinh((x - loc) / scale)


def invert_scaling(z: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
    """
    Exact inverse: sinh, then undo standardization.
    z may have extra dims (e.g. quantiles) as long as loc/scale broadcast.
    """
    return torch.sinh(z) * scale + loc