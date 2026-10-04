# scaling.py
import torch


def compute_causal_scale_stats(patches, eps=1e-5, min_obs=8):
    """
    patches: [B, V, N, P], NaN = missing. Masked (target) patches must be NaN too!
    Returns loc, scale: [B, V, N, 1]  (stats of patch i use patches 0..i only)
    """
    observed = ~torch.isnan(patches)
    filled = torch.where(observed, patches, torch.zeros_like(patches)).double()  # float64: sum-of-squares trick
    n  = observed.sum(-1).double().cumsum(-1)        # [B, V, N] #observations up to and incl. patch i
    s1 = filled.sum(-1).cumsum(-1)
    s2 = (filled ** 2).sum(-1).cumsum(-1)
    loc = s1 / n.clamp_min(1)
    scale = (s2 / n.clamp_min(1) - loc ** 2).clamp_min(0).sqrt().clamp_min(eps)

    # backfill leading patches with < min_obs points (paper uses 8)
    valid = n >= min_obs
    first = valid.long().argmax(-1, keepdim=True)
    use_own = valid | ~valid.any(-1, keepdim=True)
    loc = torch.where(use_own, loc, loc.gather(-1, first))
    scale = torch.where(use_own, scale, scale.gather(-1, first))
    return loc.to(patches.dtype).unsqueeze(-1), scale.to(patches.dtype).unsqueeze(-1)


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