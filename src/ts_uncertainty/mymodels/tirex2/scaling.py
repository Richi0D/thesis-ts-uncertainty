import torch
from torch import nn


ScalerState = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]

class Scaler:
    """Per-variate standardization over the WHOLE row (context + future window), then arcsinh.
    `fit` and `scale` are split on purpose: for streaming you fit once and then reuse the state.
    """
 
    def __init__(self, eps: float = 1e-8, use_arcsinh: bool = True, binaryaware: bool = True):
        self.eps = eps
        self.use_arcsinh = use_arcsinh
        self.binaryaware = binaryaware
 
    def fit(self, x: torch.Tensor) -> ScalerState:
        """x: [B, V, T] with NaNs (padding, missing values, the unknown future)."""

        observed = ~torch.isnan(x)                                       # [..., T] bool
        count = observed.sum(dim=-1, keepdim=True).clamp_min(1)                # [..., 1]  avoid /0 for all-NaN series
        # Replace NaN by 0 so sums ignore them (they also don't count in `count`)
        filled = torch.where(observed, x, torch.zeros_like(x))     # [..., T]
        loc = filled.sum(dim=-1, keepdim=True) / count                         # [..., 1]
        centered = torch.where(observed, x - loc, torch.zeros_like(x))  # [..., T]
        var = (centered ** 2).sum(dim=-1, keepdim=True) / count                # [..., 1]
        scale = var.sqrt().clamp_min(self.eps)                                   # [..., 1]  flat series -> no /0

        is_binary = None
        if self.binaryaware:
            is_binary = torch.all((x == 0) | (x == 1)| ~observed, dim=-1, keepdim=True) 
            loc = torch.where(is_binary, torch.zeros_like(loc), loc)
            scale = torch.where(is_binary, torch.ones_like(scale), scale)

        return loc, scale, is_binary
 
    def scale(self, x: torch.Tensor, state: ScalerState | None = None) -> tuple[torch.Tensor, ScalerState]:
        if state is None:
            state = self.fit(x)
        loc, scale, is_binary = state
        z = (x - loc) / scale
        if self.use_arcsinh:
            if is_binary is not None and is_binary.any():
                z = torch.where(is_binary, z, torch.arcsinh(z))
            else:
                z = torch.arcsinh(z)
        return z, state
 
    def re_scale(self, y: torch.Tensor, state: ScalerState) -> torch.Tensor:
        """y: [B, V, Q, T]  (note the extra quantile axis vs. the state's [B, V, 1])."""
        loc, scale, is_binary = state
        if self.use_arcsinh:
            sinh_x = torch.sinh(torch.clamp(y, -20.0, 20.0))
            if is_binary is not None and is_binary.any():
                # is_binary: (B, 1) -> broadcast over quantile and time dims
                y = torch.where(is_binary[:, :, None], y, sinh_x)
            else:
                y = sinh_x
        return y * scale[:, :, None] + loc[:, :, None]