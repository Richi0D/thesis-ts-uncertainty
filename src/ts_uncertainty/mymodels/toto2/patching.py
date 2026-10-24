# patching.py
import torch


def patchify(x: torch.Tensor, patch_size: int, pad_side: str = "left") -> torch.Tensor:
    """
    Split the last (time) axis into non-overlapping patches, padding with NaN
    to a multiple of patch_size.
        context -> pad LEFT  (the most recent values align with the forecast start)
        future  -> pad RIGHT (the first future step aligns with the forecast start)

    Args:
        x: [..., L]
    Returns:
        [..., ceil(L/P), P]
    """
    L = x.shape[-1]
    remainder = L % patch_size
    if remainder != 0:
        pad_len = patch_size - remainder
        padding = torch.full((*x.shape[:-1], pad_len), float("nan"), dtype=x.dtype, device=x.device)
        x = torch.cat([padding, x] if pad_side == "left" else [x, padding], dim=-1)
    return x.unfold(dimension=-1, size=patch_size, step=patch_size)       # [..., N, P]


def build_context_patches(context: torch.Tensor, patch_size: int, context_length: int):
    """
    Args:
        context:       [B, V, T]  context values (NaN = missing)
        context_length: C, the model's maximum context length
    Returns:
        features:       [B, V, N, P]
    """
    # The model never sees more than C steps: keep only the most recent ones
    if context.shape[-1] > context_length:
        context = context[..., -context_length:]                        # [B, V, C]
    patches = patchify(context, patch_size, pad_side="left")             # [B, V, N, P]
    return patches


def build_future_patches(future: torch.Tensor, batch_shape: tuple[int,int], num_output_patches: int,
                         patch_size: int, device=None, dtype=torch.float32):
    """
    Build the future 'slots'. Known future covariates carry values (mask=1);
    everything unknown (target, past-only covariates, padding) is NaN -> mask=0.

    Args:
        future: [B, V, H] known-future values with NaN where unknown,
                       or None if nothing about the future is known (pure forecasting).
        batch_shape:   (B, V)  needed when future is None
        num_output_patches: M, so the model predicts M*P steps (must be >= ceil(H/P))
    Returns:
        features:    [B, V, M, P]
    """
    B, V = batch_shape
    if future is None:
        # Nothing known: every future value is NaN
        patches = torch.full((B, V, num_output_patches, patch_size), float("nan"), device=device, dtype=dtype)
    else:
        patches = patchify(future, patch_size, pad_side="right")   # [B, V, ceil(H/P), P]
        n = patches.shape[2]
        if n > num_output_patches:
            raise ValueError(f"future has {n} patches but num_output_patches={num_output_patches}")
        if n < num_output_patches:                                         # extend with empty (NaN) patches
            extra = torch.full((B, V, num_output_patches - n, patch_size), float("nan"),
                               device=patches.device, dtype=patches.dtype)
            patches = torch.cat([patches, extra], dim=2)                   # [B, V, M, P]

    return patches