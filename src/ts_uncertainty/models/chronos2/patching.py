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


def _patch_features(patches: torch.Tensor, time_index: torch.Tensor, context_length: int):
    """
    Shared by context and future: build [time_enc | values | mask].

    Args:
        patches:    [B, V, N, P]  (may contain NaN = unknown/padding)
        time_index: [N*P]         integer positions relative to the forecast start
    Returns:
        features: [B, V, N, 3P]
        mask:     [B, V, N, P]   1.0 = observed/known, 0.0 = missing/unknown/padding
    """
    B, V, N, P = patches.shape
    mask = (~torch.isnan(patches)).to(patches.dtype)                       # [B, V, N, P]
    values = torch.where(mask > 0, patches, torch.zeros_like(patches))     # [B, V, N, P]
    time_enc = (time_index.to(patches.dtype) / context_length).view(N, P).expand(B, V, N, P)
    features = torch.cat([time_enc, values, mask], dim=-1)                 # [B, V, N, 3P]
    return features, mask


def build_context_patches(x_scaled: torch.Tensor, patch_size: int, context_length: int):
    """
    Args:
        x_scaled:       [B, V, T]  scaled history or future (NaN = missing)
        context_length: C, the model's maximum context length (8192 in the released model)
    Returns:
        features:       [B, V, N, 3P]
        attention_mask: [B, V, N]
    """
    # The model never sees more than C steps: keep only the most recent ones
    if x_scaled.shape[-1] > context_length:
        x_scaled = x_scaled[..., -context_length:]                        # [B, V, C]
    patches = patchify(x_scaled, patch_size, pad_side="left")             # [B, V, N, P]
    N = patches.shape[2]
    time_index = torch.arange(-N * patch_size, 0, device=patches.device)  # steps before forecast start
    features, mask = _patch_features(patches, time_index, context_length)
    attention_mask = mask.sum(dim=-1) > 0                                  # [B, V, N]
    return features, attention_mask


def build_future_patches(future_scaled: torch.Tensor, batch_shape: tuple[int,int], num_output_patches: int,
                         patch_size: int, context_length: int, device=None, dtype=torch.float32):
    """
    Build the future 'slots'. Known future covariates carry values (mask=1);
    everything unknown (target, past-only covariates, padding) is NaN -> mask=0.

    Args:
        future_scaled: [B, V, H] scaled known-future values with NaN where unknown,
                       or None if nothing about the future is known (pure forecasting).
        batch_shape:   (B, V)  needed when future_scaled is None
        num_output_patches: M, so the model predicts M*P steps (must be >= ceil(H/P))
    Returns:
        features:    [B, V, M, 3P]
        future_mask: [B, V, M, P]   1 where the future value was GIVEN (excluded from the loss later)
    """
    B, V = batch_shape
    if future_scaled is None:
        # Nothing known: every future value is NaN
        patches = torch.full((B, V, num_output_patches, patch_size), float("nan"), device=device, dtype=dtype)
    else:
        patches = patchify(future_scaled, patch_size, pad_side="right")   # [B, V, ceil(H/P), P]
        n = patches.shape[2]
        if n > num_output_patches:
            raise ValueError(f"future has {n} patches but num_output_patches={num_output_patches}")
        if n < num_output_patches:                                         # extend with empty (NaN) patches
            extra = torch.full((B, V, num_output_patches - n, patch_size), float("nan"),
                               device=patches.device, dtype=patches.dtype)
            patches = torch.cat([patches, extra], dim=2)                   # [B, V, M, P]
    time_index = torch.arange(0, num_output_patches * patch_size, device=patches.device)  # steps after forecast start
    features, future_mask = _patch_features(patches, time_index, context_length)
    return features, future_mask