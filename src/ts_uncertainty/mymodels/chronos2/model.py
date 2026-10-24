# model.py
import math
from dataclasses import dataclass

import torch
from torch import nn

from .embedding import ResidualBlock
from .layers import Encoder
from .patching import build_context_patches, build_future_patches
from .scaling import apply_scaling, compute_scale_stats, invert_scaling


@dataclass
class Chronos2Output:
    preds_scaled: torch.Tensor   # [B, V, Q, M*P]  predictions in SCALED space (used for the loss)
    loc: torch.Tensor            # [B, V, 1]
    scale: torch.Tensor          # [B, V, 1]
    future_mask: torch.Tensor    # [B, V, M*P]     1 where the future value was GIVEN as input


class Chronos2Model(nn.Module):
    def __init__(self,
                 num_layers:int,
                 d_model: int,
                 num_heads: int,
                 d_kv:int,
                 d_ff: int,
                 patch_size: int,
                 context_length: int,
                 quantiles: list[float],
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.context_length = context_length
        self.max_output_patches = context_length // patch_size
        self.num_quantiles = len(quantiles)
        # buffer = moves with .to(device), but is not a trainable parameter
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32), persistent=False)

        self.input_embedding = ResidualBlock(3 * patch_size, d_ff, d_model, dropout=dropout)
        # Row 0 = [PAD] (unused, kept for weight compatibility), row 1 = [REG]
        self.special_embedding = nn.Embedding(2, d_model)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_kv, d_ff, dropout=dropout, layer_norm_eps=layer_norm_eps)
        self.output_head = ResidualBlock(d_model, d_ff, self.num_quantiles * patch_size, dropout=dropout)

    def forward(
        self,
        context: torch.Tensor,                              # [B, V, T]  (NaN = missing)
        num_output_patches: int,                            # M
        future_covariates: torch.Tensor | None = None,      # [B, V, H]  (NaN = unknown) or None
        variate_mask: torch.Tensor | None = None,           # [B, V] bool, False = dummy/padding variate
    ) -> Chronos2Output:
        
        P, M = self.patch_size, num_output_patches
        if M > self.max_output_patches:
            raise ValueError(f"num_output_patches={M} > max_output_patches={self.max_output_patches}")
        B, V, _ = context.shape
        D = self.d_model
        Q = self.num_quantiles

        # Truncate BEFORE computing stats: the stats must describe what the model actually sees
        context = context[..., -self.context_length:]                        # [B, V, T']

        # Scaling: stats from the context only, applied to context and known future values
        loc, scale = compute_scale_stats(context)                           # [B, V, 1] each
        ctx_s = apply_scaling(context, loc, scale)                          # [B, V, T']
        fut_s = apply_scaling(future_covariates, loc, scale) if future_covariates is not None else None

        # Patching -> features
        ctx_feats, ctx_attn = build_context_patches(ctx_s, P, self.context_length)              # [B,V,N,3P], [B,V,N]
        fut_feats, fut_mask = build_future_patches(fut_s, (B, V), M, P, self.context_length,
                                                   device=context.device, dtype=context.dtype)  # [B,V,M,3P], [B,V,M,P]

        # Embedding: the SAME layer for past and future
        ctx_tok = self.input_embedding(ctx_feats)                           # [B, V, N, D]
        fut_tok = self.input_embedding(fut_feats)                           # [B, V, M, D]

        # Token sequence [ctx..., REG, fut...] and its attention mask
        ones = lambda n: torch.ones(B, V, n, dtype=torch.bool, device=context.device)
        reg = self.special_embedding.weight[1].view(1, 1, 1, D).expand(B, V, 1, D)  # [B, V, 1, D]                                              # future slots always attended
        tokens = torch.cat([ctx_tok, reg.to(ctx_tok.dtype), fut_tok], dim=2) # [B, V, S, D]
        attention_mask = torch.cat([ctx_attn, ones(1), ones(M)], dim=2)                             # [B, V, S]

        # Dummy variates: invisible everywhere (incl. their REG and future tokens)
        if variate_mask is not None:
            attention_mask = attention_mask & variate_mask[:, :, None]

        #Encoder
        hidden = self.encoder(tokens, attention_mask)                       # [B, V, S, D]

        # Read the forecast from the last M tokens (the future slots)
        fut_hidden = hidden[:, :, -M:, :]                                   # [B, V, M, D]
        out = self.output_head(fut_hidden)                                  # [B, V, M, Q*P]

        # '(q p)' layout like the official model -> [B, V, Q, M*P]
        out = out.view(B, V, M, Q, P).permute(0, 1, 3, 2, 4).reshape(B, V, Q, M * P)

        return Chronos2Output(
            preds_scaled=out,
            loc=loc,
            scale=scale,
            future_mask=fut_mask.reshape(B, V, M * P),
        )

    @torch.no_grad()
    def predict(self, context, prediction_length: int, future_covariates=None, variate_mask=None):
        """Inference in the ORIGINAL scale. Returns [B, V, Q, prediction_length]."""
        self.eval()                                                         # dropout off -> deterministic
        M = math.ceil(prediction_length / self.patch_size)
        out = self(context, M, future_covariates, variate_mask)
        # loc/scale [B, V, 1] -> [B, V, 1, 1] so they broadcast over the quantile axis
        preds = invert_scaling(out.preds_scaled, out.loc.unsqueeze(2), out.scale.unsqueeze(2))
        return preds[..., :prediction_length]  # drop the padded tail