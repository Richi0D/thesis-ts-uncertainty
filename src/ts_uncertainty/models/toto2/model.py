# model.py
import math
from dataclasses import dataclass

import torch
from torch import nn

from .embedding import PatchEmbed, ResidualBlock
from .layers import DecoderBlock
from .patching import build_context_patches, build_future_patches
from .scaling import apply_scaling, compute_causal_scale_stats, invert_scaling


@dataclass
class Toto2Output:
    preds_scaled: torch.Tensor   # [B, V, Q, M*P]  predictions in SCALED space (used for the loss)
    loc: torch.Tensor            # [B, V, 1]
    scale: torch.Tensor          # [B, V, 1]


class MyToto2Model(nn.Module):
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

        self.input_embedding = PatchEmbed(patch_size, d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_kv, d_ff, dropout, layer_norm_eps) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(d_model, layer_norm_eps)
        self.output_head = ResidualBlock(d_model, d_ff, self.num_quantiles * patch_size, dropout=dropout)

    def forward(
        self,
        context: torch.Tensor,                              # [B, V, T]  (NaN = missing)
        num_output_patches: int,                            # M
        future_covariates: torch.Tensor | None = None,      # [B, V, H]  (NaN = unknown) or None
        variate_mask: torch.Tensor | None = None,           # [B, V] bool, False = dummy/padding variate
    ) -> Toto2Output:
        
        P, M = self.patch_size, num_output_patches
        if M > self.max_output_patches:
            raise ValueError(f"num_output_patches={M} > max_output_patches={self.max_output_patches}")
        B, V, _ = context.shape
        Q = self.num_quantiles

        # Patching -> features
        ctx_feats = build_context_patches(context, P, self.context_length)              # [B,V,N,P], [B,V,N]
        fut_feats = build_future_patches(future_covariates if future_covariates is not None else None,
                                                   (B, V), M, P,
                                                   device=context.device, dtype=context.dtype)  # [B,V,M,P], [B,V,M,P]

        # Scaling: stats from the context only, applied to context and known future values
        loc, scale = compute_causal_scale_stats(ctx_feats)                           # [B, V, 1] each
        ctx_s = apply_scaling(ctx_feats, loc, scale)                          # [B, V, T']
        fut_s = apply_scaling(fut_feats, loc[:,:,-1:], scale[:,:,-1:])

        # Embedding: the SAME layer for past and future
        ctx_tok, ctx_mask = self.input_embedding(ctx_s)                           # [B, V, N, D]
        fut_tok, fut_mask = self.input_embedding(fut_s)                           # [B, V, M, D]

        tokens = torch.cat([ctx_tok, fut_tok], dim=2)
        attention_mask = torch.cat([ctx_mask, fut_mask], dim=2)

        # Dummy variates: invisible everywhere
        if variate_mask is not None:
            attention_mask = attention_mask & variate_mask[:, :, None]

        #Decoder
        for layer in self.layers:
            hidden = layer(tokens, attention_mask)   # [B, V, S, D]

        # Read the forecast from the last M tokens (the future slots)
        fut_hidden = hidden[:, :, -M:, :]                                   # [B, V, M, D]
        out = self.output_head(self.final_norm(fut_hidden))                 # [B, V, M, Q*P]

        # '(q p)' layout like the official model -> [B, V, Q, M*P]
        out = out.view(B, V, M, Q, P).permute(0, 1, 3, 2, 4).reshape(B, V, Q, M * P)

        return Toto2Output(
            preds_scaled=out,
            loc=loc[:,:,-1:],
            scale=scale[:,:,-1:],
        )

    @torch.no_grad()
    def predict(self, context, prediction_length: int, future_covariates=None, variate_mask=None):
        """Inference in the ORIGINAL scale. Returns [B, V, Q, prediction_length]."""
        self.eval()                                                         # dropout off -> deterministic
        M = math.ceil(prediction_length / self.patch_size)
        out = self(context, M, future_covariates, variate_mask)
        # loc/scale [B, V, 1] -> [B, V, 1, 1] so they broadcast over the quantile axis
        preds = invert_scaling(out.preds_scaled, out.loc, out.scale)
        return preds[..., :prediction_length]  # drop the padded tail