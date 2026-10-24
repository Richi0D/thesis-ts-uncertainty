import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Generic residual MLP for both the input patch embedding
    and the output (quantile) head.

        in_dim -> h_dim -> out_dim,  plus a linear skip in_dim -> out_dim
    """

    def __init__(self, in_dim: int, h_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = nn.SiLU()
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]  (works for any number of leading dims, e.g. [B, V, N, P])
        hid = self.act(self.hidden_layer(x))           # [..., h_dim]
        out = self.dropout(self.output_layer(hid))     # [..., out_dim]
        res = self.residual_layer(x)                   # [..., out_dim]
        return out + res                               # [..., out_dim]


class PatchEmbed(nn.Module):
    """Patch features -> tokens.  ("Input Residual MLP" in the paper's diagram)"""

    def __init__(self, patch_size, d_model, d_hidden=None):
        super().__init__()
        self.P = patch_size
        self.mlp = ResidualBlock(2 * patch_size, d_hidden or d_model, d_model)

    def forward(self, patches):
        """
        patches: [B, V, N, P] scaled values. NaN = missing in data OR CPM-masked OR future.
        Returns:
            tokens: [B, V, N, D]
            empty:  [B, V, N] bool, True = patch has no observation at all
                    (used later as key-padding mask in attention)
        """
        assert patches.shape[-1] == self.P
        missing = torch.isnan(patches)                                     # True = unobserved
        vals = patches.masked_fill(missing, 0.0)                           # NaN -> 0
        feats = torch.cat([vals, missing.to(vals.dtype)], dim=-1)          # [B, V, N, 2P]
        return self.mlp(feats), missing.all(dim=-1)