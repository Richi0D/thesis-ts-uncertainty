import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Generic residual MLP used by Chronos-2 for both the input patch embedding
    and the output (quantile) head.

        in_dim -> h_dim -> out_dim,  plus a linear skip in_dim -> out_dim
    """

    def __init__(self, in_dim: int, h_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = nn.ReLU()
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]  (works for any number of leading dims, e.g. [B, V, N, 3P])
        hid = self.act(self.hidden_layer(x))           # [..., h_dim]
        out = self.dropout(self.output_layer(hid))     # [..., out_dim]
        res = self.residual_layer(x)                   # [..., out_dim]
        return out + res                               # [..., out_dim]