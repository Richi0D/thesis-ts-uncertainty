import torch
import torch.nn.functional as F
from torch import nn


def make_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a boolean 'keep' mask into an additive mask for attention scores.
        True  -> 0           (attend normally)
        False -> very negative (softmax weight ~ 0)

    Uses finfo.min instead of -inf: if ALL keys of a query were masked, -inf would give
    softmax(-inf, ..., -inf) = NaN; finfo.min gives a harmless uniform distribution instead.
    """
    return (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min


class MultiHeadAttention(nn.Module):
    """
    Generic multi-head self-attention on [b, s, D].
    Reused for time attention (s = time) and, in C3, group attention (s = variates).
    """

    def __init__(self, d_model:int, num_heads:int, d_kv:int, dropout:float=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_kv
        inner = num_heads * d_kv

        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # Official (Mesh-TensorFlow / T5) init. q is sqrt(d_kv) smaller -> replaces the 1/sqrt(d_k) scaling.
        f, D = 0.05, d_model
        nn.init.normal_(self.q.weight, std=f * (D * d_kv) ** -0.5)
        nn.init.normal_(self.k.weight, std=f * D ** -0.5)
        nn.init.normal_(self.v.weight, std=f * D ** -0.5)
        nn.init.normal_(self.o.weight, std=f * inner ** -0.5)

    def forward(self, x: torch.Tensor, additive_mask: torch.Tensor, rope=None) -> torch.Tensor:
        """
        Args:
            x:             [b, s, D]
            additive_mask: broadcastable to [b, h, s, s], e.g. [b, 1, 1, s] (mask over keys)
            rope:          optional callable (q, k) -> (q, k); added in Step 7
        Returns:
            [b, s, D]
        """
        b, s, _ = x.shape

        # Project and split into heads: [b, s, h*dk] -> [b, s, h, dk] -> [b, h, s, dk]
        q = self.q(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)

        if rope is not None:
            q, k = rope(q, k)                                   # rotate queries/keys by position

        scores = q @ k.transpose(-1, -2)                        # [b, h, s, s]   NO 1/sqrt(dk) (see init)
        scores = scores + additive_mask                         # masked keys -> ~ -inf
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)   # softmax in fp32 for stability
        weights = self.attn_dropout(weights)

        out = weights @ v                                       # [b, h, s, dk]
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.d_k)  # [b, s, h*dk]
        return self.o(out)


class TimeSelfAttention(nn.Module):
    """
    Attention ALONG TIME, separately for every (batch element, variate) pair.
    Each series only looks at its own tokens: [ctx..., REG, fut...].
    """

    def __init__(self, d_model:int, num_heads:int, d_kv:int, dropout:float=0.1, layer_norm_eps:float=1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, layer_norm_eps)
        self.attn = MultiHeadAttention(d_model, num_heads, d_kv, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor, rope=None) -> torch.Tensor:
        """
        Args:
            x:         [B, V, S, D]
            time_mask: [B*V, 1, 1, S] additive
        """
        B, V, S, D = x.shape
        h = x.reshape(B * V, S, D)                              # every series becomes its own sequence
        attn_out = self.attn(self.norm(h), time_mask, rope)     # pre-norm -> attention
        h = h + self.dropout(attn_out)                          # residual
        return h.view(B, V, S, D)


class GroupSelfAttention(nn.Module):
    """
    Attention ACROSS VARIATES at the same sequence position.
    For every (batch element, position) pair, the V variate tokens attend to each other.
    No RoPE: variates have no natural order -> permutation-equivariant.
    """

    def __init__(self, d_model:int, num_heads:int, d_kv:int, dropout:float=0.1, layer_norm_eps:float=1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, layer_norm_eps)
        self.attn = MultiHeadAttention(d_model, num_heads, d_kv, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, group_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [B, V, S, D]
            group_mask: [B*S, 1, 1, V] additive (mask over key VARIATES)
        """
        B, V, S, D = x.shape
        # Swap variate and sequence axes, then fold positions into the batch:
        # every (b, s) becomes its own little 'sequence' of V tokens
        h = x.permute(0, 2, 1, 3).reshape(B * S, V, D)          # [B*S, V, D]
        attn_out = self.attn(self.norm(h), group_mask)          # no rope here
        h = h + self.dropout(attn_out)                          # residual
        return h.view(B, S, V, D).permute(0, 2, 1, 3)           # back to [B, V, S, D]


class FeedForward(nn.Module):
    """Position-wise MLP (D -> d_ff -> D), pre-norm + residual, no biases."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, layer_norm_eps)
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.ReLU()
        self.inner_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.wi.weight, std=0.05 * d_model ** -0.5)
        nn.init.normal_(self.wo.weight, std=0.05 * d_ff ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D] -- acts on every token independently, so any leading shape works
        h = self.wo(self.inner_dropout(self.act(self.wi(self.norm(x)))))
        return x + self.dropout(h)


class EncoderBlock(nn.Module):
    """time attention -> group attention -> feed-forward"""

    def __init__(self, d_model: int, num_heads: int, d_kv:int, d_ff: int, dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.time_attn = TimeSelfAttention(d_model, num_heads, d_kv, dropout, layer_norm_eps)
        self.group_attn = GroupSelfAttention(d_model, num_heads, d_kv, dropout, layer_norm_eps)
        self.ffn = FeedForward(d_model, d_ff, dropout, layer_norm_eps)

    def forward(self, x, time_mask, group_mask, rope=None):
        x = self.time_attn(x, time_mask, rope)     # mix along time, within each series
        x = self.group_attn(x, group_mask)         # mix across variates, at each position
        x = self.ffn(x)                            # per-token MLP
        return x


class Encoder(nn.Module):
    """Stack of encoder blocks + final RMSNorm."""

    def __init__(self, num_layers:int, d_model: int, num_heads: int, d_kv:int, d_ff: int, dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, d_kv, d_ff, dropout, layer_norm_eps) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(d_model, layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(d_kv)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, rope=None) -> torch.Tensor:
        """
        Args:
            x:              [B, V, S, D]   tokens [ctx..., REG, fut...]
            attention_mask: [B, V, S] bool, False = this token may not be READ (as a key).
                            For a dummy/padding variate set its whole row to False.
        Returns:
            [B, V, S, D]
        """
        B, V, S, _ = x.shape

        # Time attention: for each series (b, v), mask over its S positions
        time_mask = make_additive_mask(attention_mask, x.dtype).reshape(B * V, 1, 1, S)

        # Group attention: for each position (b, s), mask over its V variates
        group_mask = make_additive_mask(attention_mask.transpose(1, 2), x.dtype).reshape(B * S, 1, 1, V)

        # Positions 0..S-1 over the whole sequence [ctx..., REG, fut...]; same for all layers and heads
        cos, sin = self.rope(S, x.device)                          # [S, dk] each
        rope = lambda q, k: apply_rope(q, k, cos, sin)

        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, time_mask, group_mask, rope)
        return self.dropout(self.final_norm(x))


class RotaryEmbedding(nn.Module):
    """
    Precomputes the rotation angles. No trainable parameters.
    inv_freq is a non-persistent buffer: moves with .to(device), not saved in state_dict
    (same as the official model -> no extra keys when loading weights).
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        # One frequency per dimension PAIR: theta^(-0/d), theta^(-2/d), ..., theta^(-(d-2)/d)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))   # [dim/2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, seq_len: int, device):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)             # [S]
        freqs = torch.outer(positions, self.inv_freq.to(device))                           # [S, dim/2]  angle = pos * freq
        emb = torch.cat([freqs, freqs], dim=-1)                                            # [S, dim]    half-split layout
        return emb.cos(), emb.sin()                                                        # computed in fp32 on purpose


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1) where x1, x2 are the first/second half of the last dim."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    q, k:     [b, h, S, dk]
    cos, sin: [S, dk]   (broadcast over b and h)
    """
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot