import torch
import torch.nn.functional as F
from torch import nn


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

    def forward(self, x: torch.Tensor, key_valid: torch.Tensor, rope=None, causal=False) -> torch.Tensor:
        """
        Args:
            x:             [b, s, D]
            key_valid:     [b, s] bool, True = token has observations (may be attended to)
            causal:        True for time attention, False for variate attention
            rope:          optional callable (q, k) -> (q, k);
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

        # boolean mask, True = may attend.  [b, 1, s(query), s(key)]
        allowed = key_valid[:, None, None, :]
        if causal:
            allowed = allowed & torch.ones(s, s, dtype=torch.bool, device=x.device).tril()
        allowed = allowed | torch.eye(s, dtype=torch.bool, device=x.device)   # ALWAYS, both axes

        scores = q @ k.transpose(-1, -2)                        # [b, h, s, s]   NO 1/sqrt(dk) (see init)
        scores = scores.masked_fill(~allowed, float("-inf"))      # bool -> additive, done once
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
        attn_out = self.attn(self.norm(h), time_mask, rope, causal=True)     # pre-norm -> attention
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
        attn_out = self.attn(self.norm(h), group_mask, causal=False)          # no rope here
        h = h + self.dropout(attn_out)                          # residual
        return h.view(B, S, V, D).permute(0, 2, 1, 3)           # back to [B, V, S, D]


class GatedLinearUnitFeedForwardNetwork(nn.Module):
    """SwiGLU-based FFN with MuP scaling."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or self.adjust_hidden_dim(4 * in_dim)
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, 2 * hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout1 = nn.Dropout(ffn_dropout_p)
        self.dropout2 = nn.Dropout(ffn_dropout_p)

    @staticmethod
    def adjust_hidden_dim(dim) -> int:
        return (int(dim * 2 / 3) + 7) // 8 * 8

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        fc1_out = self.fc1(x)
        gate, x = fc1_out.chunk(2, dim=-1)
        x = self.dropout1(gate * F.silu(x))
        return self.dropout2(self.fc2(x))


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


class DecoderBlock(nn.Module):
    """Pre-norm transformer block: x + Attn(Norm(x)), then x + MLP(Norm(x))."""
    def __init__(self, d_model: int, num_heads: int, d_kv:int, d_ff: int, dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super().__init__()
        # space wise block
        self.norm1 = nn.RMSNorm(d_model, layer_norm_eps)
        self.group_attn = GroupSelfAttention(d_model, num_heads, d_kv, dropout, layer_norm_eps)
        self.norm2 = nn.RMSNorm(d_model, layer_norm_eps)
        self.group_mlp = GatedLinearUnitFeedForwardNetwork(d_model, d_ff, d_model, ffn_dropout_p=dropout)
        # time wise block
        self.norm3 = nn.RMSNorm(d_model, layer_norm_eps)
        self.time_attn = TimeSelfAttention(d_model, num_heads, d_kv, dropout, layer_norm_eps)
        self.norm4 = nn.RMSNorm(d_model, layer_norm_eps)
        self.time_mlp = GatedLinearUnitFeedForwardNetwork(d_model, d_ff, d_model, ffn_dropout_p=dropout)
        self.rope = RotaryEmbedding(d_kv) 

    def forward(self, x, empty):
        B, V, S, _ = x.shape

        # Time attention: for each series (b, v), mask over its S positions
        time_mask = ~empty.reshape(B * V, S)
        # Group attention: for each position (b, s), mask over its V variates
        group_mask = ~empty.transpose(1, 2).reshape(B * S, V)

        # Positions 0..S-1 over the whole sequence [ctx..., REG, fut...]; same for all layers and heads
        cos, sin = self.rope(S, x.device)                          # [S, dk] each
        rope = lambda q, k: apply_rope(q, k, cos, sin)

        x = x + self.group_attn(self.norm1(x), group_mask)
        x = x + self.group_mlp(self.norm2(x))
        x = x + self.time_attn(self.norm3(x), time_mask, rope)
        x = x + self.time_mlp(self.norm4(x))
        return x