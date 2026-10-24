import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def generate_square_subsequent_mask(sz: int):
    # 0 on and below the diagonal, -inf above it
    return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None=None):
    """
    q:    (batch, heads, seq_q, d_k)
    k:    (batch, heads, seq_k, d_k)
    v:    (batch, heads, seq_k, d_v)
    mask: float, broadcastable to (batch, heads, seq_q, seq_k); 0 = allowed, -inf = blocked
    returns: output (batch, heads, seq_q, d_v), weights (batch, heads, seq_q, seq_k)
    """

    # Matmul and scale
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / np.sqrt(d_k) # (batch, heads, seq_q, seq_k)
    # Additive Mask, 0 = allowed, -inf = blocked
    if mask is not None:
        scores = scores + mask
    # Softmax
    weights = torch.softmax(scores, dim=-1)
    # Weighted sum of the values
    output = weights @ v # (batch, heads, seq_q, d_v)
    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # packed Q, K, V projection, same layout as nn.MultiheadAttention
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _to_additive(self, mask:torch.Tensor, dtype):
        """bool mask (True = blocked) -> float mask (0 / -inf). Float masks pass through."""
        if mask.dtype == torch.bool:
            return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(mask, float("-inf"))
        return mask.to(dtype)

    def forward(self,
                query:torch.Tensor, key:torch.Tensor, value:torch.Tensor,
                attn_mask:torch.Tensor | None=None, key_padding_mask:torch.Tensor| None =None):
        # query: (batch, seq_q, d_model); key, value: (batch, seq_k, d_model)
        B, seq_q, _ = query.shape
        seq_k = key.size(1)

        # Split full weight matrix into Q, K, V weights
        W_q, W_k, W_v = self.in_proj_weight.chunk(3, dim=0)   # each (d_model, d_model)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)            # each (d_model,)

        # Project, create representation for q, k, v
        q = F.linear(query, W_q, b_q)   # (B, seq_q, d_model)
        k = F.linear(key,   W_k, b_k)   # (B, seq_k, d_model)
        v = F.linear(value, W_v, b_v)   # (B, seq_k, d_model)

        # Split q, k, v into smaller attention parts (heads)
        q = q.view(B, seq_q, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, seq_q, d_k)
        k = k.view(B, seq_k, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, seq_k, d_k)
        v = v.view(B, seq_k, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, seq_k, d_k)

        # Combine attn_mask and key_padding_mask into one additive mask
        mask = None
        if attn_mask is not None:
            mask = self._to_additive(attn_mask, q.dtype)  # (seq_q, seq_k)
        if key_padding_mask is not None:
            kpm = self._to_additive(key_padding_mask, q.dtype)[:, None, None, :]  # (B, 1, 1, Lk)
            mask = kpm if mask is None else mask + kpm

        # Scaled_dot_product_attention
        out, weights = scaled_dot_product_attention(q, k, v, mask)

        # Merge heads back to (batch, seq_q, d_model)
        out = out.transpose(1, 2).reshape(B, seq_q, self.d_model)   # (B, seq_q, d_model)

        # Apply out_proj
        return self.out_proj(out), weights.mean(dim=1)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor):
        # x: (B, L, d_model)
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, nhead:int, dim_feedforward:int=2048,
                 dropout:float=0.1, layer_norm_eps:float=1e-5):
        super().__init__()
        # sublayer 1: self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)

        # sublayer 2: feed-forward
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

        # Add & Norm for each sublayer
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src:torch.Tensor, src_mask:torch.Tensor | None=None, src_key_padding_mask:torch.Tensor | None=None):
        x = src

        # sublayer 1: self-attention + Add & Norm
        attn_out, _ = self.self_attn(x, x, x, attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # sublayer 2: feed-forward + Add & Norm
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)       # sublayer 1
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)  # sublayer 2 (cross)
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout)  # sublayer 3

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt:torch.Tensor, memory:torch.Tensor, tgt_mask:torch.Tensor | None=None, memory_mask:torch.Tensor | None=None,
                tgt_key_padding_mask:torch.Tensor | None=None, memory_key_padding_mask:torch.Tensor | None=None):
        x = tgt

        # sublayer 1: masked self-attention
        sa_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask,
                                   key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout1(sa_out))

        # sublayer 2: cross-attention (queries from decoder, keys/values from encoder)
        ca_out, _ = self.multihead_attn(x, memory, memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout2(ca_out))

        # sublayer 3: feed-forward
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class MyTransformer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps) for _ in range(num_encoder_layers)]
            )
        self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Devoder layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps) for _ in range(num_decoder_layers)]
            )
        self.decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)        

    def forward(self,
                src:torch.Tensor,
                tgt:torch.Tensor,
                src_mask:torch.Tensor | None=None,
                tgt_mask:torch.Tensor | None=None,
                memory_mask:torch.Tensor | None=None,
                src_key_padding_mask:torch.Tensor | None=None,
                tgt_key_padding_mask:torch.Tensor | None=None,
                memory_key_padding_mask:torch.Tensor | None=None):
        # src: (B, L_src, d_model), tgt: (B, L_tgt, d_model)

        # encoder pass
        x = src
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory = self.encoder_norm(x)

        # decoder pass
        x = tgt
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)        
        output = self.decoder_norm(x)

        return output   # (B, L_tgt, d_model)    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int = 5000, dropout:float = 0.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sin/cos pairs"
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims
        self.register_buffer("pe", pe)                 # saved with the model, but not trained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        return self.dropout(x + self.pe[: x.size(1)])


class Transformer_Model(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int,
                 horizon_length: int, target_size: int, dropout: float = 0.1,
                 use_torch: bool = True, init_constant: bool = False, max_len: int = 5000):
        super().__init__()
        self.init_constant = init_constant
        self.horizon_length = horizon_length
        self.target_size = target_size

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.query_embed = nn.Parameter(torch.empty(horizon_length, d_model))  # one query per horizon step

        if use_torch:
            self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                              dim_feedforward, dropout, batch_first=True)
        else:
            self.transformer = MyTransformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                             dim_feedforward, dropout)

        self.head = nn.Linear(d_model, target_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        for name, p in module.named_parameters(recurse=False):
            if "bias" in name:
                nn.init.zeros_(p)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(p)             # LayerNorm scale must start at 1
            else:
                if self.init_constant:
                    nn.init.constant_(p, 0.1)    # for implementation check
                else:
                    nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        B = x.size(0)
        src = self.pos_enc(self.input_proj(x))                      # (B, seq_len, d_model)
        tgt = self.query_embed.unsqueeze(0).expand(B, -1, -1)       # (B, horizon, d_model)
        out = self.transformer(src, tgt)                            # (B, horizon, d_model)
        return self.head(out)                                       # (B, horizon, target_size)