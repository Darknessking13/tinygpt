import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = self.qkv(x).split(C, dim=-1)
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))
