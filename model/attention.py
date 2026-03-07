"""
Causal Self-Attention Module.

Implements multi-head self-attention with causal masking.
Uses Flash Attention 2 for memory-efficient attention computation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention 2."""

    def __init__(self, config):
        """
        Initialize attention layer.
        
        Args:
            config: GPTConfig with d_model, n_heads, context_length, dropout, bias
        """
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        # Fused QKV projection
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Dropout
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Flash Attention 2.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        B, T, C = x.size()
        
        # Fused QKV projection
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V: (B, T, 3*C) -> (B, T, 3, n_heads, head_dim)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Flash Attention 2 (memory efficient, no materialized attention matrix)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),  # (B, n_heads, T, head_dim)
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        
        # Reshape: (B, n_heads, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.resid_dropout(self.out_proj(out))
        
        return out


if __name__ == "__main__":
    from model.config import GPTConfig
    
    config = GPTConfig()
    attn = CausalSelfAttention(config)
    
    # Test forward pass
    x = torch.randn(2, 64, config.d_model)
    out = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape mismatch!"
    print("CausalSelfAttention test passed!")
