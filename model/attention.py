"""
Causal Self-Attention Module.

Implements multi-head self-attention with causal masking.
Causal mask ensures position i can only attend to positions <= i,
which is essential for autoregressive language modeling.

Why fused QKV: Single linear layer is more efficient than three separate ones,
reduces memory reads/writes and parameter overhead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection."""

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
        
        # Fused QKV projection: one linear layer for all three
        # More efficient than separate q,k,v projections
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        
        # Output projection back to d_model
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Dropout for attention weights (regularization)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask: lower triangular matrix
        # registered as buffer so it moves with model to device
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of causal self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        B, T, C = x.size()  # batch, sequence length, embedding dimension
        
        # Fused QKV projection: (B, T, 3*C)
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V and reshape for multi-head attention
        # (B, T, 3*C) -> (B, T, 3, n_heads, head_dim) -> (3, B, n_heads, T, head_dim)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        # attn = Q @ K^T / sqrt(d_k)
        # Shape: (B, n_heads, T, T)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Apply causal mask: set future positions to -inf
        # The mask is broadcast to match (B, n_heads, T, T)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        
        # Softmax over last dimension (keys)
        attn = F.softmax(attn, dim=-1)
        
        # Apply dropout to attention weights
        attn = self.attn_dropout(attn)
        
        # Weighted sum of values: (B, n_heads, T, head_dim)
        out = attn @ v
        
        # Reshape back: (B, n_heads, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection with residual dropout
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
