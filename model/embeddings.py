"""
Token and Positional Embeddings.

Why learned positional embeddings instead of sinusoidal:
- For short contexts (256 tokens), learned embeddings are sufficient
- Learned embeddings can adapt to position patterns in training data
- Simpler to implement and debug
- Sinusoidal embeddings are better for very long sequences or extrapolation,
  which we don't need for this tiny model
"""

import torch
import torch.nn as nn


class TokenAndPositionalEmbedding(nn.Module):
    """Combined token and learned positional embeddings."""

    def __init__(self, config):
        """
        Initialize embedding layers.
        
        Args:
            config: GPTConfig with vocab_size, context_length, d_model, dropout
        """
        super().__init__()
        
        # Token embeddings: map token IDs to vectors
        # Shape: (vocab_size, d_model)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional embeddings: learned position vectors
        # Shape: (context_length, d_model)
        # Why learned: For short contexts, learned embeddings work as well as
        # sinusoidal and are simpler to implement
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Store context length for position generation
        self.context_length = config.context_length

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: embed tokens and add positional embeddings.
        
        Args:
            idx: Token indices of shape (batch, seq_len)
            
        Returns:
            Embedded tensor of shape (batch, seq_len, d_model)
        """
        B, T = idx.size()
        
        # Assert sequence length doesn't exceed context
        if T > self.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds context length {self.context_length}"
            )
        
        # Token embeddings: (B, T) -> (B, T, d_model)
        tok_emb = self.token_embedding(idx)
        
        # Position indices: [0, 1, 2, ..., T-1]
        # Shape: (T,) -> (T, d_model)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)
        
        # Combine: token + position embeddings
        # Broadcasting: (B, T, d_model) + (T, d_model) -> (B, T, d_model)
        x = tok_emb + pos_emb
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


if __name__ == "__main__":
    from model.config import GPTConfig
    
    config = GPTConfig()
    emb = TokenAndPositionalEmbedding(config)
    
    # Test forward pass
    idx = torch.randint(0, config.vocab_size, (2, 64))
    out = emb(idx)
    
    print(f"Input indices shape: {idx.shape}")
    print(f"Output embeddings shape: {out.shape}")
    assert out.shape == (2, 64, config.d_model), "Output shape mismatch!"
    print("TokenAndPositionalEmbedding test passed!")
