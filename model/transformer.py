"""
Transformer Block and TinyGPT Model.

Implements the full GPT-style decoder-only Transformer architecture.

Why pre-norm (LayerNorm before attention/FFN) instead of post-norm:
- Pre-norm stabilizes training by ensuring gradients flow through skip connections
- Post-norm can cause gradient explosion/vanishing in deep networks
- Pre-norm is standard in modern Transformers (GPT-2, LLaMA, etc.)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig
from model.attention import CausalSelfAttention
from model.embeddings import TokenAndPositionalEmbedding


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm, attention, and FFN."""

    def __init__(self, config):
        """
        Initialize transformer block.
        
        Args:
            config: GPTConfig instance
        """
        super().__init__()
        
        # Pre-norm layers (LayerNorm before attention and FFN)
        # Why pre-norm: Stabilizes training by normalizing inputs to each sublayer
        # This prevents gradient explosion in deep networks
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Multi-head causal self-attention
        self.attn = CausalSelfAttention(config)
        
        # Feed-forward network (MLP)
        # Two linear layers with GELU activation
        # d_model -> d_ff -> d_model
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x))
        
        # Pre-norm FFN with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


class TinyGPT(nn.Module):
    """GPT-style decoder-only language model."""

    def __init__(self, config: GPTConfig):
        """
        Initialize TinyGPT model.
        
        Args:
            config: GPTConfig instance with model hyperparameters
        """
        super().__init__()
        self.config = config
        
        # Token + positional embeddings
        self.embedding = TokenAndPositionalEmbedding(config)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final layer norm (before LM head)
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Language model head: project to vocab size
        # Weight tying: share weights with token embeddings
        # This reduces parameters and improves generalization
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights: lm_head and token_embedding share the same weights
        self.lm_head.weight = self.embedding.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print parameter count
        print(f"TinyGPT initialized with {self.get_num_params():,} parameters")

    def _init_weights(self, module):
        """Initialize weights with small random values for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self) -> int:
        """Return total number of model parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_params_breakdown(self) -> dict:
        """Return parameter count breakdown by component."""
        n_params = {}
        
        # Embeddings
        n_params["token_embedding"] = self.embedding.token_embedding.weight.numel()
        n_params["position_embedding"] = self.embedding.position_embedding.weight.numel()
        
        # Per block
        block_params = sum(p.numel() for p in self.blocks[0].parameters())
        n_params["per_block"] = block_params
        n_params["all_blocks"] = block_params * self.config.n_layers
        
        # Final layers
        n_params["final_ln"] = sum(p.numel() for p in self.ln_f.parameters())
        
        # LM head (tied, so 0 additional params)
        n_params["lm_head_tied"] = 0
        
        n_params["total"] = self.get_num_params()
        
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        """
        Forward pass for language modeling.
        
        Args:
            idx: Input token indices of shape (batch, seq_len)
            targets: Target token indices of shape (batch, seq_len), optional
                     If provided, computes cross-entropy loss
        
        Returns:
            Tuple of (logits, loss) where loss is None if targets not provided
        """
        # Embed tokens and positions
        x = self.embedding(idx)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy: (B, T, vocab_size) -> (B*T, vocab_size)
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            
            # Cross-entropy loss with label smoothing for regularization
            # Label smoothing prevents overconfident predictions
            loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=0.1)
        
        return logits, loss

    def configure_optimizers(self, weight_decay: float, lr: float):
        """
        Configure AdamW optimizer with separate parameter groups.
        
        Weight decay is applied only to weights (not biases or LayerNorm params)
        to prevent over-regularization of these small parameters.
        
        Args:
            weight_decay: Weight decay coefficient
            lr: Learning rate
            
        Returns:
            AdamW optimizer
        """
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for biases and LayerNorm parameters
            # LayerNorm has weight and bias that shouldn't be decayed
            if name.endswith(".bias") or "ln" in name or "layer_norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # AdamW optimizer
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        
        return optimizer

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "TinyGPT":
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model.pt file
            
        Returns:
            TinyGPT model instance
        """
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load config
        config_path = checkpoint_path.replace("model.pt", "config.json")
        if os.path.exists(config_path):
            config = GPTConfig.load(config_path)
        else:
            # Use checkpoint config if available
            config = GPTConfig.from_dict(checkpoint.get("config", {}))
        
        # Create model
        model = cls(config)
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model


if __name__ == "__main__":
    import random
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Create model
    config = GPTConfig()
    model = TinyGPT(config)
    
    # Print parameter breakdown
    print("\nParameter breakdown:")
    breakdown = model.get_num_params_breakdown()
    for name, count in breakdown.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    B, T = 2, 64
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    
    logits, loss = model(idx, targets)
    
    print(f"Input shape: {idx.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Verify shapes
    assert logits.shape == (B, T, config.vocab_size), f"Logits shape mismatch: {logits.shape}"
    assert torch.isfinite(loss), "Loss is not finite!"
    
    print("\nAll tests passed!")
