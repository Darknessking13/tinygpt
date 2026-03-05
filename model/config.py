"""
GPTConfig: All hyperparameters centralized in one dataclass.

Why these defaults for CPU/small-data training:
- vocab_size=4096: BPE vocab large enough for English, small enough for fast lookup
- context_length=256: Short enough for CPU memory, long enough for coherent text
- n_layers=6: Deeper = better, but 6 is sweet spot for 12M params on CPU
- n_heads=6: d_model/n_heads = 64 dim per head, standard ratio for attention
- d_model=384: Embedding dimension; keeps param count manageable for CPU
- d_ff=1536: FFN hidden dim = 4x d_model, standard Transformer ratio
- dropout=0.1: Regularization for small dataset, prevents overfitting
- bias=False: GPT-NeoX style; slightly faster, no performance loss
"""

from dataclasses import dataclass
import json


@dataclass
class GPTConfig:
    vocab_size: int = 8192      # bigger vocab ✅
    context_length: int = 512   # longer context ✅
    n_layers: int = 8           # deeper ✅
    n_heads: int = 8            # more heads ✅
    d_model: int = 512          # bigger embeddings ✅
    d_ff: int = 2048            # bigger FFN ✅
    dropout: float = 0.1        # less dropout (more data)
    bias: bool = False

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        """Create config from dictionary."""
        return cls(**d)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GPTConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


if __name__ == "__main__":
    config = GPTConfig()
    print(f"GPTConfig defaults: {config}")
    print(f"Config dict: {config.to_dict()}")
