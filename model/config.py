from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 8000
    context_length: int = 256
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = 1536
    dropout: float = 0.1
    batch_size: int = 16
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    warmup_iters: int = 200
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    sft_lr: float = 1e-4
    sft_max_iters: int = 1000

    def param_count(self) -> str:
        p = self.vocab_size * self.d_model * 2 + self.n_layers * 12 * self.d_model ** 2
        return f"~{p/1e6:.1f}M parameters"


PRESETS = {
    "tiny": ModelConfig(n_layers=4, n_heads=4, d_model=256, d_ff=1024,
                       context_length=128, batch_size=8, max_iters=2000),
    "small": ModelConfig(n_layers=6, n_heads=6, d_model=384, d_ff=1536,
                        context_length=256, batch_size=16, max_iters=5000),
    "medium": ModelConfig(n_layers=8, n_heads=8, d_model=512, d_ff=2048,
                         context_length=512, batch_size=32, max_iters=10000),
    "large": ModelConfig(n_layers=12, n_heads=12, d_model=768, d_ff=3072,
                        context_length=512, batch_size=32, max_iters=15000),
}
