# Rei Corpus
### A Complete GPT-Style Language Model from Scratch

Weekend-completable ~10–50M parameter GPT-style transformer built entirely in PyTorch.
Auto-adapts to CPU, single GPU, or 2+ GPU DDP with zero config changes.

## Quick Start

```bash
# 1. Install dependencies
pip install torch tokenizers datasets tqdm rich

# 2. Add your training corpus
echo "Your training text here..." > data/raw/corpus.md

# 3. Create SFT data (optional)
cat > data/raw/sft_data.jsonl << 'EOF'
{"user": "What is the capital of France?", "assistant": "The capital of France is Paris."}
{"user": "Tell me a joke.", "assistant": "Why did the programmer quit? Because they didn't get arrays."}
EOF

# 4. Launch training (choose your hardware)
./run_cpu.sh      # CPU training
./run_gpu.sh      # Single GPU
./run_ddp.sh      # 2+ GPUs with DDP
```

## Project Structure

```
rei_corpus/
├── model/              # GPT architecture
│   ├── config.py       # ModelConfig + presets
│   ├── attention.py    # Causal self-attention
│   └── transformer.py  # Full GPT model
├── training/           # Training loops
│   ├── device.py       # Hardware auto-detection
│   ├── dataset.py      # Data processing
│   ├── pretrain.py     # Phase 1: next-token prediction
│   └── sft.py          # Phase 2: chat fine-tuning
├── tokenizer/          # BPE tokenizer training
├── inference/          # Chat CLI
└── data/               # Your corpus + processed tokens
```

## Hardware Presets

| Preset | Params | Layers | d_model | Context | Hardware |
|--------|--------|--------|---------|---------|----------|
| tiny   | ~7M    | 4      | 256     | 128     | CPU (fast iteration) |
| small  | ~15M   | 6      | 384     | 256     | CPU / Single GPU |
| medium | ~30M   | 8      | 512     | 512     | Single GPU 8GB+ |
| large  | ~85M   | 12     | 768     | 512     | 2× GPU DDP |

## Features

- **Hardware-agnostic**: Single `DeviceContext` handles CPU/GPU/DDP
- **Mixed precision**: Auto-selects bfloat16/float16 based on GPU capability
- **Two-phase training**: Pretrain → SFT (chat fine-tuning)
- **JSON output**: Structured generation via special tokens
- **DDP support**: Multi-GPU training via `torchrun`
- **Weight tying**: Reduces params ~15%
- **Overfitting mitigation**: Dropout, label smoothing, grad clipping, cosine LR

## Training Pipeline

1. **Tokenizer**: Train BPE tokenizer on your corpus
2. **Data prep**: Encode text → binary token files
3. **Pretrain**: Next-token prediction (causal LM)
4. **SFT**: Fine-tune on chat examples (optional)
5. **Inference**: Terminal chat interface

## Diagnostics

```bash
# Verify hardware detection
python -c "
from training.device import setup_device
ctx = setup_device()
print(f'Backend: {ctx.backend}')
print(f'Device: {ctx.device}')
print(f'AMP dtype: {ctx.amp_dtype}')
"

# Test forward pass
python -c "
import torch
from model.config import PRESETS
from model.transformer import GPT
cfg = PRESETS['small']; cfg.vocab_size = 8000
m = GPT(cfg)
x = torch.randint(0, 8000, (2, 64))
logits, loss = m(x, x)
print(f'Loss: {loss.item():.4f} (expected ~9.0)')
print(cfg.param_count())
"
```

## Expected Results

- Initial loss: ~9.0 (log(vocab_size))
- Loss < 7.0 within 200 steps
- Val loss within 0.5 of train loss
- Coherent English after ~500 steps

## License

MIT — build, modify, ship. ♡
