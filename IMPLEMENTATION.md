# Rei Corpus — Implementation Complete ✓

## What Was Built

A complete, production-ready GPT-style language model implementation with:

- **Hardware-agnostic training**: Single codebase runs on CPU, single GPU, or multi-GPU DDP
- **~7M to ~85M parameters**: 4 presets (tiny/small/medium/large)
- **Two-phase training**: Pretraining + supervised fine-tuning (SFT)
- **Mixed precision**: Auto-selects bfloat16/float16 based on hardware
- **BPE tokenizer**: HuggingFace tokenizers with special tokens for chat
- **Terminal chat interface**: Rich CLI with JSON output support

## File Structure (15 files)

```
rei_corpus/
├── model/                      # Core architecture
│   ├── config.py              # ModelConfig + 4 presets
│   ├── attention.py           # Causal self-attention
│   └── transformer.py         # GPT model + generation
│
├── training/                   # Training infrastructure
│   ├── device.py              # ★ Hardware auto-detection
│   ├── dataset.py             # Token dataset + DDP-aware DataLoader
│   ├── pretrain.py            # Phase 1: next-token prediction
│   └── sft.py                 # Phase 2: chat fine-tuning
│
├── tokenizer/
│   └── train_tokenizer.py     # BPE tokenizer training
│
├── inference/
│   └── chat.py                # Terminal chat interface
│
├── data/
│   ├── raw/
│   │   ├── corpus.md          # Sample training corpus
│   │   └── sft_data.jsonl     # Sample chat examples
│   └── processed/             # Binary token files (created during training)
│
├── run_cpu.sh                 # Launch: CPU training
├── run_gpu.sh                 # Launch: single GPU
├── run_ddp.sh                 # Launch: 2+ GPU DDP
├── verify_setup.py            # Diagnostic script
└── README.md                  # Documentation
```

## Key Features Implemented

### 1. Hardware Auto-Detection (`training/device.py`)
- Detects CPU / single GPU / multi-GPU automatically
- Returns `DeviceContext` with device, rank, world_size, amp_dtype
- Zero manual configuration required

### 2. Model Architecture
- Pre-norm transformer blocks (LayerNorm before attention/MLP)
- Causal self-attention with proper masking
- Weight tying (lm_head ↔ token_embed)
- Dropout, label smoothing, gradient clipping

### 3. Training Features
- Gradient accumulation
- Cosine learning rate schedule with warmup
- Mixed precision (AMP) with automatic dtype selection
- DDP support via `torchrun`
- Checkpoint saving (rank-0 only in DDP)

### 4. Tokenizer
- BPE with 8000 vocab (configurable)
- Special tokens: `[BOS]`, `[EOS]`, `[PAD]`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|json|>`, `<|/json|>`

### 5. SFT (Supervised Fine-Tuning)
- Loads pretrained checkpoint
- Freezes bottom 50% of layers
- Trains only on assistant responses (user tokens masked)
- Supports JSON output via special tokens

## Quick Start

```bash
cd rei_corpus

# 1. Verify setup (optional)
python verify_setup.py

# 2. Choose your hardware and run
./run_cpu.sh      # CPU: ~4-6 hours for 'small' preset
./run_gpu.sh      # GPU: ~45 min for 'medium' preset
./run_ddp.sh      # 2× GPU: ~25 min for 'large' preset
```

## Training Pipeline

Each launch script runs 5 steps:

1. **Train tokenizer**: BPE on your corpus → `tokenizer/rei_tokenizer/`
2. **Encode corpus**: Text → binary tokens → `data/processed/{train,val}.bin`
3. **Pretrain**: Next-token prediction → `checkpoints/pretrain/ckpt_best.pt`
4. **SFT**: Chat fine-tuning → `checkpoints/sft/ckpt_final.pt`
5. **Chat**: Interactive terminal interface

## Hardware Presets

| Preset | Params | Hardware | Training Time |
|--------|--------|----------|---------------|
| tiny   | ~7M    | CPU      | ~2 hours      |
| small  | ~15M   | CPU/GPU  | ~4-6 hrs CPU, ~45 min GPU |
| medium | ~30M   | GPU 8GB+ | ~45 min       |
| large  | ~85M   | 2× GPU   | ~25 min       |

## Customization

### Use Your Own Corpus
Replace `data/raw/corpus.md` with your text (plain text or markdown).

### Add More SFT Examples
Edit `data/raw/sft_data.jsonl`:
```json
{"user": "Your question", "assistant": "Model response"}
```

### Adjust Hyperparameters
Edit `model/config.py` or create a new preset.

### Change Vocab Size
```bash
python tokenizer/train_tokenizer.py --vocab_size 16000
```

## DDP (Multi-GPU) Details

- Launch via `torchrun --nproc_per_node=N`
- Sets `LOCAL_RANK` env var automatically
- `DeviceContext` detects and initializes `nccl` backend
- Each rank processes different data shards (DistributedSampler)
- Gradients averaged across GPUs before optimizer step
- Only rank 0 logs and saves checkpoints

## Expected Results

- **Initial loss**: ~9.0 (log(vocab_size))
- **After 200 steps**: < 7.0
- **Val/train gap**: < 0.5
- **Coherent text**: After ~500 steps

## Diagnostics

```bash
# Test hardware detection
python -c "from training.device import setup_device; ctx = setup_device(); print(ctx)"

# Test model forward pass
python -c "
import torch
from model.config import PRESETS
from model.transformer import GPT
cfg = PRESETS['tiny']; cfg.vocab_size = 8000
m = GPT(cfg)
x = torch.randint(0, 8000, (2, 64))
logits, loss = m(x, x)
print(f'Loss: {loss.item():.4f}')
print(cfg.param_count())
"
```

## What Makes This Implementation Special

1. **Zero hardware switches**: Same code runs on CPU/GPU/DDP
2. **Minimal dependencies**: PyTorch + tokenizers + tqdm + rich
3. **Weekend-completable**: Friday night → Sunday afternoon
4. **Production patterns**: DDP, mixed precision, gradient accumulation
5. **Educational**: Clean, readable code with no abstractions

## Next Steps

1. **Scale up**: Use a larger corpus (5M+ tokens)
2. **Tune hyperparameters**: Adjust learning rate, batch size, context length
3. **Add more SFT data**: 50-200 examples for better chat quality
4. **Experiment with generation**: Temperature, top-k, top-p sampling
5. **Deploy**: Export to ONNX or TorchScript for production

---

**Built with ♡ — One codebase, any hardware, zero manual switches.**
