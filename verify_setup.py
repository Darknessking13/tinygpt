#!/usr/bin/env python3
"""Diagnostic script to verify Rei Corpus setup."""
import sys

print("=" * 60)
print("Rei Corpus — Setup Verification")
print("=" * 60)

# Check Python version
print(f"\n✓ Python: {sys.version.split()[0]}")

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} | VRAM: {props.total_memory / 1e9:.1f}GB")
            print(f"  bfloat16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("  Mode: CPU training")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

# Check tokenizers
try:
    import tokenizers
    print(f"✓ tokenizers: {tokenizers.__version__}")
except ImportError:
    print("✗ tokenizers not installed")

# Check other deps
for pkg in ["datasets", "tqdm", "rich"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"✓ {pkg}: {ver}")
    except ImportError:
        print(f"⚠ {pkg} not installed (optional)")

# Test device detection
print("\n" + "=" * 60)
print("Hardware Detection Test")
print("=" * 60)
try:
    from training.device import setup_device
    ctx = setup_device()
    print(f"✓ Backend: {ctx.backend}")
    print(f"  Device: {ctx.device}")
    print(f"  World size: {ctx.world_size}")
    print(f"  AMP dtype: {ctx.amp_dtype}")
    print(f"  GradScaler: {ctx.use_scaler}")
except Exception as e:
    print(f"✗ Device detection failed: {e}")

# Test model instantiation
print("\n" + "=" * 60)
print("Model Instantiation Test")
print("=" * 60)
try:
    from model.config import PRESETS
    from model.transformer import GPT
    cfg = PRESETS['tiny']
    cfg.vocab_size = 8000
    m = GPT(cfg)
    print(f"✓ Model created: {cfg.param_count()}")
    
    # Test forward pass
    x = torch.randint(0, 8000, (2, 64))
    logits, loss = m(x, x)
    print(f"  Forward pass: logits shape {logits.shape}")
    print(f"  Initial loss: {loss.item():.4f} (expected ~9.0)")
except Exception as e:
    print(f"✗ Model test failed: {e}")

print("\n" + "=" * 60)
print("Setup verification complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Add your training corpus to data/raw/corpus.md")
print("2. Run: ./run_cpu.sh (or run_gpu.sh / run_ddp.sh)")
print("3. Chat with your model!")
