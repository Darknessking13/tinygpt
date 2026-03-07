#!/bin/bash
set -e

echo "=== [DDP] Step 1: Train tokenizer ==="
python tokenizer/train_tokenizer.py \
  --corpus data/raw/corpus.md --save_dir tokenizer/rei_tokenizer --vocab_size 8000

echo "=== [DDP] Step 2: Encode corpus ==="
python training/dataset.py \
  --corpus data/raw/corpus.md \
  --tokenizer tokenizer/rei_tokenizer/tokenizer.json \
  --out_dir data/processed

echo "=== [DDP] Step 3: Pretrain (2 GPUs, preset=large, ~85M params) ==="
torchrun --standalone --nproc_per_node=2 training/pretrain.py --preset large \
  --train data/processed/train.bin --val data/processed/val.bin

echo "=== [DDP] Step 4: SFT (2 GPUs) ==="
torchrun --standalone --nproc_per_node=2 training/sft.py --preset large \
  --pretrain_ckpt checkpoints/pretrain/ckpt_best.pt \
  --sft_data data/raw/sft_data.jsonl

echo "=== [DDP] Step 5: Chat (single process, auto-selects GPU 0) ==="
python inference/chat.py --preset large

echo "Done. ♡"
