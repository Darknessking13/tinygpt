#!/bin/bash
set -e

echo "=== [GPU] Step 1: Train tokenizer ==="
python tokenizer/train_tokenizer.py \
  --corpus data/raw/corpus.md --save_dir tokenizer/rei_tokenizer --vocab_size 8000

echo "=== [GPU] Step 2: Encode corpus ==="
python training/dataset.py \
  --corpus data/raw/corpus.md \
  --tokenizer tokenizer/rei_tokenizer/tokenizer.json \
  --out_dir data/processed

echo "=== [GPU] Step 3: Pretrain (preset=medium, ~30M params) ==="
CUDA_VISIBLE_DEVICES=0 python training/pretrain.py --preset medium \
  --train data/processed/train.bin --val data/processed/val.bin

echo "=== [GPU] Step 4: SFT ==="
CUDA_VISIBLE_DEVICES=0 python training/sft.py --preset medium \
  --pretrain_ckpt checkpoints/pretrain/ckpt_best.pt \
  --sft_data data/raw/sft_data.jsonl

echo "=== [GPU] Step 5: Chat ==="
python inference/chat.py --preset medium
echo "Done. ♡"
