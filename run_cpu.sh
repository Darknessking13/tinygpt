#!/bin/bash
set -e

echo "=== [CPU] Step 1: Train tokenizer ==="
python tokenizer/train_tokenizer.py \
  --corpus data/raw/corpus.md --save_dir tokenizer/rei_tokenizer --vocab_size 8000

echo "=== [CPU] Step 2: Encode corpus ==="
python training/dataset.py \
  --corpus data/raw/corpus.md \
  --tokenizer tokenizer/rei_tokenizer/tokenizer.json \
  --out_dir data/processed

echo "=== [CPU] Step 3: Pretrain (preset=small, ~15M params) ==="
python training/pretrain.py --preset small \
  --train data/processed/train.bin --val data/processed/val.bin

echo "=== [CPU] Step 4: SFT ==="
python training/sft.py --preset small \
  --pretrain_ckpt checkpoints/pretrain/ckpt_best.pt \
  --sft_data data/raw/sft_data.jsonl

echo "=== [CPU] Step 5: Chat ==="
python inference/chat.py --preset small
echo "Done. ♡"
