#!/bin/bash
# End-to-end script: tokenize → train → sft → chat
# Run from the tinygpt directory

set -e

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Parse arguments
USE_GENERATED=false
if [ "$1" = "--use-generated" ]; then
    USE_GENERATED=true
fi

echo "=============================================="
echo "TinyGPT End-to-End Training Pipeline"
echo "=============================================="

# Append generated segments if flag is set
if [ "$USE_GENERATED" = true ]; then
    echo ""
    echo "=== Appending Generated Segments ==="
    if ls data/corpus_generated_*.md 1> /dev/null 2>&1; then
        cat data/corpus_generated_*.md >> data/corpus.md
        echo "Added generated segments to corpus"
    else
        echo "Warning: No generated segments found (corpus_generated_*.md)"
    fi
fi

# Step 1: Train Tokenizer
echo ""
echo "=== Step 1: Train Tokenizer ==="
python tokenizer/train_tokenizer.py

# Step 2: Pretrain Model
echo ""
echo "=== Step 2: Pretrain Model ==="
torchrun --nproc_per_node=2 training/trainer.py

# Step 3: Supervised Fine-Tuning
echo ""
echo "=== Step 3: Supervised Fine-Tuning ==="
python training/sft.py

# Step 4: Launch Chat
echo ""
echo "=== Step 4: Launch Chat ==="
echo "Training complete! Starting chat interface..."
python inference/chat.py
