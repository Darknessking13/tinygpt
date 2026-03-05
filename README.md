# TinyGPT

A tiny GPT-style causal language model (~12M parameters) implemented from scratch using PyTorch. Designed for educational purposes to run on a modern CPU over a weekend.

## Architecture

```
Input Tokens
     │
     ▼
┌─────────────────────────────────────┐
│     Token + Positional Embeddings    │
│         (learned positions)          │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│     Transformer Block × 6            │
│  ┌─────────────────────────────┐    │
│  │     LayerNorm               │    │
│  │          │                  │    │
│  │     Self-Attention          │◄────┼── Residual
│  │          │                  │    │
│  │     Dropout                 │    │
│  └─────────────────────────────┘    │
│                 │                    │
│  ┌─────────────────────────────┐    │
│  │     LayerNorm               │    │
│  │          │                  │    │
│  │     FFN (4× hidden)         │◄────┼── Residual
│  │          │                  │    │
│  │     Dropout                 │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│     Final LayerNorm                  │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│     LM Head (tied to embeddings)     │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│     Softmax → Vocabulary             │
└─────────────────────────────────────┘
     │
     ▼
  Next Token
```

## Requirements

```bash
pip install torch sentencepiece matplotlib
```

- PyTorch 2.0+ (for bfloat16 CPU support)
- SentencePiece (for BPE tokenization)
- Matplotlib (optional, for loss curve plotting)

## Estimated Training Time

On a modern CPU (e.g., Apple M1/M2, Intel i7/i9, AMD Ryzen 7/9):
- **Pretraining**: 30-90 minutes (20 epochs)
- **SFT**: 2-5 minutes (3 epochs)
- **Inference**: ~5-20 tokens/second

Memory requirements: ~500MB RAM for the model and training.

## Execution Steps

1. **Navigate to project directory:**
   ```bash
   cd tinygpt
   ```

2. **Run the full pipeline:**
   ```bash
   bash run_all.sh
   ```

   Or run each step individually:

   ```bash
   # Step 1: Train tokenizer
   python tokenizer/train_tokenizer.py

   # Step 2: Pretrain model
   python training/trainer.py

   # Step 3: Supervised fine-tuning
   python training/sft.py

   # Step 4: Chat with the model
   python inference/chat.py
   ```

3. **Chat with the model:**
   ```bash
   python inference/chat.py --temperature 0.8 --top_k 40 --top_p 0.9
   ```

## Configuration Tuning Guide

| Parameter | Increase Effect | Decrease Effect |
|-----------|----------------|-----------------|
| `n_layers` | More capacity, slower | Less capacity, faster |
| `n_heads` | Finer attention patterns | Coarser attention |
| `d_model` | More representation power | Less representation power |
| `d_ff` | More FFN capacity | Less FFN capacity |
| `context_length` | Longer context, more memory | Shorter context, less memory |
| `dropout` | More regularization | Less regularization |
| `lr` | Faster learning, risk instability | Slower learning, more stable |
| `batch_size` | Faster per epoch, more memory | Slower, less memory |

**Memory vs Quality Trade-offs:**
- Reduce `n_layers` or `d_model` to fit in less RAM
- Reduce `context_length` to process less text per step
- Increase `batch_size` if you have more RAM for faster training

## How SFT Works

Supervised fine-tuning takes a pretrained language model and teaches it to follow instructions. During pretraining, the model learns general language patterns from raw text. SFT then shows the model question-answer pairs, teaching it to respond in a conversational format. The model learns that when it sees `<user>`, it should generate a helpful response followed by `<assistant>`.

## Example Chat Session

```
==============================================
TinyGPT Chat
==============================================
Model: checkpoints/model_sft.pt
Max tokens: 200
Temperature: 0.8
Top-k: 40
Top-p: 0.9
==============================================
Type your message and press Enter to chat.
Type 'quit' or 'exit' to end the session.
==============================================

You: What is photosynthesis?
Assistant: Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. It occurs in the chloroplasts of plant cells and is essential for producing the oxygen we breathe. Plants use the energy from sunlight to split water molecules and combine the hydrogen with carbon dioxide to create sugars.
[8.5 tokens/sec]

You: How do birds fly?
Assistant: Birds fly by generating lift with their wings. The shape of their wings creates lower pressure above than below, pushing them upward. They also have lightweight bones, powerful chest muscles, and feathers that provide both lift and control. Different birds have different wing shapes adapted for their specific flying needs.
[7.2 tokens/sec]

You: quit
Goodbye!
```

## Known Limitations

**What this 12M model CAN do:**
- Generate grammatically correct English sentences
- Answer simple factual questions from the training domain
- Maintain short conversations with context
- Produce structured output with JSON mode

**What this model CANNOT do:**
- Reason about complex problems or math
- Answer questions outside its training data
- Maintain coherent long conversations
- Produce truly novel or creative content
- Understand nuance, humor, or sarcasm
- Provide accurate factual information (may hallucinate)

**Realistic expectations:**
- This is a ~12M parameter model trained on ~1MB of text
- It's comparable to GPT-2 small (117M params) in architecture but much smaller
- Quality is limited by model size and training data
- Best viewed as an educational tool, not a practical AI assistant

## Project Structure

```
tinygpt/
├── data/
│   ├── corpus.md              # Pretraining corpus (~800 lines)
│   └── sft_data.jsonl         # SFT chat pairs (80 examples)
├── tokenizer/
│   └── train_tokenizer.py     # SentencePiece BPE training
├── model/
│   ├── config.py              # GPTConfig dataclass
│   ├── attention.py           # Causal self-attention
│   ├── embeddings.py          # Token + positional embeddings
│   └── transformer.py         # TransformerBlock + TinyGPT
├── training/
│   ├── dataset.py             # TextDataset + SFTDataset
│   ├── trainer.py             # Pretraining loop
│   └── sft.py                 # SFT training loop
├── inference/
│   └── chat.py                # CLI chat interface
├── utils/
│   ├── logging.py             # Training logger + plotting
│   └── sampling.py            # Temperature, top-k, top-p sampling
├── checkpoints/               # Saved models (created during training)
├── logs/                      # Training logs (created during training)
├── run_all.sh                 # End-to-end script
└── README.md                  # This file
```

## Further Reading

- [Karpathy's nanoGPT](https://github.com/karpathy/nanogpt) - Inspiration for this project
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [SentencePiece](https://arxiv.org/abs/1808.06226) - Subword tokenization

## License

MIT License - Feel free to use, modify, and distribute for educational purposes.
