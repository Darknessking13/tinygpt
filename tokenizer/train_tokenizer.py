import os, argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing


def train_tokenizer(corpus_path: str, save_dir: str, vocab_size: int = 8000):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]",
                       "<|user|>", "<|assistant|>", "<|end|>",
                       "<|json|>", "<|/json|>"]
    )
    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")),
                       ("[EOS]", tokenizer.token_to_id("[EOS]"))]
    )
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(f"{save_dir}/tokenizer.json")
    print(f"Tokenizer saved. Vocab: {tokenizer.get_vocab_size()} tokens")
    return tokenizer


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/raw/corpus.md")
    p.add_argument("--save_dir", default="tokenizer/rei_tokenizer")
    p.add_argument("--vocab_size", type=int, default=8000)
    args = p.parse_args()
    train_tokenizer(args.corpus, args.save_dir, args.vocab_size)
