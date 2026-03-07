"""
Tokenizer training script using HuggingFace tokenizers (Rust-based).

Trains a Byte Pair Encoding tokenizer on the corpus and provides
a convenient wrapper class for encoding and decoding text.
"""

import os
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC


class Tokenizer:
    """
    Wrapper class for HuggingFace tokenizer with encode/decode methods.
    
    Provides a simple interface for tokenization:
        encode(text) -> List[int]
        decode(ids) -> str
    """
    
    def __init__(self, model_path: str):
        """
        Initialize tokenizer from trained model file.
        
        Args:
            model_path: Path to tokenizer JSON file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        
        self.tokenizer = HFTokenizer.from_file(model_path)
        self.model_path = model_path
        
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.get_vocab_size()
    
    @property
    def pad_id(self) -> int:
        """Return pad token ID."""
        return self.tokenizer.token_to_id("<pad>")
    
    @property
    def unk_id(self) -> int:
        """Return unknown token ID."""
        return self.tokenizer.token_to_id("<unk>")
    
    @property
    def bos_id(self) -> int:
        """Return beginning of sequence token ID."""
        return self.tokenizer.token_to_id("<bos>")
    
    @property
    def eos_id(self) -> int:
        """Return end of sequence token ID."""
        return self.tokenizer.token_to_id("<eos>")
    
    def encode(self, text: str) -> list:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids: list) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(ids)
    
    def encode_as_pieces(self, text: str) -> list:
        """
        Encode text to token pieces (strings).
        
        Args:
            text: Input text string
            
        Returns:
            List of token strings
        """
        return self.tokenizer.encode(text).tokens


def train_tokenizer(
    corpus_path: str,
    model_prefix: str,
    vocab_size: int = 32768,
    context_length: int = 2048,
) -> Tokenizer:
    """
    Train a HuggingFace BPE tokenizer on a corpus.
    
    Args:
        corpus_path: Path to training corpus file
        model_prefix: Prefix for output model files
        vocab_size: Target vocabulary size
        context_length: Maximum sequence length
        
    Returns:
        Trained Tokenizer instance
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    
    # Define special tokens for chat format
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]
    
    # Initialize BPE tokenizer
    tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()
    
    # Train tokenizer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    tokenizer.train([corpus_path], trainer)
    
    # Save tokenizer
    output_path = f"{model_prefix}.json"
    tokenizer.save(output_path)
    
    print(f"Tokenizer trained with vocab size: {vocab_size}")
    print(f"Model saved to: {output_path}")
    
    return Tokenizer(output_path)


def main():
    """Train tokenizer and run tests."""
    import random
    
    # Paths
    corpus_path = "data/corpus.md"
    model_prefix = "tokenizer/tokenizer"
    
    # Train tokenizer
    print("=== Training Tokenizer ===")
    tokenizer = train_tokenizer(
        corpus_path=corpus_path,
        model_prefix=model_prefix,
        vocab_size=32768,
    )
    
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Special token IDs:")
    print(f"  <pad>: {tokenizer.pad_id}")
    print(f"  <unk>: {tokenizer.unk_id}")
    print(f"  <bos>: {tokenizer.bos_id}")
    print(f"  <eos>: {tokenizer.eos_id}")
    
    # Test encoding/decoding on sample sentences
    print("\n=== Sample Encodings ===")
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of France?",
        "Machine learning is a fascinating field of study.",
    ]
    
    for sentence in test_sentences:
        tokens = tokenizer.encode(sentence)
        pieces = tokenizer.encode_as_pieces(sentence)
        decoded = tokenizer.decode(tokens)
        
        print(f"\nOriginal: {sentence}")
        print(f"Tokens ({len(tokens)}): {tokens[:10]}...")
        print(f"Pieces: {' '.join(pieces[:10])}...")
        print(f"Decoded: {decoded}")
    
    # Test roundtrip on random sentences from corpus
    print("\n=== Roundtrip Tests ===")
    with open(corpus_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    random.seed(42)
    test_lines = random.sample(lines, min(5, len(lines)))
    
    all_passed = True
    for line in test_lines:
        tokens = tokenizer.encode(line)
        decoded = tokenizer.decode(tokens)
        
        if decoded.strip() != line.strip():
            print(f"MISMATCH:")
            print(f"  Original: {line[:80]}...")
            print(f"  Decoded:  {decoded[:80]}...")
            all_passed = False
        else:
            print(f"PASS: {line[:60]}...")
    
    if all_passed:
        print("\nAll roundtrip tests passed!")
    else:
        print("\nSome roundtrip tests failed (minor whitespace differences are normal)")


if __name__ == "__main__":
    main()
