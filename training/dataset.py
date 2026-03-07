"""
Dataset utilities for language model training.

Implements TextDataset with sliding window for efficient data usage
and train/val split functionality.
"""

import os
import random
import torch
from torch.utils.data import Dataset, random_split


class TextDataset(Dataset):
    """
    Dataset for autoregressive language modeling.
    
    Tokenizes corpus and creates overlapping windows for training.
    Uses 50% overlap to maximize data efficiency on small corpora.
    """
    
    def __init__(
        self,
        corpus_path: str,
        tokenizer,
        context_length: int = 256,
        stride: int = None,
    ):
        """
        Initialize dataset.
        
        Args:
            corpus_path: Path to training corpus file
            tokenizer: Tokenizer instance with encode() method
            context_length: Maximum sequence length
            stride: Window stride (default: context_length // 2 for 50% overlap)
        """
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.stride = stride if stride is not None else context_length // 2
        
        # Load and tokenize corpus
        print(f"Loading corpus from: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Tokenize entire corpus
        print("Tokenizing corpus...")
        self.token_ids = tokenizer.encode(text)
        
        # Calculate number of windows (lazy - don't store them)
        self.num_windows = max(1, (len(self.token_ids) - self.context_length) // self.stride + 1)
        
        # Print stats
        print(f"\nDataset Statistics:")
        print(f"  Total tokens: {len(self.token_ids):,}")
        print(f"  Context length: {context_length}")
        print(f"  Stride: {self.stride}")
        print(f"  Number of windows: {self.num_windows:,}")
    
    def __len__(self) -> int:
        """Return number of windows."""
        return self.num_windows
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a training example.
        
        Returns:
            Tuple of (input_ids, target_ids) where targets are inputs shifted right
        """
        # Calculate window start position on-the-fly
        start = idx * self.stride
        end = start + self.context_length
        
        # Clamp to valid range
        if end > len(self.token_ids):
            end = len(self.token_ids)
            start = max(0, end - self.context_length)
        
        # Get token window - ensure we have exactly context_length tokens
        tokens = self.token_ids[start:end]
        
        # Ensure we have enough tokens (should always be true with proper clamping)
        if len(tokens) < self.context_length:
            # Pad if needed (shouldn't happen but safety check)
            pad_id = self.tokenizer.pad_id if self.tokenizer.pad_id >= 0 else 0
            tokens = tokens + [pad_id] * (self.context_length - len(tokens))
        
        # Input: all tokens except last
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        
        # Target: all tokens except first (shifted right)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids


class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning on chat data.
    
    Formats chat pairs and masks loss on prompt tokens,
    only training on completion tokens.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        context_length: int = 256,
    ):
        """
        Initialize SFT dataset.
        
        Args:
            data_path: Path to JSONL file with chat pairs
            tokenizer: Tokenizer instance
            context_length: Maximum sequence length
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"SFT data not found: {data_path}")
        
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.examples = []
        
        # Load JSONL data
        print(f"Loading SFT data from: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    import json
                    example = json.loads(line)
                    self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} examples")
        
        # Pre-tokenize examples
        self.processed = self._process_examples()
    
    def _process_examples(self) -> list:
        """
        Process examples into tokenized format with loss masks.
        """
        processed = []
        
        for example in self.examples:
            prompt = example["prompt"]
            completion = example["completion"]
            
            # Tokenize prompt and completion separately
            prompt_ids = self.tokenizer.encode(prompt)
            completion_ids = self.tokenizer.encode(completion)
            
            # Add EOS token to completion
            eos_id = self.tokenizer.eos_id
            if eos_id >= 0:
                completion_ids = completion_ids + [eos_id]
            
            # Concatenate prompt + completion
            full_ids = prompt_ids + completion_ids
            
            # Truncate to context length
            if len(full_ids) > self.context_length:
                # Truncate from the middle of prompt if needed
                excess = len(full_ids) - self.context_length
                if excess < len(prompt_ids):
                    prompt_ids = prompt_ids[excess:]
                    full_ids = prompt_ids + completion_ids
                else:
                    full_ids = full_ids[-self.context_length:]
                    prompt_ids = []
            
            # Create loss mask: 0 for prompt tokens, 1 for completion tokens
            prompt_len = len(prompt_ids)
            loss_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
            
            # Adjust for shifting (target is input shifted right)
            input_ids = full_ids[:-1]
            target_ids = full_ids[1:]
            loss_mask = loss_mask[1:]  # Shift mask to align with targets
            
            processed.append({
                "input_ids": input_ids,
                "target_ids": target_ids,
                "loss_mask": loss_mask,
            })
        
        return processed
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.processed)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a training example.
        
        Returns:
            Tuple of (input_ids, target_ids, loss_mask)
        """
        example = self.processed[idx]
        
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        target_ids = torch.tensor(example["target_ids"], dtype=torch.long)
        loss_mask = torch.tensor(example["loss_mask"], dtype=torch.float)
        
        return input_ids, target_ids, loss_mask


def train_val_split(dataset: Dataset, val_fraction: float = 0.1):
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: Dataset to split
        val_fraction: Fraction of data for validation (0.0 to 1.0)
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    total_size = len(dataset)
    val_size = int(total_size * val_fraction)
    train_size = total_size - val_size
    
    # Use fixed random seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    
    print(f"\nTrain/Val Split:")
    print(f"  Train samples: {train_size:,}")
    print(f"  Val samples: {val_size:,}")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    import sys
    
    print("=== Testing TextDataset ===\n")
    
    # Test will only work if tokenizer exists
    tokenizer_path = "tokenizer/tokenizer.json"
    corpus_path = "data/corpus.md"
    
    if not os.path.exists(tokenizer_path):
        print("Tokenizer not found. Run train_tokenizer.py first.")
        sys.exit(0)
    
    if not os.path.exists(corpus_path):
        print("Corpus not found.")
        sys.exit(0)
    
    from tokenizer.train_tokenizer import Tokenizer
    
    tokenizer = Tokenizer(tokenizer_path)
    
    # Create dataset
    dataset = TextDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        context_length=256,
    )
    
    # Verify minimum size
    if len(dataset) < 1000:
        print(f"WARNING: Only {len(dataset)} windows, expected at least 1000")
    else:
        print(f"OK: {len(dataset)} windows (>= 1000)")
    
    # Test train/val split
    train_dataset, val_dataset = train_val_split(dataset, val_fraction=0.1)
    
    # Decode random samples
    print("\n=== Sample Decodings ===")
    indices = random.sample(range(len(dataset)), min(3, len(dataset)))
    
    for i, idx in enumerate(indices):
        input_ids, target_ids = dataset[idx]
        
        # Decode
        input_text = tokenizer.decode(input_ids.tolist())
        target_text = tokenizer.decode(target_ids.tolist())
        
        print(f"\nSample {i + 1}:")
        print(f"Input: {input_text[:100]}...")
        print(f"Target: {target_text[:100]}...")
    
    print("\nDataset test complete!")
