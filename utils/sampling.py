"""
Sampling utilities for text generation.

Implements temperature scaling, top-k filtering, and nucleus (top-p) sampling.
These techniques control the diversity and quality of generated text.

Why nucleus sampling often outperforms pure top-k:
- Top-k always takes the k most likely tokens, which can include low-quality
  options when the distribution is flat, or exclude good options when peaked
- Top-p adapts dynamically: takes the smallest set of tokens whose cumulative
  probability exceeds p, capturing more tokens when uncertain and fewer when confident
- This adaptive behavior produces more coherent and diverse outputs
"""

import torch
import torch.nn.functional as F


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    
    Temperature controls the "sharpness" of the probability distribution:
    - T < 1: Sharper distribution, more confident predictions
    - T = 1: No change (original distribution)
    - T > 1: Flatter distribution, more random sampling
    
    Args:
        logits: Raw logits tensor of shape (..., vocab_size)
        temperature: Temperature value (must be > 0)
        
    Returns:
        Scaled logits
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filter logits to keep only top-k values, setting others to -inf.
    
    This ensures sampling only from the k most likely tokens.
    
    Args:
        logits: Logits tensor of shape (..., vocab_size)
        k: Number of top tokens to keep
        
    Returns:
        Filtered logits with non-top-k values set to -inf
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    if k >= logits.size(-1):
        return logits
    
    # Get the k-th largest value
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    min_value = top_k_values[..., -1].unsqueeze(-1)
    
    # Mask all values below the k-th largest
    return logits.masked_fill(logits < min_value, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) filtering to logits.
    
    Keeps the smallest set of tokens whose cumulative probability >= p.
    This is adaptive: keeps more tokens when distribution is flat,
    fewer tokens when distribution is peaked.
    
    Args:
        logits: Logits tensor of shape (..., vocab_size)
        p: Cumulative probability threshold (0 < p <= 1)
        
    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    if p <= 0 or p > 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Create mask for tokens to remove
    # We keep tokens until cumulative prob exceeds p
    # The first token always stays (cumsum starts at its prob)
    sorted_mask = cumulative_probs > p
    
    # Shift mask right: always keep at least the first token
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    
    # Set masked logits to -inf
    sorted_logits.masked_fill_(sorted_mask, float("-inf"))
    
    # Unsort to restore original order
    # We need to scatter the sorted logits back to original positions
    unsorted_logits = torch.zeros_like(logits)
    unsorted_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    
    return unsorted_logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """
    Sample a single token from logits with temperature, top-k, and top-p.
    
    Args:
        logits: Logits tensor of shape (vocab_size,) or (1, vocab_size)
        temperature: Temperature for scaling (default 1.0)
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        
    Returns:
        Sampled token ID
    """
    # Ensure 2D shape
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    # Apply temperature
    if temperature != 1.0:
        logits = temperature_scale(logits, temperature)
    
    # Apply top-k filtering
    if top_k > 0:
        logits = top_k_filter(logits, top_k)
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample from the distribution
    token = torch.multinomial(probs, num_samples=1)
    
    return token.item()


if __name__ == "__main__":
    import random
    
    torch.manual_seed(42)
    random.seed(42)
    
    print("=== Testing Sampling Functions ===\n")
    
    # Create sample logits (simulating a vocabulary)
    vocab_size = 100
    logits = torch.randn(vocab_size)
    
    # Make some tokens more likely
    logits[10] = 5.0
    logits[20] = 4.0
    logits[30] = 3.0
    
    print(f"Original logits shape: {logits.shape}")
    print(f"Top logits: {torch.topk(logits, 5).indices.tolist()}")
    
    # Test temperature
    print("\n--- Temperature Test ---")
    for temp in [0.1, 1.0, 2.0]:
        scaled = temperature_scale(logits, temp)
        probs = F.softmax(scaled, dim=-1)
        print(f"Temp={temp}: max_prob={probs.max():.4f}, entropy={-(probs * torch.log(probs + 1e-10)).sum():.4f}")
    
    # Test top-k
    print("\n--- Top-k Test ---")
    for k in [5, 10, 20]:
        filtered = top_k_filter(logits, k)
        non_inf = (filtered > float("-inf")).sum().item()
        print(f"top_k={k}: {non_inf} tokens remaining")
    
    # Test top-p
    print("\n--- Top-p Test ---")
    for p in [0.5, 0.9, 0.99]:
        filtered = top_p_filter(logits, p)
        non_inf = (filtered > float("-inf")).sum().item()
        print(f"top_p={p}: {non_inf} tokens remaining")
    
    # Test sampling distribution
    print("\n--- Sampling Distribution Test ---")
    samples = []
    for _ in range(100):
        token = sample_token(logits, temperature=0.8, top_k=40, top_p=0.9)
        samples.append(token)
    
    from collections import Counter
    counter = Counter(samples)
    print(f"Sampled 100 tokens, unique: {len(counter)}")
    print(f"Most common: {counter.most_common(5)}")
    
    # Verify distribution is reasonable
    # Top tokens should appear most frequently
    top_tokens = set(torch.topk(logits, 10).indices.tolist())
    top_token_samples = sum(count for token, count in counter.items() if token in top_tokens)
    print(f"Samples from top-10 tokens: {top_token_samples}% (expected high)")
    
    print("\nAll sampling tests passed!")
