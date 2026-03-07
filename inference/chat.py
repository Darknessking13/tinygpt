"""
CLI chat interface for interacting with trained TinyGPT.

Provides a minimal chat loop with sampling controls and optional JSON mode.
"""

import os
import sys
import argparse
import time
import json

import torch

from model.config import GPTConfig
from model.transformer import TinyGPT
from tokenizer.train_tokenizer import Tokenizer
from utils.sampling import sample_token


def generate(
    model: TinyGPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    context_length: int = 256,
    json_mode: bool = False,
    device: torch.device = None,
) -> str:
    """
    Generate text from a prompt using the model.
    
    Args:
        model: TinyGPT model instance
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Top-p (nucleus) filtering
        context_length: Maximum context length
        json_mode: Whether to append JSON instruction
        device: Device to run on
        
    Returns:
        Generated text string
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Append JSON instruction if needed
    if json_mode:
        prompt = prompt + " Respond in valid JSON only."
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    
    # Truncate if too long
    if len(input_ids) > context_length:
        input_ids = input_ids[-context_length:]
    
    # Generate tokens
    generated = []
    tokens_per_sec = 0
    
    with torch.no_grad():
        # Convert to tensor
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        start_time = time.time()
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits, _ = model(idx)
            
            # Focus on last position
            logits = logits[0, -1, :]
            
            # Sample next token
            next_token = sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            # Stop on EOS
            if next_token == tokenizer.eos_id:
                break
            
            generated.append(next_token)
            
            # Update context
            idx = torch.cat([idx, torch.tensor([[next_token]], device=device)], dim=1)
            
            # Keep within context window
            if idx.size(1) > context_length:
                idx = idx[:, -context_length:]
        
        elapsed = time.time() - start_time
        if elapsed > 0:
            tokens_per_sec = len(generated) / elapsed
    
    # Decode generated tokens
    output = tokenizer.decode(generated)
    
    return output, tokens_per_sec


def chat_loop(
    model: TinyGPT,
    tokenizer: Tokenizer,
    config: GPTConfig,
    args,
    device: torch.device,
):
    """
    Interactive chat loop.
    
    Args:
        model: TinyGPT model
        tokenizer: Tokenizer instance
        config: Model config
        args: Command line arguments
        device: Device to run on
    """
    print("\n" + "=" * 60)
    print("TinyGPT Chat")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    if args.json:
        print("JSON mode: enabled")
    print("=" * 60)
    print("Type your message and press Enter to chat.")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 60 + "\n")
    
    # Conversation history
    history = ""
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
        # Build prompt with conversation format
        prompt = f"<user> {user_input} <assistant>"
        
        # Include history if not too long
        full_prompt = history + prompt if history else prompt
        
        # Generate response
        response, tokens_per_sec = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=full_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            context_length=config.context_length,
            json_mode=args.json,
            device=device,
        )
        
        # Update history (keep last few exchanges)
        history = full_prompt + response
        if len(tokenizer.encode(history)) > config.context_length // 2:
            # Reset history if too long
            history = prompt + response
        
        # Handle JSON mode
        if args.json:
            print(f"Assistant: {response}")
            # Try to parse as JSON
            try:
                parsed = json.loads(response)
                print(f"(Valid JSON: {type(parsed).__name__})")
            except json.JSONDecodeError:
                print("(Warning: Output is not valid JSON)")
        else:
            print(f"Assistant: {response}")
        
        print(f"[{tokens_per_sec:.1f} tokens/sec]\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chat with TinyGPT")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model_sft.pt",
        help="Path to model checkpoint (default: checkpoints/model_sft.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer/tokenizer.json",
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k filtering (0 to disable)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) filtering (1.0 to disable)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Enable JSON mode for structured output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from (skips interactive mode)",
    )
    
    args = parser.parse_args()
    
    # Check model path
    model_path = args.model
    if not os.path.exists(model_path):
        # Try fallback to pretrained model
        fallback = "checkpoints/model.pt"
        if os.path.exists(fallback):
            print(f"SFT model not found, using pretrained model: {fallback}")
            model_path = fallback
        else:
            print(f"Error: Model not found at {args.model}")
            print("Run training/trainer.py first to train a model.")
            sys.exit(1)
    
    # Check tokenizer
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found at {args.tokenizer}")
        print("Run tokenizer/train_tokenizer.py first.")
        sys.exit(1)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer(args.tokenizer)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load config
    config_dir = os.path.dirname(model_path)
    config_path = os.path.join(config_dir, "config.json")
    if os.path.exists(config_path):
        config = GPTConfig.load(config_path)
    else:
        config = GPTConfig.from_dict(checkpoint.get("config", {}))
    
    model = TinyGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.get_num_params():,} parameters")
    
    # Single prompt mode
    if args.prompt:
        prompt = f"<user> {args.prompt} <assistant>"
        response, tokens_per_sec = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            context_length=config.context_length,
            json_mode=args.json,
            device=device,
        )
        print(response)
        return
    
    # Start chat
    chat_loop(model, tokenizer, config, args, device)


if __name__ == "__main__":
    main()
