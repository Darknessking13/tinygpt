"""
Supervised Fine-Tuning (SFT) for chat format.

SFT adapts a pretrained language model to follow instructions
and produce conversational responses.

Why we mask prompt tokens during SFT:
- The model already learned language patterns during pretraining
- SFT should focus on learning the response format, not relearning prompts
- Masking prevents the model from being penalized for prompt content
- This results in better instruction-following behavior
"""

import os
import sys
import random
import json

import torch
from torch.utils.data import DataLoader

from model.config import GPTConfig
from model.transformer import TinyGPT
from training.dataset import SFTDataset
from tokenizer.train_tokenizer import Tokenizer
from utils.logging import TrainingLogger


class SFTTrainer:
    """Trainer for supervised fine-tuning."""
    
    def __init__(
        self,
        model: TinyGPT,
        tokenizer: Tokenizer,
        train_dataset: SFTDataset,
        config: GPTConfig,
        batch_size: int = 8,
        lr: float = 1e-4,
        max_epochs: int = 3,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        """
        Initialize SFT trainer.
        
        Args:
            model: Pretrained TinyGPT model
            tokenizer: Tokenizer instance
            train_dataset: SFT training dataset
            config: Model configuration
            batch_size: Training batch size
            lr: Learning rate (lower than pretraining)
            max_epochs: Number of fine-tuning epochs
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Data loader
        use_pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_pin,
            collate_fn=self._collate_fn,
        )
        
        # Optimizer with lower learning rate for fine-tuning
        # No weight decay during SFT to preserve learned representations
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.0,
        )
        
        # Logger
        self.logger = TrainingLogger(log_dir=log_dir, log_file="sft.log")
        
        # Training state
        self.global_step = 0
        self.best_loss = float("inf")
        
        print(f"SFT Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Training samples: {len(train_dataset)}")
    
    def _collate_fn(self, batch):
        """Custom collate to handle variable length sequences with masks."""
        input_ids, target_ids, loss_masks = zip(*batch)
        
        # Pad sequences
        max_len = max(len(seq) for seq in input_ids)
        
        padded_inputs = []
        padded_targets = []
        padded_masks = []
        
        for inp, tgt, msk in zip(input_ids, target_ids, loss_masks):
            pad_len = max_len - len(inp)
            padded_inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
            padded_targets.append(torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)]))
            padded_masks.append(torch.cat([msk, torch.zeros(pad_len, dtype=torch.float)]))
        
        return (
            torch.stack(padded_inputs),
            torch.stack(padded_targets),
            torch.stack(padded_masks),
        )
    
    def train_step(self, batch: tuple) -> float:
        """
        Execute a single SFT training step.
        
        Args:
            batch: Tuple of (input_ids, target_ids, loss_mask)
            
        Returns:
            Loss value
        """
        input_ids, target_ids, loss_mask = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        loss_mask = loss_mask.to(self.device)
        
        # Forward pass
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
            logits, _ = self.model(input_ids)
        
        # Calculate masked loss
        # Only compute loss on completion tokens (where loss_mask = 1)
        B, T, V = logits.shape
        
        # Flatten for cross-entropy
        logits_flat = logits.view(B * T, V)
        targets_flat = target_ids.view(B * T)
        mask_flat = loss_mask.view(B * T)
        
        # Cross-entropy loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(logits_flat, targets_flat)
        
        # Apply mask and average
        masked_loss = (losses * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        
        # Backward pass
        masked_loss.backward()
        
        return masked_loss.item()
    
    def train(self):
        """Main SFT training loop."""
        self.logger.log_message("Starting SFT training...")
        
        self.model.train()
        
        for epoch in range(self.max_epochs):
            self.logger.log_message(f"\n=== Epoch {epoch + 1}/{self.max_epochs} ===")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in self.train_loader:
                # Training step
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Log every 10 steps
                if self.global_step % 10 == 0:
                    self.logger.log(
                        step=self.global_step,
                        train_loss=loss,
                    )
            
            # End of epoch
            avg_loss = epoch_loss / num_batches
            self.logger.log_message(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint()
        
        self.logger.log_message("SFT training completed!")
    
    def save_checkpoint(self):
        """Save SFT model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
        }
        
        path = os.path.join(self.checkpoint_dir, "model_sft.pt")
        torch.save(checkpoint, path)
        self.logger.log_message(f"Saved SFT checkpoint to {path}")


def main():
    """Main SFT training function."""
    # Set seeds
    torch.manual_seed(42)
    random.seed(42)
    
    # Paths
    pretrained_path = "checkpoints/model.pt"
    tokenizer_path = "tokenizer/tokenizer.model"
    sft_data_path = "data/sft_data.jsonl"
    
    # Check prerequisites
    if not os.path.exists(pretrained_path):
        print(f"Error: Pretrained model not found at {pretrained_path}")
        print("Run training/trainer.py first to pretrain the model.")
        sys.exit(1)
    
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        sys.exit(1)
    
    if not os.path.exists(sft_data_path):
        print(f"Error: SFT data not found at {sft_data_path}")
        sys.exit(1)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer(tokenizer_path)
    
    # Load pretrained model
    print("Loading pretrained model...")
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    
    # Load config from file or checkpoint
    config_path = "checkpoints/config.json"
    if os.path.exists(config_path):
        config = GPTConfig.load(config_path)
    else:
        print("Config file not found, loading from checkpoint...")
        config = GPTConfig.from_dict(checkpoint["config"])
    
    model = TinyGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Pretrained weights loaded successfully.")
    
    # Create SFT dataset
    print("Loading SFT dataset...")
    sft_dataset = SFTDataset(
        data_path=sft_data_path,
        tokenizer=tokenizer,
        context_length=config.context_length,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_dataset,
        config=config,
        batch_size=16,
        lr=2e-4,
        max_epochs=10,
    )
    
    # Train
    trainer.train()
    
    print("\nSFT complete! Run inference/chat.py to chat with the model.")


if __name__ == "__main__":
    main()
