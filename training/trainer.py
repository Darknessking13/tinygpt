"""
Training loop with all features for pretraining.

Features:
- Gradient accumulation
- Mixed precision (AMP) for CPU
- Cosine LR with warmup
- Gradient clipping
- Checkpointing
- Early stopping
- MFU estimation

Anti-overfitting measures for small datasets:
- dropout=0.1: Regularizes attention and FFN
- weight_decay=0.1: Penalizes large weights
- label_smoothing=0.1: Prevents overconfident predictions
"""

import os
import sys
import math
import time
import random
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import GPTConfig
from model.transformer import TinyGPT
from training.dataset import TextDataset, train_val_split
from tokenizer.train_tokenizer import Tokenizer
from utils.logging import TrainingLogger


class Trainer:
    """
    Trainer class for language model pretraining.
    """
    
    def __init__(
        self,
        model: TinyGPT,
        tokenizer: Tokenizer,
        train_dataset,
        val_dataset,
        config: GPTConfig,
        batch_size: int = 16,
        accum_steps: int = 4,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_ratio: float = 0.02,
        max_epochs: int = 20,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        """
        Initialize trainer.
        
        Args:
            model: TinyGPT model instance
            tokenizer: Tokenizer for encoding/decoding
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Model configuration
            batch_size: Per-micro-batch size
            accum_steps: Gradient accumulation steps
            lr: Peak learning rate
            weight_decay: Weight decay coefficient
            warmup_ratio: Fraction of training for warmup
            max_epochs: Maximum training epochs
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
        """
        # Multi-GPU setup
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.model = model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank]
            )
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Effective batch size
        self.effective_batch_size = batch_size * accum_steps
        
        # Data loaders
        use_pin = self.device.type == "cuda"
        train_sampler = None
        if self.local_rank != -1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=0,
            pin_memory=use_pin,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin,
        )
        
        # Training steps
        self.steps_per_epoch = len(train_dataset) // batch_size
        self.total_steps = self.steps_per_epoch * max_epochs
        self.warmup_steps = int(self.total_steps * warmup_ratio)
        
        # Optimizer
        self.optimizer = model.configure_optimizers(weight_decay, lr)
        
        # Learning rate scheduler (cosine with linear warmup)
        self.scheduler = self._create_scheduler()
        
        # Logger
        self.logger = TrainingLogger(log_dir=log_dir)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Print training config
        self._print_config()
    
    def _print_config(self):
        """Print training configuration."""
        print("\n" + "=" * 60)
        print("Training Configuration")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Accumulation steps: {self.accum_steps}")
        print(f"Effective batch size: {self.effective_batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"Total steps: {self.total_steps}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print("=" * 60 + "\n")
    
    def _create_scheduler(self):
        """Create cosine learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / max(1, self.warmup_steps)
            else:
                # Cosine decay
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def _estimate_mfu(self, elapsed_time: float, num_tokens: int) -> float:
        """
        Estimate Model Flops Utilization (MFU).
        
        MFU = actual tokens/sec / theoretical max tokens/sec
        This is a rough estimate for CPU training.
        
        Args:
            elapsed_time: Time in seconds
            num_tokens: Number of tokens processed
            
        Returns:
            MFU as a fraction (0 to 1)
        """
        # Rough estimate of FLOPs per token for forward + backward
        # For a transformer: ~6 * num_params FLOPs per token (forward + backward)
        num_params = self.model.get_num_params()
        flops_per_token = 6 * num_params
        
        # Actual FLOPs achieved
        actual_flops = flops_per_token * num_tokens / elapsed_time
        
        # Theoretical max for a modern CPU (~10 GFLOPS for single core)
        # This is very rough; actual varies greatly by CPU
        theoretical_flops = 10e9
        
        return actual_flops / theoretical_flops
    
    def train_step(self, batch: tuple) -> float:
        """
        Execute a single training step.
        
        Args:
            batch: Tuple of (input_ids, target_ids)
            
        Returns:
            Loss value
        """
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass with mixed precision
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
            logits, loss = self.model(input_ids, target_ids)
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.accum_steps
        
        # Backward pass
        scaled_loss.backward()
        
        return loss.item()
    
    def train(self):
        """
        Main training loop.
        """
        self.logger.log_message("Starting training...")
        
        self.model.train()
        accumulated_loss = 0.0
        step_tokens = 0
        
        for epoch in range(self.max_epochs):
            self.logger.log_message(f"\n=== Epoch {epoch + 1}/{self.max_epochs} ===")
            
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Timing
                self.logger.start_step()
                
                # Training step
                loss = self.train_step(batch)
                accumulated_loss += loss
                
                # Count tokens
                batch_tokens = batch[0].numel()
                step_tokens += batch_tokens
                self.logger.add_tokens(batch_tokens)
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accum_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Learning rate step
                    self.scheduler.step()
                    
                    # Global step
                    self.global_step += 1
                    
                    # Calculate tokens/sec
                    tokens_per_sec = self.logger.get_tokens_per_sec()
                    
                    # Logging
                    if self.global_step % 50 == 0:
                        avg_loss = accumulated_loss / self.accum_steps
                        self.logger.log(
                            step=self.global_step,
                            train_loss=avg_loss,
                            lr=self._get_lr(),
                            tokens_per_sec=tokens_per_sec,
                        )
                    
                    # Reset accumulators
                    accumulated_loss = 0.0
                    step_tokens = 0
                
                # Validation
                if self.global_step % 200 == 0 and self.global_step > 0:
                    val_loss = self.validate()
                    
                    self.logger.log(
                        step=self.global_step,
                        train_loss=accumulated_loss / self.accum_steps if accumulated_loss > 0 else 0,
                        val_loss=val_loss,
                    )
                    
                    # Checkpoint on improvement
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        self.save_checkpoint(is_best=True)
                        self.logger.log_message(f"New best val loss: {val_loss:.4f}")
                    else:
                        self.epochs_without_improvement += 1
                    
                    # Early stopping
                    if self.epochs_without_improvement >= 5:
                        self.logger.log_message("Early stopping triggered!")
                        self.save_checkpoint(is_best=False)
                        return
                    
                    self.model.train()
            
            # End of epoch
            epoch_time = time.time() - epoch_start
            self.logger.log_message(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        
        # Final checkpoint
        self.save_checkpoint(is_best=False)
        
        # Also save as model.pt for compatibility
        final_path = os.path.join(self.checkpoint_dir, "model.pt")
        latest_path = os.path.join(self.checkpoint_dir, "model_latest.pt")
        if os.path.exists(latest_path):
            import shutil
            shutil.copy(latest_path, final_path)
            self.logger.log_message(f"Saved final checkpoint to {final_path}")
        
        # Save config
        config_path = os.path.join(self.checkpoint_dir, "config.json")
        self.config.save(config_path)
        self.logger.log_message(f"Saved config to {config_path}")
        
        self.logger.log_message("Training completed!")
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation and return average loss.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
                _, loss = self.model(input_ids, target_ids)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Limit validation batches for speed
            if num_batches >= 50:
                break
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: If True, save as best model
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
        }
        
        # Save as best model
        if is_best:
            path = os.path.join(self.checkpoint_dir, "model.pt")
            torch.save(checkpoint, path)
            self.config.save(os.path.join(self.checkpoint_dir, "config.json"))
            self.logger.log_message(f"Saved best checkpoint to {path}")
        
        # Save as latest
        path = os.path.join(self.checkpoint_dir, "model_latest.pt")
        torch.save(checkpoint, path)


def main():
    """Main training function."""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Paths
    corpus_path = "data/corpus.md"
    tokenizer_path = "tokenizer/tokenizer.json"
    
    # Check prerequisites
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus not found at {corpus_path}")
        sys.exit(1)
    
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Run tokenizer/train_tokenizer.py first.")
        sys.exit(1)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer(tokenizer_path)
    
    # Create config
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    
    # Create model
    print("Creating model...")
    model = TinyGPT(config)
    
    # Create dataset
    print("Creating dataset...")
    dataset = TextDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        context_length=config.context_length,
    )
    
    # Split dataset
    train_dataset, val_dataset = train_val_split(dataset, val_fraction=0.1)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        batch_size=128,
        accum_steps=1,
        lr=5e-4,
        weight_decay=0.1,
        warmup_ratio=0.05,
        max_epochs=5,
    )
    
    # Train
    trainer.train()
    
    # Save final metrics
    trainer.logger.save_metrics()


if __name__ == "__main__":
    main()
