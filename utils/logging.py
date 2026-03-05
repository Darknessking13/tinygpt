"""
Logging utilities for training.

Provides TrainingLogger for tracking loss, learning rate, and timing.
Also includes loss curve plotting functionality.
"""

import os
import time
import json
from datetime import datetime


class TrainingLogger:
    """
    Logger that writes to both stdout and a log file.
    
    Tracks training metrics and saves them for plotting.
    """
    
    def __init__(self, log_dir: str = "logs", log_file: str = "training.log"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files
            log_file: Log filename
        """
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        
        # Create log directory if needed
        os.makedirs(log_dir, exist_ok=True)
        
        # Clear log file
        with open(self.log_file, "w") as f:
            f.write(f"Training Log - {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n")
        
        # Metrics storage
        self.metrics = {
            "steps": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "tokens_per_sec": [],
        }
        
        # Timing
        self.step_start_time = None
        self.tokens_processed = 0
    
    def start_step(self):
        """Mark the start of a step for timing."""
        self.step_start_time = time.time()
        self.tokens_processed = 0
    
    def add_tokens(self, num_tokens: int):
        """Add to token count for timing calculation."""
        self.tokens_processed += num_tokens
    
    def log(
        self,
        step: int,
        train_loss: float,
        val_loss: float = None,
        lr: float = None,
        tokens_per_sec: float = None,
    ):
        """
        Log training metrics.
        
        Args:
            step: Current training step
            train_loss: Training loss value
            val_loss: Validation loss value (optional)
            lr: Current learning rate (optional)
            tokens_per_sec: Tokens processed per second (optional)
        """
        # Store metrics
        self.metrics["steps"].append(step)
        self.metrics["train_loss"].append(train_loss)
        
        if val_loss is not None:
            self.metrics["val_loss"].append(val_loss)
        if lr is not None:
            self.metrics["learning_rate"].append(lr)
        if tokens_per_sec is not None:
            self.metrics["tokens_per_sec"].append(tokens_per_sec)
        
        # Build log message
        parts = [f"step {step:06d}", f"loss {train_loss:.4f}"]
        
        if val_loss is not None:
            parts.append(f"val_loss {val_loss:.4f}")
        if lr is not None:
            parts.append(f"lr {lr:.2e}")
        if tokens_per_sec is not None:
            parts.append(f"tokens/sec {tokens_per_sec:.0f}")
        
        message = " | ".join(parts)
        
        # Print to stdout
        print(f"[{message}]")
        
        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
    
    def log_message(self, message: str):
        """Log a plain message."""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
    
    def save_metrics(self, path: str = None):
        """Save metrics to JSON file for plotting."""
        if path is None:
            path = os.path.join(self.log_dir, "metrics.json")
        
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_tokens_per_sec(self) -> float:
        """Calculate tokens per second since start_step was called."""
        if self.step_start_time is None:
            return 0.0
        
        elapsed = time.time() - self.step_start_time
        if elapsed == 0:
            return 0.0
        
        return self.tokens_processed / elapsed


def plot_losses(log_file: str, output_path: str = None):
    """
    Plot training and validation loss curves.
    
    Args:
        log_file: Path to metrics JSON file
        output_path: Output path for plot image (default: logs/loss_curve.png)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot generation")
        return
    
    # Load metrics
    with open(log_file, "r") as f:
        metrics = json.load(f)
    
    if not metrics["steps"]:
        print("No metrics to plot")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    steps = metrics["steps"]
    ax1.plot(steps, metrics["train_loss"], label="Train Loss", color="blue")
    
    if metrics["val_loss"]:
        # Interpolate val_loss to match steps (val_loss is logged less frequently)
        val_steps = steps[::len(steps) // len(metrics["val_loss"]) + 1][:len(metrics["val_loss"])]
        ax1.plot(val_steps, metrics["val_loss"], label="Val Loss", color="red", marker="o")
    
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate
    if metrics["learning_rate"]:
        ax2.plot(steps, metrics["learning_rate"], label="Learning Rate", color="green")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        output_path = os.path.join(os.path.dirname(log_file), "loss_curve.png")
    
    plt.savefig(output_path, dpi=150)
    print(f"Loss curve saved to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    import random
    
    print("=== Testing TrainingLogger ===\n")
    
    # Create logger
    logger = TrainingLogger(log_dir="logs")
    
    # Simulate training
    logger.log_message("Starting training...")
    
    for step in range(0, 1001, 50):
        train_loss = 4.0 * (0.99 ** step) + random.uniform(-0.1, 0.1)
        val_loss = train_loss + random.uniform(0.1, 0.3) if step % 200 == 0 else None
        lr = 3e-4 * (1 - step / 1000) if step < 500 else 3e-4 * (0.1 ** ((step - 500) / 500))
        tokens_per_sec = random.uniform(5000, 6000)
        
        logger.log(
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr,
            tokens_per_sec=tokens_per_sec,
        )
    
    logger.log_message("Training complete!")
    
    # Save metrics
    logger.save_metrics()
    
    # Plot losses
    print("\n=== Generating Loss Curve ===")
    plot_losses("logs/metrics.json")
    
    print("\nLogger test complete!")
