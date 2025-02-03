"""
Trainer implementation for the LLM project.
Handles model training, evaluation, and checkpointing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import LLM

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for LLM model."""
    
    def __init__(
        self,
        model: LLM,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 10,
        warmup_steps: int = 1000,
        gradient_clip_val: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            model: LLM model instance
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Optional test data loader
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            max_epochs: Maximum number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            gradient_clip_val: Maximum gradient norm for clipping
            checkpoint_dir: Directory to save checkpoints
            device: Device to use for training
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps
        self.gradient_clip_val = gradient_clip_val
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
    
    def _setup_optimizer(self) -> AdamW:
        """Setup optimizer with weight decay."""
        # Separate parameters that should have weight decay from those that shouldn't
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
    
    def _setup_scheduler(self) -> CosineAnnealingLR:
        """Setup learning rate scheduler."""
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs * len(self.train_dataloader),
            eta_min=self.learning_rate / 10
        )
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["val_loss"]
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train_epoch(self) -> float:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        with tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}") as pbar:
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                _, loss, _ = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})
                
                self.global_step += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on given dataloader."""
        self.model.eval()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            _, loss, _ = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self) -> Dict[str, float]:
        """Train model for specified number of epochs."""
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Evaluate on validation set
            val_loss = self.evaluate(self.val_dataloader)
            val_losses.append(val_loss)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{self.max_epochs} - "
                f"Train loss: {train_loss:.4f} - "
                f"Val loss: {val_loss:.4f}"
            )
            
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        # Evaluate on test set if available
        test_loss = None
        if self.test_dataloader is not None:
            test_loss = self.evaluate(self.test_dataloader)
            logger.info(f"Test loss: {test_loss:.4f}")
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_loss": test_loss,
            "best_val_loss": best_val_loss
        }

def create_trainer(
    model: LLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: Optional[DataLoader] = None,
    **kwargs
) -> Trainer:
    """
    Create a trainer instance.
    
    Args:
        model: LLM model instance
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        test_dataloader: Optional test data loader
        **kwargs: Additional arguments to pass to Trainer
    
    Returns:
        Initialized Trainer instance
    """
    return Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        **kwargs
    )