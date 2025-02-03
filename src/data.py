"""
Data preprocessing module for the LLM project.
Handles downloading, preprocessing, and creating datasets from WikiText-2.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tqdm import tqdm

class WikiTextDataset(Dataset):
    """PyTorch Dataset for WikiText-2 data."""
    
    def __init__(
        self,
        data: List[str],
        tokenizer: Tokenizer,
        seq_length: int = 512,
        stride: int = 256
    ):
        """
        Initialize WikiText dataset.
        
        Args:
            data: List of text samples
            tokenizer: Trained tokenizer for encoding text
            seq_length: Maximum sequence length
            stride: Stride for sliding window
        """
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        
        # Encode all text and create sliding windows
        self.encoded_samples = self._prepare_samples()
        
    def _prepare_samples(self) -> List[Dict[str, torch.Tensor]]:
        """Prepare sliding window samples from text data."""
        samples = []
        
        for text in tqdm(self.data, desc="Preparing samples"):
            # Encode text
            encoding = self.tokenizer.encode(text)
            input_ids = encoding.ids
            
            # Create sliding windows
            for i in range(0, len(input_ids) - self.seq_length + 1, self.stride):
                input_slice = input_ids[i:i + self.seq_length]
                target_slice = input_ids[i + 1:i + self.seq_length + 1]
                
                # Pad if necessary
                if len(input_slice) < self.seq_length:
                    input_slice = input_slice + [self.tokenizer.token_to_id("[PAD]")] * (self.seq_length - len(input_slice))
                    target_slice = target_slice + [self.tokenizer.token_to_id("[PAD]")] * (self.seq_length - len(target_slice))
                
                samples.append({
                    "input_ids": torch.tensor(input_slice),
                    "labels": torch.tensor(target_slice),
                    "attention_mask": torch.ones(self.seq_length)
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.encoded_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encoded_samples[idx]

class DataModule:
    """Data module for handling all data-related operations."""
    
    def __init__(
        self,
        seq_length: int = 512,
        stride: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        vocab_size: int = 30000
    ):
        """
        Initialize data module.
        
        Args:
            seq_length: Maximum sequence length
            stride: Stride for sliding window
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            vocab_size: Size of vocabulary for tokenizer
        """
        self.seq_length = seq_length
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = vocab_size
        
        self.tokenizer: Optional[Tokenizer] = None
        self.train_dataset: Optional[WikiTextDataset] = None
        self.val_dataset: Optional[WikiTextDataset] = None
        self.test_dataset: Optional[WikiTextDataset] = None
    
    def prepare_data(self):
        """Download and prepare the WikiText-2 dataset."""
        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        # Train tokenizer if not exists
        if not self.tokenizer:
            self.tokenizer = self._train_tokenizer(dataset["train"]["text"])
        
        # Create datasets
        self.train_dataset = WikiTextDataset(
            dataset["train"]["text"],
            self.tokenizer,
            self.seq_length,
            self.stride
        )
        
        self.val_dataset = WikiTextDataset(
            dataset["validation"]["text"],
            self.tokenizer,
            self.seq_length,
            self.stride
        )
        
        self.test_dataset = WikiTextDataset(
            dataset["test"]["text"],
            self.tokenizer,
            self.seq_length,
            self.stride
        )
    
    def _train_tokenizer(self, texts: List[str]) -> Tokenizer:
        """Train a BPE tokenizer on the dataset."""
        # Initialize tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # Prepare trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        
        # Train tokenizer
        tokenizer.train_from_iterator(texts, trainer=trainer)
        
        return tokenizer
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

def main():
    """Main function for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 dataset")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset")
    args = parser.parse_args()
    
    if args.download or args.prepare:
        data_module = DataModule()
        data_module.prepare_data()
        print("Dataset preparation completed!")

if __name__ == "__main__":
    main()