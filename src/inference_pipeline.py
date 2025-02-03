"""
Inference pipeline for the LLM project.
Handles text generation using various sampling strategies.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from .model import LLM

class InferencePipeline:
    """Pipeline for generating text using trained LLM."""
    
    def __init__(
        self,
        model: LLM,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained LLM model
            tokenizer: Trained tokenizer for encoding/decoding text
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        # Cache special token IDs
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.eos_token_id = tokenizer.token_to_id("[SEP]")  # Using [SEP] as EOS
        self.bos_token_id = tokenizer.token_to_id("[CLS]")  # Using [CLS] as BOS
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 0,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Generate text based on prompt.
        
        Args:
            prompt: Input text to condition generation on
            max_length: Maximum length of generated sequence
            min_length: Minimum length of generated sequence
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep for sampling
            top_p: Cumulative probability for nucleus sampling
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            repetition_penalty: Penalty for repeating tokens
            pad_token_id: ID of padding token
            eos_token_id: ID of end-of-sequence token
            batch_size: Batch size for parallel generation
        
        Returns:
            List of generated text sequences
        """
        # Set default token IDs if not provided
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.eos_token_id
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt).ids
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        input_ids = input_ids.unsqueeze(0).repeat(num_return_sequences, 1)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Generate sequences
        output_sequences = self._generate_sequences(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            do_sample=do_sample,
            batch_size=batch_size
        )
        
        # Decode generated sequences
        generated_sequences = []
        for sequence in output_sequences:
            # Remove input prompt
            sequence = sequence[len(input_ids[0]):]
            # Remove padding and end tokens
            sequence = sequence[sequence != pad_token_id]
            sequence = sequence[sequence != eos_token_id]
            # Decode to text
            text = self.tokenizer.decode(sequence.tolist())
            generated_sequences.append(text)
        
        return generated_sequences
    
    def _generate_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        min_length: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        pad_token_id: int,
        eos_token_id: int,
        do_sample: bool,
        batch_size: Optional[int]
    ) -> torch.Tensor:
        """Generate sequences using specified decoding strategy."""
        batch_size = batch_size or input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # Create output sequences tensor
        output_sequences = input_ids.clone()
        
        # Create tensor to track which sequences are finished
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=self.device
        )
        
        while cur_len < max_length:
            # Prepare model inputs
            model_inputs = {
                "input_ids": output_sequences,
                "attention_mask": attention_mask
            }
            
            # Get next token logits
            outputs = self.model(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(output_sequences[i].tolist()):
                        next_token_logits[i, previous_token] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Sample next tokens
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Finished sequences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # Add generated tokens to output
            output_sequences = torch.cat([output_sequences, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=self.device)
            ], dim=1)
            
            # Update which sequences are finished
            if cur_len + 1 < min_length:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.ne(eos_token_id).long()
                )
            
            # Stop when all sequences are finished
            if unfinished_sequences.max() == 0:
                break
            
            cur_len = cur_len + 1
        
        return output_sequences
    
    @staticmethod
    def _top_k_filtering(
        logits: torch.Tensor,
        top_k: int,
        filter_value: float = -float("Inf")
    ) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        top_k = min(top_k, logits.size(-1))
        
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
        return logits
    
    @staticmethod
    def _top_p_filtering(
        logits: torch.Tensor,
        top_p: float,
        filter_value: float = -float("Inf")
    ) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
        
        return logits

def create_inference_pipeline(
    model: LLM,
    tokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> InferencePipeline:
    """
    Create an inference pipeline instance.
    
    Args:
        model: Trained LLM model
        tokenizer: Trained tokenizer
        device: Device to run inference on
    
    Returns:
        Initialized InferencePipeline instance
    """
    return InferencePipeline(model, tokenizer, device)