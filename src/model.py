"""
Model implementation for the LLM project.
Implements a decoder-only Transformer architecture for language modeling.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config):
        """
        Initialize multi-head attention.
        
        Args:
            config: Model configuration containing:
                - hidden_size: Size of hidden states
                - num_attention_heads: Number of attention heads
                - attention_dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.dropout = config.attention_dropout
        
        # Ensure hidden size is divisible by number of heads
        assert self.head_size * self.num_heads == self.hidden_size, \
            "Hidden size must be divisible by number of heads"
        
        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask tensor of shape (batch_size, seq_len)
            causal_mask: Whether to apply causal masking
        
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, hidden_size)
                - Attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project inputs to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to (batch_size, num_heads, seq_length, head_size)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        
        # Apply causal mask if requested
        if causal_mask:
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, dtype=torch.bool, device=hidden_states.device),
                diagonal=1
            )
            attention_scores.masked_fill_(causal_mask, float("-inf"))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(~attention_mask, float("-inf"))
        
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute output
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.hidden_size)
        
        # Project output
        output = self.out_proj(context)
        
        return output, attention_probs

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config):
        """
        Initialize feed-forward network.
        
        Args:
            config: Model configuration containing:
                - hidden_size: Size of hidden states
                - intermediate_size: Size of intermediate layer
                - hidden_dropout: Dropout probability
        """
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.activation = F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feed-forward network."""
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer decoder block."""
    
    def __init__(self, config):
        """
        Initialize transformer block.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for transformer block."""
        # Self-attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attention_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, attention_weights

class LLMConfig:
    """Configuration class for LLM model."""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_position_embeddings: int = 512
    ):
        """Initialize configuration."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings

class LLM(nn.Module):
    """Main LLM model implementation."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )
        
        # Tie weights between token embeddings and output projection
        self.output_projection.weight = self.token_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the model."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for LLM model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for computing loss
        
        Returns:
            Tuple of:
                - Logits tensor of shape (batch_size, seq_len, vocab_size)
                - Loss tensor if labels provided, else None
                - Last layer attention weights
        """
        batch_size, seq_length = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Process through transformer layers
        attention_weights = None
        for layer in self.layers:
            hidden_states, attention_weights = layer(
                hidden_states,
                attention_mask=attention_mask
            )
        
        # Apply final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Get logits
        logits = self.output_projection(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss, attention_weights

def create_model(
    vocab_size: int,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12
) -> LLM:
    """
    Create an instance of the LLM model.
    
    Args:
        vocab_size: Size of vocabulary
        hidden_size: Size of hidden states
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
    
    Returns:
        Initialized LLM model
    """
    config = LLMConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads
    )
    return LLM(config)
