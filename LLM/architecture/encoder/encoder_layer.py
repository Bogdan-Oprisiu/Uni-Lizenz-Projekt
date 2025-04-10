import torch
import torch.nn as nn
from architecture.fnn.feed_forward import FeedForward

from architecture.add_norm.add_norm_skip import AddNorm
from architecture.attention_heads.multi_head_attention import MultiHeadAttention
from architecture.positional_encoding.regular_positional_encoding import RegularPositionalEncoding

# -------------------------------
# Encoder Layer Implementation
# -------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): Model dimensionality.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feed-forward hidden layer (often 4*d_model).
            dropout (float): Dropout rate.
        """
        super(EncoderLayer, self).__init__()
        # Multi-head self-attention block.
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Add & Norm block for self-attention.
        self.add_norm1 = AddNorm(d_model, dropout)
        # Feed-forward network.
        self.ffn = FeedForward(d_model, d_ff, dropout)
        # Add & Norm block for the FFN.
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            mask (Tensor, optional): Mask for self-attention (shape [batch_size, seq_len, seq_len]).
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Self-attention sublayer (with residual connection).
        attn_output = self.self_attn(x, x, x, mask)
        x = self.add_norm1(x, attn_output)
        # Feed-forward sublayer (with residual connection).
        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)
        return x
