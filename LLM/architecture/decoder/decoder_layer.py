import torch.nn as nn

from architecture.add_norm.add_norm_skip import AddNorm
from architecture.attention_heads.multi_head_attention import MultiHeadAttention
from architecture.fnn.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the FFN inner layer.
            dropout (float): Dropout rate.
        """
        super(DecoderLayer, self).__init__()
        # Masked self-attention block (decoder self-attention)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Encoder-decoder (cross) attention block
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout)
        # Add & Norm layers
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            x (Tensor): Decoder input embeddings of shape [batch_size, tgt_seq_len, d_model].
            encoder_output (Tensor): Encoder output representations with shape [batch_size, src_seq_len, d_model].
            tgt_mask (Tensor or None): Mask for the target sequence (e.g. to enforce causality).
            memory_mask (Tensor or None): Mask for the encoder output if needed.
        Returns:
            Tensor: Output of the decoder layer with shape [batch_size, tgt_seq_len, d_model].
        """
        # 1. Decoder masked self-attention with skip connection and normalization.
        # x attends to itself; use tgt_mask to prevent attending to future tokens.
        self_attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.add_norm1(x, self_attn_out)

        # 2. Encoder-decoder cross-attention: decoder queries attend over encoder output.
        cross_attn_out = self.cross_attn(x, encoder_output, encoder_output, mask=memory_mask)
        x = self.add_norm2(x, cross_attn_out)

        # 3. Feed-forward network and residual connection.
        ffn_out = self.ffn(x)
        x = self.add_norm3(x, ffn_out)

        return x
