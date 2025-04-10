import math

import torch
import torch.nn as nn


class RegularPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model (int): the dimension of the model.
            dropout (float): dropout rate to apply after adding positional encodings.
            max_len (int): maximum sequence length.
        """
        super(RegularPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough PEs and register them as buffers (non-trainable).
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape becomes [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input embeddings of shape [batch_size, seq_len, d_model].
        Returns:
            Tensor: embeddings with positional encodings added.
        """
        seq_len = x.size(1)
        # Add the positional encoding (broadcasting along batch dimension)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# Example usage:
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 128
    dummy_embeddings = torch.rand(batch_size, seq_len, d_model)

    pos_encoder = RegularPositionalEncoding(d_model, dropout=0.1, max_len=5000)
    encoded = pos_encoder(dummy_embeddings)
    print("Positional Encoding applied, output shape:", encoded.shape)
