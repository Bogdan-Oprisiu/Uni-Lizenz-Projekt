import torch
import torch.nn as nn

from architecture.encoder.encoder_layer import EncoderLayer
from architecture.positional_encoding.regular_positional_encoding import RegularPositionalEncoding


# -------------------------------
# Encoder Implementation
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1, max_len=5000, use_learned_pos=False):
        """
        Args:
            num_layers (int): Number of encoder layers.
            d_model (int): Model dimensionality.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feed-forward layer.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length for positional encoding.
            use_learned_pos (bool): Whether to use a learned positional encoding instead of sinusoidal.
        """
        super(Encoder, self).__init__()
        # Positional encoding module.
        if use_learned_pos:
            from architecture.positional_encoding.learned_positional_encoding import LearnedPositionalEncoding
            self.pos_enc = LearnedPositionalEncoding(d_model, max_len=max_len)
        else:
            self.pos_enc = RegularPositionalEncoding(d_model, dropout, max_len)
        # Stack of encoder layers.
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # Final normalization (optional, but typical in many implementations).
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input embeddings of shape [batch_size, seq_len, d_model].
            mask (Tensor, optional): Mask for self-attention (shape [batch_size, seq_len, seq_len]).
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Add positional encoding.
        x = self.pos_enc(x)
        # Pass through the stack of encoder layers.
        for layer in self.layers:
            x = layer(x, mask)
        # Apply final normalization.
        x = self.layer_norm(x)
        return x


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_layers = 6
    num_heads = 8
    d_ff = 512  # common choice is 4 * d_model
    dropout = 0.1
    max_len = 5000

    # Dummy input embeddings (this would normally come from your input embedding + positional encoding)
    dummy_input = torch.rand(batch_size, seq_len, d_model)
    # Dummy mask (optional); if provided, shape should be [batch_size, seq_len, seq_len].
    # For example, you might mask out padding tokens.
    mask = None

    # Instantiate the encoder.
    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout, max_len, use_learned_pos=False)
    output = encoder(dummy_input, mask)
    print("Encoder output shape:", output.shape)  # Expected: [batch_size, seq_len, d_model]
