import torch
from torch import nn

from architecture.decoder.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, dropout=0.1, tie_output=False,
                 input_embedding=None):
        """
        Args:
            num_layers (int): Number of decoder layers.
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward network.
            vocab_size (int): Size of the output vocabulary.
            dropout (float): Dropout rate.
            tie_output (bool): Whether to tie the output projection weights with the input embedding.
            input_embedding (nn.Embedding, optional): Input embedding module for weight tying.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        # Final linear projection to output vocabulary size.
        self.output_projection = nn.Linear(d_model, vocab_size)
        if tie_output and input_embedding is not None:
            # Tie weights with input embedding.
            self.output_projection.weight = input_embedding.weight

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            x (Tensor): Target input embeddings [batch_size, tgt_seq_len, d_model].
            encoder_output (Tensor): Encoder outputs [batch_size, src_seq_len, d_model].
            tgt_mask (Tensor, optional): Mask for target sequence.
            memory_mask (Tensor, optional): Mask for encoder outputs.
        Returns:
            logits (Tensor): Logits over vocabulary [batch_size, tgt_seq_len, vocab_size].
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        return logits


# --- Example usage:
if __name__ == "__main__":
    batch_size = 2
    tgt_seq_len = 10
    src_seq_len = 12
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 6
    vocab_size = 10000

    # Dummy target embeddings (this would come from your input embedding + positional encoding)
    decoder_input = torch.rand(batch_size, tgt_seq_len, d_model)
    # Dummy encoder output (from your encoder module)
    encoder_output = torch.rand(batch_size, src_seq_len, d_model)

    # Dummy masks (for example, a causal mask for the target and possibly one for encoder output)
    tgt_mask = None  # You can create a causal mask here if needed.
    memory_mask = None

    # Instantiate the decoder.
    decoder = Decoder(num_layers, d_model, num_heads, d_ff, vocab_size, dropout=0.1)
    logits = decoder(decoder_input, encoder_output, tgt_mask, memory_mask)
    print("Decoder output logits shape:", logits.shape)  # Expected: [2, tgt_seq_len, vocab_size]
