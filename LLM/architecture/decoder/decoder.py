import torch
import torch.nn as nn

from architecture.decoder.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 dropout=0.1, tie_output=False, input_embedding=None):
        """
        Args:
            num_layers (int): Number of decoder layers.
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate.
            tie_output (bool): Whether to tie weights (not used here anymore).
            input_embedding (nn.Embedding, optional): For weight tying if relevant (not used here).
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        # Note: No final projection to vocab_size—this is handled by OutputEmbedding.

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            x (Tensor): [batch_size, tgt_seq_len, d_model] (decoder embeddings)
            encoder_output (Tensor): [batch_size, src_seq_len, d_model]
            tgt_mask (Tensor): [batch_size, tgt_seq_len, tgt_seq_len], optional
            memory_mask (Tensor): [batch_size, tgt_seq_len, src_seq_len], optional
        Returns:
            Tensor: Decoder hidden states [batch_size, tgt_seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)
        x = self.layer_norm(x)
        return x


# --- Example usage:
if __name__ == "__main__":
    batch_size = 2
    tgt_seq_len = 10
    src_seq_len = 12
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 6

    decoder_input = torch.rand(batch_size, tgt_seq_len, d_model)  # [2, 10, 128]
    encoder_output = torch.rand(batch_size, src_seq_len, d_model)  # [2, 12, 128]

    tgt_mask = None
    memory_mask = None

    decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout=0.1)
    hidden_states = decoder(decoder_input, encoder_output, tgt_mask, memory_mask)
    print("Decoder output shape:", hidden_states.shape)
    # Expected: [2, tgt_seq_len, d_model]
