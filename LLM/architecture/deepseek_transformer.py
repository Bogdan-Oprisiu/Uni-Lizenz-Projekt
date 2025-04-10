import torch
import torch.nn as nn

from architecture.decoder.decoder import Decoder
from architecture.encoder.encoder import Encoder
from architecture.input_output_embedding.input_embedding import InputEmbedding
from architecture.input_output_embedding.output_embedding import OutputEmbedding


class DeepSeekTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers_encoder, num_layers_decoder,
                 num_heads, d_ff, dropout=0.1, max_len=5000, tie_output=False, use_learned_pos=False):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of token embeddings and hidden states.
            num_layers_encoder (int): Number of layers in the encoder.
            num_layers_decoder (int): Number of layers in the decoder.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the inner FFN (often 4 * d_model).
            dropout (float): Dropout probability.
            max_len (int): Maximum sequence length for positional encoding.
            tie_output (bool): Whether to tie the output projection weights to the input token embeddings.
            use_learned_pos (bool): Whether to use learned positional encoding.
        """
        super(DeepSeekTransformer, self).__init__()

        # 1) Input Embedding
        self.input_embedding = InputEmbedding(
            vocab_size, d_model, dropout=dropout,
            max_len=max_len, use_learned_pos=use_learned_pos
        )

        # 2) Output Embedding
        # Projects decoder hidden states [batch, seq, d_model] to logits [batch, seq, vocab_size].
        self.output_embedding = OutputEmbedding(
            d_model, vocab_size, tie_weights=tie_output,
            input_embedding=self.input_embedding.token_embedding if tie_output else None
        )

        # 3) Encoder
        self.encoder = Encoder(
            num_layers_encoder, d_model, num_heads, d_ff,
            dropout=dropout, max_len=max_len, use_learned_pos=use_learned_pos
        )

        # 4) Decoder
        # Note: No vocab_size param here! The Decoder returns hidden states of shape [batch, seq, d_model].
        self.decoder = Decoder(
            num_layers_decoder, d_model, num_heads, d_ff,
            dropout=dropout, tie_output=tie_output,
            input_embedding=self.input_embedding.token_embedding if tie_output else None
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Args:
            src (Tensor): Source token IDs of shape [batch_size, src_seq_len].
            tgt (Tensor): Target token IDs of shape [batch_size, tgt_seq_len].
            src_mask (Tensor): Optional mask for encoder self-attention ([batch_size, src_seq_len, src_seq_len]).
            tgt_mask (Tensor): Optional mask for decoder self-attention ([batch_size, tgt_seq_len, tgt_seq_len]).
            memory_mask (Tensor): Optional mask for cross-attention ([batch_size, tgt_seq_len, src_seq_len]).
        Returns:
            Tensor: Logits [batch_size, tgt_seq_len, vocab_size].
        """
        # 1) Embed and encode source
        encoder_embeddings = self.input_embedding(src)  # [batch, src_seq_len, d_model]
        encoder_output = self.encoder(encoder_embeddings, src_mask)  # [batch, src_seq_len, d_model]

        # 2) Embed and decode target
        decoder_embeddings = self.input_embedding(tgt)  # [batch, tgt_seq_len, d_model]
        decoder_output = self.decoder(decoder_embeddings, encoder_output, tgt_mask, memory_mask)
        # decoder_output: [batch, tgt_seq_len, d_model]

        # 3) Project to vocab logits
        logits = self.output_embedding(decoder_output)  # [batch, tgt_seq_len, vocab_size]
        return logits


# --- Example Usage ---
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 2
    src_seq_len = 12
    tgt_seq_len = 10
    vocab_size = 10000
    d_model = 128
    num_layers_encoder = 6
    num_layers_decoder = 6
    num_heads = 8
    d_ff = 512
    dropout = 0.1
    max_len = 5000
    tie_output = True
    use_learned_pos = False

    # Dummy source & target tokens
    src_tokens = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))

    # Optional masks
    src_mask = None
    tgt_mask = None
    memory_mask = None

    model = DeepSeekTransformer(
        vocab_size, d_model, num_layers_encoder, num_layers_decoder,
        num_heads, d_ff, dropout, max_len, tie_output, use_learned_pos
    )

    output_logits = model(src_tokens, tgt_tokens, src_mask, tgt_mask, memory_mask)
    print("Model output shape:", output_logits.shape)
    # Expected: [2, 10, 10000]
