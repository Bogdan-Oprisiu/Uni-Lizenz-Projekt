import torch
import torch.nn as nn

from architecture.decoder.decoder import Decoder
from architecture.encoder.encoder import Encoder
from architecture.input_output_embedding.input_embedding import InputEmbedding


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
            tie_output (bool): Whether to tie the output projection weights to input embeddings.
            use_learned_pos (bool): Whether to use learned positional encoding.
        """
        super(DeepSeekTransformer, self).__init__()

        # Input Embedding: maps token IDs to continuous vectors and adds positional encodings.
        self.input_embedding = InputEmbedding(vocab_size, d_model, dropout=dropout,
                                              max_len=max_len, use_learned_pos=use_learned_pos)

        # Encoder: process source tokens.
        self.encoder = Encoder(num_layers_encoder, d_model, num_heads, d_ff,
                               dropout=dropout, max_len=max_len, use_learned_pos=use_learned_pos)

        # Decoder: generate output tokens based on encoder output.
        self.decoder = Decoder(num_layers_decoder, d_model, num_heads, d_ff, vocab_size,
                               dropout=dropout, tie_output=tie_output,
                               input_embedding=self.input_embedding.token_embedding if tie_output else None)

        # Optionally, you could incorporate a method to tie output projection and input embedding weights.
        # This has been handled inside the Decoder module above if tie_output is True.

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Args:
            src (Tensor): Source token IDs of shape [batch_size, src_seq_len].
            tgt (Tensor): Target token IDs of shape [batch_size, tgt_seq_len].
            src_mask (Tensor, optional): Mask for the encoder [batch_size, src_seq_len, src_seq_len].
            tgt_mask (Tensor, optional): Mask for the decoder (e.g. for causality)
                                          [batch_size, tgt_seq_len, tgt_seq_len].
            memory_mask (Tensor, optional): Optional mask over the encoder outputs
                                            [batch_size, tgt_seq_len, src_seq_len].
        Returns:
            Tensor: Logits over the vocabulary of shape [batch_size, tgt_seq_len, vocab_size].
        """
        # Embed source tokens with positional encoding.
        encoder_embeddings = self.input_embedding(src)  # shape: [batch_size, src_seq_len, d_model]
        # Pass through the encoder.
        encoder_output = self.encoder(encoder_embeddings, src_mask)  # shape: [batch_size, src_seq_len, d_model]

        # Embed target tokens (decoder input); this also adds the same positional encoding.
        decoder_embeddings = self.input_embedding(tgt)  # shape: [batch_size, tgt_seq_len, d_model]
        # Process the embeddings with the decoder, attending to encoder outputs.
        logits = self.decoder(decoder_embeddings, encoder_output, tgt_mask, memory_mask)

        return logits


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Hyperparameters:
    batch_size = 2
    src_seq_len = 12  # Length of source (encoder) sequence
    tgt_seq_len = 10  # Length of a target (decoder) sequence
    vocab_size = 10000
    d_model = 128
    num_layers_encoder = 6
    num_layers_decoder = 6
    num_heads = 8
    d_ff = 512  # Typically 4 * d_model
    dropout = 0.1
    max_len = 5000
    tie_output = True
    use_learned_pos = False  # Set to True if you prefer learned positional encoding

    # Dummy input token IDs for source and target (typically produced by your tokenizer).
    src_tokens = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))

    # Optionally, define masks (for instance, to handle padding or enforce causality in the decoder).
    src_mask = None
    tgt_mask = None
    memory_mask = None

    # Instantiate the model.
    model = DeepSeekTransformer(vocab_size, d_model, num_layers_encoder, num_layers_decoder,
                                num_heads, d_ff, dropout, max_len, tie_output, use_learned_pos)

    # Forward pass.
    output_logits = model(src_tokens, tgt_tokens, src_mask, tgt_mask, memory_mask)
    print("Model output shape:", output_logits.shape)
    # Expected shape: [batch_size, tgt_seq_len, vocab_size]
