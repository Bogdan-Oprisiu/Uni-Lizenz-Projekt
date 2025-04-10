import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1, max_len=5000, use_learned_pos=True):
        """
        Args:
            vocab_size (int): Number of tokens in the vocabulary.
            d_model (int): Dimension of the embedding space.
            dropout (float): Dropout rate to apply after adding positional encodings.
            max_len (int): Maximum length of the input sequences.
            use_learned_pos (bool): Whether to use learned positional encoding or sinusoidal.
        """
        super(InputEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Choose your positional encoding module.
        if use_learned_pos:
            from architecture.positional_encoding.learned_positional_encoding import LearnedPositionalEncoding
            self.positional_encoding = LearnedPositionalEncoding(d_model, max_len)
        else:
            from architecture.positional_encoding.regular_positional_encoding import RegularPositionalEncoding
            self.positional_encoding = RegularPositionalEncoding(d_model, dropout, max_len)

    def forward(self, x):
        """
        Args:
            x (Tensor): Token indices of shape [batch_size, seq_len].
        Returns:
            Tensor: Embeddings of shape [batch_size, seq_len, d_model] with positional encoding added.
        """
        # Get token embeddings. Shape: [batch_size, seq_len, d_model]
        embeddings = self.token_embedding(x)
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        return self.dropout(embeddings)


# Example usage:
if __name__ == "__main__":
    vocab_size = 10000
    d_model = 128
    batch_size = 2
    seq_len = 10

    # Create a dummy input of token indices.
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Initialize the input embedding module (choose learned positional encoding here).
    input_embed = InputEmbedding(vocab_size, d_model, dropout=0.1, max_len=5000, use_learned_pos=True)
    out = input_embed(dummy_input)
    print("Input embedding output shape:", out.shape)  # Expected: [2, 10, 128]
