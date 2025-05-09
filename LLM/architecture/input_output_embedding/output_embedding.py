# output_embedding.py
import torch
import torch.nn as nn


class OutputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, tie_weights=False, input_embedding=None):
        """
        Args:
            d_model (int): Model (hidden) dimension of the decoder output.
            vocab_size (int): Number of tokens in the output vocabulary.
            tie_weights (bool): Whether to tie output proj weights to the input embedding's weight.
            input_embedding (nn.Embedding, optional): The embedding for tie-weights if tie_weights=True.
        """
        super().__init__()
        self.linear_out = nn.Linear(d_model, vocab_size)
        if tie_weights and input_embedding is not None:
            # Make sure input_embedding.embedding_dim == d_model
            self.linear_out.weight = input_embedding.weight

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        returns: [batch_size, seq_len, vocab_size]
        """
        return self.linear_out(x)


# Example usage:
if __name__ == "__main__":
    # Hyperparameters and dummy data dimensions.
    batch_size = 2
    seq_len = 10
    d_model = 128
    vocab_size = 10000

    # Dummy hidden state output from your model (e.g., a Transformer block).
    dummy_hidden = torch.rand(batch_size, seq_len, d_model)

    # Option 1: Use separate output embedding weights.
    output_embed = OutputEmbedding(d_model, vocab_size, tie_weights=False)
    logits = output_embed(dummy_hidden)
    print("Logits shape (separate weights):", logits.shape)  # Expected: [2, 10, 10000]

    # Option 2: Tie weights with an input embedding.
    # First, create an input embedding for demonstration (with the same d_model and vocab_size)
    input_embedding = nn.Embedding(vocab_size, d_model)
    # Now, create an output embedding that ties its weights to input_embedding.
    output_embed_tied = OutputEmbedding(d_model, vocab_size, tie_weights=True, input_embedding=input_embedding)
    logits_tied = output_embed_tied(dummy_hidden)
    print("Logits shape (tied weights):", logits_tied.shape)  # Expected: [2, 10, 10000]
