import torch
from torch import nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): the dimension of the model.
            max_len (int): maximum sequence length.
        """
        super(LearnedPositionalEncoding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input embeddings of shape [batch_size, seq_len, d_model].
        Returns:
            Tensor: embeddings with learned positional encodings added.
        """
        batch_size, seq_len, _ = x.size()
        # Create a tensor of positions [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return x + self.pos_embed(positions)


# Example usage:
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 128
    dummy_embeddings = torch.rand(batch_size, seq_len, d_model)

    learned_pos_encoder = LearnedPositionalEncoding(d_model, max_len=5000)
    encoded = learned_pos_encoder(dummy_embeddings)
    print("Learned Positional Encoding applied, output shape:", encoded.shape)
