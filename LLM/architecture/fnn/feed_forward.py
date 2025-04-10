import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): the dimension of the model (input and output).
            d_ff (int): the dimension of the inner layer (typically 4*d_model).
            dropout (float): dropout rate applied after the activation.
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, d_model].
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        # First linear layer, activation, and dropout:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Second linear layer: project back to d_model
        x = self.linear2(x)
        return x


# Example usage:
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 128
    d_ff = 512

    dummy_input = torch.rand(batch_size, seq_len, d_model)
    ffn = FeedForward(d_model, d_ff, dropout=0.1)
    output = ffn(dummy_input)
    print("FFN output shape:", output.shape)
