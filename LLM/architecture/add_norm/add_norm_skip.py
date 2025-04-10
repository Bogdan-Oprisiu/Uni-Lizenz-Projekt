import torch
import torch.nn as nn


class AddNorm(nn.Module):
    """
    Custom module that implements a residual (skip) connection followed by layer normalization.
    This is often used in Transformer blocks to stabilize training.
    """

    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model (int): The dimension of the model (i.e., the number of features in the input).
            dropout (float): Dropout rate to apply before normalization.
        """
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        """
        Args:
            x (Tensor): The original input tensor, e.g. from a previous layer (shape: [batch_size, seq_len, d_model]).
            sublayer_output (Tensor): The output from a sublayer that is to be added to x (same shape as x).
        Returns:
            Tensor: The output tensor after applying the residual connection, dropout, and layer normalization.
        """
        # Apply dropout on the sublayer output, add the skip connection, and then normalize.
        return self.layer_norm(x + self.dropout(sublayer_output))


# --- Example usage ---
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 128

    # Dummy input tensor (e.g., from previous layer of a Transformer)
    dummy_input = torch.rand(batch_size, seq_len, d_model)
    # Dummy output from a sublayer (e.g., from multi-head attention or feed-forward network)
    dummy_sublayer_output = torch.rand(batch_size, seq_len, d_model)

    add_norm_layer = AddNorm(d_model, dropout=0.1)
    output = add_norm_layer(dummy_input, dummy_sublayer_output)

    print("Output shape:", output.shape)  # Expected: [2, 10, 128]
