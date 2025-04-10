import torch
import torch.nn as nn


class DeepSeekFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, use_gate=False, pre_norm=False):
        """
        Args:
            d_model (int): Model dimension (input and output).
            d_ff (int): Inner dimension (often 4 * d_model).
            dropout (float): Dropout rate.
            use_gate (bool): If True, apply a gated linear unit modification.
            pre_norm (bool): If True, apply layer normalization before the FFN.
        """
        super(DeepSeekFFN, self).__init__()
        self.use_gate = use_gate
        self.pre_norm = pre_norm

        # Optionally, apply layer normalization before the feed-forward network.
        if self.pre_norm:
            self.layer_norm = nn.LayerNorm(d_model)

        # First linear projection from d_model to d_ff.
        self.linear1 = nn.Linear(d_model, d_ff)
        # Activation: GELU is popular in modern architectures.
        self.gelu = nn.GELU()
        # Dropout after activation.
        self.dropout = nn.Dropout(dropout)

        if self.use_gate:
            # In a gated FFN, a second projection produces gates in the same d_ff dimension.
            self.gate = nn.Linear(d_model, d_ff)

        # Second linear projection back to d_model.
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input of shape [batch_size, seq_len, d_model].
        Returns:
            Tensor: Output of shape [batch_size, seq_len, d_model].
        """
        # Optionally apply pre-layer normalization.
        if self.pre_norm:
            x = self.layer_norm(x)

        # First projection.
        x_proj = self.linear1(x)
        x_proj = self.gelu(x_proj)
        x_proj = self.dropout(x_proj)

        # Optionally apply a gating mechanism.
        if self.use_gate:
            gate = torch.sigmoid(self.gate(x))
            x_proj = x_proj * gate

        # Second projection.
        output = self.linear2(x_proj)
        return output


# Example usage:
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 128
    d_ff = 512  # typically 4 * d_model

    dummy_input = torch.rand(batch_size, seq_len, d_model)

    # Instantiate the FFN similar to DeepSeek-3.
    ffn_deepseek = DeepSeekFFN(d_model, d_ff, dropout=0.1, use_gate=True, pre_norm=True)
    output = ffn_deepseek(dummy_input)
    print("DeepSeek-3 FFN output shape:", output.shape)  # Expected: [2, 10, 128]
