import math

import torch
import torch.nn.functional as F


# Dummy BaseAttentionHead for testing purposes
class BaseAttentionHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None):
        raise NotImplementedError


# Your StandardAttentionHead implementation
class StandardAttentionHead(BaseAttentionHead):
    def __init__(self, dim_head):
        super().__init__()
        self.dim_head = dim_head

    def forward(self, query, key, value, mask=None):
        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, value)


# Testing the attention head with random inputs
def test_attention_head():
    batch_size = 2
    seq_length = 4
    dim_head = 64  # for example

    # Create random tensors for a query, key, and value
    query = torch.rand(batch_size, seq_length, dim_head)
    key = torch.rand(batch_size, seq_length, dim_head)
    value = torch.rand(batch_size, seq_length, dim_head)

    # Optionally, create a dummy mask (for example, mask out the last token)
    mask = torch.ones(batch_size, seq_length, seq_length)
    mask[:, -1, :] = 0  # Masking out contributions from the last position

    attention_head = StandardAttentionHead(dim_head)
    output = attention_head(query, key, value, mask)
    print("Attention output shape:", output.shape)
    print(output)


if __name__ == "__main__":
    test_attention_head()
