import math

import torch
import torch.nn.functional as F

from architecture.attention_heads.base_attention_head import BaseAttentionHead


class StandardAttentionHead(BaseAttentionHead):
    def __init__(self, dim_head):
        super(StandardAttentionHead, self).__init__()
        self.dim_head = dim_head

    def forward(self, query, key, value, mask=None):
        # Compute the scaled dot-product attention scores.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_head)
        if mask is not None:
            # The mask is assumed to have shape [batch_size, tgt_seq_len, src_seq_len].
            # For any position where the mask is 0, replace the score with -infinity.
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # Compute the attention weights via softmax.
        attn = F.softmax(scores, dim=-1)
        # Replace any NaN values (which may occur when an entire row is masked) with zeros.
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        # Compute the weighted sum of the value vectors.
        output = torch.matmul(attn, value)
        return output


# --- Example usage:
if __name__ == "__main__":
    # Dummy BaseAttentionHead definition if isn't already defined:
    class BaseAttentionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, query, key, value, mask=None):
            raise NotImplementedError


    # Create dummy tensors for query, key, and value.
    batch_size = 2
    seq_length = 4
    dim_head = 64  # example dimension

    query = torch.rand(batch_size, seq_length, dim_head)
    key = torch.rand(batch_size, seq_length, dim_head)
    value = torch.rand(batch_size, seq_length, dim_head)

    # Create a dummy mask (for example, masking out the last token for each sample).
    mask = torch.ones(batch_size, seq_length, seq_length)
    mask[:, -1, :] = 0  # mask out the contributions for the last token in each query.

    attention_head = StandardAttentionHead(dim_head)
    output = attention_head(query, key, value, mask)
    print("Attention output shape:", output.shape)
    print(output)
