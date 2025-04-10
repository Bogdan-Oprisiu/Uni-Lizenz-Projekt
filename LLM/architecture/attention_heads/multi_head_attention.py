import torch
import torch.nn as nn

# Import your standard attention head from its file.
from architecture.attention_heads.standard_attention_head import StandardAttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # Ensure that d_model is divisible by num_heads.
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        # Linear layers to project (query, key, value) from d_model to d_model.
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # A single-head attention module for each head.
        self.attention_heads = nn.ModuleList([
            StandardAttentionHead(self.head_dim)
            for _ in range(num_heads)
        ])

        # Final projection after concatenating heads.
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, tgt_seq_len, d_model]
            key:   [batch_size, src_seq_len, d_model]
            value: [batch_size, src_seq_len, d_model]
            mask:  [batch_size, tgt_seq_len, src_seq_len], if provided
        Returns:
            Tensor of shape [batch_size, tgt_seq_len, d_model]
        """

        batch_size, tgt_seq_len, _ = query.size()
        _, src_seq_len, _ = key.size()

        # 1. Apply linear layers to project to d_model dimensions.
        q = self.linear_q(query)  # [batch_size, tgt_seq_len, d_model]
        k = self.linear_k(key)  # [batch_size, src_seq_len, d_model]
        v = self.linear_v(value)  # [batch_size, src_seq_len, d_model]

        # 2. Reshape into heads.
        #    We'll move the heads dimension to dim=0 so we can do q[i], k[i], v[i].
        #    After the first view, shape = [batch_size, tgt_seq_len, num_heads, head_dim].
        #    Then we permute to get [num_heads, batch_size, tgt_seq_len, head_dim].
        q = q.view(batch_size, tgt_seq_len, self.num_heads, self.head_dim)
        q = q.permute(2, 0, 1, 3)  # [num_heads, batch_size, tgt_seq_len, head_dim]

        k = k.view(batch_size, src_seq_len, self.num_heads, self.head_dim)
        k = k.permute(2, 0, 1, 3)  # [num_heads, batch_size, src_seq_len, head_dim]

        v = v.view(batch_size, src_seq_len, self.num_heads, self.head_dim)
        v = v.permute(2, 0, 1, 3)  # [num_heads, batch_size, src_seq_len, head_dim]

        # 3. Expand the mask if provided.
        #    Original mask: [batch_size, tgt_seq_len, src_seq_len].
        #    We want:       [num_heads, batch_size, tgt_seq_len, src_seq_len].
        if mask is not None:
            mask = mask.unsqueeze(0).expand(self.num_heads, -1, -1, -1)

        # 4. Run each head through StandardAttentionHead.
        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            # q[i] => shape [batch_size, tgt_seq_len, head_dim]
            # k[i], v[i] => shape [batch_size, src_seq_len, head_dim]
            head_mask = mask[i] if mask is not None else None  # => [batch_size, tgt_seq_len, src_seq_len]
            head_out = head(q[i], k[i], v[i], mask=head_mask)  # => [batch_size, tgt_seq_len, head_dim]
            head_outputs.append(head_out)

        # 5. Concatenate all heads back.
        #    head_outputs is a list of length num_heads, each [batch_size, tgt_seq_len, head_dim].
        #    So cat => [batch_size, tgt_seq_len, num_heads * head_dim] == [batch_size, tgt_seq_len, d_model].
        concatenated = torch.cat(head_outputs, dim=-1)

        # 6. Final linear projection and dropout.
        output = self.linear_out(concatenated)  # [batch_size, tgt_seq_len, d_model]
        output = self.dropout(output)
        return output


# --- Example usage:
if __name__ == "__main__":
    batch_size = 2
    tgt_seq_len = 5  # Target (decoder) sequence length
    src_seq_len = 7  # Source (encoder) sequence length
    d_model = 128
    num_heads = 8

    dummy_query = torch.rand(batch_size, tgt_seq_len, d_model)
    dummy_key = torch.rand(batch_size, src_seq_len, d_model)
    dummy_value = torch.rand(batch_size, src_seq_len, d_model)

    # Mask of shape [batch_size, tgt_seq_len, src_seq_len].
    mask = torch.ones(batch_size, tgt_seq_len, src_seq_len)
    # Example: mask out the last row in the target dimension.
    mask[:, -1, :] = 0

    mha = MultiHeadAttention(d_model, num_heads)
    output = mha(dummy_query, dummy_key, dummy_value, mask)
    print("Output shape:", output.shape)  # Expected: [2, 5, 128]
