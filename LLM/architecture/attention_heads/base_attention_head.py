from abc import ABC, abstractmethod

import torch


class BaseAttentionHead(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, query, key, value, mask=None):
        """
        Computes the attention output.

        Args:
            query, key, value: Tensors of shape [batch, seq_len, dim_head]
            mask: Optional tensor for attention masking.
        Returns:
            Tensor of shape [batch, seq_len, dim_head]
        """
        pass
