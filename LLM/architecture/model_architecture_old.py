# model_architecture_old.py

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough P.E. matrix once.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class StudentTransformer(nn.Module):
    def __init__(self, vocabulary_size, d_model=128, num_head=4,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=128, dropout=0.1):
        """
        Initializes the student transformer.

        Args:
          vocabulary_size (int): Size of the vocabulary (from your custom tokenizer).
          d_model (int): Model (hidden) dimension.
          num_head (int): Number of attention heads.
          num_encoder_layers (int): Number of encoder layers.
          num_decoder_layers (int): Number of decoder layers.
          dim_feedforward (int): Dimension of the feed-forward network.
          dropout (float): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        # Embedding layer shared by encoder and decoder.
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(d_model=d_model, nhead=num_head,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        # Final projection layer to the vocabulary.
        self.fc_out = nn.Linear(d_model, vocabulary_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
          src (Tensor): Source sequence of shape [src_seq_len, batch].
          tgt (Tensor): Target sequence of shape [tgt_seq_len, batch].
          Additional masks are optional.

        Returns:
          output (Tensor): Logits over the vocabulary for each target token.
        """
        # Embed and add positional encoding.
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)

        # Pass through the transformer
        output = self.transformer(src_emb, tgt_emb,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        output = self.fc_out(output)
        return output


if __name__ == "__main__":
    # Example: assume a custom vocabulary size of 10,000
    vocab_size = 10000
    model = StudentTransformer(vocabulary_size=vocab_size)
    print(model)
