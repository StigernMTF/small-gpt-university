import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch import Tensor


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, pad_idx: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model should be divisible by h"
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        query_matrix = self.W_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key_matrix = self.W_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value_matrix = self.W_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        attention_scores = (query_matrix @ key_matrix.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)

        attention_output = attention_weights @ value_matrix
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(attention_output)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.linear2(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            pad_idx: int,
            d_model: int = 512,
            num_heads: int = 8,
            d_ff: int = 2048,
            num_layers: int = 6,
            max_len: int = 128,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.embedding = InputEmbedding(d_model, vocab_size, pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        # 1. Mask adaptation
        if src_mask is not None and src_mask.dim() == 2:
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)

        # 2. Embeddings (scaling is already done inside)
        x = self.embedding(src)

        # 3. Positional encoding (adding positions and dropout is already done inside)
        x = self.pos_encoding(x)

        # 4. Run through layers
        for layer in self.layers:
            x = layer(x, src_mask)

        return x


if __name__ == "__main__":
    vocab_size = 58101
    pad_idx = 58100
    max_len = 128

    model = Encoder(vocab_size=vocab_size, pad_idx=pad_idx, max_len=max_len)

    batch_size = 2
    seq_len = 10
    src = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Simulate attention_mask
    src_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])

    output = model(src, src_mask)
    print(f"Input shape:  {tuple(src.shape)}")
    print(f"Mask shape before Attention: {tuple(src_mask.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print("OK")