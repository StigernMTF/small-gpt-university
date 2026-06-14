import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from encoder import InputEmbedding, PositionalEncoding, MultiHeadAttention, FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Masked Self-Attention for decoder (attention on target text with tgt_mask)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Cross-Attention (Query is from decoder, Key/Value is from encoder output memory)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Feed-Forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # LayerNorm for each of the 3 sublayers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        # x:       [batch_size, tgt_seq_len, d_model] - current decoder representations
        # memory:  [batch_size, src_seq_len, d_model] - encoder output representations

        # 1. Masked Self-Attention
        # Queries, Keys and Values are from x. tgt_mask (padding + causal) is applied.
        attn_out = self.self_attn(x, x, x, tgt_mask)  # [batch_size, tgt_seq_len, d_model]
        x = self.norm1(x + self.dropout1(attn_out))    # [batch_size, tgt_seq_len, d_model]

        # 2. Cross-Attention
        # Queries (Query) are from decoder (x), Keys (Key) and Values (Value) are from encoder (memory).
        # src_mask (padding mask for source text) is applied.
        cross_out = self.cross_attn(x, memory, memory, src_mask)  # [batch_size, tgt_seq_len, d_model]
        x = self.norm2(x + self.dropout2(cross_out))             # [batch_size, tgt_seq_len, d_model]

        # 3. Feed Forward
        ff_out = self.feed_forward(x)                  # [batch_size, tgt_seq_len, d_model]
        x = self.norm3(x + self.dropout3(ff_out))      # [batch_size, tgt_seq_len, d_model]

        return x


class Decoder(nn.Module):
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

        # Embedding layer
        self.embedding = InputEmbedding(d_model, vocab_size, pad_idx)
        # Positional encoding layer
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Decoder layers list
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> Tensor:
        # tgt:       [batch_size, tgt_seq_len] - target sequence tokens
        # memory:    [batch_size, src_seq_len, d_model] - encoder output representations
        # tgt_mask:  [batch_size, tgt_seq_len] or [batch_size, 1, tgt_seq_len, tgt_seq_len] - padding mask for tgt
        # src_mask:  [batch_size, src_seq_len] or [batch_size, 1, 1, src_seq_len] - padding mask for src

        # 1. Get dimensions of target sequence
        device = tgt.device
        tgt_seq_len = tgt.size(1)  # tgt_seq_len

        # 2. Get embeddings of target sequence
        x = self.embedding(tgt)  # [batch_size, tgt_seq_len, d_model]

        # 3. Add positional encoding
        x = self.pos_encoding(x)  # [batch_size, tgt_seq_len, d_model]

        # 4. Form causal mask (look-ahead mask)
        # Causal mask prevents decoder from looking ahead:
        # [tgt_seq_len, tgt_seq_len], where lower triangle is filled with True/1 and upper with False/0.
        causal_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, tgt_seq_len, tgt_seq_len]

        # 5. Combine target padding mask and causal mask
        if tgt_mask is not None:
            if tgt_mask.dim() == 2:
                tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_seq_len]
            tgt_mask = (tgt_mask.bool() & causal_mask)         # [batch_size, 1, tgt_seq_len, tgt_seq_len]
        else:
            tgt_mask = causal_mask  # [1, 1, tgt_seq_len, tgt_seq_len]

        # 6. Adapt source mask (src_mask) for cross-attention
        if src_mask is not None:
            if src_mask.dim() == 2:
                src_mask = src_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_seq_len]
            src_mask = src_mask.bool()

        # 7. Run through all decoder layers
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)  # [batch_size, tgt_seq_len, d_model]

        return x


if __name__ == "__main__":
    # Input parameters for testing
    vocab_size = 58101
    pad_idx = 58100
    max_len = 128
    d_model = 512

    # Initialize decoder
    decoder = Decoder(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        d_model=d_model,
        max_len=max_len,
        num_layers=3,  # reduce to 3 for testing
    )

    # Dummy tensors
    batch_size = 2
    tgt_seq_len = 10
    src_seq_len = 12

    # tgt: target tokens
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    # memory: encoder output
    memory = torch.randn(batch_size, src_seq_len, d_model)

    # Simulate attention_mask for src and tgt
    # 1 - token is considered, 0 - padding token is ignored
    src_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])  # [2, 12]

    tgt_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])  # [2, 10]

    # Run through model
    output = decoder(tgt, memory, tgt_mask, src_mask)

    # Output results and check tensor shape
    print(f"Decoder input tgt shape:     {tuple(tgt.shape)}")
    print(f"Decoder input memory shape:  {tuple(memory.shape)}")
    print(f"Decoder output shape:        {tuple(output.shape)}")

    # Ensure shape is [batch_size, tgt_seq_len, d_model]
    assert output.shape == (batch_size, tgt_seq_len, d_model), f"Unexpected output shape: {output.shape}"
    print("Decoder OK")
