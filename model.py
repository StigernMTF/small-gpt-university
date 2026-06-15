import torch
import torch.nn as nn
from torch import Tensor
from encoder import Encoder
from decoder import Decoder


class Seq2SeqTransformer(nn.Module):
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
    ):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, src_mask)
        logits = self.generator(output)
        return logits


if __name__ == "__main__":
    vocab_size = 58101
    pad_idx = 58100
    batch_size = 2
    src_seq_len = 12
    tgt_seq_len = 10

    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        num_layers=3,  # reduce to 3 layers for testing speed
    )

    src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))

    src_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])  # [2, 12]

    tgt_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])  # [2, 10]

    logits = model(src, tgt, src_mask, tgt_mask)
    print(f"Logits shape: {tuple(logits.shape)}")

    assert logits.shape == (batch_size, tgt_seq_len, vocab_size), f"Unexpected shape: {logits.shape}"
    print("Seq2SeqTransformer Assembled Successfully!")