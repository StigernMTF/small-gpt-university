import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_emb_len, d_model):
        super(PositionalEncoding, self).__init__()

        self.max_emb_len = max_emb_len
        self.d_model = d_model

        pos = torch.arange(max_emb_len)[:, None] # [[0], [1], [2], [3], [4], ...[max_emb_len]]]
        i = torch.arange(d_model)[None, :] # i = [[0, 1, 2, 3, 4, ..., d_model]]

        pe = torch.zeros(self.max_emb_len, self.d_model)
        # max_emb_len, d_model

        sin = torch.sin(pos / (10000 ** (i[:, ::2] / self.d_model)))
        # max_emb_len, d_model // 2

        cos = torch.cos(pos / (10000 ** (i[:, ::2] / self.d_model)))
        # max_emb_len, d_model // 2

        pe[:, ::2] = sin
        pe[:, 1::2] = cos

        pe = pe.unsqueeze(0)
        # 1, max_emb_len, d_model

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, emb):
        # batch_size, emb_len, input_size

        emb_len = emb.size(1)

        emb = emb + self.pe[:, :emb_len]
        # batch_size, emb_len, input_size

        emb = self.dropout(emb)

        return emb