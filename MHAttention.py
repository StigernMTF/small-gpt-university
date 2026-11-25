import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    # Removed masked=False from init as the mask is now passed in forward
    def __init__(self, input_size, head_size, num_heads, out_size, query_input_size=None):
        super(MultiHeadAttention, self).__init__()

        self.input_size = input_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.query_input_size = self.input_size if query_input_size is None else query_input_size

        self.W_Q = nn.Linear(self.query_input_size, self.num_heads * self.head_size, bias=False)
        self.W_K = nn.Linear(self.input_size, self.num_heads * self.head_size, bias=False)
        self.W_V = nn.Linear(self.input_size, self.num_heads * self.head_size, bias=False)

        self.out = nn.Linear(self.head_size * self.num_heads, self.out_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value, padding_mask=None):
        batch_size = key.size(0)
        emb_len = key.size(1)
        query_emb_len = query.size(1)

        q = self.W_Q(query)
        k = self.W_K(key)
        v = self.W_V(value)

        q = q.view(batch_size, query_emb_len, self.num_heads, self.head_size).transpose(1,2)
        k = k.view(batch_size, emb_len, self.num_heads, self.head_size).transpose(1,2)
        v = v.view(batch_size, emb_len, self.num_heads, self.head_size).transpose(1,2)

        k_T = k.transpose(2, 3)

        relevance = q @ k_T / math.sqrt(self.head_size)
        # relevance size: [batch_size, num_heads, query_emb_len, emb_len]

        if padding_mask is not None:
            # Tokenizer mask has size [batch_size, emb_len].
            # We need to transform it into [batch_size, 1, 1, emb_len] to apply it to relevance.

            # 1. Expand dimensions: [batch_size, 1, 1, emb_len]
            extended_mask = padding_mask.unsqueeze(1).unsqueeze(2)

            # 2. 0 -> -infinity to ensure 0 after softmax
            relevance = relevance.masked_fill(extended_mask == 0, float('-inf'))

        relevance = F.softmax(relevance, dim=-1)

        relevance = self.dropout(relevance)

        heads = relevance @ v
        heads = heads.transpose(1, 2)
        concat = heads.reshape(batch_size, query_emb_len, self.head_size * self.num_heads)
        out = self.out(concat)

        return out