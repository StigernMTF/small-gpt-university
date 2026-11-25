import torch.nn as nn
import MHAttention
from collections import *

class EncoderBlock(nn.Module):
    # keeping query_input_size for decoder support
    def __init__(self, input_size, head_size, num_heads, out_size, ff_hidden_size, query_input_size=None):
        super(EncoderBlock, self).__init__()

        self.input_size = input_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.query_input_size = input_size if query_input_size is None else query_input_size

        # for feed forward
        self.ff_hidden_size = ff_hidden_size

        self.attention = MHAttention.MultiHeadAttention(
            input_size=self.input_size,
            head_size=self.head_size,
            num_heads=self.num_heads,
            out_size=self.out_size,
            query_input_size=self.query_input_size
        )

        if self.query_input_size != self.out_size:
            self.adapt = nn.Linear(self.query_input_size, self.out_size)
        else:
            self.adapt = nn.Identity() # returns input without changes

        self.norm_1 = nn.LayerNorm(self.out_size)

        self.feed_forward = nn.Sequential(OrderedDict([
            ("Linear_1", nn.Linear(self.out_size, self.ff_hidden_size)),
            ("Activation", nn.ReLU()),
            ("Dropout", nn.Dropout(p=0.1)),
            ("Linear_2", nn.Linear(self.ff_hidden_size, self.out_size)),
        ]))

        self.norm_2 = nn.LayerNorm(self.out_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value, padding_mask=None):
        # batch_size, seq_len, in_size

        # Self Multi-Head Attention
        attention_out = self.attention(query, key, value, padding_mask=padding_mask)
        # batch_size, seq_len, out_size

        attention_out = self.dropout(attention_out)

        add_1_out = attention_out + self.adapt(query)
        norm_1_out = self.norm_1(add_1_out)
        # batch_size, seq_len, out_size

        # Feed Forward
        feed_forward_out = self.feed_forward(norm_1_out)
        # batch_size, seq_len, out_size

        feed_forward_out = self.dropout(feed_forward_out)

        # Add + Norm
        add_out = feed_forward_out + norm_1_out
        norm_2_out = self.norm_2(add_out)

        return norm_2_out