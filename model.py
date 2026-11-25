import torch.nn as nn
import positionalEncoding as PE
import encoder

class TransformerEncoder(nn.Module):
    def __init__(self, N, max_seq_len, num_embeddings, emb_size, att_out_size, att_head_size, num_heads, ff_hidden_size):
        super(TransformerEncoder, self).__init__()

        self.N = N
        self.max_seq_len = max_seq_len

        # for embedding_layer
        self.num_embeddings = num_embeddings # number of unique indices (vocab_size)
        self.emb_size = emb_size # embedding vector dimension, d_model

        self.att_out_size = att_out_size
        self.att_head_size = att_head_size
        self.num_heads = num_heads

        self.ff_hidden_size = ff_hidden_size

        self.embedding_layer = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.emb_size
        )
        self.positional_encoder = PE.PositionalEncoding(
            max_emb_len=self.max_seq_len,
            d_model=self.emb_size
        )

        self.encoder_blocks = nn.ModuleDict({
            f"encoder_block_{i}": encoder.EncoderBlock(
                input_size=self.emb_size if i==0 else self.att_out_size,
                head_size=self.att_head_size,
                num_heads=self.num_heads,
                out_size=self.att_out_size,
                ff_hidden_size=self.ff_hidden_size,
            ) for i in range(self.N)
        })

    def forward(self, encoder_input, padding_mask=None):
        # batch_size, seq_len

        encoder_emb = self.embedding_layer(encoder_input)
        # batch_size, seq_len, emb_size

        out = self.positional_encoder(encoder_emb)
        # batch_size, seq_len, emb_size

        for block in self.encoder_blocks.values():
            out = block(out, out, out, padding_mask=padding_mask)
        # batch_size, seq_len, att_out_size

        return out