import sentimentClassifier as sc
import model as m

def create_model(cfg):
    my_encoder = m.TransformerEncoder(
        N=cfg["N"],
        max_seq_len=cfg["max_seq_len"],
        num_embeddings=cfg["vocab_size"],
        emb_size=cfg["emb_size"],
        att_out_size=cfg["emb_size"],
        att_head_size=cfg["emb_size"] // cfg["num_heads"],  # Head size = emb_size / num_heads
        num_heads=cfg["num_heads"],
        ff_hidden_size=cfg["ff_hidden_size"],
    )

    # 3. Wrap the Encoder in our Classifier
    model = sc.SentimentClassifier(
        encoder=my_encoder,
        hidden_size=cfg["emb_size"],
        num_classes=cfg["num_classes"],
    )

    return model