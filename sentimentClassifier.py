import torch.nn as nn
import torch

class SentimentClassifier(nn.Module):
    def __init__(self, encoder, hidden_size, num_classes=2):
        super(SentimentClassifier, self).__init__()
        # Accepts a pre-assembled Transformer-encoder
        self.encoder = encoder

        # Final linear layer producing 2 logits
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, padding_mask):
        # encoder_out shape: [batch_size, seq_len, hidden_size]
        encoder_out = self.encoder(input_ids, padding_mask=padding_mask)

        # Expand mask from [batch_size, seq_len] to [batch_size, seq_len, 1]
        mask_expanded = padding_mask.unsqueeze(-1).float()

        # Padding vectors are zeroed out and no longer carry noise.
        masked_embeddings = encoder_out * mask_expanded

        # Sum vectors of all words in the sentence
        summed = torch.sum(masked_embeddings, dim=1)

        # clamp(min=1e-9) protects against division by zero
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # pooled_out shape: [batch_size, hidden_size]
        pooled_out = summed / counts

        # logits shape: [batch_size, 2]
        logits = self.classifier(pooled_out)

        return logits