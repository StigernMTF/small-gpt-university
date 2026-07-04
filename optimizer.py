import torch.optim as optim
import torch.nn as nn
import tokenizer
import torch
import sentimentClassifier as sc
import model as m

# 1. Set model hyperparameters (values can be tuned)
VOCAB_SIZE = tokenizer.vocab_size # vocabulary size for bert-base-uncased 30522
MAX_SEQ_LEN = 128  # sequence length set in tokenizer
EMB_SIZE = 256     # embedding size
NUM_HEADS = 8      # number of attention heads
FF_HIDDEN_SIZE = 512 # size of hidden layer in FeedForward
NUM_BLOCKS = 3     # number of Encoder blocks

# 2. Create the Encoder
my_encoder = m.TransformerEncoder(
    N=NUM_BLOCKS,
    max_seq_len=MAX_SEQ_LEN,
    num_embeddings=VOCAB_SIZE,
    emb_size=EMB_SIZE,
    att_out_size=EMB_SIZE,
    att_head_size=EMB_SIZE // NUM_HEADS, # Head size = emb_size / num_heads
    num_heads=NUM_HEADS,
    ff_hidden_size=FF_HIDDEN_SIZE
)

# 3. Wrap the Encoder in our Classifier
model = sc.SentimentClassifier(
    encoder=my_encoder,
    hidden_size=EMB_SIZE,
    num_classes=2
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(model)

# Loss function for classification
criterion = nn.CrossEntropyLoss()

# lr (learning rate) = 1e-4 or 5e-5 is standard for small transformers
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Number of epochs
EPOCHS = 3