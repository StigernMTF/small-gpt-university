from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "sentiment.pt"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"

MODEL_CFG = {
    "N": 3,
    "max_seq_len": 128,
    "emb_size": 256,
    "num_heads": 8,
    "ff_hidden_size": 512,
    "num_classes": 2,
}

BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 30
TARGET_ACC = 0.88
PATIENCE = 4

LABELS = {0: "negative", 1: "positive"}