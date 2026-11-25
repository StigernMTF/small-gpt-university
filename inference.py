from pathlib import Path

import torch
import torch.nn.functional as F

from build import create_model
from config import CHECKPOINT_PATH, LABELS
from data import get_tokenizer


class Predictor:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"There is no checkpoint at {checkpoint_path}"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.cfg = ckpt["config"]
        self.test_acc = ckpt.get("test_acc")

        self.model = create_model(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.tokenizer = get_tokenizer()

    @torch.inference_mode()
    def predict(self, text):
        encode = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.cfg["max_seq_len"],
            return_tensors="pt",
        )
        input_ids = encode["input_ids"].to(self.device)
        attention_mask = encode["attention_mask"].to(self.device)

        logits = self.model(input_ids, padding_mask=attention_mask)
        probs = F.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs))

        return {
            "label": LABELS[idx],
            "confidence": float(probs[idx]),
            "probability": {LABELS[i]: float(p) for i, p in enumerate(probs)},
        }