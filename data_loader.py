import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from functools import partial

class TranslationDataset(Dataset):
    """
    Dataset class for Translation.
    Takes in parallel lists of source and target sentences.
    """
    def __init__(self, src_texts, tgt_texts):
        super().__init__()
        assert len(src_texts) == len(tgt_texts), "Source and target datasets must have the same length."
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        # Returns a tuple of raw strings
        return self.src_texts[idx], self.tgt_texts[idx]


def collate_fn(batch, tokenizer, pad_idx, bos_idx, eos_idx):
    """
    Collate function to process a batch of string tuples into padded tensors.
    Uses dynamic padding up to the longest sequence in the current batch.
    """
    src_tensors = []
    tgt_tensors = []

    for src_text, tgt_text in batch:
        # a) Translate strings to lists of integer indices.
        # Here we call our custom tokenizer logic.
        if hasattr(tokenizer, 'encode'):
            # If using a real HuggingFace tokenizer
            src_encoded = tokenizer.encode(src_text, add_special_tokens=False)
            tgt_encoded = tokenizer.encode(tgt_text, add_special_tokens=False)
        else:
            # Fallback for fake testing tokenizer
            src_encoded = tokenizer(src_text)
            tgt_encoded = tokenizer(tgt_text)

        # b) Add BOS (<s>) and EOS (</s>) tokens
        src_encoded = [bos_idx] + src_encoded + [eos_idx]
        tgt_encoded = [bos_idx] + tgt_encoded + [eos_idx]

        src_tensors.append(torch.tensor(src_encoded, dtype=torch.long))
        tgt_tensors.append(torch.tensor(tgt_encoded, dtype=torch.long))

    # c) Pad to the length of the longest sequence in THIS specific batch (saves memory)
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_idx)

    # d) Return a dictionary with the ready tensors
    return {
        "src": src_padded,
        "tgt": tgt_padded
    }


def get_dataloaders(src_texts, tgt_texts, tokenizer, pad_idx, bos_idx, eos_idx, batch_size=32):
    """
    Splits the data and creates train and validation DataLoaders optimized for Google Colab.
    """
    dataset = TranslationDataset(src_texts, tgt_texts)

    # Split 99% train, 1% validation
    total_size = len(dataset)
    val_size = max(1, int(total_size * 0.01))
    train_size = total_size - val_size

    # Using generator with fixed seed if reproducibility is needed, otherwise default random split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Bind the tokenizer and tokens to the collate function
    collator = partial(
        collate_fn,
        tokenizer=tokenizer,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx
    )

    # Hardcoded parameters optimized for Google Colab (2 vCPU, 16GB GPU)
    num_workers = 2
    pin_memory = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator
    )

    return train_loader, val_loader


# Fake Tokenizer for testing purposes
def fake_tokenizer(text):
    # Simply returns a list of integer ascii values to simulate tokens
    return [ord(c) for c in text]


if __name__ == "__main__":
    # Dummy dataset (5 sentences)
    fake_src = [
        "Hello world!",
        "How are you doing today?",
        "Google Colab is great for deep learning.",
        "Dynamic padding saves memory.",
        "A short sentence."
    ]

    fake_tgt = [
        "Hallo Welt!",
        "Wie geht es dir heute?",
        "Google Colab eignet sich hervorragend für Deep Learning.",
        "Dynamisches Padding spart Speicher.",
        "Ein kurzer Satz."
    ]

    # Dummy indices
    test_pad_idx = 0
    test_bos_idx = 1
    test_eos_idx = 2

    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        src_texts=fake_src,
        tgt_texts=fake_tgt,
        tokenizer=fake_tokenizer,
        pad_idx=test_pad_idx,
        bos_idx=test_bos_idx,
        eos_idx=test_eos_idx,
        batch_size=2  # small batch size for test
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    src_batch = batch["src"]
    tgt_batch = batch["tgt"]

    print(f"Batch src shape: {tuple(src_batch.shape)}")
    print(f"Batch tgt shape: {tuple(tgt_batch.shape)}")
    print("DataLoader OK")
