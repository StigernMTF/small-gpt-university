from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import data_loader
import collections

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") #BERT tokenizer

vocab_size = tokenizer.vocab_size

def tokenize_function(examples):
    return tokenizer(
        examples["cleaned_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

split_dataset = data_loader.final_dataset.train_test_split(test_size=0.1, seed=42)

print("3. Processing texts through the tokenizer")
tokenized_datasets = split_dataset.map(tokenize_function, batched=True)

print("4. Packaging into PyTorch format")
tokenized_datasets = tokenized_datasets.remove_columns(["cleaned_text", "rating"])
tokenized_datasets.set_format("torch")

BATCH_SIZE = 32


# 1. Training DataLoader (shuffle for training)
train_dataloader = DataLoader(
    tokenized_datasets['train'],
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 2. Test DataLoader (no need to shuffle for testing)
test_dataloader = DataLoader(
    tokenized_datasets['test'],
    batch_size=BATCH_SIZE,
    shuffle=False
)


for batch in train_dataloader:
    print("\nBatch contents (training):")
    print("input_ids shape:", batch['input_ids'].shape)
    print("attention_mask shape:", batch['attention_mask'].shape)
    print("label shape:", batch['label'].shape)
    break