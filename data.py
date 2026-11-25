from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from config import TOKENIZER_DIR

def get_tokenizer():
    if TOKENIZER_DIR.exists():
        return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    return AutoTokenizer.from_pretrained('bert-base-uncased')


def get_dataloader(tokenizer, batch_size, max_len):
    from datasets import load_dataset, concatenate_datasets
    print("1. Loading data")
    dataset = load_dataset("rohan2810/amazon-movies-meta-reviews-merged", split="train")

    print("2. Keeping only required columns")
    dataset = dataset.select_columns(['rating', 'cleaned_text'])

    print("3. Removing neutral reviews and converting other")
    dataset = dataset.filter(lambda x: x['rating'] != 3)

    def convert_to_binary(example):
        example['label'] = 1 if example['rating'] > 3 else 0
        return example

    dataset = dataset.map(convert_to_binary)

    print("4. Splitting into positive and negative")
    positives = dataset.filter(lambda x: x['label'] == 1)
    negatives = dataset.filter(lambda x: x['label'] == 0)

    print("5. Selecting 30,000 random samples of each class...")
    pos_sampled = positives.shuffle(seed=42).select(range(30000))
    neg_sampled = negatives.shuffle(seed=42).select(range(30000))

    print("6. Assembling the final dataset")
    balanced_dataset = concatenate_datasets([pos_sampled, neg_sampled])
    final_dataset = balanced_dataset.shuffle(seed=42)

    print("7. Done")
    print(final_dataset[0])

    print("last step. spliting data in train/test dataset's")
    split_dataset = final_dataset.train_test_split(test_size=0.1, seed=42)
    print(split_dataset)

    def tokenize_function(examples, max_len):
        return tokenizer(
            examples["cleaned_text"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )
    print("Processing text through the tokenizer")
    tokenized_datasets = split_dataset.map(tokenize_function, batched=True)
    print("Packaging into PyTorch format")
    tokenized_datasets = tokenized_datasets.remove_columns(["cleaned_text", "rating"])
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_datasets['train'],
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        tokenized_datasets['test'],
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader