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