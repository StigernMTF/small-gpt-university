from datasets import load_dataset
ds = load_dataset("wmt/wmt14", "de-en", split="train[:100000]")


def extractData(data):
    return {
        "deutch": data["translation"]["de"],
        "english": data["translation"]["en"]
    }

flatDataset = ds.map(extractData(), remove_columns=["translation"])
splitData = flatDataset.train_test_split(test_size=0.1)

# print(endDataVersion)
# print(endDataVersion["train"][0])
