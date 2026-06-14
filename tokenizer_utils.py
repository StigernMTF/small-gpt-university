from transformers import AutoTokenizer

def getTokenizer():
    moder_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = AutoTokenizer.from_pretrained(moder_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenizePair(en_text, de_text, tokenizer, max_length=128):
    en_encoded = tokenizer(
        en_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    de_encoded = tokenizer(
        text_target=de_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return {
        "input_ids": en_encoded["input_ids"].squeeze(0),
        "attention_mask": en_encoded["attention_mask"].squeeze(0),
        "labels": de_encoded["input_ids"].squeeze(0)
    }