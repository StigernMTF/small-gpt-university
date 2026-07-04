import torch
from optimizer import *
from tokenizer import *

def predict_review(text, model, tokenizer, device, max_len=128):
    model.eval() # Switch to evaluation mode (no Dropout)

    # 1. Tokenize text (same as dataset processing)
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt' # Request PyTorch tensors
    )

    # Move to GPU
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # 2. Make prediction
    with torch.no_grad():
        logits = model(input_ids, padding_mask=attention_mask)
        prediction = torch.argmax(logits, dim=1).item()

        # Calculate model confidence from 0 to 100%
        probabilities = F.softmax(logits, dim=1)
        confidence = probabilities[0][prediction].item() * 100

    # 3. Output result
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    print(f"Review: '{text}'")
    print(f"Verdict: {sentiment} (Confidence: {confidence:.1f}%)\n")

# === TESTING! ===
predict_review("This movie is absolutely fantastic, a true masterpiece!", model, tokenizer, device)
predict_review("What a waste of time. The acting was terrible and the plot made no sense.", model, tokenizer, device)
predict_review("It is middle movie it is not good and not bad", model, tokenizer, device)