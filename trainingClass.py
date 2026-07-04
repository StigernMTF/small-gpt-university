import torch.optim as optim
from sentimentClassifier import *
from tokenizer import *
from optimizer import model, device

# 1. Training settings
EPOCHS = 3
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}")

    model.train() # Enable Dropout and weight updates
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for step, batch in enumerate(train_dataloader):
        # 1. Move data to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 2. Clear old gradients
        optimizer.zero_grad()

        # 3. Predict and calculate loss
        logits = model(input_ids, padding_mask=attention_mask)
        loss = criterion(logits, labels)

        # 4. Update weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()

        train_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

        # Print progress every 100 batches
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1:4d} | Loss: {train_loss/(step+1):.4f} | Accuracy: {(train_correct/train_total)*100:.2f}%")

    # Test
    model.eval() # Disable Dropout, set to evaluation mode
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Make predictions
            logits = model(input_ids, padding_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1)

            # Count correct answers
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

    # Summary
    epoch_train_loss = train_loss / len(train_dataloader)
    epoch_train_acc = (train_correct / train_total) * 100
    epoch_test_acc = (test_correct / test_total) * 100

    print(f"\nEPOCH {epoch + 1} SUMMARY:")
    print(f"Training: Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc:.2f}%")
    print(f"Test: {epoch_test_acc:.2f}")