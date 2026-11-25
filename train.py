import torch
import torch.nn as nn
import torch.optim as optim
import build
import data
from config import *
from data import get_tokenizer

criterion = LR
optimizer = WEIGHT_DECAY

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, padding_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = get_tokenizer()
    cfg = {**MODEL_CFG, "vocab_size": tokenizer.vocab_size}   # + размер словаря
    model = build.create_model(cfg).to(device)

    train_dataloader, test_dataloader = data.get_dataloader(tokenizer, BATCH_SIZE, cfg["max_seq_len"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Папка для весов и токенизатора (токенизатор сохраняем, чтобы серверу не нужен был интернет)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    tokenizer.save_pretrained(str(TOKENIZER_DIR))

    best_acc = 0.0     # лучшая точность на тесте за всё время
    bad_epochs = 0     # сколько эпох подряд не было улучшения

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"Epoch {epoch}")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, padding_mask=attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            if (step + 1) % 100 == 0:
                print(f"  Step {step+1:4d} | Loss: {train_loss/(step+1):.4f} | "
                      f"Accuracy: {(train_correct/train_total)*100:.2f}%")

        epoch_train_loss = train_loss / len(train_dataloader)
        epoch_train_acc = train_correct / train_total
        test_acc = evaluate(model, test_dataloader, device)

        print(f"\nEPOCH {epoch} SUMMARY:")
        print(f"Training: Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc*100:.2f}%")
        print(f"Test: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),   # сами веса
                    "config": cfg,                       # размеры модели
                    "test_acc": test_acc,
                    "epoch": epoch,
                },
                CHECKPOINT_PATH,
            )
            print(f"  >>> saved to {CHECKPOINT_PATH} (test acc {test_acc*100:.2f}%)\n")
        else:
            bad_epochs += 1

        if best_acc >= TARGET_ACC:
            print(f"Target accuracy {TARGET_ACC*100:.0f}% reached. Stopping.")
            break
        if bad_epochs >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs. Stopping.")
            break


if __name__ == "__main__":
    main()