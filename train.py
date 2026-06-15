import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Импортируем наши модули
from model import Seq2SeqTransformer
from optimization import get_loss_fn, get_optimizer, TransformerLRScheduler


# Шаг 1: Определение устройства (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Шаг 2: Вспомогательная функция для масок
def create_masks(src, tgt, pad_idx):
    """
    Создает базовые 2D-маски для исходного и целевого текста.
    True там, где находятся смысловые токены (не паддинг).
    """
    src_mask = (src != pad_idx)
    tgt_mask = (tgt != pad_idx)
    return src_mask, tgt_mask


# Шаг 4: Цикл обучения для одной эпохи
def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, pad_idx, device):
    """
    Проводит один проход по всему тренировочному датасету (1 эпоха).
    """
    model.train()
    total_loss = 0.0
    
    for step, batch in enumerate(dataloader):
        # Поддержка разных форматов выдачи DataLoader (словарь или кортеж)
        if isinstance(batch, dict):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
        else:
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            
        # Шаг 3: Логика Teacher Forcing
        # tgt_input: всё кроме последнего токена (идет в Декодер)
        # tgt_expected: всё кроме первого токена <BOS> (идет в расчет функции потерь)
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]
        
        # Создаем маски на основе tgt_input
        src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)
        
        # Прямой проход (Forward pass)
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # Меняем размерности для вычисления CrossEntropyLoss
        # logits из [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        logits = logits.reshape(-1, logits.size(-1))
        # tgt_expected из [batch_size, seq_len] -> [batch_size * seq_len]
        tgt_expected = tgt_expected.reshape(-1)
        
        # Вычисление ошибки
        loss = loss_fn(logits, tgt_expected)
        
        # Обратное распространение (Backward pass)
        loss.backward()
        
        # Обновление весов и шага планировщика
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # Вывод статистики каждые 100 шагов
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
    return total_loss / len(dataloader)


# Шаг 5: Функции выживания в Google Colab (Чекпоинты)
def save_checkpoint(model, optimizer, epoch, step, filepath):
    """Сохраняет текущие веса модели и состояние оптимизатора."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch+1} to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Загружает прогресс обучения, если файл существует."""
    if os.path.isfile(filepath):
        print(f"Found checkpoint at '{filepath}', loading...")
        # Загрузка на нужное устройство
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch + 1}")
        return start_epoch
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0


# Шаг 6: Главный цикл
def main():
    # Реальные гиперпараметры
    VOCAB_SIZE = 58101
    PAD_IDX = 58100
    BOS_IDX = 58100  # Исправляем проблему "None" из Helsinki-NLP
    EOS_IDX = 0
    NUM_EPOCHS = 10
    BATCH_SIZE = 32  # Лучше начать с 32, чтобы Colab не выдал Out of Memory

    # ПУТЬ СОХРАНЕНИЯ: Обязательно подключи Google Drive перед этим!
    # Выполни в отдельной ячейке: from google.colab import drive; drive.mount('/content/drive')
    CHECKPOINT_FILE = '/content/drive/MyDrive/transformer_checkpoint.pth'

    print(f"Initializing training on device: {device}")

    # 1. Инициализация модели (теперь ставим полноценные 6 слоев)
    model = Seq2SeqTransformer(
        vocab_size=VOCAB_SIZE,
        pad_idx=PAD_IDX,
        num_layers=6,
        max_len=256  # Оригинальный размер Трансформера
    ).to(device)

    # 2. Инициализация лосса, оптимизатора и планировщика
    loss_fn = get_loss_fn(PAD_IDX)
    optimizer = get_optimizer(model)
    scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=4000)

    # 3. Загрузка чекпоинта
    start_epoch = load_checkpoint(model, optimizer, CHECKPOINT_FILE)

    # 4. ПОДКЛЮЧЕНИЕ РЕАЛЬНЫХ ДАННЫХ
    print("Preparing real datasets...")
    # Берем данные из твоего сгенерированного словаря splitData (он уже должен быть в памяти Colab)
    src_train = splitData["train"]["english"]
    tgt_train = splitData["train"]["deutch"]

    tokenizer = getTokenizer()  # Твоя функция инициализации токенайзера

    # Создаем даталоадеры (передаем правильные токены)
    train_loader, val_loader = get_dataloaders(
        src_texts=src_train,
        tgt_texts=tgt_train,
        tokenizer=tokenizer,
        pad_idx=PAD_IDX,
        bos_idx=BOS_IDX,
        eos_idx=EOS_IDX,
        batch_size=BATCH_SIZE
    )

    # 5. Главный цикл
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,  # Передаем реальный лоадер
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            pad_idx=PAD_IDX,
            device=device
        )

        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}")

        # Сохранение на Google Drive!
        save_checkpoint(model, optimizer, epoch, step=len(train_loader), filepath=CHECKPOINT_FILE)


if __name__ == "__main__":
    main()
