import torch
import torch.nn as nn
import torch.optim as optim


def get_loss_fn(pad_idx: int):
    """
    Создает функцию потерь с игнорированием токена паддинга и сглаживанием меток.
    """
    return nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)


def get_optimizer(model: nn.Module, lr: float = 1e-4):
    """
    Создает оптимизатор AdamW со специфичными для Трансформера гиперпараметрами.
    """
    return optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)


class TransformerLRScheduler:
    """
    Планировщик скорости обучения с фазой Warmup, как описано в 'Attention is All You Need'.
    """
    def __init__(self, optimizer, d_model: int = 512, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        
        # Формула из оригинальной статьи
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5, 
            self.step_num * (self.warmup_steps ** -1.5)
        )
        
        # Применяем вычисленный lr ко всем группам параметров
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == "__main__":
    # 1. Создаем фейковую мини-модель
    model = nn.Linear(10, 10)
    
    # 2. Инициализируем оптимизатор и планировщик
    optimizer = get_optimizer(model)
    scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=4000)

    # 3. Прогоняем 10 000 шагов
    lrs = []
    for _ in range(10000):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # 4. Выводим значения для проверки графика (рост -> пик -> спад)
    print(f"LR на шаге 1: {lrs[0]:.8f}")
    print(f"LR на шаге 4000 (пик): {lrs[3999]:.8f}")
    print(f"LR на шаге 10000: {lrs[9999]:.8f}")
    
    print("Optimization Setup OK")
