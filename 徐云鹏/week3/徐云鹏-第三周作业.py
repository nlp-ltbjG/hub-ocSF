"""
多分类文本任务：5字文本中「你」的位置分类
===========================================
任务描述：对一个任意包含"你"字的五个字的文本，「你」在第几位就属于第几类。
即 5 分类任务（类别 1~5）。

模型对比：Vanilla RNN vs LSTM vs BiLSTM
依赖：pip install torch numpy matplotlib scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix
import time

# ========== 1. 超参数 ==========
VOCAB_SIZE = 50          # 常用汉字数量（含特殊token）
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_CLASSES = 5           # 位置 1~5
SEQ_LEN = 5
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
NUM_SAMPLES = 20_000      # 总样本数
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")

# ========== 2. 构造词表 ==========
# 常用汉字列表（不含"你"），用于填充非"你"位置
HAN_CHARS = list("的一不是了人在有我他这中大小上开会国为学和生大好要们出时也得家可下过天去能对方")

# 构建 char -> index 映射
char2idx = {"<PAD>": 0, "<UNK>": 1, "你": 2}
for ch in HAN_CHARS:
    char2idx[ch] = len(char2idx)

idx2char = {v: k for k, v in char2idx.items()}
VOCAB_SIZE = len(char2idx)

# ========== 3. 生成数据集 ==========
def generate_sample():
    pos = random.randint(0, 4)
    chars = ["你" if i == pos else random.choice(HAN_CHARS) for i in range(5)]
    return "".join(chars), pos

def text_to_tensor(text):
    return torch.tensor([char2idx.get(c, char2idx["<UNK>"]) for c in text], dtype=torch.long)

data = [generate_sample() for _ in range(NUM_SAMPLES)]

print("样本示例:")
for i in range(10):
    text, label = data[i]
    print(f"  '{text}' -> 类别 {label+1}")

split = int(NUM_SAMPLES * TRAIN_RATIO)
train_data, test_data = data[:split], data[split:]

class TextPositionDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        text, label = self.data[idx]
        return text_to_tensor(text), torch.tensor(label, dtype=torch.long)

train_loader = DataLoader(TextPositionDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TextPositionDataset(test_data), batch_size=BATCH_SIZE)

# ========== 4. 定义模型 ==========

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)           # (batch, seq_len, embed_dim)
        _, h_n = self.rnn(emb)            # h_n: (1, batch, hidden_dim)
        return self.fc(h_n.squeeze(0))    # (batch, num_classes)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n[-1])


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        return self.fc(torch.cat([h_fwd, h_bwd], dim=-1))


# ========== 5. 训练与评估 ==========

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels

# ========== 6. 训练循环 ==========

models_to_train = {
    "VanillaRNN": VanillaRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES),
    "LSTM": LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES),
    "BiLSTM": BiLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES),
}

results = {}

for model_name, model in models_to_train.items():
    print(f"\n{'='*50}")
    print(f"训练模型: {model_name}")
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer)
        vl, va, _, _ = evaluate(model, test_loader, criterion)
        train_losses.append(tl); train_accs.append(ta)
        test_losses.append(vl);  test_accs.append(va)
        best_acc = max(best_acc, va)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{EPOCHS} | Train Loss: {tl:.4f} Acc: {ta:.4f} | "
                  f"Test Loss: {vl:.4f} Acc: {va:.4f}")

    elapsed = time.time() - start
    _, _, preds, labels = evaluate(model, test_loader, criterion)
    results[model_name] = {
        "train_losses": train_losses, "train_accs": train_accs,
        "test_losses": test_losses, "test_accs": test_accs,
        "best_test_acc": best_acc, "elapsed": elapsed,
        "preds": preds, "labels": labels,
    }
    print(f"  >> 最佳测试Acc: {best_acc:.4f} | 耗时: {elapsed:.2f}s")

# ========== 7. 结果汇总 ==========

print(f"\n{'='*60}")
print(f"{'模型':<15} {'最佳Test Acc':<15} {'耗时(秒)':<15}")
print(f"{'-'*45}")
for n, r in results.items():
    print(f"{n:<15} {r['best_test_acc']:.4f}        {r['elapsed']:.2f}")

# ========== 8. 可视化 ==========

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {"VanillaRNN": "#FF6B6B", "LSTM": "#4ECDC4", "BiLSTM": "#45B7D1"}

for idx, (metric, title, ylabel) in enumerate([
    ("train_losses", "Train Loss", "Loss"),
    ("test_losses", "Test Loss", "Loss"),
    ("train_accs", "Train Accuracy", "Accuracy"),
    ("test_accs", "Test Accuracy", "Accuracy"),
]):
    ax = axes[idx // 2][idx % 2]
    for name, res in results.items():
        ax.plot(range(1, EPOCHS+1), res[metric], label=name, color=colors[name])
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("\n训练曲线已保存到 training_curves.png")

# ========== 9. 分类报告 ==========

for name, res in results.items():
    print(f"\n{'='*50}\n{name} - 分类报告\n{'='*50}")
    print(classification_report(
        res["labels"], res["preds"],
        target_names=[f"位置{i+1}" for i in range(NUM_CLASSES)], digits=4
    ))

# ========== 10. 错误样例 ==========

print(f"\n{'='*50}\n错误分析：测试集随机样本\n{'='*50}")
sample_indices = random.sample(range(len(test_data)), min(15, len(test_data)))
for idx in sample_indices:
    text, true_label = test_data[idx]
    print(f"\n  '{text}'  真实: 位置{true_label+1}")
    for m_name in results:
        model = models_to_train[m_name].to(DEVICE)
        model.eval()
        with torch.no_grad():
            x = text_to_tensor(text).unsqueeze(0).to(DEVICE)
            pred = model(x).argmax(dim=1).item() + 1
            mark = "✓" if pred == true_label + 1 else "✗"
            print(f"    {m_name:<10}: 位置{pred} {mark}")

print(f"\n实验完成! 期待的结果:")
print(f"  - LSTM / BiLSTM 收敛速度和最终精度应优于 VanillaRNN")
print(f"  - BiLSTM 利用前后双向信息，位置识别会更准")
print(f"  - 在数据量足够时，三个模型都能学到位置信息")
