"""
第三周交作业.py
文本多分类任务 —— "你"字位置识别（RNN 版）

任务：输入一个恰好包含一个"你"字的五字中文文本，判断"你"出现在第几位（1~5），对应类别 0~4
模型：Embedding → RNN → 取最后隐藏状态 → Linear → CrossEntropyLoss

依赖：torch >= 2.0   (pip install torch)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 5000      # 样本总数
MAXLEN      = 5         # 固定5个字
EMBED_DIM   = 64
HIDDEN_DIM  = 128
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 30
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 常用汉字池（不含"你"）
COMMON_CHARS = list(
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要"
    "她出也得里后自以会家可下而过天去能对小多然于心学之都好看起发当没"
    "成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又"
    "行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高"
    "已亲其进此话常与活正感"
)

TARGET_CHAR = '你'
SENT_LEN = 5
N_CLASSES = SENT_LEN    # 5 个位置 → 5 类


def make_sample():
    """
    生成一条样本：五字句子 + 标签
    标签 0~4 分别对应 "你" 在第 1~5 位
    """
    position = random.randint(0, SENT_LEN - 1)  # "你"出现的位置 0~4
    chars = []
    for i in range(SENT_LEN):
        if i == position:
            chars.append(TARGET_CHAR)
        else:
            chars.append(random.choice(COMMON_CHARS))
    sentence = ''.join(chars)
    return sentence, position


def build_dataset(n=N_SAMPLES):
    data = [make_sample() for _ in range(n)]
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

# ─── 词编码 ────────────────────────────────────────────────
def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class TextClassifier(nn.Module):
    """
    文本多分类器（RNN 版）
    架构：Embedding → RNN → 取最后隐藏状态 → Dropout → Linear → CrossEntropyLoss
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)              # (B, L, embed_dim)
        output, _ = self.rnn(emb)            # (B, L, hidden_dim)
        last_out = output[:, -1, :]          # (B, hidden_dim) 取最后时间步
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)           # (B, n_classes)
        return logits


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train():
    print(f"\n{'='*60}")
    print(f"模型类型: RNN")
    print(f"{'='*60}")

    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    # 统计各类别分布
    label_counts = [0] * N_CLASSES
    for _, lb in data:
        label_counts[lb] += 1
    print(f"  类别分布：{label_counts}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab),
                            batch_size=BATCH_SIZE)

    model = TextClassifier(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}  (最高: {best_val_acc:.4f})")

    # ─── 推理示例 ───
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你在干什么',      # 你 在第1位 → 类别0
        '我在找你呢',      # 你 在第4位 → 类别3
        '大家喜欢你',      # 你 在第5位 → 类别4
        '我想告诉你',      # 你 在第5位 → 类别4
        '昨天看到你',      # 你 在第5位 → 类别4
        '你好世界啊',      # 你 在第1位 → 类别0
    ]

    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred_label = logits.argmax(dim=1).item()
            position = pred_label + 1
            print(f"  [预测位置: 第{position}位]  {sent}")

    return best_val_acc


# ─── 6. 主入口 ────────────────────────────────
if __name__ == '__main__':
    train()
