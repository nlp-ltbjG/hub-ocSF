"""
设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。如果不知道怎么设计，可以选择如下任务:
对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
chars = ['我', '的', '在', '不', '了', '有', '和', '人','中','大','生','以', '上', '主','们','产','出','所','能','行','成',
         '同','学','工','书','数','输','熟','句','据','局','聚','称','城','乘','程','喔','哈','嘿','你']

def generate_data():
    random.seed()
    index = random.randint(0, 4)
    text = ''
    chars_no_ni = [c for c in chars if c != '你']
    for i in range(5):
        if index == i:
            text += '你'
        else:
            text += random.choice(chars_no_ni)
    return text

def build_dataset(n=N_SAMPLES):
    data = [generate_data() for _ in range(n)]
    return data

# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab():
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab):
    ids  = [vocab.get(ch, 1) for ch in sent]
    return ids

# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s in data]
        self.y = [sent.index('你') for sent in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )

# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)         
        lstm_out, (h_n, c_n) = self.lstm(embedded) 
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out)    
        return logits

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab()
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model = KeywordLSTM(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你今天好美',
        '她说你了吗',
        '他说他爱你',
        '他说啥你信',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred = torch.argmax(logits, dim=1).item()
            print(f"文本：{sent} → 预测位置：{pred}（实际位置：{sent.index('你')}）")


if __name__ == '__main__':
     train()
