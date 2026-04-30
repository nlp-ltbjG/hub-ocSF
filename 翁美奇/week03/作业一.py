import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 1000          
MAXLEN      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 32           # 调小一点，样本不多
EPOCHS      = 30
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

COMMON_CHARS = "的一是不了人在我有他这中到个大地出也子来们过上必比如"
DEFAULT_COUNTS = 5  

# ─── 1. 数据生成────────────────

# 方法一固定列表
def make_data_fixed():
    candidates = [
        "你好世界啊",   # 你索引0
        "我爱你中国",   # 索引1
        "地方回复你",   # 索引4
        "明天你好吗",   # 索引2
        "你我他她它",   # 索引0
        "想你的时候",   # 索引1
        "嗯好不你呀",   # 索引3
        "我说你听啊",   # 索引2
        "给你比个心",   # 索引1
        "恨你恨到死"    # 索引1
    ]
    return random.choice(candidates)

# 方法二：随机生成
def make_data_random():
    chars = list(COMMON_CHARS + "你")
    while True:
        sample = random.choices(chars, k = DEFAULT_COUNTS)
        if "你" in sample:
            return "".join(sample)
        
def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        sample = make_data_random()
        pos = sample.index('你')   # 0~4
        data.append((sample, pos))
    return data

# ─── 2. 词表与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for x, _ in data:
        for ch in x:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab):
    return [vocab.get(ch, 1) for ch in sent]

class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.long),
                torch.tensor(self.y[i], dtype=torch.long))

# ─── 3. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, DEFAULT_COUNTS)   # 5分类
    def forward(self, x, y=None):
        # x: (batch, seq_len)
        emb = self.embedding(x)                 # (b, 5, embed_dim)
        out, _ = self.rnn(emb)                  # (b, 5, hidden_dim)
        pooled = out.max(dim=1)[0]              # (b, hidden_dim)
        pooled = self.bn(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)                # (b, 5)
        if y is not None:
            loss = F.cross_entropy(logits, y)
            return loss
        return torch.softmax(logits, dim=-1)

# ─── 4. 训练函数 ────────────────────────────────────────────
def train():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    print(f"前5个样本：{data[:5]}")
    vocab = build_vocab(data)
    print(f"样本数：{len(data)}，词表大小：{len(vocab)}")
    
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]
    
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)
    
    model = KeywordRNN(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for X, y in train_loader:
            loss = model(X, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                logits = model(X)   # 注意：这里调用了forward的else分支，返回softmax概率
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch:2d} | Train Loss: {np.mean(losses):.4f} | Val Acc: {acc:.4f}")

if __name__ == '__main__':
    train()