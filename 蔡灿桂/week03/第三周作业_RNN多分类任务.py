"""
对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 5000  # 样本数量
MAXLEN      = 5     # 强制固定长度为5
EMBED_DIM   = 32    
HIDDEN_DIM  = 64    
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5     # 5个类别（对应'你'在第1到第5位）

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
FILLER_CHARS = ['我', '他', '她', '是', '的', '人', '好', '坏', '爱', '吃', '喝', '玩', '乐', '天', '地']

def generate_sentence():
    """
    生成逻辑：
    1. 随机决定'你'字的位置 (0 到 4)
    2. 生成一个长度为5的字符串，指定位置填'你'，其余位置填随机字
    3. 返回 (句子, 标签)
    """
    target_pos = random.randint(0, 4)
    
    sentence_chars = []
    for i in range(MAXLEN):
        if i == target_pos:
            sentence_chars.append('你')
        else:
            sentence_chars.append(random.choice(FILLER_CHARS))
            
    return "".join(sentence_chars), target_pos

def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        data.append(generate_sentence())
    random.shuffle(data)
    return data

# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab):
    ids = [vocab.get(ch, 1) for ch in sent]
    if len(ids) < MAXLEN:
        ids +=  * (MAXLEN - len(ids))
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
class PositionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, NUM_CLASSES) # 输出5维

    def forward(self, x):
        e, _ = self.rnn(self.embedding(x))  # (B, 5, hidden_dim)
        pooled = e.max(dim=1)            # (B, hidden_dim)
        out = self.fc(pooled)               # (B, 5) -> 输出 Logits
        return out

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = logits.argmax(dim=1)     # 评估准确率时还是取最大值索引
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total

def train():
    print("1. 生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"   样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = PositionRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"   模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")
    print("2. 开始训练...")

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
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    # ─── 6. 推理示例───────────────────────────
    print("\n3. --- 推理示例 (输出五维概率向量) ---")
    model.eval()
    
    test_sents = [
        "你好呀我在", # '你'在索引0 -> 期望接近 
        "我爱你中国", # '你'在索引1 -> 期望接近 
        "他是你爸爸", # '你'在索引2 -> 期望接近 
        "今天你去哪", # '你'在索引3 -> 期望接近 
        "世界只有你", # '你'在索引4 -> 期望接近 
        "我也爱吃瓜", # 不含'你' -> 随机分布或全低
    ]
    
    with torch.no_grad():
        for sent in test_sents:
            # 1. 编码
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            
            # 2. 模型前向传播 (得到 Logits)
            logits = model(ids)
            
            # 3. 使用 Softmax 转换为概率分布
            # dim=1 表示在类别维度上进行归一化
            probs = F.softmax(logits, dim=1)
            
            # 4. 打印结果
            # .tolist() 将 Tensor 转换为 Python 列表
            prob_vector = probs.tolist()
            
            # 格式化输出，保留4位小数
            formatted_probs = [f"{p:.4f}" for p in prob_vector]
            print(f"文本: {sent}")
            print(f"五维概率向量: {formatted_probs}")
            print("-" * 30)<websource>source_group_web_1</websource>

if __name__ == '__main__':
    train()
