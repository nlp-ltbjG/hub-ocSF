#不好意思老师，我上周提交到自己的仓库了
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42      # 固定随机种子
N_SAMPLES   = 5000    # 总数据量（每类1000条）
MAXLEN      = 5       # 固定5字长度（核心：任务要求）
EMBED_DIM   = 64      # 字符嵌入维度
HIDDEN_DIM  = 64      # RNN/LSTM隐藏层维度
LR          = 1e-3    # 学习率
BATCH_SIZE  = 64      # 批次大小
EPOCHS      = 15      # 训练轮数
TRAIN_RATIO = 0.8     # 训练/验证划分
NUM_CLASSES = 5       # 多分类：5个类别（对应“你”的5个位置）

random.seed(SEED)
torch.manual_seed(SEED)

CHAR_POOL = [
    '我', '他', '她', '的', '爱', '喜', '欢', '学', '习', '工', 
    '作', '玩', '乐', '吃', '喝', '行', '走', '看', '听', '说'
]

def generate_sample(pos):
    # 初始化5个位置的字符为随机非“你”字符
    chars = [random.choice(CHAR_POOL) for _ in range(5)]
    # 仅在指定位置替换为“你”（保证唯一）
    chars[pos] = '你'
    # 拼接为5字文本
    text = ''.join(chars)
    return text, pos

def build_dataset(n=N_SAMPLES):
    data = []
    samples_per_class = n // NUM_CLASSES  # 每类样本数
    for label in range(NUM_CLASSES):
        for _ in range(samples_per_class):
            text, _ = generate_sample(label)
            data.append((text, label))
    random.shuffle(data)  # 打乱顺序
    return data

def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]  # 未知字符用<UNK>
    ids = ids[:maxlen]                       # 截断（本任务固定5字，无截断）
    ids += [0] * (maxlen - len(ids))        # 补PAD（本任务无补）
    return ids

class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),  # 输入id序列
            torch.tensor(self.y[i], dtype=torch.long)   # 标签（long适配CrossEntropy）
        )

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, model_type='rnn', embed_dim=EMBED_DIM, 
                 hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 选择RNN/LSTM核心层
        if model_type.lower() == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif model_type.lower() == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("model_type仅支持 'rnn' / 'lstm'")
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)  # 多分类输出5维

    def forward(self, x):
        # x: (batch_size, seq_len) → (B, 5)
        embed = self.embedding(x)  # (B, 5, embed_dim)
        
        # RNN/LSTM前向计算
        if isinstance(self.rnn, nn.LSTM):
            rnn_out, (_, _) = self.rnn(embed)  # LSTM返回(out, (h_n, c_n))
        else:
            rnn_out, _ = self.rnn(embed)       # RNN返回(out, h_n)
        
        # 对序列维度做MaxPooling（提取关键信息）
        pooled = rnn_out.max(dim=1)[0]  # (B, hidden_dim)
        
        # 归一化+dropout+全连接
        pooled = self.dropout(self.bn(pooled))
        logits = self.fc(pooled)        # (B, 5)
        return logits

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            pred = torch.argmax(logits, dim=1)  # 多分类取最大概率索引
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

def train_model(model_type):
    # 数据准备
    print(f"\n===== 训练 {model_type.upper()} 模型 =====")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"样本数：{len(data)}，词表大小：{len(vocab)}")
    
    # 划分训练/验证集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]
    
    # 构建DataLoader
    train_dataset = TextDataset(train_data, vocab)
    val_dataset = TextDataset(val_data, vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 模型初始化
    device = torch.device("cpu") 
    model = TextClassifier(
        vocab_size=len(vocab),
        model_type=model_type,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(device)
    
    # 优化器与损失函数（多分类用CrossEntropyLoss）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 训练循环
    print(f"开始训练（{EPOCHS}轮）...")
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # 前向传播
            logits = model(X)
            loss = criterion(logits, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 每轮评估
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)
        
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
    
    # 5. 最终评估
    print(f"\n{model_type.upper()} 模型 - 最佳验证准确率：{best_val_acc:.4f}")
    
    # 6. 推理示例
    print("\n--- 推理示例 ---")
    model.eval()
    test_texts = [
        "你爱吃苹果",  # 你在第1位 → 0
        "我你去跑步",  # 你在第2位 → 1
        "他爱你喝茶",  # 你在第3位 → 2
        "看书听你说",  # 你在第4位 → 3
        "吃饭打游戏你" # 你在第5位 → 4
    ]
    with torch.no_grad():
        for text in test_texts:
            # 编码文本
            ids = torch.tensor([encode(text, vocab)], dtype=torch.long).to(device)
            logits = model(ids)
            pred = torch.argmax(logits, dim=1).item()
            print(f"文本：{text} → 预测“你”在第{pred+1}位（真实第{text.index('你')+1}位）")
    
    return model, best_val_acc

if __name__ == '__main__':
    # 训练RNN模型
    rnn_model, rnn_acc = train_model("rnn")
    
    # 训练LSTM模型
    lstm_model, lstm_acc = train_model("lstm")
    
    # 对比结果
    print(f"\n===== 最终对比 =====")
    print(f"RNN 验证准确率：{rnn_acc:.4f}")
    print(f"LSTM 验证准确率：{lstm_acc:.4f}")
    print(f"最优模型：{'LSTM' if lstm_acc > rnn_acc else 'RNN'}")
