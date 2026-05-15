# 作业三

# 设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
# 如果不知道怎么设计，可以选择如下任务:
# 对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----超参数---------
SEED = 42
N_SAMPLES = 4000

EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1E-3
TRAIN_RATIO = 0.8
EPOCHS = 10


    


#  1. 构造数据，生成N_SAMPLES条文本数据，每条文本长度为5，包含‘你’，‘你’在哪个位置，文本就属于哪个类别，类别从0-4
def generate_data(N_SAMPLES=N_SAMPLES):
    data = []
    # 文字列表
    chars = list("的一是在不了有人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经")
    # 将'你'随机插入到每条文本中，并记录类别
    for i in range(N_SAMPLES):
        text = random.choices(chars, k=4)  # 随机生成一个长度为5的文本
        pos = random.randint(0, 4)  # 随机选择一个位置插入'你'
        text.insert(pos, '你')  # 在指定位置插入'你'
        text = ''.join(text)  # 将列表转换为字符串
        data.append((text, pos))  # 将文本和类别组成一个元组，添加到数据列表中
    return data

# 2. 构建词表
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text, _ in data:
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab
    
## 词表编码
def encode(text, vocab, max_len=5):
    ids = [vocab.get(ch, vocab['<UNK>']) for ch in text][:max_len]
    ids += [vocab['<PAD>']] * (max_len - len(ids))
    return ids

# 3. 构造数据集
class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=5):
        self.x = [encode(text, vocab, max_len) for text, _ in data]
        self.y = [label for _, label in data]

    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return (
            torch.tensor(self.x[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )

# print(generate_data())

# 4. 构造模型，使用RNN，LSTM等模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)            # (B, L) → (B, L, E)
        _, h = self.rnn(x)               # h: (1, B, H)
        h_last = h[-1]                   # (B, H)
        logits = self.fc(h_last)         # (B, 5)
        return logits
    
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)              # (B, L) → (B, L, E)
        _, (h, _) = self.lstm(x)           # h: (1, B, H)
        h_last = h[-1]                     # (B, H)
        logits = self.fc(h_last)           # (B, 5)
        return logits

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            y_pred = model(x)
            predicted = torch.argmax(y_pred, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

# 5. 训练模型
def train():
    # 生成数据样本
    data = generate_data()
    # 构建词表
    vocab = build_vocab(data)
    print(f"样本数量：{len(data)}, 词表大小：{len(vocab)}")

    # 训练集与测试机划分
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    test_data = data[split:]

    # 构造数据集
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=32, shuffle=True)
    test_loader = DataLoader(TextDataset(test_data, vocab), batch_size=32, shuffle=False)
    
    # 构造模型
    # model = LSTMClassifier(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)  # LSTM模型
    model = RNNClassifier(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)  # RNN模型
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量：{total_params}")


    # 训练模型
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f'"训练完成！最终测试集准确率为：{evaluate(model, test_loader):.4f}')

    print("="*10)
    print("推理示例")
    print("="*10)
    model.eval()
    test_text = [
        "北京欢迎你",
        "我爱你中国",
        "你是工作狂",
        "我和你一起"
    ]

    with torch.no_grad():
        for text in test_text:
            x = torch.tensor([encode(text, vocab)], dtype=torch.long)  # (1, 5)
            logits = model(x)                                          # (1, 5)
            predicted = torch.argmax(logits, dim=1).item()             # 标量
            print(f'输入文本: "{text}", 预测类别: {predicted}')

if __name__ == "__main__":
    train()
