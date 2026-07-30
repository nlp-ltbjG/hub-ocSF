import torch
import torch.nn as nn
import torch.optim as optim
import random

# ===================== 1. 任务定义 & 数据生成 =====================
# 固定长度：5个字
SEQ_LEN = 5
# 分类数：你 在第1~5位 → 5类
NUM_CLASSES = 5
# 简单词汇表（随便选常用字，让模型能学习）
vocab = ["我", "你", "他", "爱", "吃", "饭", "睡", "觉", "玩", "乐"]
vocab_size = len(vocab)
# 字 → 索引 映射
word2idx = {w: i for i, w in enumerate(vocab)}

# 生成训练数据：自动生成 5 个字，确保有且只有一个"你"，并标注类别
def generate_data():
    data = []
    # 生成 5000 条训练数据
    for _ in range(5000):
        # 随机选择"你"的位置 0~4（对应类别 1~5）
        pos = random.randint(0, 4)
        # 生成 5 个字，指定位置放"你"，其他随机
        sentence = []
        for i in range(SEQ_LEN):
            if i == pos:
                sentence.append("你")
            else:
                sentence.append(random.choice([w for w in vocab if w != "你"]))
        # 字转索引
        idx_seq = [word2idx[w] for w in sentence]
        # 标签：位置+1（第1类~第5类）
        label = pos + 1
        data.append((idx_seq, label))
    return data

# 生成数据并转成 Tensor
data = generate_data()
train_x = torch.tensor([d[0] for d in data], dtype=torch.long)
train_y = torch.tensor([d[1] for d in data], dtype=torch.long) - 1  # 标签转 0~4


# ===================== 2. 模型定义：RNN / LSTM =====================
class TextClassifier(nn.Module):
    def __init__(self, model_type="lstm", embed_dim=16, hidden_dim=32):
        super(TextClassifier, self).__init__()
        # 词嵌入层：把字索引转成向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 选择 RNN 或 LSTM
        self.model_type = model_type
        if model_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # 分类头：输出 5 类
        self.fc = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # RNN/LSTM 前向
        if self.model_type == "rnn":
            out, hn = self.rnn(x)
        else:
            out, (hn, cn) = self.rnn(x)
        
        # 取最后一个时间步的输出做分类
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# ===================== 3. 训练配置 =====================
# 切换模型："rnn" 或 "lstm"
MODEL_TYPE = "lstm"  

model = TextClassifier(model_type=MODEL_TYPE)
criterion = nn.CrossEntropyLoss()  # 多分类损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20
batch_size = 32

print(f"开始训练 {MODEL_TYPE.upper()} 模型...")

# ===================== 4. 训练循环 =====================
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 小批量训练
    for i in range(0, len(train_x), batch_size):
        x = train_x[i:i+batch_size]
        y = train_y[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    acc = correct / total
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f} | Acc: {acc:.4f}")


# ===================== 5. 测试模型 =====================
def predict(sentence):
    """输入5个字的句子，预测"你"在第几位"""
    model.eval()
    idx = [word2idx[w] for w in sentence]
    x = torch.tensor([idx], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    pred_class = torch.argmax(out).item() + 1  # 转回 1~5
    print(f"输入：{''.join(sentence)}")
    print(f"模型预测：你 在第 {pred_class} 位\n")

# 测试案例
print("\n===== 模型测试 =====")
predict(["你", "爱", "吃", "饭", "乐"])  # 第1位
predict(["我", "你", "玩", "睡", "觉"])  # 第2位
predict(["他", "爱", "你", "吃", "饭"])  # 第3位
predict(["我", "吃", "饭", "你", "乐"])  # 第4位
predict(["他", "玩", "睡", "觉", "你"])  # 第5位
