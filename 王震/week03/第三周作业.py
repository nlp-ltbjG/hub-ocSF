import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===================== 1. 构造数据集 =====================
# 词汇表：所有用到的汉字（方便转数字编码）
vocab = ["我", "你", "他", "她", "它", "好", "世", "界", "呀", "爱", "吃", "饭", "睡", "觉"]
word2idx = {w: i for i, w in enumerate(vocab)}  # 字→数字
vocab_size = len(vocab)
num_classes = 5  # 5分类（位置0-4）
seq_len = 5      # 固定5个字

# 生成训练数据：确保每个句子只有1个"你"，且在不同位置
def generate_data():
    data = []
    labels = []
    # 构造"你"在 第1~5位 的样本
    positions = [0, 1, 2, 3, 4]
    for pos in positions:
        # 每个位置生成100个样本，让数据充足
        for _ in range(100):
            sentence = []
            for i in range(5):
                if i == pos:
                    sentence.append("你")
                else:
                    # 随机选其他字
                    sentence.append(np.random.choice([w for w in vocab if w != "你"]))
            # 转数字编码
            sentence_idx = [word2idx[w] for w in sentence]
            data.append(sentence_idx)
            labels.append(pos)
    return torch.LongTensor(data), torch.LongTensor(labels)

# 加载数据
train_data, train_labels = generate_data()

# ===================== 2. 定义模型：RNN & LSTM =====================
# 超参数
embedding_dim = 10  # 词嵌入维度
hidden_dim = 20     # 隐藏层维度
batch_size = 16
epochs = 20

# 模型1：基础RNN
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        # 词嵌入层：汉字转向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN层
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层：输出5分类
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch, 5] → [batch, 5, 10]
        out, _ = self.rnn(x)   # out: [batch, 5, 20]
        # 取最后一个时间步的输出做分类
        out = out[:, -1, :]
        out = self.fc(out)     # [batch, 5]
        return out

# 模型2：LSTM（比RNN更常用）
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ===================== 3. 训练函数 =====================
def train_model(model, model_name):
    print(f"\n===== 开始训练 {model_name} =====")
    # 损失函数+优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    for epoch in range(epochs):
        model.train()
        # 打乱数据
        idx = torch.randperm(len(train_data))
        data_shuffle = train_data[idx]
        label_shuffle = train_labels[idx]

        total_loss = 0
        correct = 0
        # 分批训练
        for i in range(0, len(data_shuffle), batch_size):
            batch_x = data_shuffle[i:i+batch_size]
            batch_y = label_shuffle[i:i+batch_size]

            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += torch.sum(pred == batch_y).item()

        # 打印日志
        acc = correct / len(train_data)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f} | Acc: {acc:.4f}")
    return model

# ===================== 4. 开始训练 =====================
rnn_model = RNNModel()
lstm_model = LSTMModel()

# 训练RNN
rnn_model = train_model(rnn_model, "RNN")
# 训练LSTM
lstm_model = train_model(lstm_model, "LSTM")

# ===================== 5. 测试模型 =====================
def predict(model, sentence):
    """输入5个字，预测"你"的位置"""
    model.eval()
    with torch.no_grad():
        idx = [word2idx[w] for w in sentence]
        x = torch.LongTensor([idx])
        output = model(x)
        pos = torch.argmax(output).item()
        return pos + 1  # 转回1~5位

# 测试案例
test_sentences = [
    "你我他她它",   # 第1位
    "我你他她它",   # 第2位
    "我他你她它",   # 第3位
    "我他她你它",   # 第4位
    "我他她它你"    # 第5位
]

print("\n===== 模型测试结果 =====")
for s in test_sentences:
    rnn_res = predict(rnn_model, s)
    lstm_res = predict(lstm_model, s)
    print(f"句子：{s}")
    print(f"  RNN预测：你在第{rnn_res}位")
    print(f"  LSTM预测：你在第{lstm_res}位")
