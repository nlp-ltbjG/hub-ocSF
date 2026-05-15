import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个文本多分类任务
任务描述：对于一个包含“你”字的五个字的文本，“你”在第几位，就属于第几类（0-4类）。
模型：RNN / LSTM

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, model_type="RNN"):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        
        if model_type == "RNN":
            self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(vector_dim, vector_dim, batch_first=True)
            
        self.classify = nn.Linear(vector_dim, 5)  # 5分类
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)        # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, _ = self.rnn(x)      # (batch_size, sen_len, vector_dim)
        
        # 对于分类任务，我们通常取最后一个时间步的输出
        if isinstance(self.rnn, nn.LSTM):
            # LSTM output is (output, (h_n, c_n))
            last_step = output[:, -1, :] 
        else:
            last_step = output[:, -1, :]
            
        y_pred = self.classify(last_step) # (batch_size, vector_dim) -> (batch_size, 5)
        
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 字符表
def build_vocab():
    chars = "你我他的一是在了不有和人这中大为上个国" # 随意构造一些常用字
    vocab = {"PAD": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['UNK'] = len(vocab)
    return vocab

# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机选择一个位置放置“你”
    target_pos = random.randint(0, sentence_length - 1)
    
    # 生成其他位置的字符（排除“你”）
    chars = list(vocab.keys())
    if "你" in chars:
        chars.remove("你")
    if "PAD" in chars:
        chars.remove("UNK")
    
    content = []
    for i in range(sentence_length):
        if i == target_pos:
            content.append("你")
        else:
            content.append(random.choice(chars))
            
    # 转化为索引
    x = [vocab.get(char, vocab['UNK']) for char in content]
    y = target_pos
    return x, y

# 构造数据集
def build_dataset(sample_num, vocab, sentence_length):
    X = []
    Y = []
    for i in range(sample_num):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

# 评估
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 参数配置
    epoch_num = 20
    batch_size = 20
    train_sample = 1000
    char_dim = 20
    sentence_length = 5
    learning_rate = 0.005
    model_type = "RNN" # 可以在这里切换 RNN 或 LSTM
    
    # 初始化
    vocab = build_vocab()
    model = TorchModel(char_dim, sentence_length, vocab, model_type)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 每一轮都重新生成一些数据，增加数据多样性
        train_x, train_y = build_dataset(train_sample, vocab, sentence_length)
        
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
            
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, float(np.mean(watch_loss))])
        
    # 绘图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
