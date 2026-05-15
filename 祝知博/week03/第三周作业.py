# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

# ─── 超参数 ────────────────────────────────────────────────
N_SAMPLES   = 4000
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20

"""

基于pytorch框架编写模型训练
实现一个多分类任务的训练:对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类

"""
# 常用汉字集合，用于生成文本
characters = "的一是在有和人我他她它不就都而要可以到你会着没有看日子们说生好" \
             "学们上大时去也能下过子来里出年得自后以家可小多然于心么" \
             "生中你了为同这那些什么从如把还又很要没最更今谁再把给"

# 字符到索引的映射，0为padding
char_to_idx = {"<PAD>": 0}
for i, char in enumerate(characters):
    if char not in char_to_idx:
        char_to_idx[char] = len(char_to_idx)
vocab_size = len(char_to_idx)

class TorchModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 5)  # 5个类别
        self.loss_fn   = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        _, h_n = self.rnn(x)   # h_n: (num_layers, batch_size, hidden_dim)
        h_n = h_n.squeeze(0)   # (batch_size, hidden_dim)
        h_n = self.bn(h_n)
        h_n = self.dropout(h_n)
        logits = self.fc(h_n)  # (batch_size, 5)
        
        if y is not None:
            return self.loss_fn(logits, y)  # 预测值和真实值计算损失
        else:
            return logits  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成包含"你"字的五个字文本，“你”在第几位，就属于第几类
def build_sample():
    """
    随机生成包含"你"字的五个字文本
    """
    # 随机选择"你"字的位置（0-4，共5个位置）
    you_position = random.randint(0, 4)
    
    # 生成五个字的列表
    result = []
    for i in range(5):
        if i == you_position:
            result.append("你")
        else:
            # 从字符集中随机选择一个字（排除"你"字，确保只有一个"你"）
            char = random.choice(characters.replace("你", ""))
            result.append(char)
    return result, you_position


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        # 将字符转换为索引
        x_idx = [char_to_idx.get(char, 0) for char in x]
        X.append(x_idx)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        logits = model(x)  # 模型预测 model.forward(x)
        y_pred = torch.argmax(logits, dim=1)  # 获取预测类别
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1  # 预测正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = EPOCHS  # 训练轮数
    batch_size = BATCH_SIZE  # 每次训练样本个数
    train_sample = N_SAMPLES  # 每轮训练总共训练的样本总数
    learning_rate = LR  # 学习率
    # 建立模型
    model = TorchModel(vocab_size, EMBED_DIM, HIDDEN_DIM)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_texts):
    model = TorchModel(vocab_size, EMBED_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        # 将文本转换为索引
        X = []
        for text in input_texts:
            x_idx = [char_to_idx.get(char, 0) for char in text]
            X.append(x_idx)
        X = torch.LongTensor(X)
        logits = model(X)  # 模型预测
        probs = torch.softmax(logits, dim=1)  # 计算概率
    for text, prob in zip(input_texts, probs):
        predicted_class = torch.argmax(prob).item()  # 获取预测类别
        max_prob = float(torch.max(prob).item())  # 获取最大概率值
        print("输入：%s, 预测“你”在第%d位, 概率：%f" % (text, predicted_class + 1, max_prob))  # 打印结果


if __name__ == "__main__":
    main()
    # 测试预测（确保所有文本都是5个字符）
    # test_texts = ["你是我的爱", "我爱你中国", "今天你好吗", "我想你了啊", "他说你是谁"]
    # predict("model.bin", test_texts)
