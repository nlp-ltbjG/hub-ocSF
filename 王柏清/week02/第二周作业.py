# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪一维最大就是哪一类
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层，输出5个类别
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失（cross_entropy内部会做softmax）
        else:
            return torch.softmax(x, dim=-1)  # 输出预测结果时使用softmax

# 生成一个样本
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)  # 返回输入向量和标签，标签是最大维度的索引

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # 统计各类别数量
    unique, counts = torch.unique(y, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"类别{class_idx.item()}有{count.item()}个样本", end="; ")
    print()
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，返回概率分布
        predicted_labels = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        correct = (predicted_labels == y).sum().item()
        wrong = len(y) - correct
        
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        # 打乱数据
        indices = torch.randperm(train_sample)
        train_x = train_x[indices]
        train_y = train_y[indices]
        
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
            
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    
    # 保存模型
    torch.save(model.state_dict(), "model_hm.bin")
    
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，返回概率
    
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()
        print("输入：%s, 预测类别：%d, 各类别概率：%s" % (vec, pred_class, res.numpy()))

if __name__ == "__main__":
    main()
    # 测试代码
    #test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
    #            [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
    #            [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
    #            [0.99349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    #predict("model.bin", test_vec)
