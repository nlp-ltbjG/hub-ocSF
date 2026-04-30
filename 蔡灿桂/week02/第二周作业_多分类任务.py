# coding:utf8

import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
说明：输入5维向量，判断哪一维数值最大，即为哪一类（共5类）。

"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        # 输出层维度改为 num_classes (5)
        self.linear = nn.Linear(input_size, num_classes)  
        # 交叉熵损失函数
        self.loss = nn.CrossEntropyLoss() 

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 输出 logits (未经过softmax的原始得分)
        logits = self.linear(x)  
        
        if y is not None:
            # CrossEntropyLoss 接收 logits 和 类别标签(整数)
            return self.loss(logits, y)  
        else:
            # 预测时，返回概率最大的类别索引
            return torch.argmax(logits, dim=1) 

# 生成一个样本
# 规律：找出5维向量中数值最大的那一维的索引
def build_sample():
    x = np.random.random(5)
    # 找到最大值的索引 (0, 1, 2, 3, 或 4)
    y = np.argmax(x) 
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # X保持FloatTensor, Y必须是LongTensor(类别索引)才能用于CrossEntropyLoss
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # 统计各类别样本数量
    unique, counts = np.unique(y.numpy(), return_counts=True)
    print("本次预测集样本分布:", dict(zip(unique, counts)))
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 此时返回的是类别索引
        # 对比预测类别和真实类别
        for y_p, y_t in zip(y_pred, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
                
    acc = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc

def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    num_classes = 5  # 5个类别
    learning_rate = 0.01
    
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            
            loss = model(x, y)  # 计算loss
            loss.backward()     # 计算梯度
            optim.step()        # 更新权重
            optim.zero_grad()   # 梯度归零
            
            watch_loss.append(loss.item())
        
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
        
    # 保存模型
    torch.save(model.state_dict(), "model_multiclass.bin")
    
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vecs):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("模型权重加载完毕，开始预测...")
    with torch.no_grad():
        # 将输入列表转为Tensor
        x = torch.FloatTensor(input_vecs)
        # 获取预测结果 (类别索引)
        predictions = model(x) 
        
        for vec, pred in zip(input_vecs, predictions):
            # 找出实际最大值所在的索引，用于验证
            true_class = np.argmax(vec)
            print("输入: %s \n -> 预测类别: %d (第%d维最大), 真实类别: %d" % (vec, int(pred), int(pred)+1, true_class))

if __name__ == "__main__":
    # 1. 训练模型
    main()
    
    # 2. 预测测试 
    # test_vec = [
    #     [0.1, 0.2, 0.9, 0.4, 0.5], # 第3维最大 (类别2)
    #     [0.8, 0.1, 0.1, 0.1, 0.1], # 第1维最大 (类别0)
    #     [0.1, 0.1, 0.1, 0.1, 0.9]  # 第5维最大 (类别4)
    # ]
    # predict("model_multiclass.bin", test_vec)
