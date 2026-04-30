# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务

规律：一个多分类任务的训练，一个随机向量，哪一维数字最大就属于第几类

"""

VEC_LEN = 10   # 固定向量长度，类别数 = VEC_LEN

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.hidden = nn.Linear(input_size, 128) # 隐藏层
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, VEC_LEN)  # 线性层
        
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    def forward(self, x, y=None):
        x = self.relu(self.hidden(x)) # 隐藏层 + ReLU
        logits = self.output(x)  # 输出层（未归一化得分
        if y is not None:
            return self.loss(logits, y)     # 训练时返回损失
        else:
            # 预测时返回概率，dim=1：对每一行做 softmax，结果每一行的概率和为 1，dim=0：对每一列
            return torch.softmax(logits, dim=1) 
# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 生成一个随机向量，哪一维数字最大就属于第几类
def build_sample():
    feature = np.random.random(VEC_LEN)          # 随机向量
    label = np.argmax(feature)               # 最大值索引（0~count-1）
    return feature, label


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y) # 注意：交叉熵要求长整型 LongTensor

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("测试样本总数：%d" % test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 调用 model()，内部触发model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            pred_class = torch.argmax(y_p).item() # .item() 将零维张量转换为 Python 的标量（int 或 float）
            true_class = y_t.item()
            if pred_class == true_class:
                correct += 1  
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    input_size = VEC_LEN  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        # 创建训练集，正常任务是读取训练集
        train_x, train_y = build_dataset(train_sample)
        model.train()
        watch_loss = []
        for i in range(0, train_sample, batch_size): # step = batch_size：每次增加 batch_size
            x = train_x[i:i+batch_size]
            y = train_y[i:i+batch_size]
            loss = model(x, y)  # 计算loss  调用 model()，内部触发 forward
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
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    print("predict--------------")
    input_size = VEC_LEN
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()
        prob = res[pred_class].item()
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred_class, prob))


if __name__ == "__main__":
    main()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843,0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392,0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)