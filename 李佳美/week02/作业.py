import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
一个多分类任务的训练:一个长度为n的随机向量，哪一维数字最大就属于第几类。
n=10
"""


class Model(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=10):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)
        # self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x,y=None):
        hidden = self.activation(self.layer1(x))  # x: N * 10   w1.T: 10 * 5  b1: 1 * 5 -> hidden: N * 5
        y_pred = self.layer1(x)  # hidden: N * 5   w2.T: 5 * 10  b2: 1 * 10 -> y: N * 10
        # y_pred = self.activation(self.layer2(2)) 
        # print(y_pred)
        # print(y)
        
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果
    

# 生成一条样本数据，样本数据长度为10，其中有一个维度比其他维度大，代表该样本属于该维度
def build_data_sample():
    x = np.random.random(10)
    return x, np.argmax(x)


# 生成样本数据集，各个样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_data_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 生成测试数据集，各个样本均匀生成
test_x, test_y = build_dataset(500)

def evaluate(model):
    model.eval() # 向模型说明开始测试
    # print(test_y)
    acc_count = 0
    with torch.no_grad():  # 不计算梯度
        y_pred = model(test_x)  # 输出预测值
        for i in range(len(test_x)):
            y_pred_class = torch.argmax(y_pred[i], dim=0)  # 取出每行最大值的索引
            if y_pred_class == test_y[i]: # 计算预测正确的个数
                acc_count += 1
    acc = acc_count / len(test_x)  # 计算准确率
    return acc

def main():
    # 配置参数
    epoch_num = 150  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 10
    hidden_size = 15
    output_size = 10
    learning_rate = 0.1  # 学习率
    # 建立模型
    model = Model(input_size, hidden_size, output_size)
    # print(model.state_dict())
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()  # 向模型说明开始训练
        epoch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)使用的是CE    
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            epoch_loss.append(loss.item())
        acc = evaluate(model)  # 测试本轮模型结果
        print("=========\n第%d轮平均loss:%f   acc:%f\n=========" % (epoch + 1, np.mean(epoch_loss), acc))
        # print()
        log.append([acc, float(np.mean(epoch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    fig, ax1 = plt.subplots()

    epochs = range(len(log))
    acc_list = [l[0] for l in log]
    loss_list = [l[1] for l in log]

    # 左侧 y 轴画 loss
    ax1.plot(epochs, loss_list, label="loss", color="red")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="lower right")

    # 右侧 y 轴画 acc
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc_list, label="acc")
    ax2.set_ylabel("acc")
    ax2.legend(loc="upper right")

    plt.show()

    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 10
    model = Model(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        result = torch.argmax(result, dim=1)  # 取出每行最大值的索引
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d" % (vec, res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = build_dataset(10)
    # print(test_vec)
    # test_vec = [
    #         [0.1853, 0.8248, 0.6405, 0.5720, 0.5520, 0.6486, 0.4829, 0.2078, 0.3059, 0.6325],
    #         [0.2987, 0.1408, 0.8709, 0.2289, 0.9692, 0.6854, 0.0942, 0.9671, 0.7859, 0.1078],
    #         [0.0526, 0.2166, 0.7111, 0.5729, 0.0158, 0.7638, 0.0233, 0.7572, 0.9027, 0.1459],
    #         [0.9850, 0.6758, 0.9126, 0.9035, 0.3952, 0.9603, 0.2384, 0.6939, 0.5844, 0.1649],
    #         [0.2360, 0.3132, 0.7040, 0.9302, 0.5895, 0.6544, 0.0607, 0.5745, 0.3449, 0.9635],
    #         [0.0791, 0.3390, 0.4034, 0.6590, 0.4154, 0.7720, 0.6773, 0.1974, 0.4565, 0.5922],
    #         [0.4354, 0.7458, 0.0747, 0.6185, 0.2023, 0.8658, 0.7113, 0.7894, 0.0173, 0.6568],
    #         [0.3341, 0.8372, 0.4604, 0.8447, 0.8898, 0.0046, 0.6977, 0.8208, 0.9691, 0.4788],
    #         [0.3446, 0.3871, 0.5055, 0.5283, 0.1503, 0.7777, 0.7359, 0.2670, 0.7592, 0.8210],
    #         [0.8911, 0.6280, 0.9303, 0.9032, 0.5657, 0.9356, 0.6700, 0.6106, 0.2148, 0.8517]
    #     ]
    # predict("model.bin", test_vec)
