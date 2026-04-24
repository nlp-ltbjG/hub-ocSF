# coding:utf8

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个多分类任务
规律：x是一个5维向量，哪一维数字最大就属于第几类（类别范围：0-4）
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        # 线性层输出维度改为 num_classes (5)
        self.linear = nn.Linear(input_size, num_classes)  
        # 多分类任务使用交叉熵损失函数
        # 注意：nn.CrossEntropyLoss 内部自带 Softmax，因此 forward 中不需要再加激活函数
        self.loss = nn.CrossEntropyLoss() 

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实类别计算损失
        else:
            return y_pred  # 输出预测的 logits


# 生成一个样本
# 随机生成一个5维向量，最大值所在的索引即为该样本的类别
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 获取最大值的索引，范围是 0 到 4
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y) # 注意：多分类标签不需要用 [] 包裹成二维
    # X 保持 FloatTensor，但 Y 必须转换为 LongTensor，这是 CrossEntropyLoss 的要求
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)
    
    # 统计各类别样本分布
    unique, counts = np.unique(y.numpy(), return_counts=True)
    distribution = dict(zip(unique, counts))
    print("本次预测集中样本分布:", distribution)
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # y_pred shape: (100, 5)
        # 获取每一行最大值的索引作为预测类别
        pred_classes = torch.argmax(y_pred, dim=1) 
        
        for y_p, y_t in zip(pred_classes, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1  # 预测正确
            else:
                wrong += 1
                
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 30       # 训练轮数 (多分类可能需要更多轮次收敛)
    batch_size = 20      # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5       # 输入向量维度
    num_classes = 5      # 分类数量
    learning_rate = 0.01 # 学习率
    
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
            loss = model(x, y) 
            loss.backward()  
            optim.step()  
            optim.zero_grad()  
            watch_loss.append(loss.item())
            
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  
        log.append([acc, float(np.mean(watch_loss))])
        
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path)) 

    model.eval()  
    with torch.no_grad():  
        result = model.forward(torch.FloatTensor(input_vec)) 
        
    for vec, res in zip(input_vec, result):
        # res 是 logits，利用 softmax 计算概率分布
        probs = torch.nn.functional.softmax(res, dim=0)
        pred_class = torch.argmax(probs).item()
        pred_prob = probs[pred_class].item()
        
        print("输入：%s \n预测类别：%d, 对应概率值：%f\n" % (vec, pred_class, pred_prob))


if __name__ == "__main__":
    main()
    # 简单的预测测试
    test_vec = [[0.88889086, 0.15229675, 0.91082123, 0.03504317, 0.88920843], # 预期类别 2
                [0.14963533, 0.5524256,  0.25758807, 0.95520434, 0.84890681], # 预期类别 3
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392], # 预期类别 0
                [0.19349776, 0.99416669, 0.12579291, 0.41567412, 0.13588940]] # 预期类别 1
    print("\n---------- 模型预测测试 ----------")
    predict("model.bin", test_vec)
