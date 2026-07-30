import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn 
import json 
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class TorchModel(nn.Module):
    def __init__(self,input_size, num_classes):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x,y)
        else:
            return torch.softmax(x,dim=1)

def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)

    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    class_count = torch.bincount(y)

    for i in range(5):
        print(f"本次测试集中，类别{i}（对应{i+1}维）的样本数是{class_count[i]}")
    correct ,wrong = 0, 0
    with torch.no_grad():
        y_pred_prob = model(x)
        y_pred = torch.argmax(y_pred_prob,dim=1)
        for y_p,y_t in zip(y_pred, y):
            if y_p == y_t:
                correct += 1 
            else:
                wrong += 1
    accuracy = correct/(correct+wrong)
    print(f"正确预测个数：{correct}, 错误预测个数：{wrong}, 正确率：{accuracy:.6f}")
    return accuracy

def main():
    epoch_num = 30
    batch_size = 20
    train_sample = 5000
    input_size = 5
    num_classes = 5
    learning_rate = 0.001
    model = TorchModel(input_size,num_classes)
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []
    train_x,train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        
        avg_loss = np.mean(watch_loss)
        print(f"=========\n第{epoch + 1}轮平均loss:{avg_loss:.6f}")
        acc = evaluate(model)
        log.append([acc, avg_loss])
    
    torch.save(model.state_dict(), "model_multiclass.bin")
    print("训练日志（每轮准确率、平均loss）：", log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 准确率曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # loss曲线
    plt.xlabel("训练轮数")
    plt.ylabel("数值")
    plt.legend()
    plt.title("多分类模型训练曲线（交叉熵损失）")
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5  
    model = TorchModel(input_size, num_classes)  
    model.load_state_dict(torch.load(model_path))
    print("加载的模型参数（线性层权重和偏置）：")
    print(model.state_dict())

    model.eval() 
    with torch.no_grad(): 
        result_prob = model(torch.FloatTensor(input_vec))
        result_class = torch.argmax(result_prob, dim=1)
    for vec, prob, cls in zip(input_vec, result_prob, result_class):
        prob_str = ", ".join([f"类别{i}（第{i+1}维）: {p:.4f}" for i, p in enumerate(prob)])
        print(f"输入向量：{np.array(vec).round(4)}")
        print(f"类别概率分布：{prob_str}")
        print(f"预测类别：{cls.item()}（对应第{cls.item()+1}维，最大值所在维度）\n")

if __name__ == "__main__":
    main()
    # 自定义测试向量
    test_vec = [
        [0.0788, 0.1523, 0.3108, 0.0350, 0.8892],  # 最大值在第5维，预期类别4
        [0.7496, 0.5524, 0.9576, 0.9552, 0.8489],  # 最大值在第3维，预期类别2
        [0.9080, 0.6748, 0.1363, 0.3468, 0.1987],  # 最大值在第1维，预期类别0
        [0.9935, 0.5942, 0.9258, 0.4157, 0.1359],  # 最大值在第1维，预期类别0
        [0.2345, 0.8765, 0.4567, 0.6789, 0.3456]   # 最大值在第2维，预期类别1
    ]
    # 调用预测函数，测试模型效果
    predict("model_multiclass.bin", test_vec)

