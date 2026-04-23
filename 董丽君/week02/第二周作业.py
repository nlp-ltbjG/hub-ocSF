
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个五分类任务
规律：x是一个5维向量，哪一维数字最大就属于第几类

"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层 (5, 5)
        # 多分类任务通常使用 CrossEntropyLoss，它内部包含了 Softmax，所以这里不需要显式添加激活层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本
# 随机生成一个5维向量，最大的数字所在的索引就是它的类别
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 注意：CrossEntropyLoss 的标签需要是 LongTensor (整数类型)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 统计各个类别的样本分布
    counts = [0] * 5
    for label in y:
        counts[label] += 1
    print(f"本次预测集中样本分布：{counts}")
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 (100, 5)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 分类类别数
    learning_rate = 0.01  # 学习率
    
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
            
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            
            watch_loss.append(loss.item())
        
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    
    # 保存模型
    torch.save(model.state_dict(), "model_multiclass.bin")
    
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

# 预测函数
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res)
        print("输入：%s, 预测类别：%d, 原始输出：%s" % (vec, pred_class, res.numpy()))

if __name__ == "__main__":
    main()
    # test_vec = [[0.1, 0.2, 0.9, 0.4, 0.5],
    #             [0.9, 0.1, 0.2, 0.3, 0.4],
    #             [0.1, 0.8, 0.2, 0.3, 0.4]]
    # predict("model_multiclass.bin", test_vec)

test_vec = [[0.1, 0.2, 0.9, 0.4, 0.3], [0.8, 0.1, 0.1, 0.0, 0.0]]
predict("model_multiclass.bin", test_vec)
