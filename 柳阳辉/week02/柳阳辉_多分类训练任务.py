import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch实现简单的找规律分类任务
规律：输入3维向量，哪一维数值最大就属于对应类别（1/2/3类）
"""

# 定义简单的分类模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim, class_num):
        super(SimpleModel, self).__init__()
        # 线性层：3维输入，3类输出
        self.layer = nn.Linear(input_dim, class_num)
        # 损失函数：交叉熵（多分类）
        self.loss_func = nn.CrossEntropyLoss()

    # 前向计算：有标签算损失，无标签返回预测值
    def forward(self, x, y=None):
        pred = self.layer(x)  # 计算预测值
        if y is not None:
            # 标签从1-3转0-2
            y = y - 1
            return self.loss_func(pred, y)
        else:
            return pred

# 生成单个样本：3维随机向量，最大值所在维度为类别
def make_one_sample():
    # 生成0-1之间的3个随机数
    x = np.random.rand(3)
    # 找最大值索引，+1转为1-3类
    label = np.argmax(x) + 1
    return x, label

# 生成批量样本
def make_dataset(sample_num):
    data_x = []
    data_y = []
    for _ in range(sample_num):
        x, y = make_one_sample()
        data_x.append(x)
        data_y.append(y)
    # 转为tensor格式
    return torch.FloatTensor(data_x), torch.LongTensor(data_y)

# 评估模型准确率
def test_model(model):
    model.eval()  # 切换到测试模式
    test_num = 100
    test_x, test_y = make_dataset(test_num)
    
    # 打印测试集各类别数量
    print("测试集样本分布：", end="")
    for i in range(1,4):
        num = (test_y == i).sum().item()
        print(f"类别{i}：{num} ", end="")
    print()
    
    # 不计算梯度，加快预测
    with torch.no_grad():
        pred_result = model(test_x)
        # 取最大值索引作为预测类别
        pred_label = torch.argmax(pred_result, dim=1)
        test_y = test_y - 1  # 转0-2索引
    
    # 统计正确/错误数量
    correct = 0
    for p, t in zip(pred_label, test_y):
        if p == t:
            correct += 1
    acc = correct / test_num
    print(f"正确数：{correct}，正确率：{acc:.4f}\n")
    return acc

# 主训练流程
def train_main():
    # 基础参数
    train_round = 15  # 训练轮数
    batch_size = 15   # 每次训练的样本数
    total_train = 3000 # 总训练样本数
    input_dim = 3     # 输入维度
    class_num = 3     # 分类数
    learn_rate = 0.02 # 学习率

    # 创建模型
    model = SimpleModel(input_dim, class_num)
    # 优化器：Adam优化
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    # 记录训练过程
    train_log = []
    # 生成训练数据
    train_x, train_y = make_dataset(total_train)

    # 开始训练
    for round in range(train_round):
        model.train()  # 切换到训练模式
        loss_list = []
        
        # 按批次训练
        for batch in range(total_train // batch_size):
            # 截取当前批次数据
            start = batch * batch_size
            end = start + batch_size
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]
            
            # 计算损失
            loss = model(batch_x, batch_y)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()
            
            loss_list.append(loss.item())
        
        # 每轮训练完测试一次
        avg_loss = np.mean(loss_list)
        print(f"第{round+1}轮训练，平均损失：{avg_loss:.4f}")
        acc = test_model(model)
        train_log.append([acc, avg_loss])
    
    # 保存模型
    torch.save(model.state_dict(), "3class_model.pth")
    
    # 绘制训练曲线
    plt.figure(figsize=(8,4))
    plt.plot([i[0] for i in train_log], label="正确率")
    plt.plot([i[1] for i in train_log], label="平均损失")
    plt.xlabel("训练轮数")
    plt.ylabel("数值")
    plt.legend()
    plt.show()

# 用训练好的模型做预测
def do_predict(model_path, test_vec):
    # 初始化模型
    model = SimpleModel(3, 3)
    # 加载训练好的参数
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 测试模式
    
    # 预测
    with torch.no_grad():
        pred = model(torch.FloatTensor(test_vec))
        pred_label = torch.argmax(pred, dim=1)
    
    # 打印结果
    for vec, p in zip(test_vec, pred_label):
        true_max = np.argmax(vec) + 1  # 真实最大值维度
        pred_class = p + 1             # 预测类别（转回1-3）
        print(f"输入向量：{vec}，真实最大维度：{true_max}，预测类别：{pred_class}")

# 主程序入口
if __name__ == "__main__":
    # 训练模型
    train_main()
    
    # 测试预测功能（取消注释可运行）
    # test_vectors = [[0.1, 0.8, 0.2], [0.9, 0.1, 0.3], [0.2, 0.3, 0.9]]
    # do_predict("3class_model.pth", test_vectors)
