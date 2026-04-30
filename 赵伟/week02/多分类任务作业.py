# coding:utf8
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
多分类任务：输入为5维向量，哪一维的数字最大，就属于第几类（类别索引0~4）
使用交叉熵损失 + Softmax 输出 5 个类别的概率分布
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出维度改为类别数
        # 注意：交叉熵损失 nn.CrossEntropyLoss 内部已包含 Softmax，因此 forward 中不再加 Softmax
        self.loss = nn.CrossEntropyLoss()  # 多分类交叉熵损失

    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, num_classes)  未归一化的分数
        if y is not None:
            # CrossEntropyLoss 要求 y 是 LongTensor，形状 (batch_size,)，值为类别索引（非 one-hot）
            y = y.squeeze().long()  # 确保形状正确，并转为 long 类型
            return self.loss(logits, y)
        else:
            # 推理时返回概率分布（经过 Softmax）
            return torch.softmax(logits, dim=1)

def build_sample():
    """生成一个5维向量样本，标签为最大值的索引（0~4）"""
    x = np.random.random(5)
    label = np.argmax(x)  # 返回最大值的索引
    return x, label

def build_dataset(total_sample_num):
    """批量生成样本"""
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])  # 保持二维形状，便于后续处理
    return torch.FloatTensor(X), torch.FloatTensor(Y)  # 标签暂时用 Float，forward 中会转为 long

def evaluate(model):
    """评估模型在100个测试样本上的准确率"""
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 统计各类别样本数（可选）
    y_np = y.squeeze().numpy().astype(int)
    class_counts = np.bincount(y_np, minlength=5)
    print(f"测试集类别分布: {class_counts} (共{test_sample_num}个样本)")

    correct = 0
    with torch.no_grad():
        y_pred_prob = model(x)           # (100, 5) 概率分布
        y_pred = torch.argmax(y_pred_prob, dim=1)  # (100,) 预测类别索引
        y_true = y.squeeze().long()      # (100,) 真实类别索引
        correct = (y_pred == y_true).sum().item()
    accuracy = correct / test_sample_num
    print(f"正确预测个数: {correct}, 正确率: {accuracy:.4f}")
    return accuracy

def main():
    # 超参数
    epoch_num = 10          # 训练轮数
    batch_size = 20         # 批大小
    train_sample = 5000     # 训练集样本总数
    input_size = 5          # 输入维度
    num_classes = 5         # 类别数
    learning_rate = 0.01    # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 生成训练数据
    train_x, train_y = build_dataset(train_sample)

    # 训练循环
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        num_batches = train_sample // batch_size
        for batch_index in range(num_batches):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)          # 计算交叉熵损失
            optim.zero_grad()           # 梯度清零
            loss.backward()             # 反向传播
            optim.step()                # 更新参数
            watch_loss.append(loss.item())
        avg_loss = np.mean(watch_loss)
        print(f"=========\n第 {epoch+1} 轮平均 loss: {avg_loss:.6f}")
        acc = evaluate(model)           # 评估准确率
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "model_multiclass.bin")
    print("模型已保存至 model_multiclass.bin")

    # 绘制训练曲线
    plt.figure()
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training Curve (5-Class Classification)")
    plt.show()

def predict(model_path, input_vec):
    """使用训练好的模型进行预测"""
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vec)
        prob = model(input_tensor)          # (batch, 5) 概率
        pred_class = torch.argmax(prob, dim=1)  # 预测类别
    for vec, p_dist, pred in zip(input_vec, prob, pred_class):
        print(f"输入: {vec}")
        print(f"  概率分布: {p_dist.numpy().round(4)}")
        print(f"  预测类别: {pred.item()}, 实际最大维度: {np.argmax(vec)}")
        print()

if __name__ == "__main__":
    main()
    # 测试预测
    test_vec = [
        [0.9, 0.1, 0.2, 0.3, 0.4],
        [0.1, 0.8, 0.1, 0.5, 0.2],
        [0.2, 0.3, 0.1, 0.4, 0.9],
        [0.3, 0.5, 0.8, 0.2, 0.1]
    ]
    print("\n===== 预测示例 =====")
    predict("model_multiclass.bin", test_vec)
