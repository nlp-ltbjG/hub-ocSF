# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 固定随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==================== 1. 自行构造数据集（找规律）====================
def build_sample():
    """生成一个样本：5维向量及其最大值所在的索引"""
    x = np.random.random(5)  # 生成5维随机向量
    max_index = np.argmax(x)  # 规律：找出最大值的索引
    return x, max_index

def build_dataset(total_sample_num):
    """构建总样本数为 total_sample_num 的数据集"""
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)  # shape: [total_sample_num, 5]
    Y_tensor = torch.LongTensor(Y)   # shape: [total_sample_num]
    return X_tensor, Y_tensor

# ==================== 2. 定义神经网络模型 ====================
class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 16),  # 5维输入到16维隐藏层
            nn.ReLU(),
            nn.Linear(16, output_size)  # 16维隐藏层到5维输出（5个类别）
        )
        
    def forward(self, x):
        return self.layer_stack(x)

# ==================== 3. 评估模型准确率 ====================
def evaluate(model, test_x, test_y):
    model.eval()
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(test_x)
        for y_p, y_t in zip(y_pred, test_y):
            # torch.max返回最大值及其索引，我们取索引（即预测的类别）
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    accuracy = correct / (correct + wrong)
    print(f"测试集准确率: {accuracy:.4f} ({correct}/{correct+wrong})")
    return accuracy

# ==================== 4. 主训练函数 ====================
def main():
    # 超参数配置
    input_size = 5      # 输入向量维度
    output_size = 5     # 输出类别数（0,1,2,3,4）
    epoch_num = 50      # 训练轮数
    batch_size = 20     # 每批样本数
    train_sample_num = 5000  # 训练样本总数
    test_sample_num = 500    # 测试样本总数
    learning_rate = 0.001
    
    # 创建模型
    model = TorchModel(input_size, output_size)
    
    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失，适用于多分类
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 构建训练集和测试集
    train_x, train_y = build_dataset(train_sample_num)
    test_x, test_y = build_dataset(test_sample_num)
    
    # 使用DataLoader进行批量加载
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 记录训练过程中的loss和acc
    log = []
    
    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()  # 梯度清零
            y_pred = model(batch_x)  # 前向传播
            loss = loss_fn(y_pred, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            watch_loss.append(loss.item())
        
        avg_loss = np.mean(watch_loss)
        print(f"Epoch {epoch+1}/{epoch_num}, 平均Loss: {avg_loss:.6f}")
        
        # 每5轮评估一次
        if (epoch + 1) % 5 == 0:
            acc = evaluate(model, test_x, test_y)
            log.append([avg_loss, acc])
    
    # 保存模型
    torch.save(model.state_dict(), "max_classifier_model.pth")
    
    # 绘制训练过程
    plt.plot(range(len(log)), [l[1] for l in log], label="准确率")
    plt.plot(range(len(log)), [l[0] for l in log], label="Loss")
    plt.legend()
    plt.xlabel("Epoch (每5轮)")
    plt.title("模型训练过程")
    plt.show()

# ==================== 5. 使用训练好的模型进行预测 ====================
def predict(model_path, input_vec):
    """对新的5维向量进行预测"""
    input_size = 5
    output_size = 5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 修复1
    model.eval()
    
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    
    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()
        # 修复2：正确格式化数组输出
        print(f"输入向量: {np.round(vec, 4)}, 预测最大值索引: {predicted_class}")

if __name__ == "__main__":
    main()
    
    # 训练完成后，使用示例进行预测
    test_vectors = [
        [0.9788, 0.1522, 0.3108, 0.0350, 0.8892],  # 第0个最大
        [0.7496, 0.5524, 0.9575, 0.9552, 0.8489],  # 第2个最大
        [0.0079, 0.6748, 0.1362, 0.3467, 0.1987],  # 第1个最大
        [0.0934, 0.5941, 0.9257, 0.4156, 0.1358]   # 第2个最大
    ]
    
    predict("max_classifier_model.pth", test_vectors)
