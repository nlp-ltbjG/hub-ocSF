import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 设置参数
input_dim = 5
num_classes = input_dim
epoch_num = 50
lr = 0.01

# 2. 生成数据
x = torch.randn(1000, input_dim)
y = torch.argmax(x, dim=1)

# 3. 定义模型
class Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

model = Model(input_dim, num_classes)

# 4. 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 5. 训练
for epoch in range(epoch_num):
    outputs = model(x)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"第{epoch+1}轮，loss：{loss.item():.4f}")

# 6. 测试准确率
with torch.no_grad():
    pred = model(x)
    pred_class = torch.argmax(pred, dim=1)
    acc = (pred_class == y).sum().item() / len(y)

print(f"\n训练完成！准确率：{acc*100:.2f}%")

# 7. 测试新样本
test_vec = torch.tensor([[0.1, 0.7, 0.2, 0.5, 0.3]])
pred_test = model(test_vec)
print("输入向量：", test_vec.numpy())
print("模型预测类别：", torch.argmax(pred_test, dim=1).item())
