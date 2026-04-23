import torch
import torch.nn as nn
import torch.optim as optim

# ===================== 1. 超参数设置 =====================
INPUT_DIM = 10    # 输入向量维度
NUM_CLASSES = 10  # 分类数
BATCH_SIZE = 128  # 增大批次，更稳定
EPOCHS = 200      # 训练轮数
DEVICE = torch.device("cpu")

# ===================== 2. 数据生成 =====================
def generate_data(batch_size, input_dim):
    # 生成随机向量
    data = torch.randn(batch_size, input_dim).to(DEVICE)
    # 标签：最大值所在索引
    labels = torch.argmax(data, dim=1).to(DEVICE)
    return data, labels

# ===================== 3. 模型（加深一点，更容易学习） =====================
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ===================== 4. 初始化 =====================
model = Classifier(INPUT_DIM, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===================== 5. 训练（每轮多训练几步！） =====================
print("开始训练...")
model.train()

for epoch in range(EPOCHS):
    # 🔥 关键修复：每一轮训练 20 次，而不是只学 1 次！
    total_acc = 0
    for _ in range(20):
        data, labels = generate_data(BATCH_SIZE, INPUT_DIM)
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total_acc += (predicted == labels).sum().item() / BATCH_SIZE
    
    avg_acc = total_acc / 20
    if (epoch + 1) % 10 == 0:
        print(f'第 [{epoch+1}/{EPOCHS}] 轮 | 损失: {loss:.4f} | 准确率: {avg_acc*100:.2f}%')

# ===================== 6. 测试 =====================
print("\n训练完成！开始测试...")
model.eval()
with torch.no_grad():
    test_data, test_labels = generate_data(20, INPUT_DIM)
    outputs = model(test_data)
    _, predictions = torch.max(outputs, 1)

    print("\n真实标签:", test_labels.cpu().numpy())
    print("模型预测:", predictions.cpu().numpy())
    
    test_acc = (predictions == test_labels).sum().item() / len(test_labels)
    print(f'\n✅ 测试准确率: {test_acc*100:.2f}%')
