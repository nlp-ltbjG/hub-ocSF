# 作业内容：完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。


# ai示例代码
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# # ==========================================
# # 1. 配置参数 (Hyperparameters)
# # ==========================================
# INPUT_DIM = 5       # 输入向量的维度。因为我们要判断哪一维最大，所以也是类别的数量 (0, 1, 2, 3, 4 共5类)
# HIDDEN_DIM = 64     # 隐藏层神经元的数量。这个数字越大，模型学习能力越强，但也更容易过拟合
# TOTAL_SAMPLES = 2000 # 总共生成多少个样本用于训练
# BATCH_SIZE = 64     # 批大小。每次训练只给模型看64个样本，而不是全部2000个，这样训练更稳定且节省内存
# EPOCHS = 20         # 训练轮数。整个数据集会被模型反复学习20次
# LEARNING_RATE = 0.01 # 学习率。控制模型每次更新参数的步长，太大可能震荡，太小收敛慢

# # ==========================================
# # 2. 数据生成函数
# # ==========================================
# def build_sample(input_dim):
#     """
#     生成单个样本：
#     1. 创建一个随机向量 x
#     2. 找到最大值所在的索引，作为标签 y
#     """
#     # np.random.random 生成 [0, 1) 之间的随机浮点数
#     # .astype(np.float32) 确保数据类型与 PyTorch 默认浮点类型一致
#     x = np.random.random(input_dim).astype(np.float32)
    
#     # np.argmax 返回数组中最大值的索引位置
#     # 例如: x=[0.1, 0.8, 0.3] -> y=1 (因为0.8最大，索引为1)
#     y = np.argmax(x) 
    
#     return x, y

# def build_dataset(total_sample_num, input_dim):
#     """
#     构建完整的数据集
#     """
#     X = [] # 用于存放所有样本的特征向量
#     Y = [] # 用于存放所有样本的标签
    
#     for _ in range(total_sample_num):
#         x, y = build_sample(input_dim)
#         X.append(x)
#         Y.append(y)
    
#     # --- 关键步骤：转换为 PyTorch Tensor ---
    
#     # 特征 X 转换为 FloatTensor (浮点型)
#     # 形状变为: [样本总数, 输入维度]，例如 [2000, 5]
#     tensor_X = torch.FloatTensor(X)
    
#     # 标签 Y 转换为 LongTensor (长整型)
#     # 注意：PyTorch 的分类损失函数 CrossEntropyLoss 要求标签必须是 Long 类型，且是一维数组
#     # 形状变为: [样本总数]，例如 [2000]
#     tensor_Y = torch.LongTensor(Y)
    
#     return tensor_X, tensor_Y

# # ==========================================
# # 3. 定义神经网络模型
# # ==========================================
# class ClassifierModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(ClassifierModel, self).__init__()
        
#         # 第一层线性变换：将输入维度映射到隐藏层维度
#         # 作用：提取特征
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
        
#         # 第二层线性变换：将隐藏层维度映射到输出维度（即类别数）
#         # 作用：输出每个类别的得分 (Logits)
#         self.linear2 = nn.Linear(hidden_dim, output_dim)
        
#         # ReLU 激活函数：引入非线性，让模型能处理复杂关系
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         """
#         前向传播过程：数据从输入流向输出的路径
#         """
#         # 1. 先经过第一层线性层，再经过 ReLU 激活
#         # x 的形状变化: [Batch, Input_Dim] -> [Batch, Hidden_Dim]
#         x = self.relu(self.linear1(x))
        
#         # 2. 再经过第二层线性层，得到最终得分
#         # x 的形状变化: [Batch, Hidden_Dim] -> [Batch, Output_Dim]
#         # 注意：这里不要加 Softmax！因为后面的损失函数 CrossEntropyLoss 内部会自动处理
#         x = self.linear2(x)
        
#         return x

# # ==========================================
# # 4. 初始化模型、损失函数和优化器
# # ==========================================

# # 实例化模型
# # 输入5维，隐藏层64维，输出5维（对应5个类别）
# model = ClassifierModel(INPUT_DIM, HIDDEN_DIM, INPUT_DIM)

# # 定义损失函数
# # CrossEntropyLoss 结合了 LogSoftmax 和 NLLLoss
# # 它负责衡量“模型预测的概率分布”与“真实标签”之间的差距
# criterion = nn.CrossEntropyLoss()

# # 定义优化器
# # SGD (随机梯度下降) 是最基础的优化算法
# # model.parameters() 告诉优化器需要更新哪些参数（即模型中的权重和偏置）
# # lr 是学习率
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# # ==========================================
# # 5. 生成数据
# # ==========================================
# print("正在生成数据...")
# # 调用之前定义的函数，得到训练用的特征和标签
# train_X, train_Y = build_dataset(TOTAL_SAMPLES, INPUT_DIM)

# # ==========================================
# # 6. 训练循环 (核心部分)
# # ==========================================
# print("开始训练...")
# for epoch in range(EPOCHS):
#     # model.train() 表示开启训练模式
#     # 这会启用 Dropout 等只在训练时生效的层（虽然本例没用到 Dropout，但这是好习惯）
#     model.train()
    
#     total_loss = 0 # 用于记录当前 Epoch 的总损失
    
#     # 计算有多少个批次
#     num_batches = TOTAL_SAMPLES // BATCH_SIZE
    
#     # 遍历每一个批次
#     for i in range(num_batches):
#         # --- 数据切片：获取当前批次的数据 ---
#         start_idx = i * BATCH_SIZE
#         end_idx = start_idx + BATCH_SIZE
        
#         # 从总数据中切出一小块
#         batch_x = train_X[start_idx:end_idx] # 当前批次的特征
#         batch_y = train_Y[start_idx:end_idx] # 当前批次的标签
        
#         # --- 第一步：前向传播 (Forward) ---
#         # 将数据传入模型，得到预测结果 (Logits)
#         y_pred = model(batch_x)
        
#         # 计算损失：比较预测值 y_pred 和真实标签 batch_y
#         loss = criterion(y_pred, batch_y)
        
#         # --- 第二步：反向传播 (Backward) ---
        
#         # 清空之前的梯度
#         # 因为 PyTorch 会累积梯度，如果不清零，梯度会越来越大导致错误
#         optimizer.zero_grad()
        
#         # 自动计算梯度
#         # loss.backward() 会根据链式法则，计算损失函数对每个参数的导数
#         loss.backward()
        
#         # --- 第三步：参数更新 (Step) ---
#         # 根据计算出的梯度和学习率，更新模型的权重
#         optimizer.step()
        
#         # 累加损失值，用于打印观察
#         # .item() 将 Tensor 转换为普通的 Python 数字
#         total_loss += loss.item()
    
#     # 每训练 5 轮，打印一次平均损失，观察模型是否在进步
#     if (epoch + 1) % 5 == 0:
#         avg_loss = total_loss / num_batches
#         print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# # ==========================================
# # 7. 验证模型效果
# # ==========================================
# print("\n正在验证模型...")

# # model.eval() 表示开启评估模式
# # 这会关闭 Dropout 等层，确保预测结果稳定
# model.eval()

# # with torch.no_grad() 上下文管理器
# # 在验证/测试时，我们不需要计算梯度，这样可以节省内存并加速计算
# with torch.no_grad():
#     # 生成100个新的测试样本
#     test_X, test_Y = build_dataset(100, INPUT_DIM)
    
#     # 让模型对测试数据进行预测
#     predictions = model(test_X)
    
#     # torch.max 返回两个值：(最大值, 最大值的索引)
#     # dim=1 表示在每一行（每个样本）中寻找最大值
#     # _ 代表我们不需要最大值具体是多少，只需要它的索引 (即预测的类别)
#     _, predicted_classes = torch.max(predictions, 1)
    
#     # 计算准确率
#     # predicted_classes == test_Y 会得到一个布尔数组 (True/False)
#     # .sum() 计算 True 的个数（即预测正确的个数）
#     # .item() 转换为数字
#     correct = (predicted_classes == test_Y).sum().item()
#     accuracy = correct / len(test_Y)
    
#     print(f"测试集准确率: {accuracy * 100:.2f}%")

#     # 打印前5个样本的详细对比，方便直观理解
#     print("\n示例预测详情:")
#     for i in range(5):
#         print(f"输入向量: {test_X[i].numpy()}")
#         print(f"  -> 真实类别: {test_Y[i].item()}")
#         print(f"  -> 预测类别: {predicted_classes[i].item()}")
#         print("-" * 20)


#个人提交代码
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 配置参数 (Hyperparameters)
INPUT_DIM = 5       
HIDDEN_DIM = 64     
TOTAL_SAMPLES = 2000 
BATCH_SIZE = 64     
EPOCHS = 20         
LEARNING_RATE = 0.01 

# 2. 数据生成函数
def build_sample(input_dim):
    """
    生成单个样本：
    1. 创建一个随机向量 x
    2. 找到最大值所在的索引，作为标签 y
    """
    # TODO: 生成 [0, 1) 之间的随机浮点数，维度为 input_dim，类型为 float32
    x = np.random.random(input_dim).astype(np.float32)
    
    # TODO: 返回数组中最大值的索引位置作为标签
    y = np.argmax(x)
    
    return x, y

def build_dataset(total_sample_num, input_dim):
    """
    构建完整的数据集
    """
    X = [] 
    Y = [] 
    
    for _ in range(total_sample_num):
        x, y = build_sample(input_dim)
        X.append(x)
        Y.append(y)
    
    # --- 关键步骤：转换为 PyTorch Tensor ---
    
    # TODO: 将列表 X 转换为 FloatTensor
    tensor_X = torch.FloatTensor(X)
    
    # TODO: 将列表 Y 转换为 LongTensor (注意分类任务标签类型要求)
    tensor_Y = torch.LongTensor(Y)
    
    return tensor_X, tensor_Y

# 3. 定义神经网络模型
class ClassifierModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassifierModel, self).__init__()
        
        # TODO: 定义第一层线性变换 (输入 -> 隐藏层)
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        
        # TODO: 定义第二层线性变换 (隐藏层 -> 输出层)
        self.linear2 = nn.Linear(hidden_dim,output_dim) 
        
        # TODO: 定义 ReLU 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播过程
        """
        # TODO: 先经过 linear1，再经过 relu 激活
        x = self.relu(self.linear1(x))
        
        # TODO: 再经过 linear2 得到最终得分 (Logits)
        # 注意：这里不要加 Softmax！
        x = self.linear2(x)
        
        return x

# 4. 初始化模型、损失函数和优化器

# TODO: 实例化模型，传入输入、隐藏、输出维度
model = ClassifierModel(INPUT_DIM,HIDDEN_DIM,output_dim=5)

# TODO: 定义多分类损失函数 (CrossEntropyLoss)
criterion = nn.CrossEntropyLoss()

# TODO: 定义优化器 (SGD)，传入模型参数和学习率
optimizer = optim.SGD(model.parameters(),lr=LEARNING_RATE)

# 5. 生成数据
print("正在生成数据...")
train_X, train_Y = build_dataset(TOTAL_SAMPLES, INPUT_DIM)


# 6. 训练循环 (核心部分)
print("开始训练...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0 
    num_batches = TOTAL_SAMPLES // BATCH_SIZE
    
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        
        batch_x = train_X[start_idx:end_idx] 
        batch_y = train_Y[start_idx:end_idx]
        
        # --- 第一步：前向传播 ---
        y_pred = model(batch_x)
        
        # TODO: 计算损失
        loss = criterion(y_pred,batch_y)
        
        # --- 第二步：反向传播 ---

        # TODO: 清空之前的梯度 (非常重要！)
        optimizer.zero_grad()
        
        # TODO: 自动计算梯度
        loss.backward()

        # --- 第三步：参数更新 ---

        # TODO: 根据梯度更新权重
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")


# 7. 验证模型效果
print("\n正在验证模型...")
model.eval()

# with torch.no_grad() 上下文管理器
# 在验证/测试时，我们不需要计算梯度，这样可以节省内存并加速计算
with torch.no_grad():
    # 生成100个新的测试样本
    test_X, test_Y = build_dataset(100, INPUT_DIM)
    
    # 让模型对测试数据进行预测
    predictions = model(test_X)
    
    # torch.max 返回两个值：(最大值, 最大值的索引)
    # dim=1 表示在每一行（每个样本）中寻找最大值
    # _ 代表我们不需要最大值具体是多少，只需要它的索引 (即预测的类别)
    _, predicted_classes = torch.max(predictions, 1)
    
    # 计算准确率
    # predicted_classes == test_Y 会得到一个布尔数组 (True/False)
    # .sum() 计算 True 的个数（即预测正确的个数）
    # .item() 转换为数字
    correct = (predicted_classes == test_Y).sum().item()
    accuracy = correct / len(test_Y)
    
    print(f"测试集准确率: {accuracy * 100:.2f}%")

    # 打印前5个样本的详细对比，方便直观理解
    print("\n示例预测详情:")
    for i in range(5):
        print(f"输入向量: {test_X[i].numpy()}")
        print(f"  -> 真实类别: {test_Y[i].item()}")
        print(f"  -> 预测类别: {predicted_classes[i].item()}")
        print("-" * 20)
