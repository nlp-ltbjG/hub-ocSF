"""
任务描述:
- 输入: 固定长度为5个汉字的文本，文本中必定包含且仅包含一个“你”字。
- 输出: 分类标签0~4，分别对应“你”字出现在第1位、第2位、第3位、第4位、第5位。
- 实验模型: RNN 和 LSTM。
- 训练后使用自定义句子进行测试，输出预测类别及概率。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# 设置随机种子，保证结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ==================== 1. 构建中文词汇表 ====================
print("=" * 60)
print("第一步：构建词汇表")
print("=" * 60)

# 基础常用汉字列表，确保包含目标字“你”
common_chars = list("的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开手十里己方现所二力三行身实长面高物问从本定见两新已正明女利最等")

# 如果基础列表中没有“你”，手动加入（其实已经包含）
if '你' not in common_chars:
    common_chars.append('你')

# 通过Unicode编码范围补充更多汉字，模拟更丰富的词表
extra_chars = [chr(i) for i in range(0x4e00, 0x4e00 + 500) if chr(i) not in common_chars]

# 合并并去重
vocab = common_chars + extra_chars
vocab = list(dict.fromkeys(vocab))

# 限制词表大小，避免嵌入层过大
VOCAB_SIZE = min(1500, len(vocab))
vocab = vocab[:VOCAB_SIZE]

# 构建字符到索引、索引到字符的映射字典
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for ch, i in char2idx.items()}

# 记录特殊字符“你”对应的索引（虽然不直接使用，但用于数据生成）
NI_CHAR = '你'
NI_IDX = char2idx[NI_CHAR]

print(f"词表大小: {VOCAB_SIZE} 个字符（包含'{NI_CHAR}'，索引为 {NI_IDX}）")
print(f"前10个字符及索引: {[(idx2char[i],i) for i in range(10)]}")


# ==================== 2. 自定义数据集类 ====================
print("\n" + "=" * 60)
print("第二步：定义数据集类 PositionDataset")
print("=" * 60)

class PositionDataset(Dataset):
    """
    位置分类数据集。
    每条数据是一个长度为 seq_len 的字符序列（索引形式），其中恰好包含一个“你”字。
    标签为“你”在序列中的位置索引（从0开始），对应类别0~4。
    """
    def __init__(self, num_samples, seq_len=5):
        """
        参数:
            num_samples: 总样本数量（自动按位置平均分配）
            seq_len: 句子长度，默认为5（任务要求）
        """
        self.seq_len = seq_len
        self.data = []   # 存储索引序列
        self.labels = [] # 存储标签

        print(f"开始生成数据集，目标样本总数: {num_samples}")

        # 每个位置生成的样本数，保证类别均衡
        samples_per_class = num_samples // seq_len
        print(f"每个类别（位置0~{seq_len-1}）生成 {samples_per_class} 个样本")

        # 排除“你”的其他字符列表
        other_chars_pool = [ch for ch in vocab if ch != NI_CHAR]

        for pos in range(seq_len):  # pos = 0,1,2,3,4
            for _ in range(samples_per_class):
                # 随机选择 seq_len-1 个其他字符
                other_chars = random.choices(other_chars_pool, k=seq_len - 1)
                # 在 pos 位置插入“你”，构成完整文本序列
                seq = other_chars[:pos] + [NI_CHAR] + other_chars[pos:]
                # 将字符序列转换为对应的索引序列
                idx_seq = [char2idx[ch] for ch in seq]
                self.data.append(idx_seq)
                # 标签就是“你”所在的位置
                self.labels.append(pos)

        # 将数据和标签一一配对后随机打乱，消除生成时的顺序规律
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)

        print(f"数据集生成完毕，总样本数: {len(self.data)}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本。
        返回:
            input_tensor: 序列张量，形状 (seq_len,)，dtype=torch.long
            label_tensor: 类别标签张量，标量，dtype=torch.long
        """
        input_tensor = torch.tensor(self.data[idx], dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_tensor, label_tensor


# ==================== 3. 模型定义 ====================
print("\n" + "=" * 60)
print("第三步：定义模型 RNNClassifier")
print("=" * 60)

class RNNClassifier(nn.Module):
    """
    基于RNN或LSTM的序列分类模型。
    结构: 嵌入层 -> RNN/LSTM层 -> 取最后一个时间步输出 -> 全连接分类层。
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, rnn_type='RNN'):
        """
        参数:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入向量的维度
            hidden_dim: RNN/LSTM 隐藏状态的维度
            num_classes: 分类类别数
            rnn_type: 'RNN' 或 'LSTM'，决定循环层的类型
        """
        super(RNNClassifier, self).__init__()
        # 词嵌入层：将整数索引转换为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 根据指定的类型创建RNN或LSTM层
        if rnn_type == 'LSTM':
            # batch_first=True 表示输入张量形状为 (batch, seq, feature)
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        else:  # 默认使用经典RNN
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        # 全连接层：将隐藏状态映射到各类别的得分
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.rnn_type = rnn_type  # 保存类型标识

    def forward(self, x):
        """
        前向传播。
        参数:
            x: 输入序列索引，形状 (batch_size, seq_len)
        返回:
            logits: 未归一化的类别得分，形状 (batch_size, num_classes)
        """
        # 1. 嵌入
        emb = self.embedding(x)                      # (batch, seq_len, embed_dim)

        # 2. RNN/LSTM 处理
        if self.rnn_type == 'LSTM':
            out, (h_n, c_n) = self.rnn(emb)          # out: (batch, seq_len, hidden_dim)
        else:
            out, h_n = self.rnn(emb)                 # out 同上

        # 3. 取序列最后一个位置的输出，它包含了整个序列的信息
        last_output = out[:, -1, :]                  # (batch, hidden_dim)

        # 4. 线性分类层
        logits = self.fc(last_output)                # (batch, num_classes)
        return logits

print("模型定义完成：包含 Embedding -> RNN/LSTM -> FC 结构")


# ==================== 4. 训练函数 ====================
print("\n" + "=" * 60)
print("第四步：定义训练函数 train_model")
print("=" * 60)

def train_model(model, train_loader, test_loader, epochs, lr, device):
    """
    训练指定的模型并返回测试准确率。
    参数:
        model: 待训练的模型
        train_loader: 训练集数据加载器
        test_loader: 测试集数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备 (cpu 或 cuda)
    返回:
        model: 训练好的模型
        final_test_acc: 最终测试准确率
    """
    criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器

    print(f"\n开始训练 {model.rnn_type} 模型，共 {epochs} 个周期，学习率 {lr}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # -------- 训练阶段 --------
        model.train()
        total_loss = 0.0        # 累计损失
        correct_train = 0       # 训练集预测正确数
        total_train = 0         # 训练集样本总数

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()                  # 清空梯度
            outputs = model(inputs)                # 前向传播
            loss = criterion(outputs, labels)      # 计算损失
            loss.backward()                        # 反向传播
            optimizer.step()                       # 更新参数

            # 累加损失（乘以批次大小得到总损失）
            total_loss += loss.item() * inputs.size(0)
            # 计算训练准确率
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_loss / total_train
        train_acc = correct_train / total_train

        # -------- 测试阶段 --------
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)
        test_acc = correct_test / total_test

        # 每个 epoch 输出一次训练状态（每5轮详细输出，第一个epoch也输出）
        if epoch == 1 or epoch % 5 == 0:
            print(f"周期 {epoch:2d}/{epochs} | 训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")

    final_test_acc = test_acc
    print(f"{model.rnn_type} 模型训练完成，最终测试准确率: {final_test_acc:.4f}")
    print("-" * 60)
    return model, final_test_acc


# ==================== 5. 测试单个句子函数 ====================
print("\n" + "=" * 60)
print("第五步：定义自定义句子预测函数 predict_sentence")
print("=" * 60)

def predict_sentence(model, sentence, device):
    """
    使用训练好的模型对给定的中文字符串进行预测。
    参数:
        model: 训练好的模型
        sentence: 长度为5的中文字符串
        device: 设备
    返回:
        pred_class: 预测的类别索引 (0~4)
        probs: 各类别的概率数组，长度为5
    """
    # 将句子中的字符转换为索引序列
    idx_list = []
    for ch in sentence:
        if ch in char2idx:
            idx_list.append(char2idx[ch])
        else:
            # 如果字符不在词表中，用一个随机的非“你”字索引代替，并给出警告
            print(f"警告：字符 '{ch}' 不在词表中，将替换为随机字符索引。")
            # 找一个不是“你”的索引，简单处理为0可能不合适，这里随机选一个
            fallback_idx = random.choice([i for i in range(VOCAB_SIZE) if i != NI_IDX])
            idx_list.append(fallback_idx)
    input_tensor = torch.tensor([idx_list], dtype=torch.long).to(device)  # 形状 (1, seq_len)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)                          # (1, num_classes)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()  # 转为概率
        pred_class = torch.argmax(logits, dim=1).item()       # 预测的类别

    return pred_class, probs


# ==================== 6. 主程序：数据准备、模型实例化、训练与测试 ====================
print("\n" + "=" * 60)
print("第六步：执行主程序")
print("=" * 60)

if __name__ == "__main__":
    # ---------- 超参数设置 ----------
    SEQ_LEN = 5                 # 文本长度
    EMBED_DIM = 128             # 词嵌入维度
    HIDDEN_DIM = 256            # 循环隐藏层维度
    NUM_CLASSES = 5             # 类别数（位置0~4）
    BATCH_SIZE = 64             # 批大小
    LEARNING_RATE = 0.001       # 学习率
    NUM_EPOCHS = 15             # 训练周期数（为了提高效率，设为15）
    TRAIN_SAMPLES = 10000       # 训练集总样本数
    TEST_SAMPLES = 2000         # 测试集总样本数

    # 选择运行设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ---------- 创建数据集 ----------
    print("\n>>> 生成训练集和测试集")
    train_dataset = PositionDataset(TRAIN_SAMPLES, SEQ_LEN)
    test_dataset = PositionDataset(TEST_SAMPLES, SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}，批次数: {len(train_loader)}")
    print(f"测试集大小: {len(test_dataset)}，批次数: {len(test_loader)}")

    # ---------- 分别训练RNN和LSTM模型 ----------
    trained_models = {}  # 保存训练好的模型

    for rnn_type in ['RNN', 'LSTM']:
        print(f"\n{'='*40}")
        print(f"准备训练 {rnn_type} 模型")
        print(f"{'='*40}")

        # 初始化模型
        model = RNNClassifier(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            rnn_type=rnn_type
        ).to(device)

        # 训练模型
        model, final_acc = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            device=device
        )

        # 保存模型（以便后续直接加载使用）
        model_path = f"{rnn_type}_position_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"{rnn_type} 模型已保存至: {model_path}")

        trained_models[rnn_type] = model

    # ---------- 使用自定义句子进行测试 ----------
    print("\n" + "=" * 60)
    print("第七步：使用自定义句子测试两个模型")
    print("=" * 60)

    # 精心设计一批测试句子，覆盖各种位置，并包括一些边界情况
    test_sentences = [
        "你天地人和",   # 预期类别: 0 (第1位)
        "人你天地和",   # 预期类别: 1 (第2位)
        "人天你地和",   # 预期类别: 2 (第3位)
        "人天地你和",   # 预期类别: 3 (第4位)
        "人天地和你",   # 预期类别: 4 (第5位)
        "学而时你之",   # 预期类别: 3 (第4位，“学而时你之”)
        "你我他它她",   # 预期类别: 0 (第1位)
        "快乐你每天",   # 预期类别: 2 (第3位)
        "一你二三四",   # 预期类别: 1 (第2位)
        "一二你三四",   # 预期类别: 2 (第3位)
        "天天你向上",   # 预期类别: 2 (第3位)
        "今天天气好",   # 注意：不包含“你”，但模型会被强制预测，观察结果
    ]

    for model_name in ['RNN', 'LSTM']:
        model = trained_models[model_name]
        print(f"\n{'='*30} {model_name} 模型预测结果 {'='*30}")
        print(f"句子 (5字)          | 预测类别 | 对应位置 | 概率分布 (类别0~4)")
        print("-" * 70)
        for sent in test_sentences:
            if len(sent) != 5:
                print(f"[跳过] '{sent}' 长度不为5，不符合任务要求")
                continue
            # 预测
            pred_class, probs = predict_sentence(model, sent, device)
            # 格式化概率输出，保留三位小数
            prob_str = np.array2string(probs, precision=3, suppress_small=True, separator=', ')
            print(f"'{sent}'              |    {pred_class}    | 第{pred_class+1}位  | {prob_str}")
        print("-" * 70)

    print("\n所有实验结束。")
