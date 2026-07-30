# 导入 PyTorch 核心库
import torch
import torch.nn as nn       # 神经网络基类
import torch.nn.functional as F  # 激活函数、softmax 等函数
import math                # 用于注意力计算的根号运算

# ===================== 1. 核心：Transformer 层（总封装） =====================
# TransformerLayer 就是一个完整的编码器层，继承 PyTorch 基类 nn.Module
class TransformerLayer(nn.Module):
    # 初始化函数：定义层内所有子模块
    # d_model: 模型特征维度（固定不变，如 768）
    # n_head: 注意力头数
    # d_k/d_v: 每个头的 Q/K/V 维度
    # d_ff: 前馈网络中间升维维度
    # dropout: 防止过拟合的随机失活概率
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()  # 调用父类初始化
        
        # 子模块 1：多头自注意力
        self.multi_head_attention = MultiHeadAttention(d_model, n_head, d_k, d_v)
        # 子模块 2：前馈神经网络
        self.feed_forward = FeedForward(d_model, d_ff)
        # 子模块 3/4：两个层归一化（稳定训练）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 子模块 5/6：两个 Dropout（防止过拟合）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 保存参数（方便后续使用）
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.n_head = n_head

    # 前向传播：数据真正流动的路径
    def forward(self, x):
        # x 输入形状：[batch_size, seq_len, d_model]
        print(f"\n=== TransformerLayer 前向传播 ===")
        print(f"输入维度: {x.shape}")

        # ===================== 第一步：多头自注意力 =====================
        residual = x  # 残差连接：保存原始输入，后面要加回来
        # 进入多头自注意力计算
        attention_output = self.multi_head_attention(x)
        # 随机失活，防止过拟合
        attention_output = self.dropout1(attention_output)

        # ===================== 第二步：Add + LayerNorm（残差1） =====================
        # 残差相加：原始输入 + 注意力输出
        # 层归一化：让数据分布更稳定，加速训练
        x = self.norm1(residual + attention_output)
        print(f"\n===Add & LayerNorm 1 输出维度: {x.shape}===")

        # ===================== 第三步：前馈神经网络 =====================
        residual = x  # 再次保存残差
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)

        # ===================== 第四步：Add + LayerNorm（残差2） =====================
        x = self.norm2(residual + ff_output)

        # 打印维度（方便调试）
        print(f"Feed-Forward 输出维度: {ff_output.shape}")
        print(f"Add & LayerNorm 2 输出维度: {x.shape}")
        print(f"=== TransformerLayer 输出维度: {x.shape} ===\n")
        
        return x  # 输出维度和输入完全一样！


# ===================== 2. 核心子模块：多头自注意力 Multi-Head Attention =====================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head    # 头数
        self.d_k = d_k          # 每个头 Q/K 维度
        self.d_v = d_v          # 每个头 V 维度

        # 三个线性层：把输入分别投影成 Q（查询）、K（键）、V（值）
        # 输入维度 d_model → 输出维度 n_head*d_k（多头拼接后的总维度）
        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)

        # 输出线性层：把多头拼接的结果 → 映射回 d_model
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, x):
        # 解包输入维度：批量大小、序列长度、模型维度
        batch_size, seq_len, d_model = x.size()

        # ===================== 步骤1：线性投影 + 拆分成多头 =====================
        # 1. 用线性层得到 Q/K/V
        # 2. view：把总维度拆成 [头数, 单头维度]
        # 3. transpose：交换维度，变成 [batch, n_head, seq_len, d_k]，方便批量计算注意力
        q = self.w_q(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_head, self.d_v).transpose(1, 2)

        # 打印调试
        print(f"\n=== Multi-Head Self-Attention 层 ===")
        print(f"q维度：{q.shape}")
        print(f"k维度：{k.shape}")
        print(f"v维度：{v.shape}")

        # ===================== 步骤2：计算注意力分数 =====================
        # 公式：Attention(Q,K,V) = softmax( Q·K^T / √d_k ) · V
        # Q·K^T：计算每个词和所有词的相关性
        # /√d_k：防止数值过大，softmax 梯度消失
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # softmax：把分数变成 0~1 的权重，总和为 1
        attention_weights = F.softmax(scores, dim=-1)

        print(f"注意力权重维度:{attention_weights.shape}")

        # ===================== 步骤3：加权求和 V =====================
        # 用注意力权重对 V 加权，得到上下文表示
        context = torch.matmul(attention_weights, v)
        print(f"Context维度(多头分离): {context.shape}")

        # ===================== 步骤4：拼接多头 + 线性映射 =====================
        # 1. transpose 换回原始维度顺序
        # 2. contiguous：保证内存连续（PyTorch 要求）
        # 3. view：把多头拼接回 [batch, seq_len, n_head*d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        print(f"Context维度(合并后): {context.shape}")

        # 最后线性层，映射回 d_model
        output = self.fc(context)

        print(f"\n=== Multi-Head Self-Attention 输出维度: {output.shape}")
        return output


# ===================== 3. 核心子模块：前馈网络 Feed Forward =====================
# 结构：线性升维 → GELU 激活 → 线性降维（还原维度）
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        # 第一层：d_model → d_ff（升维，提取更丰富特征）
        self.fc1 = nn.Linear(d_model, d_ff)
        # 第二层：d_ff → d_model（降维，恢复原始维度）
        self.fc2 = nn.Linear(d_ff, d_model)
        # 激活函数：GELU（比 ReLU 更平滑，Transformer 常用）
        self.activation = nn.GELU()

    def forward(self, x):
        print(f"\n=== Feed-Forward Network 层===")
        # 前向传播流程
        x = self.fc1(x)
        print(f"FeedForward FC1 输出维度: {x.shape}")
        
        x = self.activation(x)
        print(f"FeedForward GELU激活 输出维度: {x.shape}")
        
        x = self.fc2(x)
        print(f"FeedForward 输出维度: {x.shape}")
        return x


# ===================== 4. 测试代码 =====================
if __name__ == "__main__":
    # 超参数（和 BERT-base 配置一致）
    d_model = 768    # 模型固定维度
    n_head = 12      # 12 个注意力头
    d_k = 64         # 单头 Q/K 维度 12*64=768
    d_v = 64         # 单头 V 维度 12*64=768
    d_ff = 3072      # 前馈中间维度 4倍升维
    dropout = 0.1    # 失活概率

    # 创建一个 Transformer 层
    transformer_layer = TransformerLayer(d_model, n_head, d_k, d_v, d_ff, dropout)

    # 构造测试输入
    # 形状：[批量大小 batch, 序列长度 seq_len, 模型维度 d_model]
    batch_size = 2
    seq_len = 10
    input_data = torch.randn(batch_size, seq_len, d_model)

    # 运行前向传播
    output = transformer_layer(input_data)

    # 打印结果
    print(f"输入形状: {input_data.shape}")
    print(f"输出形状: {output.shape}")
    print("Transformer层实现成功！")
