import numpy as np
import math

# （本周第四周作业题目：）
# 尝试用pytorch实现一个transformer层。

class TransformerEncoderLayer:
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout_rate=0.1):
        """
        初始化 Transformer Encoder Layer
        
        Args:
            d_model: 隐藏层维度 (embedding dim)
            num_heads: 注意力头数
            d_ff: 前馈网络中间层维度 (通常是 d_model 的 4 倍)
            dropout_rate: Dropout 比率 (这里仅做占位，NumPy实现中暂不执行随机丢弃以简化逻辑)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # 确保 d_model 能被 num_heads 整除
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads  # 每个头的维度

        # --- 1. 初始化 Self-Attention 的权重 (Q, K, V) ---
        # W_q, W_k, W_v: [d_model, d_model]
        self.W_q = self._init_weights(d_model, d_model)
        self.W_k = self._init_weights(d_model, d_model)
        self.W_v = self._init_weights(d_model, d_model)
        
        # W_o: 输出投影矩阵 [d_model, d_model]
        self.W_o = self._init_weights(d_model, d_model)

        # --- 2. 初始化 LayerNorm 1 (Attention 后) 的参数 ---
        self.ln1_gamma = np.ones((d_model,))  # scale
        self.ln1_beta = np.zeros((d_model,))  # shift

        # --- 3. 初始化 Feed-Forward Network 的权重 ---
        # FFN 结构: Linear -> ReLU/GELU -> Linear
        # W1: [d_model, d_ff], W2: [d_ff, d_model]
        self.W_ff1 = self._init_weights(d_model, d_ff)
        self.W_ff2 = self._init_weights(d_ff, d_model)

        # --- 4. 初始化 LayerNorm 2 (FFN 后) 的参数 ---
        self.ln2_gamma = np.ones((d_model,))
        self.ln2_beta = np.zeros((d_model,))

    def _init_weights(self, fan_in, fan_out):
        """
        Xavier 初始化: 保持方差一致
        """
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))

    def layer_norm(self, x, gamma, beta):
        """
        Layer Normalization
        Args:
            x: [seq_len, d_model]
            gamma: [d_model]
            beta: [d_model]

        总结图示
        对于输入矩阵中的每一行（即每一个词向量）：
        减去均值 -> 中心移到 0
        除以标准差 -> 宽度缩放到 1
        乘以 Gamma -> 拉宽或压窄（学习最佳尺度）
        加上 Beta -> 上下移动（学习最佳位置）
        """

        
        """
        1. axis=-1 是什么意思？
        在你的代码语境中： 输入 x 的形状是 [seq_len, d_model]（例如 [10, 512]）。
        axis=0 代表 seq_len（句子长度/Token 数量）。
        axis=1 或 axis=-1 代表 d_model（隐藏层维度/特征数量）。
        操作效果： np.mean(x, axis=-1) 意味着：沿着特征维度计算均值。 
        即：对每一个 Token（每一行），计算它所有 512 个特征值的平均值。
        
        2. keepdims=True 是什么意思？ 含义：保持维度不变（Keep Dimensions）
        沿最后一个维度 (d_model) 计算均值和方差
        """
        
        # 均值
        mean = np.mean(x, axis=-1, keepdims=True)
        # 方差
        var = np.var(x, axis=-1, keepdims=True)
        
        # 归一化: (x - mean) / sqrt(var + epsilon)
        """
        epsilon ($\epsilon$): 平滑项。
        值: 1e-8 (非常小的数)。
        作用: 防止分母为 0。如果某个 token 的所有特征值都一样（方差为 0），除以 0 会导致错误。
        加上这个微小数值可以保证计算稳定。"""
        epsilon = 1e-8
        """
        x_norm: 标准化后的数据。
        公式: $(x - \text{mean}) / \sqrt{\text{var} + \epsilon}$。
        含义: 将数据转换为均值为 0，方差为 1 的标准正态分布形态。此时数据失去了原有的尺度和位置信息。
        """
        x_norm = (x - mean) / np.sqrt(var + epsilon)
        # 缩放和平移
        """
        return gamma * x_norm + beta: 重构后的数据。
        含义: 虽然 x_norm 被强制变成了标准分布，
        但通过乘以 gamma 和加上 beta，模型可以学习恢复出最适合当前任务的数据分布。
        结果形状: [seq_len, d_model]，与输入 x 相同。
        """
        return gamma * x_norm + beta

    def softmax(self, x):
        """
        Softmax 函数, 防止溢出
        Args:
            x: [..., seq_len, seq_len]
        """
        # 减去最大值以提高数值稳定性
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def gelu(self, x):
        """
        GELU 激活函数
        """
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def multi_head_attention(self, x):
        """
        多头自注意力机制
        Args:
            x: [seq_len, d_model]
        Returns:
            output: [seq_len, d_model]
        """
        batch_size = 1 # NumPy 简单实现假设 batch=1, 或者 x 为 [seq_len, d_model]
        seq_len, d_model = x.shape
        
        # 1. 线性变换生成 Q, K, V
        # Q, K, V shape: [seq_len, d_model]
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # 2. 拆分多头: Reshape 并 Transpose
        # Shape change: [seq_len, d_model] -> [seq_len, num_heads, d_k] -> [num_heads, seq_len, d_k]
        Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        # 3. 计算注意力分数: Q * K^T / sqrt(d_k)
        # Q: [num_heads, seq_len, d_k], K^T: [num_heads, d_k, seq_len]
        # scores: [num_heads, seq_len, seq_len]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)

        # 4. Softmax 归一化
        attn_weights = self.softmax(scores)

        # 5. 加权求和: Attention * V
        # attn_weights: [num_heads, seq_len, seq_len], V: [num_heads, seq_len, d_k]
        # context: [num_heads, seq_len, d_k]
        context = np.matmul(attn_weights, V)

        # 6. 合并多头: Transpose 并 Reshape
        # context: [num_heads, seq_len, d_k] -> [seq_len, num_heads, d_k] -> [seq_len, d_model]
        context = context.transpose(1, 0, 2).reshape(seq_len, d_model)

        # 7. 输出线性投影
        output = np.dot(context, self.W_o)
        
        return output

    def feed_forward(self, x):
        """
        前馈神经网络
        Args:
            x: [seq_len, d_model]
        Returns:
            output: [seq_len, d_model]
        """
        # 第一层: Linear + GELU
        # x: [seq_len, d_model], W_ff1: [d_model, d_ff]
        hidden = np.dot(x, self.W_ff1)
        hidden = self.gelu(hidden)
        
        # 第二层: Linear
        # hidden: [seq_len, d_ff], W_ff2: [d_ff, d_model]
        output = np.dot(hidden, self.W_ff2)
        
        return output

    def forward(self, x):
        """
        Transformer Encoder Layer 前向传播
        Args:
            x: [seq_len, d_model] 输入嵌入向量
        Returns:
            output: [seq_len, d_model]
        """
        # --- Sub-layer 1: Multi-Head Self-Attention ---
        # 1.1 计算 Attention 输出
        attn_output = self.multi_head_attention(x)
        
        # 1.2 残差连接 (Add)
        # 注意: 原始 Transformer 论文中是 Add & Norm
        x_add_attn = x + attn_output
        
        # 1.3 层归一化 (Norm)
        x_norm1 = self.layer_norm(x_add_attn, self.ln1_gamma, self.ln1_beta)

        # --- Sub-layer 2: Feed-Forward Network ---
        # 2.1 计算 FFN 输出
        ff_output = self.feed_forward(x_norm1)
        
        # 2.2 残差连接 (Add)
        x_add_ff = x_norm1 + ff_output
        
        # 2.3 层归一化 (Norm)
        output = self.layer_norm(x_add_ff, self.ln2_gamma, self.ln2_beta)

        return output

# --- 测试代码 ---
if __name__ == "__main__":
    # 设置随机种子以便复现
    np.random.seed(42)

    # 参数配置
    d_model = 512
    num_heads = 8
    d_ff = 2048
    seq_len = 10
    
    # 创建模拟输入: [seq_len, d_model]
    # 在实际 BERT 中，这是 Embedding 层的输出
    input_data = np.random.randn(seq_len, d_model)
    
    print(f"Input shape: {input_data.shape}")

    # 实例化 Transformer Layer
    transformer_layer = TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    
    # 前向传播
    output_data = transformer_layer.forward(input_data)
    
    print(f"Output shape: {output_data.shape}")
    
    # 验证输出形状是否正确
    assert output_data.shape == (seq_len, d_model), "Output shape mismatch!"
    print("Transformer Layer Forward Pass Successful!")
    
    # 打印部分输出值查看
    print("\nFirst 5 values of the first token's output:")
    print(output_data[0, :5])
