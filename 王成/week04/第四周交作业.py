"""
使用PyTorch实现Transformer编码器层
包含：多头注意力机制、位置编码、前馈神经网络、层归一化等组件
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    由于Transformer没有像RNN那样的序列处理顺序，需要添加位置信息来区分不同位置的token
    使用正弦和余弦函数来生成位置编码
    """
    def __init__(self, d_model, max_len=5000):
        """
        参数:
            d_model: 词向量维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算不同维度的除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).floatdddd() * (-math.log(10000.0) / d_model))
        
        # 应用正弦函数到偶数维度
        pe[:, 0::2] = torch.sin(position * div_term)
        # 应用余弦函数到奇数维度
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为buffer，不会作为模型参数更新
        self.register_buffer('pe', pe.unsqueeze(0))  # 形状: (1, max_len, d_model)
        
    def forward(self, x):
        """
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        返回:
            添加位置编码后的张量
        """
        # x.size(1) 是当前序列长度
        # 将位置编码添加到输入上，并应用dropout
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    允许模型同时关注来自不同位置的不同表示子空间的信息
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 定义线性变换层：用于生成Q, K, V和输出
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        参数:
            query, key, value: 形状均为 (batch_size, seq_len, d_model)
            mask: 可选的掩码，用于遮蔽某些位置
        返回:
            多头注意力输出，形状为 (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # 1. 线性变换并分成多个头
        # 形状变换: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力分数
        # attention_scores: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. 应用掩码（如果有）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 4. 应用softmax得到注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 5. 使用注意力权重加权value
        # context: (batch_size, num_heads, seq_len, d_k)
        context = torch.matmul(attention_probs, V)
        
        # 6. 合并多头输出
        # 形状变换: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 7. 最终线性变换
        output = self.W_o(context)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    位置级前馈神经网络
    对每个位置独立应用相同的MLP
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度（通常是d_model的4倍）
            dropout: dropout概率
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        返回:
            前馈网络输出，形状为 (batch_size, seq_len, d_model)
        """
        # 应用线性变换 -> ReLU激活 -> dropout -> 线性变换
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    包含多头自注意力和前馈神经网络两个子层，每个子层都有残差连接和层归一化
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 两个层归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, mask=None):
        """
        参数:
            src: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 可选的掩码
        返回:
            编码器层输出，形状为 (batch_size, seq_len, d_model)
        """
        # 1. 多头自注意力子层
        # 残差连接 + 层归一化 (Post-LN方式)
        src2 = self.self_attn(src, src, src, mask)  # 自注意力：Q=K=V=src
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 层归一化
        
        # 2. 前馈神经网络子层
        # 残差连接 + 层归一化 (Post-LN方式)
        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)  # 残差连接
        src = self.norm2(src)  # 层归一化
        
        return src


class TransformerEncoder(nn.Module):
    """
    完整的Transformer编码器
    由多个编码器层堆叠而成
    """
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
            max_len: 最大序列长度
        """
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # 堆叠多个编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, mask=None):
        """
        参数:
            src: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 可选的掩码
        返回:
            编码器输出，形状为 (batch_size, seq_len, d_model)
        """
        # 添加位置编码
        output = self.positional_encoding(src)
        output = self.dropout(output)
        
        # 通过所有编码器层
        for layer in self.layers:
            output = layer(output, mask)
        
        # 最后一层归一化
        output = self.norm(output)
        
        return output


# ==========================================
# 应用示例
# ==========================================

def example_usage():
    """
    演示如何使用实现的Transformer编码器
    """
    print("=== Transformer实现示例 ===\n")
    
    # 设置模型参数
    d_model = 512      # 模型维度
    num_heads = 8      # 注意力头数
    num_layers = 3     # 编码器层数
    d_ff = 2048        # 前馈网络维度
    batch_size = 4     # 批次大小
    seq_len = 32       # 序列长度
    
    # 创建Transformer编码器
    transformer = TransformerEncoder(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1
    )
    
    # 创建随机输入数据（模拟词嵌入）
    # 在实际应用中，这里应该是词嵌入矩阵
    src = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {src.shape}")
    
    # 创建掩码（可选）- 这里创建一个全1掩码，表示所有位置都可见
    mask = torch.ones(batch_size, 1, 1, seq_len)
    print(f"掩码形状: {mask.shape}\n")
    
    # 前向传播
    output = transformer(src, mask)
    print(f"输出形状: {output.shape}")
    print(f"输出均值: {output.mean().item():.4f}")
    print(f"输出标准差: {output.std().item():.4f}\n")
    
    # 测试不同组件
    print("=== 各组件单独测试 ===\n")
    
    # 测试位置编码
    pos_enc = PositionalEncoding(d_model)
    test_input = torch.randn(2, 10, d_model)
    pos_output = pos_enc(test_input)
    print(f"位置编码输入形状: {test_input.shape}")
    print(f"位置编码输出形状: {pos_output.shape}")
    print(f"位置编码后输入与输出差值均值: {(pos_output - test_input).mean().item():.4f}\n")
    
    # 测试多头注意力
    mha = MultiHeadAttention(d_model, num_heads)
    q = k = v = torch.randn(batch_size, seq_len, d_model)
    attn_output = mha(q, k, v)
    print(f"多头注意力输出形状: {attn_output.shape}\n")
    
    # 测试前馈网络
    pff = PositionwiseFeedForward(d_model, d_ff)
    ff_output = pff(torch.randn(batch_size, seq_len, d_model))
    print(f"前馈网络输出形状: {ff_output.shape}\n")
    
    print("=== 组件数量统计 ===")
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Transformer编码器总参数量: {total_params:,}")
    
    # 打印每个编码器层的参数数量
    for i, layer in enumerate(transformer.layers):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"第{i+1}层参数量: {layer_params:,}")


if __name__ == "__main__":
    example_usage()
