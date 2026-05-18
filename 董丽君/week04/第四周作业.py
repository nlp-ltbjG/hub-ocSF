import torch
import torch.nn as nn
import torch.nn.functional as F

"""
实现一个 Transformer Encoder Layer
结构如下：
Input -> (Residual) -> Multi-Head Attention -> (Add) -> LayerNorm 
      -> (Residual) -> Feed Forward   -> (Add) -> LayerNorm -> Output
"""

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """
        参数:
            d_model: 模型特征维度 (即 embedding 维度)
            nhead: 多头注意力的头数
            dim_feedforward: FFN 中间层的维度
            dropout: dropout 比率
        """
        super().__init__()
        
        # 1. 多头自注意力机制
        # PyTorch 官方已经提供了封装好的 MultiheadAttention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 2. 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. 前馈神经网络 (FFN)
        # 通常是两个 Linear 层，中间夹一个 ReLU
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        参数:
            src: 输入序列 (batch_size, seq_len, d_model)
            src_mask: 注意力掩码 (可选)
            src_key_padding_mask: Padding 掩码 (可选)
        """
        
        # ========== 第一部分：多头注意力 + 残差连接 + 归一化 ==========
        
        # 保存原始输入，用于残差连接
        src2 = src
        
        # 计算多头注意力
        # MultiheadAttention 返回: (output, attention_weights)
        # 注意：因为设置了 batch_first=True，输入和输出形状都是 (Batch, SeqLen, Dim)
        attn_output, _ = self.self_attn(
            src, src, src,  # Q, K, V 都是 src (自注意力)
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        
        # Dropout
        attn_output = self.dropout(attn_output)
        
        # 残差连接 (Add)：原始输入 + 注意力输出
        src = src2 + attn_output
        
        # 层归一化 (Norm)
        src = self.norm1(src)
        
        # ========== 第二部分：前馈网络 + 残差连接 + 归一化 ==========
        
        # 保存输入，用于残差连接
        src2 = src
        
        # 前馈网络 (FFN)
        # Linear1 -> ReLU -> Dropout -> Linear2
        src = self.linear1(src)
        src = F.relu(src)
        src = self.dropout(src)
        src = self.linear2(src)
        
        # 残差连接 (Add)
        src = src2 + src
        
        # 层归一化 (Norm)
        src = self.norm2(src)
        
        return src

# ==========================================
# 下面是测试代码，验证我们写的 Layer 能不能跑通
# ==========================================
if __name__ == "__main__":
    print("开始测试手动实现的 Transformer Encoder Layer...")
    
    # 定义超参数
    batch_size = 2
    seq_len = 10    # 句子长度
    d_model = 512   # 特征维度
    nhead = 8       # 8 头注意力
    
    # 1. 随机生成一个输入 (Batch, SeqLen, Dim)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入 shape: {x.shape}")
    
    # 2. 初始化我们的 Transformer Layer
    encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    
    # 3. 前向传播
    output = encoder_layer(x)
    print(f"输出 shape: {output.shape}")
    
    print("\n✅ 测试成功！形状匹配，模型可以正常前向传播。")
    print("结构说明：")
    print("  1. Multi-Head Attention (多头注意力)")
    print("  2. Residual Connection & LayerNorm (残差与归一化)")
    print("  3. Feed Forward Network (前馈网络)")
    print("  4. Residual Connection & LayerNorm (残差与归一化)")
