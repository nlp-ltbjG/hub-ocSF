import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 定义 Q, K, V 的线性映射层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性映射层
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        Q, K, V shape: (batch_size, num_heads, seq_len, d_k)
        """
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 如果提供了mask，将mask位置填充为极小值，softmax后其概率接近0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力权重乘以 V
        output = torch.matmul(attn_weights, V)
        return output

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # 1. 线性映射并分多头
        # shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 重塑维度以分离多头: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        # 然后转置以适应矩阵乘法: -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 合并多头
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. 经过输出线性层
        output = self.W_o(attn_output)
        return output


class PositionwiseFeedForward(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # 升维
        self.fc2 = nn.Linear(d_ff, d_model)   # 降维
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() # 目前主流使用GELU替代原始的ReLU

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    """
    完整的Transformer层 (Pre-LN 版本)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()
        
        # 子层1：多头自注意力
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # 子层2：前馈神经网络
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x shape: (batch_size, seq_len, d_model)
        mask shape: (batch_size, 1, seq_len, seq_len) 或 (batch_size, seq_len, seq_len)
        """
        # --- 注意力子层 ---
        # Pre-LN: 先Norm，再Attention，再残差相加
        residual = x
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, mask)
        attn_output = self.dropout1(attn_output)
        x = residual + attn_output
        
        # --- FFN子层 ---
        # Pre-LN: 先Norm，再FFN，再残差相加
        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        ffn_output = self.dropout2(ffn_output)
        x = residual + ffn_output
        
        return x


# ================= 测试代码 =================
if __name__ == "__main__":
    # 定义超参数
    batch_size = 2
    seq_len = 10
    d_model = 512       # 模型维度
    num_heads = 8       # 注意力头数
    d_ff = 2048         # FFN中间层维度
    dropout = 0.1
    
    # 实例化Transformer层
    transformer_layer = TransformerLayer(d_model, num_heads, d_ff, dropout)
    
    # 生成随机输入 (模拟词向量+位置编码后的输入)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 生成模拟的mask (例如: 1表示有效，0表示padding)
    # 这里假设没有padding，全1
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    
    # 前向传播
    output = transformer_layer(x, mask)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("Transformer层前向传播测试通过！")
