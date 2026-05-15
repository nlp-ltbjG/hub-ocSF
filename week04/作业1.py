import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 1. 缩放点积注意力 =====================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v shape: [batch_size, n_heads, seq_len, d_k]
        d_k = q.size(-1)
        
        # 计算注意力分数 Q*K^T / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # 如果有 mask，把需要遮蔽的位置设为 -inf，softmax 后会变成 0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 输出 = 权重 * V
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# ===================== 2. 多头注意力 =====================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model    # 模型总维度
        self.n_heads = n_heads    # 注意力头数
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 3 个线性层把输入映射为 Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.fc = nn.Linear(d_model, d_model)  # 拼接后投影
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性变换 + 拆分成多头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)
        
        # 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最后线性层 + dropout
        output = self.dropout(self.fc(attn_output))
        return output, attn_weights

# ===================== 3. 前馈网络 =====================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # 升维
        self.fc2 = nn.Linear(d_ff, d_model)   # 降维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# ===================== 4. 完整 Transformer Encoder Layer =====================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # 两个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 子层 1：自注意力 + 残差 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 子层 2：前馈网络 + 残差 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x
