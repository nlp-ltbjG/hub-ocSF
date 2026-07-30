import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead          # 每个头的维度

        # 将 Q、K、V 的线性变换合并为一个矩阵，提高效率
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, d_model]
        mask: [batch_size, seq_len] (padding mask) 或
              [batch_size, 1, seq_len] 等广播形状，True 表示需要屏蔽的位置
        """
        batch_size, seq_len, _ = x.shape

        # 线性变换并拆分成 Q、K、V
        qkv = self.qkv_proj(x)  # [B, L, 3*d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # [3, B, nhead, L, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个: [B, nhead, L, d_k]

        # 缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, nhead, L, L]

        if mask is not None:
            # 将 mask 扩展到注意力分数形状，需要屏蔽的位置填充 -inf
            # mask 形状一般为 [B, L] 或 [B, 1, 1, L] 等
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, nhead, L, L]
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, v)  # [B, nhead, L, d_k]

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)

        # 输出投影
        output = self.out_proj(attn_output)
        return output


class FeedForward(nn.Module):
    """两层全连接的前馈网络，中间使用 ReLU 激活"""
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层
    结构：x -> LayerNorm -> MultiHeadAttention -> 残差 -> LayerNorm -> FeedForward -> 残差
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        src: [batch_size, seq_len, d_model]
        src_mask: padding mask, 形状 [batch_size, seq_len] (True表示保留的位置)
        """
        # 自注意力子层
        attn_out = self.self_attn(src, mask=src_mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # 前馈子层
        ff_out = self.feed_forward(src)
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)

        return src


# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    nhead = 8
    dim_ff = 2048

    layer = TransformerEncoderLayer(d_model, nhead, dim_ff)

    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 构造一个 padding mask：假设有效长度为 [7, 5]，其余位置为 padding
    lengths = torch.tensor([7, 5])
    src_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        src_mask[i, :l] = True   # True 表示有效位置

    output = layer(x, src_mask=src_mask)
    print(output.shape)   # torch.Size([2, 10, 512])
