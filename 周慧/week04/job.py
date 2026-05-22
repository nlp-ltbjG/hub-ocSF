import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Q、K、V线性层
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # 输出线性层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape
        # 格式化qkv，并拆分多头
        qkv = (
            self.qkv(x)
            .reshape(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 计算注意力权重 MHA(x)
        attn_weights = (q @ k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ v
        # 拼接多头结果
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, embedding_dim)
        # 输出线性层
        output = self.out_proj(attn_output)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, hidden_dim=3072):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 多头自注意力 + 残差 + LayerNorm (Pre-LN 结构)
        x = x + self.self_attn(self.norm1(x))
        # FeedForward + 残差 + LayerNorm
        x = x + self.ffn(self.norm2(x))
        return x
    
def test_transformer_layer():
    # 超参数（与模型定义一致）
    batch_size = 2
    sequence_length = 10
    embed_dim = 768
    num_heads = 12
    hidden_dim = 3072

    x = torch.randn(batch_size, sequence_length, embed_dim)

    layer = TransformerLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim
    )

    # 前向传播
    output = layer(x)

    # 打印形状，验证输入输出维度一致
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 可选：检查是否有 NaN 或无穷大
    assert not torch.isnan(output).any(), "输出包含 NaN"
    assert not torch.isinf(output).any(), "输出包含 Inf"
    print("测试通过！")

if __name__ == "__main__":
    test_transformer_layer()