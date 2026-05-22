"""
面试版 Transformer Encoder
核心：Multi-Head Self-Attention / FFN / 残差 + LN / 堆叠
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
    
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, nhead):
        super().__init__()
        assert hidden % nhead == 0, "hidden must be divisible by nhead"
        self.hidden = hidden
        self.nhead = nhead
        self.d_k = hidden // nhead
        
        self.qkv = nn.Linear(hidden, hidden * 3)   # 一次性算 Q K V 或者过三个线性层
        
    def forward(self, x):
        B, T, H = x.shape
        # [B, T, 3H] -> 3 个 [B, n_head, T, d_k]
        q, k ,v = self.qkv(x).chunk(3, dim=-1) # dim=-1按最后一个维度进行分割
        q = q.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        atten = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_k)
        out = torch.softmax(atten, dim=-1) @ v # ？为什么dim = -1
        return out.transpose(1, 2).contiguous().view(B, T, H)
        
    
    
class EncoderLayer(nn.Module):
    def __init__(self, hidden, nhead, dim_feedforward, dropout):
        super().__init__()
        self.dim_feedforward = dim_feedforward
        self.atten = MultiHeadAttention(hidden, nhead)
        self.line1 = nn.Linear(hidden, dim_feedforward)
        self.line2 = nn.Linear(dim_feedforward, hidden)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.layer1 = nn.LayerNorm(hidden)
        self.layer2 = nn.LayerNorm(hidden)
        self.gelu = nn.GELU()
    
    def feedforward(self, x):
        x = self.gelu(self.drop1(self.line1(x)))
        return self.drop2(self.line2(x))
        
    def forward(self, x):
        x = self.layer1(self.atten(x) + x)
        x = self.layer2(self.feedforward(x) + x)
        return x
         

class TransformerEncoder(nn.Module):
    def __init__(self, hidden, num_layers, nhead, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(hidden, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    seq_len = 16
    hidden = 512
    nhead = 8
    num_layers = 12
    dim_feedforward = 1024
    dropout=0.0
    
    x = torch.randn(2, seq_len, hidden)        # [B, T, H] 批次、序列长度、隐藏维度
    
    # 调用 pytorch的transformer -------------------------------------------------------
    transformer_layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward,dropout, batch_first=True)
    torch_model = nn.TransformerEncoder(transformer_layer, num_layers)
    torch_y = torch_model(x)
    torch_w = torch_model.state_dict()
    print(torch_y.shape)              # [2, 16, 512]
    print(f"torch_out: {torch_y.detach().numpy()}")
    
    # 自己实现的transformer ------------------------------------------------------------
    
    diy_model = TransformerEncoder(hidden, num_layers, nhead, dim_feedforward, dropout)
    diy_model.load_state_dict(torch_w, strict=False)
    diy_y = diy_model(x)
    print(diy_y.shape)              # [2, 16, 512]
    print(f"diy_out: {diy_y.detach().numpy()}")
