import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        out, attn = self.attention(Q, K, V, mask)
        # 拼接多头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.fc(out)
        # 残差连接 + 层归一化
        return self.norm(output + query), attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.norm(out + x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        out, _ = self.self_attn(x, x, x, mask)
        out = self.ffn(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        out, _ = self.self_attn(x, x, x, tgt_mask)
        out, _ = self.cross_attn(out, enc_output, enc_output, src_mask)
        out = self.ffn(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(
        self, 
        d_model=512, 
        num_heads=8, 
        num_layers=6, 
        d_ff=2048, 
        vocab_size=10000, 
        dropout=0.1
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        # 词嵌入 + 位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # 堆叠 N 层编码器/解码器
        self.encoders = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoders = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask=None):
        # 编码器前向传播
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoders:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # 解码器前向传播
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.decoders:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 完整前向传播
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output

if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    vocab_size = 10000
    
    model = Transformer(d_model=d_model, num_heads=num_heads, vocab_size=vocab_size)
    
    src = torch.randint(0, vocab_size, (2, 10))   # 源句子：2条，每条10个词
    tgt = torch.randint(0, vocab_size, (2, 8))    # 目标句子：2条，每条8个词

    output = model(src, tgt)
    #输出形状：[batch_size, 目标序列长度, 词表大小]
    print(f"源输入形状: {src.shape}")
    print(f"目标输入形状: {tgt.shape}")
    print(f"模型输出形状: {output.shape}")
