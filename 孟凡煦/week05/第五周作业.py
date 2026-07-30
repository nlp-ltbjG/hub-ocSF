import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.utils.data import Dataset, DataLoader

# ----------------------
# 1. 位置编码 (Positional Encoding)
# 作用：给每个token添加位置信息，因为Transformer本身没有顺序概念
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)  # 不参与训练的参数
    
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]  # 添加位置编码
        return self.dropout(x)


# ----------------------
# 2. 多头注意力 (Multi-Head Attention)
# 作用：让模型同时关注不同位置的信息
# ----------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 定义4个线性层：Q、K、V的投影和输出投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """核心的注意力计算"""
        # Q: [batch, heads, seq_len, d_k]
        # K: [batch, heads, seq_len, d_k]
        # V: [batch, heads, seq_len, d_k]
        
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 添加掩码（防止看到未来的token）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 被mask的位置设为负无穷
        
        # softmax归一化得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和得到输出
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # 1. 线性投影并拆分成多头
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 合并多头并输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output


# ----------------------
# 3. 前馈神经网络 (Feed Forward Network)
# 作用：在每个位置上独立进行非线性变换
# ----------------------
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层：升维
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层：降维
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)  # GELU激活函数（比ReLU更好）
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ----------------------
# 4. Transformer解码器层 (Decoder Layer)
# 单向语言模型只需要解码器，不需要编码器
# ----------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 1. 自注意力 + 残差连接
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)  # 残差连接
        x = self.norm1(x)  # 层归一化
        
        # 2. 前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)  # 残差连接
        x = self.norm2(x)  # 层归一化
        
        return x


# ----------------------
# 5. GPT模型 (完整的单向语言模型)
# ----------------------
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入层：将token ID转换为向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # 解码器层堆叠
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层：将d_model映射到词汇表大小
        self.fc = nn.Linear(d_model, vocab_size)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def generate_mask(self, seq_len):
        """生成因果掩码，防止看到未来的token"""
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(self.embedding.weight.device)
        return mask
    
    def forward(self, x):
        """
        x: [batch_size, seq_len] - 输入的token序列
        返回: [batch_size, seq_len, vocab_size] - 每个位置的预测概率
        """
        batch_size, seq_len = x.size()
        
        # 1. 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)  # [batch, seq_len, d_model]
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        
        # 2. 生成因果掩码
        mask = self.generate_mask(seq_len)
        
        # 3. 逐层传递
        for layer in self.layers:
            x = layer(x, mask)
        
        # 4. 输出层
        logits = self.fc(x)  # [batch, seq_len, vocab_size]
        
        return logits


# ----------------------
# 6. 数据集类 (用于训练)
# ----------------------
class TextDataset(Dataset):
    def __init__(self, text, vocab, seq_len=32):
        self.text = text
        self.vocab = vocab
        self.seq_len = seq_len
        
        # 将文本转换为token ID
        self.tokens = [vocab[char] for char in text]
    
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        # 输入序列：从idx到idx+seq_len
        # 目标序列：从idx+1到idx+seq_len+1（预测下一个token）
        input_seq = torch.tensor(self.tokens[idx:idx+self.seq_len], dtype=torch.long)
        target_seq = torch.tensor(self.tokens[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return input_seq, target_seq


# ----------------------
# 7. 训练函数
# ----------------------
def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # 前向传播
            logits = model(input_seq)
            
            # 计算损失（只计算有效位置的损失）
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# ----------------------
# 8. 文本生成函数
# ----------------------
def generate_text(model, start_text, vocab, idx_to_char, max_len=100, temperature=1.0):
    """
    从start_text开始生成文本
    temperature: 控制随机性，越小越确定，越大越随机
    """
    model.eval()
    
    # 将起始文本转换为token
    tokens = [vocab[char] for char in start_text]
    input_seq = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(next(model.parameters()).device)
    
    with torch.no_grad():
        for _ in range(max_len):
            # 前向传播
            logits = model(input_seq)
            
            # 取最后一个位置的预测
            next_token_logits = logits[:, -1, :] / temperature
            
            # softmax得到概率
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 采样下一个token（可以用argmax或multinomial）
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到序列
            tokens.append(next_token.item())
            input_seq = torch.cat([input_seq, next_token], dim=1)
            
            # 如果遇到结束符，停止生成
            if idx_to_char[next_token.item()] == '<END>':
                break
    
    # 将token转换回文本
    generated_text = ''.join([idx_to_char[idx] for idx in tokens])
    return generated_text


# ----------------------
# 9. 主函数（运行示例）
# ----------------------
if __name__ == "__main__":
    # 超参数设置
    d_model = 128
    num_heads = 4
    num_layers = 3
    d_ff = 512
    dropout = 0.1
    seq_len = 32
    batch_size = 8
    epochs = 50
    lr = 1e-3
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备示例文本数据
    text = """深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的特征。
Transformer是一种基于自注意力机制的深度学习模型，由Google在2017年提出。
GPT是OpenAI开发的基于Transformer的语言模型，可以生成高质量的文本。
自然语言处理是人工智能的一个重要领域，涉及计算机与人类语言的交互。
"""
    
    # 构建词汇表
    chars = sorted(set(text))
    chars = ['<PAD>', '<END>'] + chars  # 添加特殊符号
    vocab_size = len(chars)
    
    vocab = {char: idx for idx, char in enumerate(chars)}  # 字符到ID的映射
    idx_to_char = {idx: char for idx, char in enumerate(chars)}  # ID到字符的映射
    
    print(f"词汇表大小: {vocab_size}")
    print(f"词汇表内容: {chars}")
    
    # 创建数据集和数据加载器
    dataset = TextDataset(text, vocab, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = GPT(vocab_size, d_model, num_heads, num_layers, d_ff, dropout).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    
    # 训练模型
    print("\n开始训练...")
    train(model, dataloader, optimizer, criterion, device, epochs)
    
    # 文本生成
    print("\n开始生成文本...")
    start_text = "深度学习"
    generated = generate_text(model, start_text, vocab, idx_to_char, max_len=200, temperature=0.8)
    print(f"起始文本: {start_text}")
    print(f"生成文本: {generated}")
