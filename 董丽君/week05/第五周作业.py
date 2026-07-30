import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from collections import Counter

# 超参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据预处理
class TextTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def create_dataset(data, seq_len, batch_size):
    n = len(data)
    # 确保数据长度能被 batch_size 整除，并且留出一个位置给 y
    max_len = ((n - 1) // batch_size) * batch_size + 1
    data = data[:max_len]
    x = data[:-1].view(batch_size, -1).t().contiguous()
    y = data[1:].view(batch_size, -1).t().contiguous()
    return x, y

# Transformer模型
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        embed = self.embedding(x)
        embed = self.positional_encoding(embed)
        output = self.transformer(embed)
        logits = self.fc(output)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 因为使用了 batch_first=True，所以 x 的形状是 [batch, seq_len, embed_dim]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].transpose(0, 1)
        return self.dropout(x)

# Temperature + Top-P采样
def temperature_top_p_sampling(logits, temperature=1.0, top_p=0.9):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 移除累积概率超过top_p的token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 正确处理索引
    batch_size = logits.size(0)
    for i in range(batch_size):
        logits[i, sorted_indices[i, sorted_indices_to_remove[i]]] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

# 文本生成函数
def generate_text(model, tokenizer, start_text, max_len=500, temperature=1.0, top_p=0.9):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
        generated = input_ids
        
        for _ in range(max_len):
            outputs = model(input_ids)
            logits = outputs[:, -1, :]
            next_token = temperature_top_p_sampling(logits, temperature, top_p)
            generated = torch.cat([generated, next_token], dim=1)
            input_ids = generated
            
            if generated.size(1) >= max_len:
                break
        
        return tokenizer.decode(generated[0].tolist())

# 训练函数
def train(model, x, y, epochs=50, lr=1e-4, seq_len=128):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i in range(0, x.size(0) - seq_len, seq_len):
            batch_x = x[i:i+seq_len].t().to(device)
            batch_y = y[i:i+seq_len].t().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), batch_y.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 每5轮生成一次文本
        if (epoch + 1) % 5 == 0:
            generated = generate_text(model, tokenizer, "黄金", max_len=200)
            print(f"\nGenerated text:\n{generated}\n")

# 主程序
if __name__ == "__main__":
    # 加载数据
    corpus_path = r"../week4 语言模型/循环神经网络语言模型/corpus.txt"
    text = load_corpus(corpus_path)
    print(f"Corpus length: {len(text)} characters")
    
    # 创建tokenizer
    tokenizer = TextTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # 准备训练数据
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    seq_len = 128
    batch_size = 8
    x, y = create_dataset(data, seq_len, batch_size)
    print(f"Training data shape: x={x.shape}, y={y.shape}")
    
    # 创建模型
    embed_dim = 256
    num_heads = 4
    num_layers = 3
    hidden_dim = 512
    
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    train(model, x, y, epochs=60, lr=1e-4)
    
    # 保存模型
    torch.save(model.state_dict(), 'transformer_lm.pth')
    print("Model saved as 'transformer_lm.pth'")
    
    # 生成文本示例
    print("\n--- 文本生成示例 ---")
    prompts = ["黄金", "避险资金", "美国货币政策", "大宗商品"]
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_len=300, temperature=0.8, top_p=0.9)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}\n")
