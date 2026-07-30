import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# 关键：必须在所有import torch的前面设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 超参数
batch_size = 2
seq_len = 10
d_model = 128
n_head = 2
n_layers = 1
lr = 1e-3
epochs = 50
vocab_size = 100

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=256, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len_current = x.size(1)
        mask = torch.triu(torch.ones(seq_len_current, seq_len_current), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        pos = torch.arange(0, seq_len_current).unsqueeze(0).repeat(x.size(0), 1)
        x = self.emb(x) + self.pos_emb(pos)
        x = self.transformer(x, mask=mask)
        return self.fc(x)

model = TransformerLM(vocab_size, d_model, n_head, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

data = torch.randint(0, vocab_size, (100, seq_len))

print("开始训练 Transformer 单向语言模型...")
model.train()
for epoch in range(epochs):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f}")

print("模型训练完成！")

def generate_text(model, start_tokens, max_len=5):
    model.eval()
    generated = start_tokens.clone()
    with torch.no_grad():
        for _ in range(max_len):
            context = generated[:, -seq_len:] if generated.size(1) > seq_len else generated
            out = model(context)
            next_token = out.argmax(-1)[:, -1:]
            generated = torch.cat([generated, next_token], dim=1)
    return generated

start = torch.randint(0, vocab_size, (1, 3))
result = generate_text(model, start)

print("\n文本生成结果")
print("起始 token:", start.numpy())
print("生成结果:", result.numpy())
