import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        output = self.transformer_decoder(x, x, tgt_mask=mask)
        
        return self.fc_out(output)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, vocab, seq_len):
        self.text = text
        self.vocab = vocab
        self.seq_len = seq_len
        self.data = [vocab[c] for c in text]
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len])
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1])
        return x, y


def train(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            
            loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}\n")


def generate_text(model, start_text, vocab, idx_to_char, max_len=100, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    
    input_seq = [vocab[c] for c in start_text]
    input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            output = model(input_tensor)
            last_token_logits = output[0, -1, :] / temperature
            probabilities = F.softmax(last_token_logits, dim=-1)
            
            next_token_idx = torch.multinomial(probabilities, num_samples=1).item()
            next_char = idx_to_char[next_token_idx]
            
            input_seq.append(next_token_idx)
            input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
            
            if next_char == '<END>':
                break
    
    return start_text + ''.join([idx_to_char[i] for i in input_seq[len(start_text):-1]])


def main():
    sample_text = """自然语言处理是人工智能领域的一个重要分支，它使计算机能够理解、解释和生成人类语言。
    Transformer模型是近年来自然语言处理领域最具影响力的创新之一，它基于自注意力机制，
    能够有效地处理长距离依赖关系，在机器翻译、文本生成等任务上取得了突破性进展。
    深度学习模型的训练需要大量的数据和计算资源，但随着技术的发展，这些模型的性能也在不断提升。
    文本生成是自然语言处理的一个重要应用方向，它可以用于创作文章、编写代码、生成对话等多种场景。
    通过Transformer模型，我们可以构建强大的语言模型，实现高质量的文本生成。
    语言模型的训练目标是预测下一个token的概率，通过最大化训练数据的似然概率来优化模型参数。
    在生成文本时，模型根据已生成的文本不断预测下一个最可能的token，从而逐步构建完整的文本序列。"""
    
    vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2}
    for char in sample_text:
        if char not in vocab:
            vocab[char] = len(vocab)
    
    idx_to_char = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    seq_len = 32
    batch_size = 8
    
    dataset = TextDataset(sample_text, vocab, seq_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=512,
        dropout=0.1
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("开始训练...")
    train(model, dataloader, criterion, optimizer, device, epochs=5)
    
    print("\n文本生成示例：")
    start_texts = ["自然语言处理", "Transformer模型", "深度学习"]
    for start in start_texts:
        generated = generate_text(model, start, vocab, idx_to_char, max_len=50)
        print(f"输入: {start}")
        print(f"生成: {generated}\n")


if __name__ == "__main__":
    main()
