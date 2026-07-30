"""
基于Transformer的单向语言模型训练与文本生成脚本
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── Transformer模型 ───────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, nhead, ff_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # 自注意力机制
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # 前馈网络
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, ff_dim, dropout, max_seq_len):
        super(TransformerLM, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_seq_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, nhead, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码，确保模型只能看到前面的token"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        # 获取序列长度
        seq_len = x.size(1)
        
        # 词嵌入
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 生成因果掩码
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 通过Transformer块
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        
        # 输出层
        output = self.fc_out(self.dropout(x))
        return output


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 文本生成 ───────────────────────────

def generate_text(model, start_text, char2idx, idx2char, max_length=200, temperature=1.0):
    """
    使用训练好的模型生成文本
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 将开始文本转换为索引
    tokens = [char2idx.get(c, 0) for c in start_text if c in char2idx]
    generated = tokens[:]
    
    with torch.no_grad():
        for _ in range(max_length):
            # 取最后的序列作为输入
            input_seq = torch.tensor([generated[-model.pos_encoding.pe.size(1):]], 
                                   dtype=torch.long, device=device)
            
            # 获取模型输出
            logits = model(input_seq)
            # 只取最后一个时间步的输出
            last_logits = logits[0, -1, :] / temperature
            
            # 使用softmax获取概率分布
            probs = torch.softmax(last_logits, dim=-1)
            
            # 采样下一个字符
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            
            # 如果生成结束符则停止（这里假设没有特殊的结束符）
            if next_token == char2idx.get('\n', -1) and len(generated) > len(start_text) + 10:
                break
    
    # 转换回文本
    result = ''.join([idx2char[i] for i in generated])
    return result


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--embed_dim",  type=int,   default=256)
    parser.add_argument("--nhead",      type=int,   default=8)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--ff_dim",     type=int,   default=512)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="transformer_best_model.pt")
    parser.add_argument("--generate",   action="store_true", help="生成模式")
    parser.add_argument("--start_text", default="Hello", help="生成文本的开始")
    parser.add_argument("--gen_len",    type=int, default=200, help="生成文本长度")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.generate:
        # 加载训练好的模型进行文本生成
        checkpoint = torch.load(args.save, map_location=device)
        char2idx = checkpoint['char2idx']
        idx2char = checkpoint['idx2char']
        
        model = TransformerLM(
            vocab_size=len(char2idx),
            embed_dim=args.embed_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            max_seq_len=args.seq_len
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state'])
        print(f"模型已从 {args.save} 加载")
        
        # 生成文本
        generated_text = generate_text(
            model, args.start_text, char2idx, idx2char, 
            max_length=args.gen_len, temperature=0.7
        )
        print("\n生成的文本:")
        print("="*50)
        print(generated_text)
        print("="*50)
        return

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_seq_len=args.seq_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
