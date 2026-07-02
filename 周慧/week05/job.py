import math
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# ======================== 参数配置（直接在这里修改） ========================
# 训练参数
EPOCHS = 20
SEQ_LEN = 64
BATCH_SIZE = 32              # Mac M2 上改小一点会快很多
VAL_RATIO = 0.05             # 验证集比例
LEARNING_RATE = 1e-3
DROPOUT = 0.1

# 模型参数
EMBED_DIM = 256              # 嵌入维度 / 神经元数量
NUM_HEADS = 8                # 必须能整除 EMBED_DIM，256/8=32
HIDDEN_DIM = 1024            # FFN 中间维度，通常为 EMBED_DIM * 4
NUM_LAYERS = 4               # 层数
MAX_SEQ_LEN = 512            # 最大位置编码长度（>= SEQ_LEN 即可）

# 数据文件
CORPUS_PATTERN = "corpus.txt"   # 可以是 "*.txt" 或具体文件名
SAVE_PATH = "best_model.pt"

# ======================== 数据加载 ========================
def load_corpus(pattern):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding='utf-8', errors='ignore') as f:
            texts.append(f.read())
    if not texts:
        raise FileNotFoundError(f"未找到匹配 {pattern} 的文本文件")
    return "".join(texts)

def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text]   # 所有字符都在词表中
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y

# ======================== 模型定义 ========================
class DecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, L = x.shape
        tok_emb = self.embedding(x)
        pos = torch.arange(L, device=x.device)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)

        # 生成因果掩码（下三角），确保不会看到未来信息
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        out = self.encoder(x, mask=mask)
        logits = self.fc(out)
        return logits

# ======================== 训练 / 验证 ========================
def run_epoch(model, loader, criterion, optimizer, device, train=True, use_amp=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # 混合精度前向（仅在训练时可选）
        if train and use_amp and device.type in ('cuda', 'mps'):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad() if not train else torch.enable_grad():
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

# ======================== 主程序 ========================
def main():
    # 1. 设备选择（优先 CUDA，其次 MPS，最后 CPU）
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 2. 加载语料
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(script_dir, CORPUS_PATTERN)
    text = load_corpus(corpus_path)
    print(f"语料字符数: {len(text):,}")

    # 3. 构建词表
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    # 4. 划分训练/验证集（按行随机打乱）
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - VAL_RATIO))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    # 5. 创建 Dataset 和 DataLoader
    train_ds = CharDataset(train_text, char2idx, SEQ_LEN)
    val_ds   = CharDataset(val_text,   char2idx, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # 6. 初始化模型
    model = DecoderOnlyLM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 7. 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 混合精度（仅当设备支持 autocast 时启用）
    use_amp = device.type in ('cuda', 'mps')
    if use_amp:
        print("启用混合精度训练 (float16)")

    best_val_ppl = float('inf')
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True, use_amp=use_amp)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device,
                                        train=False, use_amp=False)   # 验证时不混合精度也可

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                'model_state': model.state_dict(),
                'char2idx': char2idx,
                'idx2char': idx2char,
                'vocab_size': vocab_size,
                'embed_dim': EMBED_DIM,
                'num_heads': NUM_HEADS,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'max_seq_len': MAX_SEQ_LEN,
                'dropout': DROPOUT,
            }, SAVE_PATH)
            print(f"\n保存最佳模型到 {SAVE_PATH}")

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证困惑度: {best_val_ppl:.2f}")

if __name__ == "__main__":
    main()