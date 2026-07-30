import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import requests
import os

# --------------------- 超参数 ---------------------
BATCH_SIZE = 64
BLOCK_SIZE = 128        # 序列长度（上下文长度）
EMBED_DIM = 256         # 词嵌入维度
NUM_HEADS = 8           # 注意力头数
NUM_LAYERS = 6          # Transformer层数
DROPOUT = 0.1
LEARNING_RATE = 3e-4
EPOCHS = 10             # 训练轮数（小数据集上可快速看到效果）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------- 数据准备 ---------------------
def load_data():
    """下载或加载文本数据。默认使用Tiny Shakespeare数据集。"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "tinyshakespeare.txt"
    if not os.path.exists(filename):
        print("正在下载Tiny Shakespeare数据集...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
        except Exception as e:
            print(f"下载失败({e})，使用内置示例文本。")
            text = "Hello world! " * 5000  # 示例文本
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

class CharDataset(Dataset):
    """字符级数据集，生成 (输入序列, 目标序列) 对。"""
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}   # 字符到索引
        self.itos = {i: ch for ch, i in self.stoi.items()}  # 索引到字符
        self.vocab_size = len(chars)
        data = [self.stoi[c] for c in text]
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y

# --------------------- 模型定义 ---------------------
class PositionalEncoding(nn.Module):
    """可学习的位置编码。"""
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (seq_len, batch_size, embed_dim)
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)  # (seq_len, 1)
        return x + self.pos_embedding(positions)

class CausalTransformerLM(nn.Module):
    """基于Transformer Encoder的单向语言模型（通过因果掩码实现）。"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, block_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # 我们使用 (seq_len, batch, dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        # 构建因果掩码（上三角矩阵，True表示被遮蔽的位置）
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        )

    def forward(self, x):
        # x: (batch, seq_len) -> 转换为 (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)  # (seq_len, batch)
        seq_len, batch_size = x.shape
        tok_emb = self.token_embedding(x)  # (seq_len, batch, embed_dim)
        pos_emb = self.pos_encoding(tok_emb)
        # TransformerEncoder的src_mask：形状 (seq_len, seq_len)，True表示屏蔽
        mask = self.causal_mask[:seq_len, :seq_len]
        out = self.transformer(pos_emb, mask=mask)  # (seq_len, batch, embed_dim)
        out = self.ln_f(out)
        logits = self.head(out)  # (seq_len, batch, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        """自回归生成文本。
        start_tokens: list[int] 或 torch.LongTensor (1D)
        """
        self.eval()
        if isinstance(start_tokens, list):
            start_tokens = torch.tensor(start_tokens, dtype=torch.long, device=DEVICE)
        else:
            start_tokens = start_tokens.to(DEVICE)
        generated = start_tokens.unsqueeze(0)  # (1, seq_len)

        for _ in range(max_new_tokens):
            # 截断到最大长度
            context = generated[:, -BLOCK_SIZE:]
            logits = self.forward(context)  # (seq_len, 1, vocab_size)
            logits = logits[-1, 0, :] / temperature  # 只取最后一个时间步
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1,)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        return generated.squeeze(0).tolist()

# --------------------- 训练 ---------------------
def train():
    text = load_data()
    dataset = CharDataset(text, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"数据集大小: {len(dataset)}  |  词汇表大小: {dataset.vocab_size}")

    model = CausalTransformerLM(
        vocab_size=dataset.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)  # (seq_len, batch, vocab_size)
            # 需要将 logits 展平以计算损失
            loss = criterion(logits.view(-1, dataset.vocab_size), y.transpose(0, 1).reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}  |  Loss: {avg_loss:.4f}")

        # 每个epoch后生成一小段文本，观察效果
        if (epoch + 1) % 2 == 0:
            start_str = "The"
            start_ids = [dataset.stoi.get(c, 0) for c in start_str]
            gen_ids = model.generate(start_ids, max_new_tokens=100, temperature=0.8)
            gen_text = ''.join(dataset.itos[i] for i in gen_ids)
            print(f"--- Epoch {epoch+1} 生成示例 ---\n{gen_text}\n----------------------")

    # 最终生成
    print("\n训练完成！最终生成示例：")
    start_str = "To be, or not to be"
    start_ids = [dataset.stoi.get(c, 0) for c in start_str]
    gen_ids = model.generate(start_ids, max_new_tokens=300, temperature=0.8)
    gen_text = ''.join(dataset.itos[i] for i in gen_ids)
    print(gen_text)

if __name__ == '__main__':
    train()
