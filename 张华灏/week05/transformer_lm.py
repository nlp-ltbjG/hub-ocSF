"""
基于 Transformer 的单向（Causal）语言模型
训练 + 文本生成完整示例
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────
# 1. 从 txt 文件加载训练语料
# ─────────────────────────────────────────
DEFAULT_CORPUS_PATH = Path(__file__).parent / 'corpus.txt'


def load_corpus(path: Path | str = DEFAULT_CORPUS_PATH) -> str:
    """读取语料 txt，跳过空行和以 # 开头的注释行。"""
    path = Path(path)
    lines = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
    return '\n'.join(lines)


# ─────────────────────────────────────────
# 2. 字符级 Tokenizer
# ─────────────────────────────────────────
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab = ['<PAD>', '<BOS>', '<EOS>'] + chars
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.pad_id = self.stoi['<PAD>']
        self.bos_id = self.stoi['<BOS>']
        self.eos_id = self.stoi['<EOS>']

    def encode(self, text: str):
        return [self.stoi.get(c, self.pad_id) for c in text]

    def decode(self, ids):
        return ''.join(self.itos.get(i, '') for i in ids
                       if i not in (self.pad_id, self.bos_id, self.eos_id))

    @property
    def vocab_size(self):
        return len(self.vocab)


# ─────────────────────────────────────────
# 3. 数据集：滑动窗口切块
# ─────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int = 64):
        self.seq_len = seq_len
        ids = [tokenizer.bos_id] + tokenizer.encode(text) + [tokenizer.eos_id]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]   # (input, target)


# ─────────────────────────────────────────
# 4. Transformer 单向语言模型
# ─────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # 3, B, H, T, D
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale   # B, H, T, T

        # 因果掩码（下三角）
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)   # B, T, vocab_size


# ─────────────────────────────────────────
# 5. 文本生成（带温度 & top-k 采样）
# ─────────────────────────────────────────
@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 60,
             temperature: float = 0.8, top_k: int = 30, device='cpu'):
    model.eval()
    ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    generated = list(prompt)
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]          # B, vocab
        logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.eos_id:
            break

        generated.append(tokenizer.itos[next_id.item()])
        x = torch.cat([x, next_id], dim=1)

    return ''.join(generated)


# ─────────────────────────────────────────
# 6. 训练主程序
# ─────────────────────────────────────────
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 数据准备
    corpus = load_corpus()
    print(f"语料文件: {DEFAULT_CORPUS_PATH}")
    print(f"语料字符数: {len(corpus)}\n")

    tokenizer = CharTokenizer(corpus)
    dataset = TextDataset(corpus, tokenizer, seq_len=64)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    print(f"词表大小: {tokenizer.vocab_size}")
    print(f"训练样本数: {len(dataset)}\n")

    # 模型
    model = CausalTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128, n_heads=4, n_layers=4, max_len=128
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}\n")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # ── 训练循环 ──
    EPOCHS = 200
    print("=" * 55)
    print(f"开始训练，共 {EPOCHS} 个 epoch")
    print("=" * 55)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                          # B, T, V
            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                y.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        ppl = math.exp(avg_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    print("=" * 55)
    print("训练完成！\n")

    # ── 文本生成演示 ──
    prompts = [
        "明月几时",   # 古诗词
        "春天来了",   # 现代散文
        "老张是个",   # 人物故事
        "小明放学",   # 日常对话
        "三月，柳树", # 自然描写
        "努力的人",   # 哲思短语
    ]

    print("=" * 55)
    print("文本生成结果（temperature=0.8, top_k=30）")
    print("=" * 55)
    for prompt in prompts:
        result = generate(model, tokenizer, prompt,
                          max_new_tokens=40, temperature=0.8, top_k=30, device=device)
        print(f"提示词: 「{prompt}」")
        print(f"生成文本: {result}")
        print("-" * 40)

    print("\n高温度采样（temperature=1.2）—— 更多随机性：")
    for prompt in prompts[:2]:
        result = generate(model, tokenizer, prompt,
                          max_new_tokens=30, temperature=1.2, top_k=50, device=device)
        print(f"  [{prompt}] → {result}")

    print("\n低温度采样（temperature=0.4）—— 更确定性：")
    for prompt in prompts[:2]:
        result = generate(model, tokenizer, prompt,
                          max_new_tokens=30, temperature=0.4, top_k=10, device=device)
        print(f"  [{prompt}] → {result}")

    # 保存模型
    save_path = Path(__file__).parent / 'causal_lm.pt'
    torch.save({
        'model_state': model.state_dict(),
        'vocab': tokenizer.vocab,
        'config': dict(vocab_size=tokenizer.vocab_size, d_model=128,
                       n_heads=4, n_layers=4, max_len=128)
    }, save_path)
    print(f"\n模型已保存至: {save_path}")


if __name__ == '__main__':
    train()
