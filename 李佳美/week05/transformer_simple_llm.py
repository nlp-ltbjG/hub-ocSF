'''
任务：训练一个基于 Transformer 的单向语言模型，并完成文本生成。

数据集：训练文本路径为 DATA_PATH，可以新建一个 train.txt，把训练语料放进去。
1. RUN_MODE = "train" 表示训练模型
2. RUN_MODE = "generate" 表示加载模型并生成文本
3. DATA_PATH 是训练语料文本路径
4. CKPT_PATH 是模型保存或加载路径
5. PROMPT 是生成文本时的开头
先执行
RUN_MODE = "train"
训练完成后执行
RUN_MODE = "generate"
'''
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 0. 运行配置区：直接在这里修改参数，不需要命令行
# =========================================================

# 运行模式：
# "train"：训练模型
# "generate"：加载训练好的模型进行文本生成
RUN_MODE = "generate"

# 训练文本路径。
# 你可以新建一个 train.txt，把训练语料放进去。
# 如果该文件不存在，程序会自动使用内置示例文本进行演示。
DATA_PATH = "train.txt"

# 模型保存路径。
# 训练时会保存到这个文件；生成时会从这个文件加载模型。
CKPT_PATH = "transformer_lm_ckpt.pt"

# 是否强制使用 CPU。
# False：如果有 GPU，则自动使用 GPU；否则使用 CPU。
# True：强制使用 CPU。
USE_CPU = False

# 随机种子。
SEED = 42


# =========================================================
# 1. 训练相关参数
# =========================================================

# 训练轮数。
EPOCHS = 2  # 用2轮来测试一下
# batch size大小
BATCH_SIZE = 32
# 学习率
LEARNING_RATE = 3e-4
# AdamW 权重衰减。
WEIGHT_DECAY = 0.01
# 梯度裁剪阈值，防止梯度爆炸。
GRAD_CLIP = 1.0
# 训练集比例。
# 0.9 表示 90% 文本用于训练，10% 用于验证。
TRAIN_RATIO = 0.9
# 每隔多少个 step 打印一次训练日志。
LOG_INTERVAL = 50


# =========================================================
# 2. 模型结构参数
# =========================================================

# 最大上下文长度。
# 模型每次最多看多少个 token。
# 字符级模型中，128 表示最多看前面 128 个字符。
BLOCK_SIZE = 128

# Transformer 隐藏层维度
N_EMBD = 128
# 多头注意力的 head 数量
# 值得注意的是：N_EMBD 必须能被 N_HEAD 整除
N_HEAD = 4
# Transformer 层数
N_LAYER = 4
# Dropout 概率
DROPOUT = 0.1


# =========================================================
# 3. 文本生成参数
# =========================================================
# 文本生成的开头，类似于在大模型中向模型进行提问时的 prompt
PROMPT = "人工智能"

# 最多新生成多少个字符。
MAX_NEW_TOKENS = 100

# 采样温度。
# 越大越随机，越小越保守，常用范围：0.7 ~ 1.2。
TEMPERATURE = 1.0

# top-k 采样。
# 只从概率最高的 TOP_K 个 token 中采样，如果不想使用 top-k，可以设为 None。
TOP_K = 20


# =========================================================
# 4. 设置固定随机种子
# =========================================================
def set_seed(seed: int = 42):
    """固定 Python 和 PyTorch 的随机种子，使实验尽量可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 5. 字符级 Tokenizer
# =========================================================
class CharTokenizer:
    """
    简单的字符级 tokenizer。
    """

    def __init__(self, text: str = None, stoi: Dict[str, int] = None, itos: List[str] = None):
        """
        初始化 tokenizer。

        训练阶段：传入 text，根据训练文本自动构建词表。
        生成阶段：从 checkpoint 中加载 stoi 和 itos。
        """
        if text is not None:
            # 取出文本中出现过的所有字符，并排序，保证每次构建的词表顺序一致。
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = chars
        else:
            self.stoi = stoi
            self.itos = itos

        self.vocab_size = len(self.itos)

    def encode(self, text: str) -> List[int]:
        """把字符串编码为 token id 列表。"""
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                # 如果 prompt 中出现训练文本中没有的字符，这里直接跳过。
                # 实际项目中也可以引入 <unk> token 处理未知字符。
                print(f"[Warning] 字符 {repr(ch)} 不在词表中，已跳过。")
        return ids

    def decode(self, ids: List[int]) -> str:
        """把 token id 列表解码为字符串。"""
        return "".join([self.itos[i] for i in ids])


# =========================================================
# 6. 自回归语言模型数据集
# =========================================================
class TextDataset(Dataset):
    """
    自回归语言模型的数据集。

    假设原始 token 序列为：
    [t0, t1, t2, t3, t4, ...]

    当 block_size = 4 时，一个训练样本为：
    输入 x = [t0, t1, t2, t3]
    标签 y = [t1, t2, t3, t4]

    也就是说，模型在每个位置都要预测下一个 token。
    """

    def __init__(self, token_ids: List[int], block_size: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # 每个样本需要 block_size 个输入 token 和额外 1 个目标 token。
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        # 输入序列：idx 到 idx + block_size - 1
        x = self.data[idx: idx + self.block_size]

        # 目标序列：idx + 1 到 idx + block_size
        y = self.data[idx + 1: idx + self.block_size + 1]

        return x, y


# =========================================================
# 7. Transformer 语言模型配置
# =========================================================
@dataclass
class TransformerLMConfig:
    """保存模型结构参数，方便训练和加载时保持一致。"""

    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1


# =========================================================
# 8. 基于 Transformer 的单向语言模型
# =========================================================
class TransformerLanguageModel(nn.Module):
    """
    Decoder-only 的单向语言模型。

    注意：
    这里直接使用了 PyTorch 自带的 nn.TransformerEncoder。
    但是模型是不是“单向”，关键不在于模块名字，而在于是否使用 causal mask。
    causal mask 会禁止当前位置看到未来位置，从而实现单向自回归建模。
    """

    def __init__(self, config: TransformerLMConfig):
        super().__init__()
        self.config = config

        # token embedding：把 token id 映射成向量
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # position embedding：表示每个 token 在序列中的位置
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Transformer 编码层
        # batch_first=True 表示输入形状为 [batch_size, seq_len, hidden_dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # 多层 Transformer 堆叠
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_layer,
        )

        # 最后一层归一化，有助于训练稳定
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 语言模型输出头
        # 输出维度为 vocab_size，用于预测下一个 token 的概率分布。
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型参数。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _generate_causal_mask(self, seq_len: int, device: torch.device):
        """
        生成 causal mask。

        mask 形状：[seq_len, seq_len]

        对于第 i 个位置：
        - 可以看到第 0 到 i 个位置；
        - 不能看到第 i+1 之后的未来位置。

        未来位置会被设置为 -inf，经过 softmax 后注意力概率接近 0。
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        前向传播

        参数：
        idx: 输入 token ids，形状 [B, T]
        targets: 标签 token ids，形状 [B, T]

        返回：
        logits: 预测结果，形状 [B, T, vocab_size]
        loss: 如果传入 targets，则返回交叉熵损失；否则返回 None。
        """
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(f"输入长度 {T} 超过 block_size={self.config.block_size}")

        # token embedding：[B, T] -> [B, T, n_embd]
        tok_emb = self.token_embedding(idx)

        # position embedding：[T] -> [T, n_embd]
        positions = torch.arange(0, T, device=idx.device)
        pos_emb = self.position_embedding(positions)

        # token embedding 加上 position embedding。
        x = tok_emb + pos_emb

        # 构造单向语言模型所需的 causal mask。
        causal_mask = self._generate_causal_mask(T, idx.device)

        # Transformer 建模。
        x = self.transformer(x, mask=causal_mask)

        # 输出层。
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 交叉熵损失要求：
            # logits: [B*T, vocab_size]
            # targets: [B*T]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        """
        自回归文本生成

        生成逻辑：
        1. 输入 prompt；
        2. 模型预测下一个 token；
        3. 把预测出的 token 拼接到原序列后面；
        4. 再用新的序列继续预测；
        5. 重复直到达到 max_new_tokens。
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 如果当前序列超过最大上下文长度，只取最后 block_size 个 token。
            idx_cond = idx[:, -self.config.block_size:]

            # 得到所有位置的预测 logits。
            logits, _ = self(idx_cond)

            # 只取最后一个位置，因为文本生成只需要预测下一个 token。
            logits = logits[:, -1, :]

            # temperature 控制随机性。
            logits = logits / max(temperature, 1e-8)

            # top-k 采样：只保留概率最高的 k 个 token。
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # 转为概率分布。
            probs = F.softmax(logits, dim=-1)

            # 按概率采样下一个 token。
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接到序列后面。
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =========================================================
# 9. 读取训练文本
# =========================================================
def load_text(data_path: str) -> str:
    """
    读取训练文本。

    如果 DATA_PATH 指向的文件不存在，则使用内置示例文本。
    这样即使你没有准备数据，也可以先直接运行代码检查流程。
    """
    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[Info] 已读取训练文本：{data_path}，字符数：{len(text)}")
        return text

    print("[Warning] 未找到训练文件，使用内置示例文本进行演示训练。")
    demo_text = (
        "人工智能正在快速发展。\n"
        "语言模型可以根据前文预测下一个词或字符。\n"
        "Transformer 通过自注意力机制建模上下文关系。\n"
        "单向语言模型在生成文本时只能看到当前位置之前的信息。\n"
    )

    # 重复多次，避免示例文本过短导致训练样本太少
    return demo_text * 200


# =========================================================
# 10. 划分训练集和验证集
# =========================================================
def split_train_val(token_ids: List[int], train_ratio: float = 0.9):
    """按照比例划分训练集和验证集。"""
    n = int(len(token_ids) * train_ratio)
    train_ids = token_ids[:n]
    val_ids = token_ids[n:]
    return train_ids, val_ids


# =========================================================
# 11. 验证函数
# =========================================================
@torch.no_grad()
def evaluate(model, dataloader, device):
    """计算验证集上的平均 loss。"""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


# =========================================================
# 12. 保存和加载模型
# =========================================================
def save_checkpoint(path, model, tokenizer, config):
    """保存模型参数、词表和模型配置。"""
    ckpt = {
        "model_state_dict": model.state_dict(),
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "config": config.__dict__,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, device):
    """加载模型参数、词表和模型配置。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到模型文件：{path}。请先把 RUN_MODE 设为 'train' 训练模型。")

    ckpt = torch.load(path, map_location=device)

    tokenizer = CharTokenizer(
        stoi=ckpt["stoi"],
        itos=ckpt["itos"],
    )

    config = TransformerLMConfig(**ckpt["config"])
    model = TransformerLanguageModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, tokenizer, config


# =========================================================
# 13. 训练函数
# =========================================================
def train():
    """完整训练流程。"""
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU else "cpu")
    print(f"[Info] 使用设备：{device}")

    # 读取训练文本
    text = load_text(DATA_PATH)

    # 构建 tokenizer
    tokenizer = CharTokenizer(text=text)
    token_ids = tokenizer.encode(text)

    print(f"[Info] 词表大小 vocab_size：{tokenizer.vocab_size}")
    print(f"[Info] token 数量：{len(token_ids)}")

    # 划分训练集和验证集
    train_ids, val_ids = split_train_val(token_ids, train_ratio=TRAIN_RATIO)

    # 构造数据集
    train_dataset = TextDataset(train_ids, block_size=BLOCK_SIZE)
    val_dataset = TextDataset(val_ids, block_size=BLOCK_SIZE)

    if len(train_dataset) == 0:
        raise ValueError("训练文本太短，请增加文本长度或减小 BLOCK_SIZE。")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    # 创建模型配置
    config = TransformerLMConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    )

    # 初始化模型
    model = TransformerLanguageModel(config).to(device)

    # AdamW 是训练 Transformer 常用的优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    print("[Info] 开始训练...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_batches = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)

            # 前向传播
            _, loss = model(x, y)

            # 清空梯度
            optimizer.zero_grad(set_to_none=True)

            # 反向传播
            loss.backward()

            # 梯度裁剪，提升训练稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            # 更新参数
            optimizer.step()

            total_train_loss += loss.item()
            total_batches += 1

            if step % LOG_INTERVAL == 0:
                print(
                    f"Epoch [{epoch}/{EPOCHS}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Train Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / max(total_batches, 1)
        val_loss = evaluate(model, val_loader, device) if len(val_dataset) > 0 else float("nan")

        # PPL，即困惑度。loss 越低，PPL 越低，通常表示语言模型越好
        train_ppl = math.exp(min(avg_train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20)) if not math.isnan(val_loss) else float("nan")

        print(
            f"Epoch [{epoch}/{EPOCHS}] 完成 | "
            f"Train Loss: {avg_train_loss:.4f}, Train PPL: {train_ppl:.2f} | "
            f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}"
        )

        # 保存验证集 loss 最低的模型
        if len(val_dataset) > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(CKPT_PATH, model, tokenizer, config)
                print(f"[Info] 当前模型更优，已保存到：{CKPT_PATH}")
        else:
            save_checkpoint(CKPT_PATH, model, tokenizer, config)
            print(f"[Info] 模型已保存到：{CKPT_PATH}")

    print("[Info] 训练结束。")


# =========================================================
# 14. 文本生成函数
# =========================================================
def generate_text():
    """加载训练好的模型，并根据 PROMPT 生成文本。"""
    device = torch.device("cuda" if torch.cuda.is_available() and not USE_CPU else "cpu")
    print(f"[Info] 使用设备：{device}")

    # 加载模型和 tokenizer
    model, tokenizer, config = load_checkpoint(CKPT_PATH, device)

    # 编码 prompt
    prompt_ids = tokenizer.encode(PROMPT)

    # 如果 prompt 中所有字符都不在词表中，则默认使用词表第一个字符作为起点
    if len(prompt_ids) == 0:
        print("[Warning] PROMPT 中没有字符存在于词表，默认使用词表第一个字符作为起点。")
        prompt_ids = [0]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 执行文本生成
    out = model.generate(
        idx,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    # 解码生成结果
    generated_text = tokenizer.decode(out[0].tolist())
    print("\n========== 生成结果 ==========")
    print(generated_text)
    print("==============================\n")


# =========================================================
# 15. 主函数入口
# =========================================================
def main():
    """
    - RUN_MODE = "train"：训练模型；
    - RUN_MODE = "generate"：生成文本。
    """
    if N_EMBD % N_HEAD != 0:
        raise ValueError("N_EMBD 必须能被 N_HEAD 整除。")

    if RUN_MODE == "train":
        train()
    elif RUN_MODE == "generate":
        generate_text()
    else:
        raise ValueError("RUN_MODE 只能是 'train' 或 'generate'。")


if __name__ == "__main__":
    main()
