"""
文本生成脚本，基于训练好的语言模型。
用法:
    python generate_text.py --model_path best_model.pt --prompt "你好" --length 100
    python generate_text.py --model_path best_model.pt --interactive
"""

import argparse
import random
import torch
import torch.nn.functional as F


# ─────────────────────────── 模型定义（与训练脚本一致）───────────────────────────

def generate_causal_mask(seq_len):
    """生成因果掩码（下三角矩阵），用于语言模型的训练"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask  # (seq_len, seq_len), True 表示需要屏蔽的位置


class PositionalEncoding(torch.nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerLM(torch.nn.Module):
    """Transformer 语言模型（带因果掩码）"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        causal_mask = generate_causal_mask(seq_len).to(x.device)

        e = self.drop(self.embed(x))  # (batch, seq_len, embed_dim)
        e = self.pos_encoding(e)      # (batch, seq_len, embed_dim)

        out = self.transformer(e, mask=causal_mask)  # (batch, seq_len, embed_dim)
        logits = self.fc(self.drop(out))  # (batch, seq_len, vocab_size)
        return logits


class LM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout, num_heads=None):
        super().__init__()
        self.model_type = model_type

        if model_type == "transformer":
            self.model = TransformerLM(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.embed = torch.nn.Embedding(vocab_size, embed_dim)
            rnn_cls = torch.nn.LSTM if model_type == "lstm" else torch.nn.RNN
            self.rnn = rnn_cls(
                embed_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.drop = torch.nn.Dropout(dropout)
            self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        if self.model_type == "transformer":
            return self.model(x)
        else:
            e = self.drop(self.embed(x))
            out, _ = self.rnn(e)
            logits = self.fc(self.drop(out))   # (B, T, V)
            return logits


# ─────────────────────────── 文本生成函数 ───────────────────────────

def generate_text(model, char2idx, idx2char, prompt, max_length, temperature=1.0, device="cpu"):
    """
    生成文本

    Args:
        model: 训练好的语言模型
        char2idx: 字符到索引的映射
        idx2char: 索引到字符的映射
        prompt: 起始文本
        max_length: 最大生成长度
        temperature: 温度参数，控制随机性（越高越随机）
        device: 设备
    """
    model.eval()
    model.to(device)

    # 将 prompt 转换为索引
    if prompt:
        input_ids = [char2idx.get(c, char2idx.get("[UNK]", 1)) for c in prompt if c in char2idx]
    else:
        # 随机选择一个起始字符
        input_ids = [random.choice(list(char2idx.values()))]

    if not input_ids:
        input_ids = [random.choice(list(char2idx.values()))]

    generated = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_length):
            # 准备输入
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

            # 前向传播
            logits = model(input_tensor)  # (1, seq_len, vocab_size)

            # 获取最后一个位置的 logits
            next_token_logits = logits[0, -1, :] / temperature

            # 应用 softmax 获取概率分布
            probs = F.softmax(next_token_logits, dim=-1)

            # 采样下一个字符
            next_token = torch.multinomial(probs, num_samples=1).item()

            # 添加到生成序列
            generated.append(next_token)
            input_ids.append(next_token)

            # 可选：遇到结束符时停止
            # if next_token == char2idx.get("[EOS]"):
            #     break

    # 将索引转换为字符
    text = "".join([idx2char.get(idx, "") for idx in generated])
    return text


def generate_text_rnn(model, char2idx, idx2char, prompt, max_length, temperature=1.0, device="cpu"):
    """
    RNN/LSTM 逐字符生成（更高效，不需要每次重新计算整个序列）
    """
    model.eval()
    model.to(device)

    # 将 prompt 转换为索引
    if prompt:
        input_ids = [char2idx.get(c, char2idx.get("[UNK]", 1)) for c in prompt if c in char2idx]
    else:
        input_ids = [random.choice(list(char2idx.values()))]

    if not input_ids:
        input_ids = [random.choice(list(char2idx.values()))]

    generated = input_ids.copy()

    with torch.no_grad():
        # 初始化隐藏状态
        hidden = None

        # 先处理 prompt
        if input_ids:
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            e = model.drop(model.embed(input_tensor))
            _, hidden = model.rnn(e)

        # 逐字符生成
        for _ in range(max_length):
            # 获取最后一个字符
            last_token = torch.tensor([[generated[-1]]], dtype=torch.long).to(device)

            # 前向传播
            e = model.drop(model.embed(last_token))
            out, hidden = model.rnn(e, hidden)
            logits = model.fc(model.drop(out))

            # 获取 logits
            next_token_logits = logits[0, 0, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

    text = "".join([idx2char.get(idx, "") for idx in generated])
    return text


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_model.pt", help="模型路径")
    parser.add_argument("--prompt", default="", help="起始文本")
    parser.add_argument("--length", type=int, default=200, help="生成长度")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度参数")
    parser.add_argument("--interactive", action="store_true", help="交互式生成模式")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 加载模型
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]
    model_args = checkpoint["args"]

    print(f"模型类型: {model_args['model'].upper()}")
    print(f"词表大小: {len(char2idx)}")

    # 重建模型
    model = LM(
        vocab_size=len(char2idx),
        embed_dim=model_args["embed_dim"],
        hidden_dim=model_args["hidden_dim"],
        num_layers=model_args["num_layers"],
        model_type=model_args["model"],
        dropout=model_args["dropout"],
        num_heads=model_args.get("num_heads", 4) if model_args["model"] == "transformer" else None,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}\n")

    if args.interactive:
        # 交互式生成
        print("交互式生成模式（输入 'quit' 退出）")
        while True:
            prompt = input("\n请输入起始文本: ").strip()
            if prompt.lower() == "quit":
                break
            if not prompt:
                continue

            try:
                length = int(input("生成长度 (默认200): ") or 200)
                temperature = float(input("温度参数 (默认0.8): ") or 0.8)
            except ValueError:
                length = 200
                temperature = 0.8

            print(f"\n生成中... (长度={length}, 温度={temperature})")

            if model_args["model"] == "transformer":
                generated = generate_text(
                    model, char2idx, idx2char, prompt, length, temperature, device
                )
            else:
                generated = generate_text_rnn(
                    model, char2idx, idx2char, prompt, length, temperature, device
                )

            print(f"\n生成结果:\n{generated}\n")
    else:
        # 单次生成
        prompt = args.prompt
        if not prompt:
            print("未提供起始文本，使用随机起始字符")
            prompt = ""

        print(f"起始文本: '{prompt}'")
        print(f"生成长度: {args.length}")
        print(f"温度参数: {args.temperature}\n")

        if model_args["model"] == "transformer":
            generated = generate_text(
                model, char2idx, idx2char, prompt, args.length, args.temperature, device
            )
        else:
            generated = generate_text_rnn(
                model, char2idx, idx2char, prompt, args.length, args.temperature, device
            )

        print(f"生成结果:\n{generated}")


if __name__ == "__main__":
    main()
