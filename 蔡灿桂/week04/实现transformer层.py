"""
基于 Transformer 字符级语言模型的拼音输入法。
用法:
    python pinyin_ime_transformer.py
    python pinyin_ime_transformer.py --model_path best_model.pt --topk 8 --beam 10
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────── 拼音 → 候选汉字映射表 ───────────────────────

def _load_pinyin_map(path):
    # 从 JSON 文件读取 {音节: [候选字, ...]} 映射表
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到拼音映射表文件: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# 模块加载时为空，main() 中调用 _load_pinyin_map() 后填充
PINYIN_MAP = {}

# ─────────────────────── Transformer 语言模型 ───────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, 
                 dropout, num_heads=4, ffn_mult=4, max_len=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 可学习位置编码（替代 RNN 的隐状态传递）
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        # Transformer Encoder 层（用于自回归语言建模）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T) 字符 ID 序列
        返回: (B, T, vocab_size) 下一个字符的 logits
        """
        B, T = x.shape
        
        # 1. 词嵌入 + 位置编码
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.drop(self.embed(x) + self.pos_embed(pos_ids))
        
        # 2. 因果掩码 (Causal Mask)：防止 attending 到未来 token
        # PyTorch 中 attn_mask 为 True 表示屏蔽（不可见）
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
        # 3. Transformer 前向传播
        out = self.transformer(h, mask=causal_mask)
        out = self.norm(out)
        
        # 4. 映射到词表
        return self.fc(out)


# ─────────────────────── 拼音分词 ───────────────────────

_SYLLABLES = []

def segment(pinyin_str):
    """将拼音字符串切分为音节列表（贪心最长匹配）"""
    syllables = []
    for token in pinyin_str.strip().lower().split():
        i = 0
        while i < len(token):
            matched = next((s for s in _SYLLABLES if token[i:].startswith(s)), None)
            if matched:
                syllables.append(matched)
                i += len(matched)
            else:
                i += 1
    return syllables


# ─────────────────────── 束搜索 ───────────────────────

def beam_search(syllables, prefix, model, char2idx, idx2char, beam_size, device):
    """
    对音节列表逐字做束搜索。
    Transformer 无隐状态缓存，此处采用全序列重计算。
    """
    beams = [(0.0, "")]

    for syllable in syllables:
        # 过滤掉不在训练词表中的候选字（模型无法为其打分）
        candidates = [c for c in PINYIN_MAP.get(syllable, []) if c in char2idx]
        if not candidates:
            continue  # 该音节无可用候选，跳过（不终止整个搜索）

        new_beams = []
        for score, partial in beams:
            # 拼接历史上文与当前已生成部分，送入模型
            context = prefix + partial
            if context:
                ids = [char2idx[c] for c in context if c in char2idx]
                x = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(x)   # (1, T, vocab_size)
                # 只取最后一个时间步的输出，作为"下一字"的概率分布
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            else:
                # 上文为空时无法打分，各候选字得分相同
                log_probs = None

            for char in candidates:
                lp = log_probs[char2idx[char]].item() if log_probs is not None else 0.0
                new_beams.append((score + lp, partial + char))

        # 按累计得分降序，保留 beam_size 条最优路径
        new_beams.sort(reverse=True)
        beams = new_beams[:beam_size]

    return beams


# ─────────────────────── 交互主循环 ───────────────────────

def run(model, char2idx, idx2char, topk, beam_size, device):
    """
    交互式输入法主循环。
    用户每轮输入一段拼音，程序展示 topk 个候选转换结果；
    用户选择编号后，结果追加到已确认文字，作为下一轮的上文。
    """
    print("=" * 52)
    print("  拼音输入法（Transformer 字符级语言模型）")
    print("  输入拼音回车 → 选候选编号追加到已输入文字")
    print("  r = 重置  q = 退出")
    print("=" * 52)

    confirmed = ""  # 已确认的文字，累积作为语言模型上文

    while True:
        print(f"\n已输入: 「{confirmed}」" if confirmed else "\n已输入: （空）")
        raw = input("拼音> ").strip()

        if not raw: continue
        if raw == "q":
            print("退出。"); break
        if raw == "r":
            confirmed = ""; continue

        syllables = segment(raw)
        if not syllables:
            print("无法识别任何音节，请检查拼音拼写。"); continue

        print(f"音节: {' '.join(syllables)}")
        results = beam_search(syllables, confirmed, model, char2idx, idx2char, beam_size, device)

        if not results:
            print("无候选结果。"); continue

        print("候选:")
        for i, (score, text) in enumerate(results[:topk]):
            print(f"  [{i}] {text}  ({score:.2f})")

        choice = input("选择编号 (回车跳过): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(results):
                confirmed += results[idx][1]
            else:
                print("编号超出范围。")


# ─────────────────────── 入口 ───────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_model.pt")
    parser.add_argument("--pinyin_map", default="pinyin_map.json")
    parser.add_argument("--topk",       type=int, default=5)
    parser.add_argument("--beam",       type=int, default=10)
    args = parser.parse_args()

    # 加载拼音映射表，并重建音节排序列表
    global PINYIN_MAP, _SYLLABLES
    PINYIN_MAP = _load_pinyin_map(args.pinyin_map)
    _SYLLABLES = sorted(PINYIN_MAP.keys(), key=len, reverse=True)
    print(f"拼音表: {args.pinyin_map}  ({len(PINYIN_MAP)} 个音节)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt     = torch.load(args.model_path, map_location=device, weights_only=True)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    cfg      = ckpt["args"]

    # 兼容旧版 checkpoint 与新版 Transformer 配置
    model = LM(
        vocab_size = len(char2idx),
        embed_dim  = cfg.get("embed_dim", 256),
        hidden_dim = cfg.get("hidden_dim", 512),
        num_layers = cfg.get("num_layers", 4),
        model_type = cfg.get("model", "transformer"),
        dropout    = 0.0,
        num_heads  = cfg.get("num_heads", 4),
        ffn_mult   = cfg.get("ffn_mult", 4),
        max_len    = cfg.get("max_len", 2048)
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"模型: {args.model_path}  (Transformer, 词表 {len(char2idx)} 字)")
    run(model, char2idx, idx2char, args.topk, args.beam, device)

if __name__ == "__main__":
    main()
