"""
PyTorch Transformer 层实现
包含：多头自注意力、前馈网络、层归一化、完整 Encoder/Decoder 层
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
# 1. 多头自注意力 (Multi-Head Self-Attention)
# ─────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    将 d_model 维的输入投影到 num_heads 个子空间，
    各自计算注意力后拼接，再投影回 d_model 维。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须整除 num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q / K / V 的线性投影（合并写成三个独立线性层，便于理解）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (batch, seq_len, d_model)  →  (batch, num_heads, seq_len, d_k)
        """
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

        参数
        ----
        Q, K, V : (batch, heads, seq_len, d_k)
        mask    : (batch, 1, 1, seq_len)  或  (batch, 1, seq_len, seq_len)
                  值为 True 的位置会被屏蔽（设为 -inf）

        返回
        ----
        output  : (batch, heads, seq_len, d_k)
        weights : (batch, heads, seq_len, seq_len)  注意力权重
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = torch.matmul(weights, V)
        return output, weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数
        ----
        query, key, value : (batch, seq_len, d_model)
        mask              : 可选，屏蔽掩码

        返回
        ----
        output  : (batch, seq_len, d_model)
        weights : (batch, heads, seq_len, seq_len)
        """
        B = query.size(0)

        # 线性投影 + 分头
        Q = self.split_heads(self.W_q(query))   # (B, h, T, d_k)
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # 注意力计算
        attn_output, weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 拼接各头：(B, h, T, d_k) → (B, T, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # 输出投影
        output = self.W_o(attn_output)
        return output, weights


# ─────────────────────────────────────────
# 2. 位置前馈网络 (Position-wise Feed-Forward)
# ─────────────────────────────────────────
class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    两层线性变换，中间用 ReLU（或 GELU）激活，维度先升后降。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ─────────────────────────────────────────
# 3. 正弦位置编码 (Sinusoidal Positional Encoding)
# ─────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                       # (1, max_len, d_model)
        self.register_buffer("pe", pe)             # 不参与梯度更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────
# 4. Encoder 层
# ─────────────────────────────────────────
class TransformerEncoderLayer(nn.Module):
    """
    单个 Encoder 块：
      子层 1: 多头自注意力  + Add & Norm
      子层 2: 前馈网络      + Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 子层 1：自注意力（Pre-Norm 风格也很流行，这里用 Post-Norm）
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 子层 2：前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ─────────────────────────────────────────
# 5. Decoder 层
# ─────────────────────────────────────────
class TransformerDecoderLayer(nn.Module):
    """
    单个 Decoder 块：
      子层 1: 带因果掩码的自注意力  + Add & Norm
      子层 2: 交叉注意力            + Add & Norm
      子层 3: 前馈网络              + Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn         = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 子层 1：带因果掩码的自注意力
        attn_out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_out))

        # 子层 2：交叉注意力（query 来自 decoder，key/value 来自 encoder）
        attn_out, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(attn_out))

        # 子层 3：前馈网络
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))
        return tgt


# ─────────────────────────────────────────
# 6. 完整 Transformer（Encoder-Decoder）
# ─────────────────────────────────────────
class Transformer(nn.Module):
    """
    完整的 Encoder-Decoder Transformer，适合序列到序列任务（如机器翻译）。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.d_model = d_model

        # 词嵌入 + 位置编码
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)

        # Encoder / Decoder 堆叠
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_decoder_layers)]
        )

        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        # 输出投影到词表
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化线性层权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """生成上三角因果掩码，形状 (1, 1, seq_len, seq_len)"""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def encode(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.decoder_norm(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        参数
        ----
        src : (batch, src_len)  源序列 token id
        tgt : (batch, tgt_len)  目标序列 token id

        返回
        ----
        logits : (batch, tgt_len, tgt_vocab_size)
        """
        # 若未提供因果掩码，自动生成
        if tgt_mask is None:
            tgt_mask = self.make_causal_mask(tgt.size(1), tgt.device)

        memory = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.output_proj(decoder_out)


# ─────────────────────────────────────────
# 7. 快速验证
# ─────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SRC_VOCAB, TGT_VOCAB = 1000, 1000
    BATCH, SRC_LEN, TGT_LEN = 2, 10, 8
    D_MODEL, HEADS = 128, 4

    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        num_heads=HEADS,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
    ).to(device)

    src = torch.randint(0, SRC_VOCAB, (BATCH, SRC_LEN), device=device)
    tgt = torch.randint(0, TGT_VOCAB, (BATCH, TGT_LEN), device=device)

    logits = model(src, tgt)
    print(f"输入  src : {src.shape}")
    print(f"输入  tgt : {tgt.shape}")
    print(f"输出 logits: {logits.shape}")   # 期望 (2, 8, 1000)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数总量: {total_params:,}")