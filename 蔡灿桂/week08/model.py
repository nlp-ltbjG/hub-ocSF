"""
文本匹配模型定义

教学重点：
  1. BiEncoder（表示型）— 共享 BERT 骨干，对两句分别编码，计算余弦相似度
  2. CrossEncoder（交互型）— 两句拼接后整体送入 BERT，直接输出匹配概率
  3. Projection Head & Attention Pooling — 增强向量表达能力
  4. num_hidden_layers — 限制 BERT 层数加速训练（4 层约为全量的 1/3 时间）

使用方式：
  from model import BiEncoder, CrossEncoder, build_biencoder, build_crossencoder

依赖：
  pip install torch transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertConfig, BertModel
from typing import Optional, Literal, Dict, Any


# ── BiEncoder ─────────────────────────────────────────────────────────────

class BiEncoder(nn.Module):
    """
    表示型文本匹配：Siamese Bi-Encoder 
    """
    def __init__(
        self, 
        bert_path: str, 
        pool: Literal["cls", "mean", "max", "attention"] = "mean", 
        dropout: float = 0.1, 
        num_hidden_layers: Optional[int] = None
    ):
        super().__init__()
        assert pool in ("cls", "mean", "max", "attention"), f"不支持的池化策略: {pool}"

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        # 截断 BERT 层数（避免加载多余权重）
        if num_hidden_layers is not None:
            self.bert.encoder.layer = self.bert.encoder.layer[:num_hidden_layers]

        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert.config.hidden_size

        #  Projection Head，提升向量空间质量
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Attention Pooling 的可学习参数
        if pool == "attention":
            self.attention_weights = nn.Linear(hidden_size, 1)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        单句编码，返回 L2 归一化后的句向量 [B, H]
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        vec = self._pool(out.last_hidden_state, attention_mask)  # [B, H]
        vec = self.projection(self.dropout(vec))                 # 经过连接层
        return F.normalize(vec, p=2, dim=-1)

    def forward(self, batch_a: Dict[str, torch.Tensor], batch_b: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (emb_a, emb_b)，各形状 [B, H]"""
        emb_a = self.encode(**batch_a)
        emb_b = self.encode(**batch_b)
        return emb_a, emb_b

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """支持 4 种池化策略"""
        if self.pool == "cls":
            return last_hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        if self.pool == "mean":
            sum_h = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return sum_h / count

        if self.pool == "max":
            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values

        if self.pool == "attention":
            # 计算注意力权重 [B, L, 1]
            attn_scores = self.attention_weights(last_hidden)
            attn_scores = attn_scores + (1 - mask) * (-1e9)  # Mask 掉 padding
            attn_weights = F.softmax(attn_scores, dim=1)
            return (last_hidden * attn_weights).sum(dim=1)


# ── CrossEncoder ──────────────────────────────────────────────────────────

class CrossEncoder(nn.Module):
    """
    交互型文本匹配：Cross-Encoder 
    """
    def __init__(self, bert_path: str, dropout: float = 0.1, num_hidden_layers: Optional[int] = None):
        super().__init__()

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        # 截断 BERT 层数
        if num_hidden_layers is not None:
            self.bert.encoder.layer = self.bert.encoder.layer[:num_hidden_layers]

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """返回 logits [B, 2]，未经 softmax"""
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_vec = out.last_hidden_state[:, 0, :]  # [B, H]
        cls_vec = self.dropout(cls_vec)
        return self.classifier(cls_vec)            # [B, 2]


# ── 工厂函数 ──────────────────────────────────────────────────────────────

def build_biencoder(bert_path: str, pool: str = "mean", dropout: float = 0.1, num_hidden_layers: Optional[int] = None) -> BiEncoder:
    """构建 BiEncoder 并打印参数量。"""
    model = BiEncoder(bert_path, pool=pool, dropout=dropout, num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"BiEncoder (pool={pool}, layers={num_hidden_layers or 12})")
    return model


def build_crossencoder(bert_path: str, dropout: float = 0.1, num_hidden_layers: Optional[int] = None) -> CrossEncoder:
    """构建 CrossEncoder 并打印参数量。"""
    model = CrossEncoder(bert_path, dropout=dropout, num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"CrossEncoder (layers={num_hidden_layers or 12})")
    return model


def _print_param_info(model: nn.Module, name: str):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    bert_params = sum(p.numel() for p in model.bert.parameters()) / 1e6
    print(f"模型: {name}")
    print(f"参数量: {total:.1f}M  (BERT 骨干: {bert_params:.1f}M)")
