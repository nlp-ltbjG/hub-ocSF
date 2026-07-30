# coding:utf8
"""
文本多分类实验：使用自实现 Transformer Encoder 完成中文短文本主题分类

任务说明：
    输入一条中文短句，判断它属于哪个主题类别。

类别：
    0: 美食
    1: 体育
    2: 科技
    3: 财经

Transformer Encoder 结构（每层）：
    Multi-Head Self-Attention → Add & LayerNorm → Feed-Forward → Add & LayerNorm

模型整体结构：
    字符级 Embedding + 位置编码 → N × TransformerEncoderLayer → Mean Pooling → Linear
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ── 超参数 ────────────────────────────────────────────────────────────────────
SEED = 42
MAX_LEN = 32
D_MODEL = 64        # 词向量维度 = Transformer 隐层维度
NUM_HEADS = 4       # 注意力头数，要求 D_MODEL % NUM_HEADS == 0
D_FF = 256          # Feed-Forward 中间层维度，通常为 4 × D_MODEL
NUM_LAYERS = 2      # Transformer Encoder 层数
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCH_NUM = 10
LEARNING_RATE = 0.001
TRAIN_SAMPLE_NUM = 1600
TEST_SAMPLE_NUM = 400

random.seed(SEED)
torch.manual_seed(SEED)

# ── 数据配置（与 week03 完全相同）────────────────────────────────────────────
LABEL_NAMES = ["美食", "体育", "科技", "财经"]

CLASS_CONFIG = {
    0: {
        "objects": ["火锅", "烧烤", "面条", "蛋糕", "餐厅", "奶茶", "牛排", "披萨"],
        "actions": ["味道很好", "香气十足", "口感丰富", "非常好吃", "值得推荐"],
        "places": ["店里", "街边", "商场", "夜市", "厨房"],
    },
    1: {
        "objects": ["篮球", "足球", "网球", "比赛", "球员", "教练", "冠军", "训练"],
        "actions": ["完成进球", "拿下胜利", "状态火热", "加强防守", "打出配合"],
        "places": ["球场", "体育馆", "赛场", "训练馆", "主场"],
    },
    2: {
        "objects": ["手机", "芯片", "算法", "机器人", "模型", "系统", "软件", "数据"],
        "actions": ["完成升级", "提升性能", "发布版本", "优化体验", "训练完成"],
        "places": ["实验室", "公司", "平台", "云端", "项目组"],
    },
    3: {
        "objects": ["股票", "基金", "市场", "银行", "公司", "投资", "利润", "订单"],
        "actions": ["价格上涨", "收益增加", "成交活跃", "发布财报", "控制风险"],
        "places": ["交易所", "金融街", "会议室", "公司总部", "市场"],
    },
}

TEMPLATES = [
    "{place}的{obj}{act}",
    "今天{obj}{act}，大家讨论很多",
    "这次{obj}在{place}{act}",
    "新闻报道说{obj}{act}",
    "最近{place}里关于{obj}的话题很多",
]


# ── 数据构建（与 week03 完全相同）────────────────────────────────────────────
def build_sample(label):
    config = CLASS_CONFIG[label]
    template = random.choice(TEMPLATES)
    return template.format(
        place=random.choice(config["places"]),
        obj=random.choice(config["objects"]),
        act=random.choice(config["actions"]),
    )


def build_dataset(sample_num):
    data = []
    class_num = len(LABEL_NAMES)
    for i in range(sample_num):
        label = i % class_num
        data.append((build_sample(label), label))
    random.shuffle(data)
    return data


def build_vocab(data):
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for sentence, _ in data:
        for char in sentence:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


def encode_sentence(sentence, vocab):
    input_ids = [vocab.get(char, vocab["[UNK]"]) for char in sentence]
    input_ids = input_ids[:MAX_LEN]
    input_ids += [vocab["[PAD]"]] * (MAX_LEN - len(input_ids))
    return input_ids


class TextClassificationDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, label = self.data[index]
        input_ids = encode_sentence(sentence, self.vocab)
        return torch.LongTensor(input_ids), torch.LongTensor([label]).squeeze(0)


# ── Transformer 组件 ──────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码。
    Transformer 本身对位置无感，需要将位置信息编码后加到 Embedding 上。

    公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pe shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                                   # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维

        # 注册为 buffer（不参与梯度更新，但会随模型保存/加载）
        self.register_buffer("pe", pe.unsqueeze(0))    # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力（单头）。

    Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

    为什么要除以 √d_k：
        d_k 较大时，QKᵀ 的方差随 d_k 增长，导致 softmax 梯度接近 0（梯度消失）。
        除以 √d_k 将方差归一化，使梯度稳定。
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q/k/v shape: (batch, heads, seq, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, heads, seq, seq)

        if mask is not None:
            # mask=True 的位置填充 -inf，softmax 后趋近于 0，即忽略 PAD token
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, v)   # (batch, heads, seq, d_k)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力。

    将 d_model 均分成 num_heads 个子空间，每个头独立计算注意力，
    最后拼接所有头的输出再做一次线性投影。

    好处：不同的头可以关注序列中不同类型的依赖关系。
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # Q / K / V 三个投影矩阵
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 多头输出拼接后的投影
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, x, mask=None):
        # x: (batch, seq, d_model)
        batch_size = x.size(0)

        # 1. 线性投影 → 拆分多头
        # (batch, seq, d_model) → (batch, seq, num_heads, d_k) → (batch, num_heads, seq, d_k)
        q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 缩放点积注意力（所有头并行）
        context = self.attention(q, k, v, mask)   # (batch, heads, seq, d_k)

        # 3. 拼接多头输出 → 投影
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.w_o(context)   # (batch, seq, d_model)


class FeedForward(nn.Module):
    """
    位置前馈网络（Point-wise Feed-Forward）。

    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

    对序列中每个位置独立做相同的两层线性变换，中间维度 d_ff 通常为 4 × d_model。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    单层 Transformer Encoder。

    结构（Pre-LN 变体，训练更稳定）：
        x → LayerNorm → MHA → Dropout → + x  →
          → LayerNorm → FFN → Dropout → + x  → 输出
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # src_key_padding_mask: (batch, seq)，True 表示 PAD 位置
        # 扩展成 (batch, 1, 1, seq) 供注意力 mask 使用
        mask = None
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)

        # 子层 1：多头自注意力 + 残差 + LayerNorm（Pre-LN）
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))
        # 子层 2：前馈网络 + 残差 + LayerNorm（Pre-LN）
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TransformerClassifier(nn.Module):
    """
    完整的 Transformer 文本分类模型。

    Embedding + 位置编码 → N × TransformerEncoderLayer
        → Mean Pooling（忽略 PAD）→ Linear 分类头
    """

    def __init__(self, vocab_size, num_classes, d_model=D_MODEL, num_heads=NUM_HEADS,
                 d_ff=D_FF, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=MAX_LEN + 1, dropout=dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classify = nn.Linear(d_model, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # x: (batch, seq_len)
        pad_mask = (x == 0)   # True 表示 PAD token，注意力时需屏蔽

        # Embedding + 位置编码
        out = self.pos_encoding(self.embedding(x))   # (batch, seq, d_model)

        # 逐层 Transformer Encoder
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=pad_mask)

        out = self.norm(out)   # 最后一层额外 LayerNorm（Pre-LN 习惯）

        # Mean Pooling：对非 PAD 位置取平均，得到句向量
        non_pad = (~pad_mask).unsqueeze(-1).float()          # (batch, seq, 1)
        sentence_vec = (out * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)

        logits = self.classify(sentence_vec)   # (batch, num_classes)

        if y is not None:
            return self.loss(logits, y)
        return torch.softmax(logits, dim=1)


# ── 训练 / 评估 ───────────────────────────────────────────────────────────────

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            pred = model(x)
            pred_label = torch.argmax(pred, dim=1)
            correct += (pred_label == y).sum().item()
            total += y.size(0)
    return correct / total


def train(model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCH_NUM):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        accuracy = evaluate(model, test_loader)
        print("第%d轮，平均loss：%.4f，测试准确率：%.4f" % (epoch + 1, avg_loss, accuracy))


def predict(model, vocab, sentences):
    model.eval()
    with torch.no_grad():
        for sentence in sentences:
            input_ids = torch.LongTensor([encode_sentence(sentence, vocab)])
            result = model(input_ids)[0]
            pred_label = int(torch.argmax(result))
            print(
                "输入：%s\n  预测类别：%s，概率分布：%s\n"
                % (sentence, LABEL_NAMES[pred_label], [f"{p:.3f}" for p in result.tolist()])
            )


def main():
    train_data = build_dataset(TRAIN_SAMPLE_NUM)
    test_data = build_dataset(TEST_SAMPLE_NUM)
    vocab = build_vocab(train_data)

    train_loader = DataLoader(
        TextClassificationDataset(train_data, vocab),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        TextClassificationDataset(test_data, vocab),
        batch_size=BATCH_SIZE,
    )

    print("训练样本数：%d" % len(train_data))
    print("测试样本数：%d" % len(test_data))
    print("词表大小：%d" % len(vocab))
    print("分类类别：%s" % "、".join(LABEL_NAMES))
    print("模型结构：Transformer Encoder（%d层，%d头，d_model=%d）" % (NUM_LAYERS, NUM_HEADS, D_MODEL))

    model = TransformerClassifier(vocab_size=len(vocab), num_classes=len(LABEL_NAMES))
    print("\n参数量：%d\n" % sum(p.numel() for p in model.parameters()))

    train(model, train_loader, test_loader)

    print("\n── 预测示例 ──")
    test_sentences = [
        "这家火锅味道很好，排队的人特别多",
        "篮球比赛最后一分钟完成进球",
        "新的算法模型提升性能",
        "股票市场今天成交活跃",
    ]
    predict(model, vocab, test_sentences)


if __name__ == "__main__":
    main()
