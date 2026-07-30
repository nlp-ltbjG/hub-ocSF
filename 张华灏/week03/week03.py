import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED        = 18
N_SAMPLES   = 8000
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 128
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

LOC_WORDS = ['学校', '公司', '图书馆', '公园', '餐厅', '教室', '实验室', '咖啡馆', '操场', '宿舍']
OBJ_WORDS = ['电影', '课程', '作业', '项目', '系统', '游戏', '音乐', '产品', '报告', '文档', '论文', '计划']
ACT_WORDS = ['吃饭', '学习', '跑步', '写代码', '看书', '开会', '休息', '提交作业', '复习', '做实验', '练习', '整理资料']
ADJ_WORDS = ['好看', '有趣', '困难', '重要', '复杂', '简单', '精彩', '无聊', '有用', '耗时', '紧张', '轻松']

TEMPLATES_DECL = [
    ('我今天去了{}',            'loc', None),
    ('他正在{}',                'act', None),
    ('我们正在讨论{}',          'obj', None),
    ('这个{}比较{}',            'obj', 'adj'),
    ('{}的内容很{}',            'obj', 'adj'),
    ('我下午要去{}',            'loc', None),
    ('最近我在学习{}',          'obj', None),
    ('我们今天在{}开会',        'loc', None),
    ('他每天都在{}',            'loc', None),
    ('这个{}已经完成了',        'obj', None),
    ('我觉得{}还不错',          'obj', None),
    ('我们需要认真对待这个{}',  'obj', None),
]

TEMPLATES_QUESTION = [
    ('你今天去{}吗',            'loc', None),
    ('他正在{}吗',              'act', None),
    ('这个{}怎么样呢',          'obj', None),
    ('你为什么喜欢{}',          'obj', None),
    ('我们什么时候去{}',        'loc', None),
    ('{}在哪里',                'loc', None),
    ('这个{}难不难',            'obj', None),
    ('你有没有完成{}',          'obj', None),
    ('{}做完了吗',              'obj', None),
    ('你喜欢在{}学习吗',        'loc', None),
    ('这个{}有意思吗',          'obj', None),
    ('你平时在哪里{}',          'act', None),
]

TEMPLATES_EXCLAIM = [
    ('这个{}太{}了',            'obj', 'adj'),
    ('今天真的太{}了',          'adj', None),
    ('{}真是太{}了',            'loc', 'adj'),
    ('这个{}好{}啊',            'obj', 'adj'),
    ('这次{}太让我{}了',        'obj', 'adj'),
    ('{}真的太棒了',            'loc', None),
    ('这个{}真的很{}',          'obj', 'adj'),
    ('没想到{}这么{}',          'obj', 'adj'),
    ('今天的{}太{}了',          'obj', 'adj'),
    ('这里的{}环境真{}啊',      'loc', 'adj'),
]

TEMPLATES_COMMAND = [
    ('请你{}',                  'act', None),
    ('帮我{}',                  'act', None),
    ('不要{}',                  'act', None),
    ('快点{}',                  'act', None),
    ('记得{}',                  'act', None),
    ('必须完成{}',              'obj', None),
    ('请把{}准备好',            'obj', None),
    ('赶紧去{}',                'loc', None),
    ('不要忘记{}',              'obj', None),
    ('帮我检查一下这个{}',      'obj', None),
    ('你去{}等我',              'loc', None),
    ('把这个{}做完',            'obj', None),
]

WORD_MAP = {
    'loc': LOC_WORDS,
    'obj': OBJ_WORDS,
    'act': ACT_WORDS,
    'adj': ADJ_WORDS,
}


def fill_template(tmpl_tuple):
    tmpl, slot1, slot2 = tmpl_tuple
    w1 = random.choice(WORD_MAP[slot1])
    if slot2:
        w2 = random.choice(WORD_MAP[slot2])
        return tmpl.format(w1, w2)
    return tmpl.format(w1)


def make_declarative():
    return fill_template(random.choice(TEMPLATES_DECL))


def make_question():
    sent = fill_template(random.choice(TEMPLATES_QUESTION))
    if random.random() < 0.5:
        sent += '？'
    return sent


def make_exclamation():
    sent = fill_template(random.choice(TEMPLATES_EXCLAIM))
    if random.random() < 0.7:
        sent += '！'
    return sent


def make_command():
    sent = fill_template(random.choice(TEMPLATES_COMMAND))
    if random.random() < 0.3:
        sent += random.choice(['一下', '吧'])
    return sent


def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n // 4):
        data.append((make_declarative(), 0))
        data.append((make_question(), 1))
        data.append((make_exclamation(), 2))
        data.append((make_command(), 3))
    random.shuffle(data)
    return data


def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float),
        )

class KeywordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.lstm(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)  # (B,)
        return out

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            prob    = model(X)
            pred    = prob.argmax(dim=-1)
            correct += (pred == y.long()).sum().item()
            total   += len(y)
    return correct / total

def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = KeywordLSTM(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    LABEL_MAP = {0: '陈述', 1: '疑问', 2: '感叹', 3: '祈使'}
    test_sents = [
        '我知道你今天去公司',
        '你说的话让我觉得很奇怪',
        '服务太赞了，下次还来',
        '服务员，麻烦过来一下',
        '难到你不知道应该怎么做么',
        '你打算接下来怎么办',
        '好家伙这都能发生',
        '记得明天早点来'
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)  # (1, 4)
            pred = logits.argmax(dim=-1).item()  # 取最大值的index
            probs = torch.softmax(logits, dim=-1)[0]  # 各类概率
            conf = probs[pred].item()  # 预测类别的置信度
            print(f"  [{LABEL_MAP[pred]}({conf:.2f})]  {sent}")

if __name__ == '__main__':
    train()
