# coding:utf8
"""
文本多分类实验：使用 RNN / LSTM / GRU 完成中文短文本主题分类

任务说明：
    输入一条中文短句，判断它属于哪个主题类别。

类别：
    0: 美食
    1: 体育
    2: 科技
    3: 财经

模型结构：
    字符级编码 -> Embedding -> RNN/LSTM/GRU -> 取最后隐藏状态 -> Linear -> CrossEntropyLoss
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
MAX_LEN = 32
EMBEDDING_DIM = 64
HIDDEN_SIZE = 64
BATCH_SIZE = 32
EPOCH_NUM = 8
LEARNING_RATE = 0.001
TRAIN_SAMPLE_NUM = 1600
TEST_SAMPLE_NUM = 400

random.seed(SEED)
torch.manual_seed(SEED)


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


def build_sample(label):
    """构造单条文本样本。"""
    config = CLASS_CONFIG[label]
    template = random.choice(TEMPLATES)
    return template.format(
        place=random.choice(config["places"]),
        obj=random.choice(config["objects"]),
        act=random.choice(config["actions"]),
    )


def build_dataset(sample_num):
    """构造均衡的多分类数据集。"""
    data = []
    class_num = len(LABEL_NAMES)
    for i in range(sample_num):
        label = i % class_num
        data.append((build_sample(label), label))
    random.shuffle(data)
    return data


def build_vocab(data):
    """按中文字符建立词表。"""
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for sentence, _ in data:
        for char in sentence:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


def encode_sentence(sentence, vocab):
    """把文本转成固定长度的字符 id 序列。"""
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


class TextRNNModel(nn.Module):
    def __init__(self, vocab_size, model_type="rnn"):
        super(TextRNNModel, self).__init__()
        self.model_type = model_type.lower()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)

        if self.model_type == "rnn":
            self.sequence_layer = nn.RNN(
                EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True
            )
        elif self.model_type == "lstm":
            self.sequence_layer = nn.LSTM(
                EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True
            )
        elif self.model_type == "gru":
            self.sequence_layer = nn.GRU(
                EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True
            )
        else:
            raise ValueError("model_type 只支持 rnn、lstm、gru")

        self.classify = nn.Linear(HIDDEN_SIZE, len(LABEL_NAMES))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # x: (batch_size, max_len)
        embedding = self.embedding(x)
        output, _ = self.sequence_layer(embedding)

        # 避免取到 [PAD] 位置的输出，定位到每条真实文本的最后一个字符。
        lengths = (x != 0).sum(dim=1).clamp(min=1) - 1
        batch_index = torch.arange(x.size(0), device=x.device)#device 所在计算设备（cpu/cuda）的索引
        sentence_vector = output[batch_index, lengths]
        logits = self.classify(sentence_vector)

        if y is not None:
            return self.loss(logits, y)
        return torch.softmax(logits, dim=1)


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            y_pred = model(x)
            pred_label = torch.argmax(y_pred, dim=1)
            correct += (pred_label == y).sum().item()
            total += y.size(0)
    return correct / total


def train_one_model(model_type, train_loader, test_loader, vocab_size):
    print("\n==============================")
    print("开始训练模型：%s" % model_type.upper())
    model = TextRNNModel(vocab_size, model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH_NUM):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        accuracy = evaluate(model, test_loader)
        print(
            "第%d轮，平均loss：%.4f，测试准确率：%.4f"
            % (epoch + 1, avg_loss, accuracy)
        )

    return model


def predict(model, vocab, sentences):
    model.eval()
    with torch.no_grad():
        for sentence in sentences:
            input_ids = torch.LongTensor([encode_sentence(sentence, vocab)])
            result = model(input_ids)[0]
            pred_label = int(torch.argmax(result))
            print(
                "输入：%s，预测类别：%s，概率分布：%s"
                % (sentence, LABEL_NAMES[pred_label], result.tolist())
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

    trained_models = {}
    for model_type in ["rnn", "lstm", "gru"]:
        trained_models[model_type] = train_one_model(
            model_type, train_loader, test_loader, len(vocab)
        )

    print("\n==============================")
    print("使用 LSTM 模型进行预测示例")
    test_sentences = [
        "这家火锅味道很好，排队的人特别多",
        "篮球比赛最后一分钟完成进球",
        "新的算法模型提升性能",
        "股票市场今天成交活跃",
    ]
    predict(trained_models["lstm"], vocab, test_sentences)


if __name__ == "__main__":
    main()
