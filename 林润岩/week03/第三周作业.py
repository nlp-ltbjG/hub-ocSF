#coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于PyTorch的RNN模型
任务：判断文本中"你/我/他"任意字符的首次出现位置（未出现则为-1）
"""

class RNNPositionModel(nn.Module):
    def __init__(self, vocab_size, vector_dim, hidden_dim, sentence_length):
        super(RNNPositionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)  
        self.rnn = nn.LSTM(
            input_size=vector_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Linear(hidden_dim * 2, sentence_length + 1)  
        self.loss = nn.CrossEntropyLoss()  

    def forward(self, x, y=None):
        embed_x = self.embedding(x)  
        rnn_out, _ = self.rnn(embed_x)  
        
        global_feature = rnn_out[:, -1, :] 
        pred_logits = self.fc(global_feature)  

        if y is not None:
            return self.loss(pred_logits, y)  # 计算损失
        else:
            return torch.argmax(pred_logits, dim= -1)  # 返回预测位置索引

# 构建字符-序号映射表
def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}  # pad填充符，序号0
    for idx, char in enumerate(chars):
        vocab[char] = idx + 1
    vocab["unk"] = len(vocab)  # 未知字符序号
    return vocab

def build_sample(vocab, sentence_length):
    chars = random.choices(list(vocab.keys()), k=sentence_length)
    target_pos = -1
    for idx, char in enumerate(chars):
        if char in {"你", "我", "他"}:
            target_pos = idx
            break
    label = sentence_length if target_pos == -1 else target_pos
    input_ids = [vocab.get(c, vocab["unk"]) for c in chars]
    return input_ids, label

# 构建数据集
def build_dataset(sample_num, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_num):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 初始化模型
def build_model(vocab_size, char_dim, hidden_dim, sentence_length):
    model = RNNPositionModel(vocab_size, char_dim, hidden_dim, sentence_length)
    return model

# 模型评估：计算位置预测准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 生成200个测试样本
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 预测位置
        # 统计正确数：预测标签 == 真实标签
        correct = torch.sum(y_pred == y).item()
        acc = correct / len(y)
        print(f"测试集总数：{len(y)}")
        print(f"正确预测数：{correct}，准确率：{acc:.4f}")
        # 随机打印5个样本的预测结果
        print("示例预测结果：")
        idx_list = random.sample(range(len(x)), 5)
        for idx in idx_list:
            chars = []
            for char_id in x[idx]:
                for c, cid in vocab.items():
                    if cid == char_id:
                        chars.append(c)
                        break
            true_pos = y[idx].item()
            pred_pos = y_pred[idx].item()
            true_pos = -1 if true_pos == sentence_length else true_pos
            pred_pos = -1 if pred_pos == sentence_length else pred_pos
            print(f"文本：{''.join(chars)} | 真实位置：{true_pos} | 预测位置：{pred_pos}")
    return acc

# 主训练流程
def main():
    # 超参数配置
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 批次大小
    train_sample_num = 1000  # 每轮训练样本数
    char_dim = 32  # 字符嵌入维度
    hidden_dim = 64  # RNN隐藏层维度
    sentence_length = 8  # 文本长度
    learning_rate = 0.001  # 学习率

    # 初始化组件
    vocab = build_vocab()
    vocab_size = len(vocab)
    model = build_model(vocab_size, char_dim, hidden_dim, sentence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0.0
        batch_num = int(train_sample_num / batch_size)
        for _ in range(batch_num):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optimizer.zero_grad()  # 梯度清零
            loss = model(x, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()
        # 每轮训练后评估
        print(f"\n===== 第{epoch+1}轮训练 =====")
        print(f"平均loss：{total_loss / batch_num:.4f}")
        evaluate(model, vocab, sentence_length)

    # 保存模型和词表
    torch.save(model.state_dict(), "rnn_position_model.pth")
    with open("vocab_position.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("\n模型已保存为 rnn_position_model.pth，词表已保存为 vocab_position.json")

# 预测函数：输入文本，返回目标字符首次出现位置
def predict(model_path, vocab_path, input_strings, sentence_length):
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    vocab_size = len(vocab)
    model = build_model(vocab_size, 32, 64, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_ids = []
    for s in input_strings:
        if len(s) > sentence_length:
            s = s[:sentence_length]
        s = s.ljust(sentence_length, " ")

        ids = [vocab.get(c, vocab["unk"]) for c in s]
        input_ids.append(ids)

    input_ids = torch.LongTensor(input_ids)

    # 预测
    with torch.no_grad():
        pred_labels = model(input_ids)
    
    # 输出结果
    print("\n===== 预测结果 =====")
    for idx, s in enumerate(input_strings):
        pred_pos = pred_labels[idx].item()
        pred_pos = -1 if pred_pos == sentence_length else pred_pos
        print(f"输入文本：{s} | 目标字符首次出现位置：{pred_pos}")

if __name__ == "__main__":
    # 训练模型
    main()
    # 测试预测
    test_strings = ["a我bcdefg", "xyz你123", "abcdefgh", "他abcdefg", "mnopq我rs"]
    predict("rnn_position_model.pth", "vocab_position.json", test_strings, sentence_length=8)
