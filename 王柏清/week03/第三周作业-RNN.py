#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个nlp任务
任务：对于任意一个包含“我”字的文本，预测“我”字在文本中的位置；如果文本中没有“我”字，预测一个固定值；如果文本中有多个“我”字，预测另一个固定值。

"""

class TorchRNN(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size):
        super(TorchRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)        #RNN层
        self.classify = nn.Linear(hidden_size, (sentence_length + 2))     #线性层
        self.loss = nn.functional.cross_entropy  #loss函数采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)                   # (batch, sen_len, hidden)
        last_out = rnn_out[:, -1, :]               # 取最后一个时间步 (batch, hidden)
        logits = self.classify(last_out)           # (batch, sentence_length + 2)
        if y is not None:
            return self.loss(logits, y)   #预测值和真实值计算损失
        else:
            return torch.softmax(logits, dim=-1)                 #输出预测结果

# 构建词表
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"   # 字符集（包含“我”）
    vocab = {"pad": 0}
    for idx, char in enumerate(chars):
        vocab[char] = idx + 1                 # 每个字对应一个序号，从1开始
    vocab['unk'] = len(vocab)                # 未知字符
    return vocab

# 生成单个样本
def build_sample(vocab, sentence_length):
    # 1. 随机生成 sentence_length 个字符（可重复）
    chars = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    
    # 2. 找出所有“我”出现的位置（索引从0开始）
    positions = [i for i, ch in enumerate(chars) if ch == '我']
    
    # 3. 根据新规则计算标签
    if len(positions) == 0:
        label = sentence_length                 # 没有“我”
    elif len(positions) == 1:
        label = positions[0]                    # 单个位置索引
    else:
        label = sentence_length + 1             # 多次出现：统一固定值
    
    # 4. 将字符序列转换为对应的ID序列
    x = [vocab.get(ch, vocab['unk']) for ch in chars]
    
    return x, label

# ------------------ 构建数据集 ------------------
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    # 返回 LongTensor (X) 和 LongTensor (Y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = torch.argmax(model(x), dim=-1)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if y_p == y_t:
                correct += 1   #判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 100        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchRNN(char_dim, sentence_length, vocab, hidden_size=16)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = TorchRNN(char_dim, sentence_length, vocab, hidden_size=16)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        pred_class = torch.argmax(result[i], dim=-1).item()
        pred_prob = result[i][pred_class].item()
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, pred_class, pred_prob))



if __name__ == "__main__":
    main()
    #test_strings = ["fnvf我e", "wz你dfg", "rq我d我g", "n我kwww"]
    #predict(r"vocab.json", test_strings)
