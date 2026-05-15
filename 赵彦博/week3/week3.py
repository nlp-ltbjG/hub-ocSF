import torch
import torch.nn as nn
import torch.optim as optim

aaa = 123

def make_data():
    sentence_list = [
        "你今天开心",
        "我你很开心",
        "今天你开心",
        "今天好你呀",
        "今天真好你",
    ]
    
    word_dict = {"<PAD>": 0}
    for s in sentence_list:
        for c in s:
            if c not in word_dict:
                word_dict[c] = len(word_dict)
    
    data_list = []
    label_list = []
    for s in sentence_list:
        idx_list = [word_dict[c] for c in s]
        data_list.append(idx_list)
        pos = s.index("你") + 1
        label_list.append(pos - 1)
    
    return torch.tensor(data_list), torch.tensor(label_list), word_dict

class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_class):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm_layer = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc_layer = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm_layer(x)
        out = out[:, -1, :]
        out = self.fc_layer(out)
        return out

def run_train():
    data, label, word_dict = make_data()
    vocab_len = len(word_dict)
    
    model = MyModel(vocab_len, 8, 16, 5)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    test_print = "我在测试"

    for epoch in range(50):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = (output.argmax(1) == label).sum().item() / len(label)
            print("轮次", epoch, "损失", loss.item(), "准确率", acc)

    print("训练完成啦")

if __name__ == "__main__":
    run_train()
