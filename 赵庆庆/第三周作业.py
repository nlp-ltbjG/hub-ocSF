import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader

# ──── 词典 ──────────────────────────────────────
vocab = list("你和我他她它爱想欢喜怒哀乐好坏美丑梦星辰日月风云天山金木水火土")
word2idx = {w:i for i,w in enumerate(vocab)}
vocab_size = len(vocab)
num_classes = 5

# ──── 数据集 ──────────────────────────────────────
class MyDataset(Dataset):
    def __init__(self, n):
        self.data = []
        for _ in range(n):
            label = random.randint(0,4)
            text = [""]*5
            text[label] = "你"
            for i in range(5):
                if text[i]=="":
                    text[i] = random.choice([c for c in vocab if c!="你"])
            idx = [word2idx[c] for c in text]
            self.data.append( (idx, label) )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        x,y = self.data[i]
        return torch.tensor(x), torch.tensor(y)
    
# ────  模型 ──────────────────────────────────────
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)   # 词嵌入
        self.rnn = nn.RNN(16, 32, batch_first=True) # RNN
        self.fc = nn.Linear(32, num_classes)      # 分类

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

 
# ────  训练 ──────────────────────────────────────
def train(model, loader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    for ep in range(epochs):
        model.train()
        loss_sum, correct = 0,0
        for x,y in loader:
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred,y)
            loss.backward()
            opt.step()
            
            loss_sum += loss.item()
            correct += (pred.argmax(1)==y).sum().item()
        
        acc = correct/len(loader.dataset)
        print(f"Epoch {ep+1} | loss: {loss_sum:.2f} | acc: {acc:.2%}")

# ────  测试 ──────────────────────────────────────
def test(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x,y in loader:
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
    print(f"测试准确率: {correct/len(loader.dataset):.2%}")

# ────  预测 ──────────────────────────────────────
def predict(model, text):
    idx = [word2idx[c] for c in text]
    x = torch.tensor([idx])
    p = model(x).argmax(1).item()
    print(f"文本：{text} → 你在第 {p+1} 位")

# ────  运行 ──────────────────────────────────────
if __name__ == "__main__":
    train_loader = DataLoader(MyDataset(8000), batch_size=128, shuffle=True)
    test_loader = DataLoader(MyDataset(2000), batch_size=128)
    
    model = RNNModel()
    print("===== 训练 RNN 模型 =====")
    train(model, train_loader)
    test(model, test_loader)

    print("\n===== 预测示例 =====")
    predict(model, "我你月欢喜")
    predict(model, "天你梦星月")
    predict(model, "我和你喜乐")
    predict(model, "你爱风和云")
    predict(model, "山水星辰你")
