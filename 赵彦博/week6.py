import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

vocab_size = 200
embed_dim = 64
seq_len = 15
num_classes = 2
batch_size = 16
lr = 1e-3
epochs = 20

class FcTextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(seq_len * embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class TransformerTextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2,batch_first=True,dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)

def train_model(model, name, loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"\n===== 开始训练 {name} =====")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        for data, label in loader:
            out = model(data)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
        avg_loss = total_loss / len(loader)
        acc = correct / total
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    print(f"{name} 训练完成")

if __name__ == "__main__":
    torch.manual_seed(10)
    train_data = torch.randint(0, vocab_size, (300, seq_len))
    train_label = torch.randint(0, num_classes, (300,))
    dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model1 = FcTextClassifier()
    train_model(model1, "全连接网络", train_loader)

    model2 = TransformerTextClassifier()
    train_model(model2, "Transformer文本分类", train_loader)

    print("\n===== 效果对比总结 =====")
print("1. 全连接网络：结构简单、速度快")
print("2. Transformer：训练麻烦")
