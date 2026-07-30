import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, count=1000, size=10):
        self.count = count
        self.size = size
        self.data = torch.randn(count, size)
        self.label = torch.argmax(self.data, dim=1)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class Net(nn.Module):
    def __init__(self, in_size=10, out_size=10):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(in_size, 32)
        self.layer2 = nn.Linear(32, out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

data_size = 10
class_num = 10
batch = 32
lr = 0.001
epoch = 20

dataset = MyDataset(size=data_size)
loader = DataLoader(dataset, batch_size=batch, shuffle=True)

model = Net(in_size=data_size, out_size=class_num)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for i in range(epoch):
    total_loss = 0
    right = 0
    total_num = 0

    for inputs, targets in loader:
        outputs = model(inputs)
        loss = loss_func(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predict = torch.argmax(outputs, dim=1)
        right += (predict == targets).sum().item()
        total_num += targets.size(0)

    avg_loss = total_loss / len(loader)
    acc = right / total_num
    print(f"epoch {i+1}, loss: {avg_loss:.4f}, acc: {acc:.4f}")
