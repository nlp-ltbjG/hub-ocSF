"""
设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。如果不知道怎么设计，可以选择如下任务:对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
"""
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader


MAXLENGTH=5
EMBEDDING_DIM=64
EPOCH=50

#准备数据
dataset_list=[
"你是我唯一",
"你来自远方",
"你微笑的脸",
"你说过的话",
"你在我心中",
"你快乐吗？",
"你做过的梦",
"你爱的世界",
"你恨的过往",
"你去向何方",
"问你何时归",
"因你而改变",
"唯你不可替",
"与你共朝夕",
"为你写首诗",
"寻你千百度",
"知你心中事",
"愿你永安康",
"同你看星空",
"对你思念深",
"心中只有你",
"梦里见到你",
"何时遇见你",
"是否记得你",
"远方有个你",
"偏偏想念你",
"轻轻呼唤你",
"深深爱着你",
"静静望着你",
"慢慢走近你",
"天地你最大",
"今日你作主",
"此事你知情",
"前方你领路",
"平生你最爱",
"此番你第一",
"江湖你称雄",
"家中你掌勺",
"未来你创造",
"胜负你决定",
"明日盼你归",
"清晨唤你醒",
"倚门等你回",
"煮酒待你来",
"凭栏望你远",
"写信唤你回",
"点灯照你路",
"折梅赠你手",
"抚琴邀你和",
"备马送你行"
]
def build_data():
    data=[]
    for dataset in dataset_list:
        for i,word in enumerate(dataset):
            if word=="你":
                data.append((dataset,i))#（文字，标签）
    return data

#构建词表
def build_vocal_table(data):
    vocal_dict={'<PAD>':0,'<UNK>':1}
    for sent,label in data:
        for word in sent:
            if word not in vocal_dict:
                vocal_dict[word]=len(vocal_dict)
    return vocal_dict

#编码
def encode(vocal_dict, sent,max_length=MAXLENGTH):
    ids=[vocal_dict[ch] for ch in sent]
    ids = ids[:max_length]
    ids+=[0]*(max_length-len(ids))
    return ids#token ID

class ClassDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(vocal_dict=vocab,sent=sent) for sent,_ in data]
        self.y = [label for _,label in data]
    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )

class ClassiferNet(nn.Module):
    def __init__(self,vocal_size,d_model,hidden_dim):
        super(ClassiferNet, self).__init__()
        self.embedding=nn.Embedding(vocal_size,d_model,padding_idx=0)
        self.rnn=nn.RNN(d_model,hidden_size=hidden_dim,bias=True,batch_first=True)
        self.bn=nn.BatchNorm1d(hidden_dim)
        self.dropout=nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self,x):
        embed=self.embedding(x)
        output,hid_n=self.rnn(embed)
        pooled=output.max(dim=1)[0]
        pooled=self.dropout(self.bn(pooled))
        out=torch.sigmoid(self.fc(pooled))
        return out

def evaluate(model,val_loader):
    model.eval()
    correct=total=0
    with torch.no_grad():
        for X,y in val_loader:
            total+=1
            prob=model(X)
            pred=torch.argmax(prob,dim=1)
            correct += (pred == y).sum().item()
    return correct/total

def train():
    #准备数据
    data=build_data()
    vocal_dict=build_vocal_table(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocal_dict)}")
    print(vocal_dict)
    #数据分为训练和评估
    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]

    model=ClassiferNet(len(vocal_dict),EMBEDDING_DIM,256)
    loss_function = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    train_data=DataLoader(ClassDataset(train_data,vocal_dict),batch_size=5,shuffle=True)
    evl_data=DataLoader(ClassDataset(val_data,vocal_dict),batch_size=5)
    for epoch in range(1,EPOCH+1):
        model.train()
        total_loss=0
        for X,y in train_data:
            model.zero_grad()
            loss=loss_function(model(X),y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_data)
        acc=evaluate(model,evl_data)
        print(f"Epoch {epoch:2d}/{EPOCH}  loss={avg_loss:.4f}  val_acc={acc:.4f}")


if __name__ == '__main__':
    train()
