import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from muti_thead_transformer import Decoder
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

#构建词汇表
def prepare_data():
    with open("corpus.txt","r",encoding="utf-8") as f:
        data=f.read()
        return "".join(data)

def create_vocal(texts):
    data=sorted(set(texts))
    num2word={num:word for num,word in enumerate(data)}
    word2num={word:num for num,word in num2word.items()}
    return word2num,num2word

class MyDataset(Dataset):
    def __init__(self,texts,word2num,seq_length):
        self.seq_length=seq_length
        ids=[word2num[word] for word in texts]
        self.data=torch.tensor(ids,dtype=torch.long)

    def __len__(self):
        return max(0,len(self.data)-self.seq_length)

    def __getitem__(self, idx):
        x=self.data[idx:idx+self.seq_length]
        y=self.data[idx+1:idx+self.seq_length+1]
        return x,y



#模型
class DecoderOnly(nn.Module):
    def __init__(self,vocal_size,hidden_size,heads,n_layers,seq_length):
        super().__init__()
        self.token_embed=nn.Embedding(vocal_size,hidden_size)
        self.pos_embed=nn.Embedding(seq_length,hidden_size)
        self.decoder=Decoder(hidden_size,heads,n_layers,device)
        self.seq_length=seq_length
        self.ln=nn.LayerNorm(hidden_size)
        self.lm_head=nn.Linear(hidden_size,vocal_size)

    def forward(self,x):
        B,T=x.shape
        pos_embed=torch.arange(T,device=x.device)
        x=self.token_embed(x)+self.pos_embed(pos_embed)
        x=self.decoder(x)
        x=self.ln(x)
        logits=self.lm_head(x)
        return logits


#训练

texts=prepare_data()
word2num,num2word=create_vocal(texts)
# print(len(texts))
# print(word2num)


dataset=MyDataset(texts,word2num,seq_length=64)
loader=DataLoader(dataset,batch_size=32,shuffle=True)

model=DecoderOnly(
    vocal_size=len(word2num),
    hidden_size=768,
    heads=12,
    n_layers=3,
    seq_length=64
).to(device)

optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
loss_function=nn.CrossEntropyLoss()
def train():
    for epoch in range(50):
        model.train()
        total_loss=0
        num_batches = 0
        for x,y in loader:#x,y.shape=[B,seq_length]
            x,y=x.to(device),y.to(device)
            logits=model(x)#[B,seq_length,vocal_size]
            B,T,V=logits.shape
            loss=loss_function(logits.view(B*T,V),y.view(B*T))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            num_batches += 1
        print(f"epoch:{epoch},loss:{total_loss/num_batches}")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }, "gpt_ckpt.pt")

def generate(start_text, max_new_tokens=100):
    ckpt = torch.load("gpt_ckpt.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = [word2num[c] for c in start_text if c in word2num]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # 只保留最近 seq_length 个 token
        x_cond = x[:, -64:]

        with torch.no_grad():
            logits = model(x_cond)

        next_logits = logits[:, -1, :]
        probs = F.softmax(next_logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_id], dim=1)

    out_ids = x[0].tolist()
    return "".join(num2word[i] for i in out_ids)

if __name__=="__main__":
    # train()
    print(generate("银河基金管理公司"))
