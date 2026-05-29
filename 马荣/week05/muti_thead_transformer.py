import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class MultiheadTransformer(nn.Module):
    def __init__(self,hidden_size,heads,device):
        super().__init__()
        self.device=device
        self.heads=heads
        self.dk=hidden_size//heads
        self.qkv=nn.Linear(hidden_size,hidden_size*3)
        self.out=nn.Linear(hidden_size,hidden_size)

    def forward(self,x):
        B,T,H=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(B,T,self.heads,self.dk).transpose(1,2)
        k=k.view(B,T,self.heads,self.dk).transpose(1,2)
        v=v.view(B,T,self.heads,self.dk).transpose(1,2)#[B,head,T,dk]
        x=q@k.transpose(2,3)#[B,head,T,T]
        score=x/math.sqrt(self.dk)
        mask=torch.tril(torch.ones(size=(T,T))).to(self.device)
        score=score.masked_fill(mask==0,-1e9)
        att=F.softmax(score,dim=-1)#[B,head,T,T]
        out=att@v#[B,head,T,dk]
        out=out.transpose(1,2).contiguous().view(B,T,H)#contiguous把 tensor 在内存中重新变成“连续存储”。
        return self.out(out)

class DecoderLayer(nn.Module):
    def __init__(self,hidden_size,heads,device):
        super().__init__()
        self.attention=MultiheadTransformer(hidden_size=hidden_size,heads=heads,device=device)
        self.ln1=nn.LayerNorm(hidden_size)
        self.ln2=nn.LayerNorm(hidden_size)
        self.ffn=nn.Sequential(
            nn.Linear(hidden_size,hidden_size*3),
            nn.GELU(),
            nn.Linear(hidden_size*3,hidden_size),
        )
    def forward(self,x):
        x=x+self.attention(self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x

class Decoder(nn.Module):
    def __init__(self,hidden_size,heads,n_layers,device):
        super().__init__()
        self.decoder_layer=nn.ModuleList([DecoderLayer(hidden_size,heads,device) for _ in range(n_layers)])

    def forward(self,x):
        for layer in self.decoder_layer:
            x=layer(x)
        return x




