"""
给定一个语料，训练一个模型出来，用自己训练的模型生成文本 
文本输出通顺的句子 （rnn先来，然后是transformer）
只续写，不需要问答

当前是基于transformer的模型训练
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from argparse import ArgumentParser

from transformer import TransformerEncoder

# 1.构建数据集---------------------------------------------------
def get_text(file_path):
    """从文件读取语料"""
    try:
        with open(file_path, mode='r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
        return ""
    except Exception as e:
        print(f"读取文件出错: {e}")
        return ""
        
    
def build_vocab(text, sort=True):
    """
    构建词表
    Args:
        text: 可迭代对象，每个元素是一个词（或字符）
        sort: 是否排序以保证词表顺序稳定
    Returns:
        char2idx: dict, 词到索引的映射
        idx2char: dict, 索引到词的映射
    """
    vocab_set = set(text)
    if sort:
        vocab_list = sorted(vocab_set)   # 排序后顺序固定
    else:
        vocab_list = list(vocab_set)
    vocab_list = ['PAD', 'UND'] + vocab_list
    char2idx = {word: idx for idx, word in enumerate(vocab_list)}
    idx2char = {idx: word for word, idx in char2idx.items()}
    return char2idx, idx2char

def encode(x, char2idx):
    """翻译每一句x"""
    return [char2idx.get(ch, 1) for ch in x]


class TextDataSet(Dataset):
    def __init__(self, text, char2idx, seq_len=5):
        self.X = []
        self.Y = []
        data = encode(text, char2idx)
        # 对于每个位置 i，取 text[i : i+seq_len] 作为输入,错一位输出
        for i in range(len(data) - seq_len):
            self.X.append(data[i: i + seq_len])
            self.Y.append(data[i + 1: i + seq_len + 1])
        
    def __len__(self):
        # return max(0, len(self.data) - self.seq_len)？？
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.Y[idx], dtype=torch.long)
    
    
# 2.创建模型-----------------------------------------------------
class LaguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embed_dim, num_layers, nhead, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.transf = TransformerEncoder(hidden_dim, num_layers, nhead, dim_feedforward = 128, dropout=dropout)
        
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)           # (batch, seq_len, embed_dim)
        # 用rnn
        # out, _ = self.rnn(emb)            # out: (batch, seq_len, hidden_dim)
        
        # 用transformer
        out = self.transf(emb)
        logits = self.fc(out)         # (batch, seq_len, vocab_size)
        return logits


# 3.模型训练-----------------------------------------------------
def train():
    text = get_text(args.corpus)
    
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    
    data_set = TextDataSet(text, char2idx, args.seq_len)
    train_loader = DataLoader(data_set, args.batch_size, shuffle=True)
    
    model = LaguageModel(vocab_size, args.hidden_dim, args.embed_dim, args.num_layers, args.nhead)
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    certain = nn.CrossEntropyLoss() # 损失函数
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y_true in train_loader: # y_true （batch_size, seq_len）
            optimizer.zero_grad()
            y_pred = model(x)  # batch_size, seq_len, vocab_size
            # ！！！注意损失函数的处理: CrossEntropyLoss要求: y_pred(n, vocab_size)  y_true(n)
            loss = certain(y_pred.reshape(-1, y_pred.size(-1)),  y_true.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if epoch % args.print_every == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f}")

    
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "args": vars(args),
    }, args.save)
    print(f"模型已保存为 {args.save}")
        
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model",      default="tsf", choices=["rnn", "lstm, tsf"])
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--seq_len",    type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=64)
    parser.add_argument("--nhead",      type=int,   default=8)
    parser.add_argument("--hidden_dim", type=int,   default=64)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--save",       default="tsf_model.pt")
    # parser.add_argument("--save",       default="rnn_model.pt")
    parser.add_argument("--print_every", type=int, default=5, help="每多少轮打印一次损失")
    parser.add_argument("--corpus", default="corpus.txt", help="语料文件路径")
    
    args = parser.parse_args()

    train()
    