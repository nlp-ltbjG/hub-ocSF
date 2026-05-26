import os
import torch
import config as cfg
from torch.utils.data import Dataset, DataLoader

def clean_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    cleaned_lines = []
    for line in lines:
        if not (line.startswith('第') and '回' in line and len(line) < 30):
            cleaned_lines.append(line)
    full_text = ''.join(cleaned_lines)
    full_text = ''.join([c for c in full_text if '\u4e00' <= c <= '\u9fff' 
                        or c in '，。！？；：""''()（）、1234567890'])
    return full_text

def load_jinyong_dataset(data_dir='./jinyong'):
    all_text = ''
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                all_text += clean_text(text)
    return all_text

def build_vocab(text):
    chars = sorted(list(set(text)))
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>'] #sos-start of sequence, eos-end of sequence
    vocab = special_tokens + chars
    char2idx = {c:i for i,c in enumerate(vocab)}
    idx2char = {i:c for i,c in enumerate(vocab)}
    return vocab, char2idx, idx2char

def tokenize(text, char2idx):
    return [char2idx.get(c, char2idx['<unk>']) for c in text]

def detokenize(ids, idx2char):
    return ''.join([idx2char[i] for i in ids])

class JinyongDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        self.char2idx = char2idx
        self.tokens = tokenize(text, char2idx)
        self.num_samples = len(self.tokens) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        input_ids = torch.tensor(self.tokens[start:end], dtype=torch.long)
        labels = torch.tensor(self.tokens[start+1:end+1], dtype=torch.long)
        return input_ids, labels

def get_dataloader(text, char2idx, seq_len, batch_size, shuffle=True):
    dataset = JinyongDataset(text, char2idx, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader