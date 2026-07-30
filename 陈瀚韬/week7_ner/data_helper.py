import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=150):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        tags = item["ner_tags"]

        tag_ids = [self.label2id[t] for t in tags]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(0)
        label_ids = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_wid:
                if wid < len(tag_ids):
                    label_ids.append(tag_ids[wid])
                else:
                    label_ids.append(-100)
                prev_wid = wid
            else:
                label_ids.append(-100)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataloader(data, tokenizer, label2id, batch_size=32, shuffle=False, max_len=150):
    ds = NERDataset(data, tokenizer, label2id, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
