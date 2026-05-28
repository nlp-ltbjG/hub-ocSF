from torch.utils.data import DataLoader, Dataset
import json
from transformers import BertTokenizer
import torch
from pathlib import Path

class TnewsDataset(Dataset):
    def __init__(self, data_path, tokenizer: BertTokenizer, max_length=128):
        super().__init__()
        # 读取JSON格式的文本分类数据
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # 返回数据集总条数
        return len(self.data)

    def __getitem__(self, idx):
        # 获取单条数据
        item = self.data[idx]
        text = item["sentence"]  # 假设JSON里的文本字段叫sentence
        label = item["label"]    # 假设JSON里的标签字段叫label

        # 用BERT Tokenizer做文本编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 去掉多余的batch维度，返回模型需要的格式
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }
