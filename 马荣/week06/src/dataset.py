from torch.utils.data import DataLoader,Dataset
import json
from transformers import BertTokenizer
import torch
from pathlib import Path

class TnewsDataset(Dataset):
    def __init__(self,data_path,tokenizer:BertTokenizer,max_length=128):
        super().__init__()
        with open(data_path,"r",encoding="utf-8") as f:
            self.data=json.load(f)
        self.tokenizer=tokenizer
        self.max_length=max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        item=self.data[index]
        encoding=self.tokenizer(
            item["sentence"],
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",# 不足 max_length 时用 [PAD] 填充
            truncation=True # 超出 max_length 时截断
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       
            "attention_mask": encoding["attention_mask"].squeeze(0),  
            "token_type_ids": encoding["token_type_ids"].squeeze(0),  
            "label":          torch.tensor(item["label"], dtype=torch.long),
        }

def build_dataloaders(
    data_dir: Path,
    tokenizer: BertTokenizer,
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    一次性构建 train / val / test 三个 DataLoader。
    num_workers=0 在 Windows 上更稳定（避免多进程 pickle 问题）。
    """
    train_ds = TnewsDataset(data_dir / "train.json", tokenizer, max_length)
    val_ds   = TnewsDataset(data_dir / "val.json",   tokenizer, max_length)
    test_ds  = TnewsDataset(data_dir / "test.json",  tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"DataLoader 构建完成")
    print(f"  train: {len(train_ds)} 条, {len(train_loader)} batch")
    print(f"  val  : {len(val_ds)} 条, {len(val_loader)} batch")
    print(f"  test : {len(test_ds)} 条, {len(test_loader)} batch")

    return train_loader, val_loader, test_loader