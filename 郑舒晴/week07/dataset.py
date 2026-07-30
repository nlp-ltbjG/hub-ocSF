import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent


class PeoplesDailyDataset(Dataset):
    """处理 CoNLL 格式的数据集"""
    
    def __init__(
        self,
        file_path: Path,
        tokenizer: BertTokenizer,
        label2id: Dict[str, int],
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        
        # 读取并解析 CoNLL 格式文件
        self.samples = self._read_conll(file_path)
    
    def _read_conll(self, file_path: Path) -> List[Tuple[List[str], List[str]]]:
        """读取 CoNLL 格式文件，返回样本列表"""
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            tokens = []
            labels = []
            for line in f:
                line = line.strip()
                if not line:  # 空行表示句子结束
                    if tokens and labels:
                        samples.append((tokens, labels))
                        tokens = []
                        labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        labels.append(parts[1])
            
            # 添加最后一个样本（如果文件不以空行结尾）
            if tokens and labels:
                samples.append((tokens, labels))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens, labels = self.samples[idx]
        
        # 将标签转换为 ID
        label_ids = [self.label2id.get(label, self.label2id["O"]) for label in labels]
        
        # 使用 tokenizer 编码文本
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # 处理标签对齐
        word_ids = encoding.word_ids()
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # 特殊 token
            else:
                aligned_labels.append(label_ids[word_id])
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def build_peoples_daily_dataloaders(
    tokenizer: BertTokenizer,
    label2id: Dict[str, int],
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Path = None,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """为 peoples_daily 数据集构建 DataLoader"""
    if data_dir is None:
        data_dir = ROOT / "data" / "peoples_daily"
    
    # 创建数据集
    train_dataset = PeoplesDailyDataset(
        file_path=data_dir / "train.txt",
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )
    
    val_dataset = PeoplesDailyDataset(
        file_path=data_dir / "dev.txt",
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # 创建 id2label 映射
    id2label = {v: k for k, v in label2id.items()}
    
    return train_loader, val_loader, id2label


def get_label_mapping(data_dir: Path = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """从数据集中提取标签映射"""
    if data_dir is None:
        data_dir = ROOT / "data" / "peoples_daily"
    
    # 从训练集中提取所有唯一标签
    labels = set()
    with open(data_dir / "train.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    labels.add(parts[1])
    
    # 确保标签列表包含 "O" 并排序
    labels = sorted(labels)
    if "O" not in labels:
        labels = ["O"] + labels
    
    # 创建映射
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    return label2id, id2label
