"""
dataset.py
peoples_daily 数据集处理模块
支持 BIO 标签体系，处理子词对齐问题
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
ENTITY_TYPES = ["PER", "ORG", "LOC"]

def build_label_schema() -> tuple[list, dict, dict]:
    """构建 BIO 标签体系。"""
    labels = ["O"]
    for entity_type in ENTITY_TYPES:
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label

def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    """加载数据集记录。"""
    d = data_dir if data_dir else DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)

class PeoplesDailyDataset(Dataset):
    """peoples_daily 的 PyTorch Dataset。"""
    
    def __init__(self, records: list, tokenizer: BertTokenizer, label2id: dict, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        text = record["text"]
        entities = record.get("entities", [])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        
        offset_mapping = encoding["offset_mapping"]
        aligned_labels = [-100] * self.max_length
        
        for entity in entities:
            start = entity["start_idx"]
            end = entity["end_idx"]
            entity_type = entity["type"]
            
            for i in range(self.max_length):
                token_start, token_end = offset_mapping[i]
                if token_start == token_end == 0:
                    continue
                
                if token_start < end and token_end > start:
                    if token_start == start:
                        label = f"B-{entity_type}"
                    else:
                        label = f"I-{entity_type}"
                    aligned_labels[i] = self.label2id.get(label, self.label2id["O"])
        
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)
        
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(encoding["token_type_ids"], dtype=torch.long),
            "labels": labels_tensor,
        }

def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int,
    max_length: int,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)
    
    print(f"数据集规模：训练={len(train_records)}，验证={len(val_records)}，测试={len(test_records)}")
    
    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

"""
model.py
"""
"""
BertNER 和 BertCRFNER 模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel

def _load_bert(bert_path: str) -> BertModel:
    """加载 BERT 预训练模型。"""
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    bert = BertModel.from_pretrained(bert_path, local_files_only=True)
    transformers.logging.set_verbosity(prev)
    return bert

class BertNER(nn.Module):
    """BERT + 线性分类头。"""
    
    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state
        logits = self.classifier(self.dropout(seq_output))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

class BertCRFNER(nn.Module):
    """BERT + CRF 层。"""
    
    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        from torchcrf import CRF
        
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def _get_emissions(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state
        return self.classifier(self.dropout(seq_output))

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        
        loss = None
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")
        
        return emissions, loss

    def decode(self, input_ids, attention_mask, token_type_ids):
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)

def build_model(use_crf: bool, bert_path: str, num_labels: int, dropout: float = 0.1) -> nn.Module:
    """模型工厂函数。"""
    model_cls = BertCRFNER if use_crf else BertNER
    model = model_cls(bert_path=bert_path, num_labels=num_labels, dropout=dropout)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "BERT + CRF" if use_crf else "BERT + Linear"
    
    print(f"模型：{model_name}")
    print(f"  标签数：{num_labels}")
    print(f"  参数总量：{total_params / 1e6:.1f}M")
    print(f"  可训练参数：{trainable_params / 1e6:.1f}M")
    
    return model

"""
train.py
"""
"""
BERT NER 训练脚本（peoples_daily 数据集）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import build_label_schema, build_dataloaders
from model import build_model

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="BERT NER 训练")
    parser.add_argument("--bert_path", type=str, default="C:\\bert_model\\bert-base-chinese")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=10.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()

def evaluate_epoch(model, loader, id2label, device, use_crf):
    """验证评估函数。"""
    from seqeval.metrics import f1_score as seqeval_f1
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_golds = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            
            if use_crf:
                emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()
            
            total_loss += loss.item()
            labels_np = labels.cpu().tolist()
            
            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_np[i]
                pred_ids = pred_ids_list[i] if use_crf else pred_ids_list[i]
                
                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        pred_seq.append(id2label.get(pred_ids[j], "O") if j < len(pred_ids) else "O")
                    else:
                        pred_seq.append(id2label.get(pred_ids[j], "O"))
                
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)
    
    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1

def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum):
    """训练一轮。"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        
        _, loss = model(input_ids, attention_mask, token_type_ids, labels)
        (loss / grad_accum).backward()
        total_loss += loss.item()
        
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    remainder = len(loader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / len(loader)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    
    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（O + {len(labels) - 1} 个实体标签）")
    
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path), local_files_only=True)
    
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=DATA_DIR,
    )
    
    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(device)
    
    bert_params = list(model.bert.parameters())
    head_params = [p for n, p in model.named_parameters() if "bert" not in n]
    
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ])
    
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    
    log_data = []
    best_f1 = 0.0
    model_suffix = "crf" if args.use_crf else "linear"
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, args.grad_accum)
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_entity_f1={val_f1:.4f} | time={elapsed:.1f}s")
        
        log_data.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_entity_f1": val_f1,
            "elapsed_s": elapsed,
        })
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_entity_f1": val_f1,
            }, CKPT_DIR / f"bert_{model_suffix}_best.pt")
            print(f"★ 最优 val_entity_f1，checkpoint 已保存")
        
        torch.save({
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_entity_f1": val_f1,
        }, CKPT_DIR / f"bert_{model_suffix}_last.pt")
    
    with open(LOG_DIR / f"train_{model_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

"""
evaluate.py
"""
"""
模型评估脚本（peoples_daily 数据集）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from transformers import BertTokenizer
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report as seqeval_report

from dataset import build_label_schema, build_dataloaders
from model import build_model

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"

def count_illegal_sequences(pred_seqs: list) -> dict:
    """统计非法 BIO 序列。"""
    illegal_start = 0
    illegal_transition = 0
    
    for seq in pred_seqs:
        if seq and seq[0].startswith("I-"):
            illegal_start += 1
        
        for i in range(1, len(seq)):
            prev = seq[i-1]
            curr = seq[i]
            if curr.startswith("I-"):
                entity_type = curr[2:]
                if not prev.startswith(f"B-{entity_type}") and not prev.startswith(f"I-{entity_type}"):
                    illegal_transition += 1
    
    return {
        "illegal_start": illegal_start,
        "illegal_transition": illegal_transition,
        "total_seqs": len(pred_seqs),
        "total_illegal": illegal_start + illegal_transition,
    }

def evaluate(model, loader, id2label, device, use_crf):
    """执行评估。"""
    model.eval()
    all_preds = []
    all_golds = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            
            if use_crf:
                emissions, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).tolist()
            
            labels_np = labels.cpu().tolist()
            
            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_np[i]
                pred_ids = pred_ids_list[i] if use_crf else pred_ids_list[i]
                
                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        pred_seq.append(id2label.get(pred_ids[j], "O") if j < len(pred_ids) else "O")
                    else:
                        pred_seq.append(id2label.get(pred_ids[j], "O"))
                
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)
    
    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f1 = f1_score(all_golds, all_preds)
    
    illegal_stats = count_illegal_sequences(all_preds)
    
    return p, r, f1, all_golds, all_preds, illegal_stats

def main():
    parser = argparse.ArgumentParser(description="评估 BERT NER 模型")
    parser.add_argument("--bert_path", type=str, default="C:\\bert_model\\bert-base-chinese")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    
    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    
    model_suffix = "crf" if args.use_crf else "linear"
    ckpt_path = CKPT_DIR / f"bert_{model_suffix}_best.pt"
    
    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=num_labels,
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"加载 checkpoint（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")
    
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path), local_files_only=True)
    _, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=DATA_DIR,
    )
    
    loader = val_loader if args.split == "validation" else test_loader
    
    p, r, f1, all_golds, all_preds, illegal_stats = evaluate(model, loader, id2label, device, args.use_crf)
    
    print(f"\nEntity-level Precision: {p:.4f}")
    print(f"Entity-level Recall:    {r:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")
    
    print("\n【逐类型 F1】")
    print(seqeval_report(all_golds, all_preds, digits=4))
    
    print("\n【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头（I-X 开头）：{illegal_stats['illegal_start']} 条")
    print(f"  非法转移（B-X/I-X → I-Y, X≠Y）：{illegal_stats['illegal_transition']} 条")
    print(f"  合计非法序列：{illegal_stats['total_illegal']} 条")
    
    result = {
        "model": "BERT+CRF" if args.use_crf else "BERT+Linear",
        "split": args.split,
        "precision": p,
        "recall": r,
        "f1": f1,
        "illegal_stats": illegal_stats,
    }
    
    with open(LOG_DIR / f"eval_{model_suffix}_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
