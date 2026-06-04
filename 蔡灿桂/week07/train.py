import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import build_label_schema, build_dataloaders
from model import build_model

# ===================== 1. 工具与配置函数 =====================

ROOT = Path(__file__).parent.parent
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "cluener"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"


def seed_everything(seed: int):
    """固定所有随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证卷积操作确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_vocab_overlap(loader, tokenizer, threshold=0.02):
    """检查数据集的 UNK 比例，评估领域鸿沟"""
    total_tokens, unk_tokens = 0, 0
    for batch in loader:
        ids = batch["input_ids"].view(-1)
        total_tokens += len(ids)
        unk_tokens += (ids == tokenizer.unk_token_id).sum().item()
    
    unk_ratio = unk_tokens / max(total_tokens, 1)
    print(f"[词汇检查] 总 Token 数: {total_tokens}, UNK 数量: {unk_tokens}, UNK 比例: {unk_ratio:.4f}")
    if unk_ratio > threshold:
        print(f"⚠️ 警告: UNK 比例 ({unk_ratio:.2%}) 超过阈值 ({threshold:.0%})，建议进行领域自适应预训练(DAPT)")


# ===================== 2. 训练与评估循环 =====================

def evaluate_epoch(model, loader, id2label, device, use_crf):
    """在 loader 上评估，返回 (avg_loss, entity_f1)。"""
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0
    all_preds, all_golds = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                _, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item()
            labels_np = labels.cpu().tolist()
            
            for i in range(len(input_ids)):
                gold_seq, pred_seq = [], []
                pred_ids = pred_ids_list[i]
                
                for j, gold_id in enumerate(labels_np[i]):
                    if gold_id == -100: continue
                    gold_seq.append(id2label[gold_id])
                    p_id = pred_ids[j] if j < len(pred_ids) else -100
                    pred_seq.append(id2label.get(p_id, "O"))
                    
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, total_epochs, grad_accum, use_amp):
    """【性能优化】支持 AMP 混合精度训练的 Epoch 循环"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        # 开启 autocast 上下文
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss = model(input_ids, attention_mask, token_type_ids, labels)
            loss = loss / grad_accum
            
        # 使用 GradScaler 缩放梯度，防止 FP16 下溢
        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

    # 处理最后不足 grad_accum 的批次
    remainder = len(loader) % grad_accum
    if remainder != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


# ===================== 3. 主函数入口 =====================

def main():
    args = parse_args()
    seed_everything(args.seed)  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.use_amp  # 仅在 GPU 上启用 AMP
    print(f"设备: {device} | AMP: {'ON' if use_amp else 'OFF'}")

    # 标签体系与 Tokenizer
    labels, label2id, id2label = build_label_schema()
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    # DataLoader
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer, label2id=label2id, 
        batch_size=args.batch_size, max_length=args.max_length, data_dir=DATA_DIR
    )
    
    # 训练前检查 UNK 比例
    check_vocab_overlap(train_loader, tokenizer)

    # 模型初始化
    model = build_model(use_crf=args.use_crf, bert_path=str(args.bert_path), 
                        num_labels=len(labels), dropout=args.dropout).to(device)

    # 分层学习率设置
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    if args.use_crf: head_params += list(model.crf.parameters())
    
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # 【性能优化】初始化 GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"训练步数: {total_steps}, 预热步数: {warmup_steps}")

    # 路径准备
    run_tag = "crf" if args.use_crf else "linear"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    log_path = LOG_DIR / f"train_{run_tag}.json"

    best_f1, patience_counter = 0.0, 0
    log_records = []

    print(f"\n开始训练 ({'BERT+CRF' if args.use_crf else 'BERT+Linear'})...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, 
                                     device, epoch, args.epochs, args.grad_accum, use_amp)
                                     
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_entity_f1={val_f1:.4f} | time={elapsed:.0f}s")
        
        log_records.append({"epoch": epoch, "train_loss": round(train_loss, 6), 
                            "val_loss": round(val_loss, 6), "val_entity_f1": round(val_f1, 6), "elapsed_s": round(elapsed, 1)})

        # 早停逻辑判断
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch, "use_crf": args.use_crf, "state_dict": model.state_dict(),
                "val_entity_f1": val_f1, "label2id": label2id, "id2label": id2label, "args": vars(args)
            }, ckpt_path)
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  ✖ 验证集 F1 未提升 ({patience_counter}/{args.patience})")
            
        if patience_counter >= args.patience:
            print(f"\n⏹️ 触发早停机制，在第 {epoch} 轮提前结束训练。")
            break

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}\n  训练日志:   {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="训练 BERT NER 模型 (优化版)")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则使用线性头）")
    parser.add_argument("--use_amp", action="store_true", default=True, help="是否开启混合精度训练(AMP)")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    parser.add_argument("--patience", type=int, default=3, help="早停耐心值(Epochs)")
    
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层基础学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
