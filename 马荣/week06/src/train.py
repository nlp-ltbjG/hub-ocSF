import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from transformers import BertTokenizer,get_linear_schedule_with_warmup
from dataset import build_dataloaders
import argparse
from pathlib import Path
from model import build_model
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import time
from evaluate import evaluate_model


ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
BERT_PATH     = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR    = ROOT / "outputs"
CKPT_DIR      = OUTPUT_DIR / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 文本分类训练")
    parser.add_argument("--bert_path",      default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir",       default=str(DATA_DIR),  type=str)
    parser.add_argument("--output_dir",     default=str(OUTPUT_DIR), type=str)
    parser.add_argument("--pool",           default="cls",
                        choices=["cls", "mean", "max"],
                        help="向量提取策略：cls / mean / max")
    parser.add_argument("--epochs",         default=3,   type=int)
    parser.add_argument("--batch_size",     default=32,  type=int)
    parser.add_argument("--max_length",     default=64, type=int)
    parser.add_argument("--lr",             default=2e-5, type=float,
                        help="BERT 层学习率")
    parser.add_argument("--head_lr_mult",   default=5.0,  type=float,
                        help="分类头学习率倍数（head_lr = lr * head_lr_mult）")
    parser.add_argument("--dropout",        default=0.1,  type=float)
    parser.add_argument("--warmup_ratio",   default=0.1,  type=float,
                        help="warmup 步数占总步数的比例")
    parser.add_argument("--grad_accum",     default=1,    type=int,
                        help="梯度累积步数，显存不足时设为 2/4")
    parser.add_argument("--use_class_weight", action="store_true",
                        help="使用加权 CrossEntropyLoss 处理类别不均衡")
    return parser.parse_args()

def class_weight(data_dir,num_labels,device):
    with open(data_dir/"train.json",encoding="utf-8") as f:
        train_data=json.load(f)
    labels=np.array([item["label"] for item in train_data])
    classes=np.arange(num_labels)
    weights=compute_class_weight("balanced",classes=classes,y=labels)
    return torch.tensor(weights,dtype=float).to(device)

def train(data_dir,loader,total_epochs,grad_accum):
    args=parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    log_records  = []

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备{device}")

    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = label_map["num_labels"]
    id2name    = {int(k): v for k, v in label_map["id2name"].items()}
    print(f"类别数: {num_labels}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_loader, val_loader, _ = build_dataloaders(
        data_dir, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    model = build_model(args.bert_path, num_labels, pool=args.pool)
    model = model.to(device)

    if args.use_class_weight:
        weights = class_weight(data_dir, num_labels, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("使用加权 CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用普通 CrossEntropyLoss")


    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    for epoch in range(1,args.epochs+1):
        t0=time.time
        pbar=tqdm(loader,desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
        for step,batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)  # [B, C]
            loss   = criterion(logits, labels)

            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #梯度裁剪
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            preds=logits.argmax(dim=-1)
            total_loss+=loss.item()*labels.size(0)
            total_correct+=(preds==labels).sum().item()
            total_samples+=labels.size(0)
            pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}",
                         acc=f"{total_correct/total_samples:.4f}")
            
        avg_loss=total_loss/total_samples
        avg_acc=total_correct/total_samples
            
        val_metrics = evaluate_model(model, val_loader, device, id2name,
                                     print_report=(epoch == args.epochs))
        elapsed = time.time() - t0
        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["macro_f1"]

        log_records.append({
            "epoch": epoch, "train_loss": avg_loss, "train_acc": avg_acc,
            "val_acc": val_acc, "val_macro_f1": val_f1, "elapsed_s": elapsed,
        })

        # 只保存验证集最优的 checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            run_tag  = f"{args.pool}_weighted" if args.use_class_weight else args.pool
            ckpt_path = ckpt_dir / f"best_{run_tag}.pt"
            torch.save({
                "epoch":           epoch,
                "pool":            args.pool,
                "use_class_weight": args.use_class_weight,
                "state_dict":      model.state_dict(),
                "val_acc":         val_acc,
                "val_macro_f1":    val_f1,
                "args":            vars(args),
            }, ckpt_path)
            print(f"新最优模型已保存 → {ckpt_path}  (val_acc={val_acc:.4f})")

    run_tag  = f"{args.pool}_weighted" if args.use_class_weight else args.pool
    log_path = output_dir / f"train_log_{run_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。最优 val_acc={best_val_acc:.4f}")
    print(f"训练日志 → {log_path}")