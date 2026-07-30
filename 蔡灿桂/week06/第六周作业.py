
import argparse
import json
import time
import random
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model
from evaluate import evaluate_model

# ─────────────────── 默认路径（相对于 src/ 目录）────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
BERT_PATH     = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR    = ROOT / "outputs"
CKPT_DIR      = OUTPUT_DIR / "checkpoints"


# ─────────────────── 全局随机种子固定（保障可复现性）─────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # PyTorch 2.0+ 可选：提升 float32 矩阵乘法精度与稳定性
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    print(f"🌱 随机种子已固定: seed={seed}")


# ─────────────────── 类别权重计算 ─────────────────────────────────────────────
def compute_loss_weights(data_dir: Path, num_labels: int, device: torch.device):
    """根据训练集类别频次计算 inverse-frequency 权重。"""
    train_json = data_dir / "train.json"
    if not train_json.exists():
        raise FileNotFoundError(f"未找到训练集文件: {train_json}")
        
    with open(train_json, encoding="utf-8") as f:
        train_data = json.load(f)
        
    labels = np.array([item["label"] for item in train_data])
    classes = np.arange(num_labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    
    print("📊 类别权重（用于加权 Loss）：")
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}
        
    for i, w in enumerate(weights):
        print(f"  {i:2d} {id2name[i]:<6s}: {w:.3f}")
        
    return torch.tensor(weights, dtype=torch.float).to(device)


# ─────────────────── 单轮训练（集成 AMP + 梯度累积）───────────────────────────
def train_one_epoch(
    model, loader, optimizer, scheduler, criterion,
    device, epoch, total_epochs, grad_accum, scaler, use_amp
):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        # 1. AMP 自动混合精度前向传播
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask, token_type_ids)  # [B, C]
            loss   = criterion(logits, labels)

        # 2. 梯度累积 + 缩放反向传播
        scaler.scale(loss / grad_accum).backward()

        # 3. 达到累积步数后执行参数更新
        if (step + 1) % grad_accum == 0:
            # ⚠️ 必须先 unscale_ 再裁剪梯度，否则裁剪无效
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # 记录指标（loss.item() 始终返回 fp32 原始值，不受 scale 影响）
        preds = logits.argmax(dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}",
                         acc=f"{total_correct/total_samples:.4f}")

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


# ─────────────────── 主流程 ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BERT 文本分类训练（增强版）")
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
    parser.add_argument("--seed",           default=42,   type=int,
                        help="固定随机种子")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    print(f"使用设备: {device} | AMP 混合精度: {'✅ 启用' if use_amp else '❌ 未启用'}")

    # ── 加载 label_map ───────────────────────────────────────────────────────
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = label_map["num_labels"]
    id2name    = {int(k): v for k, v in label_map["id2name"].items()}
    print(f"类别数: {num_labels}")

    # ── Tokenizer & DataLoader ───────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_loader, val_loader, _ = build_dataloaders(
        data_dir, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # ── 模型 ────────────────────────────────────────────────────────────────
    model = build_model(args.bert_path, num_labels, pool=args.pool)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # ── Loss ────────────────────────────────────────────────────────────────
    if args.use_class_weight:
        weights = compute_loss_weights(data_dir, num_labels, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("使用加权 CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用普通 CrossEntropyLoss")

    # ── 优化器：BERT 层和分类头用不同学习率 ─────────────────────────────────
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    # 注意：total_steps 需除以 grad_accum，因为 optimizer 实际更新次数减少了
    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"总训练步数: {total_steps} | Warmup: {warmup_steps}")

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    log_records = []

    print("\n" + "="*80)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum, scaler, use_amp
        )
        
        val_metrics = evaluate_model(model, val_loader, device, id2name,
                                     print_report=(epoch == args.epochs))
        elapsed = time.time() - t0

        val_acc = val_metrics.get("accuracy", 0.0)
        val_f1  = val_metrics.get("macro_f1", 0.0)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f} | "
              f"{elapsed:.1f}s")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_macro_f1": val_f1, "elapsed_s": elapsed,
        })

        #  核心优化：基于 Macro F1 保存最优模型（更适合不均衡数据）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
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
            print(f"  ✅ 新最优模型已保存 (F1↑) → {ckpt_path}")

    # ── 保存训练日志 ─────────────────────────────────────────────────────────
    run_tag  = f"{args.pool}_weighted" if args.use_class_weight else args.pool
    log_path = output_dir / f"train_log_{run_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*80)
    print(f"训练完成。最佳验证 Macro F1: {best_val_f1:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"最优权重 → {ckpt_dir / f'best_{run_tag}.pt'}")


if __name__ == "__main__":
    main()
