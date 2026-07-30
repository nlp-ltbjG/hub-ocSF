"""
LLM SFT（监督微调）训练脚本 — 基于 LoRA 高效微调 Qwen2-0.5B-Instruct 做 NER
  1. NER 的指令微调格式：输入是文本，输出是 JSON 实体列表
  2. Loss masking：同样只在 JSON 输出部分计算 loss，prompt 全为 -100
  3. LoRA 高效微调：参数量约 0.22%，与全量微调的对比（--full_ft 开关）
  4. 生成式 NER vs 序列标注（BERT+CRF）：各自的优劣和适用场景

使用方式：
  python train_sft.py                        # LoRA，全量训练数据（默认）
  python train_sft.py --num_train 2000       # LoRA，2000 条快速演示
  python train_sft.py --epochs 1             # 快速验证流程

  # 全量微调（需显存 ≥ 16GB）
  python train_sft.py --full_ft --lr 2e-5

依赖：
  pip install torch transformers peft tqdm   # LoRA 模式
  pip install torch transformers tqdm        # 全量微调模式（不需要 peft）
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathlib import Path
import argparse
import json
import random
from tqdm import tqdm
import time

# 路径
DATA_DIR = Path(__file__).parent.parent / 'data'
MODEL_PATH = Path(__file__).parent.parent.parent / 'pretrain_models' / "Qwen2-0.5B-Instruct"
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'

SYSTEM_PROMPT = """
你是一个专业的命名实体识别（NER）助手。

【任务要求】
- 识别输入句子中的三类命名实体：人名（PER）、组织名（ORG）、地名（LOC）
- 输出格式：严格输出以下 JSON 结构，不要输出任何额外内容。

输出格式示例：
{"entities": [{"text": "李华", "type": "PER"}, {"text": "北京大学", "type": "ORG"}]}

【注意事项】
- 只输出 JSON，不要输出其他解释或标记
- 如果句子中没有实体，输出 {"entities": []}
- 确保 JSON 格式正确（双引号、逗号）
"""

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class SFTDataset(Dataset):
    """json -> tokenizer.encode ->input_ids, labels"""
    def __init__(self, data, tokenizer, max_length=256):
        print(f"Dataset 初始化，样本数: {len(data)}")
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        target = bio_to_generated(item["tokens"], item["ner_tags"])
        
        # ── Step 1：构建 prompt 文本（tokenize=False 兼容 transformers 5.x）──
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "".join(item["tokens"])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        # ── Step 2：response = JSON 字符串 + EOS ──────────────────────────────
        response_ids = (
            self.tokenizer.encode(target, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )
        
        # ── Step 3：拼接，截断 ──────────────────────────────
        input_ids = (prompt_ids + response_ids)[:self.max_length]
        
        # ── Step 4：loss mask：prompt 全 -100，只在 JSON 部分计算 loss ──────
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }
       

def bio_to_generated(tokens, ner_tags):
    """
    将 BIO 标签转换为 JSON 格式：{"entities": [{"text": "...", "type": "PER"}, ...]}
    """
    entities = []
    cur_entity_text = []
    cur_entity_type = None

    for token, tag in zip(tokens, ner_tags):
        if tag.startswith('B-'):
            # 保存上一个实体（如果有）
            if cur_entity_text:
                entities.append({
                    "text": ''.join(cur_entity_text),
                    "type": cur_entity_type
                })
            # 开始新实体
            cur_entity_text = [token]
            cur_entity_type = tag[2:]          # 去掉 "B-"，得到 PER/ORG/LOC
        elif tag.startswith('I-') and cur_entity_type is not None:
            # 继续当前实体
            cur_entity_text.append(token)
        else:
            # 非实体：结束当前实体（如果有）
            if cur_entity_text:
                entities.append({
                    "text": ''.join(cur_entity_text),
                    "type": cur_entity_type
                })
                cur_entity_text = []
                cur_entity_type = None
            # O 标签不做任何处理
    # 循环结束后，如果还有未保存的实体
    if cur_entity_text:
        entities.append({
            "text": ''.join(cur_entity_text),
            "type": cur_entity_type
        })
    # 构造最终 JSON
    return json.dumps({"entities": entities}, ensure_ascii=False)
    
# ── 加载数据 ──────────────────────────────────────────────────────────────  
def get_data():
    with open(DATA_DIR / 'train.json') as f:
        train_data = json.load(f)
    with open(DATA_DIR / 'validation.json') as f:
        val_data = json.load(f)
    with open(DATA_DIR / 'test.json') as f:
        test_data = json.load(f)
    return train_data, val_data, test_data


def train(train_loader, val_loader, tokenizer, ckpt_dir, ckpt_label, args, device):
    print(f"train_loader 长度: {len(train_loader)}")
    mode_str = "全量微调" if args.full_ft else "LoRA 微调"
    print(f"微调模式: {mode_str}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
    model = model.to(device)
    
    
     # ── LoRA 或全量微调 ────────────────────────────────────────────────────────
    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.0000")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft 库：pip install peft>=0.14.0")
        # lora微调
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        # 查看可训练参数量（通常会小于原模型参数的1%）
        model.print_trainable_parameters()
    
    # ── 优化器 ────────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps}（batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs}, lr={args.lr}）\n")
    
    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_records   = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
         
        t0 = time.time()
        pbar = tqdm(train_loader)
        print("开始进入 DataLoader 循环...")
        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss  # loss：当前 batch 的平均损失（标量张量）
            
            (loss / args.grad_accum).backward()
            
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            n_tokens      = (labels != -100).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # 处理最后不足 grad_accum 的批次
        remainder = len(train_loader) % args.grad_accum
        if remainder != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        avg_train_loss = total_loss / max(total_tokens, 1)
        # ── 验证 loss ─────────────────────────────────────────────────────────
        avg_val_loss = eval(model, val_loader, args, device)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | {elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "elapsed_s": elapsed,
        })
        # 用验证集指导模型选择，避免保存过拟合的版本
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  ✓ 最优{ckpt_label}已保存 → {ckpt_dir}  (val_loss={avg_val_loss:.4f})")
                
    return log_records, best_val_loss
        
    
def eval(model, val_loader, args, device):
    """验证"""
    model.eval()
    val_loss, val_tokens = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
            n_tokens   = (labels != -100).sum().item()
            val_loss   += outputs.loss.item() * n_tokens
            val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)
    return avg_val_loss 
            
def collate_fn(batch, pad_id):
    """不一样长度,补齐或者截断到同一长度"""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n   = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([item["input_ids"],
                                         torch.full((pad,), pad_id, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"],
                                     torch.full((pad,), -100, dtype=torch.long)]))
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long),
                                    torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }
        
    
def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 训练（LoRA / 全量微调）")
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--num_train",   default=-1,   type=int,
                        help="训练样本数，-1 使用全部 10748 条（默认）")
    parser.add_argument("--epochs",      default=3,    type=int)
    parser.add_argument("--batch_size",  default=2,    type=int)
    parser.add_argument("--grad_accum",  default=8,    type=int)
    parser.add_argument("--lr",          default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）")
    parser.add_argument("--max_length",  default=192,  type=int,
                        help="序列最大长度；NER 的 JSON 输出比分类长，建议 256")
    parser.add_argument("--full_ft",     action="store_true",
                        help="全量微调：跳过 LoRA，更新所有 495M 参数（需显存 ≥ 16GB）")
    parser.add_argument("--lora_r",      default=2,    type=int)
    parser.add_argument("--lora_alpha",  default=4,   type=int)
    parser.add_argument("--seed",        default=42,   type=int)
    return parser.parse_args()


def main():
    torch.set_num_threads(os.cpu_count())
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
    
    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    
    # ── 构建数据集 ─────────────────────────────────────────────────────────────
    train_data, val_data, _ = get_data()
    if args.num_train > 0:
        train_data = train_data[:args.num_train]
    
    # ── 加载 Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id) 
    
    train_loader = DataLoader(SFTDataset(train_data, tokenizer, args.max_length), 
                              batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(SFTDataset(val_data, tokenizer, args.max_length), 
                            batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    log_records, best_val_loss = train(train_loader, val_loader, tokenizer, ckpt_dir, ckpt_label, args, device)
    
    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_tag  = "full_ft" if args.full_ft else "sft"
    log_path = output_dir / "logs" / f"train_{log_tag}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"{ckpt_label} → {ckpt_dir}")
    print(f"\n下一步：python evaluate_sft.py 查看 entity F1 与多方对比")


main()
