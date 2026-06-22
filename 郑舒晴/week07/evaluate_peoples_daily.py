"""
在 peoples_daily 数据集上评估 BERT NER 模型
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertTokenizer
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2

from dataset import build_peoples_daily_dataloaders, get_label_mapping
from model import build_model

ROOT = Path(__file__).parent.parent
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints"


def evaluate_model(model, dataloader, id2label, device, use_crf=False):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 将数据移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # 预测
            if use_crf:
                preds = model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            else:
                preds = model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                preds = preds.cpu().numpy()
            
            # 处理预测结果和标签
            for i in range(len(labels)):
                # 获取有效长度（忽略 padding 和 -100）
                seq_len = (labels[i] != -100).sum().item()
                
                # 获取预测和真实标签
                pred = preds[i][:seq_len].tolist()
                label = labels[i][:seq_len].cpu().numpy().tolist()
                
                # 转换为标签名称
                pred_labels = [id2label[p] for p in pred]
                true_labels = [id2label[l] for l in label]
                
                all_preds.append(pred_labels)
                all_labels.append(true_labels)
    
    # 计算分类报告
    report = classification_report(
        all_labels, 
        all_preds, 
        scheme=IOB2, 
        mode='strict',
        output_dict=True
    )
    
    # 计算 F1 分数
    f1 = f1_score(all_labels, all_preds, scheme=IOB2)
    
    return report, f1


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    
    # 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))
    
    # 获取标签映射
    label2id, id2label = get_label_mapping(data_dir=DATA_DIR)
    num_labels = len(label2id)
    print(f"标签数：{num_labels}")
    
    # 加载数据
    _, val_loader, _ = build_peoples_daily_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=DATA_DIR,
    )
    
    # 加载模型
    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=num_labels,
    ).to(device)
    
    # 加载检查点
    run_tag = "peoples_daily_crf" if args.use_crf else "peoples_daily_linear"
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"已加载检查点: {ckpt_path}")
    
    # 评估
    report, f1 = evaluate_model(model, val_loader, id2label, device, args.use_crf)
    
    # 打印整体 F1 分数
    print(f"\n整体 F1 分数: {f1:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(
        report, 
        output_dict=False, 
        scheme=IOB2, 
        mode='strict'
    ))


def parse_args():
    parser = argparse.ArgumentParser(description="评估 peoples_daily 数据集上的 BERT NER 模型")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    main()
