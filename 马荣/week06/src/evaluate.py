import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

def evaluate_model(
    model,
    loader,
    device: torch.device,
    id2name: dict,
    print_report: bool = True,
) -> dict:
    """
    在给定 DataLoader 上评估模型，返回指标字典。
    可在 train.py 的每个 epoch 末调用。
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"]

            logits = model(input_ids, attention_mask, token_type_ids)
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 过滤 test 集的 -1 标签（test 集无标签）
    valid_mask = all_labels != -1
    all_preds  = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if print_report:
        label_ids   = sorted(id2name.keys())
        target_names = [id2name[i] for i in label_ids]
        print("\n分类报告：")
        print(classification_report(
            all_labels, all_preds,
            labels=label_ids,
            target_names=target_names,
            zero_division=0,
        ))

    return {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "preds":     all_preds,
        "labels":    all_labels,
    }