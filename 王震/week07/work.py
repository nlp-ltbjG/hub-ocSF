import json
import os
import sys
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# ====================== 全局配置区(统一调参) ======================
class Config:
    DATA_NAME = "validation.json"
    PRETRAIN_MODEL = "bert-base-chinese"
    BATCH_SIZE = 16
    MAX_SEQ_LEN = 512
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    WARMUP_RATIO = 0.1    # 预热步数占总训练步数比例
    TRAIN_SPLIT_RATIO = 0.8
    SAVE_DIR = "./bert_ner_new_model"
    HF_MIRROR = "https://hf-mirror.com"

cfg = Config()
os.environ["HF_ENDPOINT"] = cfg.HF_MIRROR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}")

# ====================== 工具函数：查找数据文件 ======================
def search_json_data(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    abs_path = os.path.join(script_dir, filename)
    if os.path.exists(abs_path):
        return abs_path
    print("目录文件:", os.listdir("."))
    input_path = input("找不到数据集，请输入完整文件路径：").strip().strip("'\"")
    if os.path.exists(input_path):
        return input_path
    sys.exit("数据集文件不存在，退出程序")

# ====================== 1. 数据加载与标签构建 ======================
data_path = search_json_data(cfg.DATA_NAME)
with open(data_path, "r", encoding="utf-8") as f:
    raw_data: List[Dict] = json.load(f)
print(f"加载原始样本数：{len(raw_data)}")

# 构建标签词典
all_ner_tags = set()
for item in raw_data:
    all_ner_tags.update(item["ner_tags"])
tag_list = sorted(all_ner_tags)
tag2id: Dict[str, int] = {t: idx for idx, t in enumerate(tag_list)}
id2tag: Dict[int, str] = {v: k for k, v in tag2id.items()}
label_nums = len(tag2id)
print(f"NER标签总数：{label_nums},标签:{tag_list}")

# ====================== 2. 分词+标签对齐（等价原版子词标签策略） ======================
tokenizer = BertTokenizerFast.from_pretrained(cfg.PRETRAIN_MODEL)

def align_token_label(word_list: List[str], tag_list: List[str]) -> Tuple[List[int], List[int], List[int]]:
    """
    输入原词列表+原标签，输出input_ids, label_ids, word_index
    规则：首个子词=原标签，后续子词=-100，同原版对齐逻辑
    """
    input_ids = [tokenizer.cls_token_id]
    label_ids = [-100]
    word_index = [-1]
    for word, tag in zip(word_list, tag_list):
        sub_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
        sub_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
        tag_id = tag2id[tag]
        sub_label = [tag_id] + [-100]*(len(sub_ids)-1)
        input_ids.extend(sub_ids)
        label_ids.extend(sub_label)
        word_index.extend([len(word_index)-1]*len(sub_ids)) # 记录归属原词下标

    # SEP收尾
    input_ids.append(tokenizer.sep_token_id)
    label_ids.append(-100)
    word_index.append(-1)

    # 超长截断
    if len(input_ids) > cfg.MAX_SEQ_LEN:
        input_ids = input_ids[:cfg.MAX_SEQ_LEN]
        label_ids = label_ids[:cfg.MAX_SEQ_LEN]
        word_index = word_index[:cfg.MAX_SEQ_LEN]
    return input_ids, label_ids, word_index

# 全量预处理
dataset_all = []
for sample in tqdm(raw_data, desc="数据预处理"):
    inp_ids, lab_ids, wid = align_token_label(sample["tokens"], sample["ner_tags"])
    mask = [1]*len(inp_ids)
    dataset_all.append({
        "input_ids": inp_ids,
        "attention_mask": mask,
        "labels": lab_ids,
        "word_ids": wid
    })

# 数据集划分
train_len = int(cfg.TRAIN_SPLIT_RATIO * len(dataset_all))
val_len = len(dataset_all) - train_len
train_set, val_set = random_split(dataset_all, [train_len, val_len])
print(f"训练集:{train_len},验证集:{val_len}")

# ====================== 3. Dataset + 动态padding collate ======================
class NERDataSet(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def dynamic_pad_collate(batch: List[Dict]) -> Dict:
    max_batch_len = min(max(len(x["input_ids"]) for x in batch), cfg.MAX_SEQ_LEN)
    batch_input, batch_mask, batch_label, batch_wordid = [], [], [], []
    for item in batch:
        # 截取+padding
        inp = item["input_ids"][:max_batch_len]
        mask = item["attention_mask"][:max_batch_len]
        lab = item["labels"][:max_batch_len]
        wid = item["word_ids"][:max_batch_len]
        pad_num = max_batch_len - len(inp)

        batch_input.append(torch.LongTensor(inp + [0]*pad_num))
        batch_mask.append(torch.LongTensor(mask + [0]*pad_num))
        batch_label.append(torch.LongTensor(lab + [-100]*pad_num))
        batch_wordid.append(wid + [-1]*pad_num)

    return {
        "input_ids": torch.stack(batch_input).to(device),
        "attention_mask": torch.stack(batch_mask).to(device),
        "labels": torch.stack(batch_label).to(device),
        "word_ids": batch_wordid
    }

# 构建DataLoader
train_loader = DataLoader(NERDataSet(train_set), batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=dynamic_pad_collate)
val_loader = DataLoader(NERDataSet(val_set), batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=dynamic_pad_collate)

# ====================== 4. 模型、优化器、学习率调度 ======================
model = BertForTokenClassification.from_pretrained(cfg.PRETRAIN_MODEL, num_labels=label_nums)
model.to(device)

optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
total_train_steps = len(train_loader)*cfg.EPOCHS
warmup_steps = int(total_train_steps * cfg.WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_train_steps
)

# ====================== 5. 评测函数（逻辑和原版完全等价：按原词聚合预测） ======================
def calc_metrics(true: List[int], pred: List[int], ignore=-100):
    # 过滤padding与无效标签
    filter_true = [t for t,p in zip(true,pred) if t != ignore]
    filter_pred = [p for t,p in zip(true,pred) if t != ignore]
    if len(filter_true)==0:
        return 0.0, {}, 0.0

    # 整体准确率
    acc = sum(1 for a,b in zip(filter_true,filter_pred) if a==b)/len(filter_true)

    # 单类别P/R/F1
    cls_result = {}
    total_tp, total_fp, total_fn = 0,0,0
    for lid in range(label_nums):
        tp = sum(1 for t,p in zip(filter_true,filter_pred) if t==lid and p==lid)
        fp = sum(1 for t,p in zip(filter_true,filter_pred) if t!=lid and p==lid)
        fn = sum(1 for t,p in zip(filter_true,filter_pred) if t==lid and p!=lid)
        pre = tp/(tp+fp+1e-8)
        rec = tp/(tp+fn+1e-8)
        f1 = 2*pre*rec/(pre+rec+1e-8)
        cls_result[id2tag[lid]] = {"precision":pre,"recall":rec,"f1":f1}
        total_tp += tp
        total_fp += fp
        total_fn += fn
    # micro F1
    mic_pre = total_tp/(total_tp+total_fp+1e-8)
    mic_rec = total_tp/(total_tp+total_fn+1e-8)
    mic_f1 = 2*mic_pre*mic_rec/(mic_pre+mic_rec+1e-8)
    return acc, cls_result, mic_f1

def val_model(model, loader):
    model.eval()
    all_truth, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="验证评估"):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            pred_token = torch.argmax(out.logits, dim=-1).cpu().numpy()
            label_token = batch["labels"].cpu().numpy()
            wordid_batch = batch["word_ids"]

            # 按原词聚合预测，和原代码规则一致：同word_id只取第一个有效subtoken
            for b_idx in range(len(pred_token)):
                pre_arr = pred_token[b_idx]
                lab_arr = label_token[b_idx]
                wid_arr = wordid_batch[b_idx]
                last_wid = -2
                for pos,wid in enumerate(wid_arr):
                    if wid == -1:
                        continue
                    if wid != last_wid:
                        if lab_arr[pos] != -100:
                            all_truth.append(lab_arr[pos])
                            all_preds.append(pre_arr[pos])
                        last_wid = wid
    return calc_metrics(all_truth, all_preds)

# ====================== 6. 训练循环 ======================
def train_one_epoch(model, loader, opt, sch):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(loader, desc="本轮训练"):
        opt.zero_grad()
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = out.loss
        loss.backward()
        opt.step()
        sch.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

if __name__ == "__main__":
    print("\n===== 开始NER训练 =====")
    for ep in range(1, cfg.EPOCHS+1):
        print(f"\n【Epoch {ep}/{cfg.EPOCHS}】")
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        print(f"训练平均损失: {avg_loss:.4f}")

        val_acc, cls_metric, micro_f1 = val_model(model, val_loader)
        print(f"验证Acc:{val_acc:.4f} | Micro-F1:{micro_f1:.4f}")
        for tag, m in cls_metric.items():
            print(f"{tag:10s} | P:{m['precision']:.4f} R:{m['recall']:.4f} F1:{m['f1']:.4f}")

    # 保存模型
    model.save_pretrained(cfg.SAVE_DIR)
    tokenizer.save_pretrained(cfg.SAVE_DIR)
    print(f"\n训练结束，模型&分词器保存至: {cfg.SAVE_DIR}")
