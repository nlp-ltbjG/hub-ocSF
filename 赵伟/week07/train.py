import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from tqdm import tqdm
from torch.optim import AdamW




# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# -------------------- 1. 加载数据 --------------------
def find_data_file(filename='validation.json'):
    if os.path.exists(filename):
        return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, filename)
    if os.path.exists(candidate):
        return candidate
    print(f"当前目录内容: {os.listdir('.')}")
    print(f"未找到 {filename}，请手动输入完整路径：")
    user_path = input("文件路径: ").strip().strip('"').strip("'")
    if os.path.exists(user_path):
        return user_path
    return None

data_file = find_data_file('validation.json')
if data_file is None:
    print("错误：无法找到 validation.json 文件，程序终止。")
    sys.exit(1)

print(f"找到数据文件: {data_file}")
with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"成功加载 {len(data)} 条数据")

# -------------------- 2. 提取标签 --------------------
all_tags = set()
for sample in data:
    for tag in sample['ner_tags']:
        all_tags.add(tag)
tag_list = sorted(all_tags)
print("标签列表:", tag_list)

tag2id = {tag: idx for idx, tag in enumerate(tag_list)}
id2tag = {idx: tag for tag, idx in tag2id.items()}
num_labels = len(tag2id)
print(f"标签数量: {num_labels}")

# -------------------- 3. 初始化 tokenizer --------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
MAX_LEN = 512  # BERT 最大位置编码长度

# -------------------- 4. 数据预处理：子词对齐、截断、记录 word_ids --------------------
def tokenize_and_align_labels(tokens, tags, max_len=MAX_LEN):
    input_ids = []
    labels = []
    word_ids = []
    current_token_idx = 0

    for token, tag in zip(tokens, tags):
        sub_tokens = tokenizer.tokenize(token)
        if not sub_tokens:
            sub_tokens = [tokenizer.unk_token]
        sub_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
        label_id = tag2id[tag]
        sub_labels = [label_id] + [-100] * (len(sub_ids) - 1)
        input_ids.extend(sub_ids)
        labels.extend(sub_labels)
        word_ids.extend([current_token_idx] * len(sub_ids))
        current_token_idx += 1

    # 添加 [CLS] 和 [SEP]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    labels = [-100] + labels + [-100]
    word_ids = [-1] + word_ids + [-1]
    attention_mask = [1] * len(input_ids)

    # ----- 截断至最大长度 -----
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]
        word_ids = word_ids[:max_len]
        # 注意：截断后最后一个位置可能是原始token中间，但不影响训练
    return input_ids, attention_mask, labels, word_ids

# 预处理所有数据
processed_data = []
for sample in tqdm(data, desc="预处理数据"):
    tokens = sample['tokens']
    tags = sample['ner_tags']
    input_ids, attention_mask, labels, word_ids = tokenize_and_align_labels(tokens, tags)
    processed_data.append({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'word_ids': word_ids
    })

# -------------------- 5. 划分训练集和验证集 --------------------
train_size = int(0.8 * len(processed_data))
val_size = len(processed_data) - train_size
train_data_raw, val_data_raw = torch.utils.data.random_split(processed_data, [train_size, val_size])
print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

# -------------------- 6. Dataset 和 collate_fn --------------------
class NERDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long),
            'word_ids': item['word_ids']
        }

def collate_fn(batch):
    max_len = max([len(item['input_ids']) for item in batch])
    # 如果 max_len 超过 MAX_LEN，理论上不应该发生，但保险起见
    if max_len > MAX_LEN:
        max_len = MAX_LEN
    input_ids = []
    attention_mask = []
    labels = []
    word_ids_batch = []
    for item in batch:
        # 截取至 max_len（如果原始长度超过）
        input_id = item['input_ids'][:max_len]
        attn_mask = item['attention_mask'][:max_len]
        label = item['labels'][:max_len]
        word_id = item['word_ids'][:max_len]
        pad_len = max_len - len(input_id)
        input_ids.append(torch.cat([input_id, torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([label, torch.full((pad_len,), -100, dtype=torch.long)]))
        word_ids_batch.append(word_id + [-1] * pad_len)
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
        'word_ids': word_ids_batch
    }

batch_size = 16
train_loader = DataLoader(NERDataset(train_data_raw), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(NERDataset(val_data_raw), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# -------------------- 7. 加载模型 --------------------
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
model.to(device)

# -------------------- 8. 优化器和调度器 --------------------
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 5
total_steps = len(train_loader) * epochs
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

# -------------------- 9. 评估函数 --------------------
def compute_metrics(true_labels, pred_labels, ignore_index=-100):
    filtered_true = [t for t, p in zip(true_labels, pred_labels) if t != ignore_index]
    filtered_pred = [p for t, p in zip(true_labels, pred_labels) if t != ignore_index]
    total = len(filtered_true)
    if total == 0:
        return 0.0, {}, 0.0

    accuracy = sum(1 for t, p in zip(filtered_true, filtered_pred) if t == p) / total

    class_stats = {}
    for label_id in range(num_labels):
        tp = sum(1 for t, p in zip(filtered_true, filtered_pred) if t == label_id and p == label_id)
        fp = sum(1 for t, p in zip(filtered_true, filtered_pred) if t != label_id and p == label_id)
        fn = sum(1 for t, p in zip(filtered_true, filtered_pred) if t == label_id and p != label_id)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        class_stats[id2tag[label_id]] = {'precision': precision, 'recall': recall, 'f1': f1}

    tp_total = fp_total = fn_total = 0
    for label_id in range(num_labels):
        tp = sum(1 for t, p in zip(filtered_true, filtered_pred) if t == label_id and p == label_id)
        fp = sum(1 for t, p in zip(filtered_true, filtered_pred) if t != label_id and p == label_id)
        fn = sum(1 for t, p in zip(filtered_true, filtered_pred) if t == label_id and p != label_id)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    micro_precision = tp_total / (tp_total + fp_total + 1e-8)
    micro_recall = tp_total / (tp_total + fn_total + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)

    return accuracy, class_stats, micro_f1

def evaluate(model, data_loader):
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            word_ids_list = batch['word_ids']
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            for i in range(len(input_ids)):
                word_ids = word_ids_list[i]
                pred_ids = predictions[i].cpu().numpy()
                label_ids = labels[i].cpu().numpy()
                prev_word_id = -1
                for j, w_id in enumerate(word_ids):
                    if w_id == -1:
                        continue
                    if w_id != prev_word_id:
                        if label_ids[j] != -100:
                            all_true.append(label_ids[j])
                            all_pred.append(pred_ids[j])
                        prev_word_id = w_id
    return compute_metrics(all_true, all_pred)

# -------------------- 10. 训练函数 --------------------
def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="训练"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(data_loader)

# -------------------- 11. 开始训练 --------------------
print("开始训练...")
for epoch in range(epochs):
    print(f"\n========== Epoch {epoch+1}/{epochs} ==========")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    print(f"训练损失: {train_loss:.4f}")

    val_acc, val_class_metrics, val_micro_f1 = evaluate(model, val_loader)
    print(f"验证集整体准确率: {val_acc:.4f}")
    print(f"验证集微平均F1: {val_micro_f1:.4f}")
    print("各类别指标:")
    for tag, metrics in val_class_metrics.items():
        print(f"  {tag:8s} - P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

# -------------------- 12. 保存模型 --------------------
model_save_path = "./bert_ner_model_adamw"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\n模型已保存至 {model_save_path}")
