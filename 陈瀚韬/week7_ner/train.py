import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
from pathlib import Path
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from seqeval.metrics import f1_score, classification_report
from tqdm import tqdm

from data_helper import load_data, get_dataloader
from model import BertNerModel

BERT_PATH = r"E:\pretrain_models\bert-base-chinese"
DATA_DIR = Path(__file__).parent.parent.parent / "week7序列标注问题" / "week7" / "序列标注项目" / "data" / "peoples_daily"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

USE_CRF = True
EPOCHS = 3
BATCH_SIZE = 32
MAX_LEN = 150
LR = 3e-5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    with open(DATA_DIR / "label_names.json", "r", encoding="utf-8") as f:
        label_list = json.load(f)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label_list)
    print(f"labels: {label_list}, total: {num_labels}")

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    train_data = load_data(DATA_DIR / "train.json")
    val_data = load_data(DATA_DIR / "validation.json")
    test_data = load_data(DATA_DIR / "test.json")

    train_loader = get_dataloader(train_data, tokenizer, label2id, BATCH_SIZE, shuffle=True, max_len=MAX_LEN)
    val_loader = get_dataloader(val_data, tokenizer, label2id, BATCH_SIZE, shuffle=False, max_len=MAX_LEN)
    test_loader = get_dataloader(test_data, tokenizer, label2id, BATCH_SIZE, shuffle=False, max_len=MAX_LEN)

    print(f"train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    model = BertNerModel(BERT_PATH, num_labels, use_crf=USE_CRF).to(device)

    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": LR},
        {"params": model.fc.parameters(), "lr": LR * 5},
        {"params": model.crf.parameters(), "lr": LR * 5} if USE_CRF else {"params": [], "lr": LR}
    ], weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            token_type = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(input_ids, attn_mask, token_type, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        val_f1 = evaluate(model, val_loader, id2label, device)
        print(f"Epoch {epoch} | loss: {avg_loss:.4f} | val_f1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"  -> saved best model, f1={val_f1:.4f}")

    print(f"\nTraining done. Best val f1: {best_f1:.4f}")
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", map_location=device))
    test_f1 = evaluate(model, test_loader, id2label, device, verbose=True)
    print(f"Test F1: {test_f1:.4f}")


def evaluate(model, loader, id2label, device, verbose=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            token_type = batch["token_type_ids"].to(device)
            labels = batch["labels"]

            preds = model.predict(input_ids, attn_mask, token_type)

            labels_list = labels.tolist()
            for i in range(len(input_ids)):
                gold = []
                pred = []
                for j, lb in enumerate(labels_list[i]):
                    if lb == -100:
                        continue
                    gold.append(id2label[lb])
                    if USE_CRF:
                        p_id = preds[i][j] if j < len(preds[i]) else 0
                    else:
                        p_id = preds[i][j]
                    pred.append(id2label.get(p_id, "O"))
                all_labels.append(gold)
                all_preds.append(pred)

    f1 = f1_score(all_labels, all_preds)
    if verbose:
        print(classification_report(all_labels, all_preds))
    return f1


if __name__ == "__main__":
    main()
