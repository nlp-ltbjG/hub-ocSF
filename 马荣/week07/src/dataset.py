from torch.utils.data import Dataset,DataLoader
import json
from pathlib import Path
from transformers import BertTokenizer
import torch

DATA_DIR=Path(__file__).parent.parent/"data"/"peoples_daily"

def get_label2id(data_dir):
    with open(data_dir/"label_names.json","r",encoding="utf-8") as f:
        label_list=json.load(f)
    return {label:i for i,label in enumerate(label_list)}
            
def load_data(split,data_dir):
    with open(data_dir/f"{split}.json","r",encoding="utf-8") as f:
        data_list=json.load(f)
    return data_list

class PeoplesDailyDataset(Dataset):
    def __init__(self,data_list,tokenizer:BertTokenizer,max_length):
        self.data_list=data_list
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,index):
        data=self.data_list[index]
        text_list=data["tokens"]
        label_id_list=[]
        label_list=data["ner_tags"]
        for i in label_list:
            label_id_list.append(label_list[i])

        encoding = self.tokenizer(
            text_list,
            is_split_into_words=True,#text_list 已经被你提前切好了，每个元素当成一个 word。后续使用word_ids()
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids=encoding.word_ids()
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首次出现这个字符索引：使用 BIO 标签
                if wid < len(label_id_list):
                    aligned_labels.append(label_id_list[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一字符的后续子词（中文通常不会出现，但保留正确处理）
                aligned_labels.append(-100)
        
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }

def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_data("train", data_dir)
    val_records = load_data("validation", data_dir)
    test_records = load_data("test", data_dir)

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

if __name__=="__main__":
    print(get_label2id(DATA_DIR))