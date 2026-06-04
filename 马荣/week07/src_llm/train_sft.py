import os
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

ROOT=Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "peoples_daily_entities"

ENTITY_TYPES = [
    "person", "organization","location" 
]

ENTITY_TYPE_MAP = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
    "person": "person",
    "organization": "organization",
    "location": "location",
}

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：person（人物）, organization（组织）,location（地点） \n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)

def data_to_target(data):
    entities = []
    for entity in data.get("entities", []):
        entity_text = entity.get("text", "")
        entity_type = ENTITY_TYPE_MAP.get(entity.get("type"), entity.get("type", ""))
        if not entity_text or not entity_type:
            continue
        entities.append({
            "text": entity_text,
            "type": entity_type,
        })

    return json.dumps({"entities": entities}, ensure_ascii=False)

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        super().__init__()
        self.data=data
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        item=self.data[index]
        target=data_to_target(item)
        prompt_text=self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": item["text"]},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids=self.tokenizer.encode(prompt_text,add_special_tokens=False)

        response_ids=(
            self.tokenizer.encode(target,add_special_tokens=False)
            +[self.tokenizer.eos_token_id]
        )

        input_ids=(prompt_ids+response_ids)[:self.max_length]
        labels=([-100]*len(prompt_ids)+response_ids)[:self.max_length]

        return  {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }

def main():

    #准备数据
    with open(DATA_DIR/"train.json","r",encoding="utf-8") as f:
        train_data=json.load(f)

    with open(DATA_DIR/"validation.json","r",encoding="utf-8") as f:
        validation_data=json.load(f)

    # ── 加载 Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(model_path).resolve()), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_loader=SFTDataset(train_data,tokenizer)
    validation_loader=SFTDataset(validation_data,tokenizer)

