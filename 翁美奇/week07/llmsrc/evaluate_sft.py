"""
加载 SFT checkpoint（LoRA / 全量微调），在验证集上评估 NER entity-level F1，
与 BERT+CRF 和 LLM API（zero/few-shot）多方对比

  1. 生成式 NER 的评估方式：JSON 解析 → span-level F1（与 llm_ner.py 一致）
  2. LoRA adapter 自动识别：目录含 adapter_config.json → LoRA，否则 → 全量
  3. 与 BERT+CRF 的对比：生成式 vs 序列标注，各有什么优劣

使用方式：
  python evaluate_sft.py                              # 评估 LoRA 模型（默认）
  python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt  # 评估全量微调模型
  python evaluate_sft.py --n_samples 50 --demo        # 5 条示例快速演示

依赖：
  pip install torch transformers peft
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

from pathlib import Path
import argparse
import json
import random
from tqdm import tqdm
import time

# 路径

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" 
MODEL_PATH  = ROOT.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
ADAPTER_DIR = ROOT / "outputs" / "sft_adapter"
LOG_DIR     = ROOT / "outputs" / "logs"

ENTITY_TYPES = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene",
]

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
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    

def load_model(model_path: str, ckpt_dir: str, device: torch.device):
    ckpt_path = Path(ckpt_dir)
    is_lora   = (ckpt_path / "adapter_config.json").exists()
    
    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("加载 LoRA adapter 需要 peft 库：pip install peft>=0.14.0")
        print(f"检测到 LoRA adapter，加载 base model: {model_path}")
        tokenizer  = AutoTokenizer.from_pretrained(
            str(Path(model_path).resolve()), trust_remote_code=True
        )
        # 1. 加载基座模型
        base_model = AutoModelForCausalLM.from_pretrained(
            str(Path(model_path).resolve()),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"加载 LoRA adapter: {ckpt_dir}")
        # 2. 加载 LoRA adapter
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        # 3. 合并权重（关键：接收返回值）
        model = model.merge_and_unload()
    else:
        print(f"检测到全量微调 checkpoint，直接加载: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    ckpt_type = "LoRA adapter 已合并" if is_lora else "全量微调模型"
    print(f"模型加载完成（{ckpt_type}）\n")
    return model, tokenizer

    
    
def predict(text: str, model, tokenizer, device: torch.device,
                 max_new_tokens: int = 256):
    """ """ 
    # 准备对话模板
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': text}
    ]
    
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len     = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    
       
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
        
    
def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 评估")
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--ckpt_dir",    default=str(ADAPTER_DIR),
                        help="checkpoint 目录；含 adapter_config.json → LoRA，否则 → 全量")
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--n_samples",   default=100, type=int,
                        help="验证集采样数（与 llm_ner.py 默认 100 条对齐）")
    parser.add_argument("--seed",        default=42,  type=int)
    parser.add_argument("--demo",        action="store_true",
                        help="只跑 5 条示例，快速演示")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"[错误] checkpoint 目录不存在：{ckpt_dir}")
        print("请先运行 train_sft.py 完成训练。")
        print("  LoRA（默认）保存到:   outputs/sft_adapter/")
        print("  全量微调保存到:        outputs/sft_full_ckpt/")
        return
    
    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(Path(args.data_dir) / "validation.json", encoding="utf-8") as f:
        val_data = json.load(f)
        # 加载验证集数据，默认n_samples = 100,随机100个
    with open(Path(args.data_dir) / 'validation.json') as f:
        val_data = random.sample(json.load(f), args.n_samples)
        
    random.seed(args.seed)
    n = 5 if args.demo else args.n_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}\n")
    
    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model_path, str(ckpt_dir), device)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    all_golds, all_preds = [], []
    detail_records = []
    parse_fail = 0
    t0 = time.time()
    
    for i, record  in enumerate(samples):
        tokens  = record["tokens"]
        ner_tags  = record["ner_tags"]
        
        p_set = bio_to_generated(tokens, ner_tags)
        
        raw = predict("".join(tokens), model, tokenizer, device)
        
        all_golds.append()
        
        
main()