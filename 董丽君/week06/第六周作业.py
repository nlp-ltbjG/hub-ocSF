"""
对比 LoRA SFT 和 LLM zero-shot 文本分类效果的综合验证程序

  1. 统一评估框架：使用相同的验证集、相同的解析逻辑、相同的评估指标
  2. 量化对比：准确率、无法解析率、推理耗时
  3. 定性分析：输出错误案例，分析两种方法的差异

使用方式：
  python compare_lora_vs_zero_shot.py              # 默认 200 条样本
  python compare_lora_vs_zero_shot.py --demo       # 5 条快速演示
  python compare_lora_vs_zero_shot.py --num_samples 500  # 更多样本

依赖：
  pip install torch transformers peft
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT             = Path(__file__).parent / "text_classification项目"
DATA_DIR         = ROOT / "data"
MODEL_PATH       = ROOT.parent.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
ADAPTER_DIR      = ROOT / "outputs" / "sft_adapter"
OUTPUT_DIR       = ROOT / "outputs"

# 15 个类别
LABEL_NAMES = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]

SYSTEM_PROMPT = (
    "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
    "只输出类别名称，不要输出任何其他内容。\n"
    "可选类别：" + "、".join(LABEL_NAMES)
)


def build_prompt(text: str) -> str:
    return f"新闻标题：{text}\n类别："


def load_zero_shot_model(model_path: str, device: torch.device):
    """加载原始 LLM 模型用于 zero-shot 推理"""
    print(f"加载 Zero-Shot 模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def load_lora_model(base_model_path: str, adapter_dir: str, device: torch.device):
    """加载 LoRA adapter 后的模型"""
    if not PEFT_AVAILABLE:
        raise ImportError("加载 LoRA adapter 需要 peft 库：pip install peft>=0.14.0")
    
    print(f"加载 LoRA SFT 模型: {base_model_path} + {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(base_model_path).resolve()), trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(Path(base_model_path).resolve()),
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(Path(adapter_dir).resolve()))
    model = model.merge_and_unload()  # 合并权重加速推理
    
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def classify_one(text: str, model, tokenizer, device: torch.device, max_new_tokens: int = 8) -> str:
    """单次分类推理"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(text)},
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


def parse_prediction(raw_output: str) -> str | None:
    """解析模型输出，提取类别名"""
    for name in LABEL_NAMES:
        if name in raw_output:
            return name
    return None


def evaluate_method(method_name: str, samples: list, id2name: dict, 
                    model, tokenizer, device: torch.device) -> dict:
    """评估单个方法"""
    print(f"\n{'='*60}")
    print(f"正在评估: {method_name}")
    print(f"{'='*60}")
    
    correct, total, unparseable = 0, 0, 0
    results = []
    errors = []
    t0 = time.time()

    for i, item in enumerate(samples):
        text      = item["sentence"]
        true_id   = item["label"]
        true_name = id2name[true_id]

        raw_output = classify_one(text, model, tokenizer, device)
        pred_name  = parse_prediction(raw_output)

        is_correct = (pred_name == true_name)
        if pred_name is None:
            unparseable += 1
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "text": text, "true_label": true_name,
            "pred_label": pred_name, "raw_output": raw_output,
            "correct": is_correct,
        })
        
        # 记录错误案例
        if not is_correct:
            errors.append({
                "text": text,
                "true_label": true_name,
                "pred_label": pred_name,
                "raw_output": raw_output,
            })

        # 打印进度
        status = "✓" if is_correct else ("?" if pred_name is None else "✗")
        print(f"[{i+1:3d}/{len(samples)}] {status}  "
              f"真实:{true_name:4s}  预测:{str(pred_name):4s}  |  {text[:40]}")

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0

    return {
        "method": method_name,
        "accuracy": acc,
        "total": total,
        "correct": correct,
        "unparseable": unparseable,
        "elapsed": elapsed,
        "per_sample_time": elapsed / total if total > 0 else 0,
        "results": results,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT vs Zero-Shot 对比验证")
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--adapter_dir", default=str(ADAPTER_DIR))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--num_samples", default=200, type=int)
    parser.add_argument("--seed",        default=42, type=int)
    parser.add_argument("--demo",        action="store_true", help="快速演示（5条）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 检查 LoRA adapter 是否存在
    if not Path(args.adapter_dir).exists():
        print(f"[错误] LoRA adapter 不存在: {args.adapter_dir}")
        print("请先运行: python train_sft.py")
        return

    # 加载数据
    with open(Path(args.data_dir) / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(Path(args.data_dir) / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    # 采样
    random.seed(args.seed)
    n = 5 if args.demo else args.num_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}\n")

    # ===== 1. Zero-Shot 评估 =====
    model_zs, tokenizer_zs = load_zero_shot_model(args.model_path, device)
    result_zs = evaluate_method("LLM Zero-Shot", samples, id2name, model_zs, tokenizer_zs, device)
    
    # 清理显存
    del model_zs
    torch.cuda.empty_cache()

    # ===== 2. LoRA SFT 评估 =====
    model_lora, tokenizer_lora = load_lora_model(args.model_path, args.adapter_dir, device)
    result_lora = evaluate_method("LoRA SFT", samples, id2name, model_lora, tokenizer_lora, device)

    # ===== 3. 综合对比 =====
    print(f"\n{'='*70}")
    print("            LoRA SFT vs LLM Zero-Shot 对比结果")
    print(f"{'='*70}")
    print(f"{'指标':<20} {'Zero-Shot':<20} {'LoRA SFT':<20} {'提升幅度':<10}")
    print(f"{'-'*70}")
    print(f"{'准确率':<20} {result_zs['accuracy']:.4f} {'(' + str(result_zs['correct']) + '/' + str(result_zs['total']) + ')':<15} "
          f"{result_lora['accuracy']:.4f} {'(' + str(result_lora['correct']) + '/' + str(result_lora['total']) + ')':<15} "
          f"{(result_lora['accuracy'] - result_zs['accuracy'])*100:+.1f}%")
    print(f"{'无法解析率':<20} {result_zs['unparseable']/result_zs['total']*100:.1f}% {'(' + str(result_zs['unparseable']) + '条)':<13} "
          f"{result_lora['unparseable']/result_lora['total']*100:.1f}% {'(' + str(result_lora['unparseable']) + '条)':<13} "
          f"{(result_zs['unparseable']/result_zs['total'] - result_lora['unparseable']/result_lora['total'])*100:+.1f}%")
    print(f"{'平均耗时':<20} {result_zs['per_sample_time']:.2f}s {'':<15} "
          f"{result_lora['per_sample_time']:.2f}s {'':<15}")
    print(f"{'总耗时':<20} {result_zs['elapsed']:.1f}s {'':<15} "
          f"{result_lora['elapsed']:.1f}s {'':<15}")
    print(f"{'='*70}")

    # 输出错误案例对比
    print("\n【错误案例对比】")
    print(f"Zero-Shot 错误数: {len(result_zs['errors'])}")
    print(f"LoRA SFT 错误数: {len(result_lora['errors'])}")
    
    # 找出关键差异案例（一方对一方错）
    print("\n【关键差异案例】（Zero-Shot错但LoRA对）")
    key_differences = []
    for zs_err, lora_res in zip(result_zs['results'], result_lora['results']):
        if not zs_err['correct'] and lora_res['correct']:
            key_differences.append({
                "text": zs_err['text'],
                "true_label": zs_err['true_label'],
                "zs_pred": zs_err['pred_label'],
                "lora_pred": lora_res['pred_label'],
            })
    
    for i, case in enumerate(key_differences[:5]):  # 最多显示5个
        print(f"\n案例 {i+1}:")
        print(f"  标题: {case['text']}")
        print(f"  真实: {case['true_label']}, Zero-Shot: {case['zs_pred']}, LoRA: {case['lora_pred']}")

    # 保存对比结果
    out_path = OUTPUT_DIR / "lora_vs_zero_shot_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "zero_shot": result_zs,
            "lora_sft": result_lora,
            "comparison": {
                "accuracy_gain": result_lora['accuracy'] - result_zs['accuracy'],
                "unparseable_reduction": result_zs['unparseable'] - result_lora['unparseable'],
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
