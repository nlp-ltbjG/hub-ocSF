# 命名实体识别（NER）项目

本项目实现了基于 BERT 和 LLM 的中文命名实体识别任务，支持三种实体类型：人名（PER）、地名（LOC）、机构（ORG）。

## 项目结构

```
week07/
├── data/                    # 数据集目录
│   ├── train.json           # 训练集（tokens + ner_tags 格式）
│   ├── validation.json      # 验证集
│   ├── test.json            # 测试集
│   └── label_names.json     # 标签体系
├── src/                     # BERT NER 代码
│   ├── dataset.py           # 数据集处理
│   ├── model.py             # BERT + Linear/CRF 模型
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评估脚本
│   ├── compare_results.py   # 结果对比
│   ├── explore_data.py      # 数据探索
│   └── download_data.py     # 数据下载
├── llmsrc/                  # LLM NER 代码
│   ├── train_sft.py         # LLM LoRA 微调训练
│   ├── evaluate_sft.py      # LLM SFT 模型评估
│   └── llm_ner.py          # LLM API zero-shot/few-shot 评估
├── outputs/                 # 输出目录
│   ├── checkpoints/         # BERT 模型 checkpoint
│   ├── logs/                # 训练日志
│   ├── sft_adapter/         # LLM LoRA adapter
│   └── sft_full_ckpt/       # LLM 全量微调模型
└── README.md                # 本文件
```

## 数据格式

数据集采用 `tokens + ner_tags` 格式：

```json
{
  "tokens": ["海", "钓", "比", "赛", "地", "点", "在", "厦", "门"],
  "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC"]
}
```

### 标签体系

| 标签 | 含义 |
|------|------|
| `O` | 非实体 |
| `B-PER` | 人名开头 |
| `I-PER` | 人名继续 |
| `B-ORG` | 机构开头 |
| `I-ORG` | 机构继续 |
| `B-LOC` | 地名开头 |
| `I-LOC` | 地名继续 |

### 数据规模

| 数据集 | 样本数 |
|--------|--------|
| 训练集 | 20,864 条 |
| 验证集 | 2,318 条 |
| 测试集 | 4,636 条 |

## 依赖安装

```bash
# 基础依赖（BERT 训练）
pip install torch transformers seqeval tqdm pytorch-crf

# LLM 微调依赖
pip install peft

# LLM API 评估依赖
pip install openai
```

## 使用方法

### 1. BERT NER 训练

```bash
cd src

# 训练 BERT + Linear（基线模型）
python train.py

# 训练 BERT + CRF（推荐）
python train.py --use_crf

# 自定义超参数
python train.py --use_crf --epochs 5 --batch_size 16 --lr 3e-5
```

### 2. BERT NER 评估

```bash
# 评估 BERT + Linear
python evaluate.py

# 评估 BERT + CRF
python evaluate.py --use_crf

# 在测试集上评估
python evaluate.py --use_crf --split test
```

### 3. LLM LoRA 微调

```bash
cd ../llmsrc
python train_sft.py

# 快速测试（1000 条数据）
python train_sft.py --num_train 1000 --epochs 1

# 全量微调（需显存 ≥ 16GB）
python train_sft.py --full_ft --lr 2e-5
```

### 4. LLM SFT 模型评估

```bash
# 评估 LoRA 模型（默认）
python evaluate_sft.py

# 演示模式（5条示例）
python evaluate_sft.py --demo

# 评估全量微调模型
python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt
```

### 5. LLM API 评估（Zero-shot / Few-shot）

需要设置 API Key 环境变量：

```bash
# DeepSeek API
export DEEPSEEK_API_KEY="sk-你的API密钥"

# 或使用其他 API（需修改 llm_ner.py 中的 base_url）
```

运行评估：

```bash
python llm_ner.py

# 指定样本数和模型
python llm_ner.py --n_samples 50 --model deepseek-chat
```

### 6. 结果对比

```bash
cd ../src
python compare_results.py
```

## 输出说明

| 文件/目录 | 说明 |
|-----------|------|
| `outputs/checkpoints/best_linear.pt` | BERT+Linear 最佳模型 |
| `outputs/checkpoints/best_crf.pt` | BERT+CRF 最佳模型 |
| `outputs/logs/train_*.json` | 训练日志 |
| `outputs/logs/eval_*.json` | 评估日志 |
| `outputs/sft_adapter/` | LLM LoRA adapter |
| `outputs/sft_full_ckpt/` | LLM 全量微调模型 |

## 关键配置

### train.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_crf` | False | 是否使用 CRF 层 |
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--max_length` | 128 | 序列最大长度 |
| `--lr` | 2e-5 | BERT 层学习率 |
| `--head_lr_mult` | 5.0 | 分类头学习率倍数 |

### train_sft.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 2 | 批次大小 |
| `--grad_accum` | 8 | 梯度累积步数 |
| `--max_length` | 192 | 序列最大长度 |
| `--full_ft` | False | 是否全量微调 |
| `--num_train` | -1 | 训练样本数（-1 使用全部） |
| `--lora_r` | 4 | LoRA 秩 |
| `--lora_alpha` | 8 | LoRA 缩放因子 |

### llm_ner.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_samples` | 100 | 采样数量 |
| `--model` | deepseek-chat | API 模型名称 |

## 模型对比

| 模型 | 优点 | 缺点 |
|------|------|------|
| BERT + Linear | 训练快，简单 | 可能产生非法序列 |
| BERT + CRF | 保证序列合法性 | 训练较慢 |
| LLM + LoRA | 生成式输出，泛化能力强 | 需要更多资源 |
| LLM API | 无需训练，零样本能力强 | 依赖 API，成本较高 |

## 教学重点

1. **BIO 标注规范**：理解 B-X/I-X/O 的含义和标注规则
2. **子词对齐**：BERT tokenizer 的 word_ids() 策略处理
3. **CRF 层作用**：转移矩阵学习标签依赖关系，Viterbi 解码保证合法序列
4. **Loss Masking**：忽略特殊 token 和非首子词（-100）
5. **LoRA 微调**：高效微调大型语言模型
6. **Prompt Engineering**：Zero-shot / Few-shot 设计

## 参考资料

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft)
- [DeepSeek API](https://platform.deepseek.com/)
- [CRF Layer on BiLSTM-CRF](https://arxiv.org/abs/1603.01360)
