# 文本分类实验可视化说明（只保留前1万条训练数据且epoch=1的训练）

## 项目结构

```
text_classification项目/
├── src/
│   ├── train.py              # BERT 微调训练
│   ├── evaluate.py           # BERT 评估
│   ├── visualize.py          # 可视化脚本（本说明对应）
│   └── compare_methods.py    # 旧版对比脚本
├── src_llm/
│   ├── classify_llm.py       # LLM zero-shot 推理
│   ├── train_sft.py          # LLM SFT 训练（LoRA / 全量微调）
│   └── evaluate_sft.py       # LLM SFT 评估
├── data/
│   ├── train.json            # 训练集（53,360 条）
│   ├── train_10k.json        # 训练集子集（前 10,000 条）
│   ├── val.json              # 验证集
│   └── test.json             # 测试集（标签为 -1）
├── outputs/                  # 训练日志 & 结果 JSON
└── plots/
    ├── samples/              # 样本分析图
    └── results/              # 方法对比图
```

## 快速使用

```bash
cd src
python visualize.py
```

图片自动保存到 `plots/samples/` 和 `plots/results/`。

---

## 一、样本分析图（plots/samples/）

### 1. label_distribution.png — 类别分布

**生成逻辑：** 统计训练集和验证集中每个类别的样本数量，绘制分组柱状图。

**结果含义：**
- 柱子高度 = 该类别的样本数
- 蓝色 = 训练集，橙色 = 验证集
- 如果某些类别柱子明显矮，说明**类别不均衡**，可能需要加权损失（`--use_class_weight`）

### 2. text_length_distribution.png — 文本长度分布

**生成逻辑：**
- 左图：统计所有训练样本的文本长度，绘制直方图，并标注 64/128/256 三个常用截断长度
- 右图：计算每个截断长度下的样本覆盖率，标注覆盖 95% 和 99% 所需的截断长度

**结果含义：**
- 左图：看文本长度的整体分布，是否有长尾
- 右图：帮助选择 `max_length` 参数
  - 如覆盖 95% 需 120 字 → 选 `max_length=128` 可保留 95% 样本的完整信息
  - 截断线右侧的面积 = 被截断影响的样本比例

### 3. length_by_label.png — 各类别文本长度箱线图

**生成逻辑：** 按类别分组，绘制每个类别的文本长度箱线图。

**结果含义：**
- 箱体 = 25%~75% 分位范围
- 中线 = 中位数
- 上下须线 = 1.5 倍 IQR 范围
- 如果某类箱体明显偏长/偏短，说明该类文本长度特征不同，可能影响分类效果

### 4. sample_examples.png — 样本示例

**生成逻辑：** 从训练集中为每个类别取第一条样本，生成表格。

**结果含义：** 快速了解各类别的文本内容特征，辅助理解分类难度。

---
# 训练数据：
bert取了前10000条，epoch = 1
llm-sft-lora 100条数据，epoch = 1
llm-sft  100条，epoch = 1
## 二、方法对比图（plots/results/）

### 1. bert_pooling_comparison.png — BERT 三种池化策略对比

**生成逻辑：** 读取不同池化策略（cls/mean/max）和加权 loss 变体的训练日志 JSON，绘制 2×2 子图。

| 子图 | 内容 | 数据来源 |
|------|------|---------|
| 左上 | 训练 Loss 曲线 | `train_log_*.json` 中的 `train_loss` |
| 右上 | 验证集准确率曲线 | `train_log_*.json` 中的 `val_acc` |
| 左下 | 验证集 Macro F1 曲线 | `train_log_*.json` 中的 `val_macro_f1` |
| 右下 | 最优准确率柱状图汇总 | 取每个配置的最高 `val_acc` |

**三种池化策略说明：**

| 策略 | 取值方式 | 特点 |
|------|---------|------|
| **cls** | 取 `[CLS]` token 的隐藏状态 | BERT 原始设计，[CLS] 专门用于句子表示 |
| **mean** | 对所有 token 的隐藏状态取均值（排除 padding） | 综合所有 token 信息，对长文本更鲁棒 |
| **max** | 对所有 token 的隐藏状态取最大值（排除 padding） | 捕获最显著特征，对关键词敏感 |

**加权 loss 说明：**
- `cls-weighted` = cls 池化 + `CrossEntropyLoss(weight=class_weight)`
- 类别权重 = `总样本数 / (类别数 × 该类样本数)`
- 样本少的类别权重更大，损失更重要

**结果含义：**
- Loss 曲线：下降越快、越低 → 收敛越好
- Acc/F1 曲线：越高越好，看哪种池化 + 是否加权效果最优
- 柱状图：直观对比各配置的最终效果

**生成完整数据需运行：**
```bash
python train.py --pool cls
python train.py --pool cls --use_class_weight
python train.py --pool mean
python train.py --pool mean --use_class_weight
python train.py --pool max
python train.py --pool max --use_class_weight
```

### 2. llm_methods_comparison.png — LLM 三种方式对比

**生成逻辑：** 读取 zero-shot、SFT-LoRA、SFT-全量微调的结果 JSON，绘制 1×3 子图。

| 子图 | 内容 | 数据来源 |
|------|------|---------|
| 左 | 准确率柱状图 | `llm_*_results.json` 中的 `accuracy` |
| 中 | 预测结果分布（正确/错误/无法解析） | `accuracy` + `unparseable/total` |
| 右 | 各类别准确率分组柱状图 | 逐条统计每个类别的正确率 |

**三种方式说明：**

| 方式 | 训练数据 | 参数更新 | 说明 |
|------|---------|---------|------|
| **Zero-shot** | 0 | 0 | 纯推理，靠 prompt 引导模型输出类别名 |
| **SFT-LoRA** | 5K | ~0.5% | 冻结原模型，只训练低秩适配器矩阵 |
| **SFT-全量微调** | 5K | 100% | 所有参数都参与训练 |

**"无法解析"说明：**
- LLM 生成式分类可能输出不在预定义类别中的文本（如"科技类"、"这个是科技"）
- 通过模糊匹配（`if name in raw_output`）尝试提取类别名
- 匹配不到的记为"无法解析"
- BERT 判别式分类不存在此问题

**结果含义：**
- 左图：SFT 后准确率应显著高于 zero-shot
- 中图：SFT 后"无法解析"比例应降低（模型学会了输出格式）
- 右图：看哪些类别容易混淆，哪些方法在哪些类别上有优势

**生成完整数据需运行：**
```bash
# zero-shot
cd src_llm && python classify_llm.py

# SFT-LoRA
python train_sft.py
python evaluate_sft.py

# SFT-全量微调
python train_sft.py --full_ft
python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt
```

### 3. all_methods_comparison.png — 全局对比

**生成逻辑：** 将 BERT 和 LLM 所有方法的最优准确率放在同一张柱状图中。

**结果含义：**
- 直观对比判别式（BERT）vs 生成式（LLM）的整体效果
- BERT 通常在数据充足时准确率更高（专精分类任务）
- LLM zero-shot 不需要训练数据，但准确率较低
- LLM SFT 后准确率提升，但仍有"无法解析"的风险

---

## 三、数据文件说明

| 文件 | 生成方式 | 内容 |
|------|---------|------|
| `outputs/train_log_cls.json` | `train.py --pool cls` | BERT-cls 训练日志 |
| `outputs/train_log_cls_weighted.json` | `train.py --pool cls --use_class_weight` | BERT-cls 加权训练日志 |
| `outputs/train_log_mean.json` | `train.py --pool mean` | BERT-mean 训练日志 |
| `outputs/train_log_max.json` | `train.py --pool max` | BERT-max 训练日志 |
| `outputs/llm_zero_shot_results.json` | `classify_llm.py` | LLM zero-shot 结果 |
| `outputs/llm_sft_results.json` | `evaluate_sft.py` | LLM SFT-LoRA 结果 |
| `outputs/llm_full_ft_results.json` | `evaluate_sft.py --ckpt_dir ...` | LLM 全量微调结果 |

缺失的 JSON 文件对应的图表会自动跳过或显示占位提示，不会报错。
