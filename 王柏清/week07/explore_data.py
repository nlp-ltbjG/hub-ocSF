import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT1 = Path(__file__).parent.parent
ROOT2 = Path(__file__).parent
DATA_DIR = ROOT1 / "data" / "peoples_daily"
FIG_DIR = ROOT2 / "outputs" / "figures"

# 定义合法标签集合（仅3种实体类型）
VALID_TAGS = {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"}
# 实体类型中文映射
ENTITY_LABEL_MAP = {
    "PER": "人名",
    "ORG": "机构",
    "LOC": "地点"
}


def load_split(split: str) -> list:
    """完全保留原训练/验证集加载逻辑"""
    path = DATA_DIR / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_stats(records: list) -> dict:
    """
    适配 tokens+ner_tags 格式的BIO标注统计
    仅支持 PER/ORG/LOC 三种实体类型，增加标签合法性校验
    """
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type = {}
    invalid_tag_counts = Counter()  # 统计非法标签

    for row_idx, row in enumerate(records):
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]
        text_lengths.append(len(tokens))

        # 标签长度校验
        if len(tokens) != len(ner_tags):
            print(f"⚠️  警告：第 {row_idx} 条样本 tokens 与 ner_tags 长度不一致！")
            continue

        total_entities = 0
        current_entity = None  # 格式：(实体类型, 起始索引)

        # 逐标签解析BIO序列
        for idx, tag in enumerate(ner_tags):
            # 标签合法性校验
            if tag not in VALID_TAGS:
                invalid_tag_counts[tag] += 1
                continue

            # 1. 遇到B-XXX：新实体开始
            if tag.startswith("B-"):
                # 容错：如果有未结束的实体（异常标注），先保存
                if current_entity is not None:
                    etype, start_idx = current_entity
                    entity_len = idx - start_idx
                    entity_lengths.append(entity_len)
                    entity_type_counts[etype] += 1
                    entity_text = "".join(tokens[start_idx:idx])
                    if etype not in entities_by_type:
                        entities_by_type[etype] = []
                    entities_by_type[etype].append(entity_text)
                    total_entities += 1
                # 初始化新实体
                entity_type = tag.split("-")[1]
                current_entity = (entity_type, idx)

            # 2. 遇到I-XXX：延续当前实体
            elif tag.startswith("I-"):
                if current_entity is None:
                    continue  # 异常：I无前置B，跳过
                current_type, _ = current_entity
                tag_type = tag.split("-")[1]
                # 容错：I类型与当前实体不一致，结束当前实体
                if tag_type != current_type:
                    etype, start_idx = current_entity
                    entity_len = idx - start_idx
                    entity_lengths.append(entity_len)
                    entity_type_counts[etype] += 1
                    entity_text = "".join(tokens[start_idx:idx])
                    if etype not in entities_by_type:
                        entities_by_type[etype] = []
                    entities_by_type[etype].append(entity_text)
                    total_entities += 1
                    current_entity = None

            # 3. 遇到O：结束当前实体
            else:
                if current_entity is not None:
                    etype, start_idx = current_entity
                    entity_len = idx - start_idx
                    entity_lengths.append(entity_len)
                    entity_type_counts[etype] += 1
                    entity_text = "".join(tokens[start_idx:idx])
                    if etype not in entities_by_type:
                        entities_by_type[etype] = []
                    entities_by_type[etype].append(entity_text)
                    total_entities += 1
                    current_entity = None

        # 处理句子末尾未结束的实体
        if current_entity is not None:
            etype, start_idx = current_entity
            entity_len = len(tokens) - start_idx
            entity_lengths.append(entity_len)
            entity_type_counts[etype] += 1
            entity_text = "".join(tokens[start_idx:])
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append(entity_text)
            total_entities += 1

        entity_per_sentence.append(total_entities)

    # 打印非法标签统计
    if invalid_tag_counts:
        print("\n⚠️  发现非法标签：")
        for tag, cnt in invalid_tag_counts.items():
            print(f"  {tag}: {cnt} 次")
        print("  已自动跳过这些标签进行统计\n")

    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
        "invalid_tag_counts": invalid_tag_counts
    }


def print_summary(stats_train: dict, stats_val: dict):
    """仅保留PER/ORG/LOC三种实体类型的统计输出"""
    print("=" * 70)
    print("BIO格式NER数据集统计摘要（训练+验证集）")
    print("实体类型：PER(人名)、ORG(机构)、LOC(地点)")
    print("=" * 70)

    print("\n【训练集】")
    print(f"  样本数：{len(stats_train['text_lengths'])} 条")
    if stats_train['text_lengths']:
        print(f"  文本平均长度：{sum(stats_train['text_lengths']) / len(stats_train['text_lengths']):.1f} 字")
        print(f"  文本最大长度：{max(stats_train['text_lengths'])} 字")
        print(f"  文本长度中位数：{sorted(stats_train['text_lengths'])[len(stats_train['text_lengths'])//2]} 字")
    if stats_train['entity_per_sentence']:
        print(f"  平均实体数/句：{sum(stats_train['entity_per_sentence']) / len(stats_train['entity_per_sentence']):.2f}")
    print(f"  实体总数：{sum(stats_train['entity_type_counts'].values())}")
    if stats_train['entity_lengths']:
        print(f"  平均实体长度：{sum(stats_train['entity_lengths']) / len(stats_train['entity_lengths']):.1f} 字")

    print("\n【验证集】")
    print(f"  样本数：{len(stats_val['text_lengths'])} 条")
    if stats_val['text_lengths']:
        print(f"  文本平均长度：{sum(stats_val['text_lengths']) / len(stats_val['text_lengths']):.1f} 字")
    print(f"  实体总数：{sum(stats_val['entity_type_counts'].values())}")

    print("\n【各类实体频次（训练集）】")
    for etype, cnt in sorted(stats_train["entity_type_counts"].items(), key=lambda x: -x[1]):
        cn = ENTITY_LABEL_MAP[etype]
        print(f"  {etype:15s} ({cn:8s}) : {cnt:5d} 条")

    print("\n【各类实体示例（训练集，取前5个）】")
    for etype in sorted(stats_train["entities_by_type"]):
        cn = ENTITY_LABEL_MAP[etype]
        examples = list(dict.fromkeys(stats_train["entities_by_type"][etype]))[:5]
        print(f"  {etype:15s} ({cn}) : {' | '.join(examples)}")

    print()


def plot_entity_distribution(stats_train: dict):
    """仅绘制PER/ORG/LOC三种实体类型的分布"""
    counts = stats_train["entity_type_counts"]
    if not counts:
        print("  无实体数据，跳过实体分布图表生成")
        return
    
    # 按频次排序
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    labels = [f"{k}\n({ENTITY_LABEL_MAP[k]})" for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52"], alpha=0.85, edgecolor="white")
    
    # 在柱子上标注数值
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values)*0.01, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    ax.set_title("各类实体频次分布（训练集）", fontsize=14, pad=15)
    ax.set_ylabel("实体数量", fontsize=12)
    ax.set_xlabel("实体类型", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "entity_distribution.png", dpi=150, bbox_inches="tight")
    print(f"  已保存 → {FIG_DIR / 'entity_distribution.png'}")
    plt.close()


def plot_text_length_distribution(stats_train: dict):
    """完全保留原逻辑"""
    lengths = stats_train["text_lengths"]
    if not lengths:
        print("  无文本长度数据，跳本文本长度图表生成")
        return
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    
    # 标注关键阈值
    ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, label="max_length=64")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    p95 = sorted(lengths)[int(len(lengths) * 0.95)]
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    
    ax.set_title("文本长度分布（训练集）", fontsize=14, pad=15)
    ax.set_xlabel("文本字符数", fontsize=12)
    ax.set_ylabel("样本数", fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "text_length_distribution.png", dpi=150, bbox_inches="tight")
    print(f"  已保存 → {FIG_DIR / 'text_length_distribution.png'}")
    plt.close()
    print(f"  P95 文本长度={p95}，建议 max_length=128（若P95>128可调整为256）")


def plot_entity_length_distribution(stats_train: dict):
    """完全保留原逻辑"""
    lengths = stats_train["entity_lengths"]
    if not lengths:
        print("  无实体长度数据，跳过实体长度图表生成")
        return
    
    length_counter = Counter(lengths)
    # 取前20个长度（避免过长）
    xs = sorted(length_counter.keys())[:20]
    ys = [length_counter[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(x) for x in xs], ys, color="#55A868", alpha=0.85, edgecolor="white")
    
    ax.set_title("实体长度分布（训练集，前20）", fontsize=14, pad=15)
    ax.set_xlabel("实体字符数", fontsize=12)
    ax.set_ylabel("出现次数", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "entity_length_distribution.png", dpi=150, bbox_inches="tight")
    print(f"  已保存 → {FIG_DIR / 'entity_length_distribution.png'}")
    plt.close()

    avg_len = sum(lengths) / len(lengths)
    print(f"  实体平均长度={avg_len:.1f}字，CRF对短实体边界识别优势更明显")


def main():
    parse_args()

    print("正在加载训练集...")
    train_records = load_split("train")
    print("正在加载验证集...")
    val_records = load_split("validation")

    print("\n正在统计训练集数据...")
    stats_train = collect_stats(train_records)
    print("正在统计验证集数据...")
    stats_val = collect_stats(val_records)

    print_summary(stats_train, stats_val)

    print("正在生成可视化图表...")
    plot_entity_distribution(stats_train)
    plot_text_length_distribution(stats_train)
    plot_entity_length_distribution(stats_train)

    print("\n✅ 探索完成！图表已保存到 outputs/figures/")
    print("下一步：python train.py               # 训练 BERT+Linear")
    print("         python train.py --use_crf    # 训练 BERT+CRF")


def parse_args():
    parser = argparse.ArgumentParser(description="探索 peoples_daily 数据集")
    return parser.parse_args()


if __name__ == "__main__":
    main()