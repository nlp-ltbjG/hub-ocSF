"""
文本分类实验可视化

两部分：
  1. 样本分析图 → plots/samples/
  2. 方法对比图 → plots/results/

使用方式：
  python visualize.py
"""

import json
from pathlib import Path
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

matplotlib.rcParams["axes.unicode_minus"] = False

def _find_chinese_font():
    candidates = ["SimHei", "Microsoft YaHei", "PingFang SC",
                  "Noto Sans CJK SC", "WenQuanYi Micro Hei",
                  "Arial Unicode MS"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

_CN_FONT = _find_chinese_font()
if _CN_FONT:
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SAMPLES_DIR = ROOT / "plots" / "samples"
RESULTS_DIR = ROOT / "plots" / "results"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]


def load_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
#  样本分析图 → plots/samples/
# ═══════════════════════════════════════════════════════════════════════

def plot_label_distribution():
    """样本类别分布"""
    train_data = load_json(DATA_DIR / "train.json")
    val_data = load_json(DATA_DIR / "val.json")
    if not train_data:
        return

    train_labels = Counter(item["label"] for item in train_data)
    val_labels = Counter(item["label"] for item in val_data) if val_data else {}

    label_map = load_json(DATA_DIR / "label_map.json")
    id2name = {int(k): v for k, v in label_map["id2name"].items()} if label_map else {}

    sorted_ids = sorted(train_labels.keys())
    names = [id2name.get(i, str(i)) for i in sorted_ids]
    train_counts = [train_labels[i] for i in sorted_ids]
    val_counts = [val_labels.get(i, 0) for i in sorted_ids]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(sorted_ids))
    width = 0.35

    bars1 = ax.bar(x - width / 2, train_counts, width, label="Train", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, val_counts, width, label="Val", color="#DD8452", edgecolor="white")

    for bar, cnt in zip(bars1, train_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(cnt), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("样本数")
    ax.set_title("训练集 & 验证集类别分布", fontsize=14, fontweight="bold")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(SAMPLES_DIR / "label_distribution.png", dpi=150)
    plt.close()
    print(f"  ✓ 类别分布图 → {SAMPLES_DIR / 'label_distribution.png'}")


def plot_text_length_distribution():
    """文本长度分布"""
    train_data = load_json(DATA_DIR / "train.json")
    if not train_data:
        return

    lengths = [len(item["sentence"]) for item in train_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(lengths, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
    for t, ls in [(64, "--"), (128, "-"), (256, ":")]:
        axes[0].axvline(t, color="#C44E52", linestyle=ls, alpha=0.7, label=f"max_len={t}")
    axes[0].set_title("文本长度分布", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("长度（字符数）")
    axes[0].set_ylabel("样本数")
    axes[0].legend()

    thresholds = np.arange(1, min(300, max(lengths) + 1))
    coverage = [(sum(1 for l in lengths if l <= t) / len(lengths) * 100) for t in thresholds]
    axes[1].plot(thresholds, coverage, color="#C44E52", linewidth=2)
    for target_pct in [95, 99]:
        idx = next((i for i, c in enumerate(coverage) if c >= target_pct), None)
        if idx is not None:
            axes[1].axvline(thresholds[idx], color="#C44E52", linestyle="--", alpha=0.5)
            axes[1].text(thresholds[idx] + 2, target_pct - 3,
                         f"{thresholds[idx]} 字\n覆盖 {target_pct}%", fontsize=8)
    axes[1].set_title("截断长度 vs 覆盖率", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("截断长度")
    axes[1].set_ylabel("覆盖率 (%)")

    plt.tight_layout()
    plt.savefig(SAMPLES_DIR / "text_length_distribution.png", dpi=150)
    plt.close()
    print(f"  ✓ 文本长度分布图 → {SAMPLES_DIR / 'text_length_distribution.png'}")


def plot_length_by_label():
    """各类别文本长度箱线图"""
    train_data = load_json(DATA_DIR / "train.json")
    if not train_data:
        return

    label_map = load_json(DATA_DIR / "label_map.json")
    id2name = {int(k): v for k, v in label_map["id2name"].items()} if label_map else {}

    label_lengths = {}
    for item in train_data:
        lid = item["label"]
        label_lengths.setdefault(lid, []).append(len(item["sentence"]))

    sorted_ids = sorted(label_lengths.keys())
    names = [id2name.get(i, str(i)) for i in sorted_ids]
    all_lengths = [label_lengths[i] for i in sorted_ids]

    fig, ax = plt.subplots(figsize=(14, 5))
    bp = ax.boxplot(all_lengths, labels=names, patch_artist=True, showfliers=False)
    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_ids)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("各类别文本长度分布", fontsize=14, fontweight="bold")
    ax.set_ylabel("文本长度（字符数）")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(SAMPLES_DIR / "length_by_label.png", dpi=150)
    plt.close()
    print(f"  ✓ 类别长度箱线图 → {SAMPLES_DIR / 'length_by_label.png'}")


def plot_sample_examples():
    """每类展示一个样本"""
    train_data = load_json(DATA_DIR / "train.json")
    if not train_data:
        return

    label_map = load_json(DATA_DIR / "label_map.json")
    id2name = {int(k): v for k, v in label_map["id2name"].items()} if label_map else {}

    seen = {}
    for item in train_data:
        lid = item["label"]
        if lid not in seen:
            seen[lid] = item["sentence"]
        if len(seen) == 15:
            break

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    rows = []
    for lid in sorted(seen.keys()):
        name = id2name.get(lid, str(lid))
        text = seen[lid][:40] + ("..." if len(seen[lid]) > 40 else "")
        rows.append([name, text])

    table = ax.table(cellText=rows, colLabels=["类别", "样本示例"],
                     loc="center", cellLoc="left", colWidths=[0.15, 0.75])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F5F5F5" if row % 2 == 0 else "white")

    ax.set_title("各类别样本示例", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(SAMPLES_DIR / "sample_examples.png", dpi=150)
    plt.close()
    print(f"  ✓ 样本示例图 → {SAMPLES_DIR / 'sample_examples.png'}")


# ═══════════════════════════════════════════════════════════════════════
#  方法对比图 → plots/results/
# ═══════════════════════════════════════════════════════════════════════

def plot_bert_pooling_comparison():
    """BERT 三种池化策略对比（一张图）"""
    log_mapping = {
        "cls":           ("train_log_cls.json",          "#4C72B0"),
        "cls-weighted":  ("train_log_cls_weighted.json", "#6A9BC3"),
        "mean":          ("train_log_mean.json",         "#DD8452"),
        "mean-weighted": ("train_log_mean_weighted.json","#E8A87C"),
        "max":           ("train_log_max.json",          "#55A868"),
        "max-weighted":  ("train_log_max_weighted.json", "#7EC88B"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    has_data = False
    for name, (fname, color) in log_mapping.items():
        data = load_json(OUTPUT_DIR / fname)
        if not data:
            continue
        has_data = True
        epochs = [d["epoch"] for d in data]
        train_loss = [d["train_loss"] for d in data]
        val_acc = [d["val_acc"] for d in data]
        val_f1 = [d["val_macro_f1"] for d in data]

        axes[0, 0].plot(epochs, train_loss, marker="o", color=color, label=name, linewidth=2)
        axes[0, 1].plot(epochs, val_acc, marker="s", color=color, label=name, linewidth=2)
        axes[1, 0].plot(epochs, val_f1, marker="^", color=color, label=name, linewidth=2)

    if not has_data:
        axes[0, 0].text(0.5, 0.5, "暂无数据\n请运行 train.py --pool cls/mean/max",
                        transform=axes[0, 0].transAxes, ha="center", va="center", fontsize=12)

    # 右下角：柱状图汇总
    pool_order = ["cls", "cls-weighted", "mean", "mean-weighted", "max", "max-weighted"]
    pool_colors = ["#4C72B0", "#6A9BC3", "#DD8452", "#E8A87C", "#55A868", "#7EC88B"]
    bar_names, bar_vals, bar_clrs = [], [], []
    for n, c in zip(pool_order, pool_colors):
        data = load_json(OUTPUT_DIR / log_mapping[n][0])
        if data:
            best = max(data, key=lambda x: x["val_acc"])
            bar_names.append(n)
            bar_vals.append(best["val_acc"])
            bar_clrs.append(c)

    if bar_names:
        x = np.arange(len(bar_names))
        bars = axes[1, 1].bar(x, bar_vals, color=bar_clrs, edgecolor="white", width=0.6)
        for bar, val in zip(bars, bar_vals):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(bar_names, fontsize=9)
        axes[1, 1].set_ylim(0, max(bar_vals) * 1.15)

    axes[0, 0].set_title("训练 Loss", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Train Loss")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_title("验证集准确率", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Val Accuracy")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_title("验证集 Macro F1", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Macro F1")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_title("最优准确率汇总", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Val Accuracy")

    fig.suptitle("BERT 三种池化策略对比", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bert_pooling_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ BERT池化对比图 → {RESULTS_DIR / 'bert_pooling_comparison.png'}")


def plot_llm_methods_comparison():
    """LLM 三种方式对比（zero-shot / LoRA / 全量微调，一张图）"""
    zs_data = load_json(OUTPUT_DIR / "llm_zero_shot_results.json")
    sft_data = load_json(OUTPUT_DIR / "llm_sft_results.json")
    full_ft_data = load_json(OUTPUT_DIR / "llm_full_ft_results.json")

    methods, accs, unparseable_rates, colors = [], [], [], []
    bar_colors = ["#DD8452", "#55A868", "#4C72B0"]

    if zs_data:
        methods.append("Zero-shot")
        accs.append(zs_data["accuracy"])
        unparseable_rates.append(zs_data["unparseable"] / zs_data["total"])
        colors.append(bar_colors[0])
    if sft_data:
        methods.append("SFT-LoRA")
        accs.append(sft_data["accuracy"])
        unparseable_rates.append(sft_data["unparseable"] / sft_data["total"])
        colors.append(bar_colors[1])
    if full_ft_data:
        methods.append("SFT-全量微调")
        accs.append(full_ft_data["accuracy"])
        unparseable_rates.append(full_ft_data["unparseable"] / full_ft_data["total"])
        colors.append(bar_colors[2])

    if not methods:
        print("  [跳过] 没有 LLM 结果数据")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 左图: 准确率柱状图
    bars = axes[0].bar(methods, accs, color=colors, edgecolor="white", width=0.5)
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{acc:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("验证集准确率", fontsize=12)
    axes[0].set_title("准确率对比", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, max(accs) * 1.2)
    axes[0].axhline(y=max(accs), color="gray", linestyle="--", alpha=0.3)

    # 中图: 正确/错误/无法解析 堆叠柱状图
    wrong_rates = [1 - accs[i] - unparseable_rates[i] for i in range(len(methods))]
    x = np.arange(len(methods))
    width = 0.5

    axes[1].bar(x, accs, width, label="正确", color="#55A868")
    axes[1].bar(x, wrong_rates, width, bottom=accs, label="预测错误", color="#C44E52")
    axes[1].bar(x, unparseable_rates, width,
                bottom=[accs[i] + wrong_rates[i] for i in range(len(methods))],
                label="无法解析", color="#DD8452")

    for i in range(len(methods)):
        axes[1].text(x[i], accs[i] / 2, f"{accs[i]:.1%}",
                     ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        if unparseable_rates[i] > 0.02:
            bottom_y = accs[i] + wrong_rates[i]
            axes[1].text(x[i], bottom_y + unparseable_rates[i] / 2,
                         f"{unparseable_rates[i]:.1%}",
                         ha="center", va="center", fontsize=9, color="white")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, fontsize=10)
    axes[1].set_ylabel("比例")
    axes[1].set_title("预测结果分布", fontsize=13, fontweight="bold")
    axes[1].legend(loc="upper right")

    # 右图: 各类别准确率
    def per_label_acc(results):
        label_stats = {}
        for r in results:
            true = r["true_label"]
            if true not in label_stats:
                label_stats[true] = {"correct": 0, "total": 0}
            label_stats[true]["total"] += 1
            if r["correct"]:
                label_stats[true]["correct"] += 1
        return {k: v["correct"] / v["total"] for k, v in label_stats.items()}

    datasets = []
    if zs_data:
        datasets.append(("Zero-shot", per_label_acc(zs_data["results"]), "#DD8452"))
    if sft_data:
        datasets.append(("SFT-LoRA", per_label_acc(sft_data["results"]), "#55A868"))
    if full_ft_data:
        datasets.append(("SFT-全量微调", per_label_acc(full_ft_data["results"]), "#4C72B0"))

    all_labels_set = set()
    for _, acc_dict, _ in datasets:
        all_labels_set.update(acc_dict.keys())
    all_labels = sorted(all_labels_set)

    n_methods = len(datasets)
    bar_w = 0.8 / n_methods
    x2 = np.arange(len(all_labels))

    for i, (name, acc_dict, color) in enumerate(datasets):
        vals = [acc_dict.get(l, 0) for l in all_labels]
        offset = (i - n_methods / 2 + 0.5) * bar_w
        axes[2].bar(x2 + offset, vals, bar_w, label=name, color=color, edgecolor="white")

    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(all_labels, fontsize=8)
    axes[2].set_ylabel("准确率")
    axes[2].set_title("各类别准确率", fontsize=13, fontweight="bold")
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 1.1)
    axes[2].tick_params(axis="x", rotation=40)

    fig.suptitle("LLM 不同训练方式对比 (Qwen2-0.5B)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "llm_methods_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ LLM方式对比图 → {RESULTS_DIR / 'llm_methods_comparison.png'}")


def plot_all_methods_comparison():
    """BERT vs LLM 全局对比"""
    bert_logs = {
        "BERT-cls":          "train_log_cls.json",
        "BERT-cls-weighted": "train_log_cls_weighted.json",
    }

    methods, accs, colors = [], [], []

    for name, fname in bert_logs.items():
        data = load_json(OUTPUT_DIR / fname)
        if data:
            best = max(data, key=lambda x: x["val_acc"])
            methods.append(name)
            accs.append(best["val_acc"])
            colors.append("#4C72B0")

    zs_data = load_json(OUTPUT_DIR / "llm_zero_shot_results.json")
    if zs_data:
        methods.append("Qwen2\nzero-shot")
        accs.append(zs_data["accuracy"])
        colors.append("#DD8452")

    sft_data = load_json(OUTPUT_DIR / "llm_sft_results.json")
    if sft_data:
        methods.append("Qwen2\nSFT-LoRA")
        accs.append(sft_data["accuracy"])
        colors.append("#55A868")

    full_ft_data = load_json(OUTPUT_DIR / "llm_full_ft_results.json")
    if full_ft_data:
        methods.append("Qwen2\nSFT-全量微调")
        accs.append(full_ft_data["accuracy"])
        colors.append("#4C72B0")

    if not methods:
        print("  [跳过] 没有对比数据")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, accs, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("验证集准确率", fontsize=12)
    ax.set_title("文本分类方法全局对比 (BERT vs LLM)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(accs) * 1.15)
    ax.axhline(y=max(accs), color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "all_methods_comparison.png", dpi=150)
    plt.close()
    print(f"  ✓ 全局对比图 → {RESULTS_DIR / 'all_methods_comparison.png'}")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("文本分类实验可视化")
    print("=" * 50)

    print(f"\n样本分析图 → {SAMPLES_DIR}")
    plot_label_distribution()
    plot_text_length_distribution()
    plot_length_by_label()
    plot_sample_examples()

    print(f"\n方法对比图 → {RESULTS_DIR}")
    plot_bert_pooling_comparison()
    plot_llm_methods_comparison()
    plot_all_methods_comparison()

    print(f"\n全部图片已保存")
    print(f"  样本分析: {SAMPLES_DIR}")
    print(f"  方法对比: {RESULTS_DIR}")
