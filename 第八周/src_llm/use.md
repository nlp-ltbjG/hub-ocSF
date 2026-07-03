好的，训练步骤如下：

---

## 第一步：进入项目目录

```cmd
cd c:\Users\29356\Desktop\八斗学院\week8文本匹配问题\week8 文本匹配问题\文本匹配项目\src

```

---

## 第二步：训练 BQ Corpus（3种方法）

### 方法1：BiEncoder + Cosine
```cmd
C:\Users\29356\python310\python.exe train_biencoder.py --data_dir ../data/bq_corpus --loss cosine --epochs 1
```

### 方法2：BiEncoder + Triplet
```cmd
C:\Users\29356\python310\python.exe train_biencoder.py --data_dir ../data/bq_corpus --loss triplet --epochs 1
```

### 方法3：CrossEncoder
```cmd
C:\Users\29356\python310\python.exe train_crossencoder.py --data_dir ../data/bq_corpus --epochs 1
```

---

## 第三步：训练 LCQMC（3种方法）

```cmd
C:\Users\29356\python310\python.exe train_biencoder.py --data_dir ../data/lcqmc --loss cosine --epochs 1

C:\Users\29356\python310\python.exe train_biencoder.py --data_dir ../data/lcqmc --loss triplet --epochs 1

C:\Users\29356\python310\python.exe train_crossencoder.py --data_dir ../data/lcqmc --epochs 1
```

---

## 第四步：生成图形

### BQ Corpus 图形
```cmd
C:\Users\29356\python310\python.exe explore_data.py --data_dir ../data/bq_corpus --output_dir ../outputs/bq_corpus/figures
```

### LCQMC 图形
```cmd
C:\Users\29356\python310\python.exe explore_data.py --data_dir ../data/lcqmc --output_dir ../outputs/lcqmc/figures
```

---

## 总共 8 个命令

| 数据集 | 训练 | 图形 | 合计 |
|--------|------|------|------|
| BQ Corpus | 3 | 1 | 4 |
| LCQMC | 3 | 1 | 4 |
| **总计** | **6** | **2** | **8** |

---

开始执行第一个命令吧！


是的，AFQMC 的 figures 有很多，包括：

- `label_distribution.png` - 标签分布
- `char_length_distribution.png` - 字符长度分布
- `token_length_distribution.png` - Token长度分布
- `length_diff_distribution.png` - 长度差分布
- `biencoder_sim_distributions.png` - 相似度分布
- `biencoder_badcase_dist.png` - 错误分析
- `method_comparison_bar.png` - 方法对比
- 等等

---

## 为什么我只说了 2 个

`explore_data.py` 生成基础图表（4-5个），`compare_methods.py` 生成对比图表（3-4个）。

AFQMC 的完整图表是运行了多个脚本生成的。

---

## LCQMC 和 BQ Corpus 也需要同样操作


# BQ Corpus 基础图表
```
C:\Users\29356\python310\python.exe explore_data.py --data_dir ../data/bq_corpus --output_dir ../outputs/bq_corpus/figures
```

# BQ Corpus 对比图表（需要训练完3个模型后）
C:\Users\29356\python310\python.exe compare_methods.py --data_dir ../data/bq_corpus --output_dir ../outputs/bq_corpus/figures

# LCQMC 基础图表
C:\Users\29356\python310\python.exe explore_data.py --data_dir ../data/lcqmc --output_dir ../outputs/lcqmc/figures

# LCQMC 对比图表
C:\Users\29356\python310\python.exe compare_methods.py --data_dir ../data/lcqmc --output_dir ../outputs/lcqmc/figures
```

---

## 完整流程

| 步骤 | 命令数 |
|------|--------|
| BQ Corpus 训练 | 3 |
| LCQMC 训练 | 3 |
| BQ Corpus 图形 | 2 |
| LCQMC 图形 | 2 |
| **总计** | **10** |

---

先开始训练吧！图形等训练完再生成。