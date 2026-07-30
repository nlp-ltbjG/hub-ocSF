# MyData RAG 问答系统

基于自定义 PDF 文件构建的问答系统，支持 DeepSeek API + 本地 BGE Embedding。

## 目录结构

```
mydata/
├── raw_pdf/           # 原始 PDF 文件
│   ├── DeepSeek 15天指导手册——从入门到精通.pdf
│   ├── DeepSeek从入门到精通-清华.pdf
│   ├── 以太坊白皮书中文版.pdf
│   └── 稳定币 AI智能体经济的未来探索.pdf
├── parsed/            # PDF 解析结果（JSON）
├── chunks/            # 文本分块结果
│   └── all_semantic.json
├── vectorstore/       # 向量索引
│   ├── faiss_index.bin
│   └── faiss_meta.json
├── manifest.json      # 文件清单配置
├── run_pipeline.py    # 统一入口脚本
└── README.md          # 本文件
```

## 支持的数据源

| 文件 | 内容 |
|------|------|
| DeepSeek 15天指导手册 | DeepSeek 模型使用教程 |
| DeepSeek从入门到精通-清华 | 清华大学 DeepSeek 培训资料 |
| 以太坊白皮书中文版 | 以太坊技术白皮书 |
| 稳定币 AI智能体经济的未来探索 | AI Agent 与稳定币经济研究 |

## 技术栈

| 组件 | 方案 | 是否需要 API Key |
|------|------|-----------------|
| Embedding（向量化） | 本地 BGE-small-zh-v1.5 | ❌ 免费 |
| LLM（问答生成） | DeepSeek-v4-flash | ✅ 需要 |
| 向量检索 | FAISS | ❌ 免费 |
| 关键词检索 | BM25 (jieba) | ❌ 免费 |

## 快速开始

### 1. 配置 API Key

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
LLM_PROVIDER=deepseek
EMBEDDING_MODE=bge
```

### 2. 下载本地 BGE 模型（首次运行）

```bash
# 设置镜像加速（推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型（约90MB）
python src_langchain/download_model.py
```

### 3. 构建向量索引

```bash
# 完整流程：解析 → 分块 → 建索引
python mydata/run_pipeline.py --step all

# 或分步执行
python mydata/run_pipeline.py --step parse   # 解析 PDF
python mydata/run_pipeline.py --step chunk   # 文本分块
python mydata/run_pipeline.py --step build   # 构建索引（使用本地 BGE）
```

### 4. 开始问答

```bash
# 交互式问答
python mydata/run_pipeline.py --step query

# 单次查询
python mydata/run_pipeline.py --step query --query "什么是以太坊"
```

## 使用示例

```bash
$ python mydata/run_pipeline.py --step query

MyData RAG 问答系统
LLM：deepseek-v4-flash  |  Embedding：bge
索引：mydata/vectorstore/faiss_index.bin
输入 'exit' 退出，'mode' 查看配置

问题：什么是 DeepSeek？
```

## 配置选项

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | 空 | DeepSeek API Key（必需） |
| `LLM_PROVIDER` | `deepseek` | LLM 提供商：`deepseek` 或 `dashscope` |
| `EMBEDDING_MODE` | `bge` | Embedding 方式：`bge`（本地）或 `dashscope`（API） |

### 命令行参数

```bash
python mydata/run_pipeline.py --step <步骤> [--query <问题>]
```

| 步骤 | 说明 |
|------|------|
| `parse` | 解析 raw_pdf 中的 PDF 文件 |
| `chunk` | 将解析结果分块 |
| `build` | 构建 FAISS 向量索引 |
| `query` | 启动问答界面 |
| `all` | 一键执行完整流程 |

## 添加新 PDF

1. 将 PDF 文件放入 `mydata/raw_pdf/`
2. 在 `mydata/manifest.json` 中添加条目：

```json
{
  "stock_code": "",
  "plate": "",
  "company_name": "",
  "year": "",
  "title": "新文件标题",
  "filename": "新文件.pdf",
  "source_url": "",
  "announce_id": ""
}
```

3. 重新构建索引：

```bash
python mydata/run_pipeline.py --step all
```

## 常见问题

### Q: API Key 认证失败？

请确认：
1. API Key 格式正确（以 `sk-` 开头）
2. API Key 未过期
3. `.env` 文件配置正确，无多余空格

### Q: 模型下载慢？

设置镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 切换到 DashScope？

修改 `.env` 文件：

```env
DASHSCOPE_API_KEY=sk-your-dashscope-api-key
LLM_PROVIDER=dashscope
EMBEDDING_MODE=dashscope
```

## 注意事项

- 首次构建索引可能需要几分钟（取决于 PDF 数量和大小）
- BGE 模型约 90MB，下载需联网
- DeepSeek API Key 需自行申请（[DeepSeek 官网](https://platform.deepseek.com)）
- 原有年报系统不受影响，继续使用 `python src/rag_pipeline.py`