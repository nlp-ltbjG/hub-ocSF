# MyData RAG 问答系统 — 实施计划

## 1. Summary

基于现有 `rag_annual_report` 项目的原生版 RAG 代码，为 `mydata/raw_pdf/` 中的 4 个自有 PDF 建立一套独立的问答知识库。方案核心思想是**零侵入复用**：不修改 `src/` 下任何现有文件，通过创建一个新的统一入口脚本，在运行时动态修改现有模块的全局路径变量，使其指向 `mydata/` 目录。

## 2. Current State Analysis

### 2.1 现有代码路径硬编码问题

现有 4 个核心脚本均在模块顶部通过 `Path(__file__).parent.parent / "data" / ...` 硬编码了数据目录：

| 脚本 | 硬编码路径 | 用途 |
|------|-----------|------|
| `src/parse_pdf.py` | `data/raw_pdf/`, `data/parsed/` | PDF 解析输入/输出 |
| `src/chunk_documents.py` | `data/parsed/`, `data/chunks/` | 分块输入/输出 |
| `src/build_index.py` | `data/chunks/`, `vectorstore/` | 建索引输入/输出 |
| `src/rag_pipeline.py` | `vectorstore/faiss_index.bin`, `faiss_meta.json` | 问答加载索引 |

### 2.2 用户已有数据

`mydata/raw_pdf/` 下已有 4 个 PDF：
- `DeepSeek 15天指导手册——从入门到精通.pdf`
- `DeepSeek从入门到精通-清华.pdf`
- `以太坊白皮书中文版.pdf`
- `稳定币 AI智能体经济的未来探索 总 12.5(1).pdf`

这些 PDF 主题与原有年报完全不同，需要独立的索引和知识库。

### 2.3 现有代码可复用性

- `AnnualReportParser` 类：完全可复用，它只依赖 `pdf_path` 和 `meta` 参数，不依赖目录结构
- `chunk_semantic()` / `chunk_fixed()` / `chunk_hierarchical()`：完全可复用，只操作 block 列表
- `embed_texts()` / `build_faiss_index()`：完全可复用，但依赖模块级 `CHUNKS_FILE` 和 `VECTORSTORE_DIR`
- `VectorStore` / `BM25Store` / `RAGPipeline`：完全可复用，但依赖模块级 `INDEX_PATH` 和 `META_PATH`

**结论**：所有核心逻辑均可复用，只需在导入模块后修改其全局路径常量。

## 3. Proposed Changes

### 3.1 创建 mydata 目录结构

在 `mydata/` 下创建标准数据流水线目录：

```
mydata/
├── raw_pdf/              ← 已有（4个PDF）
├── parsed/               ← 新建：解析后的 JSON
├── chunks/               ← 新建：分块后的 JSON
├── vectorstore/          ← 新建：FAISS 索引和元数据
├── manifest.json         ← 新建：PDF 元数据索引
└── run_pipeline.py       ← 新建：统一入口脚本
```

### 3.2 创建 mydata/manifest.json

手动构建一个与 `data/manifest.json` 格式兼容的清单文件，描述 mydata 中的 PDF。由于这些 PDF 不是来自巨潮资讯网，字段需要简化但保持兼容：

```json
[
  {
    "stock_code": "",
    "plate": "",
    "company_name": "",
    "year": "",
    "title": "DeepSeek 15天指导手册——从入门到精通",
    "filename": "DeepSeek 15天指导手册——从入门到精通.pdf",
    "source_url": "",
    "announce_id": ""
  },
  ...
]
```

**注意**：`parse_pdf.py` 的 `main()` 函数读取 `manifest.json` 获取 `filename` 和 `meta` 字段。只要提供 `filename`，解析流程即可正常工作。

### 3.3 创建 mydata/run_pipeline.py（核心文件）

这是整个方案的核心。该脚本作为统一入口，支持分步执行和一键执行。

**关键机制 —— Monkey-patch 路径变量**：

```python
# 在导入 src 模块前，先确保 src/ 在 sys.path 中
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 导入模块
import parse_pdf
import chunk_documents
import build_index
import rag_pipeline

# 动态修改各模块的全局路径变量，使其指向 mydata/
parse_pdf.RAW_DIR    = MYDATA_RAW_DIR
parse_pdf.PARSED_DIR = MYDATA_PARSED_DIR

chunk_documents.PARSED_DIR = MYDATA_PARSED_DIR
chunk_documents.CHUNKS_DIR = MYDATA_CHUNKS_DIR

build_index.BASE_DIR        = MYDATA_BASE_DIR
build_index.CHUNKS_DIR      = MYDATA_CHUNKS_DIR
build_index.VECTORSTORE_DIR = MYDATA_VECTORSTORE_DIR
build_index.CHUNKS_FILE     = MYDATA_CHUNKS_FILE

rag_pipeline.VECTORSTORE_DIR = MYDATA_VECTORSTORE_DIR
rag_pipeline.INDEX_PATH      = MYDATA_INDEX_PATH
rag_pipeline.META_PATH       = MYDATA_META_PATH
```

**脚本支持的命令**：

```bash
# 一键执行完整流程
python mydata/run_pipeline.py --step all

# 分步执行
python mydata/run_pipeline.py --step parse      # PDF → parsed JSON
python mydata/run_pipeline.py --step chunk      # parsed → chunks
python mydata/run_pipeline.py --step build      # chunks → FAISS 索引
python mydata/run_pipeline.py --step query      # 启动交互式问答

# 单次查询
python mydata/run_pipeline.py --step query --query "什么是DeepSeek"
```

**各 step 的实现逻辑**：

| Step | 调用方式 | 说明 |
|------|---------|------|
| `parse` | 直接调用 `parse_pdf.AnnualReportParser` | 遍历 manifest 中的 PDF，解析后保存到 `mydata/parsed/` |
| `chunk` | 直接调用 `chunk_documents.process_file()` | 对 `mydata/parsed/*.json` 执行语义分块，输出 `mydata/chunks/all_semantic.json` |
| `build` | 直接调用 `build_index.build_faiss_index()` | 加载 chunks 计算 embedding，构建 FAISS 索引到 `mydata/vectorstore/` |
| `query` | 创建 `rag_pipeline.RAGPipeline()` 实例 | 加载 mydata 的索引，启动交互式问答或单次查询 |

### 3.4 技术实现细节

#### 3.4.1 parse 步骤

```python
def step_parse():
    manifest_path = MYDATA_RAW_DIR.parent / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    
    for item in manifest:
        pdf_path = MYDATA_RAW_DIR / item["filename"]
        parser = parse_pdf.AnnualReportParser(pdf_path, meta=item)
        parser.parse()
        parser.save()   # save() 使用 parse_pdf.PARSED_DIR，已 monkey-patch
```

#### 3.4.2 chunk 步骤

```python
def step_chunk():
    parsed_files = list(MYDATA_PARSED_DIR.glob("*.json"))
    all_chunks = []
    for path in parsed_files:
        chunks = chunk_documents.process_file(path, strategy="semantic")
        all_chunks.extend(chunks)
    
    # 保存合并文件
    combined_path = MYDATA_CHUNKS_DIR / "all_semantic.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
```

#### 3.4.3 build 步骤

```python
def step_build():
    with open(MYDATA_CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    
    client = build_index.get_client()
    build_index.build_faiss_index(chunks, client)  # 使用已 patch 的 VECTORSTORE_DIR
```

#### 3.4.4 query 步骤

```python
def step_query(question=None):
    pipeline = rag_pipeline.RAGPipeline(use_bm25=True, use_rerank=False)
    
    if question:
        result = pipeline.query(question, verbose=True)
        print(result["answer"])
    else:
        # 交互式循环
        while True:
            q = input("问题：").strip()
            if q.lower() == "exit":
                break
            result = pipeline.query(q, verbose=True)
            print(result["answer"])
```

## 4. Assumptions & Decisions

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **不修改 src/ 文件** | 采用 monkey-patch 方案 | 保持原有年报系统完全可用，避免回归风险 |
| **使用原生版而非 LangChain 版** | 原生版 | 功能更完整（混合检索、Rerank、查询改写），且代码透明可调试 |
| **Embedding 方案** | 复用 DashScope text-embedding-v3 | 用户已有 API key，无需额外配置 |
| **分块策略** | semantic（默认） | 兼顾语义完整性和检索效果 |
| **manifest.json 格式** | 与现有格式兼容但字段简化 | 让 `parse_pdf.py` 的 `main()` 逻辑无需改动即可工作 |
| **是否支持 metadata 过滤** | 基础版暂不支持 | 用户的 PDF 没有 stock_code/year 等结构化元数据，过滤意义不大；如后续需要可扩展 |
| **是否支持 BM25** | 支持 | `RAGPipeline(use_bm25=True)` 直接可用，对关键词匹配有帮助 |
| **是否支持 HTTP 服务** | 第一阶段暂不实现 | 先保证 CLI 可用，后续可基于 `serve.py` 同理构建 |

## 5. Verification Steps

### 5.1 文件创建验证

```bash
# 确认目录结构
ls mydata/
# 预期输出: raw_pdf/ parsed/ chunks/ vectorstore/ manifest.json run_pipeline.py

# 确认 manifest.json 格式正确
python -c "import json; d=json.load(open('mydata/manifest.json')); print(f'共 {len(d)} 个PDF')"
```

### 5.2 parse 步骤验证

```bash
python mydata/run_pipeline.py --step parse
# 预期: mydata/parsed/ 下生成 4 个 .json 文件
ls mydata/parsed/
```

### 5.3 chunk 步骤验证

```bash
python mydata/run_pipeline.py --step chunk
# 预期: mydata/chunks/all_semantic.json 存在且包含 chunks
python -c "import json; d=json.load(open('mydata/chunks/all_semantic.json')); print(f'共 {len(d)} 个 chunks')"
```

### 5.4 build 步骤验证

```bash
python mydata/run_pipeline.py --step build
# 预期: mydata/vectorstore/ 下生成 faiss_index.bin 和 faiss_meta.json
ls mydata/vectorstore/
```

### 5.5 query 步骤验证

```bash
# 单次查询测试
python mydata/run_pipeline.py --step query --query "什么是DeepSeek"
# 预期: 返回基于 mydata PDF 内容的回答，而非年报内容

# 交互式测试
python mydata/run_pipeline.py --step query
# 输入: "以太坊的共识机制是什么"
# 预期: 返回基于以太坊白皮书的内容
```

### 5.6 隔离性验证

```bash
# 确认原有系统不受影响
python src/rag_pipeline.py --query "贵州茅台2023年营业收入"
# 预期: 仍返回年报内容，证明 mydata 知识库与原有系统完全隔离
```
