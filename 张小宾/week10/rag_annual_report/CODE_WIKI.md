# 上市公司年报 RAG 问答系统 — Code Wiki

> 文档版本：v1.0 | 最后更新：2026-07-08

---

## 目录

1. [项目概览](#一项目概览)
2. [项目架构](#二项目架构)
3. [目录结构](#三目录结构)
4. [核心模块详解](#四核心模块详解)
5. [关键类与函数](#五关键类与函数)
6. [数据流程](#六数据流程)
7. [依赖关系](#七依赖关系)
8. [运行方式](#八运行方式)
9. [评估体系](#九评估体系)
10. [配置参数](#十配置参数)

---

## 一、项目概览

### 1.1 项目定位

本项目以**上市公司年度报告智能问答**为场景，构建一套接近企业级落地标准的 **RAG（检索增强生成）系统**。

### 1.2 核心目标

| 目标 | 描述 |
|------|------|
| **数据覆盖** | 5家公司 × 3年 = 15份年报PDF，总量约85MB |
| **行业覆盖** | 消费品（茅台、五粮液）、金融（中国平安）、新能源（宁德时代）、科技（海康威视） |
| **教学价值** | 提供原生版和LangChain版两套实现，便于对比学习 |

### 1.3 两套实现对比

| 实现版本 | 定位 | 代码量 | 关键特征 |
|----------|------|--------|---------|
| **原生版** (`src/`) | 企业级生产参考 | ~700行 | 手动控制每个环节，混合检索+Rerank |
| **LangChain版** (`src_langchain/`) | 框架快速原型参考 | ~200行 | LCEL链路设计，代码简洁 |

---

## 二、项目架构

### 2.1 整体流水线

```
原始 PDF
    │
    ▼ download_reports.py
数据获取（巨潮资讯API）
    │
    ▼ parse_pdf.py
PDF解析（文字+表格+OCR+章节结构）
    │
    ▼ chunk_documents.py
文档分块（三种策略可切换）
    │
    ▼ build_index.py / build_index_lc.py
向量化 + 索引构建（DashScope API / 本地BGE）
    │
    ▼ rag_pipeline.py / rag_chain_lc.py
问答流水线：查询 → 检索 → 重排 → 生成
    │
    ▼ evaluation/
评估（RAGAS四项指标 + 消融实验）
```

### 2.2 技术选型

| 环节 | 技术方案 | 选型原因 |
|------|---------|---------|
| **数据来源** | 巨潮资讯网API | 证监会指定平台，数据公开合法 |
| **PDF解析** | pdfplumber + PyMuPDF + pytesseract | 组合策略应对复杂报表 |
| **文档分块** | 语义分块（默认） | 保留章节边界，语义完整性好 |
| **Embedding** | DashScope text-embedding-v3 / BGE-small-zh-v1.5 | 云端API vs 本地离线 |
| **向量库** | FAISS IndexFlatIP | 精确内积检索，数据量<10万时速度足够 |
| **关键词检索** | jieba + BM25Okapi | 精确匹配数字、专有名词 |
| **混合融合** | RRF（Reciprocal Rank Fusion） | 互补向量和BM25的盲区 |
| **Rerank** | BGE-reranker-base（可选） | CrossEncoder精排，提升准确率 |
| **LLM** | DashScope qwen-plus | 中文能力强，成本低 |
| **评估** | RAGAS | 自动化评估四项核心指标 |

---

## 三、目录结构

```
rag_annual_report/
├── src/                              # 原生版（DashScope API）
│   ├── download_reports.py           # 巨潮API下载PDF
│   ├── parse_pdf.py                  # PDF → 结构化JSON blocks
│   ├── chunk_documents.py            # blocks → chunks（三种策略）
│   ├── build_index.py                # chunks → FAISS向量索引
│   ├── rag_pipeline.py               # 问答流水线（BM25+向量+RRF+Rerank+LLM）
│   ├── serve.py                      # FastAPI HTTP服务
│   └── static/
│       └── index.html                # 教学可视化Web页面
│
├── src_langchain/                    # LangChain版（本地BGE + DashScope LLM）
│   ├── download_model.py             # BGE模型下载到models/目录
│   ├── build_index_lc.py             # PDF → FAISS（LangChain链路）
│   └── rag_chain_lc.py               # LCEL RAG链
│
├── evaluation/                       # 评估模块
│   ├── questions.json                # 20道标准测试题+ground truth
│   ├── evaluate.py                   # RAGAS四项指标自动评估
│   ├── compare_strategies.py         # 消融实验（策略×检索方式）
│   └── results/                      # 评估结果输出
│
├── data/                             # 数据目录
│   ├── raw_pdf/                      # 15份年报PDF（85MB）
│   ├── manifest.json                 # PDF元数据索引
│   ├── parsed/                       # 解析后的JSON（每份PDF一个）
│   └── chunks/                       # 分块后的JSON
│
├── vectorstore/                      # 向量存储
│   ├── faiss_index.bin               # 原生版索引（41MB）
│   ├── faiss_meta.json               # 元数据（15MB）
│   └── faiss_lc/                     # LangChain版索引（21MB）
│
├── models/                           # 本地模型
│   └── bge-small-zh-v1.5/            # BGE模型（~90MB）
│
├── requirements.txt                  # 依赖清单
├── ARCHITECTURE.md                   # 技术架构说明
├── USAGE_GUIDE.md                    # 代码调用与测试指南
├── PROJECT_LOG.md                    # 开发日志
└── CODE_WIKI.md                      # 本文档
```

---

## 四、核心模块详解

### 4.1 数据下载模块（download_reports.py）

**职责**：从巨潮资讯网批量下载上市公司年报PDF

**核心流程**：
1. 遍历目标股票列表（5家公司 × 3年）
2. 使用两种搜索策略查询年报：
   - 策略1：公司名+年份关键词（如"贵州茅台2023年年度报告"）
   - 策略2：字间加空格（应对"五 粮 液"存储格式）
3. 过滤非完整年报（排除摘要、英文版）
4. 下载PDF到`data/raw_pdf/`
5. 生成`manifest.json`记录元数据

**关键配置**：
- `TARGET_STOCKS`：目标股票列表（代码、板块、简称）
- `TARGET_YEARS`：目标年份列表
- `CNINFO_QUERY_URL`：巨潮查询API地址

**输出**：`data/raw_pdf/*.pdf` + `data/manifest.json`

### 4.2 PDF解析模块（parse_pdf.py）

**职责**：将原始年报PDF转换为结构化文本

**技术策略**：
- **表格提取**：pdfplumber（算法更准）
- **文字+字体信息**：PyMuPDF(fitz)（识别标题层级）
- **扫描页处理**：pytesseract（OCR降级）

**输出块类型**：
| 块类型 | 判断依据 | 处理方式 |
|--------|---------|---------|
| `title` | 字体≥14pt或加粗且行长<50字 | 单独成块，更新章节栈 |
| `table` | pdfplumber提取 | 转为Markdown格式 |
| `text` | 普通文本段落 | 累积到缓冲区 |

**数据结构**：`ParsedBlock` dataclass
- `block_type`: str（text/table/title）
- `content`: str（文字内容）
- `page_num`: int（页码）
- `section_path`: list[str]（章节路径栈）
- `is_ocr`: bool（是否经过OCR）

**输出**：`data/parsed/*.json`

### 4.3 文档分块模块（chunk_documents.py）

**职责**：对解析后的年报做分块处理

**三种分块策略**：

| 策略 | 切割依据 | chunk大小 | 优点 | 缺点 |
|------|---------|-----------|------|------|
| **fixed** | 每500字符截断，overlap=50 | 均匀 | 简单可预测 | 切断句子/表格 |
| **semantic**（默认） | 遇标题强制切，段落合并不超800字 | 不均匀 | 语义完整，边界清晰 | 块大小差异大 |
| **hierarchical** | 父块（完整章节~2000字）+子块（~400字） | 双层 | 小to大检索，精确召回 | 实现复杂 |

**chunk数据结构**：
```json
{
  "chunk_id": "600519_2023_0423",
  "content": "报告期内，公司实现营业总收入1,476.94亿元...",
  "metadata": {
    "stock_code": "600519",
    "year": "2023",
    "page_num": 56,
    "section": "第十节 > 二、财务报表 > 利润表",
    "block_types": ["text"],
    "is_ocr": false,
    "strategy": "semantic",
    "source_file": "600519_2023_贵州茅台_贵州茅台2023年年度报告.pdf"
  }
}
```

**输出**：`data/chunks/all_{strategy}.json`

### 4.4 向量索引构建模块（build_index.py）

**职责**：计算chunk embedding并构建FAISS向量索引

**核心流程**：
1. 加载分块结果
2. 按批次（10条/批）调用DashScope text-embedding-v3
3. L2归一化（使内积等价于余弦相似度）
4. 构建FAISS IndexFlatIP
5. 持久化索引和元数据

**配置参数**：
- `EMBED_MODEL`: "text-embedding-v3"
- `EMBED_DIM`: 1024
- `BATCH_SIZE`: 10（API限制）

**输出**：
- `vectorstore/faiss_index.bin`（41MB）
- `vectorstore/faiss_meta.json`（15MB）

### 4.5 RAG问答流水线（rag_pipeline.py）

**职责**：完整的问答流水线实现

**流水线流程**：
```
用户输入
    │
    ├─ [可选] 查询改写（qwen-turbo）
    │
    ├─ 向量检索（DashScope + FAISS）→ top-10
    │
    ├─ BM25关键词检索（jieba + rank_bm25）→ top-10
    │
    ├─ RRF融合 → top-19（去重后）
    │
    ├─ [可选] CrossEncoder Rerank → top-4
    │
    ├─ 相关性阈值检查（<0.25拒绝回答）
    │
    └─ LLM生成（qwen-plus）+ 引用标注
```

**核心组件**：
- `VectorStore`: 向量检索封装
- `BM25Store`: 关键词检索封装
- `RAGPipeline`: 完整流水线入口

**输出结构**：
```python
{
    "answer": "贵州茅台2023年营业收入为...",
    "citations": [
        {"index": 1, "source": "600519 2023年报 · ...", "chunk_id": "..."},
        ...
    ],
    "retrieved": [  # 完整chunk列表
        {"content": "...", "stock_code": "600519", ...},
        ...
    ]
}
```

### 4.6 HTTP服务模块（serve.py）

**职责**：提供RESTful API服务

**接口列表**：

| 接口 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 返回教学可视化页面 |
| `/health` | GET | 健康检查 |
| `/query` | POST | 标准问答接口 |
| `/query/debug` | POST | 教学调试接口（逐步返回中间结果） |

**请求/响应示例**：

```json
// POST /query 请求
{
  "question": "贵州茅台2023年营业收入是多少",
  "stock_code": "600519",
  "year": "2023"
}

// 响应
{
  "answer": "贵州茅台2023年营业收入为人民币14,769,360.50万元[2]。",
  "citations": [
    {"index": 2, "source": "600519 2023年报 · 第十节 · 第56页", "chunk_id": "..."}
  ]
}
```

### 4.7 LangChain版模块（src_langchain/）

**职责**：使用LangChain框架实现RAG链

**核心文件**：

| 文件 | 功能 |
|------|------|
| `download_model.py` | 下载BGE模型到项目目录 |
| `build_index_lc.py` | 构建LangChain版FAISS索引 |
| `rag_chain_lc.py` | LCEL链式问答实现 |

**LCEL链结构**：
```python
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

**与原生版对比**：

| 环节 | 原生版 | LangChain版 |
|------|--------|------------|
| 检索 | FAISS+BM25混合 | FAISS单路 |
| 排序 | RRF+CrossEncoder | 相似度直接排序 |
| 链路组织 | 手写流程控制 | LCEL pipe操作符 |
| 代码量 | ~300行 | ~120行 |

---

## 五、关键类与函数

### 5.1 download_reports.py

| 函数名 | 签名 | 功能 |
|--------|------|------|
| `query_annual_reports()` | `(stock_code, plate, company_name, year) -> list[dict]` | 查询年报，支持两种搜索策略 |
| `download_pdf()` | `(pdf_url, save_path) -> bool` | 下载PDF，带重试机制 |
| `sanitize()` | `(name) -> str` | 文件名清洗（移除非法字符） |

### 5.2 parse_pdf.py

| 类/函数 | 功能 |
|---------|------|
| `ParsedBlock` | dataclass，解析块数据结构 |
| `is_noise_line()` | 判断是否为页眉/页脚噪声 |
| `is_title_line()` | 判断是否为标题行 |
| `table_to_markdown()` | 将表格转为Markdown格式 |
| `detect_if_scanned()` | 启发式判断是否为扫描页 |
| `ocr_page()` | 对扫描页做OCR |
| `AnnualReportParser` | PDF解析器主类 |
| `AnnualReportParser.parse()` | 执行解析，返回blocks列表 |
| `AnnualReportParser.save()` | 保存解析结果为JSON |

### 5.3 chunk_documents.py

| 函数名 | 签名 | 功能 |
|--------|------|------|
| `chunk_fixed()` | `(text, chunk_size=500, overlap=50) -> Iterator[str]` | 固定大小分块 |
| `chunk_semantic()` | `(blocks, max_chunk_size=800, min_chunk_size=100) -> Iterator[dict]` | 语义分块（默认策略） |
| `chunk_hierarchical()` | `(blocks, parent_size=2000, child_size=400, overlap=50) -> Iterator[dict]` | 层级分块 |
| `process_file()` | `(parsed_path, strategy) -> list[dict]` | 处理单个解析文件 |

### 5.4 build_index.py

| 函数名 | 签名 | 功能 |
|--------|------|------|
| `get_client()` | `() -> OpenAI` | 获取DashScope客户端 |
| `embed_texts()` | `(client, texts, show_progress=True) -> np.ndarray` | 批量计算embedding |
| `build_faiss_index()` | `(chunks, client) -> tuple[index, meta_list]` | 构建FAISS索引 |
| `build_chroma_index()` | `(chunks, client) -> None` | 构建ChromaDB索引（可选） |

### 5.5 rag_pipeline.py

| 类/函数 | 功能 |
|---------|------|
| `VectorStore` | 向量检索封装 |
| `VectorStore.search()` | 向量检索，支持元数据过滤 |
| `BM25Store` | BM25关键词检索封装 |
| `BM25Store.search()` | BM25检索 |
| `reciprocal_rank_fusion()` | RRF混合融合算法 |
| `rerank()` | CrossEncoder精排 |
| `rewrite_query()` | 查询改写 |
| `build_context()` | 组装LLM上下文 |
| `call_llm()` | 调用LLM生成答案 |
| `RAGPipeline` | 完整RAG流水线 |
| `RAGPipeline.query()` | 执行问答查询 |

### 5.6 serve.py

| 类/函数 | 功能 |
|---------|------|
| `QueryRequest` | Pydantic模型，查询请求结构 |
| `QueryResponse` | Pydantic模型，标准响应结构 |
| `DebugResponse` | Pydantic模型，调试响应结构 |
| `index()` | 返回可视化页面 |
| `health()` | 健康检查 |
| `query()` | 标准问答接口 |
| `query_debug()` | 调试接口（逐步返回中间结果） |

### 5.7 src_langchain/

| 文件 | 关键函数 | 功能 |
|------|---------|------|
| `download_model.py` | `download()` | 下载BGE模型到本地 |
| | `verify()` | 验证模型可用性 |
| `build_index_lc.py` | `load_documents()` | 加载PDF文档 |
| | `split_documents()` | 文本分块 |
| | `get_embeddings()` | 获取BGE嵌入模型 |
| | `build_vectorstore()` | 构建FAISS向量库 |
| `rag_chain_lc.py` | `get_llm()` | 获取LLM实例 |
| | `get_vectorstore()` | 加载向量库 |
| | `build_chain()` | 构建LCEL链 |
| | `build_chain_with_sources()` | 构建带来源的链 |

### 5.8 evaluation/

| 文件 | 关键函数 | 功能 |
|------|---------|------|
| `evaluate.py` | `load_questions()` | 加载测试题集 |
| | `run_native_rag()` | 运行原生版RAG |
| | `run_langchain_rag()` | 运行LangChain版RAG |
| | `run_ragas_eval()` | RAGAS评估 |
| | `analyze_by_type()` | 按题型统计 |
| | `print_comparison()` | 两版对比输出 |
| `compare_strategies.py` | `load_ablation_questions()` | 加载消融实验题 |
| | `load_index()` | 加载指定策略的索引 |
| | `retrieve_vector()` | 向量检索 |
| | `retrieve_bm25()` | BM25检索 |
| | `retrieve_hybrid()` | RRF混合检索 |
| | `compute_metrics()` | 计算检索指标 |

---

## 六、数据流程

### 6.1 数据处理流程

```
阶段1: 数据获取
┌─────────────────────────────────────────────────────────┐
│ 巨潮资讯网API ──→ download_reports.py ──→ data/raw_pdf/ │
│                                              │         │
│                                              ▼         │
│                                       data/manifest.json│
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
阶段2: PDF解析
┌─────────────────────────────────────────────────────────┐
│ data/raw_pdf/ ──→ parse_pdf.py ──→ data/parsed/        │
│   (*.pdf)            │               (*.json)           │
│                      ├─ pdfplumber（表格）              │
│                      ├─ PyMuPDF（文字+字体）            │
│                      └─ pytesseract（OCR扫描页）        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
阶段3: 文档分块
┌─────────────────────────────────────────────────────────┐
│ data/parsed/ ──→ chunk_documents.py ──→ data/chunks/   │
│   (*.json)            │               (all_*.json)      │
│                      ├─ fixed（固定大小）               │
│                      ├─ semantic（语义分块）← 默认      │
│                      └─ hierarchical（层级分块）        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
阶段4: 向量索引
┌─────────────────────────────────────────────────────────┐
│ data/chunks/ ──→ build_index.py ──→ vectorstore/       │
│   (all_*.json)        │               faiss_index.bin   │
│                      ├─ DashScope Embedding API         │
│                      └─ FAISS IndexFlatIP               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
阶段5: 问答服务
┌─────────────────────────────────────────────────────────┐
│ 用户查询 ──→ rag_pipeline.py / serve.py ──→ 答案+引用   │
│            │                                            │
│            ├─ VectorStore（FAISS）                      │
│            ├─ BM25Store（jieba+BM25）                  │
│            ├─ RRF融合                                   │
│            ├─ CrossEncoder Rerank（可选）              │
│            └─ qwen-plus LLM                            │
└─────────────────────────────────────────────────────────┘
```

### 6.2 问答流水线数据流

```
用户输入: "贵州茅台2023年营业收入是多少"
            │
            ├─ [可选] 查询改写
            │         ↓
            │   "贵州茅台2023年营业收入"
            │
            ├─ 向量检索（DashScope + FAISS）
            │         ↓
            │   [{chunk_id, content, vec_score=0.85}, ...]
            │
            ├─ BM25检索（jieba + rank_bm25）
            │         ↓
            │   [{chunk_id, content, bm25_score=12.3}, ...]
            │
            ├─ RRF融合
            │         ↓
            │   [{chunk_id, content, rrf_score=0.028}, ...]
            │
            ├─ [可选] CrossEncoder Rerank
            │         ↓
            │   [{chunk_id, content, rerank_score=8.9}, ...]
            │
            ├─ 阈值检查（>0.25）
            │         ↓
            │   通过（继续）/ 拒绝回答
            │
            ├─ 组装上下文
            │         ↓
            │   "[1] 600519 2023年报 · 第56页\n内容..."
            │
            └─ LLM生成（qwen-plus）
                      ↓
                "贵州茅台2023年营业收入为人民币14,769,360.50万元[1]。"
```

---

## 七、依赖关系

### 7.1 核心依赖清单

| 类别 | 依赖包 | 版本 | 用途 |
|------|--------|------|------|
| **PDF解析** | pdfplumber | >=0.10.0 | 表格提取 |
| | pymupdf (fitz) | >=1.23.0 | 文字+字体信息 |
| | pytesseract | >=0.3.10 | OCR扫描页 |
| | Pillow | >=10.0.0 | OCR图像处理 |
| **Embedding** | sentence-transformers | >=2.6.0 | 本地BGE/Reranker |
| | huggingface_hub | >=0.23.0 | 模型下载 |
| **向量检索** | faiss-cpu | >=1.7.4 | 向量索引 |
| **关键词检索** | rank_bm25 | >=0.2.2 | BM25评分 |
| | jieba | >=0.42.1 | 中文分词 |
| **LLM接口** | openai | >=1.30.0 | DashScope/OpenAI兼容 |
| **LangChain** | langchain | >=0.3.0 | 框架核心 |
| | langchain-openai | >=0.2.0 | OpenAI集成 |
| | langchain-community | >=0.3.0 | 社区组件 |
| | langchain-huggingface | >=0.1.0 | HuggingFace集成 |
| **评估** | ragas | >=0.2.0 | RAG评估框架 |
| | datasets | >=2.18.0 | 数据集处理 |
| **HTTP服务** | fastapi | >=0.110.0 | Web框架 |
| | uvicorn | >=0.29.0 | ASGI服务器 |
| **工具** | numpy | >=1.24.0 | 数值计算 |
| | requests | >=2.31.0 | HTTP请求 |
| | tqdm | >=4.66.0 | 进度条 |
| | pandas | >=2.0.0 | 数据处理 |

### 7.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                        顶层入口                                     │
│              serve.py        rag_pipeline.py                        │
│                   │                 │                               │
│                   └────────┬────────┘                               │
│                            ▼                                        │
│              ┌────────────────────────┐                             │
│              │        VectorStore      │ ← FAISS, DashScope Embedding│
│              │        BM25Store        │ ← jieba, rank_bm25         │
│              │        reciprocal_rank_fusion                        │
│              │        rerank           │ ← CrossEncoder             │
│              │        call_llm         │ ← DashScope LLM            │
│              └────────────────────────┘                             │
│                            │                                        │
│                            ▼                                        │
│              vectorstore/faiss_index.bin + faiss_meta.json          │
│                            │                                        │
│                            ▼                                        │
│              build_index.py ← data/chunks/all_semantic.json         │
│                            │                                        │
│                            ▼                                        │
│              chunk_documents.py ← data/parsed/*.json                │
│                            │                                        │
│                            ▼                                        │
│              parse_pdf.py ← data/raw_pdf/*.pdf                      │
│                            │                                        │
│                            ▼                                        │
│              download_reports.py ← 巨潮资讯API                       │
│                                                                     │
│              ┌─────────────────────────────────────────────────┐    │
│              │           LangChain 版                          │    │
│              │   build_index_lc.py  ←  PyMuPDFLoader           │    │
│              │                           HuggingFaceEmbeddings │    │
│              │                           FAISS                 │    │
│              │                                                 │    │
│              │   rag_chain_lc.py  ←  LCEL Chain                │    │
│              │                           ChatOpenAI             │    │
│              └─────────────────────────────────────────────────┘    │
│                                                                     │
│              ┌─────────────────────────────────────────────────┐    │
│              │              评估模块                           │    │
│              │   evaluate.py  ←  ragas, datasets               │    │
│              │   compare_strategies.py  ←  FAISS, BM25        │    │
│              └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 八、运行方式

### 8.1 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API Key（必需）
export DASHSCOPE_API_KEY="sk-xxx"

# 下载本地BGE模型（LangChain版需要）
python src_langchain/download_model.py
```

### 8.2 原生版完整流程

```bash
# 步骤1：下载年报PDF
python src/download_reports.py

# 步骤2：解析PDF
python src/parse_pdf.py

# 步骤3：文档分块（修改STRATEGY变量切换策略）
python src/chunk_documents.py

# 步骤4：构建向量索引
python src/build_index.py

# 步骤5：问答
python src/rag_pipeline.py                      # 交互式
python src/rag_pipeline.py --query "茅台2023营收"  # 单次查询
python src/rag_pipeline.py --query "..." --stock 600519 --year 2023  # 带过滤
python src/rag_pipeline.py --query "..." --query-rewrite  # 开启查询改写
```

### 8.3 LangChain版完整流程

```bash
# 步骤1：构建向量索引
python src_langchain/build_index_lc.py

# 步骤2：问答
python src_langchain/rag_chain_lc.py              # 交互式
echo "茅台2023营收" | python src_langchain/rag_chain_lc.py  # 单次查询
```

### 8.4 HTTP服务模式

```bash
cd src
uvicorn serve:app --host 0.0.0.0 --port 8000

# 开发模式（自动重载）
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

# 调用示例
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "贵州茅台2023年营业收入是多少"}'
```

### 8.5 评估流程

```bash
# RAGAS评估
python evaluation/evaluate.py --pipeline native    # 评估原生版
python evaluation/evaluate.py --pipeline langchain # 评估LangChain版
python evaluation/evaluate.py --pipeline both      # 两版对比

# 消融实验
python evaluation/compare_strategies.py
python evaluation/compare_strategies.py --strategies semantic,hierarchical --modes vector_only,hybrid
```

---

## 九、评估体系

### 9.1 评测题集（20题）

| 题型 | 题数 | 考察能力 | 示例 |
|------|------|---------|------|
| `simple_fact` | 5 | 基础检索 | 茅台2023年营业收入？ |
| `precise_number` | 5 | BM25数字召回 | 宁德时代2021研发费用占比？ |
| `cross_doc_compare` | 4 | 多文档综合 | 茅台vs五粮液2022毛利率对比 |
| `time_trend` | 3 | 跨年版本整合 | 茅台2021-2023营收趋势 |
| `should_refuse` | 3 | 幻觉控制 | 茅台股价是多少（应拒绝）|

### 9.2 RAGAS四项指标

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **Faithfulness** | LLM判断 | 答案是否完全来自检索内容（无幻觉） |
| **Answer Relevancy** | Embedding相似度 | 答案是否切题 |
| **Context Precision** | LLM判断 | 检索内容中有用比例 |
| **Context Recall** | LLM判断 | 需要的内容被检索到了多少 |

### 9.3 消融实验矩阵

```
分块策略（3）× 检索方式（3）= 9种组合
  策略: fixed / semantic / hierarchical
  检索: vector_only / bm25_only / hybrid
```

**指标**：
- **Hit Rate @k**：前k个召回结果中是否包含目标文档
- **MRR**（Mean Reciprocal Rank）：第一个命中的平均倒数排名

---

## 十、配置参数

### 10.1 原生版关键参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `STRATEGY` | chunk_documents.py | "semantic" | 分块策略 |
| `EMBED_MODEL` | build_index.py | "text-embedding-v3" | 嵌入模型 |
| `EMBED_DIM` | build_index.py | 1024 | 向量维度 |
| `BATCH_SIZE` | build_index.py | 10 | API批处理大小 |
| `TOP_K_RETRIEVE` | rag_pipeline.py | 10 | 初始召回数 |
| `TOP_K_RERANK` | rag_pipeline.py | 4 | 最终保留数 |
| `SCORE_THRESHOLD` | rag_pipeline.py | 0.25 | 相关性阈值 |
| `LLM_MODEL` | rag_pipeline.py | "qwen-plus" | LLM模型 |

### 10.2 环境变量

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `DASHSCOPE_API_KEY` | 是 | DashScope API密钥 |
| `HF_ENDPOINT` | 否 | HuggingFace镜像地址 |

### 10.3 命令行参数

**rag_pipeline.py**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `--query` | str | 查询问题 |
| `--stock` | str | 股票代码过滤 |
| `--year` | str | 年份过滤 |
| `--query-rewrite` | flag | 开启查询改写 |
| `--no-bm25` | flag | 关闭BM25检索 |
| `--no-rerank` | flag | 关闭Rerank |

**evaluate.py**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `--pipeline` | native/langchain/both | 评估版本 |
| `--question-ids` | str | 指定题号 |
| `--skip-ragas` | flag | 跳过RAGAS打分 |

**compare_strategies.py**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `--strategies` | str | 分块策略列表 |
| `--modes` | str | 检索方式列表 |
| `--top-k` | int | 召回数量 |

---

## 附录：关键文件路径索引

| 文件 | 绝对路径 |
|------|---------|
| 项目根目录 | `/Users/zhangxiaobin/Documents/badou_ai_project/hub-ocSF/张小宾/week10/rag_annual_report/` |
| 原生版源码 | `/Users/zhangxiaobin/Documents/badou_ai_project/hub-ocSF/张小宾/week10/rag_annual_report/src/` |
| LangChain版源码 | `/Users/zhangxiaobin/Documents/badou_ai_project/hub-ocSF/张小宾/week10/rag_annual_report/src_langchain/` |
| 评估模块 | `/Users/zhangxiaobin/Documents/badou_ai_project/hub-ocSF/张小宾/week10/rag_annual_report/evaluation/` |
| 数据目录 | `/Users/zhangxiaobin/Documents/badou_ai_project/hub-ocSF/张小宾/week10/rag_annual_report/data/` |
| 向量存储 | `/Users/zhangxiaobin/Documents/badou_ai_project/hub-ocSF/张小宾/week10/rag_annual_report/vectorstore/` |
| 本地模型 | `/Users/zhangxiaobin/Documents/badou_ai_project/hub-ocSF/张小宾/week10/rag_annual_report/models/` |

---

*文档生成完成*
