"""
MyData RAG 问答系统 — 统一入口

基于 rag_annual_report 原生版代码，为 mydata/ 下的自有 PDF 建立独立知识库。
核心机制：通过 Monkey-patch 修改 src/ 模块的全局路径变量，使其指向 mydata/ 目录，
从而零侵入复用现有核心逻辑。

使用方式：
  python mydata/run_pipeline.py --step all                    # 一键执行完整流程
  python mydata/run_pipeline.py --step parse                  # 仅 PDF 解析
  python mydata/run_pipeline.py --step chunk                  # 仅文档分块
  python mydata/run_pipeline.py --step build                  # 仅构建向量索引
  python mydata/run_pipeline.py --step query                  # 交互式问答
  python mydata/run_pipeline.py --step query --query "..."    # 单次查询
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
import site

# 尝试从 .env 文件加载环境变量
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                # 分割键值，并移除值中的注释部分
                key, value = line.split("=", 1)
                # 移除值中的 # 注释
                value = value.split("#")[0].strip()
                os.environ.setdefault(key.strip(), value.strip())
    logging.info(f"已从 {env_file} 加载环境变量")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"Python: {sys.executable}")
logger.info(f"Python version: {sys.version}")

user_site = site.getusersitepackages()
logger.info(f"User site-packages: {user_site}")
if user_site not in sys.path:
    sys.path.insert(0, user_site)
    logger.info(f"Added user site-packages to sys.path")

# ── 路径配置 ──────────────────────────────────────────────────────────────────

MYDATA_DIR     = PROJECT_ROOT / "mydata"
MYDATA_RAW_DIR = MYDATA_DIR / "raw_pdf"
MYDATA_PARSED_DIR = MYDATA_DIR / "parsed"
MYDATA_CHUNKS_DIR = MYDATA_DIR / "chunks"
MYDATA_VECTORSTORE_DIR = MYDATA_DIR / "vectorstore"

MYDATA_PARSED_DIR.mkdir(parents=True, exist_ok=True)
MYDATA_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
MYDATA_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ── Monkey-patch：修改 src/ 模块的路径变量 ───────────────────────────────────

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import parse_pdf
import chunk_documents
import build_index
import rag_pipeline

# parse_pdf
parse_pdf.RAW_DIR    = MYDATA_RAW_DIR
parse_pdf.PARSED_DIR = MYDATA_PARSED_DIR

# chunk_documents
chunk_documents.PARSED_DIR = MYDATA_PARSED_DIR
chunk_documents.CHUNKS_DIR = MYDATA_CHUNKS_DIR

# build_index
build_index.BASE_DIR        = MYDATA_DIR
build_index.CHUNKS_DIR      = MYDATA_CHUNKS_DIR
build_index.VECTORSTORE_DIR = MYDATA_VECTORSTORE_DIR
build_index.CHUNKS_FILE     = MYDATA_CHUNKS_DIR / f"all_{build_index.STRATEGY}.json"

# rag_pipeline
rag_pipeline.BASE_DIR        = MYDATA_DIR
rag_pipeline.VECTORSTORE_DIR = MYDATA_VECTORSTORE_DIR
rag_pipeline.INDEX_PATH      = MYDATA_VECTORSTORE_DIR / "faiss_index.bin"
rag_pipeline.META_PATH       = MYDATA_VECTORSTORE_DIR / "faiss_meta.json"

logger.info(f"MyData 路径已配置: {MYDATA_DIR}")


# ── 各步骤实现 ────────────────────────────────────────────────────────────────

def step_parse():
    """PDF 解析：mydata/raw_pdf/ → mydata/parsed/"""
    manifest_path = MYDATA_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"找不到清单文件: {manifest_path}")
        logger.info("请确保 mydata/manifest.json 存在且包含 PDF 文件信息")
        return False

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info(f"开始解析 {len(manifest)} 个 PDF 文件...")
    success_count = 0

    for item in manifest:
        pdf_path = MYDATA_RAW_DIR / item["filename"]
        if not pdf_path.exists():
            logger.warning(f"文件不存在，跳过: {pdf_path}")
            continue

        try:
            parser = parse_pdf.AnnualReportParser(pdf_path, meta=item)
            parser.parse()
            parser.save()
            success_count += 1
        except Exception as e:
            logger.error(f"解析失败 {pdf_path.name}: {e}")

    logger.info(f"解析完成: {success_count}/{len(manifest)} 个文件成功 → {MYDATA_PARSED_DIR}")
    return success_count > 0


def step_chunk():
    """文档分块：mydata/parsed/ → mydata/chunks/"""
    parsed_files = list(MYDATA_PARSED_DIR.glob("*.json"))
    if not parsed_files:
        logger.error(f"没有找到解析结果，请先运行 --step parse")
        return False

    logger.info(f"开始分块: {len(parsed_files)} 个 parsed 文件...")
    all_chunks = []

    for path in sorted(parsed_files):
        try:
            chunks = chunk_documents.process_file(path, strategy=chunk_documents.STRATEGY)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"分块失败 {path.name}: {e}")

    # 保存合并文件
    combined_path = MYDATA_CHUNKS_DIR / f"all_{chunk_documents.STRATEGY}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"分块完成: 共 {len(all_chunks)} 个 chunk，平均 {avg_len:.0f} 字符 → {combined_path}")
    return True


def step_build(embedding_mode="bge"):
    """构建向量索引：mydata/chunks/ → mydata/vectorstore/"""
    chunks_file = build_index.CHUNKS_FILE
    if not chunks_file.exists():
        logger.error(f"找不到 {chunks_file}，请先运行 --step chunk")
        return False

    with open(chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks，开始构建索引...")

    try:
        build_index.build_faiss_index(chunks, embedding_mode=embedding_mode)
        logger.info("索引构建完成!")
        return True
    except Exception as e:
        logger.error(f"构建索引失败: {e}")
        return False


def step_query(question: str = None):
    """问答查询"""
    if not rag_pipeline.INDEX_PATH.exists():
        logger.error(f"找不到索引文件: {rag_pipeline.INDEX_PATH}")
        logger.info("请先运行 --step build 构建向量索引")
        return False

    llm_provider = os.getenv("LLM_PROVIDER", "deepseek")
    embedding_mode = os.getenv("EMBEDDING_MODE", "bge")
    
    # 显示 API Key 状态（不显示完整 key）
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
    logger.info(f"LLM Provider: {llm_provider}")
    logger.info(f"Embedding Mode: {embedding_mode}")
    if deepseek_key:
        logger.info(f"DeepSeek API Key: {deepseek_key[:8]}...{deepseek_key[-4:]}")
    if dashscope_key:
        logger.info(f"DashScope API Key: {dashscope_key[:8]}...{dashscope_key[-4:]}")
    
    try:
        pipeline = rag_pipeline.RAGPipeline(
            use_bm25=True,
            use_rerank=False,
            use_query_rewrite=False,
            llm_provider=llm_provider,
            embedding_mode=embedding_mode,
        )
        logger.info("RAG Pipeline 初始化成功")
    except Exception as e:
        logger.error(f"Pipeline 初始化失败: {e}")
        return False

    def print_result(q: str, result: dict):
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        if result.get("citations"):
            print("\n── 来源 ──")
            for c in result["citations"]:
                print(f"  {c['source']}")

    if question:
        result = pipeline.query(question, verbose=True)
        print_result(question, result)
    else:
        print(f"\nMyData RAG 问答系统")
        print(f"LLM：{pipeline.llm_model}  |  Embedding：{pipeline.embedding_mode}")
        print(f"索引：{rag_pipeline.INDEX_PATH}")
        print("输入 'exit' 退出，'mode' 查看配置\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            if q.lower() == "mode":
                print(f"BM25={'on' if pipeline.use_bm25 else 'off'}  "
                      f"Rerank={'on' if pipeline.use_rerank else 'off'}")
                continue
            result = pipeline.query(q, verbose=True)
            print_result(q, result)

    return True


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run_all():
    """一键执行完整流程"""
    if not step_parse():
        logger.error("解析步骤失败，终止流程")
        return
    if not step_chunk():
        logger.error("分块步骤失败，终止流程")
        return
    if not step_build():
        logger.error("建索引步骤失败，终止流程")
        return
    logger.info("\n✅ 完整流程执行完毕！可以运行 --step query 开始问答")


def main():
    parser = argparse.ArgumentParser(description="MyData RAG 问答系统")
    parser.add_argument(
        "--step",
        choices=["parse", "chunk", "build", "query", "all"],
        required=True,
        help="执行步骤",
    )
    parser.add_argument("--query", type=str, default=None, help="单次查询内容（仅在 --step query 时有效）")
    args = parser.parse_args()

    if args.step == "parse":
        step_parse()
    elif args.step == "chunk":
        step_chunk()
    elif args.step == "build":
        step_build()
    elif args.step == "query":
        step_query(args.query)
    elif args.step == "all":
        run_all()


if __name__ == "__main__":
    main()
