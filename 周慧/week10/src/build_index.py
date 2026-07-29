import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY = "semantic"
CHUNKS_FILE = CHUNKS_DIR / f"all_{STRATEGY}.json"

EMBED_MODEL = "BAAI/bge-large-zh"
EMBED_DIM = 1024


# ── 本地 Embedding 模型 ──────────────────────────────────────────────────────────


def load_embed_model() -> object:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise EnvironmentError(
            "请先安装 sentence-transformers: pip install sentence-transformers"
        )
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    logger.info(f"加载本地 embedding 模型: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    logger.info(f"模型加载完成，维度={EMBED_DIM}")
    return model


def embed_texts(
    model: object, texts: list[str], show_progress: bool = True
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    return embeddings


# ── FAISS 索引构建 ─────────────────────────────────────────────────────────────


def build_faiss_index(chunks: list[dict], model: object):
    import faiss

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding...")
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(model, texts)

    logger.info(f"构建 FAISS 索引，维度={EMBED_DIM}...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    meta_path = VECTORSTORE_DIR / "faiss_meta.json"

    faiss.write_index(index, str(index_path))
    logger.info(
        f"FAISS 索引已保存 → {index_path}  ({index_path.stat().st_size // 1024} KB)"
    )

    meta_list = [
        {
            "chunk_id": c["chunk_id"],
            "content": c["content"],
            "stock_code": c["metadata"].get("stock_code", ""),
            "year": c["metadata"].get("year", ""),
            "page_num": c["metadata"].get("page_num", -1),
            "section": c["metadata"].get("section", ""),
            "block_types": c["metadata"].get("block_types", []),
            "is_ocr": c["metadata"].get("is_ocr", False),
            "strategy": c["metadata"].get("strategy", ""),
            "source_file": c["metadata"].get("source_file", ""),
            "parent_content": c["metadata"].get("parent_content", ""),
            "parent_id": c["metadata"].get("parent_id", ""),
        }
        for c in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


# ── ChromaDB 索引构建（可选对比） ──────────────────────────────────────────────


def build_chroma_index(chunks: list[dict], model: object):
    try:
        import chromadb
    except ImportError:
        logger.error("请先安装 chromadb: pip install chromadb")
        return

    chroma_dir = VECTORSTORE_DIR / "chroma"
    client_db = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client_db.get_or_create_collection(
        name="annual_reports",
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"向 ChromaDB 写入 {len(chunks)} 条 chunk...")
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(model, texts)

    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        ids = [c["chunk_id"] for c in batch]
        docs = [c["content"] for c in batch]
        embs = embeddings[i : i + 100].tolist()
        metas = []
        for c in batch:
            m = dict(c["metadata"])
            m["block_types"] = ",".join(m.get("block_types") or [])
            m.pop("parent_content", None)
            metas.append(m)
        collection.add(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
        logger.info(f"  已写入 {min(i + 100, len(chunks))}/{len(chunks)}")

    logger.info(f"ChromaDB 写入完成，共 {collection.count()} 条")


# ── 主流程 ────────────────────────────────────────────────────────────────────


def main():
    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk_documents.py")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks（策略={STRATEGY}）")

    model = load_embed_model()

    build_faiss_index(chunks, model)

    # build_chroma_index(chunks, model)

    logger.info("\n索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")


if __name__ == "__main__":
    main()
