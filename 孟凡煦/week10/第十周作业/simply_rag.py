"""
简单 RAG 问答系统

功能：
  1. 加载本地 PDF
  2. 文本分块
  3. 构建向量索引（本地 BGE 模型，无需 API）
  4. 问答（使用 DeepSeek API）

使用方法：
  1. 将 PDF 放入 pdf_docs/ 目录
  2. 设置 API Key: set DEEPSEEK_API_KEY=sk-xxx
  3. 首次运行: python simple_rag.py --build
  4. 问答: python simple_rag.py --query "你的问题"

依赖：
  pip install langchain langchain-openai langchain-community langchain-huggingface faiss-cpu pymupdf sentence-transformers
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 配置
PDF_DIR = Path("pdf_docs")
VECTORSTORE_DIR = Path("vectorstore/simple_faiss")
MODELS_DIR = Path("models")
BGE_MODEL_PATH = MODELS_DIR / "bge-small-zh-v1.5"

DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """你是一个专业的文档问答助手。

回答规则：
1. 只根据【参考资料】中的内容回答
2. 若参考资料不足以回答，直接说"根据提供的资料无法回答"
3. 引用具体信息时标注来源
4. 回答简洁准确"""


def get_llm():
    """获取 LLM（DeepSeek）"""
    from langchain_openai import ChatOpenAI
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
    
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        openai_api_key=api_key,
        openai_api_base=DEEPSEEK_URL,
        temperature=0.1,
    )


def get_embeddings():
    """获取本地 Embedding 模型（BGE）"""
    from langchain_huggingface import HuggingFaceEmbeddings
    
    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    return HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pdfs():
    """加载 PDF 文件"""
    from langchain_community.document_loaders import PyMuPDFLoader
    
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"目录不存在: {PDF_DIR}，请创建并放入 PDF 文件")
    
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"{PDF_DIR} 中没有 PDF 文件")
    
    all_docs = []
    for pdf_path in sorted(pdf_files):
        logger.info(f"加载: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        all_docs.extend(docs)
        logger.info(f"  → {len(docs)} 页")
    
    logger.info(f"共加载 {len(all_docs)} 页")
    return all_docs


def split_documents(docs):
    """文档分块"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"分割为 {len(chunks)} 个 chunks")
    return chunks


def build_index(chunks):
    """构建向量索引"""
    from langchain_community.vectorstores import FAISS
    
    embeddings = get_embeddings()
    logger.info("正在构建向量索引...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    logger.info(f"索引已保存到 {VECTORSTORE_DIR}")
    return vectorstore


def load_index():
    """加载向量索引"""
    from langchain_community.vectorstores import FAISS
    
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(f"索引不存在: {VECTORSTORE_DIR}，请先运行 --build")
    
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info(f"索引加载完成，共 {vectorstore.index.ntotal} 条")
    return vectorstore


def ask_question(vectorstore, question):
    """问答"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", """参考资料：
{context}

问题：{question}"""),
    ])
    
    def format_docs(docs):
        return "\n\n".join([
            f"[来源：{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ])
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    
    return rag_chain.invoke(question)


def main():
    parser = argparse.ArgumentParser(description="简单 RAG 问答系统")
    parser.add_argument("--build", action="store_true", help="构建索引")
    parser.add_argument("--query", type=str, help="提问")
    args = parser.parse_args()
    
    if args.build:
        # 构建索引
        logger.info("=== 构建索引模式 ===")
        docs = load_pdfs()
        chunks = split_documents(docs)
        build_index(chunks)
        logger.info("索引构建完成！")
    
    elif args.query:
        # 问答模式
        logger.info("=== 问答模式 ===")
        vectorstore = load_index()
        answer = ask_question(vectorstore, args.query)
        print("\n" + "="*50)
        print("问题：", args.query)
        print("="*50)
        print("回答：\n", answer)
        print("="*50)
    
    else:
        # 交互模式
        logger.info("=== 交互问答模式 ===")
        vectorstore = load_index()
        print("\n输入问题（输入 q 退出）：\n")
        
        while True:
            question = input("> ").strip()
            if question.lower() in ["q", "quit", "exit"]:
                break
            if not question:
                continue
            
            try:
                answer = ask_question(vectorstore, question)
                print(f"\n回答：{answer}\n")
            except Exception as e:
                logger.error(f"错误: {e}")


if __name__ == "__main__":
    main()
