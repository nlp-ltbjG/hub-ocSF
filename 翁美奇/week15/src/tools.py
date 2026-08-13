"""
数值计算工具 + RAG 检索工具
- calc_profit_margin : 毛利率 = (收入 - 成本) / 收入 * 100%
- calc_asset_turnover : 资产周转率 = 收入 / 总资产
- rag_search : 从 FAISS 向量库检索财报文本
"""
import os
import json


# ========== 数值计算工具 ==========

def calc_profit_margin(revenue: float, cost: float) -> dict:
    """毛利率 = (收入 - 成本) / 收入 * 100%"""
    if revenue == 0:
        return {"error": "revenue 不能为 0"}
    value = (revenue - cost) / revenue * 100
    return {
        "indicator": "毛利率",
        "value": round(value, 2),
        "unit": "%",
        "formula": "(revenue - cost) / revenue * 100%",
    }


def calc_asset_turnover(revenue: float, total_assets: float) -> dict:
    """资产周转率 = 收入 / 总资产"""
    if total_assets == 0:
        return {"error": "total_assets 不能为 0"}
    value = revenue / total_assets
    return {
        "indicator": "资产周转率",
        "value": round(value, 4),
        "formula": "revenue / total_assets",
    }


# 数值工具的 function-calling schema（供数值计算专家子 Agent 使用）
NUMERIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calc_profit_margin",
            "description": "计算毛利率 = (营业收入 - 营业成本) / 营业收入 * 100%",
            "parameters": {
                "type": "object",
                "properties": {
                    "revenue": {"type": "number", "description": "营业收入"},
                    "cost": {"type": "number", "description": "营业成本"},
                },
                "required": ["revenue", "cost"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_asset_turnover",
            "description": "计算资产周转率 = 营业收入 / 总资产",
            "parameters": {
                "type": "object",
                "properties": {
                    "revenue": {"type": "number", "description": "营业收入"},
                    "total_assets": {"type": "number", "description": "总资产"},
                },
                "required": ["revenue", "total_assets"],
            },
        },
    },
]

# 工具名 -> 函数映射
NUMERIC_TOOL_MAP = {
    "calc_profit_margin": calc_profit_margin,
    "calc_asset_turnover": calc_asset_turnover,
}


# ========== RAG 检索工具 ==========

_VECTORSTORE = None
_EMBEDDINGS = None


def _get_embeddings():
    """获取 embedding 模型（必须与建库时使用的模型一致）。

    通过环境变量配置：
      EMBEDDING_MODEL    : embedding 模型名（默认 text-embedding-3-small）
      EMBEDDING_API_KEY  : embedding 服务密钥（默认回退 OPENAI_API_KEY）
      EMBEDDING_BASE_URL : embedding 服务地址（默认回退 OpenAI 官方）
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS
    from langchain_openai import OpenAIEmbeddings

    kwargs = {"model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")}
    if os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    if os.getenv("EMBEDDING_BASE_URL"):
        kwargs["base_url"] = os.getenv("EMBEDDING_BASE_URL")
    _EMBEDDINGS = OpenAIEmbeddings(**kwargs)
    return _EMBEDDINGS


def _get_vectorstore():
    """加载 FAISS 向量库（LangChain 存储格式：index.faiss + index.pkl）"""
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    from langchain_community.vectorstores import FAISS

    base = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "vectorstore", "faiss_lc"
    )
    _VECTORSTORE = FAISS.load_local(
        base, _get_embeddings(), allow_dangerous_deserialization=True
    )
    return _VECTORSTORE


def rag_search(query: str, k: int = 4) -> dict:
    """从财报向量库检索与 query 最相关的 k 条文本片段"""
    vs = _get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    results = []
    for d in docs:
        m = d.metadata or {}
        results.append(
            {
                "content": d.page_content,
                "stock_code": m.get("stock_code"),
                "year": m.get("year"),
                "section": m.get("section"),
            }
        )
    return {"query": query, "count": len(results), "results": results}
