"""
科技资讯问答系统

流程：
  加载科技新闻 → 向量化 → 余弦相似度检索 + 关键词匹配 → LLM生成回答

使用方式：
  export DASHSCOPE_API_KEY="sk-xxx"
  python simple_qa.py
  然后输入问题即可

支持的话题：
  - AI大模型、AI编程、AI安全、自动驾驶
  - AI算力、AI终端、AI服务、智能代理
"""

import os
import json
import re
import numpy as np
from pathlib import Path
from openai import OpenAI

BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data/chunks"
DASHSCOPE_URL  = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL    = "text-embedding-v3"
EMBED_DIM      = 1024
LLM_MODEL      = "qwen-plus"
TOP_K          = 5
SCORE_THRESHOLD = 0.15

SYSTEM_PROMPT = """你是一个专业的科技资讯助手，专门回答关于中国科技行业动态的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体数据时，在句末标注来源编号，如：营收增长50%[1]
4. 数字要精确，不得四舍五入或模糊表达
5. 回答简洁，重点突出，避免无关废话
6. 如果有多个来源涉及同一话题，请综合各来源信息给出完整回答"""


def get_client():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


class SimpleQA:
    def __init__(self, client):
        self.client = client
        self.load_data()
        self.build_index()
        print("\n支持查询的话题：")
        print("  - AI大模型、AI编程、AI安全、自动驾驶")
        print("  - AI算力、AI终端、AI服务、智能代理")

    def load_data(self):
        self.chunks = []
        data_file = DATA_DIR / "tech_news.json"
        
        if data_file.exists():
            with open(data_file, encoding="utf-8") as f:
                self.chunks = json.load(f)
            print(f"已加载 {data_file.name}: {len(self.chunks)} 条")
        else:
            raise FileNotFoundError(f"未找到数据文件: {data_file}")

    def build_index(self):
        contents = [chunk["content"] for chunk in self.chunks]
        batch_size = 32
        embeddings = []
        
        print("正在向量化数据...")
        total_batches = (len(contents) + batch_size - 1) // batch_size
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            resp = self.client.embeddings.create(
                model=EMBED_MODEL, input=batch, dimensions=EMBED_DIM
            )
            embeddings.extend([d.embedding for d in resp.data])
            if (i // batch_size + 1) % 10 == 0 or (i // batch_size + 1) == total_batches:
                print(f"  已完成 {i // batch_size + 1}/{total_batches} 批次")
        
        self.vectors = np.array(embeddings, dtype="float32")
        self.vectors = self.vectors / np.maximum(
            np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-9
        )
        print("向量化完成")

    def search(self, query):
        resp = self.client.embeddings.create(
            model=EMBED_MODEL, input=[query], dimensions=EMBED_DIM
        )
        query_vec = np.array([resp.data[0].embedding], dtype="float32")
        query_vec = query_vec / np.maximum(np.linalg.norm(query_vec), 1e-9)

        scores = self.vectors @ query_vec.T
        top_idx = np.argsort(scores[:, 0])[::-1][:TOP_K * 2]

        query_tokens = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', query.lower()))
        
        results = []
        for idx in top_idx:
            score = float(scores[idx, 0])
            chunk = self.chunks[idx]
            content = chunk["content"].lower()
            
            keyword_match = 0
            for token in query_tokens:
                if len(token) >= 2 and token in content:
                    keyword_match += 1
            
            boost_score = score
            if keyword_match > 0:
                boost_score = score + keyword_match * 0.05
            
            if boost_score < SCORE_THRESHOLD:
                continue
            
            results.append({
                "content": chunk["content"],
                "score": boost_score,
                "original_score": score,
                "keyword_match": keyword_match,
                "metadata": chunk["metadata"],
            })
        
        results.sort(key=lambda x: -x["score"])
        return results[:TOP_K]

    def build_context(self, retrieved):
        parts = []
        citations = []
        for i, item in enumerate(retrieved, 1):
            source = item["metadata"].get("source", "")
            date = item["metadata"].get("date", "")
            topic = item["metadata"].get("topic", "")

            label_parts = [f"[{i}]"]
            if topic:
                label_parts.append(f"· {topic}")
            if source:
                label_parts.append(f"· {source}")
            if date:
                label_parts.append(f"· {date}")

            label = " ".join(label_parts)
            parts.append(f"{label}\n{item['content']}")
            citations.append({"index": i, "source": label})
        return "\n\n---\n\n".join(parts), citations

    def answer(self, question):
        retrieved = self.search(question)
        if not retrieved:
            return {
                "answer": "未找到相关内容，无法回答此问题。",
                "citations": []
            }

        context, citations = self.build_context(retrieved)
        user_msg = f"【参考资料】\n{context}\n\n【问题】\n{question}\n\n请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"
        
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )
        return {
            "answer": resp.choices[0].message.content,
            "citations": citations,
            "retrieved": retrieved,
        }


def main():
    try:
        client = get_client()
    except EnvironmentError as e:
        print(e)
        print("\n设置方式：export DASHSCOPE_API_KEY=\"你的API密钥\"")
        return

    qa = SimpleQA(client)
    
    print("\n" + "="*60)
    print("科技资讯问答系统")
    print("="*60)
    print("输入 'exit' 退出\n")
    
    while True:
        try:
            q = input("问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        
        print("思考中...")
        result = qa.answer(q)
        
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        
        if result["citations"]:
            print("\n── 来源 ──")
            for c in result["citations"]:
                print(f"  {c['source']}")
        print()


if __name__ == "__main__":
    main()
