"""
LLM 客户端 - 封装 deepseek-v4-flash 调用

设计原则：
- 单一入口：所有 LLM 调用走 LLMClient
- 环境变量读取密钥，不硬编码
- 支持 System Prompt 注入（来自 SOUL.md）
- 支持重试和错误降级
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict


class LLMClient:
    """deepseek-v4-flash 客户端"""

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-v4-flash"
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        soul_path: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_API_URL", self.DEFAULT_BASE_URL)
        self.model = model or os.getenv("DEEPSEEK_MODEL", self.DEFAULT_MODEL)
        self.soul_path = soul_path
        self._system_prompt = None
        self._client = None

    @property
    def is_available(self) -> bool:
        """检查 LLM 是否可用"""
        return bool(self.api_key)

    def _get_client(self):
        """懒加载 OpenAI 客户端"""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.DEFAULT_TIMEOUT,
            )
        return self._client

    def load_soul(self, soul_path: str) -> str:
        """从 SOUL.md 加载 System Prompt"""
        try:
            content = Path(soul_path).read_text(encoding="utf-8")
            self._system_prompt = content.strip()
            return self._system_prompt
        except Exception as e:
            print(f"[LLM] 加载 SOUL.md 失败: {e}")
            return ""

    def _build_messages(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """构建消息列表"""
        messages = []

        effective_system = system_prompt or self._system_prompt
        if effective_system:
            messages.append({"role": "system", "content": effective_system})

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_input})
        return messages

    def chat(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 2,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        调用 LLM 对话接口

        Args:
            user_input: 用户输入
            system_prompt: 系统提示词（覆盖默认）
            temperature: 温度参数，0=确定性，1=创造性
            max_retries: 最大重试次数
            conversation_history: 多轮对话历史

        Returns:
            LLM 返回的文本内容
        """
        if not self.is_available:
            raise EnvironmentError(
                "DEEPSEEK_API_KEY 未设置。请执行: export DEEPSEEK_API_KEY='sk-xxx'"
            )

        messages = self._build_messages(user_input, system_prompt, conversation_history)
        client = self._get_client()

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2**attempt
                    print(
                        f"[LLM] 调用失败，{wait}s 后重试 ({attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(wait)
                else:
                    print(f"[LLM] 重试耗尽: {last_error}")
                    raise last_error

        raise last_error

    def chat_with_json_output(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> dict:
        """
        调用 LLM 并解析 JSON 输出

        用于知识总结等需要结构化输出的场景
        """
        raw = self.chat(user_input, system_prompt, temperature)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    return json.loads(raw[json_start:json_end])
                except json.JSONDecodeError:
                    pass
            print(f"[LLM] JSON 解析失败，原始输出: {raw[:200]}...")
            return {"error": "JSON解析失败", "raw": raw}

    def summarize_ppt(
        self,
        ppt_text: str,
        topic_hint: str = "",
    ) -> dict:
        """
        专用：PPT 知识总结

        将提取的 PPT 文本发送给 LLM，要求返回结构化总结 JSON
        """
        system_prompt = """你是一位资深的技术文档分析师，擅长从各类 PPT 中提炼核心知识。
请仔细分析用户提供的 PPT 文本内容，输出结构化的知识总结。

输出格式要求（严格 JSON，所有字段都是数组，每个元素是对象）：
{
  "core_concepts": [{"name": "概念名", "description": "一句话解释"}],
  "architectures": [{"name": "架构/模型名", "description": "简要说明"}],
  "comparisons": [{"a": "项目A", "b": "项目B", "difference": "核心区别"}],
  "best_practices": [{"title": "实践标题", "description": "详细说明"}],
  "applications": [{"scenario": "应用场景", "description": "说明"}],
  "supplementary_topics": ["需补充的前置知识1", "需补充的前置知识2"],
  "topic": "推断的主题英文名称(用下划线分隔)"
}

规则：
- core_concepts: 文中出现的关键技术术语，每个配一句话解释
- architectures: 文中描述的架构、模型、分层、组件结构
- comparisons: 文中涉及的技术对比，每对给出核心区别
- best_practices: 文中提到的设计原则、最佳实践
- applications: 文中描述的应用场景
- supplementary_topics: 初学者理解本文需要补充的前置知识点（3-5个）
- topic: 用简短英文命名推断的主题（如 System_Architecture, Data_Analysis, Machine_Learning 等）
- 严格输出 JSON，不要添加任何解释文字"""

        max_chars = 30000
        if len(ppt_text) > max_chars:
            ppt_text = ppt_text[:max_chars] + "\n\n...(内容已截断)"

        user_input = f"""请分析以下 PPT 文本内容，输出结构化知识总结：

{ppt_text}

（请严格按 JSON 格式输出）"""

        return self.chat_with_json_output(user_input, system_prompt, temperature=0.3)

    def supplement_knowledge(
        self,
        missing_topics: List[str],
    ) -> List[Dict]:
        """
        专用：生成补充知识的搜索策略

        对于识别出的缺失知识点，生成针对性的搜索查询和补充建议
        """
        if not missing_topics:
            return []

        system_prompt = """你是一位 AI Agent 技术课程的助教。
针对以下初学者需要掌握的前置知识点，请为每个知识点提供：
1. 一个精准的搜索查询（中文，适合在搜索引擎中查找入门教程）
2. 一句话简介（帮助初学者快速理解）
3. 学习优先级（high/medium/low）

输出格式（严格 JSON 数组）：
[
  {
    "topic": "知识点名称",
    "search_query": "搜索查询",
    "brief_intro": "一句话简介",
    "priority": "high"
  }
]

严格输出 JSON 数组，不要添加任何解释文字。"""

        user_input = f"""请为以下 {len(missing_topics)} 个知识点生成搜索策略：

{json.dumps(missing_topics, ensure_ascii=False)}"""

        raw = self.chat(user_input, system_prompt, temperature=0.3)

        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        return [
            {
                "topic": t,
                "search_query": f"{t} 入门教程 详解 2025",
                "brief_intro": "建议通过搜索引擎学习",
                "priority": "medium",
            }
            for t in missing_topics
        ]


def create_llm_client(soul_path: Optional[str] = None) -> LLMClient:
    """工厂函数：创建 LLM 客户端"""
    client = LLMClient(soul_path=soul_path)
    if soul_path and os.path.exists(soul_path):
        client.load_soul(soul_path)
    return client
