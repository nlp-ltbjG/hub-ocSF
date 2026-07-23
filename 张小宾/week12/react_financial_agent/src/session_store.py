"""
会话管理器：管理跨轮对话历史

功能：
- 内存存储会话，每个会话保存最近N轮的问答记录
- 支持创建会话、获取历史、追加轮次、删除会话
- 自动限制最大历史轮数，防止 context window 溢出

使用方式：
    from session_store import session_store
    
    # 创建新会话
    session_id = session_store.create_session()
    
    # 获取历史
    history = session_store.get_history(session_id)
    
    # 追加一轮
    session_store.append_turn(session_id, "用户问题", "助手回答")
"""

import uuid
from typing import Optional, List, Dict


class SessionStore:
    def __init__(self, max_history: int = 5):
        """
        初始化会话管理器
        
        Args:
            max_history: 每个会话最多保留的历史轮数
        """
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._max_history = max_history

    def create_session(self) -> str:
        """创建新会话，返回短ID（8位）"""
        session_id = str(uuid.uuid4())[:8]
        self._sessions[session_id] = []
        return session_id

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        获取会话历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            历史列表，每个元素为 {"question": str, "answer": str}
        """
        return self._sessions.get(session_id, [])

    def append_turn(self, session_id: str, question: str, answer: str):
        """
        追加一轮问答到会话历史
        
        Args:
            session_id: 会话ID
            question: 用户问题
            answer: 助手回答（final answer）
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        self._sessions[session_id].append({
            "question": question,
            "answer": answer,
        })
        
        # 保留最近 max_history 轮
        if len(self._sessions[session_id]) > self._max_history:
            self._sessions[session_id] = self._sessions[session_id][-self._max_history:]

    def delete_session(self, session_id: str):
        """删除会话"""
        self._sessions.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return session_id in self._sessions


# 全局单例，所有请求共享
session_store = SessionStore(max_history=5)