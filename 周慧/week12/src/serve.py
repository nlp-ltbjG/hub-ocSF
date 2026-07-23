"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI

接口：
  POST /query/manual  - 手写版 ReAct，流式返回每步
  POST /query/fc      - Function Calling 版，流式返回每步
  GET  /health        - 健康检查

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag

    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent", lifespan=lifespan)


# ── 会话管理 ───────────────────────────────────────────────────────────────────
from collections import OrderedDict
import time
import threading


class SessionManager:
    def __init__(self, ttl_seconds: int = 1800):
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def _clean_expired(self):
        now = time.time()
        expired_keys = []
        with self._lock:
            for key, value in self._sessions.items():
                if now - value["last_access"] > self._ttl:
                    expired_keys.append(key)
            for key in expired_keys:
                del self._sessions[key]

    def _create_agent(self, mode: str) -> object:
        if mode == "manual":
            from react_manual import Agent
        else:
            from react_function_calling import Agent
        return Agent()

    def get_or_create(self, session_id: str, mode: str) -> tuple:
        self._clean_expired()

        if not session_id:
            agent = self._create_agent(mode)
            return agent, "", None

        with self._lock:
            if session_id in self._sessions:
                entry = self._sessions[session_id]
                if entry["mode"] != mode:
                    entry["agent"] = self._create_agent(mode)
                    entry["mode"] = mode
                entry["last_access"] = time.time()
                self._sessions.move_to_end(session_id)
                return entry["agent"], session_id, entry["lock"]

            agent = self._create_agent(mode)
            session_lock = threading.Lock()
            self._sessions[session_id] = {
                "agent": agent,
                "mode": mode,
                "last_access": time.time(),
                "lock": session_lock,
            }
            return agent, session_id, session_lock

    def clear(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def get_session_count(self) -> int:
        self._clean_expired()
        with self._lock:
            return len(self._sessions)


session_manager = SessionManager(ttl_seconds=1800)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    max_steps: int = 10
    session_id: str = ""
    clear_history: bool = False


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(
    question: str,
    max_steps: int,
    mode: str,
    session_id: str = "",
    clear_history: bool = False,
):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。

    支持会话管理：通过 session_id 复用 Agent 实例，实现短期记忆。
    """
    global session_manager

    agent, session_id, session_lock = session_manager.get_or_create(session_id, mode)

    if clear_history and session_id:
        if mode == "manual":
            from react_manual import SYSTEM_PROMPT

            agent.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        else:
            from react_function_calling import FC_SYSTEM_PROMPT

            agent.messages = [{"role": "system", "content": FC_SYSTEM_PROMPT}]

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            if session_lock:
                with session_lock:
                    for step_data in agent.ask(question, max_steps=max_steps):
                        queue.put_nowait(step_data)
            else:
                for step_data in agent.ask(question, max_steps=max_steps):
                    queue.put_nowait(step_data)
        finally:
            queue.put_nowait(_SENTINEL)

    yield _sse(
        {"type": "start", "question": question, "mode": mode, "session_id": session_id}
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        yield _sse(step_data)

    yield _sse({"type": "done", "session_id": session_id})


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react(
            req.question, req.max_steps, "manual", req.session_id, req.clear_history
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react(
            req.question, req.max_steps, "fc", req.session_id, req.clear_history
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": os.getenv("AGENT_MODEL", "qwen-max"),
        "session_count": session_manager.get_session_count(),
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    success = session_manager.clear(session_id)
    if success:
        return {"status": "ok", "message": "会话已清除"}
    return {"status": "error", "message": "会话不存在"}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"


@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
