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

from typing import Optional

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


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:   str
    max_steps:  int = 10
    session_id: Optional[str] = None  # 新增：会话ID，可选


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str, session_id: Optional[str] = None):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。
    
    支持多轮对话：通过 session_id 关联历史上下文
    """
    from session_store import session_store
    
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    # 处理 session_id：不存在则创建新会话
    if not session_id or not session_store.has_session(session_id):
        session_id = session_store.create_session()

    # 获取历史上下文
    history = session_store.get_history(session_id)

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()
    
    final_answer = None

    def _worker():
        nonlocal final_answer
        try:
            for step_data in react_run(question, max_steps=max_steps, history=history):
                if step_data["type"] == "final":
                    final_answer = step_data["answer"]
                queue.put_nowait(step_data)
        finally:
            queue.put_nowait(_SENTINEL)

    # 发送 start 事件，携带 session_id
    yield _sse({"type": "start", "question": question, "mode": mode, "session_id": session_id})

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        yield _sse(step_data)

    # 保存本轮结果到历史
    if final_answer:
        session_store.append_turn(session_id, question, final_answer)

    yield _sse({"type": "done"})


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


# ── 会话管理接口 ──────────────────────────────────────────────────────────────
@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """获取指定会话的历史对话"""
    from session_store import session_store
    return {"session_id": session_id, "history": session_store.get_history(session_id)}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    from session_store import session_store
    session_store.delete_session(session_id)
    return {"status": "ok", "message": f"会话 {session_id} 已删除"}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
