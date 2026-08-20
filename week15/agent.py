"""mini-agent: 自实现的可下发 subagent、并行执行的多 agent 编排系统。

用法:
    pip install -r requirements.txt
    set OPENAI_API_KEY=sk-xxx                      # 必填
    set OPENAI_BASE_URL=https://api.openai.com/v1  # 可选, 兼容任意 OpenAI 格式 API
    set OPENAI_MODEL=gpt-4o-mini                   # 可选, 默认 gpt-4o-mini
    python agent.py "一句话描述你的任务"
"""

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def chat(system, messages, json_mode=False):
    """调用一次 LLM。json_mode 优先结构化输出, 接口不支持则回退。"""
    kwargs = dict(model=MODEL, messages=[{"role": "system", "content": system}] + messages)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        return _client.chat.completions.create(**kwargs).choices[0].message.content
    except Exception:
        if json_mode:
            kwargs.pop("response_format")
            return _client.chat.completions.create(**kwargs).choices[0].message.content
        raise


# ---------- 工具 ----------
def tool_shell(a):
    try:
        r = subprocess.run(a["cmd"], shell=True, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr)[-4000:] or "(无输出)"
    except Exception as e:
        return f"错误: {e}"


def tool_read(a):
    try:
        with open(a["path"], encoding="utf-8", errors="ignore") as f:
            return f.read()[-4000:]
    except Exception as e:
        return f"错误: {e}"


TOOLS = {"shell": tool_shell, "read": tool_read}


def parse_action(text):
    """从模型输出中提取 {"tool": "shell", "cmd": "..."} 形式的工具调用。"""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        a = json.loads(m.group(0))
        return a if a.get("tool") in TOOLS else None
    except Exception:
        return None


# ---------- subagent ----------
def subagent(task):
    """单个 subagent: 自主循环 —— 需要工具就调用工具, 否则输出最终答案。"""
    system = (
        f"你是 subagent, 负责完成子任务「{task['name']}」。\n"
        "需要查文件或执行命令时, 输出纯 JSON 调用工具:\n"
        '{"tool": "shell", "cmd": "命令"} 或 {"tool": "read", "path": "文件路径"}\n'
        "收到「工具结果: ...」后继续推进; 任务完成后直接输出最终答案(纯文本), 不要输出 JSON。"
    )
    messages = [{"role": "user", "content": task["description"]}]
    for _ in range(8):
        out = chat(system, messages)
        messages.append({"role": "assistant", "content": out})
        act = parse_action(out)
        if not act:
            return {"name": task["name"], "result": out}
        messages.append({"role": "user", "content": f"工具结果: {TOOLS[act['tool']](act)}"})
    return {"name": task["name"], "result": "(超过最大步数)"}


# ---------- 主 agent ----------
def run(task, n=3):
    """主 agent: 规划 -> 并行下发 subagent -> 汇总。"""
    raw = chat(
        "你是任务编排器。把任务拆成 n 个相互独立、可并行的子任务。",
        [{"role": "user", "content": f"任务: {task}\n请输出 JSON: {{\"tasks\": [{{\"id\": \"t1\", \"name\": \"\", \"description\": \"\"}}]}}, 子任务数不超过 {n} 个。"}],
        json_mode=True,
    )
    try:
        tasks = json.loads(raw)["tasks"]
    except Exception:
        sys.exit(f"[plan] 解析失败: {raw}")
    print(f"[plan] 拆分为 {len(tasks)} 个子任务, 并行执行...")
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        results = list(ex.map(subagent, tasks))
    return chat(
        "你是主 agent。综合所有 subagent 的结果, 面向原任务输出最终交付物(直接可用)。",
        [{"role": "user", "content": f"原任务: {task}\n\n各 subagent 结果:\n{json.dumps(results, ensure_ascii=False)}"}],
    )


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else (
        "写一个 Python 脚本实现快速排序, 为它编写单元测试, 并写一份使用说明。"
    )
    print(run(t))
