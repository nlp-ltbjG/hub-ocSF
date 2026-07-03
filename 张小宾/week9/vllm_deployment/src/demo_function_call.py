"""
演示 vLLM 的函数调用（function call）接口

教学重点：
  1. vLLM 的 function call 是 OpenAI 兼容的标准接口
  2. guided_json 是 function call 的底层实现基础
  3. 结合工具调用实现真正的 Agent 能力

场景：金融问答的工具路由
  用户问题 → 意图识别 → 选择合适工具 → 调用工具获取结果 → 总结回答

使用方式：
  # Linux (vLLM)：需先启动 server
  python demo_function_call.py

  # macOS（本地模式，不支持 function call）：
  python demo_function_call.py --local
"""

import argparse
import json
import time
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def get_client(local_mode: bool):
    if local_mode:
        from macos_runner import MockOpenAI
        return MockOpenAI()
    else:
        return OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")


client = None
MODEL = "qwen2-0.5b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "查询股票实时价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票代码或简称，如 600519、茅台",
                    },
                    "date": {
                        "type": "string",
                        "description": "查询日期，格式 YYYY-MM-DD，默认为今天",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_report",
            "description": "查询公司财务报表数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "公司名称"},
                    "year": {"type": "integer", "description": "年度"},
                    "quarter": {
                        "type": "integer",
                        "description": "季度，1-4，默认全年",
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["营收", "净利润", "ROE", "毛利率"],
                        "description": "财务指标",
                    },
                },
                "required": ["company", "year", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "查询公司相关新闻",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "公司名称"},
                    "limit": {"type": "integer", "description": "返回条数，默认5"},
                },
                "required": ["company"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是金融问答助手。根据用户问题，选择合适的工具进行调用。
如果需要调用工具，按工具格式输出；如果不需要，可以直接回答。"""


def mock_tool_call(name: str, args: dict[str, Any]) -> str:
    if name == "get_stock_price":
        return json.dumps({
            "symbol": args.get("symbol", "未知"),
            "price": 185.50,
            "change": "+2.3%",
            "date": args.get("date", "2024-01-15"),
        }, ensure_ascii=False)
    elif name == "get_financial_report":
        return json.dumps({
            "company": args.get("company", "未知"),
            "year": args.get("year", 2023),
            "metric": args.get("metric", "营收"),
            "value": "1250亿",
            "growth": "+15.2%",
        }, ensure_ascii=False)
    elif name == "get_news":
        return json.dumps([
            {"title": f"{args.get('company')}发布新产品", "date": "2024-01-14"},
            {"title": f"{args.get('company')}业绩预增公告", "date": "2024-01-13"},
        ], ensure_ascii=False)
    return json.dumps({"error": "unknown tool"}, ensure_ascii=False)


def run(user_msg: str, use_tools: bool, local_mode: bool = False) -> tuple[str, bool, float]:
    global client
    if client is None:
        client = get_client(local_mode)

    t0 = time.time()
    kwargs = {}
    if use_tools and not local_mode:
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=200,
        **kwargs,
    )
    content = resp.choices[0].message.content
    finish_reason = resp.choices[0].finish_reason
    has_tool_call = finish_reason == "tool_calls" or (
        hasattr(resp.choices[0].message, "tool_calls") and resp.choices[0].message.tool_calls
    )
    return content, has_tool_call, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="macOS 本地模式（不支持 function call）")
    args = parser.parse_args()

    print("=" * 75)
    print("  Demo: Function Call（工具调用）")
    print(f"  Model: {MODEL}")
    print(f"  Mode: {'macOS Local' if args.local else 'vLLM Server'}")
    print("=" * 75)

    if args.local:
        print("\n⚠️  macOS 本地模式不支持 function call\n")

    TEST_CASES = [
        "查一下茅台今天的股价",
        "贵州茅台 2023 年净利润是多少",
        "最近宁德时代有什么新闻",
        "帮我分析一下招商银行的财务状况",
    ]

    for user in TEST_CASES:
        print(f"\n▶ 用户：{user}")
        content, has_tool_call, dt = run(user, use_tools=True, local_mode=args.local)

        if has_tool_call and not args.local:
            print(f"  [工具调用] {content}")
            try:
                tool_call = json.loads(content)
                if isinstance(tool_call, dict) and "name" in tool_call:
                    result = mock_tool_call(tool_call["name"], tool_call.get("arguments", {}))
                    print(f"  [工具返回] {result}")
            except json.JSONDecodeError:
                print(f"  [工具调用] 无法解析: {content}")
        else:
            print(f"  [直接回答] {content}")


if __name__ == "__main__":
    main()