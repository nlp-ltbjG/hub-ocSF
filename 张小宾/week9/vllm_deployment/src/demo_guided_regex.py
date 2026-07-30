"""
演示 vLLM 的 guided_regex 约束解码：输出必须匹配指定正则

教学重点：
  1. guided_regex 让 LLM 输出严格符合格式（日期、手机号、邮箱、代码等）
  2. 配合正则解析下游系统时，不用再写容错逻辑
  3. 工程价值：减少 "模型说得对但格式错" 导致的下游报错

场景：从自由文本中抽取结构化字段
  - 日期标准化：任意日期表述 → YYYY-MM-DD
  - 股票代码：任意表述 → 6 位数字

使用方式：
  # Linux (vLLM)：需先启动 server
  python demo_guided_regex.py

  # macOS（本地模式，不支持 guided_regex）：
  python demo_guided_regex.py --local
"""

import argparse
import re
import time

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

# ── 任务 1：日期标准化 ────────────────────────────────────────────────
DATE_REGEX = r"\d{4}-\d{2}-\d{2}"
DATE_SYSTEM = "你是日期抽取助手。从用户输入中抽取日期，严格用 YYYY-MM-DD 格式输出，不输出任何其他文字。"
DATE_CASES = [
    "2024年5月12日",
    "2023/12/1 下午开会",
    "三月三号我去北京",
    "2024.11.30 是截止日期",
    "明天（假设今天是2026-05-11）",
    "2024 年 10 月的第一天",
]

# ── 任务 2：股票代码抽取 ──────────────────────────────────────────────
STOCK_REGEX = r"\d{6}"
STOCK_SYSTEM = "你是股票代码抽取助手。从用户输入中找到 A 股代码（6 位数字），直接输出代码，不输出任何其他文字。"
STOCK_CASES = [
    "帮我查 600000 浦发银行",
    "code: 000001 平安银行",
    "茅台的代码是 600519",
    "六零零五一九",
    "股票代码：300750（宁德时代）",
]


def run_generate(system: str, user: str, regex: str | None = None, local_mode: bool = False) -> tuple[str, float]:
    global client
    if client is None:
        client = get_client(local_mode)

    t0 = time.time()
    extra = {"guided_regex": regex} if regex and not local_mode else {}
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=30,
        extra_body=extra,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


def matches(pattern: str, text: str) -> bool:
    return bool(re.fullmatch(pattern, text))


def run_section(title: str, system: str, regex: str, cases: list[str], local_mode: bool):
    print("=" * 70)
    print(f"  {title}")
    print(f"  正则: {regex}")
    print("=" * 70)

    if local_mode:
        print("\n⚠️  macOS 本地模式不支持 guided_regex 约束解码\n")

    raw_ok = 0
    guided_ok = 0
    print(f"\n{'输入':<35}{'裸 prompt':<25}{'guided_regex':<15}")
    print("-" * 75)
    for user in cases:
        raw_out, _ = run_generate(system, user, local_mode=local_mode)
        guided_out, _ = run_generate(system, user, regex, local_mode=local_mode)
        raw_match = matches(regex, raw_out)
        guided_match = matches(regex, guided_out)
        if raw_match:
            raw_ok += 1
        if guided_match:
            guided_ok += 1
        flag_raw = "✓" if raw_match else "✗"
        flag_guided = "✓" if guided_match else "✗"
        raw_disp = raw_out[:22] + "…" if len(raw_out) > 22 else raw_out
        print(f"{user:<33}  {flag_raw} {raw_disp:<20}  {flag_guided} {guided_out}")
    n = len(cases)
    print("-" * 75)
    print(f"格式合法率：裸 prompt {raw_ok}/{n} ({100*raw_ok/n:.0f}%)  |  "
          f"guided_regex {guided_ok}/{n} ({100*guided_ok/n:.0f}%)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="macOS 本地模式（不支持 guided_regex）")
    args = parser.parse_args()

    run_section("任务 1：日期标准化 → YYYY-MM-DD",
                DATE_SYSTEM, DATE_REGEX, DATE_CASES, args.local)
    run_section("任务 2：A 股代码抽取 → 6 位数字",
                STOCK_SYSTEM, STOCK_REGEX, STOCK_CASES, args.local)

    print("=" * 70)
    print("  结论：guided_regex 保证下游解析器永远能拿到合法输入")
    print("       特别适合日期/电话/代码/邮编等有严格格式的字段")
    print("=" * 70)


if __name__ == "__main__":
    main()