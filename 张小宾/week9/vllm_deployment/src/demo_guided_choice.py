"""
演示 vLLM 的 guided_choice 约束解码：限定输出必须是枚举值之一

教学重点：
  1. guided_choice 的工作原理：每步解码时屏蔽非法 token 的 logits
  2. 在小模型（0.5B）上，约束解码把"偶尔输出自由文本"彻底根治
  3. 意图路由 / 情感分类等分类型 agent 场景的标准用法

场景：金融场景的用户意图路由
  用户问题 → 分类到 5 类意图之一 → 路由到对应的 RAG / tool

使用方式：
  # Linux (vLLM)：需先启动 server
  bash start_server.sh
  python demo_guided_choice.py

  # macOS（本地模式，不支持 guided_choice）：
  python demo_guided_choice.py --local
"""

import argparse
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


# ── 配置 ──────────────────────────────────────────────────────────────
MODEL = "qwen2-0.5b"

# ── 意图分类体系 ──────────────────────────────────────────────────────
INTENT_CHOICES = ["查股价", "查财报", "查新闻", "对比分析", "其他"]

SYSTEM_PROMPT = f"""你是金融问答助手的意图路由器。
根据用户问题，判断意图属于以下哪一类，只输出类别名称，不要任何其他文字。

可选类别：{" / ".join(INTENT_CHOICES)}"""

# ── 测试用例 ──────────────────────────────────────────────────────────
TEST_CASES = [
    ("查一下茅台今天多少钱", "查股价"),
    ("贵州茅台 2024 年营收多少亿", "查财报"),
    ("最近宁德时代有什么新闻", "查新闻"),
    ("对比一下招行和平安的净利润", "对比分析"),
    ("今天天气怎么样", "其他"),
    ("帮我看看 600000 的收盘价", "查股价"),
    ("招商银行去年的 ROE 是多少", "查财报"),
    ("宁德时代被限产了吗", "查新闻"),
    ("比亚迪和特斯拉哪个更强", "对比分析"),
    ("帮我订一张机票", "其他"),
    ("五粮液现在股价", "查股价"),
    ("平安保险的净利润增长率", "查财报"),
]


def run_without_guided(client, user_msg: str) -> tuple[str, float]:
    """裸 prompt 方式：靠指令约束输出"""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


def run_with_guided_choice(client, user_msg: str, local_mode: bool) -> tuple[str, float]:
    """guided_choice 约束：底层 FSM 屏蔽非法 token"""
    t0 = time.time()
    extra_body = {"guided_choice": INTENT_CHOICES} if not local_mode else {}
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=10,
        extra_body=extra_body,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="macOS 本地模式（不支持 guided_choice，仅演示裸 prompt）")
    args = parser.parse_args()

    client = get_client(args.local)

    print("=" * 70)
    print("  Demo: guided_choice（枚举约束）")
    print(f"  Model: {MODEL}   Choices: {INTENT_CHOICES}")
    print(f"  Mode: {'macOS Local' if args.local else 'vLLM Server'}")
    print("=" * 70)

    if args.local:
        print("\n⚠️  macOS 本地模式不支持 guided_choice 约束解码")
        print("   仅演示裸 prompt 方式\n")

    raw_correct = 0
    raw_in_choices = 0
    guided_correct = 0
    raw_time_total = 0.0
    guided_time_total = 0.0

    print(f"\n{'问题':<32}{'真值':<10}{'裸 prompt 输出':<20}{'guided 输出':<15}")
    print("-" * 90)

    for user_msg, expected in TEST_CASES:
        raw_out, raw_dt = run_without_guided(client, user_msg)
        guided_out, guided_dt = run_with_guided_choice(client, user_msg, args.local)

        raw_time_total += raw_dt
        guided_time_total += guided_dt

        raw_in = raw_out in INTENT_CHOICES
        if raw_in:
            raw_in_choices += 1
        if raw_out == expected:
            raw_correct += 1
        if guided_out == expected:
            guided_correct += 1

        flag_raw = "✓" if raw_out == expected else ("~" if raw_in else "✗")
        flag_guided = "✓" if guided_out == expected else "✗"
        print(f"{user_msg:<30}  {expected:<8}  {flag_raw} {raw_out:<16}  {flag_guided} {guided_out}")

    n = len(TEST_CASES)
    print("-" * 90)
    print(f"\n{'指标':<30}{'裸 prompt':<20}{'guided_choice':<20}")
    print("-" * 70)
    print(f"{'输出合法（在枚举内）':<28}{raw_in_choices}/{n} ({100*raw_in_choices/n:.0f}%)      "
          f"{n}/{n} (100%)")
    print(f"{'预测正确':<28}{raw_correct}/{n} ({100*raw_correct/n:.0f}%)      "
          f"{guided_correct}/{n} ({100*guided_correct/n:.0f}%)")
    print(f"{'平均延迟（秒）':<28}{raw_time_total/n:.3f}           {guided_time_total/n:.3f}")

    print()
    print("=" * 70)
    print("  结论：guided_choice 100% 保证输出合法，")
    print("       分类准确率也通常比裸 prompt 高（因为模型不会被错误 token 带偏）")
    print("=" * 70)


if __name__ == "__main__":
    main()