import time
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────
client = OpenAI(
    api_key="EMPTY",                         # vLLM server 不需要真实 key
    base_url="http://localhost:8000/v1",     # 本地 vLLM server
)
MODEL = "qwen2-0.5b"

# ── 意图分类体系 ──────────────────────────────────────────────────────
INTENT_CHOICES = ["样式布局", "逻辑功能", "框架应用", "性能优化", "其他"]

SYSTEM_PROMPT = f"""你是前端技术问答助手的意图路由器。
根据用户问题，判断意图属于以下哪一类，只输出类别名称，不要任何其他文字。

可选类别：{" / ".join(INTENT_CHOICES)}"""

# ── 测试用例 ──────────────────────────────────────────────────────────
TEST_CASES = [
    ("如何让一个div水平垂直居中", "样式布局"),
    ("Flexbox和Grid布局有什么区别", "样式布局"),
    ("CSS中position: absolute和relative的区别", "样式布局"),

    ("箭头函数和普通函数的this指向有什么不同", "逻辑功能"),
    ("如何实现深拷贝一个对象", "逻辑功能"),
    ("JavaScript的闭包是什么", "逻辑功能"),

    ("React中useEffect的依赖数组怎么用", "框架应用"),
    ("Vue3的响应式原理是基于Proxy吗", "框架应用"),
    ("Angular中如何实现双向数据绑定", "框架应用"),

    ("如何减少首屏加载时间", "性能优化"),
    ("长列表渲染有哪些优化方案", "性能优化"),
    ("如何优化Webpack打包体积", "性能优化"),

    ("今天天气怎么样", "其他"),
    ("帮我推荐一部电影", "其他"),
    ("如何设置电脑壁纸", "其他"),
]


def run_without_guided(user_msg: str) -> tuple[str, float]:
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


def run_with_guided_choice(user_msg: str) -> tuple[str, float]:
    """guided_choice 约束：底层 FSM 屏蔽非法 token"""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=10,
        extra_body={"guided_choice": INTENT_CHOICES},   # ★ 关键参数
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


def main():
    print("=" * 70)
    print("  Demo: guided_choice（枚举约束）")
    print(f"  Model: {MODEL}   Choices: {INTENT_CHOICES}")
    print("=" * 70)

    raw_correct = 0
    raw_in_choices = 0
    guided_correct = 0
    raw_time_total = 0.0
    guided_time_total = 0.0

    print(f"\n{'问题':<32}{'真值':<10}{'裸 prompt 输出':<20}{'guided 输出':<15}")
    print("-" * 90)

    for user_msg, expected in TEST_CASES:
        raw_out, raw_dt = run_without_guided(user_msg)
        guided_out, guided_dt = run_with_guided_choice(user_msg)

        raw_time_total += raw_dt
        guided_time_total += guided_dt

        # 裸 prompt 可能输出中文引号、句号、解释文字等
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
