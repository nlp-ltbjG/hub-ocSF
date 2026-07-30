"""
macOS 本地运行演示：直接调用 transformers 推理

演示内容：
  1. 基础对话
  2. JSON 输出（response_format）
  3. 意图分类

使用方式：
  python demo_macos_local.py
"""

import json
import time

from macos_runner import get_completion


def demo_chat():
    print("\n" + "=" * 60)
    print("  Demo 1: Basic Chat")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a friendly helpful assistant"},
        {"role": "user", "content": "Hello, who are you?"}
    ]
    result, dt = get_completion(messages, max_tokens=50)
    print(f"User: {messages[-1]['content']}")
    print(f"Assistant: {result}")
    print(f"Time: {dt:.2f}s")


def demo_json_output():
    print("\n" + "=" * 60)
    print("  Demo 2: JSON Output (response_format)")
    print("=" * 60)

    system_prompt = """You are a sentiment analysis assistant. Analyze the news headline and output JSON:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": number between 0.0 and 1.0,
  "keywords": ["keyword1", "keyword2"]
}
Output ONLY JSON, no other text."""

    test_cases = [
        "Apple stock hits new all-time high, revenue up 20%",
        "Tesla recalls 50000 vehicles due to safety concerns",
    ]

    for news in test_cases:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": news}
        ]

        print(f"\nNews: {news}")

        out_raw, dt_raw = get_completion(messages, max_tokens=150)
        print(f"  [raw prompt]     {dt_raw:.2f}s | {out_raw[:80]}")

        out_json, dt_json = get_completion(
            messages,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        print(f"  [response_format] {dt_json:.2f}s | {out_json[:80]}")


def demo_intent_classification():
    print("\n" + "=" * 60)
    print("  Demo 3: Intent Classification")
    print("=" * 60)

    system_prompt = """You are an intent router for financial Q&A.
Classify the user question into one category. Output ONLY the category name.

Categories: stock_price | financial_report | news | comparison | other"""

    test_cases = [
        ("What is Apple's stock price today?", "stock_price"),
        ("What was Google's revenue in 2023?", "financial_report"),
        ("Any recent news about Tesla?", "news"),
        ("Compare Apple and Microsoft earnings", "comparison"),
        ("What's the weather today?", "other"),
    ]

    print(f"\n{'Question':<40}{'Expected':<15}{'Output':<20}")
    print("-" * 75)

    for user_msg, expected in test_cases:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        result, dt = get_completion(messages, max_tokens=20)
        result = result.strip()
        flag = "✓" if result == expected else "✗"
        print(f"{user_msg:<38} {expected:<13} {flag} {result}")


def main():
    print("=" * 60)
    print("  macOS Local Inference Demo")
    print("  Model: gpt2 (CPU)")
    print("=" * 60)

    print("\nLoading model...")

    try:
        demo_chat()
        demo_json_output()
        demo_intent_classification()

        print("\n" + "=" * 60)
        print("  Demo completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()