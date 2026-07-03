"""
macOS 本地推理封装器

提供两种模式：
1. API 模式（默认）：启动 FastAPI server 模拟 OpenAI 接口
2. 直连模式：直接调用 transformers pipeline

使用方式：
    from macos_runner import get_completion
    result = get_completion(messages, model="qwen2-0.5b")
"""

import json
import time
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


_pipeline = None
_tokenizer = None
_model_loaded = False
_current_model = ""


def load_model(model_name: str = "gpt2"):
    global _pipeline, _tokenizer, _model_loaded
    if _model_loaded:
        return

    print(f"[macos_runner] Loading model: {model_name}...")
    start = time.time()

    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
        )

        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
        )

        _pipeline = pipeline(
            "text-generation",
            model=_model,
            tokenizer=_tokenizer,
        )

        _model_loaded = True
        print(f"[macos_runner] Model loaded in {time.time()-start:.1f}s")

    except Exception as e:
        print(f"[macos_runner] Failed to load model: {e}")
        raise


def build_prompt(messages: list[dict]) -> str:
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant: "
    return prompt


def get_completion(
    messages: list[dict],
    model: str = "qwen2-0.5b",
    max_tokens: int = 256,
    temperature: float = 0.0,
    response_format: Optional[dict] = None,
    extra_body: Optional[dict] = None
) -> tuple[str, float]:
    load_model()

    prompt = build_prompt(messages)

    is_json_mode = (response_format or {}).get("type") == "json_object"
    if is_json_mode:
        prompt += "{"

    t0 = time.time()
    outputs = _pipeline(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=_tokenizer.eos_token_id,
        eos_token_id=_tokenizer.eos_token_id,
    )

    response_text = outputs[0]["generated_text"]
    response_text = response_text[len(prompt):].strip()

    if is_json_mode:
        response_text = "{" + response_text

    if response_text.endswith("</s>"):
        response_text = response_text[:-4].strip()

    return response_text, time.time() - t0


class MockOpenAI:
    def __init__(self, api_key=None, base_url=None):
        pass

    class Completions:
        @staticmethod
        def create(
            model,
            messages,
            max_tokens=256,
            temperature=0.7,
            response_format=None,
            extra_body=None,
            tools=None,
            tool_choice=None,
            **kwargs
        ):
            content, dt = get_completion(
                messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
                extra_body=extra_body
            )

            class Choice:
                class Message:
                    def __init__(self, content):
                        self.content = content

                    @property
                    def tool_calls(self):
                        return None

                def __init__(self, content):
                    self.message = Choice.Message(content)
                    self.finish_reason = "stop"

            class Response:
                def __init__(self, content):
                    self.choices = [Choice(content)]

            return Response(content)

    class Chat:
        def __init__(self):
            self.completions = MockOpenAI.Completions()

    @property
    def chat(self):
        return self.Chat()


if __name__ == "__main__":
    load_model()
    print("Model loaded successfully!")

    test_messages = [
        {"role": "system", "content": "你是一个友好的助手"},
        {"role": "user", "content": "你好，你是谁？"}
    ]
    result, dt = get_completion(test_messages, max_tokens=50)
    print(f"Response ({dt:.2f}s): {result}")