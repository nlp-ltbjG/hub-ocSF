"""
macOS 替代方案：基于 transformers 的 OpenAI 兼容 server
使用 CPU 推理（macOS 无 CUDA），兼容所有 demo 脚本的 OpenAI 调用接口

支持：
  - 基础 chat completions
  - response_format={"type": "json_object"}
  - extra_body 透传（不支持 guided_json/choice/regex，但不会报错）

不支持：
  - vLLM 特有约束解码（guided_json/guided_choice/guided_regex）
  - 流式输出
  - continuous batching

使用方式：
  python macos_server.py --model Qwen/Qwen2-0.5B-Instruct
"""

import argparse
import json
import time
from threading import Thread
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODEL_NAME = "qwen2-0.5b"
pipeline_obj = None
tokenizer = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    response_format: Optional[dict] = None
    extra_body: Optional[dict] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list


app = FastAPI(title="macOS LLM Server", version="1.0")


def build_prompt(messages: list[ChatMessage]) -> str:
    result = ""
    for msg in messages:
        if msg.role == "system":
            result += f"<|system|>\n{msg.content}</s>"
        elif msg.role == "user":
            result += f"<|user|>\n{msg.content}</s>"
        elif msg.role == "assistant":
            result += f"<|assistant|>\n{msg.content}</s>"
    result += "<|assistant|>\n"
    return result


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "macos-server",
            "permission": []
        }]
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(req: ChatCompletionRequest):
    global pipeline_obj, tokenizer

    if pipeline_obj is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    prompt = build_prompt(req.messages)

    is_json_mode = (req.response_format or {}).get("type") == "json_object"
    if is_json_mode:
        prompt += "{"

    generation_kwargs = {
        "max_new_tokens": req.max_tokens or 256,
        "temperature": req.temperature or 0.7,
        "do_sample": (req.temperature or 0.7) > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if is_json_mode:
        generation_kwargs["stop_sequence"] = "</s>"

    try:
        outputs = pipeline_obj(
            prompt,
            **generation_kwargs
        )
        response_text = outputs[0]["generated_text"]
        response_text = response_text[len(prompt):].strip()

        if is_json_mode:
            response_text = "{" + response_text
            if response_text.endswith("</s>"):
                response_text = response_text[:-4].strip()

        if response_text.endswith("</s>"):
            response_text = response_text[:-4].strip()

        response = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "completion_tokens": len(tokenizer.encode(response_text)),
                "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(response_text))
            }
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_model(model_path: str):
    global pipeline_obj, tokenizer
    print(f"Loading model: {model_path}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="cpu",
    )

    pipeline_obj = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    print(f"Model loaded in {time.time()-start:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../../../pretrain_models/Qwen2-0.5B-Instruct",
                        help="Model name or path (HuggingFace hub or local)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global MODEL_NAME
    MODEL_NAME = args.model.split("/")[-1] if "/" in args.model else args.model

    load_model(args.model)

    print(f"\nServer starting on http://{args.host}:{args.port}")
    print(f"Model: {MODEL_NAME}")
    print("Ready to accept requests...")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()