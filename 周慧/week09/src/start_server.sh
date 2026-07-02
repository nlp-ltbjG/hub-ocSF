#!/bin/bash
# 启动 vLLM OpenAI 兼容 server
#
# 教学重点：
#   1. 一条命令把 HuggingFace 模型变成 OpenAI 兼容 API
#   2. 关键启动参数：max-model-len / dtype
#   3. 启动后访问 http://localhost:8000/v1/chat/completions 即可调用
#
# 使用方式（在 macOS 内执行）：
#   cd /Users/zhouhui/practice/hub-ocSF/周慧/week09/src/
#   bash start_server.sh
#
# 环境依赖：
#   py312 环境（vLLM 0.11.0 + torch 2.8）

set -e

# ── 配置 ─────────────────────────────────────────────────────
VLLM_PATH="/opt/miniconda3/envs/py312/bin/vllm"
MODEL_PATH="/Users/zhouhui/practice/pretrain_models/Qwen2-0.5B-Instruct"
SERVED_NAME="qwen2-0.5b"    # 客户端 API 里使用的模型名（与实际路径解耦）
PORT=8000
MAX_MODEL_LEN=2048          # 最大上下文长度（0.5B 模型不需要太长）
DTYPE="float32"             # CPU 模式下使用 float32（bfloat16 在某些 CPU 上有问题）


# ── 防止 torch/numpy OpenMP 冲突 ────────────────────────────
export KMP_DUPLICATE_LIB_OK=TRUE

echo "============================================"
echo "  启动 vLLM OpenAI Server (CPU模式)"
echo "  模型路径: $MODEL_PATH"
echo "  对外名称: $SERVED_NAME"
echo "  端口:     $PORT"
echo "  max_len:  $MAX_MODEL_LEN"
echo "  dtype:    $DTYPE"
echo "============================================"
echo ""
echo "启动后用以下命令测试："
echo "  curl http://localhost:${PORT}/v1/models"
echo ""

"$VLLM_PATH" serve \
    "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --enforce-eager \
    --host 127.0.0.1
