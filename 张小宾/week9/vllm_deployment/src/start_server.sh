#!/bin/bash

set -e

OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    echo "============================================"
    echo "  macOS 模式：启动 transformers 兼容 server"
    echo "============================================"

    PYTHON_PATH="/Users/zhangxiaobin/.pyenv/versions/3.10.13/bin/python"
    
    echo "使用 Python: $PYTHON_PATH"
    echo "模型: Qwen/Qwen2-0.5B-Instruct"
    echo "端口: 8000"
    echo "============================================"
    echo ""
    echo "启动后用以下命令测试："
    echo "  curl http://localhost:8000/v1/models"
    echo ""

    "$PYTHON_PATH" macos_server.py --model Qwen/Qwen2-0.5B-Instruct --port 8000

else
    MODEL_PATH="/mnt/d/badou/项目材料准备/pretrain_models/Qwen2-0.5B-Instruct"
    SERVED_NAME="qwen2-0.5b"
    PORT=8000
    MAX_MODEL_LEN=2048
    GPU_MEM_UTIL=0.6
    DTYPE="float16"

    if [ -z "$VIRTUAL_ENV" ]; then
        source ~/vllm_env/bin/activate
    fi

    export KMP_DUPLICATE_LIB_OK=TRUE

    echo "============================================"
    echo "  启动 vLLM OpenAI Server"
    echo "  模型路径: $MODEL_PATH"
    echo "  对外名称: $SERVED_NAME"
    echo "  端口:     $PORT"
    echo "  max_len:  $MAX_MODEL_LEN"
    echo "  显存占用: ${GPU_MEM_UTIL} (约 5GB / 8GB)"
    echo "============================================"
    echo ""
    echo "启动后用以下命令测试："
    echo "  curl http://localhost:${PORT}/v1/models"
    echo ""

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_NAME" \
        --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --dtype "$DTYPE" \
        --enforce-eager \
        --host 0.0.0.0
fi