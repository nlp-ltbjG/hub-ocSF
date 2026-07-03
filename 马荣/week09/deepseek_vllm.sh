CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve /slow_share/models/deepseek-r1-7b \
          --served-model-name deepseekr17b \
          --host 0.0.0.0 \
          --port 8001 \
          --dtype bfloat16 \
          --trust-remote-code \
          --max-model-len 65536 \
          --gpu-memory-utilization 0.90 \
          --data-parallel-size 1 \