#!/bin/bash
MODEL_PATH="agentrl/ReSearch-Qwen-7B-Instruct"

docker run --gpus '"device=0,1,2,3"' \
    --rm \
    --env HF_ENDPOINT=https://hf-mirror.com \
    --shm-size 32g \
    -p 30080:80 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --tp 4 \
        --context-length 20480 \
        --enable-metrics \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --port 80 \
        --trust-remote-code \
        --disable-overlap \
        --disable-radix-cache