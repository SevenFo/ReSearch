#!/bin/bash
# MODEL_PATH="/root/.cache/huggingface/hub/models--r1-researcher-qwen-2.5-b-base-rag-rl"
MODEL_PATH="/root/.cache/huggingface/hub/models--XXsongLALA-Llama-3.1-8B-instruct-RAG-RL"

docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface lmsysorg/sglang:latest bash -c "ls /root/.cache/huggingface/hub/models--r1-researcher-qwen-2.5-b-base-rag-rl" 

docker run --gpus '"device=1,2"' \
    --rm \
    --env HF_ENDPOINT=https://hf-mirror.com \
    --shm-size 32g \
    -p 30080:80 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --tp 2 \
        --context-length 20480 \
        --enable-metrics \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --port 80 \
        --trust-remote-code \
        --disable-overlap \
        --disable-radix-cache
