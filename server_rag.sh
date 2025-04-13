#!/bin/bash

cd scripts/serving && python retriever_serving.py \
    --config retriever_config.yaml \
    --num_retriever 1 \
    --port 20081