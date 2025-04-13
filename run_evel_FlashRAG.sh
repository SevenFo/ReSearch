#!/bin/bash
# 修改 DATA_DIR 为目录路径而非具体文件
DATA_DIR="/home/ps/Projects/ReSearch-1/data/longbench"
# SUBSET="hotpotqa_e"
SUBSET="musique"
SAVE_DIR="/home/ps/Projects/ReSearch-1/output/longbench_${SUBSET}"
SGL_REMOTE_URL="http://0.0.0.0:30080"
REMOTE_RETRIEVELR_URL="http://0.0.0.0:20081"
GENERATOR_MODEL="/home/ps/.cache/huggingface/hub/models--agentrl--ReSearch-Qwen-7B-Instruct/snapshots/f0787566dce64b1363746137aca5dd432ac48b9e"
cd scripts/evaluation && python run_eval.py \
    --config_path eval_config.yaml \
    --method_name research \
    --data_dir ${DATA_DIR} \
    --dataset_name ${SUBSET} \
    --split test \
    --save_dir ${SAVE_DIR} \
    --save_note research_qwen7b_ins \
    --sgl_remote_url ${SGL_REMOTE_URL} \
    --remote_retriever_url ${REMOTE_RETRIEVELR_URL} \
    --generator_model ${GENERATOR_MODEL} \
    --apply_chat True