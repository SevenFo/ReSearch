#!/bin/bash
# 修改 DATA_DIR 为目录路径而非具体文件
DATA_DIR="/home/ps/Projects/ReSearch-1/data/longbench"
# DATA_DIR="/home/ps/Projects/ReSearch-1/FlashRAG_datasets_1744472312"
# 定义要测试的数据集列表
SUBSETS=("musique" "hotpotqa" "hotpotqa_e" "2wikimqa" "2wikimqa_e" "narrativeqa" "multifieldqa_en" "multifieldqa_en_e")
DONE_SUBSETS=("hotpotqa_e" "2wikimqa_e" "multifieldqa_en_e")
SGL_REMOTE_URL="http://0.0.0.0:30080"
REMOTE_RETRIEVELR_URL="http://0.0.0.0:20081"
# GENERATOR_MODEL="/home/ps/.cache/huggingface/hub/models--r1-researcher-qwen-2.5-b-base-rag-rl"
GENERATOR_MODEL="/home/ps/.cache/huggingface/hub/models--XXsongLALA-Llama-3.1-8B-instruct-RAG-RL"
cd scripts/evaluation

# 循环遍历数据集列表
for SUBSET in "${SUBSETS[@]}"
do
    if [[ " ${DONE_SUBSETS[@]} " =~ " ${SUBSET} " ]]; then
        echo "Skipping already evaluated subset: ${SUBSET}"
        continue
    fi
    echo "Running evaluation for subset: ${SUBSET}"
    # 根据当前 SUBSET 更新 SAVE_DIR
    # SAVE_DIR="/home/ps/Projects/ReSearch-1/output/FlashRAG_datasets_1744472312_${SUBSET}"
    SAVE_DIR="/home/ps/Projects/ReSearch-1/output/longbench/r1r_llama3.1_8b_ins/${SUBSET}"
    python run_eval.py \
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
        --apply_chat True \
        --method_name r1r

    echo "Finished evaluation for subset: ${SUBSET}"
    echo "Results saved in: ${SAVE_DIR}"
    echo "----------------------------------------"
done

echo "All evaluations finished."