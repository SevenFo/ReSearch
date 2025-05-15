#!/bin/bash
# 修改 DATA_DIR 为目录路径而非具体文件
DATA_DIR="/home/ps/Projects/ReSearch-1/data/longbench"
# DATA_DIR="/home/ps/Projects/ReSearch-1/FlashRAG_datasets_1744472312"
# 定义要测试的数据集列表
SUBSETS=("musique" "hotpotqa" "hotpotqa_e" "2wikimqa" "2wikimqa_e" "narrativeqa" "multifieldqa_en" "multifieldqa_en_e")
DONE_SUBSETS=("hotpotqa_e" "hotpotqa_e","multifieldqa_en_e")
SGL_REMOTE_URL="http://0.0.0.0:30080"
REMOTE_RETRIEVELR_URL="http://0.0.0.0:20081"
REMOTE_CRAG_URL="http://172.17.0.1:11434/v1"
GENERATOR_MODEL="/home/ps/.cache/huggingface/hub/models--agentrl--ReSearch-Qwen-7B-Instruct/snapshots/f0787566dce64b1363746137aca5dd432ac48b9e"

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
    SAVE_DIR="/home/ps/Projects/ReSearch-1/output/longbench/research_crag_qwen_2_5_7b_ins/${SUBSET}"
    python run_eval.py \
        --config_path eval_config.yaml \
        --method_name research_crag \
        --data_dir ${DATA_DIR} \
        --dataset_name ${SUBSET} \
        --split test \
        --save_dir ${SAVE_DIR} \
        --save_note crag_qwen2_5_7b_ins_seprate_filter \
        --sgl_remote_url ${SGL_REMOTE_URL} \
        --remote_retriever_url ${REMOTE_RETRIEVELR_URL} \
        --generator_model ${GENERATOR_MODEL} \
        --apply_chat \
        --remote_crag_evaluator_url ${REMOTE_CRAG_URL} \
        --method_name r1r_crag \
        --retrieve_evaluate_strategy 'separate_filter' \
        --evaluate_model_type 'qwen'
    echo "Finished evaluation for subset: ${SUBSET}"
    echo "Results saved in: ${SAVE_DIR}"
    echo "----------------------------------------"
done

echo "All evaluations finished."