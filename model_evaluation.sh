#!/bin/bash

#SBATCH --partition=gpu_p6
#SBATCH --gres=gpu:2
#SBATCH --hint=nomultithread
#SBATCH --constraint=h100
#SBATCH --cpus-per-gpu=16
#SBATCH --time=02:00:00
#SBATCH --account=zln@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=logs/model_evaluation_%A_%a.out
#SBATCH --error=logs/model_evaluation_%A_%a.err

module purge

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

TESTED_MODEL=$1

TASKS=(NER charts calculs_conversation special_cases tables tables_yn_tf)
TASK=${TASKS[0]}

JUDGE_MODELS=(meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen3-32B google/gemma-3-27b-it)

uv run generation.py $TESTED_MODEL $TASK

for judge_model in "${JUDGE_MODELS[@]}"; do
    uv run evaluate.py $TESTED_MODEL $judge_model $TASK 
done

uv run group.py $TESTED_MODEL $TASK
