#!/bin/bash

#SBATCH --partition=prepost
#SBATCH --array=0-5
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=logs/model_setup_%A_%a.out
#SBATCH --error=logs/model_setup_%A_%a.err
#SBATCH --account=zln@cpu

module purge

MODELS=("meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen3-32B" "google/gemma-3-27b-it" "mistralai/Pixtral-12B-2409" "google/gemma-3-12b-it" "Qwen/Qwen2.5-VL-72B-Instruct-AWQ")

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

uv run hf download $MODEL