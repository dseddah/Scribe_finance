#!/bin/bash

models=(Qwen/Qwen2.5-VL-72B-Instruct-AWQ mistralai/Pixtral-12B-2409 google/gemma-3-12b-it Qwen/Qwen3-VL-8B-Instruct)

for model in "${models[@]}"; do
    sbatch run_model.sh $model
done

mkdir -p aggregated_results

uv run aggregate_results.py