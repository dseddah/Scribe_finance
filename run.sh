#!/bin/bash

models=(Qwen/Qwen2.5-VL-72B-Instruct-AWQ mistralai/Pixtral-12B-2409 google/gemma-3-12b-it Qwen/Qwen3-VL-8B-Instruct)

# Array to store job IDs
job_ids=()

for model in "${models[@]}"; do
    job_id=$(sbatch run_model.sh $model | awk '{print $4}')
    job_ids+=($job_id)
done

deps=$(IFS=:; echo "${job_ids[*]}")

sbatch --wrap "uv run aggregate_results.py" -A zln@cpu --dependency=afterok:$deps --output=logs/aggregate_results_A%.log --error=logs/aggregate_results_A%.err