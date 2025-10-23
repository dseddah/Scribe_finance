#!/bin/bash

TESTED_MODEL=$1
TASK=$2


uv run generation.py $TESTED_MODEL $TASK
uv run evaluate.py $TESTED_MODEL Qwen/Qwen3-32B $TASK 
uv run group.py $TESTED_MODEL $TASK
