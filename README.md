# Scribe Finance

This repository contains a benchmark to evaluate large language models (LLMs)
on several finance-related question. Most evaluation use an image as the context
(for example a chart or a table image). The only exception is the NER evaluation,
which uses a short paragraph as context.

The benchmark runs model generation, automated evaluations against several
judge models, per-task aggregation, and final aggregation across models.

## Evaluations included

The repository contains the following evaluation tasks. Each task generally
provides the model with a context (image or paragraph) and a target set of
questions or prompts.

- `NER`
	- Input: a paragraph (text)
	- Goal: named-entity recognition and extraction in a finance setting.

- `charts`
	- Input: chart images (figures)
	- Goal: read and interpret charts, answer questions about trends, values,
		and relationships shown on the chart.

- `tables`
	- Input: table images
	- Goal: extract and reason over structured table data presented as images.

- `tables_yn_tf`
	- Input: table images
	- Goal: a reformulation of the table evaluation focusing on yes/no or
		true/false style queries and paraphrase robustness.

- `special_cases`
	- Input: mostly image context; contains edge cases and intentionally tricky
		examples to probe model robustness.

- `calculs_conversation`
	- Input: image context (usually) and a multi-turn conversation where the
		model must perform calculations and carry context across turns.

- `calculs_conversation_gold`
	- Input: like `calculs_conversation` but at each question the benchmark
		provides the LLM with the gold (correct) response for the previous turn.
	- Goal: measure performance when the model is fed the correct prior replies
		(useful to isolate downstream reasoning from accumulation of previous
		generation errors).

Datasets for these tasks live in `dataset_json/` (e.g.
`Q&A_finance.dataset_charts.json`, `Q&A_finance.dataset_NER.json`, etc.).

## Key scripts and pipeline

- `generation.py` — runs the tested model to generate answers for a given task.
- `evaluate.py` — compares a tested model's outputs with judge models to
	compute evaluation metrics.
- `group_evaluation.py` — aggregates evaluation metrics per model/task.
- `aggregate_results.py` — final aggregation across tested models, producing
	repo-level summary outputs and tables.
- `run_model.sh` — SLURM job script that runs generation + evaluation +
	group aggregation for a single tested model across all tasks (used as a
	job-array; see below).
- `run.sh` — convenience script that submits `run_model.sh` for multiple
	tested models and schedules the final `aggregate_results.py` job after
	model jobs finish.

Results, logs and intermediate outputs are written to these folders:

- `logs/` — SLURM job logs and evaluation logs (see `run_model.sh`).
- `results/` — per-run outputs produced by evaluation/generation steps.
- `aggregated_results/` — aggregated metrics and result tables.

## Setup

### Prerequisites

- Python 3.12+ (the `pyproject.toml` requires >=3.12).
- A CUDA-enabled GPU environment if you plan to run large vision-enabled
	models or use vLLM for efficient inference.

Install dependencies (recommended using the repository's `uv` runner):

```bash
# install the uv runner if you don't have it
pip install uv

# install project dependencies via uv (reads pyproject)
uv sync
```

### Models
This benchmark use `meta-llama/Llama-3.3-70B-Instruct` `Qwen/Qwen3-32B` `google/gemma-3-27b-it` models to judge the tested models. Make sure you have access to these models on HuggingFace and that you have downloaded them before running the benchmark.

## How the SLURM-run pipeline works (what the scripts do)

- `run_model.sh <MODEL>` is a SLURM script designed to be submitted with
	`sbatch`. It uses a SLURM array to iterate over the task list:

	TASKS=(NER charts calculs_conversation calculs_conversation_gold special_cases tables tables_yn_tf)

	The SLURM array index selects which TASK to run for that job-array task.

	Inside each array task `run_model.sh`:
	- runs `generation.py` for the tested model and the selected task
	- runs `evaluate.py` repeatedly with several judge models to compare
		the tested model's outputs
	- runs `group_evaluation.py` to aggregate per-task metrics for the model

- `run.sh` contains an array of tested models and submits `run_model.sh` via
	`sbatch` for each model. After all model jobs are submitted, `run.sh`
	submits a dependent job that runs `uv run aggregate_results.py` and waits
	for all model jobs to complete before starting aggregation.

Notes:
- The scripts use the `uv` command to run the repository scripts (e.g.
	`uv run generation.py`), which is the project's task runner. If you don't
	have `uv`, you can run the Python modules directly (see examples below).
- The SLURM scripts assume an HPC environment. To run locally without SLURM
	you can directly call the scripts (or use `uv` if available).

## Running the benchmark

1) Run the full multi-model benchmark (on a SLURM cluster):

```bash
./run.sh
```

This will submit `run_model.sh` (via `sbatch`) for each model listed in
`run.sh` and then submit a final aggregation job depending on those jobs.

2) Submit a single model job (SLURM):

```bash
sbatch run_model.sh <huggingface-model-id-or-local-model-ref>
```

Example:

```bash
sbatch run_model.sh Qwen/Qwen3-VL-8B-Instruct
```

3) Run a single task locally (no SLURM) using `uv` (if available):

```bash
uv run generation.py <MODEL> <TASK>
# then evaluate with a judge model
uv run evaluate.py <MODEL> <JUDGE_MODEL> <TASK>
uv run group_evaluation.py <MODEL> <TASK>
```

Or run the Python modules directly (replace `uv run` with `python`):

```bash
python generation.py <MODEL> <TASK>
python evaluate.py <MODEL> <JUDGE_MODEL> <TASK>
python group_evaluation.py <MODEL> <TASK>
```

4) Logs and outputs

- SLURM stdout/stderr are stored under `logs/` (filenames come from the
	SBATCH directives in `run_model.sh` and `run.sh`).
- Per-run results and intermediate artifacts are written to `results/` and
	`aggregated_results/` depending on the script.

## Adding or changing tested models

- Edit the `models` array inside `run.sh` to add or remove tested models.
- Or submit `run_model.sh` directly with the model identifier you want to
	test (see examples above).

## Extending or adding new tasks

- Add the task name to the `TASKS` array inside `run_model.sh` and implement
	the data file(s) in `dataset_json/` and handling logic in
	`generation.py`/`evaluate.py`.

## Notes, assumptions and tips

- The benchmark primarily assumes access to an HPC/SLURM environment for
	large-scale model runs. If you want to run locally, reduce the task
	footprint and run tasks sequentially.
- `run_model.sh` is configured as a SLURM job-array. The array indices map to
	the `TASKS` list in the file. Make sure to update both if you add/remove tasks.
- If `uv` is not available on your system, run the Python scripts directly
	with your Python environment.

## Troubleshooting

- Check `logs/` for SLURM stdout/stderr when jobs fail.
- If aggregation doesn't run, ensure the initial `sbatch` submissions in
	`run.sh` succeeded and that the job IDs were captured correctly.

## Citation
This work will be published as part of the 7th Financial Narrative Processing Workshop, colocated with LREC2026. If you use this model, please cite: 
```@misc{mouilleron-etal:2026:ScribeFinance,
      title={When Tables Go Crazy: Evaluating Multimodal Models on French Financial Documents}, 
      author={Virginie Mouilleron and Théo Lasnier and Anna Mosolova and Djamé Seddah},
      year={2026},
      eprint={2602.10384},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.10384}
}
