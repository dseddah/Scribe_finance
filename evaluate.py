import json
import sys
from vllm import LLM
from pathlib import Path
from utils.ner import evaluate_ner
from utils.charts import evaluate_charts
import os
from utils.calculs_conversation import evaluate_calcul_conversation
from utils.special_cases import evaluate_special_cases
from utils.tables import evaluate_tables
from utils.tables_YN_TF import evaluate_tables_yn_tf


def main(tested_model_name: str, judge_model_name: str, task: str, datasets_path: Path):
    with open(
        f"results/{task}:{tested_model_name.replace('/', '__')}.json", "r"
    ) as json_file:
        predictions = json.load(json_file)

    judge_llm = LLM(
        model=judge_model_name,
        max_model_len=1024,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
    )

    match task:
        case "NER":
            results = evaluate_ner(judge_llm, datasets_path, predictions)
        case "charts":
            results = evaluate_charts(judge_llm, datasets_path, predictions)
        case "calculs_conversation":
            results = evaluate_calcul_conversation(
                judge_llm, datasets_path, predictions
            )
        case "special_cases":
            results = evaluate_special_cases(judge_llm, datasets_path, predictions)
        case "tables":
            results = evaluate_tables(judge_llm, datasets_path, predictions)
        case "tables_yn_tf":
            results = evaluate_tables_yn_tf(judge_llm, datasets_path, predictions)

    results.to_csv(
        f"results/{task}:{tested_model_name.replace('/', '__')}:{judge_model_name.replace('/', '__')}.csv",
        index=False,
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python eval.py <tested_model_name> <judge_model_name> <task>")
        sys.exit(1)

    tested_model_name, judge_model_name, task = sys.argv[1], sys.argv[2], sys.argv[3]

    datasets_path = Path("dataset_json_20251021-1055")

    main(tested_model_name, judge_model_name, task, datasets_path)
