import json
import sys
from vllm import LLM
from pathlib import Path
from utils.calculs_conversation_gold import generate_calcul_conversation_gold
from utils.calculs_conversation import generate_calcul_conversation
from utils.ner import generate_ner
from utils.charts import generate_charts
from utils.special_cases import generate_special_cases
from utils.tables import generate_tables
from utils.tables_yn_tf import generate_tables_yn_tf
import os


def main(model_name: str, task: str, datasets_path: Path):
    if Path(f"results/{task}:{model_name.replace('/', '__')}.json").exists():
        print(
            f"Results for model {model_name} on task {task} already exist. Skipping generation."
        )
        return

    if model_name == "mistralai/Pixtral-12B-2409":
        tested_llm = LLM(
            model=model_name,
            max_model_len=16384,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            limit_mm_per_prompt={"image": 1},
            tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            if os.environ.get("CUDA_VISIBLE_DEVICES")
            else 1,
        )
    else:
        tested_llm = LLM(
            model=model_name,
            max_model_len=16384,
            tensor_parallel_size=len(os.environ["SLURM_GPUS_ON_NODE"].split(","))
            if os.environ.get("SLURM_GPUS_ON_NODE")
            else 1,
        )

    match task:
        case "NER":
            prediction = generate_ner(tested_llm, datasets_path)
        case "charts":
            prediction = generate_charts(tested_llm, datasets_path)
        case "calculs_conversation_gold":
            prediction = generate_calcul_conversation_gold(tested_llm, datasets_path)
        case "calculs_conversation":
            prediction = generate_calcul_conversation(tested_llm, datasets_path)
        case "special_cases":
            prediction = generate_special_cases(tested_llm, datasets_path)
        case "tables":
            prediction = generate_tables(tested_llm, datasets_path)
        case "tables_yn_tf":
            prediction = generate_tables_yn_tf(tested_llm, datasets_path)

    Path("results").mkdir(exist_ok=True)
    with open(f"results/{task}:{model_name.replace('/', '__')}.json", "w") as json_file:
        json.dump(prediction, json_file)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <tested_model_name> <task>")
        sys.exit(1)

    model_name, task = sys.argv[1], sys.argv[2]
    datasets_path = Path("dataset_json")

    main(model_name, task, datasets_path)
