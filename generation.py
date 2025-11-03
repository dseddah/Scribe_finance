import json
import sys
from vllm import LLM
from pathlib import Path
from utils.calculs_conversation import generate_calcul_conversation
from utils.ner import generate_ner
from utils.charts import generate_charts
from utils.special_cases import generate_special_cases
from utils.tables import generate_tables
from utils.tables_YN_TF import generate_tables_yn_tf
import os


def main(model_name: str, task: str, datasets_path: Path):
    if model_name == "mistralai/Pixtral-12B-2409":
        tested_llm = LLM(
            model=model_name,
            max_model_len=16384,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            limit_mm_per_prompt={"image": 1},
        )
    else:
        tested_llm = LLM(model=model_name, max_model_len=16384)

    match task:
        case "NER":
            prediction = generate_ner(tested_llm, datasets_path)
        case "charts":
            prediction = generate_charts(tested_llm, datasets_path)
        case "calculs_conversation":
            prediction = generate_calcul_conversation(tested_llm, datasets_path)
        case "special_cases":
            prediction = generate_special_cases(tested_llm, datasets_path)
        case "tables":
            prediction = generate_tables(tested_llm, datasets_path)
        case "tables_yn_tf":
            prediction = generate_tables_yn_tf(tested_llm, datasets_path)

    with open(f"results/{task}:{model_name.replace('/', '__')}.json", "w") as json_file:
        json.dump(prediction, json_file)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <tested_model_name> <task>")
        sys.exit(1)

    model_name, task = sys.argv[1], sys.argv[2]
    datasets_path = Path("dataset_json")

    main(model_name, task, datasets_path)
