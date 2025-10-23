from vllm import LLM, SamplingParams
from pathlib import Path
import pandas as pd
from utils.evaluation import evaluate_answers


def load_ner_dataset(dataset_path: Path):
    # Load json dataset
    for filesname in dataset_path.iterdir():
        if filesname.suffix == ".json" and "dataset_NER" in filesname.stem:
            with open(filesname, "r") as f:
                ds = pd.read_json(f).rename(
                    columns={
                        "Context (LLM input)": "context",
                        "Answer": "answer",
                        "Question": "question",
                    }
                )[["id", "context", "question", "answer"]]
            break

    return ds


def generate_ner(llm: LLM, dataset_path: Path) -> list[str]:
    ds = load_ner_dataset(dataset_path)

    prompts = ds.apply(
        lambda row: [
            {
                "role": "system",
                "content": "You are a helpful factual QA assistant.",
            },
            {
                "role": "user",
                "content": f"Context: {row['context']}\nQuestion: {row['question']}\nAnswer the question concisely based on the context provided.",
            },
            {
                "role": "assistant",
                "content": "Answer:",
            },
        ],
        axis=1,
    ).tolist()

    outputs = llm.chat(
        prompts,
        sampling_params=SamplingParams(temperature=0, max_tokens=64, n=1),
        add_generation_prompt=False,
        continue_final_message=True,
    )

    return [output.outputs[0].text for output in outputs]


def evaluate_ner(
    judge_llm: LLM, dataset_path: Path, predictions: list[str]
) -> pd.DataFrame:
    ds = load_ner_dataset(dataset_path)

    ds["prediction"] = predictions

    ds["evaluation"] = evaluate_answers(
        judge_llm,
        ds["question"].tolist(),
        predictions,
        ds["answer"].tolist(),
    )

    return ds
